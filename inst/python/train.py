#!/usr/bin/env python3
import warnings
import logging

# ---------------------------------------------------------------------
# Silence noisy warnings/logs
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message=".*GradScaler\\(.*\\) is deprecated.*")

logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.ERROR)

import os
import argparse
import torch

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.engine.hooks import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer

from pycocotools.coco import COCO

# Throughput QoL
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def setup_cfg(args):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()

    zoo_key = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(zoo_key))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_key)

    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.AMP.ENABLED = (args.device == "cuda")

    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST  = (args.val_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.PIN_MEMORY = True
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.OPTIMIZER = "AdamW"           # for logging; builder overrides
    cfg.SOLVER.WEIGHT_DECAY = 5e-2
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_ITERS = max(1, int(0.1 * args.max_iter))
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period if args.checkpoint_period > 0 else 999_999
    cfg.MODEL.BACKBONE.FREEZE_AT = 1

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST  = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 3000
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0

    # IMPORTANT: NUM_CLASSES must be set by main(args) before this runs
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 768
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.33
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.TEST.DETECTIONS_PER_IMAGE = 600

    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1333

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg



# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
def load_and_filter_dataset(annotation_json, image_root):
    ds = load_coco_json(annotation_json, image_root)
    return [d for d in ds if len(d.get("annotations", [])) > 0]


def build_microscopy_augmentation():
    """
    Microscopy-friendly augs:
    - small-angle rotation (±10°)
    - crop BEFORE resize (brings tiny objs into regime)
    - mild color jitter
    - multi-scale resize
    """
    return [
        T.RandomRotation(angle=[-10, 10], sample_style="range"),
        T.RandomCrop("relative_range", (0.7, 0.7)),
        T.RandomFlip(prob=0.5, horizontal=True),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomSaturation(0.9, 1.1),
        T.ResizeShortestEdge(
            short_edge_length=(768, 896, 1024, 1152),
            max_size=1333,
            sample_style="choice",
        ),
    ]


def _log_coco_stats(json_path, tag):
    try:
        from pycocotools.coco import COCO
        coco = COCO(json_path)
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)
        counts = {c["name"]: len(coco.getAnnIds(catIds=[c["id"]])) for c in cats}
        logging.getLogger(__name__).info(f"[{tag}] categories: {[c['name'] for c in cats]}")
        logging.getLogger(__name__).info(f"[{tag}] instance counts: {counts}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not log COCO stats for {tag}: {e}")


# ---------------------------------------------------------------------
# Trainer (force AdamW; keep clipping)
# ---------------------------------------------------------------------
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=0.0,
            weight_decay_bias=0.0,
        )
        opt = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999))
        return maybe_add_gradient_clipping(cfg, opt)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=build_microscopy_augmentation(),
            image_format="RGB",
            # If you want to avoid shapely dep, uncomment next line:
            # instance_mask_format="bitmask",
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    from detectron2.utils.logger import setup_logger
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine.hooks import BestCheckpointer
    from detectron2.checkpoint import DetectionCheckpointer
    import logging, os

    setup_logger(name="detectron2")

    # Derive val name if not provided
    if not getattr(args, "val_dataset_name", None):
        args.val_dataset_name = args.dataset_name.replace("_train", "_val")

    # Clear any stale registrations (safe)
    for name in [args.dataset_name, args.val_dataset_name]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

    # Canonical registration (sets json_file/image_root; metadata gets hydrated on first load)
    register_coco_instances(args.dataset_name,     {}, args.annotation_json,     args.image_root)
    register_coco_instances(args.val_dataset_name, {}, args.val_annotation_json, args.val_image_root)

    # --- Force-load datasets once so Detectron2 populates thing_classes/id mapping ---
    try:
        _ = DatasetCatalog.get(args.dataset_name)      # calls load_coco_json(..., dataset_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TRAIN dataset '{args.dataset_name}'. "
            f"Check --annotation-json/--image-root paths. Underlying error: {e}"
        )
    try:
        _ = DatasetCatalog.get(args.val_dataset_name)  # calls load_coco_json(..., dataset_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load VAL dataset '{args.val_dataset_name}'. "
            f"Check --val-annotation-json/--val-image-root paths. Underlying error: {e}"
        )

    # Now metadata is populated by Detectron2
    tmeta = MetadataCatalog.get(args.dataset_name)
    vmeta = MetadataCatalog.get(args.val_dataset_name)
    tnames = list(getattr(tmeta, "thing_classes", []) or [])
    vnames = list(getattr(vmeta, "thing_classes", []) or [])

    logging.getLogger(__name__).info(f"TRAIN JSON: {getattr(tmeta, 'json_file', args.annotation_json)}")
    logging.getLogger(__name__).info(f"VAL   JSON: {getattr(vmeta, 'json_file', args.val_annotation_json)}")
    logging.getLogger(__name__).info(f"train thing_classes: {tnames}")
    logging.getLogger(__name__).info(f"val   thing_classes: {vnames}")

    if len(tnames) == 0:
        raise RuntimeError(
            "Dataset metadata has no 'thing_classes' after registration+load. "
            "Ensure your TRAIN JSON has a valid 'categories' list with 'name' fields."
        )
    if len(vnames) == 0:
        raise RuntimeError(
            "Dataset metadata has no 'thing_classes' for VAL after registration+load. "
            "Ensure your VAL JSON has a valid 'categories' list with 'name' fields."
        )
    if len(tnames) != len(vnames):
        raise RuntimeError(
            f"Class count mismatch: train={len(tnames)} vs val={len(vnames)}. "
            "Train/Val JSONs must share identical 'categories'."
        )

    # Use metadata-derived class count; if you pass --num-classes >=0, you can override this.
    if getattr(args, "num_classes", -1) is not None and args.num_classes >= 0:
        args.num_classes = int(args.num_classes)
    else:
        args.num_classes = len(tnames)

    logging.getLogger(__name__).info(f"Using NUM_CLASSES = {args.num_classes}")

    # Build config AFTER dataset registration & class resolution
    cfg = setup_cfg(args)
    default_setup(cfg, args)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    trainer = CocoTrainer(cfg)

    # Resume & keep best checkpoint by segm/AP (switch to "bbox/AP" if you prefer)
    resume = os.path.exists(os.path.join(cfg.OUTPUT_DIR, "last_checkpoint"))
    trainer.resume_or_load(resume=resume)

    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    trainer.register_hooks([
        BestCheckpointer(cfg.TEST.EVAL_PERIOD, checkpointer, val_metric="segm/AP", mode="max")
    ])

    trainer.train()





# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = default_argument_parser()

    # Data
    parser.add_argument("--dataset-name", default="shell_train")
    parser.add_argument("--val-dataset-name", default="")
    parser.add_argument("--annotation-json", default="data/shell_mixed/train/_annotations.coco.json")
    parser.add_argument("--image-root", default="data/shell_mixed/train")
    parser.add_argument("--val-annotation-json", default="data/shell_mixed/val/_annotations.coco.json")
    parser.add_argument("--val-image-root", default="data/shell_mixed/val")

    # System / output
    parser.add_argument("--output-dir", default="Detectron2_Models")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=8)

    # Solver / schedule (lean: LR is passed from caller, others are fixed in code)
    parser.add_argument("--ims-per-batch", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=12000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)  # pass batch-scaled LR from R
    parser.add_argument("--eval-period", type=int, default=500)
    parser.add_argument("--checkpoint-period", type=int, default=0)   # 0 => final only

    # Model
    parser.add_argument("--num-classes", type=int, default=-1) # -1 => infer from dataset

    # Optional detectron2 overrides
    parser.add_argument("--opts", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    launch(main, args.num_gpus, num_machines=1, machine_rank=0, dist_url="auto", args=(args,))
