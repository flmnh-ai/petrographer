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
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer

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

    # Map backbone names to model zoo keys
    backbone_map = {
        "resnet50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "resnet101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "resnext101": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
    }

    # Get backbone model zoo key
    zoo_key = backbone_map.get(args.backbone.lower(), args.backbone)

    # Load base config
    cfg.merge_from_file(model_zoo.get_config_file(zoo_key))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_key)

    # Basic settings
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.AMP.ENABLED = (args.device == "cuda")

    # Datasets
    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST  = (args.val_dataset_name,)

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # Training schedule
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period if args.checkpoint_period > 0 else 999_999

    # Evaluation period
    cfg.TEST.EVAL_PERIOD = args.eval_period

    # Tier 1 improvements: Better learning schedule
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    warmup_target = max(1, int(0.1 * args.max_iter))
    cfg.SOLVER.WARMUP_ITERS = max(100, min(warmup_target, 1000))
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000

    # Backbone freezing (0=freeze nothing, 1=freeze stem, 2=freeze stem+res2, etc.)
    cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze_at

    # Differential learning rates: backbone gets 0.1x, head gets 1.0x
    # This is standard for fine-tuning pretrained models
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # Number of classes (only thing that must be set for your dataset)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

    # Dense detection settings (for datasets with 100+ objects per image)
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000      # default: 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500       # default: 1000
    cfg.TEST.DETECTIONS_PER_IMAGE = 500           # default: 100

    # Keep scale close to 1k px tiles
    cfg.INPUT.MIN_SIZE_TRAIN = (928, 960, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024

    # Optional overrides from command line
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg



# ---------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------
def build_augmentations(cfg):
    """Augmentations: flips + color jitter"""
    return [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomRotation(angle=[0, 90, 180, 270], sample_style="choice", expand=False),
        T.RandomBrightness(0.8, 1.2),  # ±20% brightness
        T.RandomContrast(0.8, 1.2),    # ±20% contrast
        T.RandomSaturation(0.8, 1.2),  # ±20% saturation
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"
        ),
    ]


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=build_augmentations(cfg),
            image_format="RGB",
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
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="Backbone: resnet50, resnet101, resnext101, or full model zoo key")
    parser.add_argument("--freeze-at", type=int, default=2,
                        help="Freeze backbone up to this stage (0=none, 1=stem, 2=stem+res2, default=1)")

    # Optional detectron2 overrides
    parser.add_argument("--opts", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    launch(main, args.num_gpus, num_machines=1, machine_rank=0, dist_url="auto", args=(args,))
