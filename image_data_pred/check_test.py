import os
os.environ["MPLBACKEND"] = "Agg"

import sys, json, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from typing import Optional


# ✅ 로컬 경로 대응: 이 파일이 있는 폴더와 USFM 폴더를 path에 추가
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "USFM"))

from omegaconf import OmegaConf
import logging
from usdsgen.models.build import build_model

logger = logging.getLogger("infer_compare")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def guess_split_dir(data_root: Path, split: str) -> Path:
    candidates = {
        "train": ["train", "trainning_set", "training_set"],
        "val":   ["val", "val_set", "valid", "validation", "validation_set"],
        "test":  ["test", "test_set"],
    }[split]
    for name in candidates:
        p = data_root / name
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find {split} under {data_root}. Tried {candidates}")

def make_transforms(img_size: int, interpolation: str, normalize: bool):
    interp_map = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "nearest": transforms.InterpolationMode.NEAREST,
    }
    interp = interp_map.get(interpolation.lower(), transforms.InterpolationMode.BICUBIC)

    tf = [
        transforms.Resize(img_size, interpolation=interp),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if normalize:
        tf.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(tf)

from typing import Optional


def build_from_run_config(run_dir: str):
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)  # override 허용
    return build_model(cfg, logger), cfg


def build_usfm_vit(num_classes: int, img_size: int, pretrained_path: Optional[str] = None):
    cfg = OmegaConf.create({
        "task": "Cls",
        "data": {"img_size": img_size, "num_classes": num_classes},
        "model": {
            # ✅ 이 줄 추가 (중요)
            "model_type": "Cls",   # "FM"이 아니면 됨. 의미상 Cls 추천

            "num_classes": num_classes,
            "img_size": img_size,
            "model_cfg": {
                "type": "vit",
                "num_classes": num_classes,
                "backbone": {"pretrained": None},
                "name": "vit-b",
                "in_chans": 3,
                "patch_size": 16,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4,
                "qkv_bias": True,
                "attn_drop_rate": 0.0,
                "drop_path_rate": 0.1,
                "init_values": 0.1,
                "use_abs_pos_emb": False,
                "use_rel_pos_bias": True,
                "use_shared_rel_pos_bias": False,
                "use_mean_pooling": True,
                "pretrained": pretrained_path,  # baseline일 때만 넣고, finetuned는 None
            }
        }
    })
    return build_model(cfg, logger)

def load_state_dict_safely(model, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: sd = ckpt["state_dict"]
        elif "model" in ckpt:   sd = ckpt["model"]
        else:                   sd = ckpt
    else:
        sd = ckpt

    new_sd = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    return missing, unexpected

@torch.no_grad()
def eval_model(model, loader, device):
    model.to(device).eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
        probs.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)

def summarize(y_true, y_pred, class_names, title):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print("\n" + "="*80)
    print(title)
    print(f"Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}

def find_latest_ckpt(log_root: Path):
    exts = (".ckpt", ".pth", ".pt")
    files = [p for p in log_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(REPO_ROOT/"datasets/Cls/kaggle_liver_fibrosis"))
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--interpolation", type=str, default="bicubic")
    ap.add_argument("--no_normalize", action="store_true")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--baseline_pretrain", type=str, default=str(REPO_ROOT/"assets/FMweight/USFM_latest.pth"))
    ap.add_argument("--finetune_ckpt", type=str, default="auto")
    ap.add_argument("--finetune_log_root", type=str, default=str(REPO_ROOT/"logs/finetune/Cls/kaggle_liver_fibrosis/vit"))

    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    ap.add_argument("--out_dir", type=str, default=str(REPO_ROOT/"infer_outputs"))

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = guess_split_dir(data_root, "train")
    val_dir   = guess_split_dir(data_root, "val")
    test_dir  = guess_split_dir(data_root, "test")

    tf = make_transforms(args.img_size, args.interpolation, normalize=(not args.no_normalize))

    train_ds = ImageFolder(str(train_dir), transform=tf)
    val_ds   = ImageFolder(str(val_dir), transform=tf)
    test_ds  = ImageFolder(str(test_dir), transform=tf)

    class_names = test_ds.classes
    num_classes = len(class_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} | classes={class_names}")
    logger.info(f"Device: {args.device}")

    results = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "classes": class_names, "args": vars(args)}

    # 1) Baseline: USFM_latest 로드 (하지만 head는 사실상 랜덤이라 점수 낮게 나오는 게 정상)
    # 1) Fine-tuned config로 모델 구조 만들기

    model, cfg = build_from_run_config("logs/finetune/Cls/kaggle_liver_fibrosis/vit/finetune_500e_bs16/2026-02-01_01-32-59")
    y_true_b, y_pred_b, _ = eval_model(model, test_loader, args.device)
    results["baseline"] = summarize(y_true_b, y_pred_b, class_names, "Baseline: USFM_latest (random head)")

    # 2) Fine-tuned ckpt
    finetune_ckpt = args.finetune_ckpt
    if finetune_ckpt.lower() == "auto":
        ckpt = find_latest_ckpt(Path(args.finetune_log_root))
        if ckpt is None:
            raise FileNotFoundError(f"No ckpt under {args.finetune_log_root}. Set --finetune_ckpt explicitly.")
        finetune_ckpt = str(ckpt)
        logger.info(f"Auto-picked ckpt: {finetune_ckpt}")

    finetune_ckpt_path = Path(finetune_ckpt)
    if not finetune_ckpt_path.exists():
        raise FileNotFoundError(f"Fine-tuned ckpt not found: {finetune_ckpt_path}")

    finetuned_model,cfg=  build_from_run_config("logs/finetune/Cls/kaggle_liver_fibrosis/vit/finetune_500e_bs16/2026-02-01_01-32-59")
    missing, unexpected = load_state_dict_safely(finetuned_model, finetune_ckpt_path)
    logger.info(f"Loaded fine-tuned. missing={len(missing)}, unexpected={len(unexpected)}")

    y_true_f, y_pred_f, _ = eval_model(finetuned_model, test_loader, args.device)
    results["finetuned"] = summarize(y_true_f, y_pred_f, class_names, f"Fine-tuned: {finetune_ckpt_path.name}")
    results["finetuned"]["ckpt"] = str(finetune_ckpt_path)

    out_json = out_dir / "compare_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
