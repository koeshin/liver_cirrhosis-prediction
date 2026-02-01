import os, shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def prepare_usfm_from_classfolders(
    img_root: str,
    out_root: str = ".",
    dataset_name: str = "kaggle_liver_fibrosis",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    copy_mode: str = "copy"  # "copy" or "symlink"
):
    img_root = Path(img_root)
    out_root = Path(out_root)
    base = out_root / "datasets" / "Cls" / dataset_name

    # 1) 파일 목록 생성
    class_dirs = [d for d in img_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found under: {img_root}")

    rows = []
    for cdir in class_dirs:
        label = cdir.name
        for fp in cdir.rglob("*"):
            if fp.suffix.lower() in IMG_EXT:
                rows.append({"path": str(fp), "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under: {img_root}")

    print("✅ Found images per class:")
    print(df["label"].value_counts().sort_index())

    # 2) stratified split
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size / (1 - test_size),
        stratify=df_trainval["label"],
        random_state=seed
    )

    splits = [("trainning_set", df_train), ("val_set", df_val), ("test_set", df_test)]

    # 3) 폴더 생성
    for split_name, _ in splits:
        for lbl in sorted(df["label"].unique()):
            (base / split_name / lbl).mkdir(parents=True, exist_ok=True)

    # 4) 파일 복사/링크
    def place_file(src_path: Path, dst_path: Path):
        if dst_path.exists():
            return
        if copy_mode == "symlink":
            os.symlink(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    for split_name, split_df in splits:
        for _, row in split_df.iterrows():
            src = Path(row["path"])
            lbl = row["label"]
            dst = base / split_name / lbl / src.name

            # 파일명 충돌 방지(혹시 같은 이름이 있으면)
            if dst.exists():
                dst = base / split_name / lbl / f"{src.parent.name}__{src.name}"

            place_file(src, dst)

    print("\n✅ USFM dataset prepared at:", base)
    print("Use this dataset name in USFM:", dataset_name)
    return base

# --- 실행 ---
root = Path("/Users/sinjaewon/.cache/kagglehub/datasets/vibhingupta028/liver-histopathology-fibrosis-ultrasound-images/versions/2")
IMG_ROOT = root / "Dataset" / "Dataset"

# out_root="." 이면 현재 작업 폴더에 datasets/Cls/... 생성됨
base = prepare_usfm_from_classfolders(
    img_root=str(IMG_ROOT),
    out_root=".",
    dataset_name="kaggle_liver_fibrosis",
    copy_mode="symlink"  # 디스크 아끼려면 symlink 추천(맥은 보통 OK)
)