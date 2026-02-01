import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("vibhingupta028/liver-histopathology-fibrosis-ultrasound-images")
root = Path(path)
print("Downloaded root:", root)
from pathlib import Path

def find_class_root(root: Path, class_names=("F0","F1","F2","F3","F4")):
    class_names = set(class_names)
    for p in root.rglob("*"):
        if p.is_dir():
            kids = {c.name for c in p.iterdir() if c.is_dir()}
            if class_names.issubset(kids):
                return p
    return None

img_root = find_class_root(root)
print("img_root:", img_root)
print("classes:", sorted([d.name for d in img_root.iterdir() if d.is_dir()])[:10])


import os, shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def prepare_usfm_from_classfolders(
    img_root: Path,
    usfm_root: Path,
    dataset_name: str = "kaggle_liver_fibrosis",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    copy_mode: str = "symlink"  # colab은 보통 symlink가 편함(빠르고 용량 절약)
):
    base = usfm_root / "datasets" / "Cls" / dataset_name

    # 1) 파일 목록
    rows = []
    for cdir in sorted([d for d in img_root.iterdir() if d.is_dir()]):
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

    splits = [("train", df_train), ("val", df_val), ("test", df_test)]

    # 3) 폴더 생성
    labels = sorted(df["label"].unique())
    for split_name, _ in splits:
        for lbl in labels:
            (base / split_name / lbl).mkdir(parents=True, exist_ok=True)

    # 4) 복사/링크
    def place_file(src: Path, dst: Path):
        if dst.exists():
            return
        if copy_mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)

    for split_name, split_df in splits:
        for _, row in split_df.iterrows():
            src = Path(row["path"])
            lbl = row["label"]
            dst = base / split_name / lbl / src.name

            # 파일명 충돌 방지
            if dst.exists():
                dst = base / split_name / lbl / f"{src.parent.name}__{src.name}"

            place_file(src, dst)

    print("\n✅ USFM dataset prepared at:", base)
    return dataset_name

dataset_name = prepare_usfm_from_classfolders(
    img_root=img_root,
    usfm_root=Path("./USFM"),
    dataset_name="kaggle_liver_fibrosis",
    copy_mode="symlink"
)