import os
from pathlib import Path

import kagglehub

path = kagglehub.dataset_download("vibhingupta028/liver-histopathology-fibrosis-ultrasound-images")
root = Path(path)
print("Root:", root)

# 상위 몇 단계만 트리처럼 보기
def show_tree(p: Path, max_depth=2, indent=0):
    if indent//2 >= max_depth:
        return
    for x in sorted(p.iterdir())[:50]:  # 너무 많으면 50개까지만
        print("  "*indent + ("[D] " if x.is_dir() else "[F] ") + x.name)
        if x.is_dir():
            show_tree(x, max_depth=max_depth, indent=indent+1)

show_tree(root, max_depth=3)