from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        transform: Optional[Callable] = None,
        classes_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv.as_posix()}")

        self.df = pd.read_csv(self.split_csv)
        required = {"filepath", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns {sorted(required)}")

        self.transform = transform

        if classes_to_idx is None:
            labels = sorted(self.df["label"].unique().tolist())
            self.classes_to_idx = {c: i for i, c in enumerate(labels)}
        else:
            self.classes_to_idx = dict(classes_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = Path(row["filepath"])
        label_str = str(row["label"])

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path.as_posix()}")

        with Image.open(path) as im:
            im = im.convert("RGB")

        if self.transform is not None:
            x = self.transform(im)
        else:
            x = torch.from_numpy(
                __import__("numpy").array(im).transpose(2, 0, 1)
            ).float() / 255.0

        y = self.classes_to_idx[label_str]
        return x, y