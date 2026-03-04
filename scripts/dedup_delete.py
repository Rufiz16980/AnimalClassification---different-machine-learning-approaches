from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# 1,041 duplicates were removed 

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# OPTIONAL near-duplicate removal (risky). Default OFF.
ENABLE_PERCEPTUAL = False

# If perceptual enabled, only treat as duplicates if BOTH:
# - same dimensions
# - identical aHash (threshold 0). You can loosen to 1..5 if you truly want aggressive.
PERCEPTUAL_MAX_HAMMING = 0
  -

@dataclass
class ImgInfo:
    path: Path
    cls: str
    size_bytes: int


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(path: Path) -> tuple[int, int, str] | None:
    """
    Exact pixel identity hash:
    - Decode with PIL
    - Convert to RGB for consistent comparison
    - Hash raw pixel bytes + dimensions
    Returns (width, height, sha256_of_pixels)
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            data = im.tobytes()
        ph = hashlib.sha256(data).hexdigest()
        return (w, h, ph)
    except Exception:
        return None


def ahash64(path: Path) -> tuple[int, int, int] | None:
    """
    Average-hash (aHash) for near-duplicates.
    Returns (width, height, hash_int_64)
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
            im = im.convert("L").resize((8, 8))
            px = list(im.getdata())
        mean = sum(px) / 64.0
        bits = 0
        for i, v in enumerate(px):
            if v >= mean:
                bits |= (1 << i)
        return (w, h, bits)
    except Exception:
        return None


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def collect_images(prepared_dir: Path) -> list[ImgInfo]:
    infos: list[ImgInfo] = []
    for cls_dir in prepared_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for p in cls_dir.rglob("*"):
            if is_image(p):
                try:
                    infos.append(ImgInfo(path=p, cls=cls, size_bytes=p.stat().st_size))
                except OSError:
                    pass
    return infos


def move_to_quarantine(path: Path, quarantine_root: Path, cls: str) -> Path:
    """
    Move file into quarantine preserving class subfolder.
    Name-collision safe.
    """
    target_dir = quarantine_root / cls
    target_dir.mkdir(parents=True, exist_ok=True)

    dst = target_dir / path.name
    if dst.exists():
        stem = path.stem
        ext = path.suffix
        k = 1
        while True:
            cand = target_dir / f"{stem}__dup{k}{ext}"
            if not cand.exists():
                dst = cand
                break
            k += 1

    shutil.move(str(path), str(dst))
    return dst


def write_csv(rows: list[dict], out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    project_root = Path(r"F:\Projects\AnimalClassification")
    prepared = project_root / "data" / "prepared"
    if not prepared.exists():
        raise SystemExit(f"Missing: {prepared}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = project_root / "data" / "_duplicates_quarantine" / timestamp
    report_dir = project_root / "reports"
    report_csv = report_dir / f"dedup_report_{timestamp}.csv"

    print(f"Prepared:   {prepared}")
    print(f"Quarantine: {quarantine}")
    print(f"Report:     {report_csv}")
    print()

    images = collect_images(prepared)
    print(f"Found {len(images):,} images")

    # ---------------------------
    # Phase 1: Byte-identical
    # ---------------------------
    print("\n[1/3] Removing BYTE-identical duplicates (SHA256 of file bytes)...")
    by_file_hash: dict[str, list[ImgInfo]] = defaultdict(list)
    for info in tqdm(images, desc="Hashing files (SHA256)"):
        try:
            h = sha256_file(info.path)
            by_file_hash[h].append(info)
        except OSError:
            pass

    to_quarantine: list[ImgInfo] = []
    kept = 0
    for h, group in by_file_hash.items():
        if len(group) == 1:
            kept += 1
            continue
        # keep first, quarantine rest
        group_sorted = sorted(group, key=lambda x: str(x.path))
        kept += 1
        to_quarantine.extend(group_sorted[1:])

    print(f"Byte-identical duplicate files to remove: {len(to_quarantine):,}")

    report_rows: list[dict] = []
    removed_set = set()

    for info in tqdm(to_quarantine, desc="Quarantining byte-identical"):
        if not info.path.exists():
            continue
        dst = move_to_quarantine(info.path, quarantine, info.cls)
        removed_set.add(str(info.path))
        report_rows.append({
            "phase": "byte_identical",
            "original": str(info.path),
            "quarantined_to": str(dst),
        })

    # refresh list after removals
    images = [i for i in images if i.path.exists()]
    print(f"Remaining after phase 1: {len(images):,}")

    # ---------------------------
    # Phase 2: Pixel-identical
    # ---------------------------
    print("\n[2/3] Removing PIXEL-identical duplicates (decoded RGB pixels)...")
    by_pixel: dict[tuple[int, int, str], list[ImgInfo]] = defaultdict(list)

    for info in tqdm(images, desc="Computing pixel hashes"):
        ph = pixel_hash(info.path)
        if ph is None:
            continue
        by_pixel[ph].append(info)

    pixel_dups = []
    for key, group in by_pixel.items():
        if len(group) > 1:
            group_sorted = sorted(group, key=lambda x: str(x.path))
            pixel_dups.extend(group_sorted[1:])

    print(f"Pixel-identical duplicate files to remove: {len(pixel_dups):,}")

    for info in tqdm(pixel_dups, desc="Quarantining pixel-identical"):
        if not info.path.exists():
            continue
        dst = move_to_quarantine(info.path, quarantine, info.cls)
        report_rows.append({
            "phase": "pixel_identical",
            "original": str(info.path),
            "quarantined_to": str(dst),
        })

    # refresh list after removals
    images = [i for i in images if i.path.exists()]
    print(f"Remaining after phase 2: {len(images):,}")

    # ---------------------------
    # Phase 3: Perceptual (optional)
    # ---------------------------
    if ENABLE_PERCEPTUAL:
        print("\n[3/3] Removing PERCEPTUAL duplicates (aHash, optional + risky)...")
        by_ah: dict[tuple[int, int, int], list[ImgInfo]] = defaultdict(list)

        for info in tqdm(images, desc="Computing aHash"):
            ah = ahash64(info.path)
            if ah is None:
                continue
            by_ah[ah].append(info)

        # exact aHash match groups first
        candidates = []
        for key, group in by_ah.items():
            if len(group) > 1:
                group_sorted = sorted(group, key=lambda x: str(x.path))
                candidates.extend(group_sorted[1:])

        print(f"Perceptual duplicate candidates to remove: {len(candidates):,}")

        # If you set PERCEPTUAL_MAX_HAMMING > 0, you’d need pairwise checking.
        # We keep it conservative here: exact aHash match only (hamming=0),
        # and also only if dimensions are identical (already in key).
        for info in tqdm(candidates, desc="Quarantining perceptual"):
            if not info.path.exists():
                continue
            dst = move_to_quarantine(info.path, quarantine, info.cls)
            report_rows.append({
                "phase": "perceptual_ahash_exact",
                "original": str(info.path),
                "quarantined_to": str(dst),
            })

        images = [i for i in images if i.path.exists()]
        print(f"Remaining after phase 3: {len(images):,}")
    else:
        print("\n[3/3] Perceptual dedup is OFF (ENABLE_PERCEPTUAL=False).")

    write_csv(report_rows, report_csv)

    print("\nDone.")
    print(f"Quarantined files: {len(report_rows):,}")
    print(f"Report written to: {report_csv}")
    print(f"Quarantine folder: {quarantine}")


if __name__ == "__main__":
    # Avoid PIL decompression bomb warnings from huge images if any
    Image.MAX_IMAGE_PIXELS = None
    main()