from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = ["cats", "dogs", "wildlife"]


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def human_bytes(n: int) -> str:
    # Simple human-readable bytes
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} PB"


def folder_size_bytes(folder: Path) -> int:
    total = 0
    for p in folder.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    project_root = Path(r"F:\Projects\AnimalClassification")
    prepared = project_root / "data" / "prepared"

    if not prepared.exists():
        raise SystemExit(f"Missing prepared folder: {prepared}")

    # Collect images per class
    class_files: dict[str, list[Path]] = {}
    class_counts: dict[str, int] = {}
    class_sizes: dict[str, int] = {}

    total_images = 0
    total_bytes = 0

    for cls in CLASSES:
        cls_dir = prepared / cls
        files = [p for p in cls_dir.rglob("*") if is_image(p)] if cls_dir.exists() else []
        class_files[cls] = files
        class_counts[cls] = len(files)
        class_sizes[cls] = folder_size_bytes(cls_dir) if cls_dir.exists() else 0

        total_images += class_counts[cls]
        total_bytes += class_sizes[cls]

    print("=== Dataset summary ===")
    print(f"Root: {prepared}")
    print(f"Total images: {total_images:,}")
    print(f"Total size:   {human_bytes(total_bytes)}")
    print()

    print("=== Per-class counts / distribution / size ===")
    for cls in CLASSES:
        cnt = class_counts[cls]
        pct = (cnt / total_images * 100.0) if total_images else 0.0
        print(f"{cls:8s}  {cnt:8,d}  ({pct:6.2f}%)   {human_bytes(class_sizes[cls])}")
    print()

    # Duplicate detection (exact duplicates by SHA256)
    # For speed, group by file size first (only same-size files can be duplicates)
    all_images = []
    for cls in CLASSES:
        all_images.extend(class_files[cls])

    size_groups: dict[int, list[Path]] = defaultdict(list)
    for p in all_images:
        try:
            size_groups[p.stat().st_size].append(p)
        except OSError:
            pass

    # Hash only groups with >1 file
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    candidates = sum(1 for sz, paths in size_groups.items() if len(paths) > 1)
    print("=== Duplicate check (exact, SHA256) ===")
    print(f"File-size groups with potential duplicates: {candidates:,}")

    hashed_files = 0
    for sz, paths in size_groups.items():
        if len(paths) < 2:
            continue
        for p in paths:
            try:
                digest = sha256_file(p)
                hash_to_paths[digest].append(p)
                hashed_files += 1
            except OSError:
                pass

    dup_groups = [(h, ps) for h, ps in hash_to_paths.items() if len(ps) > 1]
    dup_files = sum(len(ps) for _, ps in dup_groups)
    print(f"Hashed files: {hashed_files:,}")
    print(f"Duplicate groups: {len(dup_groups):,}")
    print(f"Files involved in duplicates: {dup_files:,}")
    print()

    # Print top duplicate groups (limit output)
    if dup_groups:
        print("Top duplicate groups (showing up to 10 groups):")
        dup_groups.sort(key=lambda x: len(x[1]), reverse=True)
        for i, (h, ps) in enumerate(dup_groups[:10], start=1):
            print(f"\nGroup {i} (count={len(ps)}):")
            for p in ps[:10]:  # limit per group
                print(f"  {p}")
            if len(ps) > 10:
                print(f"  ... ({len(ps)-10} more)")
    else:
        print("No exact duplicates found (by SHA256).")

    print("\n=== Near-duplicate hint (same file size, not guaranteed) ===")
    # This is just a hint; many non-duplicates can share size.
    big_size_groups = sorted(
        [(sz, paths) for sz, paths in size_groups.items() if len(paths) > 5],
        key=lambda x: len(x[1]),
        reverse=True
    )
    if big_size_groups:
        sz, paths = big_size_groups[0]
        print(f"Largest same-size group: size={sz} bytes, count={len(paths)}")
        print("Example paths:")
        for p in paths[:10]:
            print(f"  {p}")
        if len(paths) > 10:
            print(f"  ... ({len(paths)-10} more)")
    else:
        print("No large same-size groups detected.")


if __name__ == "__main__":
    main()