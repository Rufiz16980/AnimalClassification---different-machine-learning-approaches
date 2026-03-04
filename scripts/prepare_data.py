import os
import re
import zipfile
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Hugging Face datasets (optional but used for your HF arrow dump)
from datasets import load_from_disk


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs(prepared_dir: Path):
    (prepared_dir / "cats").mkdir(parents=True, exist_ok=True)
    (prepared_dir / "dogs").mkdir(parents=True, exist_ok=True)
    (prepared_dir / "wildlife").mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def sanitize_name(s: str) -> str:
    # Make a safe-ish filename chunk
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    return s.strip("_")[:120] if s else "img"


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def copy_image(src: Path, dst_dir: Path, prefix: str):
    # collision-safe filename
    ext = src.suffix.lower()
    base = sanitize_name(src.stem)
    out = dst_dir / f"{prefix}_{base}{ext}"
    i = 1
    while out.exists():
        out = dst_dir / f"{prefix}_{base}_{i}{ext}"
        i += 1
    shutil.copy2(src, out)


def unzip_all_zips_in_dir(zips_dir: Path, out_dir: Path):
    zips = sorted([p for p in zips_dir.glob("*.zip") if p.is_file()])
    if not zips:
        print(f"[unzip] No .zip files found in: {zips_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for z in tqdm(zips, desc=f"Unzipping in {zips_dir.name}"):
        # extract into a subfolder named like the zip (without .zip)
        target = out_dir / z.stem
        target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(target)
        except zipfile.BadZipFile:
            print(f"[unzip] Bad zip, skipping: {z}")


def ingest_microsoft(microsoft_dir: Path, prepared_dir: Path):
    # expects ...\microsoft\PetImages\Cat and Dog
    cat_dir = microsoft_dir / "PetImages" / "Cat"
    dog_dir = microsoft_dir / "PetImages" / "Dog"
    if not cat_dir.exists() or not dog_dir.exists():
        print(f"[microsoft] Expected PetImages/Cat and PetImages/Dog under: {microsoft_dir}")
        return

    for cls, src_dir in [("cats", cat_dir), ("dogs", dog_dir)]:
        dst_dir = prepared_dir / cls
        for p in tqdm(list(src_dir.rglob("*")), desc=f"Copy Microsoft -> {cls}"):
            if p.is_file() and is_image_file(p):
                if verify_image(p):
                    copy_image(p, dst_dir, prefix="ms")
                # silently skip corrupted


def ingest_afhq(afhq_dir: Path, prepared_dir: Path):
    # expects ...\AFHQv2\cat dog wild (or wildlife)
    mapping = {
        "cat": "cats",
        "dog": "dogs",
        "wild": "wildlife",
        "wildlife": "wildlife",
    }

    if not afhq_dir.exists():
        print(f"[afhq] Missing dir: {afhq_dir}")
        return

    subdirs = [d for d in afhq_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[afhq] No subfolders found in: {afhq_dir}")
        return

    for d in subdirs:
        key = d.name.lower().strip()
        if key not in mapping:
            continue
        dst = prepared_dir / mapping[key]
        for p in tqdm(list(d.rglob("*")), desc=f"Copy AFHQv2/{d.name} -> {mapping[key]}"):
            if p.is_file() and is_image_file(p):
                if verify_image(p):
                    copy_image(p, dst, prefix="afhq")


def ingest_afd_unzipped(afd_face_images_dir: Path, prepared_dir: Path):
    # After unzipping, you’ll have species folders. Everything goes to wildlife.
    if not afd_face_images_dir.exists():
        print(f"[afd] Missing dir: {afd_face_images_dir}")
        return

    wildlife_dir = prepared_dir / "wildlife"
    # Copy images from all subfolders (species)
    imgs = []
    for p in afd_face_images_dir.rglob("*"):
        if p.is_file() and is_image_file(p):
            imgs.append(p)

    if not imgs:
        print(f"[afd] No images found under: {afd_face_images_dir}")
        return

    for p in tqdm(imgs, desc="Copy AFD -> wildlife"):
        if verify_image(p):
            copy_image(p, wildlife_dir, prefix="afd")


def hf_find_columns(split):
    cols = split.column_names
    # guess image column
    img_col = None
    for c in ["image", "img", "pixel_values", "pixels"]:
        if c in cols:
            img_col = c
            break

    # guess label column
    label_col = None
    for c in ["label", "labels", "class", "category"]:
        if c in cols:
            label_col = c
            break

    return img_col, label_col


def hf_label_to_classname(ds, label_value):
    # Works if label is int with features['label'].names, or already a string
    feats = ds.features
    if isinstance(label_value, str):
        s = label_value.lower()
    else:
        # attempt int -> name
        try:
            if "label" in feats and hasattr(feats["label"], "names") and feats["label"].names:
                s = feats["label"].names[int(label_value)].lower()
            else:
                s = str(label_value).lower()
        except Exception:
            s = str(label_value).lower()

    # map to cats/dogs
    if "cat" in s:
        return "cats"
    if "dog" in s:
        return "dogs"
    # fallback: unknown -> skip (you only want cats/dogs from this HF dataset)
    return None


def export_hf_saved_to_disk(hf_saved_dir: Path, prepared_dir: Path, max_items: int = 0):
    if not hf_saved_dir.exists():
        print(f"[hf] Missing dir: {hf_saved_dir}")
        return

    ds = load_from_disk(str(hf_saved_dir))  # usually a DatasetDict with train/val
    splits = ds.keys() if hasattr(ds, "keys") else ["train"]

    for split_name in splits:
        split = ds[split_name] if hasattr(ds, "__getitem__") else ds
        img_col, label_col = hf_find_columns(split)

        if img_col is None or label_col is None:
            print(f"[hf] Could not find image/label columns in split '{split_name}'. Columns: {split.column_names}")
            return

        n = len(split)
        if max_items and max_items > 0:
            n = min(n, max_items)

        for i in tqdm(range(n), desc=f"Export HF {split_name} -> images"):
            row = split[i]
            cls = hf_label_to_classname(split, row[label_col])
            if cls is None:
                continue

            img = row[img_col]
            # Hugging Face Image type supports .save(); sometimes returns PIL.Image directly
            try:
                if hasattr(img, "convert"):
                    pil = img
                else:
                    pil = img.convert("RGB")
            except Exception:
                try:
                    pil = img["image"].convert("RGB")
                except Exception:
                    continue

            # ensure RGB, save as jpg
            pil = pil.convert("RGB")
            out_dir = prepared_dir / cls
            out_path = out_dir / f"hf_{split_name}_{i:06d}.jpg"
            try:
                pil.save(out_path, format="JPEG", quality=95)
            except Exception:
                pass


def main():
    project_root = Path(r"F:\Projects\AnimalClassification")
    raw_root = project_root / "data" / "datasets_raw"
    prepared_dir = project_root / "data" / "prepared"

    ensure_dirs(prepared_dir)

    # 1) Locate AFD folder robustly (handles ( ) vs （ ） and weird naming)
    afd_parent = None
    for cand in raw_root.iterdir():
        if cand.is_dir() and cand.name.strip().lower().startswith("afd"):
            afd_parent = cand
            break
    
    if afd_parent is None:
        print("[afd] Could not locate any folder starting with 'AFD' under datasets_raw")
    else:
        afd_zip_dir = afd_parent / "face images"
    
        if not afd_zip_dir.exists():
            print(f"[afd] Found AFD folder: {afd_parent}, but missing 'face images' subfolder.")
        else:
            unzip_all_zips_in_dir(afd_zip_dir, afd_zip_dir)
            ingest_afd_unzipped(afd_zip_dir, prepared_dir)

    # 3) Copy AFHQv2 -> prepared
    afhq_dir = raw_root / "AFHQv2"
    ingest_afhq(afhq_dir, prepared_dir)

    # 4) Copy Microsoft PetImages -> prepared (skip corrupted)
    microsoft_dir = raw_root / "microsoft"
    ingest_microsoft(microsoft_dir, prepared_dir)

    # 5) Export Hugging Face saved_to_disk -> prepared cats/dogs
    hf_dir = raw_root / "huggin_face"
    export_hf_saved_to_disk(hf_dir, prepared_dir, max_items=0)

    print("\nDone.")
    print(f"Prepared dataset at: {prepared_dir}")
    for cls in ["cats", "dogs", "wildlife"]:
        count = sum(1 for _ in (prepared_dir / cls).rglob("*") if _.is_file() and is_image_file(_))
        print(f"  {cls}: {count} images")


if __name__ == "__main__":
    main()