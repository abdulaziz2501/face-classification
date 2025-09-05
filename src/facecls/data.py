from pathlib import Path
import random, shutil, zipfile
from torchvision import datasets
from torch.utils.data import DataLoader
from .augmentations import get_transforms

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def unzip_if_needed(zip_path: Path, interim_dir: Path):
    interim_dir.mkdir(parents=True, exist_ok=True)
    mark = interim_dir / ".unzipped"
    if mark.exists():
        return interim_dir
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(interim_dir)
    mark.touch()
    return interim_dir

def has_images(dirp: Path):
    return any(dirp.rglob(f"*{ext}") for ext in IMG_EXTS)

def make_splits_if_needed(dataset_root: Path, processed_dir: Path, splits):
    # Agar train/valid/test mavjud bo'lmasa -> class-per-dir ni 80/10/10 bo'lib ko'chiramiz
    if any((processed_dir / s).exists() for s in ["train","valid","test"]):
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir() and has_images(d)]
    for cls in class_dirs:
        imgs = []
        for ext in IMG_EXTS:
            imgs += list(cls.glob(f"*{ext}"))
        random.shuffle(imgs)
        n = len(imgs); n_tr = int(splits[0]*n); n_val = int(splits[1]*n); n_te = n - n_tr - n_val
        parts = {"train": imgs[:n_tr], "valid": imgs[n_tr:n_tr+n_val], "test": imgs[n_tr+n_val:]}
        for split, files in parts.items():
            out = processed_dir / split / cls.name
            out.mkdir(parents=True, exist_ok=True)
            for f in files: shutil.copy2(f, out / f.name)

def build_dataloaders(cfg):
    zip_path = Path(cfg["paths"]["zip_path"])
    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    img_size = cfg["data"]["img_size"]
    bs = cfg["data"]["batch_size"]; nw = cfg["data"]["num_workers"]
    splits = cfg["data"]["splits"]

    unzip_if_needed(zip_path, interim_dir)

    # dataset_root ni topish (unzipped ichida eng yuqori darajadagi class papkalar)
    candidates = [p for p in interim_dir.rglob("*") if p.is_dir() and any((p / x).is_dir() for x in [".",".."])]
    dataset_root = interim_dir
    # oddiy heuristika: class-per-dir bor joyni topamiz
    for p in interim_dir.rglob("*"):
        if p.is_dir():
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if subdirs and any(has_images(sd) for sd in subdirs):
                dataset_root = p
                break

    # agar processed tayyor bo'lmasa, yaratamiz
    make_splits_if_needed(dataset_root, processed_dir, splits)

    tf_train, tf_eval = get_transforms(img_size)

    train_ds = datasets.ImageFolder(processed_dir / "train", transform=tf_train)
    val_ds   = datasets.ImageFolder(processed_dir / "valid", transform=tf_eval)
    test_ds  = None
    if (processed_dir / "test").exists():
        test_ds = datasets.ImageFolder(processed_dir / "test", transform=tf_eval)

    loaders = {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True),
        "valid": DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True) if test_ds else None
    }
    return loaders, train_ds.classes
