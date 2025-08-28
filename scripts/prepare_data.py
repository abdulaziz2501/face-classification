import yaml
from pathlib import Path
from facecls.data import unzip_if_needed, make_splits_if_needed, has_images

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    from facecls.data import build_dataloaders  # just to reuse heuristics
    # Build once to trigger unzip/split
    loaders, classes = build_dataloaders(cfg)
    print("Prepared. Classes:", classes)
