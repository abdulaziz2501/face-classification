import yaml
from pathlib import Path

# Always resolve repository root no matter where you run from
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "default.yaml"

if not CFG_PATH.exists():
    raise FileNotFoundError(f"Config not found: {CFG_PATH}  "
                            f"(current cwd: {Path.cwd()})")

# Add src to sys.path so `import facecls` works without install
import sys
sys.path.insert(0, str(ROOT / "src"))

from src.facecls.data import build_dataloaders

if __name__ == "__main__":
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    loaders, classes = build_dataloaders(cfg)
    print("Prepared. Classes:", classes)
