import yaml, torch, torch.nn as nn
from facecls.data import build_dataloaders
from facecls.model_zoo import build_model
from facecls.engine import evaluate
from facecls.utils import device

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    dev = device()
    loaders, classes = build_dataloaders(cfg)
    model = build_model(cfg["model"]["backbone"], len(classes), cfg["model"]["freeze_backbone"]).to(dev)

    ckpt = torch.load(cfg["paths"]["best_ckpt"], map_location=dev)
    model.load_state_dict(ckpt["state_dict"])
    crit = nn.CrossEntropyLoss()
    split = "Test" if loaders["test"] is not None else "Valid"
    loader = loaders["test"] or loaders["valid"]
    evaluate(model, loader, crit, dev, classes, split=split)
