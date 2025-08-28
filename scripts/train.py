import yaml, torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from src.facecls.utils import set_seed, device
from src.facecls.data import build_dataloaders
from src.facecls.model_zoo import build_model
from src.facecls.engine import train_one_epoch, evaluate

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    set_seed(cfg["seed"]); dev = device()

    loaders, classes = build_dataloaders(cfg)
    num_classes = len(classes)
    model = build_model(cfg["model"]["backbone"], num_classes, cfg["model"]["freeze_backbone"]).to(dev)

    crit = nn.CrossEntropyLoss()
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sch = ReduceLROnPlateau(opt, mode="min", factor=cfg["train"]["scheduler"]["factor"],
                            patience=cfg["train"]["scheduler"]["patience"], verbose=True)
    scaler = GradScaler(enabled=cfg["train"]["mixed_precision"])

    best_loss = 1e9; best_path = cfg["paths"]["best_ckpt"]
    for ep in range(1, cfg["train"]["epochs"]+1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], opt, crit, dev, scaler, amp=cfg["train"]["mixed_precision"])
        v_loss, v_acc = evaluate(model, loaders["valid"], crit, dev, classes, split="Valid")
        sch.step(v_loss)
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save({"state_dict": model.state_dict(),
                        "classes": classes,
                        "img_size": cfg["data"]["img_size"]}, best_path)
            print(f"Saved best to {best_path} (val_loss={best_loss:.4f})")

    # optional: final test
    if loaders["test"] is not None:
        evaluate(model, loaders["test"], crit, dev, classes, split="Test")
