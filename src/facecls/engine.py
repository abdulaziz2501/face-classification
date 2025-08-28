import torch, torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp=True):
    model.train(); total, correct, running = 0, 0, 0.0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            out = model(x); loss = criterion(out, y)
            loss.backward(); optimizer.step()
        running += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device, classes, split="Valid"):
    model.eval(); total, correct, running = 0, 0, 0.0
    all_pred, all_true = [], []
    for x, y in tqdm(loader, desc=f"Eval-{split}"):
        x, y = x.to(device), y.to(device)
        out = model(x); loss = criterion(out, y)
        running += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        all_pred += pred.cpu().tolist(); all_true += y.cpu().tolist()
    print(f"{split} | loss={running/total:.4f} acc={correct/total:.4f}")
    print(classification_report(all_true, all_pred, target_names=classes, digits=4))
    print(confusion_matrix(all_true, all_pred))
    return running/total, correct/total
