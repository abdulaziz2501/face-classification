import yaml, torch
from PIL import Image
from torchvision import transforms
from src.facecls.model_zoo import build_model
from src.facecls.utils import device

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    dev = device()
    ckpt = torch.load(cfg["paths"]["best_ckpt"], map_location=dev)
    classes = ckpt["classes"]; img_size = ckpt["img_size"]

    model = build_model("resnet18", len(classes), False).to(dev)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    import sys
    img_path = sys.argv[1]
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(img); pred = logits.argmax(1).item()
    print(f"Predicted class: {classes[pred]}")
