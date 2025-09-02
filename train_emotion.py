import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_fer2013 import FER2013Dataset
from emotion_model import EmotionCNN
from tqdm import tqdm
import os

CSV_PATH = "../data/fer2013.csv"
SAVE_PATH = "../models/emotion_cnn.pt"
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 7
IMG_SIZE = 48
NUM_WORKERS = 2

def accuracy(logits, y):
    preds = logits.argmax(1)
    return (preds == y).float().mean().item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_set = FER2013Dataset(CSV_PATH, split="Training", target_size=IMG_SIZE)
    val_set   = FER2013Dataset(CSV_PATH, split="PublicTest", target_size=IMG_SIZE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            total_acc += (logits.argmax(1) == y).float().sum().item()
            n += x.size(0)
            pbar.set_postfix(loss=total_loss/n, acc=total_acc/n)

        # validation
        model.eval()
        val_acc, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_acc += (logits.argmax(1) == y).float().sum().item()
                val_n += x.size(0)
        val_acc /= val_n
        print(f"[Val] acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model -> {SAVE_PATH}")

    print("Best val acc:", best_val)

if __name__ == "__main__":
    main()
