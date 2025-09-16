
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from tqdm import tqdm

# Import the upgraded DIST wrapper
from DiT_DIST_updated import DiT_basic

# =====================
# Example dataset (replace with your real dataset)
# =====================
class DummyDataset(Dataset):
    def __init__(self, length=100, in_ch=6, size=224):
        super().__init__()
        self.length = length
        self.in_ch = in_ch
        self.size = size
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        before = torch.randn(self.in_ch, self.size, self.size)
        after  = torch.randn(self.in_ch, self.size, self.size)
        label  = torch.randint(0, 2, (1,)).item()
        return before, after, label

# =====================
# Train / Validate
# =====================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for before, after, label in tqdm(loader, desc='Training', leave=False):
        before, after, label = before.to(device), after.to(device), torch.tensor(label, device=device, dtype=torch.long)
        optimizer.zero_grad()
        loss, acc = model(before, after, labels_MP=label, labels_LNM=label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * before.size(0)
        total_acc  += acc * before.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for before, after, label in tqdm(loader, desc='Validation', leave=False):
            before, after, label = before.to(device), after.to(device), torch.tensor(label, device=device, dtype=torch.long)
            loss, acc = model(before, after, labels_MP=label, labels_LNM=label)
            total_loss += loss.item() * before.size(0)
            total_acc  += acc * before.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

if __name__ == '__main__':
    # Hyper-params
    batch_size = 4
    epochs = 5
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    train_set = DummyDataset(length=200, in_ch=6)
    val_set   = DummyDataset(length=50,  in_ch=6)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)

    # Model
    model = DiT_basic(basic_model='dist', in_ch=6, loss_f='focal', output_type='score').to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.85, 0.998), weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train loop
    best_val_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
              f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            print('Saved new best model!')
