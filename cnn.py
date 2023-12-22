import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_data, train_epoch, evaluate, score
import os
import wandb

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

def train_cnn(epochs: int = 20, batch_size: int = 128, learning_rate: float = 1e-4,
              device: str = 'cpu', fold: int = -1, logging: bool = False, save_folder: str = '', ckpt_frequency: int = 0):
    device = torch.device(device)
    model = ImageClassifier()
    model.to(device)
    if logging:
        wandb.init(project="cifar")
        wandb.watch(model, log_freq=4)

    train_dataloader, val_dataloader, test_dataloader = load_data(batch_size, fold=fold, resize=False)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    print("Begin training loop...")
    for epoch in range(epochs):
        train_loss = train_epoch(train_dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, logging=logging)
        val_loss = evaluate(val_dataloader=val_dataloader, model=model, device=device)
        print(f"{f'Fold {fold + 1}/5 ' if fold > -1 else ''}Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        if logging:
            wandb.log({ "val": val_loss })
        if ckpt_frequency > 0 and (epoch + 1) % ckpt_frequency == 0:
            print(f"Saving checkpoint for epoch {epoch + 1} to {save_folder if save_folder != '' else 'current working directory'}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_folder, f"cifar_cnn{f'_fold{fold + 1}' if fold > -1 else ''}_epoch{epoch + 1}.pt"))

    print("Evaluating model on CIFAR-10 test dataset...")
    test_loss, test_acc = score(test_dataloader=test_dataloader, model=model, device=device)
    print(f"{f'Fold {fold + 1}/5 ' if fold > -1 else ''}Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}")
    
    print(f"Saving model to {save_folder if save_folder != '' else 'current working directory'}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc
    }, os.path.join(save_folder, f"cifar_cnn{f'_fold{fold + 1}' if fold > -1 else ''}.pt"))