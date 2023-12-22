import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_data, train_epoch, evaluate, score
import os
import wandb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1 if in_channels == out_channels else 2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        skip = self.skip(x)
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        return F.relu(skip + x)
    

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(64)
        
        self.block1 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64)
        )
        
        self.block2 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256)
        )

        self.block4 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=512),
            ResidualBlock(in_channels=512, out_channels=512)
        )
        
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


def train_resnet(epochs: int = 50, batch_size: int = 128, learning_rate: float = 0.1,
              device: str = 'cpu', fold: int = -1, logging: bool = False, save_folder: str = ''):
    device = torch.device(device)
    model = ImageClassifier()
    model.to(device)
    if logging:
        wandb.init(project="cifar")
        wandb.watch(model, log_freq=4)

    train_dataloader, val_dataloader, test_dataloader = load_data(batch_size, fold=fold, resize=False, crop=True)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.001)

    print("Begin training loop...")
    for epoch in range(epochs):
        train_loss = train_epoch(train_dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, logging=logging)
        val_loss = evaluate(val_dataloader=val_dataloader, model=model, device=device)
        scheduler.step(val_loss)
        if logging:
            wandb.log({ "lr": optimizer.param_groups[0]['lr'], "val": val_loss })
        print(f"{f'Fold {fold}/5 ' if fold > -1 else ''}Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

    print("Evaluating model on CIFAR-10 test dataset...")
    test_loss, test_acc = score(test_dataloader=test_dataloader, model=model, device=device)
    print(f"{f'Fold {fold}/5 ' if fold > -1 else ''}Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}")
    
    print(f"Saving model to {save_folder if save_folder != '' else 'current working directory'}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc
    }, os.path.join(save_folder, f"cifar_resnet{f'_fold{fold}' if fold > -1 else ''}.pt"))