import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.models import vit_b_16, ViT_B_16_Weights
from utils import load_data, train_epoch, evaluate, score
import os
import math
from typing import Tuple
import wandb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, n_patches: int):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(n_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, n_patches, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        d_embed = d_head * n_heads
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_embed)

        self.out_proj = nn.Linear(d_embed, d_model)

    def forward(self, x):
        b, p, d = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(b, p, self.n_heads, 3 * self.d_head).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        logits = q @ k.transpose(-2, -1)
        logits /= self.d_head ** -0.5

        out = F.softmax(logits, dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, p, -1)

        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_heads: int, d_feedforward: int, dropout: float = 0.0):
        super(EncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, d_head, n_heads)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_feedforward, d_model)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attention_out = self.attention(self.layer_norm1(x))
        x = x + self.dropout(attention_out)
        
        fcn_out = self.fcn(self.layer_norm2(x))
        x = x + self.dropout(fcn_out)
        return x


class ImageClassifier(nn.Module):
    def __init__(self, n_patches: int, patch_size: Tuple[int, int], 
                 d_model: int, d_head: int, n_heads: int, d_feedforward: int, n_layers: int): # d_head = d_model // n_heads
        super(ImageClassifier, self).__init__()

        self.d_model = d_model
        self.patch_size = patch_size

        # a preprocessing convolutional layer proven to boost ViT training performance (https://arxiv.org/abs/2106.14881)
        self.conv_proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = PositionalEncoding(d_model, n_patches + 1)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_head, n_heads, d_feedforward) for _ in range(n_layers)])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 10)
        )

    
    def forward(self, x):
        b, c, h, w = x.shape

        patches = patches.reshape(b, self.d_model, (h // self.patch_size[0]) * (w // self.patch_size[1])).permute(0, 2, 1)

        embeddings = self.pos_encoding(torch.cat((self.cls_token.repeat(b, 1, 1), patches), dim=1))

        for layer in self.encoder:
            embeddings = layer(embeddings)

        out = self.head(embeddings[:, 0])
        return out


def train_vit(epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-4,
              device: str = 'cpu', pretrained: bool = True, fold: int = -1, logging: bool = False, save_folder: str = '', ckpt_frequency: int = 0):
    device = torch.device(device)
    if pretrained:
        model = model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(768, 10)
    else:
        model = ImageClassifier(196, (16, 16), 768, 64, 12, 3072, 12)
    model.to(device)
    if logging:
        wandb.init(project="cifar")
        wandb.watch(model, log_freq=16)

    train_dataloader, val_dataloader, test_dataloader = load_data(batch_size, fold=fold)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    print("Begin training loop...")
    for epoch in range(epochs):
        train_loss = train_epoch(train_dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, logging=logging)
        val_loss = evaluate(val_dataloader=val_dataloader, model=model, device=device)
        print(f"{f'Fold {fold + 1}/5 ' if fold > -1 else ''}Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        if logging:
            wandb.log({ "lr": optimizer.param_groups[0]['lr'], "val": val_loss })
        if ckpt_frequency > 0 and (epoch + 1) % ckpt_frequency == 0:
            print(f"Saving checkpoint for epoch {epoch + 1} to {save_folder if save_folder != '' else 'current working directory'}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_folder, f"cifar_vit{'_pretrained' if pretrained else ''}{f'_fold{fold + 1}' if fold > -1 else ''}_epoch{epoch + 1}.pt"))

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
    }, os.path.join(save_folder, f"cifar_vit{'_pretrained' if pretrained else ''}{f'_fold{fold + 1}' if fold > -1 else ''}.pt"))