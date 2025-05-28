# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------
import argparse
import numpy as np
import os
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader
from monai.utils import set_determinism

from examples.build_multimodal_dataset import build_multimodal_dataset


def get_process_memory_usage(pid):
    """Returns the memory usage (RSS) of a process in megabytes."""
    try:
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # RSS in megabytes
    except psutil.NoSuchProcess:
        return 0  # Process no longer exists


def get_process_tree_memory_usage(pid):
    """Returns the total memory usage of a process and all its children."""
    total_memory = 0
    try:
        process = psutil.Process(pid)
        total_memory += get_process_memory_usage(pid)

        # Recursively add memory usage of child processes
        for child in process.children(recursive=True):
            total_memory += get_process_memory_usage(child.pid)

    except psutil.NoSuchProcess:
        pass  # process already finished.
    return total_memory



# MBConv block with DropConnect and SiLU activation, as used in EfficientNet.
class MBConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, drop_connect_rate=0.2):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.drop_connect_rate = drop_connect_rate

        self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)

        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)

        self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.act = nn.SiLU()

    def drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep_prob = 1 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor

    def forward(self, x):
        identity = x

        out = self.act(self.expand_bn(self.expand(x)))
        out = self.act(self.depthwise_bn(self.depthwise(out)))
        out = self.project_bn(self.project(out))

        if self.use_res_connect:
            out = self.drop_connect(out)
            return identity + out
        else:
            return out

# Lightweight Encoder Network
class EfficientNetLite2D(nn.Module):
    def __init__(self, in_channels=3, width_mult=1.0, out_dim=128):
        super().__init__()
        def c(ch):  # channel scaling function
            return max(8, int(ch * width_mult))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.SiLU()
        )

        self.block1 = MBConv2D(c(32), c(48), stride=1, expand_ratio=4)
        self.block2 = MBConv2D(c(48), c(64), stride=2, expand_ratio=4)
        self.block3 = MBConv2D(c(64), c(96), stride=2, expand_ratio=4)
        self.block4 = MBConv2D(c(96), c(128), stride=1, expand_ratio=4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim
        #self.proj = nn.Linear(c(128), out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class MultiModalAttentionNet_Lite(nn.Module):
    def __init__(self, num_modalities=3, d_model=256, output_dim=1, nhead=4):
        super().__init__()

        # Replace with new EfficientNetLite2D encoder
        self.encoders = nn.ModuleList([
            EfficientNetLite2D(in_channels=3, width_mult=0.5, out_dim=None)  # out_dim is unused in feature maps
            for _ in range(num_modalities)
        ])

        # Feature projection to d_model dimension
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, d_model, kernel_size=1),  # scale the network width 0.5, so 128*0.5=64
                nn.BatchNorm2d(d_model),
                nn.ReLU(),
            )
            for _ in range(num_modalities)
        ])

        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, imgs: list):
        token_list = []

        for i, img in enumerate(imgs):
            feat = self.encoders[i].stem(img)
            feat = self.encoders[i].block1(feat)
            feat = self.encoders[i].block2(feat)
            feat = self.encoders[i].block3(feat)
            feat = self.encoders[i].block4(feat)  # [B, 128, H', W']

            proj_feat = self.proj[i](feat)        # [B, d_model, H', W']
            B, C, H, W = proj_feat.shape
            tokens = proj_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            token_list.append(tokens)

        x = torch.cat(token_list, dim=1)  # [B, M*H*W, d_model]
        x = x.transpose(0, 1)             # [M*H*W, B, d_model]
        x = self.attn(x)                 # [M*H*W, B, d_model]
        x = x.mean(dim=0)                # [B, d_model]

        return self.classifier(x)



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Focal Loss to handle class imbalance.

        Args:
        - gamma (float): Focusing parameter. Higher values reduce the relative loss for well-classified examples.
        - alpha (Tensor, optional): Class weighting factor. Should be a tensor of shape (num_classes,).
        - reduction (str): Specifies the reduction mode ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute Focal Loss.

        Args:
        - logits (Tensor): Model outputs (raw scores before softmax), shape (batch_size, num_classes).
        - targets (Tensor): Ground-truth labels, shape (batch_size,).

        Returns:
        - Tensor: Computed focal loss.
        """
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        targets_one_hot = F.one_hot(
            targets, num_classes=logits.shape[1]
        ).float()  # One-hot encode targets

        ce_loss = -targets_one_hot * torch.log(
            probs + 1e-9
        )  # Cross-entropy loss (adding epsilon for numerical stability)
        focal_weight = (1 - probs) ** self.gamma  # Compute focal weight

        if self.alpha is not None:
            alpha_weight = self.alpha[targets].unsqueeze(1)  # Apply class weighting
            ce_loss *= alpha_weight

        loss = focal_weight * ce_loss  # Apply focal loss adjustment

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # If 'none', return per-sample loss




# Training function
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data_iter_step, batch in enumerate(tqdm(dataloader)):
        # input shape list of modality where each modalty has [BS, 3, 256, 256]
        images, labels = (
            [v.to(device) for v in batch["frames"]],
            batch["label"].to(device).long(),
        )
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images[0].size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        # profiler.step()

    return running_loss / total, correct / total


# Validation function with additional metrics
def validate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data_iter_step, batch in enumerate(dataloader):
            images, labels = (
                [v.to(device) for v in batch["frames"]],
                batch["label"].to(device).long(),
            )

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * images[0].size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)

    n_classes = all_probs.shape[1]
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    if n_classes == 2:
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        all_labels_bin = label_binarize(all_labels, classes=[0, 1]).ravel()
        auc_roc = roc_auc_score(all_labels_bin, all_probs[:, 1])
        auc_pr = average_precision_score(all_labels_bin, all_probs[:, 1])
    else:
        cm = confusion_matrix(all_labels, all_preds)
        sensitivity = recall_score(all_labels, all_preds, average='macro')
        specificity = None  # not defined
        all_labels_bin = label_binarize(all_labels, classes=np.arange(n_classes))
        auc_roc = roc_auc_score(all_labels_bin, all_probs, multi_class='ovo', average='macro')
        auc_pr = average_precision_score(all_labels_bin, all_probs, average='macro')

    metrics = {
        'Loss': val_loss / len(dataloader.dataset),
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Balanced Accuracy': bal_acc,
        'F1 Score': f1,
        'MCC': mcc
    }
    return metrics, cm


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")


def train_one_epoch_reg(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    for data_iter_step, batch in enumerate(tqdm(dataloader)):
        images, labels = (
            [v.to(device) for v in batch["frames"]],
            batch["label"].to(device).float(),
        )

        optimizer.zero_grad()
        outputs = model(images) # [B,1]
        loss = loss_fn(outputs.squeeze(1), labels) # huber_loss expect inputs have the same shape
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images[0].size(0)
        all_preds.extend(outputs.squeeze(1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        total += labels.size(0)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return running_loss / total, mae, r2


# Validation function with additional metrics
def validate_reg(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data_iter_step, batch in enumerate(dataloader):
            images, labels = (
                [v.to(device) for v in batch["frames"]],
                batch["label"].to(device).float(),
            )
            outputs = model(images)
            loss = loss_fn(outputs.squeeze(1), labels)
            val_loss += loss.item() * images[0].size(0)

            all_preds.extend(outputs.squeeze(1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute regression metrics
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    metrics = {
        "Loss": val_loss / len(dataloader.dataset),
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2,
    }
    return metrics


def get_args_parser():

    parser = argparse.ArgumentParser(
        "MultiModalAttentionNet_Lite for Image Classification", add_help=False
    )

    parser.add_argument(
        "--modalities",
        required=True,
        type=str,
        nargs="+",
        help="Patient dataset types (you can specify multiple)",
    )

    parser.add_argument(
        "--concept_id", default=-1, type=int, help="anatomic region to process"
    )


    parser.add_argument(
        "--task_mode",
        default="classification",
        type=str,
        choices=["regression", "classification"],
        help="nb_classes should be 1 when regression",
    )

    parser.add_argument(
        "--fast_slice_access",
        action="store_true",
        default=False,
        help="Fast low-variance slice sampling",
    )

    parser.add_argument(
        "--cache_rate",
        default=0.,
        type=float,
        help="Proportion of dataset to cache between epochs",
    )

    # Training parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU")
    parser.add_argument(
        "--val_batch_size", default=16, type=int, help="Validation batch size"
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs"
    )

    # Model parameters
    parser.add_argument("--input_size", default=224, type=int, help="Input image size")
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="Number of classification categories",
    )

    # Optimizer & Learning Rate
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for schedulers",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Number of warmup epochs"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/path/to/data/", type=str, help="Dataset path"
    )

    # Device and computation settings
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training/testing"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", default=10, type=int, help="Number of data loading workers"
    )

    # Checkpointing & Logging
    parser.add_argument(
        "--output_dir", default="./output", help="Path to save model checkpoints"
    )
    parser.add_argument("--log_dir", default="./logs", help="Path for logging")

    # Evaluation mode
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")

    args = parser.parse_args()

    return parser


# Training loop with model saving
def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    save_path,
):
    best_val_loss = float("inf")
    writer = SummaryWriter(args.log_dir)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics, cm = validate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["Loss"])

        writer.add_scalar("val/loss", val_metrics["Loss"], epoch + 1)
        writer.add_scalar("val/acc", val_metrics["Accuracy"], epoch + 1)
        writer.add_scalar("train/loss", train_loss, epoch + 1)
        writer.add_scalar("train/acc", train_acc, epoch + 1)

        main_pid = os.getpid()  # Get the PID of the main process
        total_memory_used = get_process_tree_memory_usage(main_pid)
        writer.add_scalar("max_mem", total_memory_used, epoch + 1)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        for key, value in val_metrics.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: N/A") # multiclass doesn't calculate specificity

        if val_metrics["Loss"] < best_val_loss:
            best_val_loss = val_metrics["Loss"]
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path)  # for DataParallel
            else:
                torch.save(model.state_dict(), save_path)
            print("Model saved with improved validation loss.")
            # Generate confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")
            plt.title(f"Confusion Matrix @ Epoch {epoch+1}")
            plt.savefig(os.path.join(args.output_dir, f"confusion_matrix_epoch_{epoch+1}.png"))
            plt.close()

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    plt.close()


# Training loop with model saving (Regression)
def train_reg(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    save_path):

    best_val_loss = float("inf")
    writer = SummaryWriter(log_dir=args.log_dir or "./runs/default")

    for epoch in range(epochs):
        train_loss, train_mae, train_r2 = train_one_epoch_reg(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate_reg(model, val_loader, loss_fn, device)

        writer.add_scalar("val/loss", val_metrics["Loss"], epoch + 1)
        writer.add_scalar("val/mae", val_metrics["MAE"], epoch + 1)
        writer.add_scalar("val/r2", val_metrics["R2 Score"], epoch + 1)
        writer.add_scalar("train/loss", train_loss, epoch + 1)
        writer.add_scalar("train/mae", train_mae, epoch + 1)
        writer.add_scalar("train/r2", train_r2, epoch + 1)

        # Log memory usage
        main_pid = os.getpid()
        total_memory_used = get_process_tree_memory_usage(main_pid) # this function is the same for 3d
        writer.add_scalar("max_mem", total_memory_used, epoch + 1)

        # Print progress
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R2: {train_r2:.4f}")
        for key, value in val_metrics.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: N/A") # multiclass doesn't calculate specificity

        if val_metrics["Loss"] < best_val_loss:
            best_val_loss = val_metrics["Loss"]
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path)  # for DataParallel
            else:
                torch.save(model.state_dict(), save_path)
            print("Model saved with improved validation loss.")



def main(args):
    # Ensure reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_determinism(seed)

    # Define the device
    device = torch.device(args.device)

    # Load the datasets
    dataset_train = build_multimodal_dataset(is_train=True, args=args)
    dataset_val = build_multimodal_dataset(is_train=False, args=args)

    # Create DataLoaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )


    # Initialize Multimodal model
    model = MultiModalAttentionNet_Lite(num_modalities=len(args.modalities), output_dim=args.nb_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr) #, weight_decay=1e-4)


    # Replace with actual dataloaders
    train_loader = data_loader_train
    val_loader = data_loader_val
    save_path = os.path.join(args.output_dir, "best_model.pth")

    if args.nb_classes > 1:
        criterion = FocalLoss(gamma=2.0)
        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            epochs=args.epochs,
            save_path=save_path,
    )
    else: # regression
        criterion = nn.HuberLoss(delta=1.0)
        train_reg(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            epochs=args.epochs,
            save_path=save_path,
        )
    main_pid = os.getpid()  # Get the PID of the main process
    total_memory_used = get_process_tree_memory_usage(main_pid)
    print(f"Total memory usage (process tree): {total_memory_used:.2f} MB")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
