# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import numpy as np
import os
from tqdm import tqdm
from typing import Optional
from pathlib import Path
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
    r2_score
)
from sklearn.preprocessing import label_binarize
import psutil
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from torchvision import transforms

from examples.build_dataset import build_dataset
from aireadi_loader.simpledatasets import AIReadiDataset
from models.model import EfficientNetB0Classifier, DinoTransformer

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

        images, labels = (
            batch["img"].to(device),
            batch["labels"].to(device).long(),
        )
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        predicted = torch.reshape(predicted, labels.size())
        X = predicted==labels
        correct += (predicted == labels).sum().item()
       
    
        total += labels.size(0)
        # profiler.step()

    return running_loss / total, correct / total


# Validation function with additional metrics
def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data_iter_step, batch in enumerate(tqdm(dataloader)):

            images, labels = (
                batch["img"].to(device),
                batch["labels"].to(device).long(),
            )

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    
    n_classes = all_probs.shape[1]
    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds).astype(int)
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
        'MCC': mcc,
        'Epoch': epoch
    }
    return metrics


def train_one_epoch_reg(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    for data_iter_step, batch in enumerate(tqdm(dataloader)):
        images, labels = (
            batch["frames"].to(device),
            batch["label"].to(device).float(),
        )

        optimizer.zero_grad()
        outputs = model(images) # [B,1]
        loss = loss_fn(outputs.squeeze(1), labels) # huber_loss expect inputs have the same shape
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
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
        for data_iter_step, batch in enumerate(tqdm(dataloader)):
            images, labels = (
                batch["frames"].to(device),
                batch["label"].to(device).float(),
            )
            outputs = model(images)
            loss = loss_fn(outputs.squeeze(1), labels)
            val_loss += loss.item() * images.size(0)

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
        "EfficientNetB0 for Image Classification", add_help=False
    )

    parser.add_argument(
        "--patient_dataset_type",
        default="slice",
        type=str,
        choices=["slice", "center_slice", "volume"],
        help="patient dataset type",
    )
    parser.add_argument(
        "--imaging",
        default="oct",
        type=str,
        choices=["oct", "cfp", "ir", "octa"],
        help="imaging type",
    )
    parser.add_argument(
        "--manufacturers_model_name",
        default="Spectralis",
        type=str,
        choices=["Spectralis", "Maestro2", "Triton", "Cirrus", "Eidon", "All"],
        help="device type",
    )
    parser.add_argument(
        "--anatomic_region",
        default="Macula",
        type=str,
        help="anatomic region to process",
    )
    parser.add_argument(
        "--octa_enface_imaging",
        default=None,
        type=str,
        choices=["superficial", "deep", "outer_retina", "choriocapillaris", None],
        help="OCTA enface slab type",
    )

    parser.add_argument(
        "--concept_id", default=-1, type=int, help="anatomic region to process"
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
        default=2,
        type=int,
        help="Number of classification categories",
    )

    # Optimizer & Learning Rate
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

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

    #label
    parser.add_argument(
        "--label", default="mhoccur_ca, Cancer (any type)", help="label to predict"
    )

    parser.add_argument(
        "--dataset_config_path", default="/path/to/data/", type=str, help="Dataset path"
    )

    parser.add_argument(
        "--img_dir", default="/path/to/data/", type=str, help="Dataset path"
    )

    parser.add_argument( "--img_type", default="cfp",type=str, help = "image modality")
    args = parser.parse_args()

    return parser

def write_logging_results(file_path, metrics):
    if os.path.exists(file_path):
        df1 = pd.read_csv(file_path, index_col=0) 
        df2 = pd.DataFrame(metrics, index = [metrics["Epoch"]]) 
        df3 = pd.concat([df1, df2], axis=0)  
        df3.to_csv(file_path)
    else:
        df = pd.DataFrame(metrics, index=[metrics["Epoch"]])
        df.to_csv(file_path)

# Training loop with model saving (Classification)
def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    save_path,
):
    best_val_loss = float("inf")
    writer = SummaryWriter(args.log_dir)
    
    

    train_file_path = save_path+"//train_result.csv"
    val_file_path = save_path+"//val_result.csv"
    test_file_path = save_path+"//test_result.csv"

    for epoch in range(epochs):
  
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        train_metrics = {"Epoch": epoch +1, "Train Loss": train_loss, "Train Acc": train_acc}
        val_metrics = validate(model, val_loader, loss_fn, device, epoch+1)
        
        test_metrics = validate(model, test_loader, loss_fn, device, epoch+1)
        write_logging_results(train_file_path, train_metrics)
        write_logging_results(val_file_path, val_metrics)
        write_logging_results(test_file_path, test_metrics)

        writer.add_scalar("test/loss", test_metrics["Loss"], epoch+1)
        writer.add_scalar("test/acc", test_metrics["Accuracy"], epoch+1)
        writer.add_scalar("test/auroc", test_metrics["AUC-ROC"], epoch+1)
        writer.add_scalar("val/loss", val_metrics["Loss"], epoch + 1)
        writer.add_scalar("val/acc", val_metrics["Accuracy"], epoch + 1)
        writer.add_scalar("val/auroc", val_metrics["AUC-ROC"], epoch+1) 
        writer.add_scalar("train/loss", train_loss, epoch + 1)
        writer.add_scalar("train/acc", train_acc, epoch + 1)
        

        main_pid = os.getpid()  # Get the PID of the main process
        total_memory_used = get_process_tree_memory_usage(main_pid)
        writer.add_scalar("max_mem", total_memory_used, epoch + 1)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print("VAL!")
        for key, value in val_metrics.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: N/A") # multiclass doesn't calculate specificity
        print("TEST!")
        for key, value in test_metrics.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: N/A") # multiclass doesn't calculate specificity

        if val_metrics["Loss"] < best_val_loss:
            best_val_loss = val_metrics["Loss"]
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path+"//best_model.pth")  # for DataParallel
            else:
                torch.save(model.state_dict(), save_path + "//best_model.pth")
            print("Model saved with improved validation loss.")


# Training loop with model saving (Regression)
def train_reg(
    model,
    train_loader,
    val_loader,
    test_loader,
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
        test_metrics = validate_reg(model, test_loader, loss_fn, device)

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
        for key, value in test_metrics.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: N/A") # multiclass doesn't calculate specificity

        if val_metrics["Loss"] < best_val_loss:
            best_val_loss = val_metrics["Loss"]
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path+"//best_model.pth")  # for DataParallel
            else:
                torch.save(model.state_dict(), save_path+"//best_model.pth")
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
    df = pd.read_csv(args.dataset_config_path)
    imgpath = args.img_dir
    if args.img_type == "ir":
        transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=15),  
        transforms.RandomCrop(size=224), # Convert to 3-channel RGB
        transforms.ToTensor()   
                            # Convert to tensor [0,1]
    ])  
        transform_test= transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(size=224),       # Adjust size as needed
        transforms.ToTensor()
    ])

    if args.img_type == "cfp":
        transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=15),  
        transforms.RandomCrop(size=224), # Convert to 3-channel RGB
        transforms.ToTensor()   
                            # Convert to tensor [0,1]
    ])  
        transform_test= transforms.Compose([
        transforms.CenterCrop(size=224),       # Adjust size as needed
        transforms.ToTensor()
    ])


    dataset_train = AIReadiDataset(df = df, img_path = imgpath, img_type = args.img_type, split="train", transform = transform_train, label=args.label)
    dataset_val = AIReadiDataset(df = df, img_path = imgpath, img_type = args.img_type, split="val", transform = transform_test, label=args.label)
    dataset_test = AIReadiDataset(df = df, img_path = imgpath, img_type = args.img_type, split="test", transform = transform_test, label=args.label)
#    dataset_train = build_dataset(split="train", args=args)
   # dataset_val = build_dataset(split="val", args=args)
  #  dataset_test = build_dataset(split="test", args=args)
    print(len(dataset_train),len(dataset_val), len(dataset_test))
    is_iterable_dataset = isinstance(dataset_train, IterableDataset)
    dataloader_shuffle = False if is_iterable_dataset else True

    # Create DataLoaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=dataloader_shuffle,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    # Initialize EfficientNet-B0 model
    model = DinoTransformer().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr) #, weight_decay=1e-4)


    # Replace with actual dataloaders
    train_loader = data_loader_train
    val_loader = data_loader_val
    test_loader = data_loader_test
    save_path = args.output_dir

    if args.nb_classes > 1:
        criterion = FocalLoss(gamma=2.0)
        train(
            model,
            train_loader,
            val_loader,
            test_loader,
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
            test_loader,
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
