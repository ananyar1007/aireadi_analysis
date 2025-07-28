# Copyright (c) 2025 Yuka Kihara and collaborators.
# All rights reserved.
#
# This source code is licensed under the terms found in the LICENSE file
# in the root directory of this source tree.
# --------------------------------------------------------

from examples.parse_args import get_args_parser
import numpy as np
import os
from tqdm import tqdm
from typing import Optional
from pathlib import Path
import datetime
import json
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
from aireadi_loader.AIREADIdataset import AIReadiDataset
from models.model import EfficientNetB0Classifier, DinoTransformer, TransformerFusion

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

def create_model_input(dct, device):
    model_input_dct = {}
    
    if "cfp" in dct:
        cfp = dct["cfp"]
        cfp = [img.to(device) for img in cfp]
        model_input_dct["cfp"] = cfp
    if "ir" in dct:
        ir = dct["ir"]
        ir = [img.to(device) for img in ir]
        model_input_dct["ir"] = ir

    if "clinical_meta" in dct:
        model_input_dct["clinical_meta"] = dct["clinical_meta"].to(device)
    return model_input_dct 
# Training function
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data_iter_step, dct in enumerate(tqdm(dataloader)):

        model_input_dct = create_model_input(dct, device)
        optimizer.zero_grad()
        outputs = model(model_input_dct)
        labels = dct["labels"].to(device)
        B = labels.size(0)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item() * B
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
       
    
        total += labels.size(0)

    return running_loss / total, correct / total


# Validation function with additional metrics
def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data_iter_step, dct in enumerate(tqdm(dataloader)):

            
            model_input_dct = create_model_input(dct, device)
            outputs = model(model_input_dct)
            labels = dct["labels"].to(device)
            B = labels.size(0)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * B

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
    auroc = {'Probs' : all_probs[:, 1],
        'Labels' : all_labels} 
    print(metrics)
    return metrics, auroc


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
    best_val_auc = 0
    writer = SummaryWriter(args.log_dir)
    
    

    train_file_path = save_path+"//train_result.csv"
    val_file_path = save_path+"//val_result.csv"
    test_file_path = save_path+"//test_result.csv"
    waiting_ctr = 0
    for epoch in range(epochs):
        waiting_ctr+=1
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        train_metrics = {"Epoch": epoch +1, "Train Loss": train_loss, "Train Acc": train_acc}
        val_metrics, predictions = validate(model, val_loader, loss_fn, device, epoch+1)
        test_metrics, predictions = validate(model, test_loader, loss_fn, device, epoch+1)

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
       
        if val_metrics["AUC-ROC"] > best_val_auc:
            probs = predictions['Probs']
            labels = predictions['Labels'] 
            dct_auroc = {'Probs': probs, 'Labels': labels}
            df = pd.DataFrame(dct_auroc)
            df.to_csv(save_path + "//Predictions.csv")
            best_val_auc = val_metrics["AUC-ROC"]
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path+"//best_model_" + str(epoch+1)+".pth")  # for DataParallel
            else:
                torch.save(model.state_dict(), save_path+"//best_model_" + str(epoch+1)+".pth")
            waiting_ctr = 0
            print("Model saved with improved validation auc.") 
        else:
            if(waiting_ctr>=2):
                print("No improvement - STOPPING!!")
                exit(0)


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
    
    
    ir_train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),                   # upscale to 256×256
        transforms.RandomCrop((224, 224)),               # random 224×224 patch
        transforms.RandomHorizontalFlip(),                  # randomly flip horizontally
        transforms.RandomRotation(degrees=30),         # new rotation augmentation
        transforms.ToTensor() ,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225])  
                            # Convert to tensor [0,1]
    ]) 

    ir_test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),                   # fixed 224×224
        transforms.ToTensor() ,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225])  
                            # Convert to tensor [0,1]
    ]) 


    cfp_train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),                   
        transforms.RandomCrop((224, 224)),               
        transforms.RandomRotation(degrees=30),   
        transforms.RandomHorizontalFlip(),# new rotation augmentation
        transforms.ToTensor() ,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225])  
                            # Convert to tensor [0,1]
    ]) 

    cfp_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),                   # fixed 224×224
        transforms.ToTensor() ,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225])  
                            # Convert to tensor [0,1]
    ]) 
    
    dataset_train = AIReadiDataset(df=df, img_type = args.img_type, img_path_cfp = args.cfp_img_path, img_path_ir=args.ir_img_path,split="train", cfp_transform=cfp_train_transforms, ir_transform = ir_train_transforms, label=args.label, clinical_data=args.clinical_data)
    dataset_val = AIReadiDataset(df=df, img_type = args.img_type, img_path_cfp = args.cfp_img_path, img_path_ir=args.ir_img_path,split="val", cfp_transform=cfp_test_transforms, ir_transform = ir_test_transforms, label=args.label, clinical_data=args.clinical_data)
    dataset_test = AIReadiDataset(df=df, img_type = args.img_type, img_path_cfp = args.cfp_img_path, img_path_ir=args.ir_img_path,split="test", cfp_transform=cfp_test_transforms, ir_transform = ir_test_transforms, label=args.label, clinical_data=args.clinical_data)
   
   
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

    model = TransformerFusion(len(args.clinical_data), args.num_layers, args.dropout)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, weight_decay=1e-4)


    # Replace with actual dataloaders
    train_loader = data_loader_train
    val_loader = data_loader_val
    test_loader = data_loader_test
    save_path = args.output_dir

    if args.nb_classes > 1:
        criterion =nn.CrossEntropyLoss() 
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
    else: 
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
    # Save args to a JSON file
    date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, '{}_{}'.format(args.experiment_name, date_string))
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=4, sort_keys=True)
        fh.write('\n')
    args.output_dir = output_dir
    main(args)
