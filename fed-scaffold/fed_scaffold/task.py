import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss, \
    multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import os

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

# Suppress annoying UserWarnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# FLAIR dataset coarse-grained labels mapping
COARSE_LABELS = {
    "structure": 0, "equipment": 1, "material": 2, "outdoor": 3,
    "plant": 4, "food": 5, "animal": 6, "liquid": 7,
    "art": 8, "interior_room": 9, "light": 10, "recreation": 11,
    "celebration": 12, "fire": 13, "music": 14, "games": 15,
    "religion": 16}
ID2LABEL = {v: k for k, v in COARSE_LABELS.items()}
NUM_CLASSES = len(COARSE_LABELS)

EPOCHS = 5
LR = 5e-4
WEIGHT_DECAY = 1e-5

DROPOUT = 0.3  # Dropout value for classification head

ALPHA = 0.1
MU = 0.1


class FLAIRDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]

        # 1. Handle Image
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 2. Handle Labels (Multi-label)
        string_labels = item['labels']
        label_indices = [COARSE_LABELS[label] for label in string_labels]
        multi_hot_label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        multi_hot_label[label_indices] = 1.0

        return image, multi_hot_label


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """Get data transforms for training/validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def load_data_random(partition_id: int, num_partitions: int) -> DataLoader:  # load_data_v1
    """Create and load partitions with sequential data (IID)"""
    print(f"Loading data for client {partition_id}/{num_partitions}...")

    # Load the full training and validation sets
    train_hf = load_dataset("apple/flair", split="train", trust_remote_code=True)

    # --- IID Partitioning Logic ---
    total_train_size = len(train_hf)
    partition_size = total_train_size // num_partitions
    start_idx = partition_id * partition_size
    # Ensure the last partition gets all remaining data
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else total_train_size

    # Select the partition for the current client
    client_train_data = train_hf.select(range(start_idx, end_idx))

    # Create datasets with transforms
    train_dataset = FLAIRDataset(client_train_data, transform=get_transforms(is_train=True))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3, pin_memory=True)

    print(f"Client {partition_id} training set size: {len(train_dataset)}")
    return train_loader


def load_data_userid(partition_id: int, num_partitions: int) -> DataLoader:  # load_data_v2
    """Load original FLAIR partitions for a specific user_id. (Non-IID)"""
    print(f"Loading data for client {partition_id}/{num_partitions}...")

    # Load the full training and validation sets
    train_hf = load_dataset("apple/flair", split="train", trust_remote_code=True)

    # --- Partitioning by user_id ---
    unique_users = sorted(set(train_hf["user_id"]))  # all unique user_ids
    assert num_partitions == len(unique_users), (
        f"num_partitions={num_partitions} but dataset has {len(unique_users)} unique users"
    )

    # Pick the user_id for this client
    user_id = unique_users[partition_id]
    user_indices = [i for i, uid in enumerate(train_hf["user_id"]) if uid == user_id]

    # Select only this user's samples
    client_train_data = train_hf.select(user_indices)

    # Wrap in your dataset class with transforms
    train_dataset = FLAIRDataset(client_train_data, transform=get_transforms(is_train=True))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=False)

    print(f"Client {partition_id} (user_id={user_id}) training set size: {len(train_dataset)}")
    return train_loader


def load_data_primary_label(  # load_data
        partition_id: int,
        num_partitions: int,
        alpha: float = ALPHA,  # The key parameter to control non-IID level
) -> DataLoader:
    """Create and load partitions using DirichletPartitioner using a given label. (Non-IID)"""
    print(f"Loading data for client {partition_id}/{num_partitions}...")

    # Preprocess dataset (adds "primary_label")
    def add_primary_label(ds):
        return ds.map(lambda batch: {"primary_label": [labels[0] for labels in batch["labels"]]}, batched=True)

    fds = FederatedDataset(
        dataset="apple/flair",
        preprocessor=add_primary_label,
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=200,  # number of clients
                partition_by="primary_label",  # primary_label is the most skewed, using a random label as a label for
                # partitioning is not as skewed
                alpha=alpha,  # concentration parameter (smaller = more skew)
                seed=42,
            )
        },
    )

    client_train_data = fds.load_partition(partition_id)

    # Wrap into PyTorch datasets
    train_dataset = FLAIRDataset(client_train_data, transform=get_transforms(is_train=True))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3, pin_memory=True)

    print(f"Client {partition_id} training set size: {len(train_dataset)}")

    return train_loader


def load_val() -> DataLoader:
    fds = FederatedDataset(dataset="apple/flair", partitioners={"test": 1})
    val_hf = fds.load_split("test")
    val_dataset = FLAIRDataset(val_hf, transform=get_transforms(is_train=False))

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True)

    return val_loader


def get_model() -> nn.Module:
    """Create MobileNetV2 model for FLAIR classification."""
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )

    return model


def train(model: nn.Module, train_loader: DataLoader, epochs: int, device: torch.device) -> Dict:
    """Train the model and return metrics."""
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss


def test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """Test the model and return comprehensive metrics for multi-label classification."""
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            preds = (torch.sigmoid(output) > 0.5).int()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_loss = test_loss / len(test_loader)

    accuracy = accuracy_score(all_labels, all_preds)  # Exact match ratio
    h_loss = hamming_loss(all_labels, all_preds)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    ml_cm = multilabel_confusion_matrix(all_labels, all_preds)

    return {
        "test_loss": avg_loss,
        "test_accuracy_exact_match": accuracy,
        "hamming_loss": h_loss,
        "test_micro_f1": micro_f1,
        "test_macro_f1": macro_f1,
        "multilabel_confusion_matrix": ml_cm,
        "true_labels": all_labels
    }


# --- Flower specific functions ---
def apply_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Apply parameters to the model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
