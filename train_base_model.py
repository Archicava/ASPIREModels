"""
Base Model for Autism Spectrum Disorder (ASD) Detection.

A feedforward neural network trained on clinical features for ASD screening.
Uses the top 8 most predictive features (84% of predictive power).

Usage:
    python train_base_model.py
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
)
import joblib

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Activation function name mapping
ACTIVATIONS = {
    "ReLU": th.nn.ReLU,
    "LeakyReLU": th.nn.LeakyReLU,
    "ELU": th.nn.ELU,
    "GELU": th.nn.GELU,
    "SiLU": th.nn.SiLU,
    "Tanh": th.nn.Tanh,
}

# Fixed configuration (not part of grid search)
FIXED_CONFIG = {
    # Paths
    "data_dir": Path(__file__).parent.parent / "aspire" / "data",
    "output_dir": Path(__file__).parent / "_output",

    # System
    "device": "cuda" if th.cuda.is_available() else "cpu",
    "seed": 42,

    # Training constraints
    "num_epochs": 100,
    "early_stopping_patience": 15,
    "save_every_n_epochs": 10,
    "test_size": 0.2,
    "val_size": 0.15,

    # Minimum precision threshold for early stopping
    "min_precision": 0.80,
}

# Grid search configuration
# Each key can be a single value or a list of values to search over
GRID_SEARCH_CONFIG = {
    # Training hyperparameters
    "batch_size": [32],
    "learning_rate": [1e-3, 5e-4, 1e-4],

    # Model architecture
    "hidden_sizes": [
        [64, 32],      # Original architecture
        [32, 16],      # Smaller
        [128, 64, 32], # Deeper
    ],
    "dropout": [0.2, 0.3, 0.4],
    "activation": ["ReLU"],
    "batchnorm": [True],

    # Output
    "nr_classes": [2],  # Binary: Healthy, ASD
}

# TOP 8 FEATURES (84% predictive power)
SELECTED_FEATURES = [
    'Developmental milestones- global delay (G), motor delay (M), cognitive delay (C)',
    'IQ/DQ',
    'ICD',
    'Language disorder Y= present, N=absent',
    'Language development: delay, normal=N, absent=A',
    'Dysmorphysm y=present, no=absent',
    'Behaviour disorder- agressivity, agitation, irascibility',
    'Neurological Examination; N=normal, text = abnormal; free cell = examination not performed ???'
]

# Simplified feature names for easier reference
FEATURE_NAMES_SIMPLE = [
    'developmental_milestones',
    'iq_dq',
    'intellectual_disability',
    'language_disorder',
    'language_development',
    'dysmorphism',
    'behaviour_disorder',
    'neurological_exam'
]

# Feature value mappings for user-friendly input
FEATURE_VALUES = {
    'developmental_milestones': {'N': 'Normal', 'G': 'Global delay', 'M': 'Motor delay', 'C': 'Cognitive delay'},
    'iq_dq': 'numeric (0-150)',
    'intellectual_disability': {'N': 'None', 'F70.0': 'Mild', 'F71': 'Moderate', 'F72': 'Severe'},
    'language_disorder': {'N': 'No', 'Y': 'Yes'},
    'language_development': {'N': 'Normal', 'delay': 'Delayed', 'A': 'Absent'},
    'dysmorphism': {'NO': 'No', 'Y': 'Yes'},
    'behaviour_disorder': {'N': 'No', 'Y': 'Yes'},
    'neurological_exam': {'N': 'Normal', 'other': 'Abnormal (describe)'}
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and combine ASD and healthy patient datasets.

    Args:
        data_dir: Path to data directory containing Excel files

    Returns:
        (features_df, labels) tuple
    """
    asd_path = data_dir / 'patients_db.xlsx'
    healthy_path = data_dir / 'healthy_patients_gpt2.xlsx'

    asd_df = pd.read_excel(asd_path, sheet_name='asd_patients')
    healthy_df = pd.read_excel(healthy_path, sheet_name='healthy_patients')

    combined_df = pd.concat([asd_df, healthy_df], ignore_index=True)

    X = combined_df[SELECTED_FEATURES].copy()
    y = combined_df['Diagnosis'].values

    return X, y


# ==============================================================================
# PREPROCESSING
# ==============================================================================

class ClinicalPreprocessor:
    """
    Preprocessor for clinical features.

    Handles:
    - Numeric features: StandardScaler normalization
    - Categorical features: LabelEncoder + StandardScaler
    - Missing values: Imputation with defaults
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_cols = ['IQ/DQ']
        self.categorical_cols = [f for f in SELECTED_FEATURES if f != 'IQ/DQ']
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        """Fit preprocessor on training data."""
        X = X.copy()

        # Convert IQ/DQ to numeric
        X['IQ/DQ'] = pd.to_numeric(X['IQ/DQ'], errors='coerce')

        # Fit label encoders for categorical columns
        for col in self.categorical_cols:
            X[col] = X[col].fillna('_missing_').astype(str)
            all_values = list(X[col].unique()) + ['_missing_', '_unknown_']
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(all_values)

        # Fit scaler on all features (after encoding)
        X_encoded = self._encode(X)
        self.scaler.fit(X_encoded)
        self.fitted = True

        return self

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """Encode features without scaling."""
        X = X.copy()
        X['IQ/DQ'] = pd.to_numeric(X['IQ/DQ'], errors='coerce').fillna(70)

        for col in self.categorical_cols:
            X[col] = X[col].fillna('_missing_').astype(str)
            known_classes = set(self.label_encoders[col].classes_)
            X[col] = X[col].apply(lambda x: x if x in known_classes else '_unknown_')
            X[col] = self.label_encoders[col].transform(X[col])

        return X[SELECTED_FEATURES].values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted preprocessor."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        X_encoded = self._encode(X)
        return self.scaler.transform(X_encoded)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_config(self) -> dict:
        """Get preprocessor configuration for saving."""
        return {
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'label_encoder_classes': {
                col: list(le.classes_) for col, le in self.label_encoders.items()
            }
        }


# ==============================================================================
# DATASET
# ==============================================================================

class ClinicalDataset(th.utils.data.Dataset):
    """
    PyTorch Dataset for clinical feature data.

    Handles tabular data with optional GPU preloading.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, device: str = "cpu"):
        """
        Args:
            features: numpy array of shape (N, num_features)
            labels: numpy array of shape (N,)
            device: Device to store tensors on
        """
        self.device = device

        # Convert to tensors
        self.features = th.from_numpy(features).float()
        self.labels = th.from_numpy(labels).float().unsqueeze(1)

        # Move to device
        if device != "cpu":
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    def get_memory_usage(self) -> str:
        """Return string describing memory usage."""
        feat_bytes = self.features.element_size() * self.features.nelement()
        label_bytes = self.labels.element_size() * self.labels.nelement()
        total_kb = (feat_bytes + label_bytes) / 1024
        return f"{total_kb:.2f} KB on {self.device}"


# ==============================================================================
# MODEL
# ==============================================================================

class BaseASDDetector(th.nn.Module):
    """
    Configurable feedforward neural network for ASD detection.

    Supports grid search over architecture hyperparameters.

    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes (e.g., [64, 32])
        dropout: Dropout rate for regularization
        batchnorm: Whether to use batch normalization
        activation: Activation function class
        nr_classes: Number of output classes (2 for binary)

    Input: (B, input_size)
    Output: (B, 1) probabilities for binary classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = None,
        dropout: float = 0.3,
        batchnorm: bool = True,
        activation: type = th.nn.ReLU,
        nr_classes: int = 2,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        # Store config for reference
        self.config = {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "batchnorm": batchnorm,
            "activation": activation.__name__ if hasattr(activation, '__name__') else str(activation),
            "nr_classes": nr_classes,
        }

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(th.nn.Linear(prev_size, hidden_size))
            if batchnorm:
                layers.append(th.nn.BatchNorm1d(hidden_size))
            layers.append(activation())
            layers.append(th.nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer (binary classification with sigmoid)
        layers.append(th.nn.Linear(prev_size, 1))
        layers.append(th.nn.Sigmoid())

        self.network = th.nn.Sequential(*layers)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_one_epoch(
    model: th.nn.Module,
    loader: th.utils.data.DataLoader,
    criterion: th.nn.Module,
    optimizer: th.optim.Optimizer,
    device: str,
    data_on_gpu: bool = False
) -> float:
    """
    Train for one epoch.

    Args:
        model: The model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        data_on_gpu: If True, skip .to(device) calls

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in loader:
        if not data_on_gpu:
            features = features.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return total_loss / total_samples


def validate(
    model: th.nn.Module,
    loader: th.utils.data.DataLoader,
    criterion: th.nn.Module,
    device: str,
    data_on_gpu: bool = False
) -> tuple[float, dict]:
    """
    Validate model.

    Args:
        model: The model to validate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use
        data_on_gpu: If True, skip .to(device) calls

    Returns:
        (loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with th.no_grad():
        for features, labels in loader:
            if not data_on_gpu:
                features = features.to(device)
                labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)

            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(all_labels)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }

    return avg_loss, metrics


def print_metrics(metrics: dict, class_names: list[str] = None):
    """Print metrics summary."""
    if class_names is None:
        class_names = ["Healthy", "ASD"]

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

class LoggerManager:
    """
    Centralized logging manager for grid search training.

    Creates:
    - Master log: Complete log of all runs in grid_search directory
    - Run logs: Individual log file for each run
    - Console output: Real-time feedback
    """

    def __init__(self, grid_dir: Path):
        self.grid_dir = grid_dir
        self.logs_dir = grid_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.master_logger = self._create_master_logger()
        self.run_logger = None
        self.current_run_idx = None

    def _create_master_logger(self) -> logging.Logger:
        """Create master logger that logs everything."""
        logger = logging.getLogger("grid_search_master")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        master_log_path = self.logs_dir / "master_training.log"
        fh = logging.FileHandler(master_log_path)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def create_run_logger(self, run_idx: int, run_name: str, run_dir: Path) -> logging.Logger:
        """Create a dedicated logger for a specific run."""
        if self.run_logger is not None:
            self._close_run_logger()

        self.current_run_idx = run_idx

        logger_name = f"run_{run_idx:03d}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers = []

        run_logs_dir = run_dir / "logs"
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        run_log_path = run_logs_dir / "training.log"

        central_run_log_path = self.logs_dir / f"run_{run_idx:03d}_{run_name}.log"

        fh_run = logging.FileHandler(run_log_path)
        fh_run.setLevel(logging.INFO)

        fh_central = logging.FileHandler(central_run_log_path)
        fh_central.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh_run.setFormatter(formatter)
        fh_central.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh_run)
        logger.addHandler(fh_central)
        logger.addHandler(ch)

        self.run_logger = logger
        return logger

    def _close_run_logger(self):
        """Close and cleanup current run logger."""
        if self.run_logger is not None:
            for handler in self.run_logger.handlers[:]:
                handler.close()
                self.run_logger.removeHandler(handler)
            self.run_logger = None

    def log_master(self, message: str, level: str = "info"):
        """Log to master logger only."""
        log_func = getattr(self.master_logger, level.lower(), self.master_logger.info)
        log_func(message)

    def close(self):
        """Close all loggers."""
        self._close_run_logger()
        for handler in self.master_logger.handlers[:]:
            handler.close()
            self.master_logger.removeHandler(handler)


def setup_grid_search_dir(base_dir: Path) -> Path:
    """Create timestamped grid search output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_dir = base_dir / f"grid_search_{timestamp}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    return grid_dir


def save_checkpoint(
    model: th.nn.Module,
    optimizer: th.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if hasattr(model, "config"):
        checkpoint["model_config"] = model.config
    th.save(checkpoint, path)


# ==============================================================================
# GRID SEARCH
# ==============================================================================

def generate_grid_combinations(grid_config: dict) -> list[dict]:
    """
    Generate all combinations of hyperparameters from grid config.

    Args:
        grid_config: Dict where each value is a list of options

    Returns:
        List of dicts, each representing one hyperparameter combination
    """
    keys = list(grid_config.keys())
    values = [grid_config[k] if isinstance(grid_config[k], list) else [grid_config[k]]
              for k in keys]

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def config_to_run_name(config: dict) -> str:
    """
    Generate a short descriptive name for a run based on its config.

    Args:
        config: Hyperparameter configuration dict

    Returns:
        String name for the run
    """
    parts = []

    # Hidden sizes
    hidden = config.get("hidden_sizes", [64, 32])
    if hidden:
        parts.append(f"H{'_'.join(map(str, hidden))}")

    # Learning rate
    lr = config.get("learning_rate", 1e-3)
    parts.append(f"lr{lr:.0e}")

    # Dropout
    drop = config.get("dropout", 0.3)
    parts.append(f"d{drop}")

    # Activation
    activ = config.get("activation", "ReLU")
    parts.append(activ)

    # BatchNorm
    if not config.get("batchnorm", True):
        parts.append("noBN")

    return "_".join(parts)


# ==============================================================================
# SINGLE RUN TRAINING
# ==============================================================================

def train_single_run(
    run_config: dict,
    fixed_config: dict,
    train_loader: th.utils.data.DataLoader,
    val_loader: th.utils.data.DataLoader,
    output_dir: Path,
    run_idx: int,
    total_runs: int,
    logger_manager: LoggerManager = None,
    data_on_gpu: bool = False,
) -> dict:
    """
    Train a single model with given hyperparameters.

    Args:
        run_config: Hyperparameters for this run
        fixed_config: Fixed configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory for this run's outputs
        run_idx: Current run index (1-based)
        total_runs: Total number of runs
        logger_manager: LoggerManager instance
        data_on_gpu: If True, data is preloaded on GPU

    Returns:
        Dict with run results
    """
    # Create output directories
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Setup logging
    run_name = config_to_run_name(run_config)
    if logger_manager is not None:
        logger = logger_manager.create_run_logger(run_idx, run_name, output_dir)
    else:
        logger = logging.getLogger(f"run_{run_idx}")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        fh = logging.FileHandler(output_dir / "logs" / "training.log")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info("=" * 70)
    logger.info(f"RUN {run_idx}/{total_runs}: {run_name}")
    logger.info("=" * 70)

    # Save run config
    run_config_serializable = deepcopy(run_config)
    for k, v in run_config_serializable.items():
        if isinstance(v, (list, tuple)):
            run_config_serializable[k] = list(v)

    with open(output_dir / "logs" / "run_config.json", "w") as f:
        json.dump(run_config_serializable, f, indent=2)

    logger.info(f"Config: {run_config}")

    # Get activation function
    activation_name = run_config.get("activation", "ReLU")
    activation = ACTIVATIONS.get(activation_name, th.nn.ReLU)

    # Get input size from data loader
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1]

    # Create model
    logger.info("\nCreating model...")
    model = BaseASDDetector(
        input_size=input_size,
        hidden_sizes=run_config.get("hidden_sizes", [64, 32]),
        dropout=run_config.get("dropout", 0.3),
        batchnorm=run_config.get("batchnorm", True),
        activation=activation,
        nr_classes=run_config.get("nr_classes", 2),
    )
    model = model.to(fixed_config["device"])
    logger.info(f"Model config: {model.config}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = th.nn.BCELoss()
    optimizer = th.optim.Adam(
        model.parameters(),
        lr=run_config.get("learning_rate", 1e-3),
        weight_decay=1e-4
    )
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training history
    history = {
        "run_config": run_config_serializable,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_recall": [],
        "val_precision": [],
        "val_f1": [],
    }

    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    best_recall = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, fixed_config["num_epochs"] + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, fixed_config["device"],
            data_on_gpu=data_on_gpu
        )

        # Validate
        val_loss, metrics = validate(
            model, val_loader, criterion, fixed_config["device"],
            data_on_gpu=data_on_gpu
        )

        scheduler.step(metrics["recall"])

        # Update history
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(metrics["recall"])
        history["val_precision"].append(metrics["precision"])
        history["val_f1"].append(metrics["f1"])

        # Log progress
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{fixed_config['num_epochs']} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Recall: {metrics['recall']:.4f} | "
                        f"F1: {metrics['f1']:.4f}")

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, metrics,
            output_dir / "checkpoints" / "latest.pth"
        )

        # Save best model (optimize for recall with minimum precision)
        if metrics["recall"] > best_recall and metrics["precision"] >= fixed_config["min_precision"]:
            best_recall = metrics["recall"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, metrics,
                output_dir / "checkpoints" / "best_model.pth"
            )
            logger.info(f"  -> New best model! Recall: {metrics['recall']:.4f}")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % fixed_config["save_every_n_epochs"] == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth"
            )

        # Save history
        with open(output_dir / "logs" / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if patience_counter >= fixed_config["early_stopping_patience"]:
            logger.info(f"\nEarly stopping at epoch {epoch} "
                        f"(no improvement for {patience_counter} epochs)")
            break

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best recall: {best_recall:.4f} (epoch {best_epoch})")

    # Load best model and get final metrics
    best_checkpoint_path = output_dir / "checkpoints" / "best_model.pth"
    if best_checkpoint_path.exists():
        checkpoint = th.load(best_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    _, final_metrics = validate(
        model, val_loader, criterion, fixed_config["device"],
        data_on_gpu=data_on_gpu
    )

    logger.info(f"\nFinal validation metrics:")
    print_metrics(final_metrics)

    return {
        "run_idx": run_idx,
        "run_name": config_to_run_name(run_config),
        "config": run_config_serializable,
        "best_recall": best_recall,
        "best_epoch": best_epoch,
        "final_metrics": final_metrics,
        "total_params": total_params,
        "epochs_trained": len(history["epochs"]),
        "output_dir": str(output_dir),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """
    Main function that runs grid search over hyperparameter configurations.
    """
    # Set seed
    th.manual_seed(FIXED_CONFIG["seed"])
    np.random.seed(FIXED_CONFIG["seed"])

    # Generate all hyperparameter combinations
    combinations = generate_grid_combinations(GRID_SEARCH_CONFIG)
    total_runs = len(combinations)

    # Setup grid search output directory
    grid_dir = setup_grid_search_dir(FIXED_CONFIG["output_dir"])

    # Initialize logger manager
    logger_manager = LoggerManager(grid_dir)
    logger = logger_manager.master_logger

    logger.info("=" * 70)
    logger.info("GRID SEARCH - ASD DETECTION")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {grid_dir}")
    logger.info(f"Total configurations to test: {total_runs}")
    logger.info(f"Device: {FIXED_CONFIG['device']}")
    logger.info(f"Epochs per run: {FIXED_CONFIG['num_epochs']}")
    logger.info(f"Early stopping patience: {FIXED_CONFIG['early_stopping_patience']}")
    logger.info("")

    # Save grid search configuration
    grid_config_serializable = {
        "fixed_config": {k: str(v) if isinstance(v, Path) else v
                         for k, v in FIXED_CONFIG.items()},
        "grid_search_config": {k: [list(x) if isinstance(x, (list, tuple)) else x for x in v]
                               if isinstance(v, list) else v
                               for k, v in GRID_SEARCH_CONFIG.items()},
        "total_combinations": total_runs,
        "selected_features": SELECTED_FEATURES,
        "feature_names_simple": FEATURE_NAMES_SIMPLE,
    }
    with open(grid_dir / "grid_search_config.json", "w") as f:
        json.dump(grid_config_serializable, f, indent=2)

    # Load data
    logger.info("Loading data...")
    X, y = load_data(FIXED_CONFIG["data_dir"])

    logger.info(f"Total samples: {len(y)}")
    logger.info(f"Class distribution: ASD={sum(y)}, Healthy={len(y)-sum(y)}")
    logger.info(f"Features: {len(SELECTED_FEATURES)}")

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=FIXED_CONFIG["test_size"],
        stratify=y, random_state=FIXED_CONFIG["seed"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=FIXED_CONFIG["val_size"],
        stratify=y_train_val, random_state=FIXED_CONFIG["seed"]
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = ClinicalPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save preprocessor
    joblib.dump(preprocessor, grid_dir / "preprocessor.joblib")

    # Determine if we should preload data to GPU
    device = FIXED_CONFIG["device"]
    data_on_gpu = device != "cpu"

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ClinicalDataset(X_train_proc, y_train, device=device if data_on_gpu else "cpu")
    val_dataset = ClinicalDataset(X_val_proc, y_val, device=device if data_on_gpu else "cpu")
    test_dataset = ClinicalDataset(X_test_proc, y_test, device=device if data_on_gpu else "cpu")

    logger.info(f"Train dataset memory: {train_dataset.get_memory_usage()}")
    logger.info(f"Val dataset memory: {val_dataset.get_memory_usage()}")
    logger.info(f"Test dataset memory: {test_dataset.get_memory_usage()}")
    logger.info("")

    # Track all results
    all_results = []

    # Run grid search
    for run_idx, run_config in enumerate(combinations, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"STARTING RUN {run_idx}/{total_runs}")
        logger.info("=" * 70)

        batch_size = run_config.get("batch_size", 32)

        # Create data loaders
        num_workers = 0 if data_on_gpu else 4
        pin_memory = not data_on_gpu

        train_loader = th.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_loader = th.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Create run output directory
        run_name = config_to_run_name(run_config)
        run_dir = grid_dir / f"run_{run_idx:03d}_{run_name}"

        # Train this configuration
        result = train_single_run(
            run_config=run_config,
            fixed_config=FIXED_CONFIG,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=run_dir,
            run_idx=run_idx,
            total_runs=total_runs,
            logger_manager=logger_manager,
            data_on_gpu=data_on_gpu,
        )

        all_results.append(result)

        # Save intermediate results
        with open(grid_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info("")
        logger.info(f"Run {run_idx} complete: Recall = {result['best_recall']:.4f}")
        logger.info(f"Best epoch: {result['best_epoch']}, Params: {result['total_params']:,}")

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("=" * 70)

    # Sort by best recall
    sorted_results = sorted(all_results, key=lambda x: x["best_recall"], reverse=True)

    logger.info("")
    logger.info("TOP 5 CONFIGURATIONS:")
    logger.info(f"{'Rank':<6} {'Recall':<10} {'Params':<12} {'Run Name'}")
    logger.info("-" * 70)
    for i, result in enumerate(sorted_results[:5], 1):
        logger.info(f"{i:<6} {result['best_recall']:<10.4f} {result['total_params']:<12,} {result['run_name']}")

    # Evaluate best model on test set
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATING BEST MODEL ON TEST SET")
    logger.info("=" * 70)

    best_run = sorted_results[0]
    best_checkpoint_path = Path(best_run["output_dir"]) / "checkpoints" / "best_model.pth"

    if best_checkpoint_path.exists():
        checkpoint = th.load(best_checkpoint_path)

        # Recreate model
        activation_name = best_run["config"].get("activation", "ReLU")
        activation = ACTIVATIONS.get(activation_name, th.nn.ReLU)

        best_model = BaseASDDetector(
            input_size=X_train_proc.shape[1],
            hidden_sizes=best_run["config"].get("hidden_sizes", [64, 32]),
            dropout=best_run["config"].get("dropout", 0.3),
            batchnorm=best_run["config"].get("batchnorm", True),
            activation=activation,
        )
        best_model.load_state_dict(checkpoint["model_state_dict"])
        best_model = best_model.to(device)

        test_loader = th.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0 if data_on_gpu else 4,
            pin_memory=not data_on_gpu
        )

        _, test_metrics = validate(
            best_model, test_loader, th.nn.BCELoss(), device,
            data_on_gpu=data_on_gpu
        )

        logger.info(f"\nTest set metrics for best model ({best_run['run_name']}):")
        print_metrics(test_metrics)

        # Add test metrics to results
        sorted_results[0]["test_metrics"] = test_metrics

    # Save final summary
    summary = {
        "total_runs": total_runs,
        "best_run": sorted_results[0],
        "all_results_sorted": sorted_results,
    }
    with open(grid_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {grid_dir}")
    logger.info(f"Best configuration: {sorted_results[0]['run_name']}")
    logger.info(f"Best validation recall: {sorted_results[0]['best_recall']:.4f}")
    logger.info("")
    logger.info("Log files:")
    logger.info(f"  Master log: {grid_dir}/logs/master_training.log")
    logger.info(f"  Individual run logs: {grid_dir}/logs/run_XXX_*.log")
    logger.info("=" * 70)

    # Cleanup
    logger_manager.close()

    return all_results


if __name__ == "__main__":
    main()
