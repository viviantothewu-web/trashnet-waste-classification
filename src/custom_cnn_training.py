"""Train the shared Custom CNN on TrashNet with a chosen seed."""

from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn

from src.custom_cnn import CustomCNN
from src.training import train_model


CUSTOM_CNN_EPOCHS = 50
CUSTOM_CNN_LR = 0.001


def run_custom_cnn(train_loader, val_loader, device, training_seed, save_dir="saved_models", num_epochs=CUSTOM_CNN_EPOCHS):
    """Train from scratch and save the best validation checkpoint."""
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    seed_dir = Path(save_dir) / f"Seed_{training_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / f"customcnn_seed{training_seed}.pth"

    model = CustomCNN(num_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CUSTOM_CNN_LR)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_path=str(checkpoint_path),
    )

    with (seed_dir / f"customcnn_seed{training_seed}_history.json").open("w") as f:
        json.dump({"seed": training_seed, "epochs": num_epochs,
                   "learning_rate": CUSTOM_CNN_LR,
                   "best_validation_accuracy": max(history["val_accuracy"]),
                   "history": history}, f, indent=2)

    # Use the best validation weights for any later predictions.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model, history, checkpoint_path
