
# SHARED TRAINING FUNCTIONS
# Used by Vivian, Joshua, and Devan.
# Do not change independently between experiments.

import torch


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one complete pass through the training dataset.
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, 1)

        correct_predictions += (predictions == labels).sum().item()
        total_images += labels.size(0)

    epoch_loss = running_loss / total_images
    epoch_accuracy = correct_predictions / total_images

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set without updating weights.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)

            correct_predictions += (predictions == labels).sum().item()
            total_images += labels.size(0)

    epoch_loss = running_loss / total_images
    epoch_accuracy = correct_predictions / total_images

    return epoch_loss, epoch_accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_path
):
    """
    Train a model for multiple epochs and save the best weights
    based on validation accuracy.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"  Best model saved "
                f"(Val Acc: {best_val_accuracy:.4f})"
            )

    return history
