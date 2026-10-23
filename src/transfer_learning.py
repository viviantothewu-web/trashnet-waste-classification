
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from src.training import train_model


STAGE1_EPOCHS = 5
STAGE1_LR = 0.001

STAGE2_EPOCHS = 5
STAGE2_LR = 0.0001

criterion = nn.CrossEntropyLoss()


def create_pretrained_model(model_name, num_classes=6):

    if model_name == "vgg16":
        model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )

        for param in model.parameters():
            param.requires_grad = False

        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)


    elif model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        for param in model.parameters():
            param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)


    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2
        )

        for param in model.parameters():
            param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)


    else:
        raise ValueError(
            "Choose 'vgg16', 'resnet50', or 'mobilenetv2'"
        )

    return model


def unfreeze_model(model):

    for param in model.parameters():
        param.requires_grad = True

    return model


def run_transfer_learning(
    model_name,
    train_loader,
    val_loader,
    device,
    training_seed
):

    model_dir = Path("saved_models")
    model_dir.mkdir(exist_ok=True)

    model = create_pretrained_model(model_name)
    model = model.to(device)

    # Stage 1
    stage1_path = model_dir / (
        f"{model_name}_seed{training_seed}_stage1.pth"
    )

    stage1_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR
    )

    stage1_history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        stage1_optimizer,
        device,
        STAGE1_EPOCHS,
        stage1_path
    )

    model.load_state_dict(
        torch.load(stage1_path, map_location=device)
    )

    # Stage 2
    unfreeze_model(model)

    stage2_path = model_dir / (
        f"{model_name}_seed{training_seed}_stage2.pth"
    )

    stage2_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=STAGE2_LR
    )

    stage2_history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        stage2_optimizer,
        device,
        STAGE2_EPOCHS,
        stage2_path
    )

    return model, stage1_history, stage2_history
