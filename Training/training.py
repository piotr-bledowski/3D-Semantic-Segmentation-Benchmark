import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable
from Training.metrics import overall_accuracy, update_accuracy, confusion_matrix, intersection_over_union, update_intersection_over_union
from tqdm import tqdm

def plot_confusion_matrix(matrix: torch.Tensor) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(matrix, annot=True, fmt='d', 
                cmap='Blues', ax=ax,
                xticklabels=range(1, 15), 
                yticklabels=range(1, 15))
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    # ax.set_title(f'{Con} - Epoch {epoch}')
    # plt.tight_layout()
    
    return fig


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: optim.Optimizer,
        device: str,
        logger: SummaryWriter,
        log_interval: int,
        previous_global_steps: int
    ) -> tuple[float, int]:

    model.train()
    total_loss = 0
    global_step = previous_global_steps

    progress_bar = tqdm(
        train_loader,
        desc='Training',
        unit='Batch',
        position=1,
        leave=False
    )

    for batch_index, (points, labels, lengths) in enumerate(progress_bar):
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels, lengths)
        loss.backward()
        optimizer.step()

        if batch_index % log_interval == 0:
            probabilites = F.softmax(outputs, dim=-1)
            accuracy = overall_accuracy(probabilites, labels, lengths)
            mIoU = intersection_over_union(probabilites, labels, lengths)[0]

            logger.add_scalar('Train/Loss', loss.item(), global_step)
            logger.add_scalar('Train/Accuracy', 100.0 * accuracy, global_step)
            logger.add_scalar('Train/Mean_IoU', 100.0 * mIoU, global_step)

        total_loss += loss.item()
        global_step += 1

    progress_bar.close()

    total_loss /= len(train_loader)

    return total_loss, global_step

def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        device: str
    ) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:

    model.eval()
    total_loss = 0

    total_correct = 0
    total_points = 0
    matrix = torch.zeros((14, 14), dtype=torch.int64)
    intersections = torch.zeros((14,), dtype=torch.float32)
    unions = torch.zeros((14,), dtype=torch.float32)

    eps = 1e-6

    progress_bar = tqdm(
        test_loader,
        desc='Validating',
        unit='Batch',
        position=1,
        leave=False
    )

    with torch.no_grad():
        for points, labels, lengths in progress_bar:
            points = points.to(device)
            labels = labels.to(device)

            outputs = model(points)
            total_loss += criterion(outputs, labels, lengths.long())

            probabilites = F.softmax(outputs, dim=-1)

            batch_correct, batch_points = update_accuracy(probabilites, labels, lengths)
            total_correct += batch_correct
            total_points += batch_points

            matrix += confusion_matrix(probabilites, labels, lengths)

            batch_intersections, batch_unions = update_intersection_over_union(probabilites, labels, lengths)
            intersections += batch_intersections
            unions += batch_unions

    progress_bar.close()

    total_loss /= len(test_loader)
    accuracy = total_correct / total_points
    ious = (intersections + eps) / (unions + eps)
    mean_iou = ious.mean().item()

    return total_loss, accuracy, mean_iou, ious, matrix

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: optim.Optimizer,
        device: str,
        logger: SummaryWriter,
        num_epochs: int,
        log_interval: int,
    ) -> nn.Module:

    global_step = 0

    progress_bar = tqdm(
        range(num_epochs),
        desc='Overall Progress',
        unit='Epoch',
        position=0,
        leave=True
    )

    model = model.to(device)

    for epoch in progress_bar:
        train_loss, global_step = train_epoch(model, train_loader, criterion, optimizer, device, logger, log_interval, global_step)

        val_loss, accuracy, mean_iou, ious, matrix = evaluate(model, test_loader, criterion, device)

        tqdm.write(f'Epoch {epoch + 1} completed:')
        tqdm.write(f'- Training loss: {train_loss}')
        tqdm.write(f'- Validation loss: {val_loss}')
        tqdm.write(f'- Validation accuracy: {accuracy}')
        tqdm.write(f'- Validation mean IoU: {mean_iou}')
        tqdm.write('-' * 15)

        logger.add_scalar('Train/Total_Loss', train_loss, epoch)
        logger.add_scalar('Val/Loss', val_loss, epoch)
        logger.add_scalar('Val/Accuracy', 100.0 * accuracy, epoch)
        logger.add_scalar('Val/Mean_Iou', 100.0 * mean_iou, epoch)
        logger.add_tensor('Val/Ious', 100.0 * ious, epoch)
        # logger.add_figure('Val/Confusion_Matrix', plot_confusion_matrix(matrix), epoch)

    progress_bar.close()

    return model
