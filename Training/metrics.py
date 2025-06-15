import torch

def overall_accuracy(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Calculates the overall accuracy from padded, one-hot encoded tensors. The predictions must be passed through softmax first.

    Args:
        predictions (torch.Tensor): Probabilities for each class (B, N, C).
        labels (torch.Tensor): Padded, one-hot encoded labels (B, N, C).
        mask (torch.Tensor): Actual lengths of each sample (B).

    Returns:
        float: Overall accuracy.
    """

    B, _, _ = labels.shape

    correct = 0
    for batch_id in range(B):
        length = mask[batch_id]
        predicted_classes = predictions[batch_id, :length].argmax(-1)
        label_classes = labels[batch_id, :length].argmax(-1)
        correct += (label_classes == predicted_classes).sum().item()

    return correct / mask.sum().item()


def update_accuracy(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """
    Calculates the number of correctly predicted points, and the total number of points in a batch. Used for calculating accuracy on the entire validation set. The predictions must be passed through softmax first.

    Args:
        predictions (torch.Tensor): Probabilities for each class (B, N, C).
        labels (torch.Tensor): Padded, one-hot encoded labels (B, N, C).
        mask (torch.Tensor): Actual lengths of each sample (B).

    Returns:
        tuple[float, float]: Number of correctly predicted points and total number of points in a batch.
    """

    B, _, _ = labels.shape

    correct = 0
    for batch_id in range(B):
        length = mask[batch_id]
        predicted_classes = predictions[batch_id, :length].argmax(-1)
        label_classes = labels[batch_id, :length].argmax(-1)
        correct += (label_classes == predicted_classes).sum().item()

    return correct, mask.sum().item()

def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the confusion matrix from padded, one-hot encoded tensors. The predictions must be passed through softmax first.

    Args:
        predictions (torch.Tensor): Probabilities for each class (B, N, C).
        labels (torch.Tensor): Padded, one-hot encoded labels (B, N, C).
        mask (torch.Tensor): Actual lengths of each sample (B).

    Returns:
        torch.Tensor: Confusion matrix (C, C).
    """

    B, _, C = labels.shape

    matrix = torch.zeros((C, C), dtype=torch.int64)
    for batch_id in range(B):
        length = mask[batch_id]
        predicted_classes = predictions[batch_id, :length].argmax(-1)
        label_classes = labels[batch_id, :length].argmax(-1)

        for i in range(C):
            predictions_for_ith_class = predicted_classes[label_classes == i]
            for j in range(C):
                matrix[i, j] += (predictions_for_ith_class == j).sum().item()

    return matrix


def intersection_over_union(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[float, torch.Tensor]:
    """
    Calculates the per class, as well as mean, IoU from padded, one-hot encoded tensors. The predictions must be passed through softmax first.

    Args:
        predictions (torch.Tensor): Probabilities for each class (B, N, C).
        labels (torch.Tensor): Padded, one-hot encoded labels (B, N, C).
        mask (torch.Tensor): Actual lengths of each sample (B).

    Returns:
        tuple[float, torch.Tensor]: Mean IoU and per class IoUs.
    """

    B, _, C = labels.shape
    eps = 1e-6

    ious = torch.zeros((C,), dtype=torch.float32)
    for class_id in range(C):
        intersection = 0
        union = 0

        for batch_id in range(B):
            length = mask[batch_id]
            labels_mask = labels[batch_id, :length, class_id] == 1
            predictions_mask = predictions[batch_id, :length].argmax(-1) == class_id

            intersection += torch.logical_and(labels_mask, predictions_mask).sum().item()
            union += torch.logical_or(labels_mask, predictions_mask).sum().item()

        ious[class_id] = (intersection + eps) / (union + eps)

    return ious.mean().item(), ious


def update_intersection_over_union(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the per class intersections and unions in a batch. Used for calculating IoU on the entire validation set. The predictions must be passed through softmax first.

    Args:
        predictions (torch.Tensor): Probabilities for each class (B, N, C).
        labels (torch.Tensor): Padded, one-hot encoded labels (B, N, C).
        mask (torch.Tensor): Actual lengths of each sample (B).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Per class intersections and unions in a batch
    """

    B, _, C = labels.shape

    intersections = torch.zeros((C,), dtype=torch.float32)
    unions = torch.zeros((C,), dtype=torch.float32)

    for class_id in range(C):
        for batch_id in range(B):
            length = mask[batch_id]
            labels_mask = labels[batch_id, :length, class_id] == 1
            predictions_mask = predictions[batch_id, :length].argmax(-1) == class_id

            intersections[class_id] += torch.logical_and(labels_mask, predictions_mask).sum().item()
            unions[class_id] += torch.logical_or(labels_mask, predictions_mask).sum().item()

    return intersections, unions
