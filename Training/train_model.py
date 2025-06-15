"""
This code is adjusted and tested to data structure of PointNet.
"""

import torch.nn as nn
import torch
import torch.optim as optim
import tqdm
import os
import pickle
import numpy as np
import torch.nn.functional as F


def masked_onehot_cross_entropy(
    logits: torch.Tensor,
    targets_onehot: torch.Tensor,
    pad_starts: torch.Tensor,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Args:
        logits: Tensor of shape (B, L, C), nieskala­rowane predykcje (logity).
        targets_onehot: Tensor of shape (B, L, C), one-hot etykiety.
        pad_starts: Tensor of shape (B,), dtype=torch.int32 lub torch.int64,
                    indeks (w każdej sekwencji) od którego zaczyna się padding.
        eps: mała stała, by uniknąć log(0).

    Returns:
        loss: pojedynczy skalar – średnia cross-entropy po wszystkich niemaskowanych
              tokenach w batchu.
    """
    B, L, C = logits.shape
    # 1) log-softmax po klasach
    log_probs = F.log_softmax(logits, dim=-1) # (B, L, C), model jest po softmaksie

    # 2) cross-entropy per-token: - sum_y y * log p
    #    dzięki temu, że targets_onehot jest one-hot, to redukuje się do -log p_true
    token_loss = -torch.sum(targets_onehot * log_probs, dim=-1)  # (B, L)

    # 3) maska: 1 dla prawdziwych tokenów (< pad_start), 0 dla paddingu
    #    utworzymy tensor pozycji 0..L-1, a potem porównamy z pad_starts
    device = logits.device
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
    pad_starts = pad_starts.to(device).long()
    mask = (positions < pad_starts.unsqueeze(1)).float()  # (B, L)

    # 4) aplikacja maski i redukcja
    masked_loss = token_loss * mask  # (B, L)
    total_non_pad = mask.sum()

    # Chronimy przed dzieleniem przez zero (wszystkie tokeny to padding?)
    if total_non_pad.item() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = masked_loss.sum() / total_non_pad
    return loss


import torch


def accuracy_from_one_hot(label: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    Oblicza procent trafionych klas pomiędzy tensorami label i prediction.

    Args:
        label (torch.Tensor): tensor one-hot o wymiarach (batch_size, num_examples, num_classes)
        prediction (torch.Tensor): tensor z prawdopodobieństwami po softmaksie, o tych samych wymiarach

    Returns:
        float: procent trafnych predykcji (0–100)
    """
    # Pobieramy indeksy klas: argmax wzdłuż osi klas
    label_classes = label.argmax(dim=-1)
    predicted_classes = prediction.argmax(dim=-1)

    # Porównanie predykcji z etykietą
    correct = (label_classes == predicted_classes).float()

    # Obliczamy średnią trafność i przeliczamy na %
    accuracy = correct.mean().item()
    return accuracy


import torch
from typing import List, Tuple, Optional

def preprocess_batch_to_train_format(
    x: List[torch.Tensor],
    y: List[List[str]],
    mapping: List[str],
    cut: Optional[int] = None,
    sampling: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """
    Pads variable-length point clouds, one-hot encodes labels, and optionally samples points.

    Args:
        x (List[torch.Tensor]): Each element has shape (N_i, D), N_i varies per sample.
        y (List[List[str]]): List of string labels per point, length matches N_i.
        mapping (List[str]): Class names ordered so their index matches the one-hot label position.
        cut (int, optional): If provided, truncate/pad sequences to this length (N_max = cut).
        sampling (float, optional): Fraction in (0,1] of points to randomly sample per example before padding.
                                   If provided, each sample will be randomly downsampled to
                                   max(int(N_i * sampling), 1) points.

    Returns:
        batch_input (torch.Tensor): Padded point clouds, shape (B, D, N_out).
        label (torch.Tensor): One-hot labels, shape (B, N_out, num_classes).
        input_output_mapping (torch.Tensor): Original (or sampled) lengths per example, shape (B,).
        cont (bool): Whether to continue training (False if batch size == 1).
    """

    # Validate sampling
    if sampling is not None:
        if not (0 < sampling <= 1.0):
            raise ValueError(f"sampling must be in (0,1], got {sampling}")
        x_sampled, y_sampled = [], []
        for xi, yi in zip(x, y):
            N_i = xi.shape[0]
            k = max(int(N_i * sampling), 1)
            # Randomly choose k indices without replacement
            perm = torch.randperm(N_i, device=xi.device)[:k]
            xi_sub = xi[perm]
            # perm is torch tensor, convert to python indices for list
            idxs = perm.cpu().tolist()
            yi_sub = [yi[idx] for idx in idxs]
            x_sampled.append(xi_sub)
            y_sampled.append(yi_sub)
        x, y = x_sampled, y_sampled

    # Compute lengths after sampling (or original if no sampling)
    input_output_mapping = torch.tensor([i.shape[0] for i in x], dtype=torch.int32)

    # Determine maximum length
    max_length = int(input_output_mapping.max().item())
    if cut is not None:
        max_length = min(max_length, cut)

    # Prepare input tensor: shape (B, max_length, D)
    B = len(x)
    D = x[0].shape[-1]
    batch_input = torch.zeros((B, max_length, D), device=x[0].device, dtype=x[0].dtype)
    for i, sample in enumerate(x):
        n_pts = min(sample.shape[0], max_length)
        batch_input[i, :n_pts, :] = sample[:n_pts]

    # Prepare label tensor: shape (B, max_length, num_classes)
    num_classes = len(mapping)
    label = torch.zeros((B, max_length, num_classes), dtype=torch.float32, device=batch_input.device)
    for i, sample_labels in enumerate(y):
        for j, lbl in enumerate(sample_labels):
            if j >= max_length: # jakby się tu zepsuło to dać >=
                break
            idx = mapping.index(lbl)
            if idx == -1:
                raise ValueError(f"W zbiorze danych są etykiety których nie ma w 'mapping' tj. {lbl}")
            label[i, j, idx] = 1.0

    # Transpose input to (B, D, max_length)
    batch_input = batch_input.transpose(1, 2)

    # Update mapping if cut applied
    if cut is not None:
        input_output_mapping = torch.clamp(input_output_mapping, max=cut)

    # Determine continuation flag
    cont = (B > 1)

    return batch_input, label, input_output_mapping, cont



def train_epoch(model, train_loader, criterion, optimizer, device, mapping, cut, sampling):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        points, labels, input_output_mapping, cont = preprocess_batch_to_train_format(batch["x"], batch["y"], mapping, cut=cut, sampling = sampling)
        if not cont:
            continue
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _, _ = model(points)

        #print(outputs.mean(dim=1))
        #print(labels.mean(dim=1))

        loss = criterion(outputs, labels, input_output_mapping)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device, mapping, cut, sampling):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    if len(test_loader) == 0:
        return None, None
    with torch.no_grad():
        for batch in test_loader:
            points, labels, input_output_mapping, cont = preprocess_batch_to_train_format(batch["x"], batch["y"], mapping, cut=cut, sampling = sampling)
            if not cont:
                continue
            points = points.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(points)
            loss = criterion(outputs, labels, input_output_mapping)
            total_loss += loss.item()

            correct += accuracy_from_one_hot(labels, outputs) * input_output_mapping.sum().item()

            total += input_output_mapping.sum().item()
    accuracy = correct / total
    return total_loss / len(test_loader), accuracy





def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader, mapping,  device = None, lr = 0.001, epochs = 20, print_records = False,
                records_dir: str = None, records_filename: str = None, cut: int=None, sampling: float = None) -> torch.nn.Module:
    """

    Args:
        model:
        train_loader:
        test_loader:
        device: device to perform computations
        lr: learning rate
        epochs:
        print_records: Does user wants to print loss in stout
        records_dir: If not None it will save results of evaluation and training to give directory.

    Returns: trained model

    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not records_dir is None:
        if not os.path.exists(records_dir):
            os.makedirs(records_dir)

        train_losses = []
        val_losses = []
        val_metrics = []


    criterion = masked_onehot_cross_entropy

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)





    for epoch in tqdm.tqdm(range(epochs)):
        torch.cuda.empty_cache()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, mapping, cut=cut, sampling = sampling)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, mapping, cut=cut, sampling = sampling)

        if not records_dir is None:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_metrics.append(val_acc)
        if print_records:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}, Val Accuracy: {val_acc}")

    if not records_dir is None:
        with open(os.path.join(records_dir, f"{records_filename}.pkl"), "wb") as f:
            to_dump = {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_metrics}
            pickle.dump(to_dump, f)
    return model

