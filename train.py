import torch
from data_processing.chunked_datasets import create_chunked_dataloaders
from models.PointNeXt.PointNeXt import PointNeXt
from models.PointNetpp.PointNetpp import PointNetpp
from models.PointNet.PointNet import PointNetSeg
from Training.train_model import train_model
import os
import argparse

DATA_DIR_PATH = "data_chunked"
TRAINING_DIR_PATH = "saved_training"
MODEL_SAVE_PATH = os.path.join(TRAINING_DIR_PATH, "models")
TRAINING_HISTORY_PATH = os.path.join(TRAINING_DIR_PATH, "history")
BATCH_SIZE = 2
NUM_CLASSES = 13  # dla S3DIS
s3dis_classes = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter"
]

if __name__ == '__main__':
    os.makedirs(TRAINING_DIR_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(TRAINING_HISTORY_PATH, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a selected model.")
    parser.add_argument("model", help="Name of the model to train.", choices=['PointNet', 'PointNet++', 'PointNeXt'])
    args = parser.parse_args()

    # # Create dataloaders with optimized loading
    # # UWAGA!! dany plik z mapowaniem: chunked_s3dis_index_mapping.pkl nie wspiera jeżeli kożystamy z podzbioru zbioru
    # # danego na dysku - trzeba dać require_index_file=False
    train_loader, test_loader = create_chunked_dataloaders(
        DATA_DIR_PATH,
        batch_size=BATCH_SIZE,
        require_index_file=False
    )

    # !!!! TYLKO DO TESTÓW - w przykładowych danych nie ma ich wystarczająco dużo aby zrobić zbiór testowy (walidacyjny)
    test_loader = train_loader # usunąć w normalnym treningu

    model_name = args.model
    if model_name == 'PointNet':
        raw_model = PointNetSeg(part_classes=NUM_CLASSES)
    elif model_name == 'PointNet++':
        raw_model = PointNetpp(part_classes=NUM_CLASSES)
    elif model_name == 'PointNeXt':
        raw_model = PointNeXt(part_classes=NUM_CLASSES)

    model_trained = train_model(raw_model, train_loader, test_loader, s3dis_classes, print_records=True,
                        records_dir=TRAINING_HISTORY_PATH, records_filename=model_name, epochs=30, sampling=None, cut=1000)

    with open(os.path.join(MODEL_SAVE_PATH, f"{model_name}.pt"), "wb") as f:
        torch.save(model_trained.state_dict(), f)
