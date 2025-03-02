import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from ultra_semi_net import UltraSemiNet
from train import train_ultraseminet
from datasets import MyLabeledDataset, MyUnlabeledDataset
from transformations import transform, transform_mask

labeled_image_dir = '/home/ufaqkhan/UltraSemiNet/Dataset/labeled/original'
labeled_mask_dir = '/home/ufaqkhan/UltraSemiNet/Dataset/labeled/groundtruth'
unlabeled_image_dir = '/home/ufaqkhan/UltraSemiNet/Dataset/unlabeled'
labeled_dataset = MyLabeledDataset(
    image_dir=labeled_image_dir,
    mask_dir=labeled_mask_dir,
    transform=transform,
    transform_mask=transform_mask
)
unlabeled_dataset = MyUnlabeledDataset(
    image_dir=unlabeled_image_dir,
    transform=transform
)
labeled_loader = DataLoader(labeled_dataset, batch_size=4, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=True)
model = UltraSemiNet(in_channels=3, num_classes=2, alpha=0.99)  # or in_channels=1 if grayscale
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_ultraseminet(
    student_teacher_model=model,
    dataloader_labeled=labeled_loader,
    dataloader_unlabeled=unlabeled_loader,
    optimizer=optimizer,
    num_epochs=10,
    temperature=0.07,
    lambda_sat=0.5,
    lambda_aldc=0.5,
    save_path="model.pth"
)
