# ----------------------
# Corrected Imports
# ----------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # REQUIRED IMPORT
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse

# ----------------------
# Enhanced Transforms
# ----------------------
class RandomCutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, img):
        if np.random.rand() > 0.3:  # 30% probability
            return img
            
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)
        
        # Create mixed image
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img[index, :, bbx1:bbx2, bby1:bby2]
        return img

    @staticmethod
    def rand_bbox(size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

transform_labeled = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.ToTensor(),
])

transform_unlabeled = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ColorJitter(0.3, 0.3, 0.3, 0.2),
    T.GaussianBlur(3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    RandomCutMix(alpha=1.0),
    T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

transform_mask = T.Compose([
    T.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    T.PILToTensor(),
    lambda x: x.squeeze(0).long()
])

# ----------------------
# Dataset Classes
# ----------------------
class CervicalDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_labeled = mask_dir is not None
        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                              if f.endswith(('.png', '.jpg'))])
        if self.is_labeled:
            self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                                 if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.is_labeled:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            return transform_labeled(image), transform_mask(mask)
        return transform_unlabeled(image)

# ----------------------
# Enhanced U-Net Model
# ----------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        att = self.conv(x)
        return x * self.sigmoid(att)

class CervicalUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.attention = AttentionBlock(256)
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, num_classes, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward_features(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        bn = self.bottleneck(p2)
        att = self.attention(bn)
        return self.feature_pool(att).squeeze(-1).squeeze(-1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        bn = self.bottleneck(p2)
        bn = self.attention(bn)
        
        # Decoder
        d1 = self.up1(bn)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        return self.seg_head(d2)

# ----------------------
# Semi-Supervised Framework
# ----------------------
class SemiSupervisedModel(nn.Module):
    def __init__(self, num_classes=3, alpha=0.999):
        super().__init__()
        self.student = CervicalUNet(num_classes=num_classes)
        self.teacher = CervicalUNet(num_classes=num_classes)
        self.alpha = alpha
        self._init_teacher()
        
        # Loss parameters
        self.lambda_unsup = 0.1
        self.lambda_sat = 0.1
        self.conf_thresh = 0.65

    def _init_teacher(self):
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.copy_(s_param.data)

    @torch.no_grad()
    def update_teacher(self, global_step):
        alpha = min(1 - 1/(global_step/100 + 1), self.alpha)
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1-alpha)

    def forward(self, x):
        return self.student(x)

# ----------------------
# Loss Functions
# ----------------------
def compute_sat_loss(student_feats, teacher_feats, temperature=0.1):
    student_feats = F.normalize(student_feats, p=2, dim=1)
    teacher_feats = F.normalize(teacher_feats, p=2, dim=1)
    
    sim_matrix = torch.mm(student_feats, teacher_feats.t()) / temperature
    pos_sim = torch.diag(sim_matrix)
    neg_sim = (sim_matrix.sum(dim=1) - pos_sim) / (sim_matrix.size(1) - 1)
    
    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)).mean()
    return loss

class AdaptiveLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, preds, targets):
        # Cross-Entropy
        ce_loss = F.cross_entropy(preds, targets, weight=self.class_weights)
        
        # Dice Loss
        smooth = 1e-6
        preds_soft = F.softmax(preds, dim=1)
        targets_oh = F.one_hot(targets, num_classes=preds.shape[1]).permute(0,3,1,2).float()
        
        intersection = (preds_soft * targets_oh).sum(dim=(2,3))
        union = preds_soft.sum(dim=(2,3)) + targets_oh.sum(dim=(2,3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
        
        return ce_loss + dice_loss.mean()

# ----------------------
# Training Utilities
# ----------------------
def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.argmax(1)
    return (2.0 * (pred * target).sum() + smooth) / (pred.sum() + target.sum() + smooth)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    labeled_ds = CervicalDataset(args.labeled_img, args.labeled_mask)
    unlabeled_ds = CervicalDataset(args.unlabeled_img)
    
    # Class weights
    class_counts = torch.zeros(3)
    for _, mask in labeled_ds:
        class_counts += torch.bincount(mask.flatten(), minlength=3)
    class_weights = 1.0 / (class_counts / class_counts.sum()).to(device)
    
    # Data loaders
    labeled_loader = DataLoader(labeled_ds, batch_size=args.batch_size, shuffle=True, 
                               num_workers=2, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size*2, shuffle=True,
                                 num_workers=2, pin_memory=True)
    
    # Model
    model = SemiSupervisedModel(num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Losses
    sup_criterion = AdaptiveLoss(class_weights)
    unsup_criterion = AdaptiveLoss()
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        sup_loss_total = 0.0
        unsup_loss_total = 0.0
        dice_total = 0.0
        
        # Dynamic parameters
        current_thresh = 0.65 + min(epoch/args.epochs, 1)*0.25  # 0.65 → 0.9
        current_lambda_unsup = min(epoch/10 * 0.5, 0.5)
        current_lambda_sat = 0.2 * (1 - epoch/args.epochs)
        
        pbar = tqdm(zip(labeled_loader, unlabeled_loader), 
                   total=min(len(labeled_loader), len(unlabeled_loader)), 
                   desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for (labeled_x, labeled_y), unlabeled_x in pbar:
            labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)
            unlabeled_x = unlabeled_x.to(device)
            
            # Supervised Forward
            student_preds = model(labeled_x)
            sup_loss = sup_criterion(student_preds, labeled_y)
            
            # Unsupervised Forward
            with torch.no_grad():
                teacher_preds = model.teacher(unlabeled_x)
                pseudo_probs = F.softmax(teacher_preds, dim=1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
                mask = (max_probs > current_thresh).float()
                
            unsup_loss = 0.0
            if mask.sum() > 0:
                student_u_preds = model(unlabeled_x)
                unsup_loss = unsup_criterion(student_u_preds, pseudo_labels) * mask.mean()
            
            # Feature Alignment Loss
            with torch.no_grad():
                t_features = model.teacher.forward_features(unlabeled_x)
            s_features = model.student.forward_features(unlabeled_x)
            sat_loss = compute_sat_loss(s_features, t_features)
            
            # Total Loss
            total_loss = sup_loss + current_lambda_unsup*unsup_loss + current_lambda_sat*sat_loss
            
            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_teacher(global_step)
            global_step += 1
            
            # Metrics
            sup_loss_total += sup_loss.item()
            unsup_loss_total += unsup_loss.item() if unsup_loss != 0 else 0
            total_loss += total_loss.item()
            dice_total += dice_score(student_preds, labeled_y).item()
            
            pbar.set_postfix({
                "Sup": f"{sup_loss.item():.3f}",
                "Unsup": f"{unsup_loss.item():.3f}" if unsup_loss != 0 else "0.000",
                "SAT": f"{sat_loss.item():.3f}",
                "Dice": f"{dice_total/(pbar.n+1):.3f}",
                "Total": f"{total_loss.item():.3f}"
            })
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        avg_loss = total_loss / len(labeled_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")
        
        print(f"Epoch {epoch+1} | Sup: {sup_loss_total/len(labeled_loader):.3f} "
              f"Unsup: {unsup_loss_total/len(unlabeled_loader):.3f} "
              f"Dice: {dice_total/len(labeled_loader):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_img", type=str, default="dataset/labeled_data/images")
    parser.add_argument("--labeled_mask", type=str, default="dataset/labeled_data/labels")
    parser.add_argument("--unlabeled_img", type=str, default="dataset/unlabeled_data/images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)