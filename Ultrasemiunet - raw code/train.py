import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from losses import sat_loss, aldc_loss, compute_hardness_map

def train_ultraseminet(
    student_teacher_model,
    dataloader_labeled,
    dataloader_unlabeled,
    optimizer,
    num_epochs=10,
    temperature=0.07,
    lambda_sat=0.5,
    lambda_aldc=0.5,
    save_path="model.pth"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_teacher_model = student_teacher_model.to(device)
    criterion_ce = nn.CrossEntropyLoss()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        student_teacher_model.train()
        running_loss = 0.0
        steps = 0
        steps_per_epoch = min(len(dataloader_labeled), len(dataloader_unlabeled))

        # Create the tqdm progress bar
        pbar = tqdm(
            zip(dataloader_labeled, dataloader_unlabeled),
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )
        for (x_l, y_l), x_u in pbar:
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # Supervised loss
            logits_l = student_teacher_model(x_l)
            sup_loss = criterion_ce(logits_l, y_l)

            # Pseudo-label generation (teacher side)
            with torch.no_grad():
                logits_u_teacher = student_teacher_model.teacher_net(x_u)
                pseudo_labels = torch.argmax(logits_u_teacher, dim=1)
            
            # Student forward on unlabeled data
            logits_u_student = student_teacher_model(x_u)
            unsup_loss_ce = criterion_ce(logits_u_student, pseudo_labels)

            # Features for SAT loss
            features_student_u = F.adaptive_avg_pool2d(logits_u_student, (1,1)).squeeze(-1).squeeze(-1)
            features_teacher_u = F.adaptive_avg_pool2d(logits_u_teacher, (1,1)).squeeze(-1).squeeze(-1)

            # Negative examples by shuffling
            batch_size = features_student_u.size(0)
            indices = torch.randperm(batch_size, device=device)
            neg_features = features_student_u[indices]

            # SAT loss
            sat_loss_val = sat_loss(features_student_u, features_teacher_u, neg_features, temperature)

            # ALDC loss on labeled data (using hardness map from teacher)
            with torch.no_grad():
                logits_l_hard = student_teacher_model.teacher_net(x_l)
            hardness_map = compute_hardness_map(logits_l_hard)
            mask = (hardness_map > 0.5).float()

            aldc_val = aldc_loss(logits_l, y_l.unsqueeze(1), mask, temperature)

            # Total loss
            total_loss = sup_loss + unsup_loss_ce + lambda_sat*sat_loss_val + lambda_aldc*aldc_val

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # EMA update for teacher
            student_teacher_model._update_teacher()

            running_loss += total_loss.item()
            steps += 1

            # Update the progress bar
            pbar.set_postfix({
                "SupLoss": f"{sup_loss.item():.4f}",
                "UnsupLoss": f"{unsup_loss_ce.item():.4f}",
                "SAT": f"{sat_loss_val.item():.4f}",
                "ALDC": f"{aldc_val.item():.4f}",
                "Total": f"{total_loss.item():.4f}"
            })

        epoch_loss = running_loss / steps if steps > 0 else 0.0
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(student_teacher_model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with loss={epoch_loss:.4f}")

    print("Training complete!")
