import torch
import torch.nn.functional as F

def sat_loss(anchor, pos, neg, temperature=0.07):
    # Dot products for anchor-positive and anchor-negative
    sim_pos = (anchor * pos).sum(dim=1) / temperature
    sim_neg = (anchor * neg).sum(dim=1) / temperature
    
    # SAT Loss is based on softmax cross-entropy
    logits = torch.stack([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)

def compute_hardness_map(logits):
    probs = torch.softmax(logits, dim=1)
    ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
    return ent

def aldc_loss(features, labels, mask, temperature=0.07):
    B, C, H, W = features.shape
    features_2d = features.permute(0,2,3,1).reshape(-1, C) 
    labels_2d = labels.reshape(-1)     
    mask_2d = mask.reshape(-1) > 0.5

    idxs = torch.where(mask_2d)[0]
    if len(idxs) < 2:
        return torch.tensor(0.0, device=features.device)  # no "hard" region

    # Pick random anchor in the masked region
    anchor_idx = idxs[torch.randint(0, len(idxs), (1,))]
    anchor_feat = features_2d[anchor_idx]  # shape (C,)
    anchor_label = labels_2d[anchor_idx]
    same_label_idx = idxs[(labels_2d[idxs] == anchor_label)]
    if len(same_label_idx) < 2:
        return torch.tensor(0.0, device=features.device)
    pos_idx = same_label_idx[torch.randint(0, len(same_label_idx), (1,))]
    pos_feat = features_2d[pos_idx]
    diff_label_idx = idxs[(labels_2d[idxs] != anchor_label)]
    if len(diff_label_idx) < 1:
        return torch.tensor(0.0, device=features.device)
    neg_idx = diff_label_idx[torch.randint(0, len(diff_label_idx), (1,))]
    neg_feat = features_2d[neg_idx]

    # Same form as sat_loss
    sim_pos = (anchor_feat * pos_feat).sum() / temperature
    sim_neg = (anchor_feat * neg_feat).sum() / temperature
    logits = torch.stack([sim_pos, sim_neg], dim=0).unsqueeze(0)
    labels_val = torch.zeros(1, dtype=torch.long, device=features.device)

    return F.cross_entropy(logits, labels_val)
