import torch

def compute_group_avg(losses, group_idx, group):
    # compute observed counts and mean loss for each group
    group_map = (group_idx == torch.arange(group).unsqueeze(1).long().to(device)).float().to(device)
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count==0).float() # avoid nans
    
    group_loss = (group_map @ losses.view(-1))/group_denom
    return group_loss, group_count

def compute_robust_loss(group_loss, group_count, group_weights, step_size):
    adjusted_loss = group_loss
    adjusted_loss = adjusted_loss/(adjusted_loss.sum())
    group_weights = group_weights * torch.exp(step_size*adjusted_loss.data)
    group_weights = group_weights/group_weights.sum()

    robust_loss = group_loss @ group_weights
    return robust_loss, group_weights