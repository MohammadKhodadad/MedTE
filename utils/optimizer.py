import math
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

def get_optimizer_and_scheduler(model, base_lr=0.001, warmup_steps=1000, total_steps=10000, eta_min=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(eta_min / base_lr, cosine_decay)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    return optimizer, scheduler