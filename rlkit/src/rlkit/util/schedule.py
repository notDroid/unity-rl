import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class cosine_annealing_with_linear_warmup:
    def __init__(self, total_epochs, warmup_epochs, max_lr = 1e-3, initial_lr = 1e-5):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.initial_lr = initial_lr
    
    def __call__(self, epoch):
        # Linear warmup
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * epoch / self.warmup_epochs
        # Cosine annealing
        else:
            lr = self.max_lr * 0.5 * (1 + np.cos((epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs) * np.pi))

        return lr
    
class CosineWithLinearWarmupSchedule(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, max_lr=1e-3, initial_lr=1e-5, last_epoch=-1):
        self.policy = cosine_annealing_with_linear_warmup(
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            max_lr=max_lr,
            initial_lr=initial_lr,
        )
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        lr = self.policy(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]