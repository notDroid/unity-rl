from .ppo import *
from rlkit.modules import AutomaticEntropyModule

class PPOWithEntropySchedule(PPOTrainer):
    def __init__(self, create_env, train_config):
        super().__init__(create_env, train_config)

    def load_state(self, entropy_scheduler = None, *args, **kwargs):
        self.entropy_scheduler = entropy_scheduler

        return super().load_state(*args, **kwargs)
    
    def _prepare_train(self):
        # Update entropy
        if self.entropy_scheduler:
            entropy_coef = self.entropy_scheduler(self.gen)
            self.logger.acc({"entropy_coef": entropy_coef})

        return super()._prepare_train()   


'''
Some thoughts on Dynamic Entropy: 
It doesn't work that well. 
In high dim action spaces the space of "correct" answers seem to be a higher percent of the space.
This means that good solutions can still be high entropy.
So setting a target entropy of say -dim(A) for continuous action spaces drives alpha to 0 and leads to a worse solution.
Better off just scheduling the entropy, like phase 1: 1e-2, phase 2: 1e-3.
'''
class PPOWithDynamicEntropy(PPOTrainer):
    def __init__(self, create_env, train_config):
        super().__init__(create_env, train_config)

    def _load_config(self, train_config):
        self.alpha_coef = train_config["alpha_coef"]
        return super()._load_config(train_config)

    def load_state(self, entropy_module: AutomaticEntropyModule, *args, **kwargs):
        self.entropy_module = entropy_module
        return super().load_state(*args, **kwargs)
    
    def _train_step(self, batch, epoch, j):
        # -------------- a. Compute Loss --------------
        with torch.no_grad():
            alpha = self.entropy_module.alpha
        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=(self.amp_dtype==torch.float16)):
            loss_data = self.loss_module(batch)
            minus_entropy = loss_data["loss_entropy"].mean()
            loss = loss_data["loss_objective"].mean() + loss_data["loss_critic"].mean() + alpha * minus_entropy
            
        # Automatic Entropy
        loss += self.alpha_coef * self.entropy_module(minus_entropy.detach())

        # -------------- b. KL Safety Check --------------
        kl_approx = loss_data["kl_approx"].mean().cpu().item()
        if self.kl_soft_clip is not None and kl_approx > self.kl_soft_clip:
            self.early_stop += 1
            if self.early_stop >= self.early_stop_threshold: 
                print(f"Early stopped at ({epoch}, {j})")
                return
        if self.kl_hard_clip is not None and kl_approx > self.kl_hard_clip:
            print(f"Skipping iteration ({epoch}, {j}) with kl > kl_hard_clip: {kl_approx} > {self.kl_hard_clip}")
            return
        
        # -------------- c. Optimization Step --------------
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=self.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # -------------- d. Metric Update --------------
        weight = float(batch.batch_size[0])
        if self.logger: 
            self.logger.acc(ppo_loss_td_to_dict(loss_data, weight), mode='ema')
            self.logger.acc({"alpha": alpha.cpu().item()}, mode='ema')
        return
    
    def _ckpt_step(self, metrics):
        state_obj={
            "generation": self.gen + 1,
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "entropy_state_dict": self.entropy_module.state_dict(),
        }
        if self.best_metric_key:
            state_obj[self.best_metric_key] =  metrics[self.best_metric_key]
        
        self.checkpointer.save_progress(state_obj)