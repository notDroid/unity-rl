from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_class, instantiate
from rlkit.templates import ppo_log_keys
from rlkit.utils import SimpleMetricModule
from rlkit.templates import PPOStateManager
import torch
todict = lambda x: OmegaConf.to_container(x, resolve=True)

def maybe_add_key(log_keys, condition, key):
    if condition and key not in log_keys:
        log_keys.append(key)

class PPOStateBuilder:
    def build(self, config: DictConfig, agent, loss_module) -> dict:

        # Initialize State
        state = {'agent': agent, 'loss_module': loss_module}
        state = self._build_core_state(config, state)
        state.update(self._build_optional_components(config, state))
        
        
        # Restore or Reset State
        self.ppo_sm = PPOStateManager(config.trainer.params)
        if config.get("continue_", True):
            state = self._restore_state(state)
        else:
            state = self.ppo_sm.reset_state(state)
            
        return state
    
    def _build_core_state(self, config, state):
        """Build required state components"""
        optimizer = instantiate(
            config.state.components.optimizer,
            params=state['loss_module'].parameters(),
        )
        return {
            'agent': state['agent'],
            'loss_module': state['loss_module'],
            'optimizer': optimizer,
            'start_generation': 0,
        }
    
    def _build_optional_components(self, config, state):
        """Build optional components like logger, checkpointer, etc."""
        components = {}
        
        if config.state.components.get('logger'):
            # Instantiate logger with inferred keys if not provided
            if not config.state.components.logger.get("keys"):
                log_keys = self._get_log_keys(config)
                components['logger'] = instantiate(config.state.components.logger, keys=log_keys)
            else:
                components['logger'] = instantiate(config.state.components.logger)
        
        if config.state.components.get('checkpointer'):
            components['checkpointer'] = instantiate(config.state.components.checkpointer)

        if config.state.components.get('lr_scheduler'):
            lr_scheduler = instantiate(
                config.state.components.lr_scheduler, 
                optimizer=state['optimizer'],
            )
            components['lr_scheduler'] = lr_scheduler

        if config.state.components.get('scaler'):
            components['scaler'] = torch.amp.GradScaler()
        
        if config.state.components.get('metric_module'):
            components['metric_module'] = instantiate(config.state.components.metric_module)
        else:
            components['metric_module'] = SimpleMetricModule(mode='approx')

        return components
    
    def _restore_state(self, state):
        checkpointer = state.get('checkpointer', None)
        if checkpointer:
            checkpoint = checkpointer.load_progress()
            if checkpoint:
                return self.ppo_sm.restore_checkpoint(state, checkpoint)
            else:
                return self.ppo_sm.reset_state(state)
        else:
            raise Warning("No checkpointer found in state; cannot restore past state.")
        
    def _get_log_keys(self, config):
        if config.state.components.logger.get("keys"):
            return list(config.state.components.logger.keys)
        log_keys = ppo_log_keys.copy()
        maybe_add_key(log_keys, "lr_scheduler" in config.state.components, 'lr')
        if config.loss.get('loss_params') and config.loss.loss_params.get("inverse_dynamics_coeff"):
            maybe_add_key(log_keys, config.loss.loss_params.get("inverse_dynamics_coeff") > 0, 'inverse_dynamics_loss')
        return log_keys


