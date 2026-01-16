from omegaconf import DictConfig, OmegaConf
from rlkit.templates import PPOBasic, PPOTrainConfig, PPOState
todict = lambda x: OmegaConf.to_container(x, resolve=True)

class PPOTrainerBuilder:
    def build(self, config: DictConfig, create_env, state):
        raise self._make_ppo_trainer(config, create_env, state)
    
    def _make_ppo_trainer(self, config, create_env, state):
        # Train config
        train_config = todict(config.trainer.params)
        if 'device' not in train_config: train_config['device'] = config.device
        train_config = PPOTrainConfig(**train_config)
        state = PPOState(**state)

        # Assemble Trainer
        trainer = PPOBasic(
            create_env=create_env,
            ppo_config=train_config,
            ppo_state=state,
            verbose=config.get("verbose", True),
        )
        return trainer