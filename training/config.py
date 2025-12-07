import yaml
from pathlib import Path


class Config:
    """
    Configuration class for TD3 MEMG EMS
    """
    
    # ============ AGENT CONFIG ============
    AGENT_CONFIG = {
        'state_dim': 10,
        'action_dim': 3,
        'hidden_dim': 256,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,  # Soft update coefficient
        'replay_buffer_size': int(1e6),
        'batch_size': 256,
        'device': 'cpu'  # Change to 'cuda' if available
    }
    
    # ============ ENVIRONMENT CONFIG ============
    ENV_CONFIG = {
        'forecast_horizon': 24,  # Timesteps per episode (15 min each = 6 hours)
        'timestep_min': 15,
        'max_episodes': 50,
        'eval_interval': 10,
        'checkpoint_interval': 10
    }
    
    # ============ TRAINING CONFIG ============
    TRAINING_CONFIG = {
        'num_episodes': 50,
        'initial_exploration_steps': 5000,  # Steps of random actions to fill buffer
        'eval_interval': 10,
        'checkpoint_interval': 10,
        'eval_episodes': 5,
        'save_dir': './checkpoints'
    }
    
    # ============ MEMG CONSTRAINTS ============
    BATTERY_CONFIG = {
        'soc_min': 0.1,
        'soc_max': 0.9,
        'P_ch_max': 50.0,  # kW
        'P_dch_max': 50.0,  # kW
        'E_capacity': 100.0,  # kWh
        'eta_ch': 0.95,
        'eta_dch': 0.95,
        'ramp_rate': 10.0  # kW/15min
    }
    
    CHP_CONFIG = {
        'P_elec_min': 10.0,  # kW
        'P_elec_max': 80.0,  # kW
        'Q_heat_per_elec': 1.2,  # kWth per kWe
        'ramp_rate': 5.0,  # kW/15min
        'min_on_time': 2,
        'min_off_time': 2,
        'startup_cost': 50.0
    }
    
    GRID_CONFIG = {
        'P_import_max': 100.0,  # kW
        'P_export_max': 80.0,  # kW
        'voltage_min': 0.95,
        'voltage_max': 1.05,
        'frequency_min': 49.5,  # Hz
        'frequency_max': 50.5
    }
    
    RENEWABLE_CONFIG = {
        'P_pv_max': 60.0,  # kW
        'P_wt_max': 40.0  # kW
    }
    
    # ============ PRICE & COST CONFIG ============
    PRICE_CONFIG = {
        'price_import': 0.15,  # €/kWh
        'price_export': 0.08,  # €/kWh
        'price_gas': 0.05,  # €/kWh
    }
    
    # ============ REWARD CONFIG ============
    REWARD_CONFIG = {
        'w_cost': 1.0,
        'w_risk': 0.3,
        'w_constraint': 2.0,
        'w_battery': 0.1,
        'w_thermal': 0.5,
        'cvar_alpha': 0.95,
        'constraint_violation_penalty': 10.0
    }
    
    @staticmethod
    def to_dict():
        """
        Return all configurations as dictionary
        """
        return {
            'agent': Config.AGENT_CONFIG,
            'environment': Config.ENV_CONFIG,
            'training': Config.TRAINING_CONFIG,
            'battery': Config.BATTERY_CONFIG,
            'chp': Config.CHP_CONFIG,
            'grid': Config.GRID_CONFIG,
            'renewable': Config.RENEWABLE_CONFIG,
            'price': Config.PRICE_CONFIG,
            'reward': Config.REWARD_CONFIG
        }
    
    @staticmethod
    def save_config(filepath='config.yaml'):
        """
        Save configuration to YAML file
        """
        config_dict = Config.to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Configuration saved to {filepath}")
    
    @staticmethod
    def load_config(filepath='config.yaml'):
        """
        Load configuration from YAML file
        """
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
