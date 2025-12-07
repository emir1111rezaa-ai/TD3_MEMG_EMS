# TD3 MEMG EMS - Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training Example

### 1. Basic Training (50 episodes, CPU)

```bash
python main.py --train --episodes 50 --device cpu
```

### 2. Training with GPU

```bash
python main.py --train --episodes 50 --device cuda
```

### 3. Save Configuration

```bash
python main.py --save-config
```

This creates `config.yaml` with all settings.

## Evaluation

### 1. Evaluate After Training

```bash
python main.py --eval --checkpoint ./checkpoints/td3_episode_50.pt
```

### 2. Evaluate a Specific Checkpoint

```bash
python main.py --eval --checkpoint ./checkpoints/td3_episode_30.pt --device cuda
```

## Python API

### Create and Train Agent

```python
from training.trainer import TD3Trainer

# Initialize trainer
trainer = TD3Trainer(
    state_dim=10,
    action_dim=3,
    hidden_dim=256,
    device='cpu'
)

# Train for 100 episodes
trainer.train(num_episodes=100, checkpoint_interval=10)

# Evaluate
results = trainer.evaluate(num_episodes=10)
print(f"Mean Cost: {results['mean_cost']:.2f} EUR")
print(f"Mean Violations: {results['mean_violations']:.1f}")
```

### Load and Test Checkpoint

```python
from training.trainer import TD3Trainer

trainer = TD3Trainer(device='cuda')
trainer.load_checkpoint('./checkpoints/td3_episode_50.pt')

# Test trained agent
results = trainer.evaluate(num_episodes=20)
```

### Direct Agent Usage

```python
from td3_agent.td3_core import TD3Agent
from memg_env.microgrid import MEMGEnvironment

# Initialize
agent = TD3Agent(state_dim=10, action_dim=3, device='cpu')
env = MEMGEnvironment(forecast_horizon=24)

# Single episode
state = env.reset()
for step in range(24):
    action = agent.select_action(state, eval_mode=True)  # No exploration
    next_state, reward, done, info = env.step(
        action,
        p_pv_avail=30.0,    # kW
        p_wt_avail=20.0,    # kW
        p_elec_load=40.0,   # kW
        q_heat_load=50.0    # kWth
    )
    
    print(f"Step {step}: Reward={reward:.2f}, Cost={info['cost']:.2f} EUR")
    state = next_state
```

## Output

During training, you'll see:

```
======================================================================
Starting TD3 Training on MEMG Environment
Episodes: 50 | Device: cpu
======================================================================

Episode 5/50 | Reward: -28.45 | Cost: 35.20€ | Violations: 7.2
Episode 10/50 | Reward: -25.12 | Cost: 28.50€ | Violations: 4.1
Episode 15/50 | Reward: -22.80 | Cost: 22.30€ | Violations: 2.3
  ✓ Checkpoint saved at episode 10
...

======================================================================
Training completed!
Final metrics:
  - Mean Reward: -18.52
  - Mean Cost: 16.25€
  - Mean Violations: 0.8
======================================================================
```

## Checkpoints

Checkpoints are saved in `./checkpoints/` every 10 episodes:

```
checkpoints/
├── td3_episode_10.pt      # Agent weights
├── stats_episode_10.json  # Training statistics
├── td3_episode_20.pt
├── stats_episode_20.json
└── ...
```

Load a checkpoint:

```python
trainer.load_checkpoint('./checkpoints/td3_episode_50.pt')
```

## Tuning Hyperparameters

Edit `training/config.py`:

```python
# Agent learning rates
AGENT_CONFIG = {
    'lr_actor': 3e-4,    # Increase to learn faster
    'lr_critic': 3e-4,
    'gamma': 0.99,       # Discount factor
    'tau': 0.005,        # Soft update rate
    'batch_size': 256,   # Smaller = faster, larger = stable
}

# Reward weights
REWARD_CONFIG = {
    'w_cost': 1.0,       # Cost minimization weight
    'w_risk': 0.3,       # Risk (CVaR) weight
    'w_constraint': 2.0, # Constraint violation penalty
}
```

## Debugging

### Check environment

```python
from memg_env.microgrid import MEMGEnvironment

env = MEMGEnvironment()
state = env.reset()

print(f"State shape: {state.shape}")
print(f"State: {state}")

for step in range(10):
    action = [0.5, 0.3, -0.2]  # Test action
    next_state, reward, done, info = env.step(action, 30, 20, 40, 50)
    print(f"Step {step}: reward={reward:.2f}, cost={info['cost']:.2f}€")
```

### Check constraint violations

```python
from memg_env.constraints import ConstraintChecker

checker = ConstraintChecker()
is_valid, violations = checker.check_battery_constraints(
    soc=0.5,          # Current SOC
    p_charge=30.0,    # Charging power (kW)
    p_discharge=0.0   # Discharging power (kW)
)

print(f"Valid: {is_valid}, Violations: {violations}")
print(checker.violations)
```

## Expected Results

After 50 episodes of training:

| Metric | Value |
|--------|-------|
| Mean Cost | 15-20 EUR |
| Mean Violations | 0-2 |
| Mean Reward | -18 to -22 |
| CVaR (95%) | 25-35 EUR |

## Performance Tips

1. **Use GPU**: `--device cuda` for 10x faster training
2. **Larger batch size**: Increases stability but slower per-step
3. **More hidden units**: Better learning but slower
4. **Higher learning rate**: Faster learning but may diverge

## Troubleshooting

### Out of Memory
Reduce `batch_size` in config.py:
```python
AGENT_CONFIG = {'batch_size': 128}
```

### Training is slow
Use GPU:
```bash
python main.py --train --device cuda
```

### Agent not learning
Increase learning rate:
```python
AGENT_CONFIG = {'lr_actor': 1e-3, 'lr_critic': 1e-3}
```

## Next Steps

1. ✅ Run `python main.py --train --episodes 50`
2. ✅ Check results in `./checkpoints/`
3. ✅ Evaluate with `python main.py --eval --checkpoint ./checkpoints/td3_episode_50.pt`
4. ✅ Modify config and experiment!

---

For more details, see [README.md](README.md)
