# TD3 MEMG EMS - Technical Architecture

## System Overview

```
┌────────────────────────────────────────┐
│          TD3 MEMG Energy Management System                      │
└────────────────────────────────────────┘
    ┌─────────────┌─────────────┌─────────────┌────────────┌
    │  MEMG Env  │  TD3 Agent  │  Reward Fn  │  Trainer  │
    └─────────────┘─────────────┘─────────────┘────────────┘
```

## Module Hierarchy

### 1. **TD3 Agent (`td3_agent/`)**

#### Core Components

```python
TD3Agent
├── Actor Network (DNN)
│   ├── FC1(state_dim -> 256)
│   ├── ReLU
│   ├── FC2(256 -> 256)
│   ├── ReLU
│   └── FC3(256 -> action_dim) -> Tanh
│
├── Critic Network (Dual Q-Networks)
│   ├── Q1(state, action) -> scalar
│   └── Q2(state, action) -> scalar
│
├── Target Networks
│   ├── Actor_target
│   └── Critic_target
│
└── Replay Buffer
    ├── store(state, action, reward, next_state, done)
    └── sample(batch_size)
```

#### TD3 Update Equations

**Critic Update (every step):**
```
Q_loss = MSE(Q(s, a), r + γ * min(Q'(s', a')))
where a' = Actor'(s') + noise
```

**Actor Update (every d steps, d=2):**
```
Actor_loss = -Q(s, Actor(s)).mean()
```

**Soft Target Update:**
```
Q'_param = τ * Q_param + (1 - τ) * Q'_param
τ = 0.005 (polyak average)
```

### 2. **MEMG Environment (`memg_env/`)**

#### State Space (10D)

```python
state = [
    0: p_pv_norm             # [0, 1] normalized PV availability
    1: p_wt_norm             # [0, 1] normalized wind availability  
    2: p_elec_load_norm      # [0, 1] normalized electric load
    3: q_heat_load_norm      # [0, 1] normalized thermal load
    4: soc_battery           # [0.1, 0.9] battery state of charge
    5: chp_status            # {0, 1} CHP on/off
    6: price_elec_norm       # [0, 1] normalized electricity price
    7: hour_of_day_norm      # [0, 1] normalized hour (0-23)
    8: sin(2π*hour/24)      # Cyclic hour encoding
    9: cos(2π*hour/24)      # Cyclic hour encoding
]
```

#### Action Space (3D)

```python
action = [
    0: p_bat_setpoint   ∈ [-1, 1] -> [-50, +50] kW
                        (negative: discharge, positive: charge)
    1: p_chp_setpoint   ∈ [0, 1] -> [0, 80] kW
    2: p_grid_setpoint  ∈ [-1, 1] -> [-80, +100] kW
                        (negative: export, positive: import)
]
```

#### Power Flow Dispatch Algorithm

```
1. Renewable generation (PV + WT) available
2. Calculate remaining load after renewable supply
3. If deficit:
   a. Try battery discharging (if SoC > min)
   b. Try CHP (if output > min power)
   c. Use grid import (if within limit)
4. If surplus:
   a. Try battery charging (if SoC < max)
   b. Export excess to grid (if within limit)
5. Calculate thermal balance with CHP + gas boiler
6. Compute total cost
```

#### Constraints Enforcement

```
Battery:
  SoC ∈ [0.1, 0.9]        # State of charge limits
  P_ch ≤ 50 kW            # Max charging power
  P_dch ≤ 50 kW           # Max discharging power
  ramp ≤ 10 kW/15min      # Max ramp rate
  η_ch = 0.95, η_dch = 0.95

CHP:
  P ∈ [10, 80] kW when ON
  Q_heat = 1.2 * P_elec
  ramp ≤ 5 kW/15min
  min_on = 2, min_off = 2 timesteps

Grid:
  P_import ≤ 100 kW
  P_export ≤ 80 kW
  V ∈ [0.95, 1.05] pu
  f ∈ [49.5, 50.5] Hz
```

### 3. **Reward Function (`reward_function/`)**

#### CVaR-Based Risk-Aware Reward

```python
Reward = -cost - constraint_penalty - battery_wear - risk_penalty

where:
  cost = electricity_cost + gas_cost - export_income
  
  constraint_penalty = w_constraint * (
    num_battery_violations +
    num_chp_violations +
    num_grid_violations
  )
  
  battery_wear = 0.1 * |P_bat| / P_dch_max
  
  risk_penalty = λ * (CVaR_α - E[cost])
                where α = 0.95, λ = 0.3
```

#### CVaR Computation

```
Given cost history C = {c1, c2, ..., cn}:

1. Sort costs: C_sorted
2. Find (1-α) quantile: idx = ceil((1-α)*n)
3. VaR = C_sorted[idx]
4. CVaR = mean(C_sorted[idx:])  # Mean of worst cases
5. Risk = CVaR - E[C]
```

For α=0.95: focus on worst 5% scenarios

### 4. **Training Loop (`training/`)**

#### Episode Structure

```
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    episode_cost = 0
    
    # Generate synthetic profiles for this episode
    pv_profile = synthetic_pv(24)
    wt_profile = synthetic_wt(24)
    load_profile = synthetic_load(24)
    heat_profile = synthetic_heat(24)
    
    for step in range(24):  # 24 * 15min = 6 hours
        # Exploration: add noise to action
        action = agent.select_action(state, eval_mode=False)
        noise = N(0, 0.2) clipped to [-0.5, 0.5]
        action += noise
        
        # Environment step
        next_state, reward, done, info = env.step(
            action,
            pv_profile[step],
            wt_profile[step],
            load_profile[step],
            heat_profile[step]
        )
        
        # Store experience
        agent.replay_buffer.add(
            state, action, reward, next_state, done
        )
        
        # Train from batch
        train_metrics = agent.train(batch_size=256)
        
        # Update state
        state = next_state
        episode_reward += reward
        episode_cost += info['cost']
    
    # Save checkpoint every 10 episodes
    if episode % 10 == 0:
        agent.save_checkpoint(f"td3_episode_{episode}.pt")
```

## Data Flow Diagram

```
                    ┌─────────────────────┐
                    │  MEMG Environment   │
                    │  Power Dispatch     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  State (10D)        │
                    │  Reward             │
                    │  Info               │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Replay Buffer      │
                    │  Store Experience   │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
    │  Actor  │          │ Critic  │          │ Target  │
    │ Network │◄─────────│ Network │──────────│Networks │
    └────┬────┘          └────▲────┘          └────────┘
         │                    │
    Action                 Q-Loss
         │                    │
         └────────┬───────────┘
                  │
        ┌─────────▼──────────┐
        │  Optimization      │
        │  (Adam)            │
        └────────────────────┘
```

## Performance Characteristics

### Computational Complexity

| Component | Per-Step Time | Memory |
|-----------|---------------|--------|
| Environment step | O(1) | ~10 floats |
| Agent action | O(256² * 2) | ~500 KB (networks) |
| Critic update | O(batch_size * hidden²) | ~1 MB (buffers) |
| Actor update | O(batch_size * hidden²) | ~1 MB |
| Total per step | ~10 ms (CPU) | ~2 MB |

### Scaling

- **State dim**: 10 → increases actor input
- **Action dim**: 3 → increases critic input
- **Hidden dim**: 256 → scales networks O(hidden²)
- **Batch size**: 256 → scales training time linearly
- **Buffer size**: 1M transitions → ~100 MB RAM

## Key Design Decisions

### 1. Twin Q-Networks
**Why**: Reduces overestimation bias in off-policy RL
```
min(Q1, Q2) < avg(Q1, Q2) < max(Q1, Q2)
```

### 2. Delayed Actor Update
**Why**: Stabilizes policy learning by waiting for critic convergence
```
Update critic every step, actor every 2 steps
```

### 3. Target Policy Smoothing
**Why**: Reduces exploitability by adding noise to target actions
```
a' = clip(π(s') + clip(ε, -c, c))
```

### 4. CVaR Reward
**Why**: Risk-aware control for worst-case scenarios
```
Minimize both mean cost and tail risk
```

### 5. Synthetic Profiles
**Why**: Enables training without real-world data
```
PV: sinusoidal + random
Wind: random walk
Load: peak hours + variation
```

## Extension Points

### 1. **Multi-Agent TD3**
```python
# For multiple microgrids with coordination
from td3_agent.td3_core import TD3Agent

agents = [
    TD3Agent(...) for _ in range(num_microgrids)
]

# Shared critic for coordination
shared_critic = Critic(total_state_dim, total_action_dim)
```

### 2. **Model-Based Planning**
```python
# Learn environment dynamics
from dynamics_model import DynamicsModel

model = DynamicsModel(state_dim, action_dim)
# Use learned model for lookahead
```

### 3. **Attention Mechanisms**
```python
# For temporal dependencies
from nn.attention import MultiHeadAttention

class AttentionActor(nn.Module):
    def __init__(self):
        self.attention = MultiHeadAttention(...)
```

### 4. **Constraint-Aware Learning**
```python
# Lagrangian Actor-Critic
from lac import ConstrainedActor

actor = ConstrainedActor(
    state_dim, action_dim,
    constraint_fn=lambda s, a: check_feasibility(s, a)
)
```

## Testing & Validation

### Unit Tests
```python
# Test constraint checker
def test_battery_constraints():
    checker = ConstraintChecker()
    assert checker.check_battery_constraints(0.5, 30, 0) == (True, 0)
    assert checker.check_battery_constraints(0.05, 30, 0) == (False, 1)

# Test environment
def test_power_balance():
    env = MEMGEnvironment()
    state = env.reset()
    next_state, reward, done, info = env.step(...)
    assert abs(power_balance) < tolerance
```

### Integration Tests
```python
# Full training pipeline
def test_training_loop():
    trainer = TD3Trainer()
    trainer.train(num_episodes=5)
    assert len(trainer.episode_rewards) == 5
    assert trainer.agent.total_it > 100  # At least 20 steps/episode
```

---

**Author**: emir1111rezaa  
**Last Updated**: 2025-12-07
