# TD3 Agent for Multi-Energy Microgrid (MEMG) Energy Management System

**Twin Delayed DDPG (TD3)** implementation for optimal energy management in a **Multi-Energy Microgrid** environment.

## 🎯 Overview

This repository contains a state-of-the-art implementation of the TD3 reinforcement learning algorithm applied to real-time energy management in microgrids with multiple energy carriers:

- **Electrical Bus**: PV panels, wind turbines, battery storage, CHP unit, main grid
- **Thermal Bus**: CHP heat output, gas boiler
- **Gas Bus**: CHP fuel consumption, boiler fuel supply

## 🚀 Key Features

### 1. **TD3 Algorithm Implementation**
   - Twin Q-networks to reduce overestimation bias
   - Delayed policy updates for stability
   - Target policy smoothing with noise clipping
   - Soft target network updates (polyak averaging)

### 2. **MEMG Environment**
   - Realistic physical constraints:
     - Battery: SOC limits, power ramps, round-trip efficiency
     - CHP: Min/max power, thermal coupling, startup costs
     - Grid: Import/export limits, voltage/frequency bounds
   - Power balance enforcement
   - Efficient dispatch algorithm

### 3. **Risk-Aware Control**
   - **CVaR (Conditional Value at Risk)** integration
   - Minimization of tail risk (worst-case scenarios)
   - Robust decision-making under uncertainty

### 4. **Modular Architecture**
   ```
   TD3_MEMG_EMS/
   ├── td3_agent/          # Core TD3 algorithm
   ├── memg_env/           # MEMG environment & constraints
   ├── reward_function/    # CVaR reward function
   ├── training/           # Training loop & configuration
   └── main.py             # Entry point
   ```

## 💻 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, Pandas

### Setup

```bash
# Clone repository
git clone https://github.com/emir1111rezaa-ai/TD3_MEMG_EMS.git
cd TD3_MEMG_EMS

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Quick Start

### Training

Train the TD3 agent for 50 episodes:

```bash
python main.py --train --episodes 50 --device cpu
```

**Options:**
- `--episodes`: Number of training episodes (default: 50)
- `--device`: Device to use `cpu` or `cuda` (default: cpu)
- `--save-config`: Save configuration to `config.yaml`

### Evaluation

Evaluate a trained agent:

```bash
python main.py --eval --checkpoint ./checkpoints/td3_episode_50.pt --device cpu
```

## 🛰 Algorithm Details

### TD3 (Twin Delayed DDPG)

TD3 addresses overestimation bias in Q-learning through three mechanisms:

1. **Twin Q-Networks**: Take minimum of two Q-networks
2. **Delayed Policy Update**: Update actor less frequently than critic
3. **Target Policy Smoothing**: Add noise to target actions

### MEMG Energy Dispatch

**Power Balance**:
```
P_PV + P_WT + P_CHP + P_grid_import = P_load + P_bat_charge + P_grid_export
```

**Actions** (normalized to [-1, 1]):
- `a1`: Battery charge/discharge setpoint
- `a2`: CHP output
- `a3`: Grid import/export

## 📁 File Structure

### `td3_agent/`
- **`td3_core.py`**: Main TD3 agent class
- **`networks.py`**: Actor/Critic networks
- **`replay_buffer.py`**: Experience replay buffer

### `memg_env/`
- **`microgrid.py`**: MEMG environment
- **`constraints.py`**: Physical constraints

### `reward_function/`
- **`cvar_reward.py`**: Risk-aware reward

### `training/`
- **`trainer.py`**: Training loop
- **`config.py`**: Configuration

## 🎬 Example Usage

```bash
# Train for 50 episodes
python main.py --train --episodes 50

# Evaluate trained agent
python main.py --eval --checkpoint ./checkpoints/td3_episode_50.pt

# Save configuration
python main.py --save-config
```

## 📚 References

- Fujimoto et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods" (ICML)
- GitHub: [sfujim/TD3](https://github.com/sfujim/TD3)

## 👨‍💻 Author

Developed by **emir1111rezaa**

## 📄 License

MIT License
