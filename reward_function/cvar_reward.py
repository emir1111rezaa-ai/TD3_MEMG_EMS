import numpy as np
from collections import deque


class CVaRReward:
    """
    Conditional Value at Risk (CVaR) Based Reward Function
    
    CVaR-Risk-Aware: Penalizes high-cost scenarios (tail risk)
    Instead of just minimizing mean cost, we also minimize worst-case scenarios
    
    Formula: CVaR_alpha(X) = E[X | X > VaR_alpha(X)]
    where VaR_alpha is the alpha-quantile of the distribution
    """
    
    def __init__(self, alpha=0.95, window_size=100):
        """
        Initialize CVaR reward function
        
        Args:
            alpha: Confidence level (0.95 = focus on worst 5% scenarios)
            window_size: Rolling window for cost history
        """
        self.alpha = alpha
        self.window_size = window_size
        self.cost_history = deque(maxlen=window_size)
        self.constraint_violation_history = deque(maxlen=window_size)
    
    def compute_cvar(self, costs):
        """
        Compute CVaR from cost samples
        
        Args:
            costs: Array of costs
        
        Returns:
            CVaR value
        """
        if len(costs) == 0:
            return 0.0
        
        costs_sorted = np.sort(costs)
        n = len(costs_sorted)
        
        # Find the (1-alpha) quantile index
        quantile_idx = int(np.ceil((1 - self.alpha) * n)) - 1
        quantile_idx = max(0, quantile_idx)
        
        var = costs_sorted[quantile_idx]
        
        # CVaR: mean of values >= VaR
        cvar = costs_sorted[quantile_idx:].mean()
        
        return cvar
    
    def compute_var(self, costs):
        """
        Compute Value at Risk (VaR) - the quantile
        """
        if len(costs) == 0:
            return 0.0
        
        costs_sorted = np.sort(costs)
        n = len(costs_sorted)
        
        quantile_idx = int(np.ceil((1 - self.alpha) * n)) - 1
        quantile_idx = max(0, quantile_idx)
        
        return costs_sorted[quantile_idx]
    
    def update_history(self, cost, constraint_violations):
        """
        Update cost and constraint violation history
        """
        self.cost_history.append(cost)
        self.constraint_violation_history.append(constraint_violations)
    
    def get_risk_metrics(self):
        """
        Compute risk metrics from history
        
        Returns:
            Dictionary with VaR, CVaR, mean cost, std cost
        """
        if len(self.cost_history) == 0:
            return {
                'var': 0.0,
                'cvar': 0.0,
                'mean_cost': 0.0,
                'std_cost': 0.0,
                'max_cost': 0.0
            }
        
        costs = np.array(list(self.cost_history))
        
        return {
            'var': self.compute_var(costs),
            'cvar': self.compute_cvar(costs),
            'mean_cost': costs.mean(),
            'std_cost': costs.std(),
            'max_cost': costs.max()
        }
    
    def compute_risk_reward(self, base_cost, constraint_penalty):
        """
        Compute reward with CVaR risk consideration
        
        Risk-Aware Reward = -base_cost - lambda * (CVaR - mean_cost) - constraint_penalty
        
        Where:
        - base_cost: Immediate operational cost
        - lambda: Risk aversion coefficient (0.5 typical)
        - CVaR - mean_cost: Tail risk (worst-case scenarios)
        - constraint_penalty: Penalty for constraint violations
        
        Args:
            base_cost: Immediate cost from this timestep
            constraint_penalty: Penalty for violated constraints
        
        Returns:
            Risk-aware reward
        """
        lambda_risk = 0.3  # Risk aversion coefficient
        
        # Basic reward
        reward = -base_cost - constraint_penalty
        
        # Add risk component if we have history
        if len(self.cost_history) > 10:
            metrics = self.get_risk_metrics()
            mean_cost = metrics['mean_cost']
            cvar = metrics['cvar']
            
            # Penalize high CVaR (tail risk)
            tail_risk = cvar - mean_cost
            reward -= lambda_risk * tail_risk
        
        return reward
    
    def get_tail_risk_penalty(self):
        """
        Get current tail risk penalty for logging
        """
        if len(self.cost_history) <= 10:
            return 0.0
        
        metrics = self.get_risk_metrics()
        return metrics['cvar'] - metrics['mean_cost']


class RiskAwareEMSReward:
    """
    Multi-Objective Risk-Aware Reward for MEMG-EMS
    
    Objectives:
    1. Minimize operational cost
    2. Minimize tail risk (CVaR)
    3. Minimize constraint violations
    4. Encourage battery health (smooth operation)
    5. Achieve thermal comfort
    """
    
    def __init__(self, alpha_cvar=0.95):
        self.cvar_reward = CVaRReward(alpha=alpha_cvar)
        
        # Reward weights
        self.w_cost = 1.0
        self.w_risk = 0.3
        self.w_constraint = 2.0
        self.w_battery = 0.1
        self.w_thermal = 0.5
    
    def compute_episode_reward(self, cost, constraint_penalty, p_bat, 
                               thermal_discomfort=0.0):
        """
        Compute multi-objective reward
        
        Args:
            cost: Operational cost (€)
            constraint_penalty: Constraint violation penalty
            p_bat: Battery power (kW) - for smoothness
            thermal_discomfort: Heat balance error (kWth)
        
        Returns:
            Composite reward
        """
        # Base cost penalty
        cost_penalty = self.w_cost * cost
        
        # Constraint penalty
        constraint_penalty_weighted = self.w_constraint * constraint_penalty
        
        # Battery smoothness bonus (penalize large power changes)
        battery_penalty = self.w_battery * abs(p_bat) / 50.0
        
        # Thermal comfort penalty
        thermal_penalty = self.w_thermal * thermal_discomfort
        
        # Total reward
        reward = -(cost_penalty + constraint_penalty_weighted + 
                   battery_penalty + thermal_penalty)
        
        # Update CVaR history
        self.cvar_reward.update_history(cost, max(0, constraint_penalty))
        
        # Add risk component
        reward += -self.w_risk * self.cvar_reward.get_tail_risk_penalty()
        
        return reward
    
    def get_training_metrics(self):
        """
        Get metrics for monitoring training
        """
        metrics = self.cvar_reward.get_risk_metrics()
        metrics['tail_risk_penalty'] = self.cvar_reward.get_tail_risk_penalty()
        return metrics
