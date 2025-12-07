import numpy as np
from .constraints import ConstraintChecker, ComponentConstraints


class MEMGEnvironment:
    """
    Multi-Energy Microgrid (MEMG) Environment for RL Training
    
    Components:
    - Electric Bus: PV, WT, Battery, CHP, Grid connection
    - Thermal Bus: CHP, Gas boiler, Thermal storage (implicit)
    - Gas Bus: CHP, Gas boiler, Grid supply
    
    State Space:
    [P_pv_avail, P_wt_avail, P_elec_load, Q_heat_load, 
     SOC_battery, CHP_status, T_ambient, Price_elec, 
     Hour_of_day, Day_of_week]
    
    Action Space:
    [P_bat_charge, P_chp_output, P_grid_import]
    All normalized to [-1, 1] and scaled to actual power ranges
    """
    
    def __init__(self, forecast_horizon=24, timestep_min=15):
        """
        Initialize MEMG environment
        
        Args:
            forecast_horizon: Lookahead window (timesteps)
            timestep_min: Duration of each timestep (minutes)
        """
        self.forecast_horizon = forecast_horizon
        self.timestep_min = timestep_min
        self.constraint_checker = ConstraintChecker()
        
        # Component constraints
        self.battery = self.constraint_checker.battery
        self.chp = self.constraint_checker.chp
        self.grid = self.constraint_checker.grid
        
        # State space dimension
        self.state_dim = 10
        
        # Action space dimension
        self.action_dim = 3
        
        # Initialize state
        self._reset_state()
    
    def _reset_state(self):
        """
        Initialize state variables
        """
        self.soc_battery = 0.5  # 50% initial charge
        self.p_pv_available = 0.0
        self.p_wt_available = 0.0
        self.p_elec_load = 30.0
        self.q_heat_load = 40.0
        self.chp_is_on = False
        self.chp_startup_cost = 0.0
        self.price_elec_import = 0.15  # €/kWh
        self.price_elec_export = 0.08  # €/kWh
        self.price_gas = 0.05  # €/kWh
        self.timestep = 0
        self.cumulative_cost = 0.0
        self.constraint_violations = 0
    
    def reset(self):
        """
        Reset environment for new episode
        
        Returns: Initial state
        """
        self._reset_state()
        return self._get_state()
    
    def step(self, action, p_pv_avail, p_wt_avail, p_elec_load, q_heat_load):
        """
        Execute one timestep of environment
        
        Args:
            action: Normalized action [a1, a2, a3] in [-1, 1]
            p_pv_avail: Available PV power (kW)
            p_wt_avail: Available WT power (kW)
            p_elec_load: Electric load demand (kW)
            q_heat_load: Heat load demand (kWth)
        
        Returns:
            next_state, reward, done, info
        """
        # Denormalize actions to physical ranges
        action = np.clip(action, -1.0, 1.0)
        
        # Action mapping:
        # a1: Battery charge/discharge (-1 to 1) -> (-50 to +50 kW)
        # a2: CHP output (0 to 1) -> (0 to 80 kW)
        # a3: Grid import/export (-1 to 1) -> (-80 to +100 kW)
        
        p_bat_setpoint = action[0] * 50.0  # Positive = charging
        p_chp_setpoint = ((action[1] + 1.0) / 2.0) * self.chp.P_elec_max  # [0, 1] -> [0, P_max]
        p_grid_setpoint = action[2] * 50.0  # Positive = importing
        
        # Update available resources
        self.p_pv_available = np.clip(p_pv_avail, 0, self.constraint_checker.renewable.P_pv_max)
        self.p_wt_available = np.clip(p_wt_avail, 0, self.constraint_checker.renewable.P_wt_max)
        self.p_elec_load = np.clip(p_elec_load, 
                                   self.constraint_checker.load.P_elec_min,
                                   self.constraint_checker.load.P_elec_max)
        self.q_heat_load = np.clip(q_heat_load,
                                   self.constraint_checker.load.Q_heat_min,
                                   self.constraint_checker.load.Q_heat_max)
        
        # ========== COMPUTE ACTUAL POWER FLOWS ==========
        p_bat_actual, p_chp_actual, p_grid_actual, cost = self._compute_power_dispatch(
            p_bat_setpoint, p_chp_setpoint, p_grid_setpoint
        )
        
        # ========== UPDATE BATTERY STATE OF CHARGE ==========
        energy_change = (p_bat_actual * self.battery.eta_ch if p_bat_actual > 0 
                        else p_bat_actual / self.battery.eta_dch) * (self.timestep_min / 60.0)
        self.soc_battery += energy_change / self.battery.E_capacity
        self.soc_battery = np.clip(self.soc_battery, self.battery.soc_min, self.battery.soc_max)
        
        # ========== CHECK CONSTRAINTS ==========
        self.constraint_checker.clear_violations()
        constraint_penalty = self._evaluate_constraints(p_bat_actual, p_chp_actual, p_grid_actual)
        
        # ========== COMPUTE REWARD ==========
        reward = self._compute_reward(cost, constraint_penalty, p_bat_actual)
        
        self.cumulative_cost += cost
        self.constraint_violations += self.constraint_checker.get_violation_count()
        
        # Update timestep
        self.timestep += 1
        done = self.timestep >= self.forecast_horizon
        
        info = {
            'cost': cost,
            'constraint_violations': self.constraint_checker.get_violation_count(),
            'soc_battery': self.soc_battery,
            'p_bat': p_bat_actual,
            'p_chp': p_chp_actual,
            'p_grid': p_grid_actual,
            'cumulative_cost': self.cumulative_cost
        }
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _compute_power_dispatch(self, p_bat_sp, p_chp_sp, p_grid_sp):
        """
        Resolve power balance and compute actual power flows
        Uses constraint-aware dispatch
        
        Returns:
            p_bat_actual, p_chp_actual, p_grid_actual, cost
        """
        # Initialize power flows
        p_pv = self.p_pv_available
        p_wt = self.p_wt_available
        p_bat_ch = 0.0
        p_bat_dch = 0.0
        p_chp = 0.0
        p_grid_import = 0.0
        p_grid_export = 0.0
        cost = 0.0
        
        # Step 1: Try to satisfy load with renewable + battery
        renewable_supply = p_pv + p_wt
        remaining_load = self.p_elec_load - renewable_supply
        
        if remaining_load > 0:
            # Need additional power
            if p_bat_sp > 0 and self.soc_battery > self.battery.soc_min:
                # Use battery discharging
                p_bat_dch = min(p_bat_sp, remaining_load, 
                               (self.soc_battery - self.battery.soc_min) * self.battery.E_capacity / (self.timestep_min / 60.0))
                remaining_load -= p_bat_dch
            
            # Try CHP
            if p_chp_sp > self.chp.P_elec_min:
                p_chp = np.clip(p_chp_sp, self.chp.P_elec_min, self.chp.P_elec_max)
                remaining_load -= p_chp
            
            # Grid import for remaining
            if remaining_load > 0:
                p_grid_import = np.clip(remaining_load, 0, self.grid.P_import_max)
                cost += p_grid_import * self.price_elec_import
        
        elif remaining_load < 0:
            # Excess power - charge battery or export
            excess = -remaining_load
            
            if p_bat_sp < 0 and self.soc_battery < self.battery.soc_max:
                # Charge battery
                p_bat_ch = min(abs(p_bat_sp), excess,
                              (self.battery.soc_max - self.soc_battery) * self.battery.E_capacity / (self.timestep_min / 60.0))
                excess -= p_bat_ch
            
            # Export excess to grid
            if excess > 0:
                p_grid_export = np.clip(excess, 0, self.grid.P_export_max)
                cost -= p_grid_export * self.price_elec_export  # Negative cost (income)
        
        # Heat balance (simplified - CHP covers part of heat load)
        heat_from_chp = p_chp * self.chp.Q_heat_per_elec
        
        # Gas boiler covers remaining heat demand
        q_heat_deficit = max(0, self.q_heat_load - heat_from_chp)
        cost += q_heat_deficit * self.price_gas  # Gas cost
        
        # Total battery power
        p_bat_total = p_bat_dch - p_bat_ch
        
        # Total grid power
        p_grid_total = p_grid_import - p_grid_export
        
        return p_bat_total, p_chp, p_grid_total, cost
    
    def _evaluate_constraints(self, p_bat, p_chp, p_grid):
        """
        Evaluate constraint violations and return penalty
        """
        penalty = 0.0
        violation_weight = 10.0  # €/violation
        
        # Battery constraints
        _, n_viol = self.constraint_checker.check_battery_constraints(
            self.soc_battery, 
            max(0, -p_bat),  # Charging
            max(0, p_bat)    # Discharging
        )
        penalty += n_viol * violation_weight
        
        # CHP constraints
        _, n_viol = self.constraint_checker.check_chp_constraints(p_chp, p_chp > 0)
        penalty += n_viol * violation_weight
        
        # Grid constraints
        _, n_viol = self.constraint_checker.check_grid_constraints(
            max(0, p_grid),   # Import
            max(0, -p_grid)   # Export
        )
        penalty += n_viol * violation_weight
        
        return penalty
    
    def _compute_reward(self, cost, constraint_penalty, p_bat):
        """
        Compute reward function
        Reward = -cost - constraint_penalty + battery_smoothness_bonus
        """
        # Operational cost
        reward = -cost - constraint_penalty
        
        # Bonus for smooth battery operation (reduces degradation)
        battery_smoothness_bonus = -0.1 * abs(p_bat) / self.battery.P_dch_max
        reward += battery_smoothness_bonus
        
        return reward
    
    def _get_state(self):
        """
        Construct state vector
        
        Returns: Normalized state [0, 1] or [-1, 1]
        """
        # Normalize state components
        p_pv_norm = self.p_pv_available / self.constraint_checker.renewable.P_pv_max
        p_wt_norm = self.p_wt_available / self.constraint_checker.renewable.P_wt_max
        p_elec_norm = self.p_elec_load / self.constraint_checker.load.P_elec_max
        q_heat_norm = self.q_heat_load / self.constraint_checker.load.Q_heat_max
        soc_norm = self.soc_battery
        chp_status = float(self.chp_is_on)
        
        # Time features
        hour = (self.timestep * self.timestep_min / 60) % 24
        hour_norm = hour / 24.0
        
        # Price
        price_norm = self.price_elec_import / 0.3  # Normalize
        
        state = np.array([
            p_pv_norm,
            p_wt_norm,
            p_elec_norm,
            q_heat_norm,
            soc_norm,
            chp_status,
            price_norm,
            hour_norm,
            np.sin(2 * np.pi * hour / 24),  # Hour sin
            np.cos(2 * np.pi * hour / 24)   # Hour cos
        ], dtype=np.float32)
        
        return state
    
    def get_info(self):
        """
        Return environment information
        """
        return {
            'soc': self.soc_battery,
            'cumulative_cost': self.cumulative_cost,
            'constraint_violations': self.constraint_violations,
            'timestep': self.timestep
        }
