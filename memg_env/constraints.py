import numpy as np
from enum import Enum


class ComponentConstraints:
    """
    Physical and operational constraints for MEMG components
    """
    
    class BatteryConstraints:
        """Battery Energy Storage System (BESS) Constraints"""
        def __init__(self):
            self.soc_min = 0.1  # Minimum state of charge (10%)
            self.soc_max = 0.9  # Maximum state of charge (90%)
            self.P_ch_max = 50.0  # Max charging power (kW)
            self.P_dch_max = 50.0  # Max discharging power (kW)
            self.E_capacity = 100.0  # Total capacity (kWh)
            self.eta_ch = 0.95  # Charging efficiency
            self.eta_dch = 0.95  # Discharging efficiency
            self.ramp_rate = 10.0  # Max power change rate (kW/15min)
    
    class CHPConstraints:
        """Combined Heat and Power (CHP) Unit Constraints"""
        def __init__(self):
            self.P_elec_min = 10.0  # Min electric power (kW)
            self.P_elec_max = 80.0  # Max electric power (kW)
            self.Q_heat_per_elec = 1.2  # Heat output ratio (kWth per kWe)
            self.ramp_rate = 5.0  # Max power change rate (kW/15min)
            self.min_on_time = 2  # Minimum on-time (timesteps)
            self.min_off_time = 2  # Minimum off-time (timesteps)
            self.startup_cost = 50.0  # Startup cost (€)
    
    class GridConstraints:
        """Main Grid Connection Constraints"""
        def __init__(self):
            self.P_import_max = 100.0  # Max import power (kW)
            self.P_export_max = 80.0  # Max export power (kW)
            self.voltage_min = 0.95  # Min voltage (pu)
            self.voltage_max = 1.05  # Max voltage (pu)
            self.frequency_min = 49.5  # Min frequency (Hz)
            self.frequency_max = 50.5  # Max frequency (Hz)
    
    class RenewableConstraints:
        """Renewable Energy Source Constraints"""
        def __init__(self):
            self.P_pv_max = 60.0  # Max PV output (kW)
            self.P_wt_max = 40.0  # Max wind output (kW)
            # Power is bounded by weather conditions (set during episode)
    
    class LoadConstraints:
        """Load Constraints"""
        def __init__(self):
            self.P_elec_min = 5.0  # Min electric load (kW)
            self.P_elec_max = 100.0  # Max electric load (kW)
            self.Q_heat_min = 10.0  # Min heat load (kWth)
            self.Q_heat_max = 150.0  # Max heat load (kWth)


class ConstraintChecker:
    """
    Validates that MEMG operations satisfy physical constraints
    """
    
    def __init__(self):
        self.battery = ComponentConstraints.BatteryConstraints()
        self.chp = ComponentConstraints.CHPConstraints()
        self.grid = ComponentConstraints.GridConstraints()
        self.renewable = ComponentConstraints.RenewableConstraints()
        self.load = ComponentConstraints.LoadConstraints()
        self.violations = []
    
    def check_battery_constraints(self, soc, p_charge, p_discharge):
        """
        Check battery state of charge and power constraints
        
        Returns: (is_valid, violation_count)
        """
        violations = 0
        
        # SOC bounds
        if soc < self.battery.soc_min or soc > self.battery.soc_max:
            violations += 1
            self.violations.append(f"Battery SOC out of bounds: {soc:.3f}")
        
        # Charging power limit
        if p_charge > self.battery.P_ch_max:
            violations += 1
            self.violations.append(f"Battery charge power exceeds limit: {p_charge:.2f} > {self.battery.P_ch_max}")
        
        # Discharging power limit
        if p_discharge > self.battery.P_dch_max:
            violations += 1
            self.violations.append(f"Battery discharge power exceeds limit: {p_discharge:.2f} > {self.battery.P_dch_max}")
        
        return violations == 0, violations
    
    def check_chp_constraints(self, p_elec, is_on):
        """
        Check CHP power constraints
        """
        violations = 0
        
        if is_on:
            if p_elec < self.chp.P_elec_min:
                violations += 1
                self.violations.append(f"CHP below minimum power: {p_elec:.2f} < {self.chp.P_elec_min}")
            if p_elec > self.chp.P_elec_max:
                violations += 1
                self.violations.append(f"CHP above maximum power: {p_elec:.2f} > {self.chp.P_elec_max}")
        
        return violations == 0, violations
    
    def check_grid_constraints(self, p_import, p_export):
        """
        Check grid power exchange constraints
        """
        violations = 0
        
        if p_import > self.grid.P_import_max:
            violations += 1
            self.violations.append(f"Grid import exceeds limit: {p_import:.2f} > {self.grid.P_import_max}")
        
        if p_export > self.grid.P_export_max:
            violations += 1
            self.violations.append(f"Grid export exceeds limit: {p_export:.2f} > {self.grid.P_export_max}")
        
        return violations == 0, violations
    
    def check_power_balance(self, p_pv, p_wt, p_chp, p_elec_load, 
                           p_bat_ch, p_bat_dch, p_import, p_export):
        """
        Check power balance: generation = demand
        Allows small tolerance for numerical errors
        """
        tolerance = 1.0  # kW
        
        # Generation side
        generation = p_pv + p_wt + p_chp + p_import
        
        # Demand side
        demand = p_elec_load + p_bat_ch + p_export
        
        # Account for battery discharging as generation
        generation += p_bat_dch
        
        imbalance = abs(generation - demand)
        
        if imbalance > tolerance:
            self.violations.append(f"Power balance violation: {imbalance:.2f} kW")
            return False, 1
        
        return True, 0
    
    def clear_violations(self):
        """Clear violation log"""
        self.violations = []
    
    def get_violation_count(self):
        """Return number of violations"""
        return len(self.violations)
