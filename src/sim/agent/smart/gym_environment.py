import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim.data.data_manager import data_manager
import os
from dotenv import load_dotenv

load_dotenv()
max_capacity = int(os.getenv("MAX_CAPACITY", "10"))
tariff = float(os.getenv("TARIFF", "0.75"))

interval_str = os.getenv("INTERVAL", "1,0")
hour_interval, minute_interval = map(int, interval_str.split(","))

class HEMSEnvironment(gym.Env):
    """Custom Gymnasium Environment for Home Energy Management System using SAC"""
    
    metadata = {"render_modes": []}
    
    def __init__(self, battery_max_capacity=max_capacity, tariff=tariff, 
                 hour_interval=hour_interval, minute_interval=minute_interval, max_steps=None):
        super().__init__()
        
        self.battery_max_capacity = battery_max_capacity
        self.tariff = tariff
        self.hour_interval = hour_interval
        self.minute_interval = minute_interval
        
        # Calculate max steps based on interval
        if max_steps is None:
            total_minutes_per_day = 24 * 60
            interval_minutes = hour_interval * 60 + minute_interval
            self.max_steps = total_minutes_per_day // interval_minutes
        else:
            self.max_steps = max_steps
        
        self.current_step = 0
        
        # Action space: continuous values [0, 1] for allocation percentages
        # [prod_to_cons_pct, prod_to_battery_pct, prod_to_grid_pct, 
        #  battery_to_cons_pct, battery_to_grid_pct, grid_to_battery_pct, grid_to_cons_pct]
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )
        
        # Observation space: 
        # [battery_level_normalized, price_normalized, solar_production_normalized, consumption_normalized,
        #  hour_sin, hour_cos, minute_sin, minute_cos]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Normalization constants (ADJUST BASED ON DATA!!!)
        self.max_price = 0.15  # €/kWh
        self.max_production = 5.0  # kW
        self.max_consumption = 3.0  # kW
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cur_capacity = 0
        self.balance = 0.0
        self.time_stamp = (0, 0)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        price, solar, wind, consumption = data_manager.get_model_data_entry(time_stamp=self.time_stamp)
        
        # Normalize values
        battery_normalized = self.cur_capacity / self.battery_max_capacity
        price_normalized = min(price / self.max_price, 1.0)
        solar_normalized = min(solar / self.max_production, 1.0)
        consumption_normalized = min(consumption / self.max_consumption, 1.0)
        
        # Cyclical time encoding
        hour, minute = self.time_stamp
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        minute_sin = np.sin(2 * np.pi * minute / 60)
        minute_cos = np.cos(2 * np.pi * minute / 60)
        
        return np.array([
            battery_normalized,
            price_normalized,
            solar_normalized,
            consumption_normalized,
            hour_sin,
            hour_cos,
            minute_sin,
            minute_cos
        ], dtype=np.float32)
    
    def _update_time(self):
        hour, minute = self.time_stamp
        minute += self.minute_interval
        hour += self.hour_interval
        
        if minute >= 60:
            minute -= 60
            hour += 1
        if hour >= 24:
            hour -= 24
        
        self.time_stamp = (hour, minute)
    
    def step(self, action):
        price, solar, wind, consumption = data_manager.get_model_data_entry(
            time_stamp=self.time_stamp
        )
        
        # Extract action percentages
        prod_to_cons_pct = action[0]
        prod_to_battery_pct = action[1]
        prod_to_grid_pct = action[2]
        battery_to_cons_pct = action[3]
        battery_to_grid_pct = action[4]
        grid_to_battery_pct = action[5]
        grid_to_cons_pct = action[6]
        
        # Phase 1: Production allocation
        production_to_consumption = min(prod_to_cons_pct * consumption, solar, consumption)
        remaining_production = solar - production_to_consumption
        remaining_consumption = consumption - production_to_consumption
        
        # Production to battery
        max_battery_charge = self.battery_max_capacity - self.cur_capacity
        production_to_battery = min(prod_to_battery_pct * max_battery_charge, remaining_production)
        remaining_production -= production_to_battery
        
        # Production to grid
        production_to_grid = min(prod_to_grid_pct * remaining_production, remaining_production)
        
        # Phase 2: Battery allocation
        # Battery to consumption
        battery_to_consumption = min(battery_to_cons_pct * self.cur_capacity, 
                                    remaining_consumption, self.cur_capacity)
        remaining_consumption -= battery_to_consumption
        remaining_battery = self.cur_capacity - battery_to_consumption
        
        # Battery to grid (sell stored energy)
        battery_to_grid = min(battery_to_grid_pct * remaining_battery, remaining_battery)
        
        # Phase 3: Grid allocation
        # Grid to battery (buy cheap energy to store)
        max_battery_charge_remaining = self.battery_max_capacity - (self.cur_capacity + production_to_battery - battery_to_consumption - battery_to_grid)
        grid_to_battery = grid_to_battery_pct * max(0, max_battery_charge_remaining)
        
        # Grid to consumption (buy energy for immediate use)
        grid_to_consumption = min(grid_to_cons_pct * remaining_consumption, remaining_consumption)
        
        # Calculate rewards and penalties
        reward = 0
        penalty = 0
        
        # Revenue from selling to grid
        revenue_from_production = production_to_grid * price * self.tariff
        revenue_from_battery = battery_to_grid * price * self.tariff
        total_revenue = revenue_from_production + revenue_from_battery
        
        # Cost from buying from grid
        cost_for_consumption = grid_to_consumption * price
        cost_for_battery = grid_to_battery * price
        total_cost = cost_for_consumption + cost_for_battery
        
        # Penalty for unmet consumption (critical)
        unmet_consumption = consumption - (production_to_consumption + battery_to_consumption + grid_to_consumption)
        if unmet_consumption > 0.01:  # tolerance
            penalty += 1000 * unmet_consumption
        
        # Penalty for wasted production
        wasted_production = solar - (production_to_consumption + production_to_battery + production_to_grid)
        if wasted_production > 0.01:
            penalty += 100 * wasted_production
        
        # Small penalty for battery degradation (encourage efficient use)
        battery_usage_penalty = 0.01 * (production_to_battery + grid_to_battery + battery_to_consumption + battery_to_grid)
        
        # Penalty for buying expensive energy to charge battery (discourage inefficiency)
        if price > 0.05 and grid_to_battery > 0:
            penalty += 50 * grid_to_battery * price
        
        # Reward = profit - penalties
        reward = total_revenue - total_cost - penalty - battery_usage_penalty
        
        # Update battery capacity
        battery_net_change = production_to_battery + grid_to_battery - battery_to_consumption - battery_to_grid
        self.cur_capacity += battery_net_change
        self.cur_capacity = np.clip(self.cur_capacity, 0, self.battery_max_capacity)
        
        # Update balance
        self.balance += total_revenue - total_cost
        
        # Update time and step
        self._update_time()
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "balance": self.balance,
            "battery_level": self.cur_capacity,
            "unmet_consumption": unmet_consumption,
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "production_to_grid": production_to_grid,
            "battery_to_grid": battery_to_grid,
            "grid_to_battery": grid_to_battery
        }
        
        return self._get_observation(), reward, terminated, truncated, info
