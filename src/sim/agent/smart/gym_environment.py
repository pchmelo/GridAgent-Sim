import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim.data.data_manager import DataManager
import os
from dotenv import load_dotenv

load_dotenv()
max_capacity = int(os.getenv("MAX_CAPACITY", "10"))
tariff = float(os.getenv("TARIFF", "0.75"))

interval_str = os.getenv("INTERVAL", "1,0")
hour_interval, minute_interval = map(int, interval_str.split(","))

class HEMSEnvironment(gym.Env):
    """HEMS Environment with improved arbitrage reward function"""
    
    metadata = {"render_modes": []}
    
    def __init__(self, battery_max_capacity=max_capacity, tariff=tariff, 
                 hour_interval=hour_interval, minute_interval=minute_interval, 
                 max_steps=None, date=None):
        super().__init__()
        
        self.battery_max_capacity = battery_max_capacity
        self.tariff = tariff
        self.hour_interval = hour_interval
        self.minute_interval = minute_interval
        self.date = date
        
        if date:
            self.data_manager = DataManager(date=date)
            self.data_manager.start_data_collection(date)
        else:
            from sim.data.data_manager import data_manager
            self.data_manager = data_manager
        
        if max_steps is None:
            total_minutes_per_day = 24 * 60
            interval_minutes = hour_interval * 60 + minute_interval
            self.max_steps = total_minutes_per_day // interval_minutes
        else:
            self.max_steps = max_steps
        
        self.current_step = 0
        self.price_history = []
        self.avg_price = 0.0
        self.min_price_seen = float('inf')
        self.max_price_seen = 0.0
        self.battery_cost_basis = 0.0
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, -1, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.max_price = 0.5
        self.max_production = 5.0
        self.max_consumption = 3.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cur_capacity = 0
        self.balance = 0.0
        self.time_stamp = (0, 0)
        self.price_history = []
        self.avg_price = 0.0
        self.min_price_seen = float('inf')
        self.max_price_seen = 0.0
        self.battery_cost_basis = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        price, solar, wind, consumption = self.data_manager.get_model_data_entry(time_stamp=self.time_stamp)
        
        battery_normalized = self.cur_capacity / self.battery_max_capacity
        price_normalized = np.clip(price / self.max_price, 0, 1)
        solar_normalized = np.clip(solar / self.max_production, 0, 1)
        consumption_normalized = np.clip(consumption / self.max_consumption, 0, 1)
        
        hour, _ = self.time_stamp
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        if len(self.price_history) > 0:
            price_vs_avg = np.clip((price - self.avg_price) / (self.avg_price + 0.01) + 0.5, 0, 1)
        else:
            price_vs_avg = 0.5
        
        if self.cur_capacity > 0 and self.battery_cost_basis > 0:
            battery_value = np.clip((price * self.tariff - self.battery_cost_basis) / (self.battery_cost_basis + 0.01) + 0.5, 0, 1)
        else:
            battery_value = 0.5
        
        if len(self.price_history) >= 3:
            recent_trend = (price - self.price_history[-3]) / (self.avg_price + 0.01)
            price_trend = np.clip(recent_trend + 0.5, 0, 1)
        else:
            price_trend = 0.5
        
        remaining_hours = (self.max_steps - self.current_step) / self.max_steps
        
        return np.array([
            battery_normalized,
            price_normalized,
            solar_normalized,
            consumption_normalized,
            hour_sin,
            hour_cos,
            price_vs_avg,
            battery_value,
            price_trend,
            remaining_hours,
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
    
    def _get_price_percentile(self, price):
        """Calculate where current price falls in the day's price range"""
        if self.max_price_seen <= self.min_price_seen:
            return 0.5
        return (price - self.min_price_seen) / (self.max_price_seen - self.min_price_seen)
    
    def step(self, action):
        price, solar, wind, consumption = self.data_manager.get_model_data_entry(
            time_stamp=self.time_stamp
        )
        
        self.price_history.append(price)
        self.avg_price = np.mean(self.price_history)
        self.min_price_seen = min(self.min_price_seen, price)
        self.max_price_seen = max(self.max_price_seen, price)
        
        prod_to_cons_pct = action[0]
        prod_to_battery_pct = action[1]
        prod_to_grid_pct = action[2]
        battery_to_cons_pct = action[3]
        battery_to_grid_pct = action[4]
        grid_to_battery_pct = action[5]
        grid_to_cons_pct = action[6]
        
        battery_before = self.cur_capacity
        
        production_to_consumption = min(prod_to_cons_pct * consumption, solar, consumption)
        remaining_production = solar - production_to_consumption
        remaining_consumption = consumption - production_to_consumption
        
        max_battery_charge = self.battery_max_capacity - self.cur_capacity
        production_to_battery = min(prod_to_battery_pct * max_battery_charge, remaining_production)
        remaining_production -= production_to_battery
        
        production_to_grid = remaining_production
        
        battery_to_consumption = min(battery_to_cons_pct * self.cur_capacity, 
                                    remaining_consumption, self.cur_capacity)
        remaining_consumption -= battery_to_consumption
        remaining_battery = self.cur_capacity - battery_to_consumption
        
        battery_to_grid = min(battery_to_grid_pct * remaining_battery, remaining_battery)
        
        max_battery_charge_remaining = self.battery_max_capacity - (
            self.cur_capacity + production_to_battery - battery_to_consumption - battery_to_grid
        )
        grid_to_battery = grid_to_battery_pct * max(0, max_battery_charge_remaining)
        grid_to_consumption = max(0, remaining_consumption)
        
        energy_sold = production_to_grid + battery_to_grid
        energy_bought = grid_to_consumption + grid_to_battery
        
        revenue = energy_sold * price * self.tariff
        cost = energy_bought * price
        step_profit = revenue - cost
        
        energy_added = production_to_battery + grid_to_battery
        energy_removed = battery_to_consumption + battery_to_grid
        
        if energy_added > 0:
            solar_opportunity_cost = production_to_battery * price * self.tariff
            grid_purchase_cost = grid_to_battery * price
            new_energy_cost = (solar_opportunity_cost + grid_purchase_cost) / energy_added
            
            if self.cur_capacity > 0:
                total_energy = self.cur_capacity + energy_added
                self.battery_cost_basis = (
                    self.battery_cost_basis * self.cur_capacity + 
                    new_energy_cost * energy_added
                ) / total_energy
            else:
                self.battery_cost_basis = new_energy_cost
        
        battery_net_change = production_to_battery + grid_to_battery - battery_to_consumption - battery_to_grid
        self.cur_capacity = np.clip(self.cur_capacity + battery_net_change, 0, self.battery_max_capacity)
        
        self.balance += step_profit
        
        self._update_time()
        self.current_step += 1
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        price_percentile = self._get_price_percentile(price)
        
        # === REWARD FUNCTION ===
        reward = 0.0
        
        # Component 1: Scaled base profit
        reward += step_profit * 10
        
        # Component 2: Arbitrage reward with continuous scaling
        if battery_to_grid > 0 and self.battery_cost_basis > 0:
            sell_price = price * self.tariff
            margin = (sell_price - self.battery_cost_basis) / (self.battery_cost_basis + 0.001)
            arbitrage_reward = margin * battery_to_grid * 5
            reward += arbitrage_reward
        
        # Component 3: Continuous price-aware charging reward
        if grid_to_battery > 0:
            # Reward buying at low prices, penalize buying at high prices
            charging_quality = (0.5 - price_percentile) * 2
            reward += charging_quality * grid_to_battery * 2
        
        # Component 4: Continuous price-aware selling reward
        if battery_to_grid > 0:
            # Reward selling at high prices, penalize selling at low prices
            selling_quality = (price_percentile - 0.5) * 2
            reward += selling_quality * battery_to_grid * 2
        
        # Component 5: Strategic battery holding reward
        # Reward holding battery when price is low (anticipate higher prices)
        if self.cur_capacity > 0 and price_percentile < 0.3:
            hold_reward = self.cur_capacity * 0.05 * (0.3 - price_percentile)
            reward += hold_reward
        
        # Component 6: Penalize holding full battery when price is high
        if self.cur_capacity > self.battery_max_capacity * 0.8 and price_percentile > 0.7:
            hold_penalty = -0.1 * (price_percentile - 0.7) * self.cur_capacity
            reward += hold_penalty
        
        # Component 7: Reward using battery instead of buying from grid
        if battery_to_consumption > 0:
            # Avoided cost
            avoided_cost = battery_to_consumption * price
            reward += avoided_cost * 0.5
        
        # Component 8: End of day evaluation
        if terminated:
            # Value remaining battery, penalize if it held too much
            battery_value = self.cur_capacity * self.avg_price * self.tariff
            
            # If battery is more than 50% full at end, missed selling opportunities
            if self.cur_capacity > self.battery_max_capacity * 0.5:
                missed_opportunity_penalty = (self.cur_capacity - self.battery_max_capacity * 0.5) * self.max_price_seen * self.tariff * 0.3
                reward += battery_value * 0.3 - missed_opportunity_penalty
            else:
                reward += battery_value * 0.5
            
            # Bonus for ending with positive balance
            if self.balance > 0:
                reward += self.balance * 2
        
        info = {
            "balance": self.balance,
            "battery_level": self.cur_capacity,
            "step_profit": step_profit,
            "revenue": revenue,
            "cost": cost,
            "price": price,
            "avg_price": self.avg_price,
            "price_percentile": price_percentile,
            "battery_cost_basis": self.battery_cost_basis,
            "energy_sold": energy_sold,
            "energy_bought": energy_bought,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
