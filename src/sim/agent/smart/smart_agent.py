import os
from dotenv import load_dotenv
from stable_baselines3 import SAC
import numpy as np

from sim.data.data_manager import data_manager
from log.log_controller import log_controller

load_dotenv()
max_capacity = int(os.getenv("MAX_CAPACITY", "10"))
tariff = float(os.getenv("TARIFF", "0.75"))

class SmartAgent:
    
    log_type = "smart_input"
    
    def __init__(self, battery_max_capacity=max_capacity, tariff=tariff, model_path=None):
        self.battery_max_capacity = battery_max_capacity
        self.tariff = tariff

        # Normalization constants (MUST MATCH TRAINING ENVIRONMENT!!!)
        self.max_price = 0.2
        self.max_production = 5.0
        self.max_consumption = 3.0
        
        # Price thresholds (MUST MATCH TRAINING ENVIRONMENT!!!)
        self.price_high_threshold = 0.08
        self.price_low_threshold = 0.04

        # Load trained model
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "best_model.zip")
        
        if os.path.exists(model_path):
            self.model = SAC.load(model_path)
            print(f"Loaded SAC model from {model_path}")
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path}."
            )
    
    def smart_decision(self, balance, cur_capacity, cur_hour):
        self.balance = balance
        self.cur_capacity = cur_capacity
        
        price, solar_production, wind_production, consumption = data_manager.get_model_data_entry(time_stamp=cur_hour)
        
        self.price = price
        self.solar_production = solar_production
        self.wind_production = wind_production
        self.consumption = consumption
        
        log_controller.log_message(
            f"\nSmart Decision - Hour: {cur_hour}, Balance: {balance}, Current Capacity: {cur_capacity}",
            self.log_type
        )

        log_controller.log_message(
            f"Smart Data - Price: {price}, Solar Production: {solar_production}, "
            f"Wind Production: {wind_production}, Consumption: {consumption}", 
            self.log_type
        )
        
        # Prepare observation for the model
        obs = self.get_observation(cur_hour)
        
        # Get action from trained model
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to actual energy flows
        actions = self.convert_action_to_flows(action)
        
        log_controller.log_message(
            f"Smart Actions: {actions}, Balance: {self.balance}, Battery Capacity: {self.cur_capacity}", 
            self.log_type
        )
        
        return actions, self.balance, self.cur_capacity
    
    def get_observation(self, time_stamp):
        """Convert current state to observation vector (MUST MATCH TRAINING ENVIRONMENT!!!)"""
        hour, minute = time_stamp
        
        # Normalize values
        battery_normalized = self.cur_capacity / self.battery_max_capacity
        price_normalized = min(self.price / self.max_price, 1.0)
        solar_normalized = min(self.solar_production / self.max_production, 1.0)
        consumption_normalized = min(self.consumption / self.max_consumption, 1.0)
        
        # Cyclical time encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        minute_sin = np.sin(2 * np.pi * minute / 60)
        minute_cos = np.cos(2 * np.pi * minute / 60)
        
        # Price indicators
        price_high_signal = 1.0 if self.price > self.price_high_threshold else 0.0
        price_low_signal = 1.0 if self.price < self.price_low_threshold else 0.0
        
        return np.array([
            battery_normalized,
            price_normalized,
            solar_normalized,
            consumption_normalized,
            hour_sin,
            hour_cos,
            minute_sin,
            minute_cos,
            price_high_signal,
            price_low_signal,
        ], dtype=np.float32)
    
    def convert_action_to_flows(self, action):
        """Convert normalized action to actual energy flows (all 7 actions)"""
        actions = []
        
        # Tolerance for floating point operations
        TOLERANCE = 1e-6
        
        # Extract action percentages
        prod_to_cons_pct = float(action[0])
        prod_to_battery_pct = float(action[1])
        prod_to_grid_pct = float(action[2])
        battery_to_cons_pct = float(action[3])
        battery_to_grid_pct = float(action[4])
        grid_to_battery_pct = float(action[5])
        grid_to_cons_pct = float(action[6])
        
        # Phase 1: Production allocation
        # Production to consumption
        production_to_consumption = min(
            prod_to_cons_pct * self.consumption, 
            self.solar_production, 
            self.consumption
        )
        remaining_production = self.solar_production - production_to_consumption
        remaining_consumption = self.consumption - production_to_consumption
        
        # Production to battery
        max_battery_charge = self.battery_max_capacity - self.cur_capacity
        production_to_battery = min(
            prod_to_battery_pct * max_battery_charge, 
            remaining_production
        )
        remaining_production -= production_to_battery
        
        # Production to grid
        production_to_grid = min(
            prod_to_grid_pct * remaining_production, 
            remaining_production
        )
        
        # Phase 2: Battery allocation
        # Battery to consumption
        battery_to_consumption = min(
            battery_to_cons_pct * self.cur_capacity,
            remaining_consumption,
            self.cur_capacity
        )
        remaining_consumption -= battery_to_consumption
        remaining_battery = self.cur_capacity - battery_to_consumption
        
        # Battery to grid
        battery_to_grid = min(
            battery_to_grid_pct * remaining_battery, 
            remaining_battery
        )
        
        # Phase 3: Grid allocation
        # Grid to battery (buy cheap energy to store)
        max_battery_charge_remaining = self.battery_max_capacity - (
            self.cur_capacity + production_to_battery - battery_to_consumption - battery_to_grid
        )
        grid_to_battery = grid_to_battery_pct * max(0, max_battery_charge_remaining)
        
        # Grid to consumption (buy energy for immediate use) - ENSURE ALL CONSUMPTION IS MET
        grid_to_consumption = max(0, remaining_consumption)
        
        # Clamp to zero if very small (floating point errors)
        if abs(grid_to_consumption) < TOLERANCE:
            grid_to_consumption = 0
        if abs(production_to_consumption) < TOLERANCE:
            production_to_consumption = 0
        if abs(production_to_battery) < TOLERANCE:
            production_to_battery = 0
        if abs(production_to_grid) < TOLERANCE:
            production_to_grid = 0
        if abs(battery_to_consumption) < TOLERANCE:
            battery_to_consumption = 0
        if abs(battery_to_grid) < TOLERANCE:
            battery_to_grid = 0
        if abs(grid_to_battery) < TOLERANCE:
            grid_to_battery = 0
        
        # Update balance and capacity
        revenue_from_production = production_to_grid * self.price * self.tariff
        revenue_from_battery = battery_to_grid * self.price * self.tariff
        cost_for_consumption = grid_to_consumption * self.price
        cost_for_battery = grid_to_battery * self.price
        
        self.balance += revenue_from_production + revenue_from_battery
        self.balance -= cost_for_consumption + cost_for_battery
        
        battery_net_change = production_to_battery + grid_to_battery - battery_to_consumption - battery_to_grid
        self.cur_capacity += battery_net_change
        
        # Clamp battery capacity
        self.cur_capacity = max(0, min(self.cur_capacity, self.battery_max_capacity))
        
        # Build actions list (all 7 actions)
        if production_to_consumption > TOLERANCE:
            actions.append({"production_to_consumption": production_to_consumption})
        if production_to_battery > TOLERANCE:
            actions.append({"production_to_battery": production_to_battery})
        if production_to_grid > TOLERANCE:
            actions.append({"production_to_grid": production_to_grid})
        if battery_to_consumption > TOLERANCE:
            actions.append({"battery_to_consumption": battery_to_consumption})
        if battery_to_grid > TOLERANCE:
            actions.append({"battery_to_grid": battery_to_grid})
        if grid_to_battery > TOLERANCE:
            actions.append({"grid_to_battery": grid_to_battery})
        if grid_to_consumption > TOLERANCE:
            actions.append({"grid_to_consumption": grid_to_consumption})
        
        return actions

smart_agent = SmartAgent()