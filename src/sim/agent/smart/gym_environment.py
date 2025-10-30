import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sim.data.data_manager import data_manager

class HEMSEnvironment(gym.Env):
    """Custom Gymnasium Environment for Home Energy Management System"""
    
    metadata = {'render_modes': ['human'], 'render_fps': 1}
    
    def __init__(self, battery_max_capacity=10, tariff=0.75, max_hours=24):
        super(HEMSEnvironment, self).__init__()
        
        self.battery_max_capacity = battery_max_capacity
        self.tariff = tariff
        self.max_hours = max_hours
        
        # Define action space: 7 continuous actions in [-1, 1]
        # [prod_to_cons, prod_to_battery, prod_to_grid, 
        #  battery_to_cons, battery_to_grid, grid_to_cons, grid_to_battery]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )
        
        # Define observation space
        # [balance, battery_capacity, price, solar_production, consumption, hour_of_day]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, battery_max_capacity, np.inf, np.inf, np.inf, 23], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize state
        self.balance = 0
        self.cur_capacity = battery_max_capacity / 2
        self.cur_hour = 0
        self.steps_taken = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.balance = 0
        self.cur_capacity = self.battery_max_capacity / 2
        self.cur_hour = 0
        self.steps_taken = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Get current energy data
        price, solar_production, wind_production, consumption = data_manager.get_model_data_entry(self.cur_hour)
        
        # Store previous state for reward calculation
        prev_balance = self.balance
        prev_capacity = self.cur_capacity
        
        # Execute action and update state
        actions_dict, self.balance, self.cur_capacity = self._action_to_energy_decisions(
            action, price, solar_production, consumption
        )
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_balance, 
            prev_capacity,
            actions_dict,
            price
        )
        
        # Update time
        self.cur_hour = (self.cur_hour + 1) % 24
        self.steps_taken += 1
        
        # Check if episode is done
        terminated = False  # Could add failure conditions
        truncated = self.steps_taken >= self.max_hours
        
        observation = self._get_observation()
        info = self._get_info(actions_dict)
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation"""
        price, solar_production, wind_production, consumption = data_manager.get_model_data_entry(self.cur_hour)
        
        return np.array([
            self.balance / 1000.0,  # Normalize
            self.cur_capacity / self.battery_max_capacity,
            price,
            solar_production / 10.0,
            consumption / 10.0,
            self.cur_hour / 23.0  # Normalize hour
        ], dtype=np.float32)
    
    def _get_info(self, actions_dict=None):
        """Get additional info"""
        info = {
            'balance': self.balance,
            'battery_capacity': self.cur_capacity,
            'hour': self.cur_hour,
            'steps': self.steps_taken
        }
        if actions_dict:
            info['actions'] = actions_dict
        return info
    
    def _action_to_energy_decisions(self, action, price, solar_production, consumption):
        """Convert continuous actions to energy management decisions
        
        Action mapping (all in range [-1, 1], normalized to [0, 1] for proportions):
        action[0]: prod_to_cons
        action[1]: prod_to_battery
        action[2]: prod_to_grid
        action[3]: battery_to_cons
        action[4]: battery_to_grid
        action[5]: grid_to_cons
        action[6]: grid_to_battery
        """
        actions_dict = {
            "consumption_action": {},
            "production_action": {},
            "battery_actions": {}
        }
        
        # Normalize actions to [0, 1]
        normalized_actions = np.clip((action + 1) / 2, 0, 1)
        
        # Production allocation
        prod_to_cons = min(normalized_actions[0] * solar_production, consumption)
        remaining_prod = solar_production - prod_to_cons
        
        available_battery_space = max(0, self.battery_max_capacity - self.cur_capacity)
        prod_to_battery = min(normalized_actions[1] * remaining_prod, available_battery_space)
        remaining_prod -= prod_to_battery
        
        prod_to_grid = remaining_prod
        
        # Consumption allocation
        remaining_cons = consumption - prod_to_cons
        available_battery = self.cur_capacity + prod_to_battery
        battery_to_cons = min(normalized_actions[3] * available_battery, remaining_cons)
        remaining_cons -= battery_to_cons
        
        grid_to_cons = max(0, remaining_cons)
        
        # Battery to grid
        current_battery = self.cur_capacity + prod_to_battery - battery_to_cons
        battery_to_grid = normalized_actions[4] * current_battery
        
        # Grid to battery
        current_battery_after_discharge = current_battery - battery_to_grid
        remaining_space = max(0, self.battery_max_capacity - current_battery_after_discharge)
        grid_to_battery = normalized_actions[6] * remaining_space
        
        # Build actions dictionary
        actions_dict["production_action"] = {
            "production_to_consumption": float(prod_to_cons),
            "production_to_battery": float(prod_to_battery),
            "production_to_grid": float(prod_to_grid)
        }
        
        actions_dict["consumption_action"] = {
            "energy_from_production": float(prod_to_cons),
            "energy_from_battery": float(battery_to_cons),
            "energy_from_grid": float(grid_to_cons)
        }
        
        actions_dict["battery_actions"] = {
            "production_to_battery": float(prod_to_battery),
            "grid_to_battery": float(grid_to_battery),
            "battery_to_consumption": float(battery_to_cons),
            "battery_to_grid": float(battery_to_grid)
        }
        
        # Update state
        battery_net_change = prod_to_battery + grid_to_battery - battery_to_cons - battery_to_grid
        new_capacity = np.clip(self.cur_capacity + battery_net_change, 0, self.battery_max_capacity)
        
        # Calculate balance
        revenue = (prod_to_grid + battery_to_grid) * price * self.tariff
        cost = (grid_to_cons + grid_to_battery) * price
        new_balance = self.balance + revenue - cost
        
        return actions_dict, new_balance, new_capacity
    
    def _calculate_reward(self, prev_balance, prev_capacity, actions_dict, price):
        """Calculate reward for the current step"""
        # 1. Financial reward - profit increase
        balance_change = self.balance - prev_balance
        balance_reward = balance_change * 1.0
        
        # 2. Grid independence penalty
        grid_consumption = actions_dict["consumption_action"].get("energy_from_grid", 0)
        grid_penalty = -grid_consumption * price * 0.5
        
        # 3. Battery health reward - penalize extreme states
        battery_ratio = self.cur_capacity / self.battery_max_capacity
        if 0.2 <= battery_ratio <= 0.8:
            battery_reward = 0.1
        else:
            battery_reward = -0.1
        
        # 4. Self-consumption bonus
        self_consumption = actions_dict["consumption_action"].get("energy_from_production", 0)
        self_consumption_bonus = self_consumption * 0.05
        
        # Total reward
        total_reward = balance_reward + grid_penalty + battery_reward + self_consumption_bonus
        
        return float(total_reward)
    
    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == "human":
            print(f"Hour: {self.cur_hour}, Balance: ${self.balance:.2f}, "
                  f"Battery: {self.cur_capacity:.2f}/{self.battery_max_capacity} kWh")