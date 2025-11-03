import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from sim.data.data_manager import data_manager

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, s_, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            dones.append(d)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2

class SACAgent:
    def __init__(self, state_dim=5, action_dim=7, battery_max_capacity=10, tariff=0.75):
        self.battery_max_capacity = battery_max_capacity
        self.tariff = tariff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Temperature parameter
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 256
        
    def get_state(self, balance, cur_capacity, cur_hour):
        """Convert environment variables to state vector"""
        price, solar_production, wind_production, consumption = data_manager.get_model_data_entry(time_stamp=cur_hour)
        state = np.array([
            balance / 1000.0,  # Normalize balance
            cur_capacity / self.battery_max_capacity,  # Normalized capacity
            price,
            solar_production / 10.0,  # Normalize production
            consumption / 10.0  # Normalize consumption
        ], dtype=np.float32)
        return state
    
    def select_action(self, state, evaluate=False):
        """Select action using the actor network"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        
        return action.cpu().detach().numpy()[0]
    
    def action_to_energy_decisions(self, action, balance, cur_capacity, cur_hour):
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
        price, solar_production, wind_production, consumption = data_manager.get_model_data_entry(time_stamp=cur_hour)
        
        actions_dict = {
            "consumption_action": {},
            "production_action": {},
            "battery_actions": {}
        }
        
        # Normalize actions to [0, 1]
        normalized_actions = (action + 1) / 2
        
        # Production allocation
        prod_to_cons = min(normalized_actions[0] * solar_production, consumption)
        remaining_prod = solar_production - prod_to_cons
        
        available_battery_space = max(0, self.battery_max_capacity - cur_capacity)
        prod_to_battery = min(normalized_actions[1] * remaining_prod, available_battery_space)
        remaining_prod -= prod_to_battery
        
        prod_to_grid = remaining_prod
        
        # Consumption allocation
        remaining_cons = consumption - prod_to_cons
        available_battery = cur_capacity + prod_to_battery
        battery_to_cons = min(normalized_actions[3] * available_battery, remaining_cons)
        remaining_cons -= battery_to_cons
        
        grid_to_cons = max(0, remaining_cons)
        
        # Battery to grid
        current_battery = cur_capacity + prod_to_battery - battery_to_cons
        battery_to_grid = normalized_actions[4] * current_battery
        
        # Grid to battery
        current_battery_after_discharge = current_battery - battery_to_grid
        remaining_space = max(0, self.battery_max_capacity - current_battery_after_discharge)
        grid_to_battery = normalized_actions[6] * remaining_space
        
        # Update actions dict
        actions_dict["production_action"] = {
            "production_to_consumption": prod_to_cons,
            "production_to_battery": prod_to_battery,
            "production_to_grid": prod_to_grid
        }
        
        actions_dict["consumption_action"] = {
            "energy_from_production": prod_to_cons,
            "energy_from_battery": battery_to_cons,
            "energy_from_grid": grid_to_cons
        }
        
        actions_dict["battery_actions"] = {
            "production_to_battery": prod_to_battery,
            "grid_to_battery": grid_to_battery,
            "battery_to_consumption": battery_to_cons,
            "battery_to_grid": battery_to_grid
        }
        
        # Update balance and capacity
        battery_net_change = prod_to_battery + grid_to_battery - battery_to_cons - battery_to_grid
        new_capacity = np.clip(cur_capacity + battery_net_change, 0, self.battery_max_capacity)
        
        # Calculate balance
        revenue = (prod_to_grid + battery_to_grid) * price * self.tariff
        cost = (grid_to_cons + grid_to_battery) * price
        new_balance = balance + revenue - cost
        
        return actions_dict, new_balance, new_capacity
    
    def calculate_reward(self, balance, prev_balance, cur_capacity, consumption_from_grid):
        """Calculate reward for the agent"""
        # Reward components:
        # 1. Positive reward for increasing balance
        balance_reward = (balance - prev_balance) * 0.1
        
        # 2. Penalty for using grid energy
        grid_penalty = -consumption_from_grid * 0.5
        
        # 3. Bonus for maintaining battery level
        battery_reward = (cur_capacity / self.battery_max_capacity) * 0.2
        
        total_reward = balance_reward + grid_penalty + battery_reward
        return total_reward
    
    def train(self):
        """Train the SAC agent"""
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_target
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """Save model weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filename)
    
    def load(self, filename):
        """Load model weights"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

# Global agent instance
sac_agent = SACAgent()

def smart_decision(balance, cur_capacity, cur_hour):
    """Interface function for the smart agent"""
    state = sac_agent.get_state(balance, cur_capacity, cur_hour)
    action = sac_agent.select_action(state, evaluate=True)

    return sac_agent.action_to_energy_decisions(action, balance, cur_capacity, cur_hour)
