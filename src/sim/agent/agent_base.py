import os
from dotenv import load_dotenv
from sim.data.data_manager import data_manager

from mesa import Agent

from sim.agent.baseline.baseline_agent import baseline_agent
#from sim.agent.smart.smart_agent import smart_decision

load_dotenv()
agent_type = os.getenv("AGENT_TYPE", "smart")

class HEMSAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        m = self.model

        # Make a decision based on the agent type and validate it
        while True:
            if agent_type == "smart":
                #actions, new_balance, new_capacity = smart_decision(m.balance, m.cur_capacity, m.cur_hour)
                if self.validate_actions(actions, m.cur_capacity, m.cur_hour, m.battery_capacity):
                    break
            elif agent_type == "basic":
                actions, new_balance, new_capacity = baseline_agent.baseline_decision(m.balance, m.cur_capacity, m.cur_hour)
                if self.validate_actions(actions, m.cur_capacity, m.cur_hour, m.battery_capacity):
                    break
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        # Update model state based on decision
        m.balance = new_balance
        m.cur_capacity = new_capacity

    #TODO Implement Wind Production Configuration
    def validate_actions(self, actions: dict, cur_capacity, cur_hour, battery_max_capacity):
        res = True
        acc_consumption, acc_production, acc_battery = 0, 0, 0

        _, solar_production, wind_production, consumption = data_manager.get_model_data_entry(cur_hour)

        for action_dict in actions:
            for key, value in action_dict.items():
                if key == "production_to_consumption" or key == "production_to_battery" or key == "production_to_grid":
                    acc_production += value
                elif key == "grid_to_consumption" or key == "production_to_consumption" or key == "battery_to_consumption":
                    acc_consumption += value
                elif key == "grid_to_battery" or key == "production_to_battery":
                    acc_battery += value
                elif key == "battery_to_consumption" or key == "battery_to_grid":
                    acc_battery -= value
        
        if acc_consumption < consumption:
            res = False
        if acc_production > solar_production:
            res = False
        if cur_capacity + acc_battery > battery_max_capacity or cur_capacity + acc_battery < 0:
            res = False

        return res
