import os
from dotenv import load_dotenv
from data.data_manager import data_manager

from mesa import Agent

from ..agent.baseline.baseline_agent import baseline_decision
from ..agent.smart.smart_agent import smart_decision

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
                actions, new_balance, new_capacity = smart_decision(m.balance, m.cur_capacity, m.cur_hour)
                if self.validate_actions(actions, m.cur_capacity, m.cur_hour, m.battery_max_capacity):
                    break
            else:
                actions, new_balance, new_capacity = baseline_decision(m.balance, m.cur_capacity, m.cur_hour)
                if self.validate_actions(actions, m.cur_capacity, m.cur_hour, m.battery_max_capacity):
                    break

        # Update model state based on decision
        m.balance = new_balance
        m.cur_capacity = new_capacity

    #TODO Implement Wind Production Configuration
    def validate_actions(self, actions: dict, cur_capacity, cur_hour, battery_max_capacity):
        res = True

        price, solar_production, wind_production, consumption = data_manager.get_model_data_entry(cur_hour)

        for label, actions in actions.items():
            if label == "consumption_action":
                acc = 0
                for action, value in actions.items():
                    acc += value
                
                if acc != consumption:
                    return False
                            

            elif label == "production_action":
                acc = 0
                for action, value in actions.items():
                    acc += value
                
                if acc != solar_production:
                    return False

            elif label == "battery_actions":
                acc = 0
                for action, value in actions.items():
                    if action in ["production_to_battery", "grid_to_battery"]:
                        acc += value
                    elif action in ["battery_to_consumption", "battery_to_grid"]:
                        acc -= value

                if acc + cur_capacity > battery_max_capacity or acc + cur_capacity < 0:
                    return False

        return res
        
        
