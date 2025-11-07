import os
from dotenv import load_dotenv
from sim.data.data_manager import data_manager

from mesa import Agent

from sim.agent.baseline.baseline_agent import baseline_agent
from sim.agent.smart.smart_agent import smart_agent

from sim.data.data_manager import data_manager
from log.log_controller import log_controller

load_dotenv()
agent_type = os.getenv("AGENT_TYPE", "smart")

class HEMSAgent(Agent):

    log_type = "action_validation"

    def __init__(self, model):
        super().__init__(model)

    def step(self):
        m = self.model

        # Make a decision based on the agent type and validate it
        while True:
            if agent_type == "smart":
                actions, new_balance, new_capacity = smart_agent.smart_decision(m.balance, m.cur_capacity, m.cur_hour)
                if self.validate_actions(actions, m.cur_capacity, m.cur_hour, m.battery_capacity):
                    data_manager.update_time_stamp(m.cur_hour)
                    break
            elif agent_type == "basic":
                actions, new_balance, new_capacity = baseline_agent.baseline_decision(m.balance, m.cur_capacity, m.cur_hour)
                if self.validate_actions(actions, m.cur_capacity, m.cur_hour, m.battery_capacity):
                    data_manager.update_time_stamp(m.cur_hour)
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
        
        # Tolerance for floating point comparison
        TOLERANCE = 1e-3

        _, solar_production, wind_production, consumption = data_manager.get_model_data_entry(cur_hour)
        log_controller.log_message(f"Action Validation - Hour: {cur_hour}, Solar Production: {solar_production}, Wind Production: {wind_production}, Consumption: {consumption}", self.log_type)

        for action_dict in actions:
            for key, value in action_dict.items():
                if key == "production_to_consumption" or key == "production_to_battery" or key == "production_to_grid":
                    acc_production += value
                
                if key == "grid_to_consumption" or key == "production_to_consumption" or key == "battery_to_consumption":
                    acc_consumption += value
                
                if key == "grid_to_battery" or key == "production_to_battery":
                    acc_battery += value
                
                if key == "battery_to_consumption" or key == "battery_to_grid":
                    acc_battery -= value

        # Validate production doesn't exceed available
        if acc_production > solar_production + TOLERANCE:
            log_controller.log_message(f"Action Validation Failed - Production Exceeded: {acc_production} > {solar_production}", self.log_type)
            res = False

        # Validate consumption is met (with tolerance)
        if acc_consumption < consumption - TOLERANCE:
            log_controller.log_message(f"Action Validation Failed - Consumption Not Suppressed: {acc_consumption} < {consumption}", self.log_type)
            res = False

        # Validate battery constraints
        new_capacity = cur_capacity + acc_battery
        if new_capacity < -TOLERANCE:
            log_controller.log_message(f"Action Validation Failed - Battery Capacity Below Zero: {new_capacity} < 0", self.log_type)
            res = False
        
        if new_capacity > battery_max_capacity + TOLERANCE:
            log_controller.log_message(f"Action Validation Failed - Battery Capacity Exceeded: {new_capacity} > {battery_max_capacity}", self.log_type)
            res = False

        if res:
            log_controller.log_message(f"Action Validation Passed", self.log_type)

        return res
