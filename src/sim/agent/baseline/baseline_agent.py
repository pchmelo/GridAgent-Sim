import os
from dotenv import load_dotenv

from sim.data.data_manager import data_manager
from log.log_controller import log_controller

load_dotenv()
max_capacity = os.getenv("MAX_CAPACITY")
tariff = os.getenv("TARIFF", 0.75)

class BaselineAgent:

    log_type = "baseline_input"

    def __init__(self, battery_max_capacity = int(max_capacity), tariff = float(tariff)):
        self.battery_max_capacity = battery_max_capacity
        self.tariff = tariff

    def baseline_decision(self, balance, cur_capacity, cur_hour):
        self.balance = balance
        self.cur_capacity = cur_capacity
        
        self.price, self.solar_production, self.wind_production, self.consumption = data_manager.get_model_data_entry(time_stamp=cur_hour)

        return self.policy()

    #TODO Implement Wind Production Configuration
    def policy(self):
        self.actions = []

        if self.solar_production >= self.consumption:
            current_production = self.solar_production - self.consumption

            self.actions.append({"production_to_consumption": self.consumption})

            if current_production + self.cur_capacity >= self.battery_max_capacity:
                current_production = current_production - (self.battery_max_capacity - self.cur_capacity)
                self.cur_capacity = self.battery_max_capacity

                self.actions.append({"production_to_battery": self.battery_max_capacity - self.cur_capacity})

                self.balance += current_production * self.price * self.tariff
                self.actions.append({"production_to_grid": current_production})

            else:
                self.cur_capacity += current_production
                self.actions.append({"production_to_battery": current_production})

        else:
            current_consumption = self.consumption - self.solar_production
            self.actions.append({"production_to_consumption": self.solar_production})

            if current_consumption <= self.cur_capacity:
                self.cur_capacity -= current_consumption
                self.actions.append({"battery_to_consumption": current_consumption})

            else:
                current_consumption = current_consumption - self.cur_capacity
                self.actions.append({"battery_to_consumption": self.cur_capacity})

                self.cur_capacity = 0

                self.balance -= current_consumption * self.price * self.tariff
                self.actions.append({"grid_to_consumption": current_consumption})


        log_controller.log_message(f"Baseline Actions: {self.actions}, Balance: {self.balance}, Battery Capacity: {self.cur_capacity}", self.log_type)

        return self.actions, self.balance, self.cur_capacity
    
    '''
    Label Available:   
        production_action:
            - production_to_consumption
            - production_to_battery
            - production_to_grid

        battery_action:
            - battery_to_consumption
            - battery_to_grid

        grid action:
            - grid_to_battery
            - grid_to_consumption
    '''

baseline_agent = BaselineAgent()