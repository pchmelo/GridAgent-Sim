import copy
from sim.data.data_manager import data_manager


class BaselineModel:
    def __init__(self, battery_max_capacity=10, tariff = 0.75):
        self.battery_max_capacity = battery_max_capacity
        self.tariff = tariff

    def baseline_decision(self, balance, cur_capacity, cur_hour):
        self.balance = balance
        self.cur_capacity = cur_capacity
        
        self.price, self.solar_production, self.wind_production, self.consumption = data_manager.get_model_data_entry(cur_hour)
    

    #TODO Implement Wind Production Configuration
    def policy(self):
        self.actions = {
            "consumption_action": [],
            "production_action": [],
            "battery_actions": []
        }

        if self.solar_production >= self.consumption:
            current_production = self.solar_production - self.consumption

            self.consumption_action("energy_from_production", current_production)
            self.production_action("production_to_consumption", self.consumption)

            if current_production + self.cur_capacity >= self.battery_max_capacity:
                cur_capacity_t = copy.deepcopy(self.cur_capacity)
                self.cur_capacity += current_production
                self.battery_action("production_to_battery", current_production)
                
                current_production = current_production - (self.cur_capacity - cur_capacity_t)
                self.production_action("production_to_battery", self.cur_capacity - cur_capacity_t)

                self.balance += current_production * self.price * self.tariff
                self.production_action("production_to_grid", current_production)

            else:
                self.cur_capacity += current_production
                self.battery_action("production_to_battery", current_production)
                self.production_action("production_to_battery", current_production)

        else:
            current_consumption = self.consumption - self.solar_production
            self.consumption_action("energy_from_production", self.solar_production)
            self.production_action("production_to_consumption", self.solar_production)

            if current_consumption <= self.cur_capacity:
                self.cur_capacity -= current_consumption
                self.battery_action("battery_to_consumption", current_consumption)
                self.consumption_action("energy_from_battery", current_consumption)

            else:
                current_consumption = current_consumption - self.cur_capacity
                self.consumption_action("energy_from_battery", self.cur_capacity)

                self.cur_capacity = 0
                self.battery_action("battery_to_consumption", current_consumption)

                self.balance -= current_consumption * self.price * self.tariff
                self.consumption_action("energy_from_grid", current_consumption)
        
        return self.actions, self.balance, self.cur_capacity
    
    '''
    Label Available:
        consumption_action:
            - energy_from_production
            - energy_from_battery
            - energy_from_grid
        
        production_action:
            - production_to_consumption
            - production_to_battery
            - production_to_grid


        battery_action:
            - production_to_battery
            - grid_to_battery
            - battery_to_consumption
            - battery_to_grid
    '''

    def consumption_action(self, label, energy):
        self.actions["consumption_action"].append({label: energy})

    def production_action(self, label, energy):
        self.actions["production_action"].append({label: energy})

    def battery_action(self, label, energy):
        self.actions["battery_actions"].append({label: energy})

baseline_model = BaselineModel()