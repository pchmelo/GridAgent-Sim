import os
from dotenv import load_dotenv
from mesa import Model
from mesa.datacollection import DataCollector
from sim.agent.agent_base import HEMSAgent

load_dotenv()

class HEMSModel(Model):

    def __init__(self, agent_type="smart"):
        super().__init__()
        self.agent_type = agent_type

        # Initialize model parameters
        self.battery_capacity = int(os.getenv("MAX_CAPACITY"))
        self.interval_str = os.getenv("INTERVAL", "1,0")
        self.hour_interval, self.minute_interval = map(int, self.interval_str.split(","))

        self.steps = self.get_steps()
        self.cur_capacity = 0.0
        self.actions = []
        self.price = 0.0
        self.solar_production = 0.0
        self.wind_production = 0.0
        self.consumption = 0.0
        self.old_capacity = 0.0
        self.cur_hour = (0, 0)

        # Initialize goal
        self.balance = 0.0

        # Create agent
        agent = HEMSAgent(self, agent_type)
        self.agents.add(agent)

        # Data collector setup
        self.datacollector = DataCollector(
            model_reporters={
                "Current_Hour": lambda m: f"{m.cur_hour[0]:02}:{m.cur_hour[1]:02}",
                "Solar_Production": lambda m: m.solar_production,
                "Wind_Production": lambda m: m.wind_production,
                "Consumption": lambda m: m.consumption,
                "Current_Capacity": lambda m: m.old_capacity,
                "Price": lambda m: m.price,
                "Actions": lambda m: m.format_actions(m.actions),
                "Balance": lambda m: m.balance,
                "New_Capacity": lambda m: m.cur_capacity
            }
        )

    def setup_configs(self, simulation_configs):
        self.simulation_configs = simulation_configs

        self.battery_capacity = simulation_configs.battery_max_capacity
        self.interval_str = simulation_configs.interval
        
        if simulation_configs.interval is None or simulation_configs.interval == 0:
            # Default to 1 hour intervals if None or 0 is provided
            self.hour_interval = 1
            self.minute_interval = 0
        elif simulation_configs.interval == 60:
            self.hour_interval = 1
            self.minute_interval = 0
        else:
            self.hour_interval = 0
            self.minute_interval = simulation_configs.interval
        
        # Recalculate steps with new intervals
        self.steps = self.get_steps()
            
        agent = HEMSAgent(self, self.agent_type, simulation_configs)
        self.agents.add(agent)


    def step(self):
        self.cur_hour = self.update_time(self.cur_hour)
        self.agents.do("step")
        self.datacollector.collect(self)

    def get_steps(self):
        return 24 * 60 // (self.hour_interval * 60 + self.minute_interval)

    def update_time(self, current_time):
        hour, minute = current_time

        minute += self.minute_interval
        hour += self.hour_interval

        if minute >= 60:
            minute -= 60
            hour += 1
        if hour >= 24:
            hour -= 24

        return (hour, minute)

    def format_actions(self, actions):
        """Format actions list into a string: 'key: value, key: value, ...'"""
        if not actions:
            return "No actions"
        
        action_strings = []
        for action_dict in actions:
            for key, value in action_dict.items():
                action_strings.append(f"{key}: {value:.4f}")
        
        return ", ".join(action_strings)
