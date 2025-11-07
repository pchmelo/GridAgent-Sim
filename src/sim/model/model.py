import os
from dotenv import load_dotenv

from mesa import Model
from mesa.datacollection import DataCollector

from sim.agent.agent_base import HEMSAgent

load_dotenv()
interval_str = os.getenv("INTERVAL", "1,0")
hour_interval, minute_interval = map(int, interval_str.split(","))
max_capacity = int(os.getenv("MAX_CAPACITY"))

def get_steps():
    return 24 * 60 // (hour_interval * 60 + minute_interval)

def update_time(current_time):
    hour, minute = current_time

    minute += minute_interval
    hour += hour_interval

    if minute >= 60:
        minute -= 60
        hour += 1
    if hour >= 24:
        hour -= 24 

    return (hour, minute)

class HEMSModel(Model):
    def __init__(self):
        super().__init__()

        # Initialize model parameters
        self.steps = get_steps()
        self.battery_capacity = max_capacity
        self.cur_capacity = 0
        self.cur_hour = (0, 0)

        # Initialize goal
        self.balance = 0.0

        # Create agent
        agent = HEMSAgent(self)
        self.agents.add(agent)

        # Data collector setup
        self.datacollector = DataCollector(
            model_reporters={
                "Balance": lambda m: m.balance,
                "Current_Capacity": lambda m: m.cur_capacity,
                "Current_Hour": lambda m: f"{m.cur_hour[0]:02}:{m.cur_hour[1]:02}",
            }
        )

    def step(self):
        self.cur_hour = update_time(self.cur_hour)
        self.agents.do("step")
        self.datacollector.collect(self)
