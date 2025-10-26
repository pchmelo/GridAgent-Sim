import os
from dotenv import load_dotenv

from mesa import Agent

load_dotenv()
agent_type = os.getenv("AGENT_TYPE", "smart")

class HEMSAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        m = self.model

        """
        # Make a decision based on the agent type
        if agent_type == "smart":
            new_balance, new_capacity = smart_decision(m.balance, m.cur_capacity, m.cur_hour)
        else:
            new_balance, new_capacity = baseline_decision(m.balance, m.cur_capacity, m.cur_hour)

        # Update model state based on decision
        m.balance = new_balance
        m.cur_capacity = new_capacity
        """
