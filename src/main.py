from json import load

from sim import agent
from sim.model.model import HEMSModel

import os
from dotenv import load_dotenv

load_dotenv()
mode = os.getenv("MODE")
agent = os.getenv("AGENT")

if __name__ == "__main__":
    model = HEMSModel()

    for i in range(model.steps):
        model.step()