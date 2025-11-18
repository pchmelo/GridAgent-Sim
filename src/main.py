from datetime import datetime
import os
from dotenv import load_dotenv
from sim.data.json_result_manager import json_result_manager
from sim.model.model import HEMSModel
from sim.agent.smart.train import train_sac_agent

load_dotenv()

mode = os.getenv("MODE")
agent = os.getenv("AGENT_TYPE")

if __name__ == "__main__":
    if mode == "run_model":

        model = HEMSModel()

        for i in range(model.steps):
            model.step()

        results = model.datacollector.get_model_vars_dataframe()
        filename = f"{agent}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        results.to_csv(os.path.join("src", "log", "results", agent, filename))

    elif mode == "train":

        train_sac_agent(
            total_timesteps=200000,
            use_gpu=True,
            n_envs=4
        )

    else:
        print("Invalid MODE in .env file. Please set MODE to 'run_model' or 'train'.")
