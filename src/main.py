import os
from dotenv import load_dotenv
from sim.data.json_result_manager import json_result_manager
from sim.agent.smart.train import train_sac_agent

load_dotenv()

mode = os.getenv("MODE")

if __name__ == "__main__":
    if mode == "run_model":
        from sim.model.model import HEMSModel

        """Run Smart Agent"""
        model = HEMSModel(agent_type="smart")

        for i in range(model.steps):
            model.step()

        results = model.datacollector.get_model_vars_dataframe()
        json_result_manager.save_to_json_file(results, agent_type="smart")

        """Run Basic Agent"""
        model = HEMSModel(agent_type="basic")

        for i in range(model.steps):
            model.step()

        results = model.datacollector.get_model_vars_dataframe()
        json_result_manager.save_to_json_file(results, agent_type="basic")

        json_result_manager.calculate_final_results()

    elif mode == "train":

        train_sac_agent(
            total_timesteps=1_000_000,
            use_gpu=True,
            n_envs=4,
            resume_from="src/sim/agent/smart/models/best_model.zip"
        )

    else:
        print("Invalid MODE in .env file. Please set MODE to 'run_model' or 'train'.")
