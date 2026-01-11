# MS Household Energy Production
The MS Household Energy Production project is a comprehensive simulation framework designed to evaluate Reinforcement Learning (RL) agents against baseline strategies for household energy management. By integrating a Mesa-based agent simulation with Stable-Baselines3 (SAC) and a Streamlit dashboard, the system provides a full pipeline from data ingestion to interactive visualization. Whether you are synthesizing realistic demand patterns through the REN API or orchestrating complex agent interactions within the HEMSModel, the project logic ensures a seamless flow of data from raw time-series inputs to the decision-making matrices of the smart agent.

## Features
- Simulation of home energy flows: solar, consumption and market prices.
- Smart agent using SAC (stable-baselines3) and a baseline agent for comparison.
- Web GUI to inspect simulation steps, metrics and plots.
- Export of results to CSV/JSON for analysis.

## Requirements
Python 3.12 \
pip 

## Running with .venv
create virtual environment (if you prefer) \
pip install -r src/requirements.txt \
make run 

## Project Architecture
docs/: project documentation and design notes. \
src/ — main application code.

[MS/MS_Household_Energy_Production/src/main.py](http://_vscodecontentref_/5): runner; reads MODE (run_model, train, gui_mode) and launches simulations, training, or the Streamlit GUI. \
[MS/MS_Household_Energy_Production/src/requirements.txt](http://_vscodecontentref_/7): Python dependencies. \
src/gui/ — Streamlit UI and components. \
[MS/MS_Household_Energy_Production/src/gui/gui.py](http://_vscodecontentref_/8): Streamlit app entry (interactive controls, plots, and dashboards). \
src/gui/gui_components.py: UI helper components. \
src/sim/ — simulation core (model, environment, collectors). \
src/sim/model.py / HEMSModel: simulation loop and environment dynamics. \
src/sim/data/: data ingestion/result management (JSON exporters). \
src/sim/agent/: agent implementations and training assets (smart agent, SAC training, saved models). \
src/log/ — logging utilities and saved log files. \
src/plots/ — plotting helpers for analysis and GUI.
