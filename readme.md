# GridAgent-Sim
The MS Household Energy Production project is a comprehensive simulation framework designed to evaluate Reinforcement Learning (RL) agents against baseline strategies for household energy management. By integrating a Mesa-based agent simulation with Stable-Baselines3 (SAC) and a Streamlit dashboard, the system provides a full pipeline from data ingestion to interactive visualization. Whether you are synthesizing realistic demand patterns through the REN API or orchestrating complex agent interactions within the HEMSModel, the project logic ensures a seamless flow of data from raw time-series inputs to the decision-making matrices of the smart agent.

## Features
- Simulation of home energy flows: solar, consumption and market prices.
- Smart agent using SAC (stable-baselines3) and a baseline agent for comparison.
- Web GUI to inspect simulation steps, metrics and plots.
- Export of results to CSV/JSON for analysis.

## GUI Overview

The interactive Streamlit-based GUI provides an intuitive interface for configuring, running, and analyzing energy management simulations.

### Home Screen

![Home Screen](docs/home.png)

*The main interface featuring the EMS-Sim title with options to start a new simulation or view previous results.*

### Simulation Parameters

![Simulation Parameters](docs/simulation_parameters.png)

*Configuration panel where users set interval time steps, maximum battery capacity, tariff rates, and choose between API integration or CSV upload for data input.*

### Data Verification

![API Data Graphs](docs/api_graphs.png)

*Interactive visualization of collected data showing consumption patterns, market prices, and solar production over 24 hours. Users verify data accuracy before running the simulation.*

### Smart Agent Actions

![Smart Agent Actions](docs/smart_actions.png)

*Real-time display of agent decisions showing input metrics (solar production, consumption, battery capacity, price) and output actions (balance, battery level) with a visual energy flow diagram.*

### Final Results Overview

![Final Results](docs/final_results.png)

*Comprehensive comparison between smart agent and baseline agent performance, including cumulative balance, battery usage, and detailed decision logs.*

### Results Analysis Plots

![Results Plots](docs/plot_results.png)

*Analytical charts comparing agent performance over time, displaying energy balance trends, battery state evolution, and cost efficiency metrics.*

## Requirements
Python 3.12 \
pip 

## Running with .venv
create virtual environment (if you prefer) \
pip install -r src/requirements.txt \
make run 
or
python -m streamlit run src/gui/gui.py

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
