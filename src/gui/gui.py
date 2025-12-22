import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
import time
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json

from sim.data.data_manager import data_manager
#from sim.simulation_manager import SimulationManager

st.set_page_config(layout="wide")

def create_log(log : str):
    return f'<p class="run">{log}</p>'

@st.dialog("Confirm Data Collection",width='large')
def confirm_collection_modal(date_str):
    data_dir = os.path.join(r"sim/data/datafiles", date_str)

    # 2. Check if directory exists
    if os.path.exists(data_dir):
        datasets = os.listdir(data_dir)

        if datasets:
            # 2. Create ALL tabs at once (Streamlit requirement)
            tabs = st.tabs(datasets)

        for tab, filename in zip(tabs, datasets):
            with tab:
                file_path = os.path.join(data_dir, filename)
                df = pd.read_csv(file_path)
                
                st.caption(f"Source: '{filename}'")

                df["Time (Hour)"] = pd.to_datetime(df["Time (Hour)"])

                if filename == "consumption.csv":
                    st.line_chart(df,x="Time (Hour)",y="Consumption (kW)")

                elif filename == "market_prices.csv":
                    st.line_chart(df,x="Time (Hour)",y="Price (€/kWh)")

                elif filename == "wind_production.csv" or filename == "solar_production.csv":
                    st.line_chart(df,x="Time (Hour)",y="Production (kW)")  
    
    st.write(f"Is this data correct?")

    col1, col2, _ = st.columns([1, 1, 8])
    
    with col1:
        if st.button("Yes", type="primary",use_container_width=True):
            st.session_state.show_calendar = False
            st.session_state.selected_date = date_str
            st.session_state.data_confirmed = True
            st.rerun()

    with col2:
        if st.button("No",use_container_width=True):
            st.session_state.show_calendar = False
            st.session_state.data_confirmed = True
            st.rerun()

@st.dialog("Simulation Overview",width='large')
def display_simulation_overview(json_path : str = None):
    tabs = st.tabs(["Smart","Basic"])

    with open(json_path, "r") as file:
        json_data = json.loads(file.read())

    rows = []
    for agent_type in ['smart', 'basic']:
        agent_data = json_data.get(agent_type, {})
        
        for timestamp, details in agent_data.items():
            row = {
                'Agent_Type': agent_type,
                'Time': timestamp
            }
            
            if 'input_data' in details:
                row.update(details['input_data'])

            if 'output_data' in details:
                row.update(details['output_data'])
                
            if 'Actions' in details:
                for action in details['Actions']:
                    row.update(action)
                    
            rows.append(row)

    # Create the main DataFrame
    df_simulation = pd.DataFrame(rows)
    df_simulation = df_simulation.rename(columns={"New_Capacity" : "Battery"})
    df_simulation["Time"] = pd.to_datetime(df_simulation["Time"])

    all_columns = [
    "Solar_Production", 
    "Consumption", 
    "Battery",
    "Price",
    "Balance"]

    with tabs[0]:
        df_smart = df_simulation[df_simulation["Agent_Type"] == "smart"]
        st.caption("Smart Agent Simulation Overview")

        scale = st.selectbox( "Choose a scale for the polts!",
                ("Default", "Logarithmic"))
        selected_cols = st.multiselect(
                "Select metrics to display", 
                options=all_columns, 
                default=["Solar_Production"] 
            )
        
        if scale == "Default":
            # 3. Display the chart
            if selected_cols:
                st.line_chart(df_smart, x="Time", y=selected_cols)
            else:
                st.info("Select one or more metrics from the dropdown above to start visualizing.")
        elif scale == "Logarithmic":
            df_smart_log = df_smart.copy()
            for col in all_columns:
                df_smart_log[col] = df_smart_log[col].apply(lambda x: np.log(x) if x > 0 else 0)

            # 3. Display the chart
            if selected_cols:
                st.line_chart(df_smart_log, x="Time", y=selected_cols)
            else:
                st.info("Select one or more metrics from the dropdown above to start visualizing.")
    
    with tabs[1]:
        df_basic = df_simulation[df_simulation["Agent_Type"] == "basic"]
        st.caption("Basic Agent Simulation Overview")
        selected_cols = st.multiselect(
            "Select metrics to display", 
            options=all_columns, 
            default=["Solar_Production"],key="SC1"
        )

        # 3. Display the chart
        if selected_cols:
            st.line_chart(df_basic, x="Time", y=selected_cols)
        else:
            st.info("Select one or more metrics from the dropdown above to start visualizing.")

def display_input_metrics(input_data,timestamp):
   # 1. Big Timestamp Header
    st.markdown(f"## 🕒 Time: `{timestamp}`")
    
    meta = {
        "Solar_Production": {"icon": "☀️", "unit": "kW"},
        "Consumption": {"icon": "🏠", "unit": "kW"},
        "Current_Capacity": {"icon": "🔋", "unit": "kWh"},
        "Price": {"icon": "💰", "unit": "/kWh"}
    }
    
    # 2. Loop through and display large text
    for key, value in input_data.items():
        if key != "Wind_Production":
            m = meta.get(key, {"icon": "📊", "unit": ""})
            label = key.replace('_', ' ')
            
            # We use a div with inline CSS for custom sizing
            # Adjust '24px' to '30px' if you want it even larger
            st.markdown(
                f"""
                <div style="font-size: 24px; font-family: sans-serif; margin-bottom: 10px;">
                    {m['icon']} <b>{label}</b>: 
                    <span style="color: #60A5FA; font-family: monospace;">{value:.4f}</span> {m['unit']}
                </div>
                """, 
                unsafe_allow_html=True
            )

def display_output_metrics(output_data):
    # 2. Mapping keys to Emojis and Units
    meta = {
        "Balance": {"icon": "⚖️", "unit": "€"},
        "New_Capacity": {"icon": "🔋", "unit": "kWh"}
    }
    
    # 3. Loop through and display large text
    for key, value in output_data.items():
        m = meta.get(key, {"icon": "📝", "unit": ""})
        label = key.replace('_', ' ')
        
        # Color logic: Red if balance is negative, Green if positive
        val_color = "#FF4B4B" if key == "Balance" and value < 0 else "#60A5FA"
        if key == "Balance" and value >= 0: val_color = "#00FF00"

        st.markdown(
            f"""
            <div style="font-size: 24px; font-family: sans-serif; margin-bottom: 10px;">
                {m['icon']} <b>{label}</b>: 
                <span style="color: {val_color}; font-family: monospace;">{value:.4f}</span> {m['unit']}
            </div>
            """, 
            unsafe_allow_html=True
        )

def draw_ems_map(actions_list):
    def get_val(key):
        for action in actions_list:
            if key in action:
                return action[key]
        return 0.0

     # 1. Extract values
    b2c = get_val("battery_to_consumption")
    b2g = get_val("battery_to_grid")
    g2b = get_val("grid_to_battery")
    g2c = get_val("grid_to_consumption")
    p2c = get_val("production_to_consumption") # New
    p2b = get_val("production_to_battery")     # New

    # 2. Determine visibility
    b2c_style = "display: block;" if b2c > 0 else "display: none;"
    b2g_style = "display: block;" if b2g > 0 else "display: none;"
    g2b_style = "display: block;" if g2b > 0 else "display: none;"
    g2c_style = "display: block;" if g2c > 0 else "display: none;"
    p2c_style = "display: block;" if p2c > 0 else "display: none;"
    p2b_style = "display: block;" if p2b > 0 else "display: none;"

    svg_code = f"""
    <div style="display: flex; justify-content: center; background-color: #0e1117; border-radius: 15px; padding: 10px;">
        <svg viewBox="0 0 600 350" width="600" height="350" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
                    <path d="M0,0 L10,5 L0,10 Z" fill="#555" />
                </marker>
                <style>
                    .flow-line {{ stroke-dasharray: 8; animation: dash 1s linear infinite; }}
                    @keyframes dash {{ from {{ stroke-dashoffset: 16; }} to {{ stroke-dashoffset: 0; }} }}
                    .label {{ fill: #888; font-size: 12px; font-family: sans-serif; }}
                    .val {{ font-size: 14px; font-weight: bold; font-family: sans-serif; }}
                    .node-text {{ font-size: 40px; }}
                </style>
            </defs>

            <text x="100" y="50" class="node-text" text-anchor="middle">☀️</text>
            <text x="100" y="80" class="label" text-anchor="middle">Solar</text>

            <text x="500" y="50" class="node-text" text-anchor="middle">🏠</text>
            <text x="500" y="80" class="label" text-anchor="middle">Consumption</text>
            
            <text x="100" y="270" class="node-text" text-anchor="middle">🔋</text>
            <text x="100" y="300" class="label" text-anchor="middle">Battery</text>
            
            <text x="500" y="270" class="node-text" text-anchor="middle">⚡</text>
            <text x="500" y="300" class="label" text-anchor="middle">Grid</text>

            <g style="{p2c_style}">
                <path d="M140,40 L460,40" stroke="#fbbf24" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="300" y="30" fill="#fbbf24" class="val" text-anchor="middle">{p2c:.2f} kW</text>
            </g>

            <g style="{p2b_style}">
                <path d="M100,90 L100,220" stroke="#fbbf24" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="90" y="160" fill="#fbbf24" class="val" text-anchor="middle" transform="rotate(-90, 90, 160)">{p2b:.2f} kW</text>
            </g>

            <g style="{b2c_style}">
                <path d="M140,240 L460,70" stroke="#00FF00" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="250" y="150" fill="#00FF00" class="val" transform="rotate(-28, 250, 150)">{b2c:.2f} kW</text>
            </g>

            <g style="{g2c_style}">
                <path d="M500,220 L500,90" stroke="#FACC15" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="515" y="160" fill="#FACC15" class="val" text-anchor="middle" transform="rotate(90, 515, 160)">{g2c:.2f} kW</text>
            </g>

            <g style="{b2g_style}">
                <path d="M140,265 L460,265" stroke="#FF4B4B" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="300" y="255" fill="#FF4B4B" class="val" text-anchor="middle">{b2g:.2f} kW</text>
            </g>
            
            <g style="{g2b_style}">
                <path d="M460,285 L140,285" stroke="#60A5FA" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="300" y="310" fill="#60A5FA" class="val" text-anchor="middle">{g2b:.2f} kW</text>
            </g>
        </svg>
    </div>
    """
    return svg_code

@st.fragment
def ems_monitor_smart(smart_data):
    smart_keys = [i for i in smart_data.keys()]
    col1, col2, col3 = st.columns([1, 2, 1])
    ems_map_window = col2.empty()

    with col2:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c3:
            forward_click = st.button("Fordward Step", use_container_width=True)
        with c1:
            backwards_click = st.button("Backward Step", use_container_width=True)
        with c2:
            run_simulation = st.button("Run Simulation",type="primary", use_container_width=True)
    
    if forward_click:
        st.session_state.index += 1
        st.session_state.index %= len(smart_keys)
    
    if backwards_click:
        st.session_state.index -= 1
        st.session_state.index %= len(smart_keys)

    input_data = smart_data[smart_keys[st.session_state.index]]["input_data"]
    actions = smart_data[smart_keys[st.session_state.index]]["Actions"]
    output_data = smart_data[smart_keys[st.session_state.index]]["output_data"]

    with col1:
        st.title("Input data")
        display_input_metrics(input_data,smart_keys[st.session_state.index])

    with ems_map_window:
        st.title("Actions")
        components.html(draw_ems_map(actions) , height=330)
    
    with col3:
        st.title("Output Data")
        display_output_metrics(output_data)
        if st.session_state.simulation_run:
            overview_button = st.button("Simulation Overview")

            if overview_button:
                if result_files:
                    display_simulation_overview(str(result_files[0]))
    
    if run_simulation:
        config = {
        "selected_date": st.session_state.get("selected_date"),
        "interval": st.session_state.get("interval"),
        "max_capacity": st.session_state.get("max_capacity"),
        "tariff": st.session_state.get("tariff"),
        "complex_mode": st.session_state.get("complex_mode"),
        "used_api":True if st.session_state.get("selected_date") else False
        }

        json_data = json.dumps(config, indent=4)

        #sim_manager = SimulationManager()

        print(10*"-")
        print(config)
        print(10*"-")

        """json_res = sim_manager.start_simulation(config=config,
                                        df_solar_production=st.session_state.solar_data,
                                        df_wind_production=st.session_state.wind_data,
                                        df_consumption=st.session_state.consumption_data,
                                        df_price=st.session_state.market_data)"""
        
        print(10*"-")
        #print(json_res)
        print(10*"-")
        
        st.session_state.simulation_run = True

        st.rerun()

        for i in range(5):
            st.session_state.logs.append(create_log(f"Step {i+1} complete — {datetime.now().strftime("%H:%M:%S")}"))
            logs_html = "".join(st.session_state.logs)

            # rebuild entire HTML each time (nonce changes to force rerender)
            terminal_placeholder.html(
                f"""
        {terminal_style}
        <!-- Hidden toggle input -->
        <input type="checkbox" id="toggleTerminal">

        <!-- Terminal container -->
            <!-- nonce: {time.time()} -->
        <div class="terminal-container" id="terminalContainer">
            <div class="terminal-header">
            <!-- Label controls the checkbox -->
            <label for="toggleTerminal" class="terminal-toggle"></label>
            <span>Logs</span>
            <div></div>
            </div>
            <div class="terminal-body" id="terminalBody">
            {logs_html}
            </div>
        </div>
        """
            )

            time.sleep(0.5)

@st.fragment
def ems_monitor_basic(basic_data):
    basic_keys = [i for i in basic_data.keys()]
    col1, col2, col3 = st.columns([1, 2, 1])
    ems_map_window = col2.empty()

    with col2:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c3:
            forward_click_1 = st.button("Fordward Step",key="FS1",use_container_width=True)
        with c1:
            backwards_click_1 = st.button("Backward Step",key="BS1", use_container_width=True)
        with c2:
            run_simulation_1 = st.button("Run Simulation",type="primary",key="RS1", use_container_width=True)
    
    if forward_click_1:
        st.session_state.index_basic += 1
        st.session_state.index_basic %= len(basic_keys)
    
    if backwards_click_1:
        st.session_state.index_basic -= 1
        st.session_state.index_basic %= len(basic_keys)

    input_data = basic_data[basic_keys[st.session_state.index_basic]]["input_data"]
    actions = basic_data[basic_keys[st.session_state.index_basic]]["Actions"]
    output_data = basic_data[basic_keys[st.session_state.index_basic]]["output_data"]

    with col1:
        st.title("Input data")
        display_input_metrics(input_data,basic_keys[st.session_state.index_basic])

    with ems_map_window:
        st.title("Actions")
        components.html(draw_ems_map(actions) , height=330)
    
    with col3:
        st.title("Output Data")
        display_output_metrics(output_data)
        if st.session_state.simulation_run:
            overview_button = st.button("Simulation Overview",key="OV1")

            if overview_button:
                if result_files:
                    display_simulation_overview(str(result_files[0]))
    
    if run_simulation_1:
        config = {
        "selected_date": st.session_state.get("selected_date"),
        "interval": st.session_state.get("interval"),
        "max_capacity": st.session_state.get("max_capacity"),
        "tariff": st.session_state.get("tariff"),
        "complex_mode": st.session_state.get("complex_mode"),
        "used_api":True if st.session_state.get("selected_date") else False
        }

        json_data = json.dumps(config, indent=4)

        #sim_manager = SimulationManager()

        print(10*"-")
        print(config)
        print(10*"-")

        """json_res = sim_manager.start_simulation(config=config,
                                        df_solar_production=st.session_state.solar_data,
                                        df_wind_production=st.session_state.wind_data,
                                        df_consumption=st.session_state.consumption_data,
                                        df_price=st.session_state.market_data)"""
        
        print(10*"-")
        #print(json_res)
        print(10*"-")
        
        st.session_state.simulation_run = True

        st.rerun()

        for i in range(5):
            st.session_state.logs.append(create_log(f"Step {i+1} complete — {datetime.now().strftime("%H:%M:%S")}"))
            logs_html = "".join(st.session_state.logs)

            # rebuild entire HTML each time (nonce changes to force rerender)
            terminal_placeholder.html(
                f"""
        {terminal_style}
        <!-- Hidden toggle input -->
        <input type="checkbox" id="toggleTerminal">

        <!-- Terminal container -->
            <!-- nonce: {time.time()} -->
        <div class="terminal-container" id="terminalContainer">
            <div class="terminal-header">
            <!-- Label controls the checkbox -->
            <label for="toggleTerminal" class="terminal-toggle"></label>
            <span>Logs</span>
            <div></div>
            </div>
            <div class="terminal-body" id="terminalBody">
            {logs_html}
            </div>
        </div>
        """
            )

            time.sleep(0.5)

if "show_calendar" not in st.session_state:
    st.session_state.show_calendar = False

if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False

if "data_confirmed" not in st.session_state:
    st.session_state.data_confirmed = False

if "inserting_data" not in st.session_state:
    st.session_state.inserting_data = False

if "selected_date" not in st.session_state:
    st.session_state.selected_date = None

if "interval" not in st.session_state:
    st.session_state.interval = None

if "max_capacity" not in st.session_state:
    st.session_state.max_capacity = None

if "tariff" not in st.session_state:
    st.session_state.tariff = None

if "complex_mode" not in st.session_state:
    st.session_state.complex_mode = False

if "logs" not in st.session_state:
    st.session_state.logs = [create_log(str(i)) for i in range(5)]

if "consumption_data" not in st.session_state:
    st.session_state.consumption_data = None

if "market_data" not in st.session_state:
    st.session_state.market_data = None

if "solar_data" not in st.session_state:
    st.session_state.solar_data = None

if "wind_data" not in st.session_state:
    st.session_state.wind_data = None

if "index" not in st.session_state:
    st.session_state.index = 0

if "index_basic" not in st.session_state:
    st.session_state.index_basic = 0

if "svg_code" not in st.session_state:
    st.session_state.svg_code = "<div></div>"

results_dir = Path(__file__).parent.parent / "sim" / "data" / "results" / "final_results"
result_files = sorted(results_dir.glob("final_results_*.json"), reverse=True)

if result_files:
    with open(result_files[0], "r") as file:
        json_data = json.loads(file.read())
else:
    json_data = {"smart": {}, "basic": {}}

smart_data = json_data["smart"] if "smart" in json_data.keys() else None
basic_data = json_data["basic"] if "basic" in json_data.keys() else None

with st.sidebar:
    # AGENT_TYPE toggle
    agent_type = st.toggle("Toggle complex mode", value=True)
    # INTERVAL slider
    interval = st.slider("INTERVAL (minutes)", min_value=0, max_value=60, value=0, step=15)
    # MAX_CAPACITY
    max_capacity = st.number_input("Max Battery Capacity (kWh)", min_value=0, max_value=1000,value=10)
    # TARIFF
    tariff = st.number_input("TARIFF", min_value=0.0, max_value=1.0,value=0.75, step=0.01,
                             help="The second parameter is the relationship between import and export tariffs. For example, if tariff = 75%, this means that the selling price is 75% of the purchase price")

    cole, colf,_ = st.columns([5,5,8])
    with cole:
        confirm_config = st.button("Confirm",type="primary",use_container_width=True)
        insert_csv = st.button("Insert CSV",use_container_width=True)

        if st.session_state.consumption_data != None and st.session_state.market_data != None and st.session_state.solar_data != None and st.session_state.wind_data != None:
            remove_csv = st.button("Remove CSV",use_container_width=True)
        
            if remove_csv:
                st.session_state.consumption_data = None
                st.session_state.market_data = None
                st.session_state.solar_data = None
                st.session_state.wind_data = None
                st.rerun()

    with colf:
        cancel_config = st.button("Cancel",use_container_width=True,key="cancel_config")
        use_api = st.button("Use api",use_container_width=True)

        if st.session_state.selected_date != None:
            remove_api = st.button("Remove API data",use_container_width=True)

            if remove_api:
                st.session_state.selected_date = None
                st.rerun()

    if cancel_config:
        st.session_state.inserting_data = False
        st.rerun()

    if confirm_config:
        st.session_state.inserting_data = False
        st.session_state.complex_mode = agent_type
        st.session_state.interval = interval
        st.session_state.max_capacity = max_capacity
        st.session_state.tariff = tariff
        st.rerun()

    if insert_csv:
        st.session_state.show_calendar = False

        uploaded_file_1 = st.file_uploader("Insert consumption data", type=["csv"], key="consumption_uploader")
        if uploaded_file_1:
            df = pd.read_csv(uploaded_file_1)
            st.session_state.consumption_data = df
            st.success("File uploaded successfully!")
            st.rerun()
        
        uploaded_file_2 = st.file_uploader("Insert market price data", type=["csv"], key="market_uploader")
        if uploaded_file_2:
            df = pd.read_csv(uploaded_file_2)
            st.session_state.market_data = df
            st.success("File uploaded successfully!")
            st.rerun()
        
        uploaded_file_3 = st.file_uploader("Insert solar production data", type=["csv"], key="solar_uploader")
        if uploaded_file_3:
            df = pd.read_csv(uploaded_file_3)
            st.session_state.solar_data = df
            st.success("File uploaded successfully!")
            st.rerun()
        
        uploaded_file_4 = st.file_uploader("Insert wind production data", type=["csv"], key="wind_uploader")
        if uploaded_file_4:
            df = pd.read_csv(uploaded_file_4)
            st.session_state.wind_data = df
            st.success("File uploaded successfully!")
            st.rerun()

    if use_api:
        st.session_state.show_calendar = True

    calendar_placeholder = st.empty()

    if st.session_state.show_calendar:
        #Calender to select date for REN API
        with calendar_placeholder.container():
            default_date = datetime.now() - timedelta(days=1)
            default_date =  default_date.strftime("%Y-%m-%d")

            col3, _ = st.columns([2,2])

            with col3:
                selected_date = st.date_input("Select a date", value=default_date,max_value=default_date,key="api_date").strftime("%Y-%m-%d")

            col1, col2 = st.columns([1, 1])

            with col1:
                confirm_button = st.button("Confirm date",type="primary",use_container_width=True)
            
            with col2:
                cancel_button = st.button("Cancel",use_container_width=True)

            if cancel_button:
                calendar_placeholder.empty()

            if confirm_button:
                if data_manager.start_data_collection(selected_date):
                    confirm_collection_modal(selected_date)

if st.session_state.data_confirmed:
    st.session_state.data_confirmed = False
    calendar_placeholder.empty()

terminal_style = """
<style>
/* Hidden checkbox acts as the toggle state */
#toggleTerminal {
    display: none;
}

/* Terminal container */
.terminal-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 32px;
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: "Consolas", monospace;
    border-top: 1px solid #333;
    display: flex;
    flex-direction: column;
    transition: height 0.3s ease;
    z-index: 9999;
    overflow: hidden;
}

/* When checkbox is NOT checked → expanded */
#toggleTerminal:not(:checked) ~ .terminal-container {
    height: 32px;
}

/* When checkbox IS checked → minimized */
#toggleTerminal:checked ~ .terminal-container {
    height: 220px;
}

/* Terminal header */
.terminal-header {
    background: #2d2d2d;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.3rem 0.8rem;
    color: #ccc;
    font-size: 0.9rem;
    user-select: none;
}

/* The clickable toggle button */
.terminal-toggle {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    background: #ffbd2e;
    cursor: pointer;
    transition: background 0.2s ease;
}

/* Change color when minimized */
#toggleTerminal:checked ~ .terminal-container .terminal-toggle {
    background: #27c93f;
}

.terminal-body {
    flex: 1;
    overflow-y: auto;
    padding: 0.8rem;
    font-size: 0.9rem;
    line-height: 1.4;
}

.terminal-body p { margin: 0; white-space: pre-wrap; }
.terminal-body p.info { color: #9cdcfe; }
.terminal-body p.action { color: #ce9178; }
.terminal-body p.run { color: #b5cea8; }
</style>

"""

terminal_placeholder = st.empty()

logs_html = "".join(st.session_state.logs)

terminal_placeholder.html(
    f"""
    {terminal_style}
    <!-- Hidden toggle input -->
    <input type="checkbox" id="toggleTerminal">

    <!-- Terminal container -->
     <!-- nonce: {time.time()} -->
    <div class="terminal-container" id="terminalContainer">
      <div class="terminal-header">
        <!-- Label controls the checkbox -->
        <label for="toggleTerminal" class="terminal-toggle"></label>
        <span>Logs</span>
        <div></div>
      </div>
      <div class="terminal-body" id="terminalBody">
        {logs_html}
      </div>
    </div>
    """)

if result_files:
    result_tabs = st.tabs(["Smart","Basic","Final Results"])
    with result_tabs[0]:
        ems_monitor_smart(smart_data)

    with result_tabs[1]:
        ems_monitor_basic(basic_data)

    with result_tabs[2]:
        final_results = json_data["final_results"]

        # 2. Format Data for Table
        comparison_data = {
            "Metric": ["Balance (€)", "Consumption Savings (kW)"],
            "Smart Agent": [
                f"{final_results['smart_agent_balance']:.2f}", 
                f"{final_results['smart_agent_consumption_saving']:.2f}"
            ],
            "Basic Agent": [
                f"{final_results['basic_agent_balance']:.2f}", 
                f"{final_results['basic_agent_consumption_saving']:.2f}"
            ]
        }

        df = pd.DataFrame(comparison_data)

        # 3. Display Header and Table
        st.header("🏆 Smart vs. Basic Agent Performance")

        # Styling the table to look professional
        st.table(df)

        # 4. Impact Highlight
        st.metric(
            label="Balance Difference (Smart - Basic)", 
            value=f"{final_results['agent_balance_difference']:.2f} €",
            delta_color="inverse",
            help=f"""
            **Value:** `{final_results['agent_balance_difference']:.2f}`
            - **What it is:** The direct comparison between the two agents. 
            - A **negative** value indicates the Basic Agent outperformed the Smart Agent in this specific run.
            """
        )

        # 4. Impact Highlight
        st.metric(
            label="Total Consumption Cost", 
            value=f"{final_results['total_consumption_cust']:.2f} €",
            delta_color="inverse",
            help=f"""
            **Value:** `{final_results['total_consumption_cust']:.2f}`
            - **What it is:** The theoretical total cost if the customer had no Solar/Battery system at all. 
            - This serves as the 'worst-case scenario' baseline to measure how much both agents improved the situation.
            """
        )

        st.divider()

        # 5. Explanatory Tips (The "What does this mean?" section)
        st.subheader("💡 Metric Explanations")

        with st.expander("🔍 Balance"):
            st.write("""
            **What it is:** The net profit or loss generated by the agent during the simulation.
            - **Smart Agent Balance:** Total revenue from selling to the grid minus costs of buying from the grid and battery usage.
            - **Basic Agent Balance:** The benchmark performance, usually representing a standard 'dumb' strategy (e.g., just covering consumption).
            """)

        with st.expander("📊 Consumption Saving"):
            st.write("""
            **What it is:** The amount of energy (kW) the agent successfully diverted from the grid by using Solar or Battery power.
            - **Higher is better:** It means the agent is maximizing self-sufficiency and reducing the home's reliance on external energy.
            """)
else:
    st.warning("No result files available, run simulation first!")
    default_data = {"00:00":{
            "input_data": {
                "Solar_Production": 0.0,
                "Wind_Production": 0.0,
                "Consumption": 0.0,
                "Current_Capacity": 0.0,
                "Price": 0.0
            },
            "Actions": [
                {
                    "grid_to_battery": 0
                },
                {
                    "grid_to_consumption": 0
                }
            ],
            "output_data": {
                "Balance": 0,
                "New_Capacity": 0
            }
        }
    }

    default_final_results = {
        "smart_agent_balance": 0,
        "basic_agent_balance": 0,
        "agent_balance_difference": 0,
        "total_consumption_cust": 0,
        "basic_agent_consumption_saving": 0,
        "smart_agent_consumption_saving": 0
    }

    result_tabs = st.tabs(["Smart","Basic","Final Results"])
    with result_tabs[0]:
        ems_monitor_smart(default_data)

    with result_tabs[1]:
        ems_monitor_basic(default_data)

    with result_tabs[2]:
        # 2. Format Data for Table
        comparison_data = {
            "Metric": ["Balance (€)", "Consumption Savings (kW)"],
            "Smart Agent": [
                f"{default_final_results['smart_agent_balance']:.2f}", 
                f"{default_final_results['smart_agent_consumption_saving']:.2f}"
            ],
            "Basic Agent": [
                f"{default_final_results['basic_agent_balance']:.2f}", 
                f"{default_final_results['basic_agent_consumption_saving']:.2f}"
            ]
        }

        df = pd.DataFrame(comparison_data)

        # 3. Display Header and Table
        st.header("🏆 Smart vs. Basic Agent Performance")

        # Styling the table to look professional
        st.table(df)

        # 4. Impact Highlight
        st.metric(
            label="Balance Difference (Smart - Basic)", 
            value=f"{default_final_results['agent_balance_difference']:.2f} €",
            delta_color="inverse",
             help=f"""
            **Value:** `{default_final_results['agent_balance_difference']:.2f}`
            - **What it is:** The direct comparison between the two agents. 
            - A **negative** value indicates the Basic Agent outperformed the Smart Agent in this specific run.
            """
        )

        # 4. Impact Highlight
        st.metric(
            label="Total Consumption Cost", 
            value=f"{default_final_results['total_consumption_cust']:.2f} €",
            delta_color="inverse",
             help=f"""
            **Value:** `{default_final_results['total_consumption_cust']:.2f}`
            - **What it is:** The theoretical total cost if the customer had no Solar/Battery system at all. 
            - This serves as the 'worst-case scenario' baseline to measure how much both agents improved the situation.
            """
        )

        st.divider()

        # 5. Explanatory Tips (The "What does this mean?" section)
        st.subheader("💡 Metric Explanations")

        with st.expander("🔍 Balance"):
            st.write("""
            **What it is:** The net profit or loss generated by the agent during the simulation.
            - **Smart Agent Balance:** Total revenue from selling to the grid minus costs of buying from the grid and battery usage.
            - **Basic Agent Balance:** The benchmark performance, usually representing a standard 'dumb' strategy (e.g., just covering consumption).
            """)

        with st.expander("📊 Consumption Saving"):
            st.write("""
            **What it is:** The amount of energy (kW) the agent successfully diverted from the grid by using Solar or Battery power.
            - **Higher is better:** It means the agent is maximizing self-sufficiency and reducing the home's reliance on external energy.
            """)
