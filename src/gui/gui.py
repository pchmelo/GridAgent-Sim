import streamlit as st
import streamlit.components.v1 as components
import time
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
from sim.data.data_manager import data_manager
import json
import os
#from sim.simulation_manager import SimulationManager
import sys
import os
from pathlib import Path

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
    df_simulation["Time"] = pd.to_datetime(df_simulation["Time"])

    with tabs[0]:
        df_smart = df_simulation[df_simulation["Agent_Type"] == "smart"]
        st.caption("Smart Agent Simulation Overview")

        st.line_chart(df_smart,x="Time",y="Solar_Production")

        st.line_chart(df_smart,x="Time",y="Wind_Production")

        st.line_chart(df_smart,x="Time",y="Consumption")

        st.line_chart(df_smart,x="Time",y="New_Capacity")

        st.line_chart(df_smart,x="Time",y="Current_Capacity")

        st.line_chart(df_smart,x="Time",y="Price")
    
    with tabs[1]:
        df_basic = df_simulation[df_simulation["Agent_Type"] == "basic"]
        st.caption("Basic Agent Simulation Overview")

        st.line_chart(df_basic,x="Time",y="Solar_Production")

        st.line_chart(df_basic,x="Time",y="Wind_Production")

        st.line_chart(df_basic,x="Time",y="Consumption")

        st.line_chart(df_basic,x="Time",y="New_Capacity")

        st.line_chart(df_basic,x="Time",y="Current_Capacity")

        st.line_chart(df_basic,x="Time",y="Price")

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

result_tabs = st.tabs(["Smart","Basic"])
with result_tabs[0]:
    col1, col2, col3= st.columns([3, 3, 3])

    text_window_1 = st.empty()
    text_window_4 = st.empty()
    with col1:
        st.header("Input Data")
        text_window_1 = st.html(
                """
                <div style='
                    height: 400px;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    padding: 10px;
                    overflow-y: auto;
                    border-radius: 5px;
                    border: 1px solid #333;
                '>
                """ + "<br>".join(st.session_state.logs) + "</div>"
            )

    text_window_2 = st.empty()
    text_window_5 = st.empty()
    with col2:
        st.header("Action")
        text_window_2 = st.html(
                """
                <div style='
                    height: 400px;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    padding: 10px;
                    overflow-y: auto;
                    border-radius: 5px;
                    border: 1px solid #333;
                '>
                """ + "<br>".join(st.session_state.logs) + "</div>"
            )

    text_window_3 = st.empty()
    text_window_6 = st.empty()
    with col3:
        st.header("Output Data")
        text_window_3 = st.html(
                """
                <div style='
                    height: 400px;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    padding: 10px;
                    overflow-y: auto;
                    border-radius: 5px;
                    border: 1px solid #333;
                '>
                """ + "<br>".join(st.session_state.logs) + "</div>"
            )
    with col2:
        # make inner columns equal width but spacious
        c1, c2, c3 = st.columns([1, 1, 1])
        with c3:
            forward_click = st.button("Fordward Step", use_container_width=True)
        with c1:
            backwards_click = st.button("Backward Step", use_container_width=True)
        with c2:
            run_simulation = st.button("Run Simulation",type="primary", use_container_width=True)

with result_tabs[1]:
    col1, col2, col3= st.columns([3, 3, 3])

    text_window_4 = st.empty()
    with col1:
        st.header("Input Data")
        text_window_4 = st.html(
                """
                <div style='
                    height: 400px;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    padding: 10px;
                    overflow-y: auto;
                    border-radius: 5px;
                    border: 1px solid #333;
                '>
                """ + "<br>".join(st.session_state.logs) + "</div>"
            )

    text_window_5 = st.empty()
    with col2:
        st.header("Action")
        text_window_5 = st.html(
                """
                <div style='
                    height: 400px;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    padding: 10px;
                    overflow-y: auto;
                    border-radius: 5px;
                    border: 1px solid #333;
                '>
                """ + "<br>".join(st.session_state.logs) + "</div>"
            )

    text_window_6 = st.empty()
    with col3:
        st.header("Output Data")
        text_window_6 = st.html(
                """
                <div style='
                    height: 400px;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    padding: 10px;
                    overflow-y: auto;
                    border-radius: 5px;
                    border: 1px solid #333;
                '>
                """ + "<br>".join(st.session_state.logs) + "</div>"
            )
    with col2:
        # make inner columns equal width but spacious
        c1, c2, c3 = st.columns([1, 1, 1])
        with c3:
            forward_click_1 = st.button("Fordward Step", key="FS1",use_container_width=True)
        with c1:
            backwards_click_1 = st.button("Backward Step",key="BS1", use_container_width=True)

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

    """SimulationManager.start_smulation(config=config,
                                      df_solar_production=st.session_state.solar_data,
                                      df_wind_production=st.session_state.wind_data,
                                      df_consumption=st.session_state.consumption_data,
                                      df_price=st.session_state.market_data)"""
    
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

with open(r"C:\Users\gabib\Desktop\Mestrado_IA\MS\MS_Household_Energy_Production\src\log\files\final_results_20251121_123019.json","r") as file:
    json_data = json.loads(file.read())

smart_data = json_data["smart"] if "smart" in json_data.keys() else None
basic_data = json_data["basic"] if "basic" in json_data.keys() else None

smart_keys = [i for i in smart_data.keys()]

input_data = smart_data[smart_keys[0]]["input_data"].items()
actions =[i.items() for i in smart_data[smart_keys[0]]["Actions"]]
output_data = smart_data[smart_keys[0]]["output_data"].items()

basic_keys = [i for i in basic_data.keys()]

basic_input_data = basic_data[basic_keys[0]]["input_data"].items()
basic_actions =[i.items() for i in basic_data[basic_keys[0]]["Actions"]]
basic_output_data = basic_data[basic_keys[0]]["output_data"].items()

text_window_1.html(
        """
        <div style='
            height: 400px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: Consolas, monospace;
            padding: 10px;
            overflow-y: auto;
            border-radius: 5px;
            border: 1px solid #333;
        '>
        """ + "<br>".join(["Timestamp : " + smart_keys[st.session_state.index]] + [i[0] + " : " + str(i[1]) for i in input_data]) + "</div>"
    )

text_window_2.html(
        """
        <div style='
            height: 400px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: Consolas, monospace;
            padding: 10px;
            overflow-y: auto;
            border-radius: 5px;
            border: 1px solid #333;
        '>
        """ + "<br>".join([j[0]  + " : " + str(j[1]) for i in actions for j in i]) + "</div>"
    )

text_window_3.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([i[0] + " : " + str(i[1]) for i in output_data]) + "</div>"
        )

text_window_4.html(
    """
    <div style='
        height: 400px;
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: Consolas, monospace;
        padding: 10px;
        overflow-y: auto;
        border-radius: 5px;
        border: 1px solid #333;
    '>
    """ + "<br>".join(["Timestamp : " + basic_keys[st.session_state.index_basic]] + [i[0] + " : " + str(i[1]) for i in basic_input_data]) + "</div>"
)

text_window_5.html(
    """
    <div style='
        height: 400px;
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: Consolas, monospace;
        padding: 10px;
        overflow-y: auto;
        border-radius: 5px;
        border: 1px solid #333;
    '>
    """ + "<br>".join([j[0]  + " : " + str(j[1]) for i in basic_actions for j in i]) + "</div>"
)

text_window_6.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([i[0] + " : " + str(i[1]) for i in basic_output_data]) + "</div>"
        )


if forward_click:
    st.session_state.index += 1
    input_data = smart_data[smart_keys[st.session_state.index]]["input_data"].items()
    actions =[i.items() for i in smart_data[smart_keys[st.session_state.index]]["Actions"]]
    output_data = smart_data[smart_keys[st.session_state.index]]["output_data"].items()
    
    text_window_1.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join(["Timestamp : " + smart_keys[st.session_state.index]] + [i[0] + " : " + str(i[1]) for i in input_data]) + "</div>"
        )
    
    text_window_2.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([j[0]  + " : " + str(j[1]) for i in actions for j in i]) + "</div>"
        )
    
    text_window_3.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([i[0] + " : " + str(i[1]) for i in output_data]) + "</div>"
        )

if backwards_click:
    st.session_state.index -= 1
    input_data = smart_data[smart_keys[st.session_state.index]]["input_data"].items()
    actions =[i.items() for i in smart_data[smart_keys[st.session_state.index]]["Actions"]]
    output_data = smart_data[smart_keys[st.session_state.index]]["output_data"].items()
    
    text_window_1.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join(["Timestamp : " + smart_keys[st.session_state.index]] + [i[0] + " : " + str(i[1]) for i in input_data]) + "</div>"
        )
    
    text_window_2.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([j[0]  + " : " + str(j[1]) for i in actions for j in i]) + "</div>"
        )
    
    text_window_3.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([i[0] + " : " + str(i[1]) for i in output_data]) + "</div>"
        )

if forward_click_1:
    st.session_state.index_basic += 1
    basic_input_data = basic_data[basic_keys[st.session_state.index_basic]]["input_data"].items()
    basic_actions =[i.items() for i in basic_data[basic_keys[st.session_state.index_basic]]["Actions"]]
    basic_output_data = basic_data[basic_keys[st.session_state.index_basic]]["output_data"].items()
    
    text_window_1.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join(["Timestamp : " + basic_keys[st.session_state.index_basic]] + [i[0] + " : " + str(i[1]) for i in basic_input_data]) + "</div>"
        )
    
    text_window_2.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([j[0]  + " : " + str(j[1]) for i in basic_actions for j in i]) + "</div>"
        )
    
    text_window_3.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([i[0] + " : " + str(i[1]) for i in basic_output_data]) + "</div>"
        )

if backwards_click_1:
    st.session_state.index_basic -= 1
    basic_input_data = basic_data[basic_keys[st.session_state.index_basic]]["input_data"].items()
    basic_actions =[i.items() for i in basic_data[basic_keys[st.session_state.index_basic]]["Actions"]]
    basic_output_data = basic_data[basic_keys[st.session_state.index_basic]]["output_data"].items()
    
    text_window_4.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join(["Timestamp : " + basic_keys[st.session_state.index_basic]] + [i[0] + " : " + str(i[1]) for i in basic_input_data]) + "</div>"
        )
    
    text_window_5.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([j[0]  + " : " + str(j[1]) for i in basic_actions for j in i]) + "</div>"
        )
    
    text_window_6.html(
            """
            <div style='
                height: 400px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                padding: 10px;
                overflow-y: auto;
                border-radius: 5px;
                border: 1px solid #333;
            '>
            """ + "<br>".join([i[0] + " : " + str(i[1]) for i in basic_output_data]) + "</div>"
        )

with col3:
    if st.session_state.simulation_run:
        overview_button = st.button("Simulation Overview")

        if overview_button:
            display_simulation_overview(r"C:\Users\gabib\Desktop\Mestrado_IA\MS\MS_Household_Energy_Production\src\log\files\final_results_20251121_123019.json")
