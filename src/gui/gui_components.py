import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.dialog("Confirm Data Collection",width='large')
def confirm_collection_modal(date_str):
    data_dir = os.path.join(src_dir, "sim", "data", "datafiles", date_str)

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
def display_simulation_overview(json_data):
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

    # 1. Create a "Wide" version of the agent data
    smart_data = df_simulation[df_simulation["Agent_Type"] == "smart"][["Time", "Battery", "Balance"]].rename(
        columns={"Battery": "Battery_Smart", "Balance": "Balance_Smart"})

    basic_data = df_simulation[df_simulation["Agent_Type"] == "basic"][["Time", "Battery", "Balance"]].rename(
        columns={"Battery": "Battery_Basic", "Balance": "Balance_Basic"}
    )

    # 2. Merge them back to the original
    df_simulation = df_simulation.merge(smart_data, on="Time", how="left").merge(basic_data, on="Time", how="left")

    all_columns = [
    "Solar_Production", 
    "Consumption", 
    "Price",
    "Balance_Smart",
    "Balance_Basic",
    "Battery_Basic",
    "Battery_Smart",]

    st.caption("Agent Simulation Overview")

    scale = st.selectbox( "Choose a scale for the polts.",
            ("Default", "Logarithmic"),help="The scale of a plot defines the mathematical relationship between a data point's value and its physical position on an axis, determining how intervals are spaced and visualized.")
    
    selected_cols = st.multiselect(
            "Select metrics to display", 
            options=all_columns, 
            default=["Solar_Production"] 
        )
    
    if scale == "Default":
        # 3. Display the chart
        if selected_cols:
            st.line_chart(df_simulation, x="Time", y=selected_cols,height=500)

            if st.button("Save Plot",type="primary"):
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_simulation["Time"], df_simulation[selected_cols])
                ax.set_title("Energy Simulation Results")

                # Display in Streamlit
                st.pyplot(fig)

                # Save to file
                fig.savefig("plots/simulation_results.png", dpi=300)
                st.success("Plot saved as simulation_results.png")
        else:
            st.info("Select one or more metrics from the dropdown above to start visualizing.")

    elif scale == "Logarithmic":
        df_log = df_simulation.copy()

        for col in all_columns:
            df_log[col] = df_log[col].apply(lambda x: np.log(x) if x > 0 else 0)

        # 3. Display the chart
        if selected_cols:
            st.line_chart(df_log, x="Time", y=selected_cols,height=500)

            if st.button("Save Plot",type="primary"):
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_log["Time"], df_log[selected_cols])
                ax.set_title("Energy Simulation Results")

                # Display in Streamlit
                st.pyplot(fig)

                # Save to file
                fig.savefig("plots/simulation_results.png", dpi=300)
                st.success("Plot saved as simulation_results.png")
        else:
            st.info("Select one or more metrics from the dropdown above to start visualizing.")

def display_input_metrics(input_data,timestamp):
   # 1. Big Timestamp Header
    st.markdown(
                f"""
                <div style="font-size: 20px; font-family: sans-serif; margin-bottom: 10px;">
                    🕒 <b>Time</b>: 
                    <span style="color: #60A5FA; font-family: monospace;">{timestamp}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    meta = {
        "Solar_Production": {"icon": "☀️", "unit": "kWh"},
        "Consumption": {"icon": "🏠", "unit": "kWh"},
        "Current_Capacity": {"icon": "🔋", "unit": "kWh"},
        "Price": {"icon": "💰", "unit": "/kWh"}
    }
    
    # 2. Loop through and display large text
    for key, value in input_data.items():
        if key != "Wind_Production":
            m = meta.get(key, {"icon": "📊", "unit": ""})
            label = key.replace('_', ' ')
            
            #Div with inline CSS for custom sizing
            st.markdown(
                f"""
                <div style="font-size: 20px; font-family: sans-serif; margin-bottom: 10px;">
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
            <div style="font-size: 20px; font-family: sans-serif; margin-bottom: 10px;">
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
                <text x="300" y="30" fill="#fbbf24" class="val" text-anchor="middle">{p2c:.2f} kWh</text>
            </g>

            <g style="{p2b_style}">
                <path d="M100,90 L100,220" stroke="#fbbf24" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="90" y="160" fill="#fbbf24" class="val" text-anchor="middle" transform="rotate(-90, 90, 160)">{p2b:.2f} kWh</text>

            </g>

            <g style="{b2c_style}">
                <path d="M140,240 L460,70" stroke="#00FF00" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="250" y="150" fill="#00FF00" class="val" transform="rotate(-28, 250, 150)">{b2c:.2f} kWh</text>

            </g>

            <g style="{g2c_style}">
                <path d="M500,220 L500,90" stroke="#FACC15" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="515" y="160" fill="#FACC15" class="val" text-anchor="middle" transform="rotate(90, 515, 160)">{g2c:.2f} h</text>

            </g>

            <g style="{b2g_style}">
                <path d="M140,265 L460,265" stroke="#FF4B4B" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="300" y="255" fill="#FF4B4B" class="val" text-anchor="middle">{b2g:.2f} kWh</text>
            </g>
            
            <g style="{g2b_style}">
                <path d="M460,285 L140,285" stroke="#60A5FA" stroke-width="3" fill="none" class="flow-line" marker-end="url(#arrow)" />
                <text x="300" y="310" fill="#60A5FA" class="val" text-anchor="middle">{g2b:.2f} kWh</text>
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
        c1, c3 = st.columns([1,1])
        with c3:
            forward_click = st.button("Fordward Step", use_container_width=True)
        with c1:
            backwards_click = st.button("Backward Step", use_container_width=True)
    
    if forward_click:
        st.session_state.index += 1
        st.session_state.index %= len(smart_keys)
        st.rerun(scope="fragment")
    
    if backwards_click:
        st.session_state.index -= 1
        st.session_state.index %= len(smart_keys)
        st.rerun(scope="fragment")

    with col1:
        _, c4,_ = st.columns([1,5,1])
        with c4:
            st.header("Input data")
        # Use an expander as the "click" mechanism
        timestamp = st.selectbox(
            "Choose a simulation step:",
            options=smart_keys,
            key="ts_choice",
            placeholder=smart_keys[st.session_state.index],
            index=st.session_state.index
        )

        st.session_state.index = smart_keys.index(timestamp)

        input_data = smart_data[timestamp]["input_data"]
        display_input_metrics(input_data,timestamp)

    with ems_map_window:
            actions = smart_data[timestamp]["Actions"]
            components.html(draw_ems_map(actions) , height=330)
    with col3:
        _, c5,_ = st.columns([1,6,1])
        with c5:
            st.header("Output Data")

        output_data = smart_data[timestamp]["output_data"]
        display_output_metrics(output_data)

@st.fragment
def ems_monitor_basic(basic_data):
    basic_keys = [i for i in basic_data.keys()]
    col1, col2, col3 = st.columns([1, 2, 1])
    ems_map_window = col2.empty()

    with col2:
        c1, c3 = st.columns([1,1])
        with c3:
            forward_click_1 = st.button("Fordward Step",key="FS1",use_container_width=True)
        with c1:
            backwards_click_1 = st.button("Backward Step",key="BS1", use_container_width=True)
    
    if forward_click_1:
        st.session_state.index_basic += 1
        st.session_state.index_basic %= len(basic_keys)
        st.rerun(scope="fragment")
    
    if backwards_click_1:
        st.session_state.index_basic -= 1
        st.session_state.index_basic %= len(basic_keys)
        st.rerun(scope="fragment")

    with col1:
        _, c6,_ = st.columns([1,5,1])
        with c6:
            st.header("Input data")

        # Use an expander as the "click" mechanism
        timestamp2 = st.selectbox(
            "Choose a simulation step:",
            options=basic_keys,
            key="ts_choice2",
            placeholder=basic_keys[st.session_state.index_basic],
            index=st.session_state.index_basic
        )

        st.session_state.index_basic = basic_keys.index(timestamp2)

        input_data = basic_data[timestamp2]["input_data"]
        display_input_metrics(input_data,timestamp2)

    with ems_map_window:
        st.title("Actions")
        actions = basic_data[timestamp2]["Actions"]
        components.html(draw_ems_map(actions) , height=330)
    
    with col3:
        _, c7,_ = st.columns([1,6,1])
        with c7:
            st.header("Output Data")
        output_data = basic_data[timestamp2]["output_data"]
        display_output_metrics(output_data)

def render_results(json_data):
    smart_data = json_data["smart"] if "smart" in json_data.keys() else None
    basic_data = json_data["basic"] if "basic" in json_data.keys() else None

    result_tabs = st.tabs(["Smart Agent Actions","Basic Agent Actions","Final Results","Model Statistics"])
    with result_tabs[0]:
        ems_monitor_smart(smart_data)

    with result_tabs[1]:
        ems_monitor_basic(basic_data)

    with result_tabs[2]:
        overview_button = st.button("Simulation Overview")

        if overview_button:
            display_simulation_overview(json_data)
                    
        final_results = json_data["final_results"]

        # 2. Format Data for Table
        comparison_data = {
            "Metric": ["Balance (€)", "Consumption Savings (kWh)"],
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
            **What it is:** The amount of energy (kWh) the agent successfully diverted from the grid by using Solar or Battery power.
            - **Higher is better:** It means the agent is maximizing self-sufficiency and reducing the home's reliance on external energy.
            """)
    
    with result_tabs[3]:
        csv_dir = os.path.join(src_dir, "sim", "agent", "smart", "models","csv_exports")
        
        if os.path.exists(csv_dir):
            for col, filename in zip(st.columns([2,2]), os.listdir(csv_dir)):
                with col:
                    file_path = os.path.join(csv_dir, filename)
                    df = pd.read_csv(file_path)
                    st.line_chart(df,x="Step",y="Value")