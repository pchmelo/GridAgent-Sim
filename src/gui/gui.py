import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import json

from sim.data.data_manager import data_manager
from sim.simulation_manager import SimulationManager

from gui_components import *

st.set_page_config(layout="wide",initial_sidebar_state="expanded")

zoom = 1.3

st.markdown(f"""
    <style>
        /* Increase base font size instead of zooming the whole container */
        html {{
            font-size: {zoom * 100}%; 
        }}
        
        /* Ensure sliders take up the full relative width of their parent */
        div[data-baseweb="slider"] {{
            width: 100%;
        }}
    </style>
    """, unsafe_allow_html=True)

def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                        url("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        /* Making the text more readable against the image */
        h1, h2, h3, p, .stMarkdown {{
            color: white !important;
        }}
        
        /* Optional: Styling the metric boxes to be semi-transparent */
        [data-testid="stMetricValue"] {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.session_state.backgroungd_on = True

def remove_background():
    """
    Overrides previous background images and resets the UI 
    to standard Streamlit colors.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background: none !important;
            background-color: #0E1117 !important; /* Default Streamlit Dark Color */
        }
        /* Reset text colors if you changed them to white for the image */
        h1, h2, h3, p, span, .stMarkdown {
            color: inherit !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.session_state.backgroungd_on = False

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
    st.session_state.interval = 15

if "max_capacity" not in st.session_state:
    st.session_state.max_capacity = 10

if "tariff" not in st.session_state:
    st.session_state.tariff = 75

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

if "previous_results" not in st.session_state:
    st.session_state.previous_results = False

if "new_simulation" not in st.session_state:
    st.session_state.new_simulation = False

if "backgroungd_on" not in st.session_state:
    st.session_state.backgroungd_on = False

if "inserting" not in st.session_state:
    st.session_state.inserting = False

if "data_inserted" not in st.session_state:
    st.session_state.data_inserted = False

results_dir = Path(__file__).parent.parent / "sim" / "data" / "results" / "final_results"
image_dir = Path(__file__).parent.parent / "gui" / "Images"

homescreen = st.empty()

#Render app home screen
with homescreen.container():
    set_background()
    _,cw,_ = st.columns([1,6,1])
    with cw:
        st.title("EMS-Sim: Energy Management System Simulator")
        _,cy,cz,_= st.columns([1,1,1,1])
        with cz:
            st.image(f"{image_dir}\LogotipoSI-759468593.png",width=250)
        with cy:
            st.image(f"{image_dir}\OIP-220066441.png",width=250)
        
    st.write("---")

    # Action area
    col1, col2 = st.columns([1, 1])
    with col1:
        new_simulation = st.empty()
        if new_simulation.button("New Simulation",use_container_width=True,type="primary"):
            st.session_state.new_simulation = True
            st.session_state.previous_results = False

    with col2:
        previous_results = st.empty()
        if previous_results.button("Previous Results",use_container_width=True,type="primary"):
            st.session_state.previous_results = True
            st.session_state.new_simulation = False

if st.session_state.new_simulation:
    if st.session_state.backgroungd_on:
        remove_background()

    new_simulation.empty()
    previous_results.empty()

    co,_ =  st.columns([3,7])
    with co:
        if st.button("<< Back",key="B1",use_container_width=True):
            st.session_state.new_simulation = False
            st.rerun()

    with st.sidebar:
        # INTERVAL slider
        interval = st.slider("Interval (minutes)", min_value=15, max_value=60, value=15, step=15,key="interval_slider_fixed")
        # MAX_CAPACITY
        max_capacity = st.number_input("Max Battery Capacity (kWh)", min_value=1, max_value=1000,value=10)
        # TARIFF
        tariff = st.number_input("Tariff", min_value=0.0, max_value=1.0,value=0.75, step=0.01,
                                help="The second parameter is the relationship between import and export tariffs. For example, if tariff = 75%, this means that the selling price is 75% of the purchase price")

        cole, colf= st.columns([5,5])
        with cole:
            insert_csv = st.button("Insert CSV",use_container_width=True)

        with colf:
            use_api = st.button("Use API",use_container_width=True)

        reset_config = st.button("Clear data",use_container_width=True,key="reset_config")

        if reset_config:
            st.session_state.selected_date = None
            st.session_state.inserting_data = False
            st.session_state.interval = 15
            st.session_state.max_capacity = 10
            st.session_state.tariff = 75
            st.session_state.consumption_data = None
            st.session_state.market_data = None
            st.session_state.solar_data = None
            st.session_state.wind_data = None
            st.rerun()

        if insert_csv:
            st.session_state.show_calendar = False
            st.session_state.selected_date = None
            st.session_state.inserting = True

        if use_api:
            st.session_state.consumption_data = None
            st.session_state.market_data = None
            st.session_state.solar_data = None
            st.session_state.wind_data = None
            st.session_state.inserting = False
            st.session_state.data_inserted = False
            st.session_state.show_calendar = True

        if st.session_state.inserting:
            uploaded_file_1 = st.file_uploader("Insert consumption data", type=["csv"], key="consumption_uploader")
            if uploaded_file_1:
                df = pd.read_csv(uploaded_file_1)
                st.session_state.consumption_data = df
                st.success("File uploaded successfully!")
            
            uploaded_file_2 = st.file_uploader("Insert market price data", type=["csv"], key="market_uploader")
            if uploaded_file_2:
                df = pd.read_csv(uploaded_file_2)
                st.session_state.market_data = df
                st.success("File uploaded successfully!")
            
            uploaded_file_3 = st.file_uploader("Insert solar production data", type=["csv"], key="solar_uploader")
            if uploaded_file_3:
                df = pd.read_csv(uploaded_file_3)
                st.session_state.solar_data = df
                st.success("File uploaded successfully!")
            
            uploaded_file_4 = st.file_uploader("Insert wind production data", type=["csv"], key="wind_uploader")
            if uploaded_file_4:
                df = pd.read_csv(uploaded_file_4)
                st.session_state.wind_data = df
                st.success("File uploaded successfully!")
            
            if (uploaded_file_1 != None) and (uploaded_file_2 != None) and (uploaded_file_3 != None) and  (uploaded_file_4 != None):
                st.session_state.inserting = False
                st.session_state.data_inserted = True
                st.rerun()

        calendar_placeholder = st.empty()

        #Render the calendar
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

        #Check if all the needed data was introduced
        if (st.session_state.selected_date != None) or st.session_state.data_inserted:
                #Print the input data
                if st.session_state.selected_date != None:
                    st.session_state.interval = interval
                    st.session_state.max_capacity = max_capacity
                    st.session_state.tariff = tariff
                    st.info(f"Data Loaded: \n\n Selected date: {st.session_state.selected_date} \n\n Interval: {st.session_state.interval} minutes \n\n Max Capacity: {st.session_state.max_capacity} kWh \n\n Tariff: {st.session_state.tariff}%")

                #Print the input data
                if st.session_state.data_inserted:
                    st.session_state.interval = interval
                    st.session_state.max_capacity = max_capacity
                    st.session_state.tariff = tariff
                    st.info(f"Data Loaded: \n\n Interval: {st.session_state.interval} minutes \n\n Max Capacity: {st.session_state.max_capacity} kWh \n\n Tariff: {st.session_state.tariff}%")

                #Start the simulation
                if st.button("Run Simulation",type="primary",key="RS1", use_container_width=True):
                    with st.spinner("Simulating Agent Actions..."):
                        config = {
                        "selected_date": st.session_state.get("selected_date"),
                        "interval": st.session_state.get("interval"),
                        "max_capacity": st.session_state.get("max_capacity"),
                        "tariff": st.session_state.get("tariff"),
                        "used_api":True if st.session_state.get("selected_date") else False
                        }

                        json_data = json.dumps(config, indent=4)

                        sim_manager = SimulationManager()

                        json_res = sim_manager.start_simulation(config=config,
                                                        df_solar_production=st.session_state.solar_data,
                                                        df_wind_production=st.session_state.wind_data,
                                                        df_consumption=st.session_state.consumption_data,
                                                        df_price=st.session_state.market_data)
                        
                    st.success("Simulation Complete!")
                    st.session_state.simulation_run = True
                    st.rerun()

    #Remove the calendar after selecting the date
    if st.session_state.data_confirmed:
        st.session_state.data_confirmed = False
        calendar_placeholder.empty()

    #Display the results of the simulation
    if st.session_state.simulation_run:
        homescreen.empty()
        result_files = sorted(results_dir.glob("final_results_*.json"), reverse=True)

        if result_files:
            with open(result_files[0], "r") as file:
                json_data = json.loads(file.read())
        
            render_results(json_data)

#Loads the previous results page
if st.session_state.previous_results:
    if st.session_state.backgroungd_on:
        remove_background()

    new_simulation.empty()
    previous_results.empty()
    homescreen.empty()
    
    cy,_ =  st.columns([3,7])
    with cy:
        if st.button("<< Back",key="B0",use_container_width=True):
            st.session_state.previous_results = False
            st.rerun()
        
    result_files = sorted(results_dir.glob("final_results_*.json"), reverse=True)
    
    previous_file = st.selectbox("Previous Simulations",result_files)

    if previous_file:
        with open(previous_file, "r") as file:
                json_data = json.loads(file.read())

        render_results(json_data)
