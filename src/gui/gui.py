import streamlit as st
import streamlit.components.v1 as components
import time
from datetime import datetime
import pandas as pd
#from sim.data.data_manager import data_manager
st.set_page_config(layout="wide")

def create_log(log : str):
    return f'<p class="run">{log}</p>'

if "show_calendar" not in st.session_state:
    st.session_state.show_calendar = False

if "inserting_data" not in st.session_state:
    st.session_state.inserting_data = False

# Create log state in app
if "logs" not in st.session_state:
    st.session_state.logs = [create_log(str(i)) for i in range(5)]

if "data" not in st.session_state:
    st.session_state.data = None

cola, colb, colc, cold, cole = st.columns([1, 1, 1, 1, 1])
with cola:
    # make inner columns equal width but spacious
    ca, cb = st.columns([1, 1])
    with ca:
        insert_clicked = st.button("Insert Data", use_container_width=True)
    with cb:
        api_clicked = st.button("Use API", use_container_width=True) 
        if api_clicked:
            st.session_state.show_calendar = True  # remember to show calendar

calendar_placeholder = st.empty()


#Calender to select date for REN API
if st.session_state.show_calendar:
    with calendar_placeholder.container():
        default_date = datetime.now().strftime("%Y-%m-%d")
        selected_date = st.date_input("Select a date", value=default_date,key="api_date").strftime("%Y-%m-%d")

        confirm_button = st.button("Confirm date")
        cancel_button = st.button("Cancel")

        if cancel_button:
            st.session_state.show_calendar = False
            calendar_placeholder.empty()

        if confirm_button:
            #data_manager.start_data_collection(selected_date)
            st.session_state.show_calendar = False
            calendar_placeholder.empty()

# Button functionality
if insert_clicked:
    st.session_state.inserting_data = True
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="uploader")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.dataframe(df)
        st.success("File uploaded successfully!")

config_placeholder = st.empty()

if st.session_state.inserting_data:
    with config_placeholder.container():
        # AGENT_TYPE toggle
        agent_type = st.toggle("Toggle complex mode", value=True)
        # INTERVAL slider
        interval = st.slider("INTERVAL", min_value=0.0, max_value=1.0, value=0.0, step=0.15)
        # MAX_CAPACITY
        max_capacity = st.number_input("MAX_CAPACITY", min_value=0, value=10**10)
        # TARIFF
        tariff = st.number_input("TARIFF", min_value=0.0, max_value=1.0,value=0.75, step=0.01)

        confirm_config = st.button("Confirm config")
        cancel_config = st.button("Cancel")

        if cancel_config:
            st.session_state.inserting_data = False
            config_placeholder.empty()

        if confirm_config:
            st.session_state.inserting_data = False
            config_placeholder.empty()


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

# --- Text window above buttons ---
text_window = st.container()  # This will hold the text display

with text_window:
    # Create a scrollable box using st.markdown with styling
    st.markdown(
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
        """ + "<br>".join(st.session_state.logs) + "</div>",
        unsafe_allow_html=True
    )

col1, col2, col3, col4, col5 = st.columns([1, 1, 4, 1, 1])
with col3:
    # make inner columns equal width but spacious
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        forward_click = st.button("Fordward Step", use_container_width=True)
    with c2:
        backwards_click = st.button("Backward Step", use_container_width=True)
    with c3:
        run_simulation = st.button("Run Simulation", use_container_width=True)

# --- Simulate appending logs dynamically ---
if run_simulation:
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
