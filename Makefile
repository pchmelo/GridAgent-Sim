PYTHON := python

.PHONY: run 

# Run Streamlit directly against the GUI entry
run:
	$(PYTHON) -m streamlit run src/gui/gui.py

