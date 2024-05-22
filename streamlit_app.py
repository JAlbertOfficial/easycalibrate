
################################################################
# Packages
###############################################################

import streamlit as st
from streamlit_option_menu import option_menu

###############################################################
# Define page titles
###############################################################

home = "Home"
basic_calibration = "Basic Calibration"
improved_calibration = "Improved Calibration"
references = "References"

###############################################################
# Function to render home page
###############################################################

def render_home():
    st.subheader("Home Page")
    st.write("Welcome to the Home page!")

###############################################################
# Function to render basic calibration page
###############################################################

def render_basic_calibration():
    st.subheader("Basic Calibration Page")
    st.write("This is the Basic Calibration page!")

###############################################################
# Function to render improved calibration page
###############################################################

def render_improved_calibration():
    st.subheader("Improved Calibration Page")
    st.write("This is the Improved Calibration page!")

###############################################################
# Function to render references page
###############################################################

def render_references():
    st.subheader("References Page")
    st.write("This is the References page!")

###############################################################
# Define the navigation menu
###############################################################

selected = option_menu(
    menu_title=None,
    options=[home, basic_calibration, improved_calibration, references],
    orientation="horizontal"
)

###############################################################
# Render the selected page
###############################################################

if selected == home:
    render_home()
elif selected == basic_calibration:
    render_basic_calibration()
elif selected == improved_calibration:
    render_improved_calibration()
elif selected == references:
    render_references()
