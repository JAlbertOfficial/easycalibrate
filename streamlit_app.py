
################################################################
# Packages
###############################################################

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt



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
    
    # User inputs
    x_input = st.text_area("Enter x values (comma separated)", "")
    y_input = st.text_area("Enter y values (comma separated)", "")
    
    if st.button("Perform classic calibration"):
        # Convert input strings to lists
        x_values = [float(i) for i in x_input.split(',') if i.strip()]
        y_values = [float(i) for i in y_input.split(',') if i.strip()]
        
        # Check if both lists have the same length
        if len(x_values) == len(y_values):

            # Create a dataframe
            data = {'x': x_values, 'y': y_values}
            df = pd.DataFrame(data)
            st.dataframe(df)  

            # Create scatterplot 
            fig, ax = plt.subplots()
            ax.plot(df['x'], df['y'], 'o')
            ax.set_xlabel('Nominal values')
            ax.set_ylabel('Signal')
            ax.set_title("Ordinary Least Square Calibration")

            # Display the plot
            st.pyplot(fig)

            plt.plot(df.x,df.y,'o')          
        else:
            st.error("The number of x values must be equal to the number of y values.")

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
