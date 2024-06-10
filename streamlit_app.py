
################################################################
# Packages
###############################################################

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


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
    
    # Default values for x and y
   # Default values for x and y
    default_x_values = ("0.050, 0.050, 0.050, 0.125, 0.125, 0.125, "
                    "0.500, 0.500, 0.500, 1.250, 1.250, 1.250, "
                    "2.500, 2.500, 2.500, 5.000, 5.000, 5.000, "
                    "12.500, 12.500, 12.500, 25.000, 25.000, 25.000")
    default_y_values = ("9.102775, 9.102971, 9.096732, 22.658198, 22.882701, 22.830106, "
                    "77.690938, 75.064287, 80.320072, 197.030149, 197.390646, 196.477779, "
                    "388.543543, 382.672992, 378.273372, 844.937521, 799.804932, 799.695752, "
                    "1996.367224, 1987.843702, 1969.842072, 3901.977880, 3786.692867, 3762.291002")

    # User inputs
    x_input = st.text_area("Enter x values (comma separated)", default_x_values)
    y_input = st.text_area("Enter y values (comma separated)", default_y_values)
    
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
            
            # Perform linear regression
            X = np.array(x_values).reshape(-1, 1)
            y = np.array(y_values)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Plot the data and the regression line
            fig, ax = plt.subplots()
            ax.plot(df['x'], df['y'], 'o', label='Data points')
            ax.plot(df['x'], y_pred, '-', label='Regression line')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Scatter plot of x vs y with regression line')
            ax.legend()
            
            # Display the plot in Streamlit
            st.pyplot(fig)
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
