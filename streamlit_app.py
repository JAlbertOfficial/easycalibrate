################################################################
# Packages
###############################################################

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
from scipy.interpolate import make_interp_spline



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
    st.header("Home Page")
    st.write("Welcome to the Home page!")

###############################################################
# Function to render basic calibration page
###############################################################

###############################################################
# Function to render basic calibration page
###############################################################

def render_basic_calibration():
    st.header("Basic Calibration Page")
    st.subheader("Import Data")

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
            st.subheader("View raw data")
            st.dataframe(df)
            
            # Perform linear regression
            X = np.array(x_values).reshape(-1, 1)
            y = np.array(y_values)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Plot the data and the regression line
            st.subheader("Calibration plot")
            fig, ax = plt.subplots()
            ax.plot(df['x'], df['y'], 'o', label='Data points')
            ax.plot(df['x'], y_pred, '-', label='Regression line')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            
            # Display the plot in Streamlit
            st.pyplot(fig)

            # Display Calibration function section
            st.subheader("Calibration function")

            # Show formula of fitted linear regression
            st.markdown("**Formula of fitted linear regression:**")
            st.write(f"y = {model.coef_[0]} * x + {model.intercept_}")

            # Show slope of the fitted regression line
            st.markdown("**Slope of the fitted regression line:**")
            st.write(model.coef_[0])

            # Show intercept of the fitted regression line
            st.markdown("**Intercept of the fitted regression line:**")
            st.write(model.intercept_)
            
            # Display Model evaluation section
            st.subheader("Model evaluation")

            # Calculate adjusted R-squared
            n = len(y)
            p = 1  # number of predictors
            r_squared = model.score(X, y)
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            st.markdown("**Adjusted R-squared**:")
            st.write(adjusted_r_squared)

            # Calculate mean squared errors (MSE)
            mse = mean_squared_error(y, y_pred)
            st.markdown("**Mean Squared Error (MSE)**:")
            st.write(mse)

            # Calculate root mean squared errors (RMSE)
            rmse = np.sqrt(mse)
            st.markdown("**Root Mean Squared Error (RMSE)**:")
            st.write(rmse)

            # Display Model assumptions - Homoscedasticity section
            st.subheader("Model assumptions - Homoscedasticity")

            # Display Residual plots subsection
            st.markdown("**Residual plots**")

            # Calculate residuals
            residuals = y - y_pred

            # Residuals vs x plot
            fig_resid, ax_resid = plt.subplots()
            ax_resid.scatter(df['x'], residuals)
            ax_resid.axhline(y=0, color='black', linestyle='--')
            ax_resid.set_xlabel('x')
            ax_resid.set_ylabel('Residuals')
            ax_resid.set_title("Residuals vs x")
            st.pyplot(fig_resid)
            
            # Calculate standardized residuals
            standardized_residuals = residuals / np.std(residuals)

            # square root of the absolute value of standardized residuals vs x plot
            fig_sqrt_std_res, ax_sqrt_std_res = plt.subplots()
            ax_sqrt_std_res.scatter(df['x'], np.sqrt(np.abs(standardized_residuals)))
            ax_sqrt_std_res.set_xlabel('x')
            ax_sqrt_std_res.set_ylabel('sqrt(|Standardized Residuals|)')
            ax_sqrt_std_res.set_title("sqrt(|Standardized Residuals|) vs x")
            st.pyplot(fig_sqrt_std_res)

            # Calculate relative error
            x_calc = (y - model.intercept_) / model.coef_[0]
            relative_error = (x_calc - df['x']) / df['x']

            # Relative Error vs x plot
            fig_rel_error, ax_rel_error = plt.subplots()
            ax_rel_error.scatter(df['x'], relative_error)
            ax_rel_error.axhline(y=0, color='black', linestyle='--')
            ax_rel_error.set_xlabel('x')
            ax_rel_error.set_ylabel('Relative Error')
            ax_rel_error.set_title("Relative Error vs x")
            st.pyplot(fig_rel_error)

        else:
            st.error("The number of x values must be equal to the number of y values.")


###############################################################
# Function to render improved calibration page
###############################################################

def render_improved_calibration():
    st.header("Improved Calibration Page")
    st.write("This is the Improved Calibration page!")


###############################################################
# Function to render references page
###############################################################

def render_references():
    st.header("References Page")
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
