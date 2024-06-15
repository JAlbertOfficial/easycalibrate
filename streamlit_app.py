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
from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.compat as lzip
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as smd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.stats.diagnostic as sm_diagnostic

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

def bc_import_data():
    st.header("Import Data")
    st.write("The data can be entered manually or from an uploaded spreadsheet file.")

    data_source = st.radio("Select data import method:", ("Manual Input", "Upload File"))

    if data_source == "Manual Input":
        st.subheader("Enter Data Manually")
        st.write("The x-values represent the known concentrations or quantities of the analyte being measured. "
                 "The y-values correspond to the measured signal produced by the analyte at each known concentration.")
        # Default values for demonstration
        default_x_values = ("0.050, 0.050, 0.050, 0.125, 0.125, 0.125, "
                            "0.500, 0.500, 0.500, 1.250, 1.250, 1.250, "
                            "2.500, 2.500, 2.500, 5.000, 5.000, 5.000, "
                            "12.500, 12.500, 12.500, 25.000, 25.000, 25.000")
        default_y_values = ("9.102775, 9.102971, 9.096732, 22.658198, 22.882701, 22.830106, "
                            "77.690938, 75.064287, 80.320072, 197.030149, 197.390646, 196.477779, "
                            "388.543543, 382.672992, 378.273372, 844.937521, 799.804932, 799.695752, "
                            "1996.367224, 1987.843702, 1969.842072, 3901.977880, 3786.692867, 3762.291002")

        # Grid layout for input fields
        col1, col2 = st.columns(2)

        with col1:
            x_label = st.text_input("Enter a name for the x-variable:", "Concentration[mg/L]")

        with col2:
            y_label = st.text_input("Enter a name for the y-variable:", "Peak area")

        st.write("")  # empty line for spacing

        with col1:
            x_input = st.text_area("Enter x-values (comma separated):", default_x_values)

        with col2:
            y_input = st.text_area("Enter y-values (comma separated):", default_y_values)

        st.write("")  # empty line for spacing

        # Button to import data
        if st.button("Import data"):
            x_values = [float(i) for i in x_input.split(',') if i.strip()]
            y_values = [float(i) for i in y_input.split(',') if i.strip()]

            if len(x_values) == len(y_values):
                data = {'x': x_values, 'y': y_values}
                df = pd.DataFrame(data)
                st.session_state['df'] = df
                st.session_state['x_label'] = x_label
                st.session_state['y_label'] = y_label

                st.success("Data imported successfully.")
                st.session_state['data_imported'] = True
                st.session_state['current_section'] = "View Data"
                st.experimental_rerun()
            else:
                st.error("The number of x-values must be equal to the number of corresponding y-values.")

    elif data_source == "Upload File":
        st.subheader("Upload File")
        uploaded_file = st.file_uploader("Select a CSV, XLSX, ODS, or XLS file", type=["csv", "xlsx", "ods", "xls"])

        if uploaded_file is not None:
            x_label = st.text_input("Enter a name for the x-variable:", "Concentration[mg/L]")
            y_label = st.text_input("Enter a name for the y-variable:", "Peak area")

            if st.button("Import data"):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, header=0, sep=None, engine='python')
                    else:
                        df = pd.read_excel(uploaded_file, header=0)  # Load file with headers

                    if len(df.columns) < 2:
                        st.error("The uploaded file should have at least two columns (x and y values).")
                    else:
                        # Drop rows with non-numeric values
                        df = df.dropna().apply(pd.to_numeric, errors='coerce').dropna()

                        if df.empty:
                            st.error("The uploaded file does not contain valid numeric data.")
                        else:
                            df.columns = ['x', 'y']  # Assign column names after loading the file
                            st.session_state['df'] = df

                            st.write("The x-values represent the known concentrations or quantities of the analyte being measured. "
                                     "The y-values correspond to the measured signal produced by the analyte at each known concentration.")

                            st.session_state['x_label'] = x_label
                            st.session_state['y_label'] = y_label

                            st.success("Data imported successfully.")
                            st.session_state['data_imported'] = True
                            st.session_state['current_section'] = "View Data"
                            st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

def bc_train_model():
    st.subheader("Train Calibration Model")
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        x_label = st.session_state['x_label']
        y_label = st.session_state['y_label']
        X = df[['x']].values
        y = df['y'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        st.session_state['model'] = model
        st.session_state['y_pred'] = y_pred

        st.success("Calibration model trained successfully.")
        st.session_state['model_trained'] = True
        st.session_state['current_section'] = "Calibration Plot"
        st.experimental_rerun()

def bc_raw_data():
    st.subheader("Calibration Data")
    if 'df' in st.session_state:
        st.dataframe(st.session_state['df'])
        st.write("")  # empty line for spacing
        if st.button("Fit calibration model"):
            bc_train_model()
    else:
        st.error("No data available. Please import data first.")

def bc_calibration_plot():
    st.subheader("Calibration Plot")
    if 'df' in st.session_state and 'model' in st.session_state:
        df = st.session_state['df']
        model = st.session_state['model']
        y_pred = st.session_state['y_pred']
        x_label = st.session_state['x_label']
        y_label = st.session_state['y_label']
        fig, ax = plt.subplots()
        ax.plot(df['x'], df['y'], 'o', label='Data points')
        ax.plot(df['x'], y_pred, '-', label='Regression line')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

        st.pyplot(fig)
    else:
        st.error("No data or model available. Please import data first.")

def bc_calibration_function():
    st.subheader("Calibration Function")
    if 'model' in st.session_state:
        model = st.session_state['model']
        x_label = st.session_state['x_label']
        y_label = st.session_state['y_label']
        st.markdown("Formula of fitted linear regression:")
        st.write(f"{y_label} = {model.coef_[0]} * {x_label} + {model.intercept_}")
        st.markdown("Slope of the fitted regression line:")
        st.write(model.coef_[0])
        st.markdown("Intercept of the fitted regression line:")
        st.write(model.intercept_)
    else:
        st.error("No model available. Please import data first.")

def bc_model_evaluation():
    st.subheader("Model Evaluation")
    if 'model' in st.session_state:
        model = st.session_state['model']
        y = st.session_state['df']['y']
        y_pred = st.session_state['y_pred']
        n = len(y)
        p = 1  # number of predictors
        r_squared = model.score(np.array(st.session_state['df']['x']).reshape(-1, 1), y)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        st.markdown("Adjusted R-squared:")
        st.write(adjusted_r_squared)
        st.markdown("Mean Squared Error (MSE):")
        st.write(mse)
        st.markdown("Root Mean Squared Error (RMSE):")
        st.write(rmse)
    else:
        st.error("No model available. Please import data first.")

def bc_model_assumptions():
    st.header("Model Assumptions")
    
    # Homoscedasticity
    st.subheader("Homoscedasticity")
    if 'df' in st.session_state and 'model' in st.session_state:
        df = st.session_state['df']
        model = st.session_state['model']
        residuals = st.session_state['df']['y'] - st.session_state['y_pred']
        fit = smf.ols('y ~ x', data=df).fit()
        test_result_bp = sm_diagnostic.het_breuschpagan(fit.resid, fit.model.exog)
        st.markdown("**Breusch-Pagan Test for Homoscedasticity**")
        st.write("P-value:", test_result_bp[1])
        st.write("Degrees of freedom:", test_result_bp[2])
        st.write("F-statistic:", test_result_bp[3])

        if test_result_bp[1] > 0.05:
            st.write("The Breusch-Pagan test for Homoscedasticity is not significant (p > 0.05), indicating that the residuals have constant variance (homoscedasticity).")
        else:
            st.write("The Breusch-Pagan test for Homoscedasticity is significant (p <= 0.05), suggesting that the residuals may not have constant variance (heteroscedasticity).")
            st.write("Considering the potential violation of homoscedasticity assumption, further diagnostics or transformations may be necessary.")
            st.write("The distribution of residuals and the test result should be visually verified with the following residual plots.")

        st.markdown("**Residual plots**")
        fig_resid, ax_resid = plt.subplots()
        ax_resid.scatter(df['x'], residuals)
        ax_resid.axhline(y=0, color='black', linestyle='--')
        ax_resid.set_xlabel(st.session_state['x_label'])
        ax_resid.set_ylabel("Residuals")
        ax_resid.set_title("Residual plot")
        st.pyplot(fig_resid)
    else:
        st.error("No data or model available. Please import data first.")
    
    # Normality of Residuals
    st.subheader("Normality of Residuals")
    if 'df' in st.session_state and 'model' in st.session_state:
        df = st.session_state['df']
        residuals = st.session_state['df']['y'] - st.session_state['y_pred']
        shapiro_test = stats.shapiro(residuals)
        st.markdown("**Shapiro-Wilk Test for Normality of Residuals**")
        st.write("P-value:", shapiro_test[1])

        if shapiro_test[1] > 0.05:
            st.write("The Shapiro-Wilk test for Normality of Residuals is not significant (p > 0.05), indicating that the residuals are normally distributed.")
        else:
            st.write("The Shapiro-Wilk test for Normality of Residuals is significant (p <= 0.05), suggesting that the residuals may not be normally distributed.")
            st.write("Considering the potential violation of normality assumption, further diagnostics or transformations may be necessary.")
            st.write("The distribution of residuals and test result should be visually verified with the following Q-Q plot.")
        st.markdown("**Q-Q plot of residuals**")
        fig_qq, ax_qq = plt.subplots()
        sm.qqplot(residuals, line='s', ax=ax_qq)
        ax_qq.set_title("Q-Q plot of residuals")
        st.pyplot(fig_qq)
    else:
        st.error("No data or model available. Please import data first.")

def render_basic_calibration():
    if st.session_state.get('data_imported'):
        if st.session_state.get('model_trained'):
            bc_section = st.sidebar.radio(
                "",
                ["Import Data", "View Data", "Calibration Plot", "Calibration Function", 
                "Model Evaluation", "Model Assumptions"],
                index=["Import Data", "View Data", "Calibration Plot", "Calibration Function", 
                       "Model Evaluation", "Model Assumptions"].index(st.session_state.get('current_section', "Import Data"))
            )
        else:
            bc_section = st.sidebar.radio(
                "Navigate Basic Calibration",
                ["Import Data", "View Data"],
                index=["Import Data", "View Data"].index(st.session_state.get('current_section', "Import Data"))
            )
    else:
        bc_section = st.sidebar.radio(
            "Navigate Basic Calibration",
            ["Import Data"]
        )

    if bc_section == "Import Data":
        bc_import_data()

    elif bc_section == "View Data":
        bc_raw_data()

    elif bc_section == "Calibration Plot":
        bc_calibration_plot()

    elif bc_section == "Calibration Function":
        bc_calibration_function()

    elif bc_section == "Model Evaluation":
        bc_model_evaluation()

    elif bc_section == "Model Assumptions":
        bc_model_assumptions()

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
