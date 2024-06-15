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
    st.subheader("Import Data")
    default_x_values = ("0.050, 0.050, 0.050, 0.125, 0.125, 0.125, "
                        "0.500, 0.500, 0.500, 1.250, 1.250, 1.250, "
                        "2.500, 2.500, 2.500, 5.000, 5.000, 5.000, "
                        "12.500, 12.500, 12.500, 25.000, 25.000, 25.000")
    default_y_values = ("9.102775, 9.102971, 9.096732, 22.658198, 22.882701, 22.830106, "
                        "77.690938, 75.064287, 80.320072, 197.030149, 197.390646, 196.477779, "
                        "388.543543, 382.672992, 378.273372, 844.937521, 799.804932, 799.695752, "
                        "1996.367224, 1987.843702, 1969.842072, 3901.977880, 3786.692867, 3762.291002")
    x_label = st.text_input("Enter the label for x", "Concentration[mg/L]")
    y_label = st.text_input("Enter the label for y", "Peakarea")
    x_input = st.text_area("Enter x values (comma separated)", default_x_values)
    y_input = st.text_area("Enter y values (comma separated)", default_y_values)

    if st.button("Perform classic calibration"):
        x_values = [float(i) for i in x_input.split(',') if i.strip()]
        y_values = [float(i) for i in y_input.split(',') if i.strip()]

        if len(x_values) == len(y_values):
            data = {'x': x_values, 'y': y_values}
            df = pd.DataFrame(data)
            st.session_state['df'] = df
            st.session_state['x_label'] = x_label
            st.session_state['y_label'] = y_label

            X = np.array(x_values).reshape(-1, 1)
            y = np.array(y_values)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.session_state['model'] = model
            st.session_state['y_pred'] = y_pred

            st.success("Data imported and model trained successfully.")
        else:
            st.error("The number of x values must be equal to the number of y values.")

def bc_raw_data():
    st.subheader("Calibration Data")
    if 'df' in st.session_state:
        st.dataframe(st.session_state['df'])
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
    st.subheader("Model evaluation")
    if 'model' in st.session_state:
        model = st.session_state['model']
        y = st.session_state['df']['y']
        y_pred = st.session_state['y_pred']
        n = len(y)
        p = 1 # number of predictors
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

def bc_model_assumptions_homoscedasticity():
    st.subheader("Model Assumptions - Homoscedasticity")
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
   
def bc_model_assumptions_normality():
    st.subheader("Model Assumptions - Normality of Residuals")
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
    st.header("Basic Calibration Page")     
    bc_section = st.sidebar.radio(
        "Navigate Basic Calibration",
        ["Import Data", "View Raw Data", "Calibration Plot", "Calibration Function", 
        "Model evaluation", "Model Assumptions - Homoscedasticity", "Model Assumptions - Normality of Residuals"]
    )

    if bc_section == "Import Data":
        bc_import_data()

    elif bc_section == "View Raw Data":
        bc_raw_data()

    elif bc_section == "Calibration Plot":
        bc_calibration_plot()

    elif bc_section == "Calibration Function":
        bc_calibration_function()

    elif bc_section == "Model evaluation":
        bc_model_evaluation()

    elif bc_section == "Model Assumptions - Homoscedasticity":
        bc_model_assumptions_homoscedasticity()

    elif bc_section == "Model Assumptions - Normality of Residuals":
        bc_model_assumptions_normality()
  

###############################################################
# Function to render improved calibration page
###############################################################

def render_improved_calibration():
    st.header("Improved Calibration Page")
    st.write("This is the Improved Calibration page!")

    with st.expander("Import Data"):
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
        x_label = st.text_input("Enter the label for x", "Concentration[mg/L]")
        y_label = st.text_input("Enter the label for y", "Peakarea")
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

            # Perform ordinary linear regression
            model_ordinary = sm.OLS(df['y'], sm.add_constant(df['x'])).fit()

            # Dictionary to store all models
            models = {
                'ordinary_linear_regression': model_ordinary
            }

            # Calculate weighted values
            df['x0.5'] = 1 / np.sqrt(df['x'])
            df['x1'] = 1 / df['x']
            df['x2'] = 1 / (df['x'] ** 2)
            df['y0.5'] = 1 / np.sqrt(df['y'])
            df['y1'] = 1 / df['y']
            df['y2'] = 1 / (df['y'] ** 2)
            df['sdy0.5'] = 1 / np.sqrt(df.groupby('x')['y'].transform('std'))
            df['sdy1'] = 1 / df.groupby('x')['y'].transform('std')
            df['sdy2'] = 1 / (df.groupby('x')['y'].transform('std') ** 2)
            df['sde0.5'] = 1 / np.abs(model_ordinary.resid) ** 0.5
            df['sde1'] = 1 / np.abs(model_ordinary.resid)
            df['sde2'] = 1 / (np.abs(model_ordinary.resid) ** 2)

            # Store weights in the models dictionary
            for weight_scheme in ['x0.5', 'x1', 'x2', 'y0.5', 'y1', 'y2', 'sdy0.5', 'sdy1', 'sdy2', 'sde0.5', 'sde1', 'sde2']:
                model = sm.WLS(df['y'], sm.add_constant(df['x']), weights=df[weight_scheme]).fit()
                models[weight_scheme] = model

            # Calculate relative errors and x_calc for each model
            re_results = []
            for model_name, model in models.items():
                x_calc = (df['y'] - model.params[0]) / model.params[1]
                relative_error = (x_calc - df['x']) / df['x']
                df[f'{model_name}_x_calc'] = x_calc
                df[f'{model_name}_re'] = relative_error
                re_results.append({
                    'Model': model_name,
                    'Relative Error': relative_error.mean()
                })

            # Calculate evaluation metrics for each model
            results = []
            for model_name, model in models.items():
                # Make predictions
                y_pred = model.predict(sm.add_constant(df['x']))

                # Calculate adjusted R-squared
                n = len(df['y'])
                p = 1  # number of predictors
                r_squared = model.rsquared
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

                # Calculate mean squared error (MSE)
                mse = np.mean((df['y'] - y_pred) ** 2)

                # Calculate root mean squared error (RMSE)
                rmse = np.sqrt(mse)

                # Calculate AIC
                aic = model.aic

                # Calculate BIC (also known as NIC in some contexts)
                bic = model.bic

                # Calculate SRE and MRE
                sre = np.abs(df[f'{model_name}_re']).sum()
                mre = np.abs(df[f'{model_name}_re']).mean()

                results.append({
                    'Model': model_name,
                    'Adjusted R-squared': adjusted_r_squared,
                    'SRE': sre,
                    'MRE': mre,
                    'Mean Squared Error (MSE)': mse,
                    'Root Mean Squared Error (RMSE)': rmse,
                    'AIC': aic,
                    'BIC (NIC)': bic
                })

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)

            # Find the model with the lowest MRE
            # Find the model with the lowest MRE
            min_mre_model_name = results_df.loc[results_df['MRE'].idxmin(), 'Model']
            min_mre_model = models[min_mre_model_name]

            # Expander - View raw data
            with st.expander("View raw data"):
                st.dataframe(df[['x', 'y']])

            # Expander - Improved Calibration Plot
            with st.expander("Improved Calibration Plot"):
                # Plot the data and the regression line
                fig, ax = plt.subplots()
                ax.plot(df['x'], df['y'], 'o', label='Data points')
                ax.plot(df['x'], min_mre_model.predict(sm.add_constant(df['x'])), '-', label='Regression line')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()

                # Display the plot in Streamlit
                st.pyplot(fig)

            # Expander - Improved Calibration Function
            with st.expander("Improved Calibration Function"):
                st.markdown("**Formula of fitted linear regression:**")
                st.write(f"{y_label} = {min_mre_model.params[1]} * {x_label} + {min_mre_model.params[0]}")

                # Show slope of the fitted regression line
                st.markdown("**Slope of the fitted regression line:**")
                st.write(min_mre_model.params[1])

                # Show intercept of the fitted regression line
                st.markdown("**Intercept of the fitted regression line:**")
                st.write(min_mre_model.params[0])

            # Expander - Calibration plots
            with st.expander("Calibration plots"):
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # OLS regression plot
                axes[0].scatter(df['x'], df['y'], label='Actual')
                axes[0].plot(df['x'], model_ordinary.predict(sm.add_constant(df['x'])), color='red',
                             label='OLS Regression')
                axes[0].set_xlabel(x_label)
                axes[0].set_ylabel(y_label)
                axes[0].set_title('OLS Regression Plot')
                axes[0].legend()

                # WLS regression plot with lowest MRE
                axes[1].scatter(df['x'], df['y'], label='Actual')
                axes[1].plot(df['x'], min_mre_model.predict(sm.add_constant(df['x'])), color='green',
                             label=f'WLS Regression ({min_mre_model_name})')
                axes[1].set_xlabel(x_label)
                axes[1].set_ylabel(y_label)
                axes[1].set_title(f'WLS Regression Plot ({min_mre_model_name})')
                axes[1].legend()

                st.pyplot(fig)

            # Expander - View model evaluation metrics
            with st.expander("View model evaluation metrics"):
                st.dataframe(results_df)

            # Expander - Model evaluation plots
            with st.expander("Model evaluation plots"):
                # Sort results dataframe based on the specified metric
                def sort_dataframe(metric, ascending=True):
                    sorted_df = results_df.sort_values(by=metric, ascending=ascending)
                    model_names = sorted_df['Model']
                    colors = ['red' if model == 'ordinary_linear_regression' else 'green' if
                              model == min_mre_model_name else 'blue' for model in model_names]
                    return sorted_df, model_names, colors

                # Define metrics to plot
                metrics_to_plot = ['Adjusted R-squared', 'SRE', 'MRE', 'Mean Squared Error (MSE)',
                                   'Root Mean Squared Error (RMSE)', 'AIC', 'BIC (NIC)']

                # Create subplots
                fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 6 * len(metrics_to_plot)))

                for i, metric in enumerate(metrics_to_plot):
                    # Sort dataframe based on the current metric
                    ascending = False if metric == 'Adjusted R-squared' else True
                    sorted_df, model_names, colors = sort_dataframe(metric, ascending=ascending)

                    # Plot the bar chart
                    ax = axes[i]
                    ax.bar(model_names, sorted_df[metric], color=colors)
                    ax.set_xlabel('Model')
                    ax.set_ylabel(metric)
                    ax.set_title(f'Model Comparison by {metric}')

                    # Set y-scale
                    if metric == 'Adjusted R-squared':
                        ax.set_yscale('log')

                    # Add horizontal grid lines
                    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

                plt.tight_layout()
                st.pyplot(fig)

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
