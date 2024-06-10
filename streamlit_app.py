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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.stats.diagnostic as smd

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

def render_basic_calibration():
    st.header("Basic Calibration Page")

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
            with st.expander("View raw data"):
                st.dataframe(df)
            
            # Perform linear regression
            X = np.array(x_values).reshape(-1, 1)
            y = np.array(y_values)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Plot the data and the regression line
            with st.expander("Calibration plot"):
                fig, ax = plt.subplots()
                ax.plot(df['x'], df['y'], 'o', label='Data points')
                ax.plot(df['x'], y_pred, '-', label='Regression line')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()
            
                # Display the plot in Streamlit
                st.pyplot(fig)

            # Display Calibration function section
            with st.expander("Calibration function"):
                st.markdown("**Formula of fitted linear regression:**")
                st.write(f"{y_label} = {model.coef_[0]} * {x_label} + {model.intercept_}")

                # Show slope of the fitted regression line
                st.markdown("**Slope of the fitted regression line:**")
                st.write(model.coef_[0])

                # Show intercept of the fitted regression line
                st.markdown("**Intercept of the fitted regression line:**")
                st.write(model.intercept_)

            # Display Model evaluation section
            with st.expander("Model evaluation"):
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
            with st.expander("Model assumptions - Homoscedasticity"):
                st.subheader("Model assumptions - Homoscedasticity")

                # Calculate residuals
                residuals = y - y_pred         
    
                # Conduct the Breusch-Pagan test
                fit = smf.ols('y_values ~ x_values', data=df).fit()
                names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']

                # Get the test result
                test_result = sms.het_breuschpagan(fit.resid, fit.model.exog)

                # Display the test result
                st.markdown("**Breusch-Pagan Test for Homoscedasticity**")

                # Get the test result
                test_result_bp = sms.het_breuschpagan(fit.resid, fit.model.exog)

                # Display Breusch-Pagan test statistics
                st.write("P-value:", test_result_bp[1])
                st.write("Degrees of freedom:", test_result_bp[2])
                st.write("F-statistic:", test_result_bp[3])

                # Interpret Breusch-Pagan test results
                if test_result_bp[1] > 0.05:
                    st.write("The Breusch-Pagan test for Homoscedasticity is not significant (p > 0.05), indicating that the residuals have constant variance (homoscedasticity).")
                else:
                    st.write("The Breusch-Pagan test for Homoscedasticity is significant (p <= 0.05), suggesting that the residuals may not have constant variance (heteroscedasticity).")
                    st.write("Considering the potential violation of homoscedasticity assumption, further diagnostics or transformations may be necessary.")
                    st.write("The distribution of residuals and the test result should be visually verified with the following residual plots.")

                # Display Residual plots subsection
                st.markdown("**Residual plots**")

                # Residuals vs x plot
                fig_resid, ax_resid = plt.subplots()
                ax_resid.scatter(df['x'], residuals)
                ax_resid.axhline(y=0, color='black', linestyle='--')
                ax_resid.set_xlabel(x_label)
                ax_resid.set_ylabel('Residuals')
                ax_resid.set_title(f"Residuals vs {x_label}")
                st.pyplot(fig_resid)
                
                # Calculate standardized residuals
                standardized_residuals = residuals / np.std(residuals)

                # square root of the absolute value of standardized residuals vs x plot
                fig_sqrt_std_res, ax_sqrt_std_res = plt.subplots()
                ax_sqrt_std_res.scatter(df['x'], np.sqrt(np.abs(standardized_residuals)))
                ax_sqrt_std_res.set_xlabel(x_label)
                ax_sqrt_std_res.set_ylabel('sqrt(|Standardized Residuals|)')
                ax_sqrt_std_res.set_title(f"sqrt(|Standardized Residuals|) vs {x_label}")
                st.pyplot(fig_sqrt_std_res)

                # Calculate relative error
                x_calc = (y - model.intercept_) / model.coef_[0]
                relative_error = (x_calc - df['x']) / df['x']

                # Relative Error vs x plot
                fig_rel_error, ax_rel_error = plt.subplots()
                ax_rel_error.scatter(df['x'], relative_error)
                ax_rel_error.axhline(y=0, color='black', linestyle='--')
                ax_rel_error.set_xlabel(x_label)
                ax_rel_error.set_ylabel('Relative Error')
                ax_rel_error.set_title(f"Relative Error vs {x_label}")
                st.pyplot(fig_rel_error)
            
            # Model assumptions - Normality of Residuals
            with st.expander("Model assumptions - Normality of Residuals"):
                st.subheader("Model assumptions - Normality of Residuals")
                st.markdown("**Shapiro-Wilk Test for Normality of Residuals**")

                test_statistic_sw, p_value_sw = shapiro(fit.resid)

                # Display Shapiro-Wilk test statistics
                st.write("W-statistic:", test_statistic_sw)
                st.write("P-value:", p_value_sw)

                # Interpret Shapiro-Wilk test results
                if p_value_sw > 0.05:
                    st.write("The Shapiro-Wilk test for normality of residuals is not significant (p > 0.05), indicating that the residuals are normally distributed.")
                else:
                    st.write("The Shapiro-Wilk test for normality of residuals is significant (p <= 0.05), suggesting that the residuals may not be normally distributed.")
                    st.write("Considering the non-normal distribution of residuals, weighted calibration may be advisable.")
                    st.write("The distribution of residuals and the test result should be visually verified with the following QQ plot.")

                st.markdown("**Quantile-Quantile Plot**")

                # Generate QQ plot
                fig_qq, ax_qq = plt.subplots()
                sm.qqplot(standardized_residuals, line ='45', ax=ax_qq)
                ax_qq.set_xlabel('Theoretical quantiles')
                ax_qq.set_ylabel('Standardized residuals')
                ax_qq.set_title('QQ Plot of Standardized Residuals')
                st.pyplot(fig_qq)

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
