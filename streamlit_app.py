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
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr, ttest_ind
import statsmodels.api as sm

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
    st.subheader("Customize Calibration Plot")
    if 'df' in st.session_state and 'model' in st.session_state:
        df = st.session_state['df']
        model = st.session_state['model']
        y_pred = st.session_state['y_pred']
        x_label_default = st.session_state['x_label']
        y_label_default = st.session_state['y_label']

        with st.expander("Labels"):
            # Layout für Checkboxen und Eingabefelder in einem 2x2-Grid
            col1, col2 = st.columns(2)

            with col1:
                show_r_squared = st.checkbox("Show adjusted R-squared", value=True)
                if show_r_squared:
                    r_squared_decimals = st.number_input("Decimal places for R-squared", min_value=1, max_value=10, value=3, key="r2_decimals")

                x_label = st.text_input("X-axis label:", x_label_default)

            with col2:
                show_equation = st.checkbox("Show calibration equation", value=True)
                if show_equation:
                    equation_decimals = st.number_input("Decimal places for equation", min_value=1, max_value=10, value=3, key="eq_decimals")

                y_label = st.text_input("Y-axis label:", y_label_default)

            title = st.text_input("Plot title:", "Calibration Plot")

        with st.expander("Size"):
            # Slider für die Breite und Höhe des Plots
            width = st.slider("Plot width (cm)", min_value=3, max_value=24, value=14)
            height = st.slider("Plot height (cm)", min_value=3, max_value=24, value=12)

        with st.expander("Colors and Shapes"):
            # Auswahl für Farben und Formen
            col1, col2 = st.columns(2)

            with col1:
                point_color = st.color_picker("Pick a color for points", "#1f77b4")
                line_color = st.color_picker("Pick a color for the line", "#ff7f0e")

            with col2:
                # Verfügbare Marker
                markers = {
                    'Circle': 'o',
                    'Square': 's',
                    'Diamond': 'D',
                    'Up Triangle': '^',
                    'Down Triangle': 'v',
                    'Left Triangle': '<',
                    'Right Triangle': '>',
                    'Pentagon': 'p',
                    'Hexagon': 'h'
                }

                marker_labels = list(markers.keys())
                marker_display = ["● Circle", "■ Square", "◆ Diamond", "▲ Up Triangle", "▼ Down Triangle", "◀ Left Triangle", "▶ Right Triangle", "⬟ Pentagon", "⬢ Hexagon"]
                selected_marker_label = st.selectbox("Select marker style for points", marker_display)
                selected_marker = markers[marker_labels[marker_display.index(selected_marker_label)]]

                line_styles = {
                    'Solid': '-',
                    'Dashed': '--',
                    'Dash-dot': '-.',
                    'Dotted': ':'
                }

                line_style_labels = list(line_styles.keys())
                line_style_display = ["Solid", "Dashed", "Dash-dot", "Dotted"]
                selected_line_style_label = st.selectbox("Select line style", line_style_display)
                selected_line_style = line_styles[selected_line_style_label]

        # Berechnung des Konfidenzintervalls
        X = df[['x']].values
        se = np.sqrt(np.sum((y_pred - df['y'])**2) / (len(df) - 2))
        t_val = stats.t.ppf(1 - 0.025, df=len(df) - 2)
        ci = t_val * se * np.sqrt(1 / len(df) + (X - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
        ci = ci.flatten()  # Sicherstellen, dass ci 1-dimensional ist

        # Plot erstellen
        fig, ax = plt.subplots(figsize=(width / 2.54, height / 2.54))  # Konvertieren von cm zu Zoll
        ax.plot(df['x'], df['y'], marker=selected_marker, linestyle='None', color=point_color)  # Datenpunkte
        ax.plot(df['x'], y_pred, linestyle=selected_line_style, color=line_color)  # Regressionslinie
        ax.fill_between(df['x'], y_pred - ci, y_pred + ci, color=line_color, alpha=0.1)  # Konfidenzintervall

        # Achsenbeschriftungen
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Berechnung des adjustierten R-Quadrats
        n = len(df)
        p = 1  # Anzahl der Prädiktoren
        r_squared = model.score(X, df['y'])
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        # Anzeige der Kalibriergleichung und des adjustierten R-Quadrats basierend auf Benutzeroptionen
        equation_text = ""
        if show_r_squared:
            equation_text += f"adj. R² = {adjusted_r_squared:.{r_squared_decimals}f}\n"
        if show_equation:
            equation_text += f"y = {model.coef_[0]:.{equation_decimals}f} * x + {model.intercept_:.{equation_decimals}f}"

        if equation_text:
            ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        st.pyplot(fig)
    else:
        st.error("No data or model available. Please import data first.")


def lsqfity(X, Y):
    """
    Calculate a "MODEL-1" least squares fit.

    The line is fit by MINIMIZING the residuals in Y only.

    The equation of the line is:     Y = my * X + by.

    Equations are from Bevington & Robinson (1992)
    Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
    pp: 104, 108-109, 199.

    Data are input and output as follows:

    my, by, ry, smy, sby = lsqfity(X,Y)
    X     =    x data (vector)
    Y     =    y data (vector)
    my    =    slope
    by    =    y-intercept
    ry    =    correlation coefficient
    smy   =    standard deviation of the slope
    sby   =    standard deviation of the y-intercept

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)

    # Calculate the sums.

    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sx2 = np.sum(X ** 2)
    Sxy = np.sum(X * Y)
    Sy2 = np.sum(Y ** 2)

    # Calculate re-used expressions.
    num = n * Sxy - Sx * Sy
    den = n * Sx2 - Sx ** 2

    # Calculate my, by, ry, s2, smy and sby.
    my = num / den
    by = (Sx2 * Sy - Sx * Sxy) / den
    ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

    diff = Y - by - my * X

    s2 = np.sum(diff * diff) / (n - 2)
    smy = np.sqrt(n * s2 / den)
    sby = np.sqrt(Sx2 * s2 / den)

    return my, by, ry, smy, sby

def calculate_lod_loq(sigma, slope):
    lod = 3.3 * sigma / slope
    loq = 10 * sigma / slope
    return lod, loq

def bc_calibration_metrics():
    if 'model' in st.session_state:
        model = st.session_state['model']
        
        with st.expander("Calibration Coefficients", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Slope of the fitted regression line:")
            with col2:
                st.write(model.coef_[0])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Intercept of the fitted regression line:")
            with col2:
                st.write(model.intercept_)
        
        with st.expander("Goodness of Fit", expanded=True):
            y = st.session_state['df']['y']
            y_pred = st.session_state['y_pred']
            n = len(y)
            p = 1  # number of predictors
            r_squared = model.score(np.array(st.session_state['df']['x']).reshape(-1, 1), y)
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Adjusted R-squared:")
            with col2:
                st.write(adjusted_r_squared)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Mean Squared Error (MSE):")
            with col2:
                st.write(mse)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Root Mean Squared Error (RMSE):")
            with col2:
                st.write(rmse)
        
        with st.expander("Sensitivity - LOD and LOQ", expanded=True):
            st.markdown("""
                According to the International Conference on Harmonization (ICH) guidelines, the limit of detection (LOD) 
                is determined using the relation <b>LOD = 3.3 * (σ/S)</b> where <b>σ</b> is the standard deviation 
                of the response and <b>S</b> is the slope of the calibration curve. The standard deviation of the response 
                can be obtained either by measuring the standard deviation of the blank response, by calculating the 
                residual standard deviation of the regression line, by calculating the standard deviation of the y-intercept 
                of the regression line, or by calculating <b>S<sub>y/x</sub></b>, i.e., the standard error of the estimate.
            """, unsafe_allow_html=True)

            X = st.session_state['df']['x']
            Y = st.session_state['df']['y']
            slope = model.coef_[0]

            method = st.radio(
                "Select method to estimate σ (standard deviation):",
                ("Standard deviation of blank response", 
                 "Residual standard deviation of the regression line",
                 "Standard deviation of the y-intercept of the regression line",
                 "Standard error of estimate"))

            if method == "Standard deviation of blank response":
                blank_y_values = st.text_area("Enter y-values of blank response (comma separated):")
                if blank_y_values:
                    blank_y = [float(i) for i in blank_y_values.split(',') if i.strip()]
                    if len(blank_y) >= 5:
                        sigma = np.std(blank_y)
                    else:
                        st.error("Please enter at least 5 blank y-values.")
                else:
                    st.error("Please enter the y-values of blank response.")

            elif method == "Residual standard deviation of the regression line":
                residuals = Y - model.predict(X.values.reshape(-1, 1))
                sigma = np.std(residuals)

            elif method == "Standard deviation of the y-intercept of the regression line":
                _, _, _, _, sby = lsqfity(X, Y)
                sigma = sby

            elif method == "Standard error of estimate":
                residuals = Y - model.predict(X.values.reshape(-1, 1))
                sigma = np.sqrt(np.sum(residuals ** 2) / (len(Y) - 2))

            if st.button("Estimate LOD and LOQ"):
                lod, loq = calculate_lod_loq(sigma, slope)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Limit of Detection (LOD):")
                with col2:
                    st.write(lod)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Limit of Quantification (LOQ):")
                with col2:
                    st.write(loq)
    else:
        st.error("No model available. Please import data first.")


def bc_model_assumptions():
    st.header("Model Assumptions")
    
    st.subheader("Background")
    st.write("""
        Calibration functions are linear regression models, and these models need to meet certain assumptions to ensure their validity.
        Two critical assumptions are homoscedasticity and normality of residuals. Homoscedasticity means that the variance of the residuals 
        is constant across all levels of the independent variable, ensuring that the model's predictions are reliable and independent of the value of x.
        Normality of residuals indicates that the residuals (differences between observed and predicted values) follow a normal distribution, which is crucial 
        for making valid inferences from the model.
    """)

    if 'df' in st.session_state and 'model' in st.session_state:
        df = st.session_state['df']
        model = st.session_state['model']
        residuals = st.session_state['df']['y'] - st.session_state['y_pred']
        df['residuals'] = residuals
        fit = smf.ols('y ~ x', data=df).fit()
        test_result_bp = sm_diagnostic.het_breuschpagan(fit.resid, fit.model.exog)
        shapiro_test = stats.shapiro(residuals)

        st.subheader("Testing of Model Assumptions")
        
        with st.expander("Breusch-Pagan Test for Homoscedasticity", expanded=True):
            bp_significant = test_result_bp[1] <= 0.05
            bp_text = f"<span style='background-color: {'#90EE90' if not bp_significant else '#FFCCCB'}'>{'Homoscedasticity' if not bp_significant else 'Heteroscedasticity'}</span>"
            p_value_bp = "<0.001" if test_result_bp[1] < 0.001 else f"{test_result_bp[1]:.3f}"
            f_value_bp = "<0.001" if test_result_bp[3] < 0.001 else f"{test_result_bp[3]:.3f}"
            st.markdown(f"Breusch-Pagan test suggests {bp_text} (p = {p_value_bp}, F = {f_value_bp}).", unsafe_allow_html=True)

        with st.expander("Shapiro-Wilk Test for Normality of Residuals", expanded=True):
            sw_significant = shapiro_test[1] <= 0.05
            sw_text = f"<span style='background-color: {'#90EE90' if not sw_significant else '#FFCCCB'}'>{'Normality' if not sw_significant else 'Non-Normality'}</span>"
            p_value_sw = "<0.001" if shapiro_test[1] < 0.001 else f"{shapiro_test[1]:.3f}"
            w_value_sw = "<0.001" if shapiro_test[0] < 0.001 else f"{shapiro_test[0]:.3f}"
            st.markdown(f"Shapiro-Wilk Test suggests {sw_text} (p = {p_value_sw}, W = {w_value_sw}).", unsafe_allow_html=True)

        st.subheader("Diagnostic Plots")

        with st.expander("Customize Residual Plot", expanded=False):
            st.markdown("**Labels**")
            x_label_default = st.session_state['x_label']
            y_label_default = "Residuals"
            title_default = "Residual Plot"

            col1, col2 = st.columns(2)

            with col1:
                show_bp_p = st.checkbox("Show p-value (Breusch-Pagan)", value=True)
                if show_bp_p:
                    bp_p_decimals = st.number_input("Decimal places for p-value (Breusch-Pagan)", min_value=1, max_value=10, value=3, key="bp_p_decimals")

                x_label = st.text_input("X-axis label:", x_label_default, key="resid_x_label")

            with col2:
                show_bp_f = st.checkbox("Show F-value (Breusch-Pagan)", value=True)
                if show_bp_f:
                    bp_f_decimals = st.number_input("Decimal places for F-value (Breusch-Pagan)", min_value=1, max_value=10, value=3, key="bp_f_decimals")

                y_label = st.text_input("Y-axis label:", y_label_default, key="resid_y_label")

            title = st.text_input("Plot title:", title_default, key="resid_title")

            st.markdown("**Size**")
            width = st.slider("Plot width (cm)", min_value=3, max_value=24, value=14, key="resid_width")
            height = st.slider("Plot height (cm)", min_value=3, max_value=24, value=12, key="resid_height")

            st.markdown("**Colors and Shapes**")
            col1, col2 = st.columns(2)

            with col1:
                point_color = st.color_picker("Pick a color for points", "#1f77b4", key="resid_point_color")
                line_color = st.color_picker("Pick a color for the line", "#ff7f0e", key="resid_line_color")

            with col2:
                markers = {
                    'Circle': 'circle',
                    'Square': 'square',
                    'Diamond': 'diamond',
                    'Up Triangle': 'triangle-up',
                    'Down Triangle': 'triangle-down',
                    'Left Triangle': 'triangle-left',
                    'Right Triangle': 'triangle-right',
                    'Pentagon': 'pentagon',
                    'Hexagon': 'hexagon'
                }

                marker_labels = list(markers.keys())
                marker_display = ["● Circle", "■ Square", "◆ Diamond", "▲ Up Triangle", "▼ Down Triangle", "◀ Left Triangle", "▶ Right Triangle", "⬟ Pentagon", "⬢ Hexagon"]
                selected_marker_label = st.selectbox("Select marker style for points", marker_display, key="resid_marker_display")
                selected_marker = markers[marker_labels[marker_display.index(selected_marker_label)]]

                line_styles = {
                    'Solid': 'solid',
                    'Dashed': 'dash',
                    'Dash-dot': 'dashdot',
                    'Dotted': 'dot'
                }

                line_style_labels = list(line_styles.keys())
                line_style_display = ["Solid", "Dashed", "Dash-dot", "Dotted"]
                selected_line_style_label = st.selectbox("Select line style", line_style_display, key="resid_line_style_display")
                selected_line_style = line_styles[selected_line_style_label]

        fig_resid = px.scatter(x=df['x'], y=residuals, labels={'x': x_label, 'y': y_label}, title=title)
        fig_resid.update_traces(marker=dict(color=point_color, symbol=selected_marker))
        fig_resid.add_shape(type='line', x0=df['x'].min(), y0=0, x1=df['x'].max(), y1=0, line=dict(color='black', dash='dash'))
        
        mean_residuals = df.groupby('x')['residuals'].mean().reset_index()
        mean_residuals = mean_residuals.sort_values('x')
        fig_resid.add_trace(go.Scatter(x=mean_residuals['x'], y=mean_residuals['residuals'], mode='lines', name='Mean Residuals', line=dict(color=line_color, dash=selected_line_style)))

        annotation_text = ""
        if show_bp_p:
            bp_p_threshold = 10**(-bp_p_decimals)
            if test_result_bp[1] < bp_p_threshold:
                annotation_text += f"p (Breusch-Pagan) < {bp_p_threshold:.{bp_p_decimals}f}<br>"
            else:
                annotation_text += f"p (Breusch-Pagan) = {test_result_bp[1]:.{bp_p_decimals}f}<br>"
        if show_bp_f:
            bp_f_threshold = 10**(-bp_f_decimals)
            if test_result_bp[3] < bp_f_threshold:
                annotation_text += f"F (Breusch-Pagan) < {bp_f_threshold:.{bp_f_decimals}f}"
            else:
                annotation_text += f"F (Breusch-Pagan) = {test_result_bp[3]:.{bp_f_decimals}f}"
        
        if show_bp_p or show_bp_f:
            fig_resid.add_annotation(
                x=0.95, y=0.05, showarrow=False, text=annotation_text, xref="paper", yref="paper", 
                xanchor="right", yanchor="bottom", align="right", bgcolor="white", font=dict(color='black')
            )

        fig_resid.update_layout(showlegend=False)
        st.plotly_chart(fig_resid, use_container_width=True)

        with st.expander("Customize Q-Q Plot", expanded=False):
            st.markdown("**Labels**")
            x_label_default = "Theoretical Quantiles"
            y_label_default = "Sample Quantiles"
            title_default = "Q-Q plot of residuals"

            col1, col2 = st.columns(2)

            with col1:
                show_sw_p = st.checkbox("Show p-value (Shapiro-Wilk)", value=True)
                if show_sw_p:
                    sw_p_decimals = st.number_input("Decimal places for p-value (Shapiro-Wilk)", min_value=1, max_value=10, value=3, key="sw_p_decimals")

                x_label = st.text_input("X-axis label:", x_label_default, key="qq_x_label")

            with col2:
                show_sw_w = st.checkbox("Show W-value (Shapiro-Wilk)", value=True)
                if show_sw_w:
                    sw_w_decimals = st.number_input("Decimal places for W-value (Shapiro-Wilk)", min_value=1, max_value=10, value=3, key="sw_w_decimals")

                y_label = st.text_input("Y-axis label:", y_label_default, key="qq_y_label")

            title = st.text_input("Plot title:", title_default, key="qq_title")

            st.markdown("**Size**")
            width = st.slider("Plot width (cm)", min_value=3, max_value=24, value=14, key="qq_width")
            height = st.slider("Plot height (cm)", min_value=3, max_value=24, value=12, key="qq_height")

            st.markdown("**Colors and Shapes**")
            col1, col2 = st.columns(2)

            with col1:
                point_color = st.color_picker("Pick a color for points", "#1f77b4", key="qq_point_color")
                line_color = st.color_picker("Pick a color for the line", "#ff7f0e", key="qq_line_color")

            with col2:
                markers = {
                    'Circle': 'circle',
                    'Square': 'square',
                    'Diamond': 'diamond',
                    'Up Triangle': 'triangle-up',
                    'Down Triangle': 'triangle-down',
                    'Left Triangle': 'triangle-left',
                    'Right Triangle': 'triangle-right',
                    'Pentagon': 'pentagon',
                    'Hexagon': 'hexagon'
                }

                marker_labels = list(markers.keys())
                marker_display = ["● Circle", "■ Square", "◆ Diamond", "▲ Up Triangle", "▼ Down Triangle", "◀ Left Triangle", "▶ Right Triangle", "⬟ Pentagon", "⬢ Hexagon"]
                selected_marker_label = st.selectbox("Select marker style for points", marker_display, key="qq_marker_display")
                selected_marker = markers[marker_labels[marker_display.index(selected_marker_label)]]

                line_styles = {
                    'Solid': 'solid',
                    'Dashed': 'dash',
                    'Dash-dot': 'dashdot',
                    'Dotted': 'dot'
                }

                line_style_labels = list(line_styles.keys())
                line_style_display = ["Solid", "Dashed", "Dash-dot", "Dotted"]
                selected_line_style_label = st.selectbox("Select line style", line_style_display, key="qq_line_style_display")
                selected_line_style = line_styles[selected_line_style_label]

        qq_plot = sm.qqplot(residuals, line='s')
        qq_plot_fig = go.Figure()
        qq_plot_data = qq_plot.gca().get_lines()
        
        qq_x = qq_plot_data[0].get_xdata()
        qq_y = qq_plot_data[0].get_ydata()
        qq_line_x = qq_plot_data[1].get_xdata()
        qq_line_y = qq_plot_data[1].get_ydata()
        
        qq_plot_fig.add_trace(go.Scatter(x=qq_x, y=qq_y, mode='markers', name='Q-Q Plot', marker=dict(color=point_color, symbol=selected_marker)))
        qq_plot_fig.add_trace(go.Scatter(x=qq_line_x, y=qq_line_y, mode='lines', name='Q-Q Line', line=dict(color=line_color, dash=selected_line_style)))

        annotation_text = ""
        if show_sw_p:
            sw_p_threshold = 10**(-sw_p_decimals)
            if shapiro_test[1] < sw_p_threshold:
                annotation_text += f"p (Shapiro-Wilk) < {sw_p_threshold:.{sw_p_decimals}f}<br>"
            else:
                annotation_text += f"p (Shapiro-Wilk) = {shapiro_test[1]:.{sw_p_decimals}f}<br>"
        if show_sw_w:
            sw_w_threshold = 10**(-sw_w_decimals)
            if shapiro_test[0] < sw_w_threshold:
                annotation_text += f"W (Shapiro-Wilk) < {sw_w_threshold:.{sw_w_decimals}f}"
            else:
                annotation_text += f"W (Shapiro-Wilk) = {shapiro_test[0]:.{sw_w_decimals}f}"
        
        if show_sw_p or show_sw_w:
            qq_plot_fig.add_annotation(
                x=0.95, y=0.05, showarrow=False, text=annotation_text, xref="paper", yref="paper", 
                xanchor="right", yanchor="bottom", align="right", bgcolor="white", font=dict(color='black')
            )

        qq_plot_fig.update_layout(showlegend=False, title=title, xaxis_title=x_label, yaxis_title=y_label)
        st.plotly_chart(qq_plot_fig, use_container_width=True)

        st.subheader("Conclusion")
        
        if bp_significant or sw_significant:
            st.markdown("""
                <div style='background-color: #FFCCCB; padding: 10px; border-radius: 5px;'>
                Based on the null hypothesis significance tests, the assumptions for a linear calibration model are not met. 
                If these assumptions are violated, the accuracy of the calibration line can be significantly affected, especially in the lower range of the calibration.
                To determine whether the accuracy of the model decreases at the lower end of the calibration range, it is important to examine the relative errors (percentage difference from the true x-values) at smaller x-values. 
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #90EE90; padding: 10px; border-radius: 5px;'>
                Based on the null hypothesis significance tests, the assumptions for a linear calibration model are met. 
                However, it is essential to visually verify these results with the residual plot and Q-Q plot.
                To determine whether the accuracy of the model decreases at the lower end of the calibration range, it is important to examine the relative errors (percentage difference from the true x-values) at smaller x-values.
                </div>
            """, unsafe_allow_html=True)
        
    else:
        st.error("No data or model available. Please import data first.")


def bc_relative_errors():
    st.header("Relative Errors")
    
    if 'df' in st.session_state and 'model' in st.session_state:
        df = st.session_state['df']
        model = st.session_state['model']
        
        x_calc = (df['y'] - model.intercept_) / model.coef_[0]
        relative_error = 100 * (x_calc - df['x']) / df['x']
        df['relative_error'] = relative_error

        sre = np.abs(df['relative_error']).sum()
        mre = np.abs(df['relative_error']).mean()
        pearson_corr, _ = pearsonr(df['x'], np.abs(df['relative_error']))
        spearman_corr, _ = spearmanr(df['x'], np.abs(df['relative_error']))
        
        lower_half_error = np.abs(df[df['x'] <= df['x'].median()]['relative_error'])
        upper_half_error = np.abs(df[df['x'] > df['x'].median()]['relative_error'])

        lower_mean = lower_half_error.mean()
        upper_mean = upper_half_error.mean()

        t_stat, p_value = ttest_ind(lower_half_error, upper_half_error, equal_var=False)

        if abs(pearson_corr) >= 0.50:
            pearson_text = "strong"
        elif 0.30 <= abs(pearson_corr) < 0.50:
            pearson_text = "moderate"
        else:
            pearson_text = "low"

        if abs(spearman_corr) >= 0.50:
            spearman_text = "strong"
        elif 0.30 <= abs(spearman_corr) < 0.50:
            spearman_text = "moderate"
        else:
            spearman_text = "low"

        if p_value > 0.05:
            significance_text = f"""
            The relative error in the lower half of the calibration is {lower_mean:.2f}% and {upper_mean:.2f}% in the upper half of the calibration. 
            There is no significant difference in the relative errors between the lower and upper halves of the calibration (Two-Sample T-test, t = {t_stat:.3f}, p = {p_value:.3f}).
            """
        else:
            if lower_mean > upper_mean:
                significance_text = f"""
                The mean relative error in the lower half of the calibration ({lower_mean:.2f}%) is significantly higher than in the upper half ({upper_mean:.2f}%) (Two-Sample T-test, t = {t_stat:.3f}, p = {p_value:.3f}).
                """
            else:
                significance_text = f"""
                The mean relative error in the lower half of the calibration ({lower_mean:.2f}%) is significantly lower than in the upper half ({upper_mean:.2f}%) (Two-Sample T-test, t = {t_stat:.3f}, p = {p_value:.3f}).
                """

        st.subheader("Background")
        st.write("""
        The analysis of relative errors is essential in calibration studies to understand how the model performs across different concentration levels.
        By examining the relative errors, we can assess whether the model's predictions are consistently accurate or if there are specific ranges where 
        the model tends to overestimate or underestimate the actual values.
        """)

        st.subheader("Analysis of Relative Errors")

        with st.expander("1. How much do the estimated x-values deviate from the actual x-values?", expanded=False):
            st.write("""
            - With the basic calibration model, the estimated x-value deviates by ±30% from the actual x-value.
            """)

        with st.expander("2. Is the calibration function less accurate in the lower range of the calibration?", expanded=False):
            st.write(f"""
            - {significance_text.strip()}
            """)

        with st.expander("3. Is there a linear relationship between the size of the x-value and the size of the relative estimation error?", expanded=False):
            st.write(f"""
            - The Pearson correlation of {pearson_corr:.3f} indicates a {pearson_text} linear relationship between x and the relative estimation error for x.
            """)

        with st.expander("4. Is there a non-linear relationship between the size of the x-value and the size of the relative estimation error?", expanded=False):
            st.write(f"""
            - The Spearman correlation of {spearman_corr:.3f} indicates a {spearman_text} non-linear relationship between x and the relative estimation error for x.
            """)

        st.subheader("Relative Error Plot")
        
        with st.expander("Labels", expanded=False):
            x_label_default = st.session_state['x_label']
            y_label_default = "Relative Error (%)"
            title_default = "Relative Error Plot"

            col1, col2 = st.columns(2)

            with col1:
                show_sre = st.checkbox("Show SRE", value=True)
                if show_sre:
                    sre_decimals = st.number_input("Decimal places for SRE", min_value=1, max_value=10, value=1, key="sre_decimals")

                x_label = st.text_input("X-axis label:", x_label_default)

            with col2:
                show_mre = st.checkbox("Show MRE", value=True)
                if show_mre:
                    mre_decimals = st.number_input("Decimal places for MRE", min_value=1, max_value=10, value=1, key="mre_decimals")

                y_label = st.text_input("Y-axis label:", y_label_default)

            title = st.text_input("Plot title:", title_default)

        with st.expander("Size", expanded=False):
            width = st.slider("Plot width (cm)", min_value=3, max_value=24, value=14)
            height = st.slider("Plot height (cm)", min_value=3, max_value=24, value=12)

        with st.expander("Colors and Shapes", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                point_color = st.color_picker("Pick a color for points", "#1f77b4")
                line_color = st.color_picker("Pick a color for the line", "#ff7f0e")

            with col2:
                markers = {
                    'Circle': 'circle',
                    'Square': 'square',
                    'Diamond': 'diamond',
                    'Up Triangle': 'triangle-up',
                    'Down Triangle': 'triangle-down',
                    'Left Triangle': 'triangle-left',
                    'Right Triangle': 'triangle-right',
                    'Pentagon': 'pentagon',
                    'Hexagon': 'hexagon'
                }

                marker_labels = list(markers.keys())
                marker_display = ["● Circle", "■ Square", "◆ Diamond", "▲ Up Triangle", "▼ Down Triangle", "◀ Left Triangle", "▶ Right Triangle", "⬟ Pentagon", "⬢ Hexagon"]
                selected_marker_label = st.selectbox("Select marker style for points", marker_display)
                selected_marker = markers[marker_labels[marker_display.index(selected_marker_label)]]

                line_styles = {
                    'Solid': 'solid',
                    'Dashed': 'dash',
                    'Dash-dot': 'dashdot',
                    'Dotted': 'dot'
                }

                line_style_labels = list(line_styles.keys())
                line_style_display = ["Solid", "Dashed", "Dash-dot", "Dotted"]
                selected_line_style_label = st.selectbox("Select line style", line_style_display)
                selected_line_style = line_styles[selected_line_style_label]

        fig = px.scatter(x=df['x'], y=relative_error, labels={'x': x_label, 'y': y_label}, title=title)
        fig.update_traces(marker=dict(color=point_color, symbol=selected_marker))
        fig.add_shape(type='line', x0=df['x'].min(), y0=0, x1=df['x'].max(), y1=0, line=dict(color='black', dash='dash'))
        
        mean_relative_errors = df.groupby('x')['relative_error'].mean().reset_index()
        mean_relative_errors = mean_relative_errors.sort_values('x')
        fig.add_trace(go.Scatter(x=mean_relative_errors['x'], y=mean_relative_errors['relative_error'], mode='lines', name='Mean Relative Error', line=dict(color=line_color, dash=selected_line_style)))

        if show_sre or show_mre:
            annotation_text = ""
            if show_sre:
                annotation_text += f"SRE = {sre:.{sre_decimals}f}%<br>"
            if show_mre:
                annotation_text += f"MRE = {mre:.{mre_decimals}f}%"
            
            # Add annotation in the lower right corner with padding
            fig.add_annotation(
                x=0.95, y=0.1, showarrow=False, text=annotation_text, xref="paper", yref="paper", 
                xanchor="right", yanchor="bottom", align="right", bgcolor="white", font=dict(color='black')
            )

        fig.update_layout(showlegend=False)  # Remove the legend

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Conclusion")
        
        if abs(pearson_corr) >= 0.30 or abs(spearman_corr) >= 0.30 or p_value <= 0.05:
            st.markdown("""
            <div style='background-color: #FFCCCB; padding: 10px; border-radius: 5px;'>
            It seems that the precision in the lower range of the calibration is negatively affected. The improved calibration is strongly recommended!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #90EE90; padding: 10px; border-radius: 5px;'>
            There is no indication that the precision in the lower range of the calibration is negatively affected.
            However, it is still recommended to try the improved calibration to further enhance the overall performance of the calibration.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("No data or model available. Please import data first.")

def render_basic_calibration():
    if st.session_state.get('data_imported'):
        if st.session_state.get('model_trained'):
            bc_section = st.sidebar.radio(
                "",
                ["Import Data", "View Data", "Calibration Plot", "Calibration Metrics", 
                "Model Assumptions", "Relative Errors"],
                index=["Import Data", "View Data", "Calibration Plot", "Calibration Metrics", 
                       "Model Assumptions", "Relative Errors"].index(st.session_state.get('current_section', "Import Data"))
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

    elif bc_section == "Calibration Metrics":
        bc_calibration_metrics()

    elif bc_section == "Model Assumptions":
        bc_model_assumptions()
    
    elif bc_section == "Relative Errors":
        bc_relative_errors()

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
