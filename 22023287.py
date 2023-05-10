# -*- coding: utf-8 -*-
"""
Created on Wed May 10 03:29:11 2023

@author: BINEESHA BABY
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.stats


def heatmap_corr(df, size=6, title=None, title_fontweight='bold'):
    """
    Create a heatmap of the correlation matrix for a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        size (int): Size of the heatmap figure.
        title (str): Title of the heatmap plot.
        title_fontweight (str): Font weight of the title.

    Returns:
        None
    """
    corr = df.corr()

    plt.figure(figsize=(size, size))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    if title:
        plt.title(title, fontweight=title_fontweight)

    plt.show()


def backscale(arr, df_min, df_max):
    """
    Backscale the cluster centers using the minimum and maximum values of the original DataFrame.

    Args:
        arr (np.ndarray): Array of cluster centers.
        df_min (pd.Series): Minimum values of the original DataFrame.
        df_max (pd.Series): Maximum values of the original DataFrame.

    Returns:
        np.ndarray: Backscaled cluster centers.
    """
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]
    return arr


def get_diff_entries(df1, df2, column):
    """
    Compare mismatching entries in a specific column between two DataFrames.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        column (str): Column to compare.

    Returns:
        list: List of mismatching entries.
    """
    try:
        df_out = pd.merge(df1, df2, on=column, how="outer")
        df_in = pd.merge(df1, df2, on=column, how="inner")
        df_in["exists"] = "Y"
        df_merge = pd.merge(df_out, df_in, on=column, how="outer")
        df_diff = df_merge[(df_merge["exists"] != "Y")]
        diff_list = df_diff[column].to_list()
        return diff_list
    except (KeyError, ValueError) as e:
        print(f"Error occurred while comparing entries: {e}")
        return []


def fit_curve(x, y, model_func):
    """
    Fit a curve to the data.

    Args:
        x (np.ndarray): Input x values.
        y (np.ndarray): Input y values.
        model_func (callable): Model function to fit.

    Returns:
        tuple: Tuple containing the best-fit parameters and the covariance matrix.
    """
    try:
        popt, pcov = curve_fit(model_func, x, y)
        return popt, pcov
    except RuntimeError as e:
        print(f"Error occurred while fitting a curve to the data: {e}")
        return None, None


def linear_func(x, a, b):
    """Linear function: f(x) = a*x + b"""
    return a * x + b

def confidence_fit(x, y, func, params, covariance, alpha=0.05):
    """Estimates confidence ranges for the fitted function.

    Args:
        x (array-like): Input data points.
        y (array-like): Target values.
        func (callable): Function to fit the data.
        params (array-like): Estimated parameters of the fitted function.
        covariance (ndarray): Covariance matrix of the fitted parameters.
        alpha (float, optional): Significance level for confidence interval. Defaults to 0.05.

    Returns:
        list: Confidence ranges for each data point.

    """
    p = len(params)
    n = len(x)
    dof = max(0, n - p)

    # Residuals
    residuals = y - func(x, *params)

    # Residual sum of squares
    ssr = np.sum(residuals ** 2)

    # Estimated standard deviation of the residuals
    s = np.sqrt(ssr / dof)

    # t-value for the confidence level
    t = scipy.stats.t.ppf(1 - alpha / 2, df=dof)

    # Confidence ranges
    ranges = []
    for i in range(len(x)):
        x_i = x[i]
        range_i = t * s * np.sqrt(1 + 1 / n + (x_i - np.mean(x)) ** 2 / ((n - 1) * np.var(x)))
        ranges.append(range_i)

    return ranges


def err_ranges(x, y, popt, pcov, model_func, alpha=0.05):
    """
    Estimate the confidence range for the best-fit curve.

    Args:
        x (np.ndarray): Input x values.
        y (np.ndarray): Input y values.
        popt (np.ndarray): Best-fit parameters.
        pcov (np.ndarray): Covariance matrix.
        model_func (callable): Model function used for fitting.
        alpha (float): Significance level for confidence intervals.

    Returns:
        np.ndarray: Lower and upper bounds of the confidence range.
    """
    try:
        y_fit = model_func(x, *popt)
        resid = y - y_fit
        n = len(x)
        dof = n - len(popt)
        t_val = t.ppf(1 - alpha / 2, df=dof)
        mse = np.sum(resid ** 2) / dof
        cov_diag = np.diag(pcov)
        err_range = t_val * np.sqrt(mse * (1 + cov_diag))
        lower = y_fit - err_range
        upper = y_fit + err_range
        return lower, upper
    except (ValueError, TypeError) as e:
        print(f"Error occurred while estimating confidence range: {e}")
        return None, None


def predict_with_ci(
    x_pred: np.ndarray,
    popt: np.ndarray,
    pcov: np.ndarray,
    model_func: callable,
    alpha: float = 0.05
) -> tuple:
    """
    Make predictions with confidence intervals.

    Args:
        x_pred (np.ndarray): Array of x values to make predictions.
        popt (np.ndarray): Best-fit parameters.
        pcov (np.ndarray): Covariance matrix.
        model_func (callable): Model function to use for predictions.
        alpha (float): Significance level for confidence intervals.

    Returns:
        tuple: Tuple containing the lower and upper bounds of the confidence intervals.
    """
    try:
        y_pred = model_func(x_pred, *popt)
        resid = y - model_func(x, *popt)
        n = len(x)
        dof = n - len(popt)
        t_val = t.ppf(1 - alpha / 2, df=dof)
        lower = y_pred - t_val * np.sqrt(np.diag(pcov))  # Calculate lower bound
        upper = y_pred + t_val * np.sqrt(np.diag(pcov))  # Calculate upper bound
        return lower, upper
    except (ValueError, TypeError) as e:
        print(f"Error occurred while making predictions with confidence intervals: {e}")
        return None, None


def polynomial_model(x, *coefficients):
    """
    Polynomial regression model function.

    Args:
        x (np.ndarray): Input x values.
        coefficients: Coefficients of the polynomial regression model.

    Returns:
        np.ndarray: Predicted y values.
    """
    return np.polyval(coefficients, x)


# Step 1: Data Selection and Preprocessing
data = pd.read_csv("D:/pos/API_19_DS2_en_csv_v2_5361599.csv", skiprows=4)
unique_values = data['Indicator Name'].unique()
print(unique_values)

# Select the indicators you want to use for clustering
selected_indicators = [
    'CO2 emissions (metric tons per capita)',
    'Population growth (annual %)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Forest area (% of land area)',
   
   'CO2 emissions (kg per 2017 PPP $ of GDP)',
    'Agriculture, forestry, and fishing, value added (% of GDP)'
]

# Filter the data to include only the selected indicators
df = data[data['Indicator Name'].isin(selected_indicators)]

# Pivot the data to have countries as rows and indicators as columns
df_pivot = df.pivot_table(index='Country Name', columns='Indicator Name', values='2019', aggfunc='mean')

df_pivot = df_pivot.fillna(df_pivot.mean())
df_pivot = df_pivot.astype(float)

# Print the shape of df_pivot to check if it has the expected dimensions
print("Shape of df_pivot:", df_pivot.shape)

# Print the first few rows of df_pivot to inspect the data
print(df_pivot.head())
print(df_pivot.describe())

# Normalize the data
df_min = df_pivot.min()
df_max = df_pivot.max()
df_normalized = (df_pivot - df_min) / (df_max - df_min)

# Perform K-means clustering
n_clusters = 3  # Set the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(df_normalized)

# Add cluster labels as a new column in the DataFrame
df_pivot['Cluster'] = cluster_labels

# step 2 :Heatmap Visualization
heatmap_corr(df_pivot[selected_indicators], title='Heatmap of the Correlation Matrix', title_fontweight='bold')


# Step 3: Visualization
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^']  # Marker shapes for each cluster
for i in range(n_clusters):
    cluster_data = df_normalized[cluster_labels == i]
    plt.scatter(cluster_data[selected_indicators[1]],
                cluster_data[selected_indicators[0]],
                label=f'Cluster {i+1}')

plt.xlabel(selected_indicators[1])
plt.ylabel(selected_indicators[0])
plt.title('Clustering of Countries - ' + selected_indicators[0], fontweight='bold')  # Add the indicator name to the title
plt.legend()
plt.colorbar()  # Add colorbar
plt.show()

# Step 4: Backscaling cluster centers
cluster_centers_normalized = kmeans.cluster_centers_
cluster_centers_backscaled = backscale(cluster_centers_normalized, df_min, df_max)

# Step 5: Visualization with cluster centers
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^']  # Marker shapes for each cluster
for i in range(n_clusters):
    cluster_data = df_pivot[df_pivot['Cluster'] == i]
    plt.scatter(cluster_data[selected_indicators[1]],
                cluster_data[selected_indicators[0]],
                marker=markers[i], label=f'Cluster {i+1}')

# Plot the cluster centers
plt.scatter(cluster_centers_backscaled[:, selected_indicators.index(selected_indicators[1])],
            cluster_centers_backscaled[:, selected_indicators.index(selected_indicators[0])],
            marker='X', s=100, color='black', label='Cluster Centers')

plt.xlabel(selected_indicators[1])
plt.ylabel(selected_indicators[0])
plt.title('Clustering of Countries - ' + selected_indicators[0], fontweight='bold')  # Add the indicator name to the title
plt.legend()
plt.colorbar()  # Add colorbar
plt.show()

# Step 6: Check for mismatching entries
column_to_check = selected_indicators[1]
df_cluster0 = df_pivot[df_pivot['Cluster'] == 0]
df_cluster1 = df_pivot[df_pivot['Cluster'] == 1]
mismatching_entries = get_diff_entries(df_cluster0, df_cluster1, column_to_check)

print(f'Mismatching entries in {column_to_check}: {mismatching_entries}')

# Step 10: Model Fitting and Prediction
x = df_pivot['Population growth (annual %)'].values
y = df_pivot['CO2 emissions (metric tons per capita)'].values

# Reshape x to match the dimensions of y
x = x.reshape(-1, 1)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# Fit the polynomial regression model to the data
poly_regression_model = LinearRegression()
poly_regression_model.fit(x_poly, y)

# Generate predictions for a range of x values
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = poly_regression_model.predict(x_range_poly)

# Estimate confidence range using predict_with_ci function
lower, upper = predict_with_ci(x_range, poly_regression_model.coef_, poly_regression_model._residues, polynomial_model)

# Check if any of the variables is None
if x_range is not None and y_range_pred is not None and lower is not None and upper is not None:
    # Convert x_range, lower, and upper to 1D arrays if they are not None
    x_range = x_range.flatten() if x_range is not None else None
    lower = lower.flatten() if lower is not None else None
    upper = upper.flatten() if upper is not None else None

    # Plot the polynomial regression line if x_range and y_range_pred are not None
    if x_range is not None and y_range_pred is not None:
        # Repeat y_range_pred to match the length of x_range
        y_range_pred = np.full_like(x_range, y_range_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(x_range, y_range_pred, color='red', label='Polynomial Regression')

    # Fill the confidence range if x_range, lower, and upper are not None
    if x_range is not None and lower is not None and upper is not None:
        plt.fill_between(x_range.flatten(), lower.flatten(), upper.flatten(), color='gray', alpha=0.2, label='Confidence Range')

    # Plot the actual data points
    plt.scatter(x, y, label='Actual Data')
    
    # Add gridlines
    plt.grid(True)
    plt.xlabel('Population growth (annual %)')
    plt.ylabel('CO2 emissions per capita')
    plt.title('Polynomial Regression', fontweight='bold')
    plt.legend()
    plt.show()

# Make predictions for new data
new_data = np.arange(1, 31).reshape(-1, 1) / 10.0
new_data_poly = poly.transform(new_data)
predictions = poly_regression_model.predict(new_data_poly)
print("Predictions:", predictions)

# Scatter plot of the actual data
plt.scatter(x, y, label='Actual Data')

# Plot the polynomial regression curve
plt.plot(x_range, y_range_pred, color='red', label='Polynomial Regression')
plt.xlabel('Population growth (annual %)')
plt.ylabel('CO2 emissions per capita')
plt.title('Polynomial Regression',fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()

# Plot the predicted values
new_data_pred = poly_regression_model.predict(new_data_poly)
plt.scatter(new_data.flatten(), new_data_pred, color='red', label='Predicted Data')
plt.grid(True)
plt.xlabel('Population growth ')
plt.ylabel('CO2 emissions per capita')
plt.title('Polynomial Regression of Prediction', fontweight='bold')
plt.legend()
plt.show()

# Step 11: Model Fitting and Prediction
x = df_pivot['Population growth (annual %)'].values
y = df_pivot['CO2 emissions (metric tons per capita)'].values

# Fit the data using curve_fit
params, covariance = curve_fit(linear_func, x, y)

# Get the parameter estimates and the associated errors
a_est, b_est = params
a_err, b_err = np.sqrt(np.diag(covariance))

# Predict values in the future
x_pred = np.arange(1, 31)
y_pred = linear_func(x_pred, a_est, b_est)

# Calculate confidence ranges using the err_ranges function
confidence_ranges = confidence_fit(x, y, linear_func, params, covariance)

# Obtain only 30 values from confidence_ranges
confidence_ranges = confidence_ranges[:30]
print(333333333333)
print(x_pred)
print(6666666666666)
print(y_pred)
print(88888888888)
print(confidence_ranges)

# Plot the data points, best fitting function, and confidence range
plt.scatter(x, y, label='Data')
plt.plot(x_pred, y_pred, label='Best Fit')
plt.fill_between(x_pred, y_pred - confidence_ranges, y_pred + confidence_ranges, alpha=0.3, label='Confidence Range')
plt.xlabel('Population growth (annual %)')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.title('Curve Fitting with Confidence Range', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()