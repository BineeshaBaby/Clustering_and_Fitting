# -*- coding: utf-8 -*-
"""
Created on Tue May  9 19:45:37 2023

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

def heatmap_corr(df, size=6):
    """
    Create a heatmap of the correlation matrix for a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        size (int): Size of the heatmap figure.

    Returns:
        None
    """
    corr = df.corr()
    plt.figure(figsize=(size, size))
    sns.heatmap(corr, cmap='coolwarm', annot=True)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
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
try:
    data = pd.read_csv("D:/pos/API_19_DS2_en_csv_v2_5361599.csv", skiprows=4)
except FileNotFoundError as e:
    print(f"Error occurred while loading the data: {e}")
    exit()

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
df_pivot = df.pivot_table(index='Country Name', columns='Indicator Name', values='2018', aggfunc='mean')

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

# Step 3: Visualization
heatmap_corr(df_pivot[selected_indicators])

plt.figure(figsize=(10, 6))
markers = ['o', 's', '^']  # Marker shapes for each cluster
for i in range(n_clusters):
    cluster_data = df_normalized[cluster_labels == i]
    plt.scatter(cluster_data[selected_indicators[1]],
                cluster_data[selected_indicators[0]],
                label=f'Cluster {i+1}')

plt.xlabel(selected_indicators[1])
plt.ylabel(selected_indicators[0])
plt.title('Clustering of Countries - ' + selected_indicators[0])  # Add the indicator name to the title
plt.legend()
plt.colorbar()  # Add colorbar
plt.show()

# Step 4: Backscaling cluster centers
cluster_centers_normalized = kmeans.cluster_centers_
cluster_centers_backscaled = backscale(cluster_centers_normalized, df_min, df_max)

# Step 5: Visualization
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
plt.title('Clustering of Countries - ' + selected_indicators[0])  # Add the indicator name to the title
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

# Get the coefficients of the polynomial regression model
polynomial_coefficients_fit = poly_regression_model.coef_

# Generate predictions for a range of x values
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = poly_regression_model.predict(x_range_poly)

# Estimate confidence range using err_ranges function
lower, upper = err_ranges(x, y, x_range, y_range_pred, poly_regression_model)
# Check if any of the variables is None
if x_range is None or y_range_pred is None or lower is None or upper is None:
    print("Error: One or more variables is not defined.")
else:
    # Convert x_range, lower, and upper to 1D arrays if they are not None
    x_range = x_range.flatten() if x_range is not None else None
    lower = lower.flatten() if lower is not None else None
    upper = upper.flatten() if upper is not None else None

    # Plot the polynomial regression line if x_range and y_range_pred are not None
    if x_range is not None and y_range_pred is not None:
        # Repeat y_range_pred to match the length of x_range
        y_range_pred = np.full_like(x_range, y_range_pred)

        plt.plot(x_range, y_range_pred, color='red', label='Polynomial Regression')

    # Fill the confidence range if x_range, lower, and upper are not None
    if x_range is not None and lower is not None and upper is not None:
        plt.fill_between(x_range, lower, upper, color='gray', alpha=0.2, label='Confidence Range')
        
    # Other plot configurations and labels
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Actual Data')
    plt.xlabel('Population growth (annual %)')
    plt.ylabel('CO2 emissions per capita')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()
    



# Scatter plot of the actual data
plt.scatter(x, y, label='Actual Data')

# Plot the polynomial regression curve
plt.plot(x_range, y_range_pred, color='red', label='Polynomial Regression')

plt.xlabel('Population growth (annual %)')
plt.ylabel('CO2 emissions per capita')
plt.title('Polynomial Regression')
plt.legend()
plt.show()


# Step 12: Compare countries within and across clusters
# Select a country from each cluster
countries_per_cluster = []
for i in range(n_clusters):
    countries_per_cluster.append(df_pivot[df_pivot['Cluster'] == i].index[0])

# Print the selected countries
print("Selected countries per cluster:")
for i, country in enumerate(countries_per_cluster):
    print(f"Cluster {i+1}: {country}")

# Compare countries within a cluster
cluster_index = 0  # Select a cluster index
countries_within_cluster = df_pivot[df_pivot['Cluster'] == cluster_index].index

# Compare countries across clusters
cluster_indices = [0, 1]  # Select cluster indices to compare
countries_across_clusters = df_pivot[df_pivot['Cluster'].isin(cluster_indices)].index

