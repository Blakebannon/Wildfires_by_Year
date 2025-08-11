This is a basic portfolio piece that shows gathering data, cleaning data, finding a model that best fits said data and generating a predictive machine learning model to predict fire behavior based on annually available data. 

This is not currently an example of a finished piece and is a work in progress. All relevant Python code for this project currently can be found below. 

import pandas as pd

chunk_size = 1000
for chunk in pd.read_csv("F:\\Wildfire Data Analysis\\CSV Attribute Table Exports\\USGS_Wildland_Fire_Combined_Dataset.csv",
                         chunksize=chunk_size,
                         low_memory=False,
                         on_bad_lines='skip'):
    print(chunk.head())  # Process or inspect each chunk
    break  # Stop after first chunk for testing

    # Problematic dataset with some corrupted cells due to varied delimiters. Adjusting import method.

    import csv

    input_file = "F:\\Wildfire Data Analysis\\CSV Attribute Table Exports\\USGS_Wildland_Fire_Combined_Dataset.csv"
    output_file = "F:\\Wildfire Data Analysis\\CSV Attribute Table Exports\\cleaned_dataset.csv"

    with open(input_file, 'r', encoding='latin1') as infile, open(output_file, 'w', encoding='utf-8',
                                                                  newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            try:
                writer.writerow(row)  # Write valid rows
            except:
                continue  # Skip problematic rows

    # Try reading the cleaned file
    df = pd.read_csv(output_file, low_memory=False)
    df.head(10)

print(df.columns)

#dropping columns that we do not need for this analysis.
columns_to_drop = ['ï»¿OID_', 'USGS_Assigned_ID', 'Fire_Polygon_Tier', 'Fire_Attribute_Tiers', 'Source_Datasets', 'Listed_Fire_Codes', 'Listed_Fire_IDs',
    'Listed_Fire_IRWIN_IDs', 'Listed_Fire_Cause_Class', 'Listed_Rx_Reported_Acres', 'Listed_Map_Digitize_Methods', 'Listed_Notes', 'Processing_Notes',
    'Wildfire_and_Rx_Flag', 'Overlap_Within_1_or_2_Flag', 'Circleness_Scale', 'Circle_Flag',
    'Exclude_From_Summary_Rasters', 'Shape_Length', 'Shape_Area', 'Wildfire_Notice', 'Listed_Fire_Dates']

df = df.drop(columns=columns_to_drop, errors='ignore')

df.head()

# Fill non-finite values (NaN, inf, -inf) with 0, round, and convert to int
df['GIS_Acres'] = df['GIS_Acres'].fillna(0).replace([float('inf'), -float('inf')], 0).round().astype(int)
df['GIS_Hectares'] = df['GIS_Hectares'].fillna(0).replace([float('inf'), -float('inf')], 0).round().astype(int)

# Verify the changes
print(df[['GIS_Acres', 'GIS_Hectares']].head())
print(df[['GIS_Acres', 'GIS_Hectares']].dtypes)

df.head()

df.tail()

# Convert Fire_Year to numeric, coercing invalid values to NaN
df['Fire_Year'] = pd.to_numeric(df['Fire_Year'], errors='coerce')

# Dop rows with NaN in Fire_Year
df = df[df['Fire_Year'].notna()]

# Convert to integer to ensure whole numbers
df['Fire_Year'] = df['Fire_Year'].astype(int)

# Filter rows where Fire_Year is between 1950 and 2020
df = df[(df['Fire_Year'] >= 1950) & (df['Fire_Year'] <= 2020)]

# Verify the filtered DataFrame
print(df['Fire_Year'].describe())  # Summary statistics for Fire_Year
print(df['Fire_Year'].dtype)      # Confirm data type
print(df.head())                  # Show first few rows

# Replace NaN in Listed_Fire_Causes with "Unknown"
df['Listed_Fire_Causes'] = df['Listed_Fire_Causes'].fillna('Unknown')

# Verify the changes
print(df['Listed_Fire_Causes'].isna().sum())  # Should be 0 if all NaN are replaced
print(df['Listed_Fire_Causes'].value_counts())  # Show distribution of values
print(df[['Listed_Fire_Causes']].head())  # Show first few rows

df.head()

df.tail()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style for better visuals
sns.set_style("whitegrid")

# Create a boxplot of GIS_Acres by Fire_Year
plt.figure(figsize=(12, 6))  # Set figure size
sns.boxplot(x='Fire_Year', y='GIS_Acres', data=df)

# Customize the plot
plt.title('Distribution of Fire Sizes (GIS_Acres) by Year (1950–2020)', fontsize=14)
plt.xlabel('Fire Year', fontsize=12)
plt.ylabel('Fire Size (Acres)', fontsize=12)
plt.xticks(rotation=90)  # Rotate x-axis labels for readability

# Show the plot
plt.tight_layout()
plt.show()

print("Data type of GIS_Acres:", df['GIS_Acres'].dtype)

# Check summary statistics for GIS_Acres
print(df['GIS_Acres'].describe())
print("Max GIS_Acres:", df['GIS_Acres'].max())
print("Min GIS_Acres:", df['GIS_Acres'].min())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")

# Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fire_Year', y='GIS_Acres', data=df)

# Set y-axis range and ticks
plt.ylim(0, 1000000)  # Set y-axis from 0 to 1,000,000 acres
plt.yticks([0, 200000, 400000, 600000, 800000, 1000000],
           ['0', '200K', '400K', '600K', '800K', '1M'])  # Custom tick labels

# Customize plot
plt.title('Distribution of Fire Sizes (GIS_Acres) by Year (1950–2020)', fontsize=14)
plt.xlabel('Fire Year', fontsize=12)
plt.ylabel('Fire Size (Acres)', fontsize=12)
plt.xticks(rotation=90)

# Show plot
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

# Aggregate total GIS_Acres by Fire_Year
yearly_data = df.groupby('Fire_Year')['GIS_Acres'].sum().reset_index()

# Create scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Fire_Year', y='GIS_Acres', data=yearly_data, s=100, color='blue', label='Total Acres Burned')

# Calculate and plot linear regression trend line
slope, intercept, r_value, p_value, std_err = linregress(yearly_data['Fire_Year'], yearly_data['GIS_Acres'])
trend_line = slope * yearly_data['Fire_Year'] + intercept
plt.plot(yearly_data['Fire_Year'], trend_line, color='red', linestyle='--', label=f'Trend Line (R²={r_value**2:.2f})')

# Customize plot
plt.title('Total Fire Acres Burned per Year (1950–2020) with Trend Line', fontsize=14)
plt.xlabel('Fire Year', fontsize=12)
plt.ylabel('Total Fire Size (Acres)', fontsize=12)
plt.ylim(0, yearly_data['GIS_Acres'].max() * 1.1)  # Add 10% headroom for y-axis
plt.xticks(rotation=45, ticks=range(1950, 2021, 5))  # Show every 5 years for clarity
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Print trend line statistics
print(f"Slope: {slope:.2f} acres/year")
print(f"R²: {r_value**2:.2f}")
print(f"P-value: {p_value:.4f}")

# Aggregate total GIS_Acres by Fire_Year
yearly_data = df.groupby('Fire_Year')['GIS_Acres'].sum().reset_index()

# Check summary statistics
print("Yearly GIS_Acres Summary:")
print(yearly_data['GIS_Acres'].describe())
print("Max GIS_Acres (yearly sum):", yearly_data['GIS_Acres'].max())
print("Min GIS_Acres (yearly sum):", yearly_data['GIS_Acres'].min())
print(yearly_data.head(10))  # Show first 10 years

# Aggregate total GIS_Acres by Fire_Year
yearly_data = df.groupby('Fire_Year')['GIS_Acres'].sum().reset_index()

# Create scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Fire_Year', y='GIS_Acres', data=yearly_data, s=100, color='blue', label='Total Acres Burned')

# Linear trend line
slope, intercept, r_value, p_value, std_err = linregress(yearly_data['Fire_Year'], yearly_data['GIS_Acres'])
trend_line = slope * yearly_data['Fire_Year'] + intercept
plt.plot(yearly_data['Fire_Year'], trend_line, color='red', linestyle='--', label=f'Trend Line (R²={r_value**2:.2f})')

# Set y-axis scale explicitly
max_acres = max(yearly_data['GIS_Acres'].max() * 1.1, 10000000)  # Use data max + 10% or 10M
plt.ylim(0, max_acres)
plt.yticks([0, 2000000, 4000000, 6000000, 8000000, 10000000],
           ['0', '2M', '4M', '6M', '8M', '10M'])  # Custom ticks in millions

# Customize plot
plt.title('Total Fire Acres Burned per Year (1950–2020) with Trend Line', fontsize=14)
plt.xlabel('Fire Year', fontsize=12)
plt.ylabel('Total Fire Size (Acres) Cumalitve per Year', fontsize=12)
plt.xticks(rotation=45, ticks=range(1950, 2021, 5))  # Every 5 years
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Print trend line statistics
print(f"Slope: {slope:.2f} acres/year")
print(f"R²: {r_value**2:.2f}")
print(f"P-value: {p_value:.4f}")

# Aggregate total GIS_Acres by Fire_Year
yearly_data = df.groupby('Fire_Year')['GIS_Acres'].sum().reset_index()

# Filter for 1985–2020
yearly_data = yearly_data[(yearly_data['Fire_Year'] >= 1985) & (yearly_data['Fire_Year'] <= 2020)]

# Verify the data
print("Yearly GIS_Acres Summary (1985–2020):")
print(yearly_data['GIS_Acres'].describe())
print("Max GIS_Acres:", yearly_data['GIS_Acres'].max())
print("Min GIS_Acres:", yearly_data['GIS_Acres'].min())
print(yearly_data.head(10))

# Create scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Fire_Year', y='GIS_Acres', data=yearly_data, s=100, color='blue', label='Total Acres Burned')

# Linear trend line
slope, intercept, r_value, p_value, std_err = linregress(yearly_data['Fire_Year'], yearly_data['GIS_Acres'])
trend_line = slope * yearly_data['Fire_Year'] + intercept
plt.plot(yearly_data['Fire_Year'], trend_line, color='red', linestyle='--', label=f'Trend Line (R²={r_value**2:.2f})')

# Set y-axis scale
max_acres = max(yearly_data['GIS_Acres'].max() * 1.1, 10000000)  # Use data max + 10% or 10M
plt.ylim(0, max_acres)
plt.yticks([0, 2000000, 4000000, 6000000, 8000000, 10000000],
           ['0', '2M', '4M', '6M', '8M', '10M'])

# Customize plot
plt.title('Total Fire Acres Burned per Year (1985–2020) with Trend Line', fontsize=14)
plt.xlabel('Fire Year', fontsize=12)
plt.ylabel('Total Fire Size (Acres)', fontsize=12)
plt.xticks(rotation=45, ticks=range(1985, 2021, 5))  # Every 5 years
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Print trend line statistics
print(f"Slope: {slope:.2f} acres/year")
print(f"R²: {r_value**2:.2f}")
print(f"P-value: {p_value:.4f}")

# Aggregate total GIS_Acres by Fire_Year (1985–2020)
yearly_data = df.groupby('Fire_Year')['GIS_Acres'].sum().reset_index()
yearly_data = yearly_data[(yearly_data['Fire_Year'] >= 1985) & (yearly_data['Fire_Year'] <= 2020)]

# Prepare features (X) and target (y)
X = yearly_data[['Fire_Year']].values  # Feature: Fire_Year
y = yearly_data['GIS_Acres'].values     # Target: Total GIS_Acres

# Verify data
print("Data Shape:", X.shape, y.shape)
print(yearly_data.head())
print("Max GIS_Acres:", yearly_data['GIS_Acres'].max())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

print("Linear Regression:")
print(f"R²: {lr_r2:.2f}")
print(f"MSE: {lr_mse:.2f}")

# --- Model 2: Polynomial Regression (Degree 2) ---
poly = PolynomialFeatures(degree=2)
polyreg = make_pipeline(poly, LinearRegression())
polyreg.fit(X_train, y_train)
poly_pred = polyreg.predict(X_test)
poly_r2 = r2_score(y_test, poly_pred)
poly_mse = mean_squared_error(y_test, poly_pred)

print("\nPolynomial Regression (Degree 2):")
print(f"R²: {poly_r2:.2f}")
print(f"MSE: {poly_mse:.2f}")

# --- Model 3: Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

print("\nRandom Forest Regressor:")
print(f"R²: {rf_r2:.2f}")
print(f"MSE: {rf_mse:.2f}")

# Create future years for prediction (2021–2050)
future_years = np.array(range(2021, 2051)).reshape(-1, 1)
all_years = np.vstack([X, future_years])  # Combine historical and future years

# Predictions for all years
lr_future_pred = lr_model.predict(all_years)
poly_future_pred = polyreg.predict(all_years)
rf_future_pred = rf_model.predict(all_years)

# Plot historical data and predictions
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Fire_Year', y='GIS_Acres', data=yearly_data, s=100, color='blue', label='Historical Data')

# Plot predictions
plt.plot(all_years, lr_future_pred, color='red', linestyle='--', label='Linear Regression')
plt.plot(all_years, poly_future_pred, color='green', linestyle='--', label='Polynomial Regression (Deg 2)')
plt.plot(all_years, rf_future_pred, color='purple', linestyle='--', label='Random Forest')

# Customize plot
plt.title('Total Fire Acres Burned (1985–2020) and Predictions to 2050', fontsize=14)
plt.xlabel('Fire Year', fontsize=12)
plt.ylabel('Total Fire Size (Acres)', fontsize=12)
plt.ylim(0, max(y.max(), lr_future_pred.max(), poly_future_pred.max(), rf_future_pred.max()) * 1.1)
plt.yticks([0, 2000000, 4000000, 6000000, 8000000, 10000000],
           ['0', '2M', '4M', '6M', '8M', '10M'])
plt.xticks(rotation=45, ticks=range(1985, 2051, 5))
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Print predictions for select future years
future_df = pd.DataFrame({
    'Year': all_years.flatten(),
    'Linear_Pred': lr_future_pred,
    'Poly_Pred': poly_future_pred,
    'RF_Pred': rf_future_pred
})
print("Predictions for 2030, 2040, 2050:")
print(future_df[future_df['Year'].isin([2030, 2040, 2050])])

# Aggregate total GIS_Acres by Fire_Year (1985–2020)
yearly_data = df.groupby('Fire_Year')['GIS_Acres'].sum().reset_index()
yearly_data = yearly_data[(yearly_data['Fire_Year'] >= 1985) & (yearly_data['Fire_Year'] <= 2020)]

# Check distribution
print("Yearly GIS_Acres Summary:")
print(yearly_data['GIS_Acres'].describe())
plt.figure(figsize=(8, 4))
sns.histplot(yearly_data['GIS_Acres'], kde=True)
plt.title('Distribution of Total GIS_Acres per Year')
plt.xlabel('Total GIS_Acres')
plt.show()

# Log-transform GIS_Acres (add 1 to avoid log(0))
yearly_data['Log_GIS_Acres'] = np.log1p(yearly_data['GIS_Acres'])

# Prepare features (exclude GIS_Acres and Log_GIS_Acres)
X = yearly_data.drop(columns=['GIS_Acres', 'Log_GIS_Acres']).values
y = yearly_data['Log_GIS_Acres'].values  # Use log-transformed target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
print("Linear Regression (Log-Transformed):")
print(f"R²: {lr_r2:.2f}")
print(f"MSE: {lr_mse:.2f}")

# --- Polynomial Regression (Degree 2) ---
poly = PolynomialFeatures(degree=2)
polyreg = make_pipeline(poly, LinearRegression())
polyreg.fit(X_train, y_train)
poly_pred = polyreg.predict(X_test)
poly_r2 = r2_score(y_test, poly_pred)
poly_mse = mean_squared_error(y_test, poly_pred)
print("\nPolynomial Regression (Degree 2, Log-Transformed):")
print(f"R²: {poly_r2:.2f}")
print(f"MSE: {poly_mse:.2f}")

# --- XGBoost Regressor ---
xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
print("\nXGBoost Regressor (Log-Transformed):")
print(f"R²: {xgb_r2:.2f}")
print(f"MSE: {xgb_mse:.2f}")

# Create future years (2021–2050)
future_years = pd.DataFrame({'Fire_Year': range(2021, 2051)})
# Use mean values for cause and type features
mean_features = yearly_data.drop(columns=['Fire_Year', 'GIS_Acres', 'Log_GIS_Acres']).mean()
future_data = pd.concat([future_years.assign(**mean_features), yearly_data.drop(columns=['GIS_Acres', 'Log_GIS_Acres'])], ignore_index=True)

# Prepare X for all years
X_all = future_data.values

# Predict log-transformed values and convert back to acres
lr_future_pred = np.expm1(lr_model.predict(X_all))  # Inverse of log1p
poly_future_pred = np.expm1(polyreg.predict(X_all))
xgb_future_pred = np.expm1(xgb_model.predict(X_all))

# Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Fire_Year', y='GIS_Acres', data=yearly_data, s=100, color='blue', label='Historical Data')
plt.plot(future_data['Fire_Year'], lr_future_pred, color='red', linestyle='--', label='Linear Regression')
plt.plot(future_data['Fire_Year'], poly_future_pred, color='green', linestyle='--', label='Polynomial Regression')
plt.plot(future_data['Fire_Year'], xgb_future_pred, color='purple', linestyle='--', label='XGBoost')

# Customize plot
plt.title('Total Fire Acres Burned (1985–2020) and Predictions to 2050', fontsize=14)
plt.xlabel('Fire Year', fontsize=12)
plt.ylabel('Total Fire Size (Acres)', fontsize=12)
plt.ylim(0, max(yearly_data['GIS_Acres'].max(), lr_future_pred.max(), poly_future_pred.max(), xgb_future_pred.max()) * 1.1)
plt.yticks([0, 2000000, 4000000, 6000000, 8000000, 10000000], ['0', '2M', '4M', '6M', '8M', '10M'])
plt.xticks(rotation=45, ticks=range(1985, 2051, 5))
plt.legend()

plt.tight_layout()
plt.show()

# Save predictions
predictions = pd.DataFrame({
    'Year': future_data['Fire_Year'],
    'Linear_Pred_Acres': lr_future_pred,
    'Poly_Pred_Acres': poly_future_pred,
    'XGB_Pred_Acres': xgb_future_pred
})
print("Predictions for 2030, 2040, 2050:")
print(predictions[predictions['Year'].isin([2030, 2040, 2050])])
