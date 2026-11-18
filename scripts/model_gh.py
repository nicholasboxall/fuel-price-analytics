import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "e10_average_usd_oil_cpi.csv", header=0)

"""
Ask for Plots

    Set these to 0 to remove output(s)
    Set these to 1 to view the output(s)
"""

plot_p_t = 1 # Plot peaks and troughs of price
plot_func = 1 # Plot fuel cycle estimator function
peaks_analysis_cycle_length = 1 # Plot peak to peak length vs price change between peaks
plot_peaks_analysis_daily_cycle_length = 1 # Plot days since last peak vs price difference from last peak
print_model = 1 # Print model and variables
plot_X_y = 1 # Plot model elements
plot_model = 1 # Plot model predicted value vs actual value
plot_residuals = 1 # Plot model residuals

"""
Data Manipulation
"""

df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S")

df['Crude Oil Price AUD'] = df['AUD to 1 USD'] * df['Crude Oil Price']

df['Crude Oil Price AUD'] = df['Crude Oil Price AUD'].ffill()

df['OVX Index'] = df['OVX Index'].ffill()

# Create Moving Averages
moving_average_days = [7, 14, 35, 70]

for days in moving_average_days:
    df["Average of Price " + f"{days} Day Moving Average"] = (
        df["Average of Price"]
        .rolling(window=days, min_periods=days)
        .mean()
    )

# Fill CPI linearly
df["Normalized CPI"] = pd.to_numeric(
    df["Normalized CPI"], errors="coerce"
)
df["Daily Normalized CPI"] = df["Normalized CPI"].interpolate(
    method="linear"
)

# Fill Fuel Excise from last known figure
df["Fuel Excise (c)"] = pd.to_numeric(
    df["Fuel Excise (c)"], errors="coerce"
)
df["Daily Fuel Excise"] = df["Fuel Excise (c)"].ffill()

# Create % Change columns
for col in df.columns[1:]:
    df[col + " Change"] = df[col].diff()
    
"""
Oil Volatility
"""

df["Oil Log Return"] = np.log(
    df["Crude Oil Price AUD"] /
    df["Crude Oil Price AUD"].shift(1)
)

df['Oil Log Return'] = 100 * df['Oil Log Return']

moving_average_days = [7, 14, 35]
for days in moving_average_days:
    df["Oil Log Return " + f"{days} Day MA"] = (
        df["Oil Log Return"]
        .rolling(window=days, min_periods=days)
        .mean()
    )


"""
Peaks and Troughs
"""

daily_or_ma = 0

if daily_or_ma == 0:
    # Settings for Daily
    prominence = 4
    distance = 20
    col = 'Average of Price'
elif daily_or_ma == 1:
    # Settings for 7 Day Moving Average
    prominence = 1
    distance = 10
    col = 'Average of Price 7 Day Moving Average'

# Identify peaks
peaks, _ = find_peaks(
    df[col],
    prominence=prominence,
    distance=distance
)

# Identify troughs by finding peaks in the inverted price series
troughs, _ = find_peaks(
    -df[col],
    prominence=prominence,
    distance=distance
)

# Create Days Since Last Peak
df["Days Since Last Peak"] = np.nan

peak_dates = df.loc[peaks, "Date"]

for peak_idx in peaks:
    df.loc[peak_idx:, "Days Since Last Peak"] = (
        df.loc[peak_idx:, "Date"] - df.loc[peak_idx, "Date"]
    ).dt.days

peak_intervals = peak_dates.diff().dt.days.dropna()

# print("Average peak-to-peak cycle:",
#       peak_intervals.mean(), "days")

# print("Median peak-to-peak cycle:",
#       peak_intervals.median(), "days")

trough_dates = df.loc[troughs, "Date"]

trough_intervals = trough_dates.diff().dt.days.dropna()

# print("Average trough-to-trough cycle:",
#       trough_intervals.mean(), "days")

# Plot Peaks and Troughs

if plot_p_t == 1:
    plt.figure(figsize=(16, 7))
    plt.plot(
        df["Date"],
        df["Average of Price"],
        alpha=0.3,
        label="Daily Fuel Price"
    )
    plt.plot(
        df["Date"],
        df[col],
        label=col
    )
    plt.scatter(
        df.loc[peaks, "Date"],
        df.loc[peaks, [col]],
        color="red",
        label="Peaks",
        zorder=3
    )
    plt.scatter(
        df.loc[troughs, "Date"],
        df.loc[troughs, col],
        color="green",
        label="Troughs",
        zorder=3
    )
    plt.plot(
        df["Date"],
        df["Days Since Last Peak"],
        label="Days Since Last Peak"
    )
    plt.axhline(
        y=peak_intervals.median(),
        color="red",
        linestyle="--",
        label=f"Intervals Median: {peak_intervals.median()} days"
    )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Fuel Price (c/L)")
    plt.title(f"Australian Fuel Pricing Cycles - Prominence: {prominence:.2f} cents, Distance: {distance:.2f} days")
    plt.show()


"""
Create Lags and Peaks Metric
"""

lag_variables = [
    'Average of Price',
    "Crude Oil Price AUD",
    "Daily Fuel Excise",
    'Oil Log Return 7 Day MA'
]

for column in lag_variables:
    for lag in [1, 3, 7, 10, 14, 21]:
        df[f"{column} Lag {lag}"] = df[column].shift(lag)


median_peak = peak_intervals.median()
average_peak = peak_intervals.mean()

def fuel_cycle_metric(x, length_between_peaks):
    return np.where(
        x >= length_between_peaks,
        0,
        0.05 * ((x)**2 - x*length_between_peaks)
        #0.05* (x - peak_type/2)**2 - 0.05* (-peak_type/2)**2
    )

x = np.linspace(0, 50, 500)

if plot_func == 1:
    y = fuel_cycle_metric(x, median_peak)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2)
    
    plt.xlabel('Days Since Last Peak')
    plt.ylabel('Fuel Cycle Metric')
    plt.title('Fuel Cycle Metric Function')
    plt.grid(True)
    plt.show()

peak_length = median_peak # Change this number to change the metric, set as median_peak
df['Days Since Last Peak Metric'] = fuel_cycle_metric(
    df['Days Since Last Peak'],
    peak_length
)

"""
Peaks Analysis
"""

if peaks_analysis_cycle_length == 1:
    # Create peak-to-peak analysis dataset
    peak_analysis = pd.DataFrame({
        "Peak Date": df.loc[peaks, "Date"].values,
        "Peak Price": df.loc[peaks, 'Average of Price'].values
    })
    
    # Previous peak date and price
    peak_analysis["Previous Peak Date"] = peak_analysis["Peak Date"].shift(1)
    peak_analysis["Previous Peak Price"] = peak_analysis["Peak Price"].shift(1)
    
    # Cycle length
    peak_analysis["Cycle Length"] = (
        peak_analysis["Peak Date"] -
        peak_analysis["Previous Peak Date"]
    ).dt.days
    
    # Difference between consecutive peaks
    peak_analysis["Price Difference"] = (
        peak_analysis["Peak Price"] -
        peak_analysis["Previous Peak Price"]
    )
    
    # Percentage difference
    peak_analysis["Percentage Change"] = (
        peak_analysis["Price Difference"] /
        peak_analysis["Previous Peak Price"] * 100
    )
    
    # Classify cycle based on median
    median_cycle = peak_intervals.median()
    
    peak_analysis["Cycle Type"] = np.where(
        peak_analysis["Cycle Length"] < median_cycle,
        "Shorter",
        "Longer"
    )
    
    # Remove first peak (no previous peak)
    peak_analysis = peak_analysis.dropna().reset_index(drop=True)
    
    #print(peak_analysis)
    
    summary = peak_analysis.groupby("Cycle Type").agg(
        Number_of_Cycles=("Price Difference", "count"),
        Average_Cycle_Length=("Cycle Length", "mean"),
        Average_Price_Change=("Price Difference", "mean"),
        Median_Price_Change=("Price Difference", "median"),
        Average_Percentage_Change=("Percentage Change", "mean"),
        Percentage_Higher=("Price Difference",
                           lambda x: (x > 0).mean() * 100),
        Percentage_Lower=("Price Difference",
                          lambda x: (x < 0).mean() * 100)
    )
    
    # print("\nPeak-to-Peak Comparison:")
    # print(summary.round(2))
    
    plt.figure(figsize=(13, 7))
    
    plt.scatter(
        peak_analysis["Cycle Length"],
        peak_analysis["Price Difference"],
        alpha=0.7,
        s=60
    )
    
    plt.axhline(0, color="black", linewidth=1)
    
    plt.axvline(
        median_cycle,
        color="red",
        linestyle="--",
        label=f"Median Cycle: {median_cycle:.1f} days"
    )
    
    plt.xlabel("Peak-to-Peak Cycle Length (Days)", fontsize=14)
    plt.ylabel("Change in Peak Price (c/L)", fontsize=14)
    
    plt.title(
        "Relationship Between Cycle Length and Subsequent Peak Price",
        fontsize=16
    )
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

peaks_analysis_daily_cycle_length = 1

if peaks_analysis_daily_cycle_length == 1:   
    # Get peak dates and prices
    peak_dates = df.loc[peaks, "Date"].values
    peak_prices = df.loc[peaks, col].values
    
    # Find most recent peak for every date
    last_peak_indices = np.searchsorted(
        peaks,
        np.arange(len(df)),
        side="right"
    ) - 1
    
    # Initialise columns
    df["Days Since Last Peak"] = np.nan
    df["Last Peak Price"] = np.nan
    df["Price Difference from Last Peak"] = np.nan
    
    # Only include dates occurring on or after the first detected peak
    valid = last_peak_indices >= 0
    
    # Assign last peak price
    df.loc[valid, "Last Peak Price"] = peak_prices[
        last_peak_indices[valid]
    ]
    
    # Calculate days since last peak
    df.loc[valid, "Days Since Last Peak"] = (
        df.loc[valid, "Date"].values -
        peak_dates[last_peak_indices[valid]]
    ).astype("timedelta64[D]").astype(int)
    
    # Calculate difference from last peak
    df.loc[valid, "Price Difference from Last Peak"] = (
        df.loc[valid, col].values -
        df.loc[valid, "Last Peak Price"].values
    )
    
    # Peak intervals
    peak_intervals = pd.Series(peak_dates).diff().dt.days.dropna()
    
    if plot_peaks_analysis_daily_cycle_length == 1:
        plt.figure(figsize=(10, 6))
    
        # Individual observations
        plt.scatter(
            df["Days Since Last Peak"],
            df["Price Difference from Last Peak"],
            alpha=0.3,
            s=10,
            label="Individual Daily Observations"
        )
        
        # Average price difference
        daily_average_mean = df.groupby(
            "Days Since Last Peak"
        )["Price Difference from Last Peak"].mean()
        
        plt.plot(
            daily_average_mean.index,
            daily_average_mean.values,
            color="red",
            linewidth=2.5,
            label="Average Price Difference"
        )
        
        # Average price difference
        daily_average_median = df.groupby(
            "Days Since Last Peak"
        )["Price Difference from Last Peak"].median()
        
        plt.plot(
            daily_average_median.index,
            daily_average_median.values,
            color="purple",
            linewidth=2.5,
            label="Median Price Difference"
        )
        
        plt.plot(
            x,
            fuel_cycle_metric(x,peak_length),
            color="orange",
            linewidth=4,
            label="Days Since Last Peak Metric Function"
        )
        
        plt.axhline(0, color="black", linewidth=1)
        
        plt.axvline(
            peak_intervals.median(),
            color="red",
            linestyle="--",
            alpha=0.6,
            label=f"Median Peak Cycle: {peak_intervals.median():.1f} days"
        )
        
        plt.axvline(
            peak_intervals.mean(),
            color="purple",
            linestyle="--",
            alpha=0.6,
            label=f"Mean Peak Cycle: {peak_intervals.mean():.1f} days"
        )
        
        plt.xlabel("Days Since Last Peak", fontsize=14)
        plt.ylabel("Price Difference from Last Peak (c/L)", fontsize=14)
        
        plt.title(
            "Individual Daily Fuel Price Movement Since Last Peak",
            fontsize=16
        )
        
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


"""
Columns:
              ['Date', 'Average of Price', 'Count of Price', 'AUD to 1 USD',
              'Crude Oil Price', 'Normalized CPI', 'Fuel Excise (c)',
              'Crude Oil Price AUD', 'Average of Price 7 Day Moving Average',
              'Average of Price 14 Day Moving Average',
              'Average of Price 35 Day Moving Average',
              'Average of Price 70 Day Moving Average', 'Daily Normalized CPI',
              'Daily Fuel Excise', 'Average of Price Change', 'Count of Price Change',
              'AUD to 1 USD Change', 'Crude Oil Price Change',
              'Normalized CPI Change', 'Fuel Excise (c) Change',
              'Crude Oil Price AUD Change',
              'Average of Price 7 Day Moving Average Change',
              'Average of Price 14 Day Moving Average Change',
              'Average of Price 35 Day Moving Average Change',
              'Average of Price 70 Day Moving Average Change',
              'Daily Normalized CPI Change', 'Daily Fuel Excise Change',
              'Oil Log Return', 'Oil Log Return 7 Day MA', 'Oil Log Return 14 Day MA',
              'Oil Log Return 35 Day MA', 'Days Since Last Peak',
              'Average of Price Lag 1', 'Average of Price Lag 3',
              'Average of Price Lag 7', 'Average of Price Lag 10',
              'Average of Price Lag 14', 'Average of Price Lag 21',
              'Crude Oil Price AUD Lag 1', 'Crude Oil Price AUD Lag 3',
              'Crude Oil Price AUD Lag 7', 'Crude Oil Price AUD Lag 10',
              'Crude Oil Price AUD Lag 14', 'Crude Oil Price AUD Lag 21',
              'Daily Fuel Excise Lag 1', 'Daily Fuel Excise Lag 3',
              'Daily Fuel Excise Lag 7', 'Daily Fuel Excise Lag 10',
              'Daily Fuel Excise Lag 14', 'Daily Fuel Excise Lag 21',
              'Oil Log Return 7 Day MA Lag 1', 'Oil Log Return 7 Day MA Lag 3',
              'Oil Log Return 7 Day MA Lag 7', 'Oil Log Return 7 Day MA Lag 10',
              'Oil Log Return 7 Day MA Lag 14', 'Oil Log Return 7 Day MA Lag 21',
              'Days Since Last Peak Metric', 'Last Peak Price',
              'Price Difference from Last Peak', 'Log Price', 'Predicted',
              'Residuals']
"""

"""
Modelling
"""

X = df[
    [
     'Average of Price Lag 7',
     'Crude Oil Price AUD Lag 7',
     'Daily Fuel Excise',
     'Days Since Last Peak Metric',
     'Daily Normalized CPI'
    ]
]

df['Log Price'] = np.log(df['Average of Price'])


y_col = "Average of Price"
#y_col = "Average of Price 7 Day Moving Average"
#y_col = "Log Price"
y = df[y_col]

data = pd.concat([y, X], axis=1).dropna()

X = sm.add_constant(data[X.columns])
y = data[y_col]

model = sm.WLS(y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 14}
)


if print_model == 1:
    print('---------------------- MODEL ----------------------')
    print(model.summary())
    print('')
  


if plot_X_y == 1:
    plt.figure(figsize=(10, 6)) 
    for i in data:   
        plt.plot(df['Date'],df[i], label=i)
    plt.legend()
    plt.title('All Model Elements')
    plt.show()

df['Predicted'] = model.predict(X) 
df['Residuals'] = df[y_col] - df['Predicted']

if plot_model == 1:
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df[y_col], label=y_col)
    plt.plot(df['Date'], df['Predicted'], label = 'Predicted')
    #plt.plot(df['Date'], df['Residuals'], label = 'Error')
    #plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Real vs Prediction')
    plt.legend()
    plt.show()


if plot_residuals == 1:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Predicted'], df['Residuals'], alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Fuel Price')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted {y_col}')
    plt.grid()
    plt.show()
    
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df[y_col], df['Residuals'], alpha=0.4)
    # plt.axhline(0, color='red', linestyle='--')
    # plt.xlabel('Fuel Price')
    # plt.ylabel('Residuals')
    # plt.title(f'Residuals vs {y_col}')
    # plt.grid()
    # plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Date'], df['Residuals'], alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Date')
    plt.grid()
    plt.show()


