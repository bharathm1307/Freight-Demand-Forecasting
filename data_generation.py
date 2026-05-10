import pandas as pd
import numpy as np

np.random.seed(42)

# CONFIG
n_rows = 100000

# 10 YEARS DATE RANGE (ORDERED)
date_range = pd.date_range(start="2014-01-01", end="2023-12-31", periods=n_rows)

regions = ["North", "South", "East", "West", "Central"]
cargo_types = ["Electronics", "FMCG", "Automobile", "Agriculture", "Pharma"]
shipping_modes = ["Road", "Rail", "Air", "Sea"]
weather_conditions = ["Clear", "Rain", "Storm", "Fog"]
delivery_statuses = ["On Time", "Delayed", "Cancelled"]
handling = ["None", "Fragile", "Hazardous", "Perishable"]

# CREATE DATAFRAME
df = pd.DataFrame({
    "Serial_No": np.arange(1, n_rows + 1),
    "Date": date_range
})

# DERIVED
df["Month"] = df["Date"].dt.month
df["Hour"] = df["Date"].dt.hour

# INDIAN SEASONS
def get_indian_season(m):
    if m in [12, 1, 2]: return "Winter"
    elif m in [3, 4, 5]: return "Summer"
    elif m in [6, 7, 8, 9]: return "Monsoon"
    else: return "Post-Monsoon"

df["Season"] = df["Month"].apply(get_indian_season)

# TIME OF DAY
def get_time_of_day(h):
    if 5 <= h < 12: return "Morning"
    elif 12 <= h < 17: return "Afternoon"
    elif 17 <= h < 21: return "Evening"
    else: return "Night"

df["Time_of_Day"] = df["Hour"].apply(get_time_of_day)

# RANDOM FEATURES
df["Origin"] = np.random.choice(regions, n_rows)
df["Destination"] = np.random.choice(regions, n_rows)
df["Cargo_Type"] = np.random.choice(cargo_types, n_rows)
df["Shipping_Mode"] = np.random.choice(shipping_modes, n_rows)
df["Weather"] = np.random.choice(weather_conditions, n_rows, p=[0.6, 0.2, 0.1, 0.1])
df["Special_Handling"] = np.random.choice(handling, n_rows)

# BASE VOLUME
base_volume = np.random.normal(100, 20, n_rows)

season_factor = df["Season"].map({
    "Winter": 0.9,
    "Summer": 1.1,
    "Monsoon": 0.8,
    "Post-Monsoon": 1.2
})

cargo_factor = df["Cargo_Type"].map({
    "Electronics": 1.3,
    "FMCG": 1.1,
    "Automobile": 0.9,
    "Agriculture": 1.2,
    "Pharma": 1.0
})

weather_factor = df["Weather"].map({
    "Clear": 1.0,
    "Rain": 0.85,
    "Storm": 0.7,
    "Fog": 0.8
})

df["Freight_Volume"] = (base_volume * season_factor * cargo_factor * weather_factor).round(2)

# PRICE
mode_factor = df["Shipping_Mode"].map({
    "Road": 1.0,
    "Rail": 0.8,
    "Air": 1.5,
    "Sea": 0.7
})

df["Price_per_Unit"] = (200 / (df["Freight_Volume"] + 1) * 100 * mode_factor).round(2)

# DELIVERY STATUS
def delivery_logic(row):
    if row["Weather"] in ["Storm", "Fog"]:
        return np.random.choice(["Delayed", "Cancelled"], p=[0.7, 0.3])
    elif row["Special_Handling"] == "Hazardous":
        return np.random.choice(["On Time", "Delayed"], p=[0.7, 0.3])
    else:
        return np.random.choice(delivery_statuses, p=[0.8, 0.15, 0.05])

df["Delivery_Status"] = df.apply(delivery_logic, axis=1)

# CLEANUP
df.drop(columns=["Month", "Hour"], inplace=True)

# SAVE
df.to_csv("synthetic_freight_data_10years.csv", index=False)

print("Dataset generated:", df.shape)
print(df.head())