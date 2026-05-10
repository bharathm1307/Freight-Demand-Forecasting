import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("synthetic_freight_data_10years.csv")

# =========================
# 2. DATE HANDLING
# =========================
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

# =========================
# 3. FEATURE ENGINEERING
# =========================
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Hour"] = df["Date"].dt.hour

# TIME SERIES MEMORY
df["lag_1"] = df["Freight_Volume"].shift(1)
df["lag_2"] = df["Freight_Volume"].shift(2)
df["rolling_mean_3"] = df["Freight_Volume"].rolling(3).mean()
df["rolling_mean_7"] = df["Freight_Volume"].rolling(7).mean()

df = df.dropna()

df["is_weekend"] = df["Date"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

# =========================
# 4. ENCODE CATEGORICALS
# =========================
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# =========================
# 5. TARGET + FEATURES
# =========================
y = df["Freight_Volume"]

drop_cols = ["Freight_Volume", "Date", "Serial_No", "Price_per_Unit"]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# =========================
# 6. SPLIT
# =========================
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# 7. MODEL
# =========================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 8. EVALUATION
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# =========================
# 9. FEATURE IMPORTANCE
# =========================
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop Features:\n", importance.sort_values(ascending=False).head(10))

# =========================
# 10. SAVE
# =========================
joblib.dump(model, "freight_model_xgb.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model trained and saved.")