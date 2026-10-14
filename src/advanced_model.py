import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Load cleaned data
train = pd.read_csv('data/processed/train_FD001_clean.csv')
test = pd.read_csv('data/processed/test_FD001_clean.csv')
test_rul = pd.read_csv('data/processed/test_FD001_rul.csv')

# Load our list of useless sensors from Phase 3
with open('data/processed/useless_sensors.txt') as f:
    useless_sensors = f.read().splitlines()

sensor_cols = [col for col in train.columns if col.startswith('sensor_') and col not in useless_sensors]
print("Using sensors:", sensor_cols)

def add_rolling_features(df, sensor_cols, window=5):
    df = df.sort_values(['unit_number', 'time_cycles'])
    for col in sensor_cols:
        df[f'{col}_rollmean'] = df.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    return df

train = add_rolling_features(train, sensor_cols)
test = add_rolling_features(test, sensor_cols)

print(train.columns.tolist())

# Build final feature list: raw sensors + rolling averages + operating settings
feature_cols = sensor_cols + [f'{col}_rollmean' for col in sensor_cols] + ['op_setting_1', 'op_setting_2', 'op_setting_3']

X_train = train[feature_cols]
y_train = train['RUL']

# For test: take the last row per engine (same approach as Phase 4)
test_last = test.groupby('unit_number').last().reset_index()
X_test = test_last[feature_cols]
y_test = test_rul['RUL']

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)

print(f"Random Forest MAE: {rf_mae:.2f} cycles")

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)

print(f"XGBoost MAE: {xgb_mae:.2f} cycles")