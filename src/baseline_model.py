import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load cleaned data from earlier phases
train = pd.read_csv('data/processed/train_FD001_clean.csv')
test = pd.read_csv('data/processed/test_FD001_clean.csv')
test_rul = pd.read_csv('data/processed/test_FD001_rul.csv')

print(train.shape, test.shape, test_rul.shape)

# Dumb baseline: always predict the average RUL
avg_rul = train['RUL'].mean()
print(f"Average RUL in training data: {avg_rul:.2f}")

dumb_predictions = [avg_rul] * len(test_rul)
dumb_mae = mean_absolute_error(test_rul['RUL'], dumb_predictions)
print(f"Dumb baseline MAE: {dumb_mae:.2f} cycles")

# Prepare features: drop non-sensor columns, and drop the useless sensors we found in Phase 3
with open('data/processed/useless_sensors.txt') as f:
    useless_sensors = f.read().splitlines()

feature_cols = [col for col in train.columns if col.startswith('sensor_') and col not in useless_sensors]
feature_cols += ['op_setting_1', 'op_setting_2', 'op_setting_3']

print("Features we'll use:", feature_cols)
print("Number of features:", len(feature_cols))

# Train linear regression on all training rows
X_train = train[feature_cols]
y_train = train['RUL']

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained!")

# For test data: take the LAST row per engine (most recent reading before cutoff)
test_last = test.groupby('unit_number').last().reset_index()
X_test = test_last[feature_cols]

# Predict
predictions = model.predict(X_test)

# Evaluate against the true RUL values
linreg_mae = mean_absolute_error(test_rul['RUL'], predictions)
print(f"Linear Regression MAE: {linreg_mae:.2f} cycles")