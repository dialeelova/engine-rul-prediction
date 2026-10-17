import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We'll reload everything needed cleanly for this analysis
train = pd.read_csv('data/processed/train_FD001_clean.csv')
test = pd.read_csv('data/processed/test_FD001_clean.csv')
test_rul = pd.read_csv('data/processed/test_FD001_rul.csv')

print(test_rul.describe())

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

with open('data/processed/useless_sensors.txt') as f:
    useless_sensors = f.read().splitlines()

sensor_cols = [col for col in train.columns if col.startswith('sensor_') and col not in useless_sensors]

def add_rolling_features(df, sensor_cols, window=5):
    df = df.sort_values(['unit_number', 'time_cycles'])
    for col in sensor_cols:
        df[f'{col}_rollmean'] = df.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    return df

train = add_rolling_features(train, sensor_cols)
test = add_rolling_features(test, sensor_cols)

feature_cols = sensor_cols + [f'{col}_rollmean' for col in sensor_cols] + ['op_setting_1', 'op_setting_2', 'op_setting_3']

X_train = train[feature_cols]
y_train = train['RUL']

test_last = test.groupby('unit_number').last().reset_index()
X_test = test_last[feature_cols]
y_test = test_rul['RUL']

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)

print("Retrained. MAE:", mean_absolute_error(y_test, predictions))
# Build a results dataframe: true RUL, predicted RUL, and error, per engine
results = pd.DataFrame({
    'unit_number': test_last['unit_number'],
    'true_rul': y_test.values,
    'predicted_rul': predictions,
})
results['error'] = results['predicted_rul'] - results['true_rul']
results['abs_error'] = results['error'].abs()

# Which engines had the worst predictions?
worst = results.sort_values('abs_error', ascending=False).head(10)
print("10 worst predictions:")
print(worst)

# Is there a relationship between true RUL and prediction error?
plt.figure(figsize=(10, 5))
plt.scatter(results['true_rul'], results['error'], alpha=0.6)
plt.axhline(0, color='red', linestyle='--', label='Perfect prediction')
plt.xlabel('True RUL (cycles)')
plt.ylabel('Prediction Error (predicted - true)')
plt.title('Prediction Error vs True RUL')
plt.legend()
plt.savefig('notebooks/error_vs_true_rul.png')
print("Error analysis plot saved!")

# Correlation between true RUL and error
correlation = results['true_rul'].corr(results['error'])
print(f"Correlation between true RUL and error: {correlation:.3f}")

# Simulate a maintenance policy: schedule maintenance when predicted RUL drops below a threshold
threshold = 30  # cycles

results['maintenance_flagged'] = results['predicted_rul'] < threshold
results['was_actually_critical'] = results['true_rul'] < threshold

# How many did we correctly flag vs miss vs false-alarm?
true_positives = ((results['maintenance_flagged']) & (results['was_actually_critical'])).sum()
false_negatives = ((~results['maintenance_flagged']) & (results['was_actually_critical'])).sum()
false_positives = ((results['maintenance_flagged']) & (~results['was_actually_critical'])).sum()
true_negatives = ((~results['maintenance_flagged']) & (~results['was_actually_critical'])).sum()

print(f"\nMaintenance policy simulation (threshold = {threshold} cycles):")
print(f"Correctly flagged as critical (true positives): {true_positives}")
print(f"Missed critical engines (false negatives) — DANGEROUS: {false_negatives}")
print(f"Unnecessarily flagged (false positives) — wasteful but safe: {false_positives}")
print(f"Correctly left alone (true negatives): {true_negatives}")
# Test a more conservative threshold to see the tradeoff in action
def evaluate_threshold(threshold, results):
    flagged = results['predicted_rul'] < threshold
    critical = results['true_rul'] < threshold

    tp = ((flagged) & (critical)).sum()
    fn = ((~flagged) & (critical)).sum()
    fp = ((flagged) & (~critical)).sum()
    tn = ((~flagged) & (~critical)).sum()

    total_critical = tp + fn
    catch_rate = tp / total_critical if total_critical > 0 else 0

    print(f"\nThreshold = {threshold} cycles:")
    print(f"  Caught {tp}/{total_critical} critical engines ({catch_rate:.1%})")
    print(f"  Missed (dangerous): {fn}")
    print(f"  Unnecessary flags (wasteful): {fp}")

evaluate_threshold(30, results)
evaluate_threshold(45, results)
evaluate_threshold(60, results)