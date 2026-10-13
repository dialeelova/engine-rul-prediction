import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned training data we saved in Phase 2
train = pd.read_csv('data/processed/train_FD001_clean.csv')

print(train.shape)
print(train['unit_number'].nunique())  # how many unique engines
# Look at engine #1's full lifetime
engine_1 = train[train['unit_number'] == 1]

plt.figure(figsize=(10, 5))
plt.plot(engine_1['time_cycles'], engine_1['sensor_2'])
plt.xlabel('Time (cycles)')
plt.ylabel('Sensor 2 reading')
plt.title('Engine #1 — Sensor 2 over its lifetime')
plt.savefig('notebooks/engine1_sensor2.png')
print("Plot saved!")

# Compare sensor 2 across multiple engines
plt.figure(figsize=(10, 5))
for unit in [1, 2, 3, 4, 5]:
    engine_data = train[train['unit_number'] == unit]
    plt.plot(engine_data['time_cycles'], engine_data['sensor_2'], label=f'Engine {unit}')

plt.xlabel('Time (cycles)')
plt.ylabel('Sensor 2 reading')
plt.title('Sensor 2 across 5 different engines')
plt.legend()
plt.savefig('notebooks/sensor2_multiple_engines.png')
print("Comparison plot saved!")

# Check which sensors vary vs which stay constant
sensor_cols = [col for col in train.columns if col.startswith('sensor_')]
sensor_std = train[sensor_cols].std().sort_values()

print("Sensors with LOWEST variation (possibly useless):")
print(sensor_std.head(10))

# Visual comparison: a "dead" sensor vs an informative one
engine_1 = train[train['unit_number'] == 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(engine_1['time_cycles'], engine_1['sensor_1'])
axes[0].set_title('Sensor 1 (near-zero variation)')
axes[0].set_xlabel('Time (cycles)')

axes[1].plot(engine_1['time_cycles'], engine_1['sensor_2'])
axes[1].set_title('Sensor 2 (shows degradation trend)')
axes[1].set_xlabel('Time (cycles)')

plt.tight_layout()
plt.savefig('notebooks/useless_vs_useful_sensor.png')
print("Comparison saved!")

# Save list of low-variation sensors to drop later
useless_sensors = sensor_std[sensor_std < 0.01].index.tolist()
print("Sensors to drop:", useless_sensors)

with open('data/processed/useless_sensors.txt', 'w') as f:
    f.write('\n'.join(useless_sensors))