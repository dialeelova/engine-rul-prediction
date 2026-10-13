import pandas as pd

# Column names based on readme.txt (21 real sensors, not 26 — known NASA readme quirk)
columns = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
columns += [f'sensor_{i}' for i in range(1, 22)]  # sensors 1 through 21

train = pd.read_csv(
    'data/raw/cmapss_raw/train_FD001.txt',
    sep=r'\s+',
    header=None,
    names=columns,
    usecols=range(26)  # ignore any trailing phantom columns from whitespace
)

print(train.shape)
print(train.head())

# Calculate RUL: for each engine, find its max cycle (failure point)
max_cycles = train.groupby('unit_number')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']

train = train.merge(max_cycles, on='unit_number')
train['RUL'] = train['max_cycle'] - train['time_cycles']
train = train.drop('max_cycle', axis=1)

print(train[['unit_number', 'time_cycles', 'RUL']].head(10))

# Load test data (same column structure as train)
test = pd.read_csv(
    'data/raw/cmapss_raw/test_FD001.txt',
    sep=r'\s+',
    header=None,
    names=columns,
    usecols=range(26)
)

# Load the true RUL values (one value per engine, for the LAST cycle in the test file)
test_rul = pd.read_csv(
    'data/raw/cmapss_raw/RUL_FD001.txt',
    sep=r'\s+',
    header=None,
    names=['RUL']
)
test_rul['unit_number'] = test_rul.index + 1  # engine numbers start at 1, not 0

print(test.shape)
print(test_rul.shape)
print(test_rul.head())

# Save cleaned data to data/processed/
train.to_csv('data/processed/train_FD001_clean.csv', index=False)
test.to_csv('data/processed/test_FD001_clean.csv', index=False)
test_rul.to_csv('data/processed/test_FD001_rul.csv', index=False)

print("Saved cleaned files to data/processed/")