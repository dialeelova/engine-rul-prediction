import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load cleaned data
train = pd.read_csv('data/processed/train_FD001_clean.csv')
test = pd.read_csv('data/processed/test_FD001_clean.csv')
test_rul = pd.read_csv('data/processed/test_FD001_rul.csv')

with open('data/processed/useless_sensors.txt') as f:
    useless_sensors = f.read().splitlines()

feature_cols = [col for col in train.columns if col.startswith('sensor_') and col not in useless_sensors]
feature_cols += ['op_setting_1', 'op_setting_2', 'op_setting_3']

print("Using features:", len(feature_cols))

# Scale features to 0-1 range
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

# Cap RUL at 125 (standard practice for this dataset)
train['RUL'] = train['RUL'].clip(upper=125)

# Scale RUL to 0-1 as well, for consistent training
rul_scaler = MinMaxScaler()
train['RUL_scaled'] = rul_scaler.fit_transform(train[['RUL']])

def create_sequences(df, feature_cols, seq_length=30, target_col='RUL'):
    sequences = []
    labels = []
    for unit in df['unit_number'].unique():
        engine_data = df[df['unit_number'] == unit].sort_values('time_cycles')
        features = engine_data[feature_cols].values
        rul = engine_data[target_col].values

        if len(features) >= seq_length:
            for i in range(len(features) - seq_length + 1):
                sequences.append(features[i:i+seq_length])
                labels.append(rul[i+seq_length-1])

    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = create_sequences(train, feature_cols, target_col='RUL_scaled')
print("Sequence shape:", X_train_seq.shape)
print("Labels shape:", y_train_seq.shape)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

model = LSTMModel(input_size=len(feature_cols))
print(model)

X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq).view(-1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 60
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.5f}")

def create_test_sequences(df, feature_cols, seq_length=30):
    sequences = []
    unit_ids = []
    for unit in df['unit_number'].unique():
        engine_data = df[df['unit_number'] == unit].sort_values('time_cycles')
        features = engine_data[feature_cols].values

        if len(features) >= seq_length:
            sequences.append(features[-seq_length:])
        else:
            padding = np.repeat(features[0:1], seq_length - len(features), axis=0)
            padded = np.vstack([padding, features])
            sequences.append(padded)
        unit_ids.append(unit)

    return np.array(sequences), unit_ids

X_test_seq, test_unit_ids = create_test_sequences(test, feature_cols)
print("Test sequence shape:", X_test_seq.shape)

model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_seq)
    lstm_predictions = model(X_test_tensor).numpy().flatten()

lstm_predictions = rul_scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()
lstm_predictions = np.clip(lstm_predictions, 0, 125)

true_rul_capped = test_rul.set_index('unit_number').loc[test_unit_ids, 'RUL'].clip(upper=125).values

lstm_mae = mean_absolute_error(true_rul_capped, lstm_predictions)
print(f"LSTM MAE: {lstm_mae:.2f} cycles")
