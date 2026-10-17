# Turbofan Engine Remaining Useful Life (RUL) Prediction

Predicting how many operational cycles a jet engine has left before failure, using NASA's C-MAPSS turbofan degradation simulation dataset.

## Problem

Companies operating industrial machinery face a costly trade-off: replace parts too early and waste money, or wait too long and risk unexpected failure. This project builds and compares several models to predict Remaining Useful Life (RUL) from sensor data, then evaluates how such predictions could inform a real maintenance decision policy.

## Dataset

NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset — specifically the FD001 subset: 100 training engines run to failure, 100 test engines cut off before failure, with 21 sensor readings and 3 operational settings recorded per cycle.

## Approach

1. **Data cleaning** — parsed raw sensor logs, corrected a column-count mismatch in NASA's documentation, computed RUL labels for training data
2. **Exploratory analysis** — identified 7 of 21 sensors with near-zero variance (uninformative, excluded from modeling)
3. **Feature engineering** — added rolling averages to capture degradation trends, not just noisy snapshots
4. **Modeling** — progressed from a naive baseline through classical ML to deep learning
5. **Error analysis & policy simulation** — evaluated practical maintenance-decision trade-offs, not just raw accuracy

## Results

| Model | MAE (cycles) |
|---|---|
| Naive baseline (predict average) | 40.88 |
| Linear Regression | 25.59 |
| Random Forest | 24.72 |
| XGBoost | 23.17 |
| **LSTM (best)** | **20.20** |

MAE = Mean Absolute Error, in engine cycles. Lower is better.

## Key Findings

- **Feature engineering mattered as much as model complexity** — rolling average features dominated XGBoost's top 10 most important features, with `sensor_4`'s rolling average alone accounting for ~36% of total importance.
- **Initial LSTM training failed silently** — an early attempt scored worse than the naive baseline (67.91 MAE) due to a scale mismatch between inputs and targets. Scaling the target variable resolved this, improving results to 20.20 MAE.
- **The model's worst errors were systematically overestimates** — the dangerous direction for maintenance planning, since it implies more remaining life than actually exists.
- **Maintenance threshold is a real trade-off, not a fixed answer**: at a 30-cycle threshold, the model caught 68% of critical engines; raising the threshold to 60 cycles improved the catch rate to 87.2%, at the cost of more unnecessary maintenance flags.

## Project Structure
data/raw/ Original NASA C-MAPSS files
data/processed/ Cleaned datasets, useless sensor list
notebooks/ Saved analysis plots
src/load_data.py Data loading and RUL calculation
src/explore_data.py Exploratory analysis
src/baseline_model.py Naive baseline and linear regression
src/advanced_model.py Random Forest, XGBoost, feature importance
src/lstm_model.py LSTM deep learning model
src/insights.py Error analysis and maintenance policy simulation


## How to Run

```bash
pip install pandas numpy scikit-learn xgboost torch matplotlib seaborn
python src/load_data.py
python src/explore_data.py
python src/baseline_model.py
python src/advanced_model.py
python src/lstm_model.py
python src/insights.py
``` 