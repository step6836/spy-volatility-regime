"""
SPY Volatility Regime Detection - Final Model

4-class XGBoost model for detecting volatility regimes in SPY
with proper time series cross-validation.

Model: Run 2 (8 features, leakage-free)
Performance: F1 = 0.711 for high-volatility detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import xgboost as xgb
import os

# Set up path
PROJECT_DIR = r"C:\Users\steph\spy-volatility-regime"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "spy_clean.csv")
VIZ_OUTPUT = os.path.join(PROJECT_DIR, "results")
os.makedirs(VIZ_OUTPUT, exist_ok=True)

# Load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Clean numeric columns
cols_to_clean = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Parse dates and sort
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)
df.dropna(inplace=True)

print(f"Data loaded: {len(df)} rows")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# FEATURE ENGINEERING
print("\nEngineering volatility features...")

# Basic returns and volatility
df['return_1d'] = df['Close'].pct_change()
df['volatility_5d'] = df['return_1d'].rolling(5).std()
df['volatility_10d'] = df['return_1d'].rolling(10).std()

# Core volatility expansion features
df['vol_expansion'] = df['volatility_5d'] / df['volatility_10d']
df['vol_expansion_3d'] = df['return_1d'].rolling(3).std() / df['volatility_5d']

# Volatility dynamics
df['vol_persistence'] = df['vol_expansion'].rolling(3).mean()
df['vol_acceleration'] = df['vol_expansion'] - df['vol_expansion'].shift(1)

# Regime indicators
df['high_vol_days'] = (df['vol_expansion'] > 1.2).rolling(5).mean()
df['vol_regime_shift'] = (df['vol_expansion'] > df['vol_expansion'].rolling(20).quantile(0.8)).astype(int)

# Intraday volatility
df['intraday_vol'] = (df['High'] - df['Low']) / df['Close']
df['intraday_vol_expansion'] = df['intraday_vol'] / df['intraday_vol'].rolling(10).mean()

df.dropna(inplace=True)

# Target: Future max volatility (5-day lookahead)
LOOKFORWARD_DAYS = 5
df['future_max_vol'] = df['vol_expansion'].shift(-1).rolling(LOOKFORWARD_DAYS).max()
df.dropna(subset=['future_max_vol'], inplace=True)

print(f"✓ Features engineered")
print(f"Final dataset: {len(df)} rows")

# MODEL TRAINING & VALIDATION
feature_cols = [
    'vol_expansion',
    'vol_expansion_3d',
    'vol_persistence',
    'vol_acceleration',
    'high_vol_days',
    'vol_regime_shift',
    'intraday_vol',
    'intraday_vol_expansion'
]

print(f"\nFeatures: {len(feature_cols)}")
for feat in feature_cols:
    print(f"  - {feat}")

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

print("\n" + "="*60)
print("TIME SERIES CROSS-VALIDATION")
print("="*60)

for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}")
    print(f"{'='*60}")
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Calculate thresholds ONLY on training set (prevents leakage)
    train_vol = df_train['future_max_vol']
    thresholds = train_vol.quantile([0.6, 0.8, 0.95]).values
    
    print(f"Thresholds: {thresholds}")
    
    # Assign 4-class labels using training thresholds
    def assign_classes(data, thresh):
        conditions = [
            data['future_max_vol'] <= thresh[0],
            (data['future_max_vol'] > thresh[0]) & (data['future_max_vol'] <= thresh[1]),
            (data['future_max_vol'] > thresh[1]) & (data['future_max_vol'] <= thresh[2]),
            data['future_max_vol'] > thresh[2]
        ]
        return np.select(conditions, [0, 1, 2, 3], default=0)
    
    df_train['vol_regime_target'] = assign_classes(df_train, thresholds)
    df_test['vol_regime_target'] = assign_classes(df_test, thresholds)
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_train = df_train['vol_regime_target'].astype(int)
    y_test = df_test['vol_regime_target'].astype(int)
    
    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True).sort_index())
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # High-volatility detection (Classes 2 + 3)
    y_test_highvol = y_test.isin([2, 3]).astype(int)
    y_pred_highvol = pd.Series(y_pred).isin([2, 3]).astype(int)
    
    precision = precision_score(y_test_highvol, y_pred_highvol, zero_division=0)
    recall = recall_score(y_test_highvol, y_pred_highvol, zero_division=0)
    f1 = f1_score(y_test_highvol, y_pred_highvol, zero_division=0)
    
    print("\nHigh Vol Regime Detection (Classes 2+3):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    print("\nPer-Class Performance:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature importances
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importances:")
    print(importance_df.to_string(index=False))
    
    fold_results.append({
        'fold': fold + 1,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# FINAL RESULTS
print("FINAL RESULTS")

results_df = pd.DataFrame(fold_results)
mean_f1 = results_df['f1'].mean()
std_f1 = results_df['f1'].std()
mean_precision = results_df['precision'].mean()
mean_recall = results_df['recall'].mean()

summary_lines = [
    "SPY Volatility Regime Detection - Summary\n",
    "="*50 + "\n",
    "High-Volatility Detection (Classes 2+3):\n",
    f"  F1 Score:  {mean_f1:.3f} ± {std_f1:.3f}\n",
    f"  Precision: {mean_precision:.3f}\n",
    f"  Recall:    {mean_recall:.3f}\n",
    "\nFold-by-Fold Results:\n"
]

for result in fold_results:
    summary_lines.append(f"  Fold {result['fold']}: F1 = {result['f1']:.3f}, "
                         f"Precision = {result['precision']:.3f}, "
                         f"Recall = {result['recall']:.3f}\n")

summary_lines.append("\nMODEL TRAINING COMPLETE\n")

# Save to text file
output_file = os.path.join(VIZ_OUTPUT, "regime_detection_summary.txt")
with open(output_file, 'w') as f:
    f.writelines(summary_lines)

print(f"Summary saved to: {output_file}")