import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import xgboost as xgb
import os

# Set up paths
PROJECT_DIR = r"C:\Users\steph\spy-volatility-regime"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "spy_clean.csv")
VIZ_OUTPUT = os.path.join(PROJECT_DIR, "results")
os.makedirs(VIZ_OUTPUT, exist_ok=True)

# Load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

cols_to_clean = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)
df.dropna(inplace=True)

print(f"Data loaded: {len(df)} rows, date range: {df['Date'].min()} to {df['Date'].max()}")

# FEATURE ENGINEERING - OPTIMIZED SET
print("OPTIMIZED 11-FEATURE MODEL")

# Basic features
df['return_1d'] = df['Close'].pct_change()
df['return_5d'] = df['Close'].pct_change(5)
df['volatility_5d'] = df['return_1d'].rolling(5).std()
df['volatility_10d'] = df['return_1d'].rolling(10).std()

# ORIGINAL 8 VOLATILITY FEATURES (Run 2 version)
print("\nEngineering original 8 features...")
df['vol_expansion'] = df['volatility_5d'] / df['volatility_10d']
df['vol_expansion_3d'] = df['return_1d'].rolling(3).std() / df['volatility_5d']
df['vol_persistence'] = df['vol_expansion'].rolling(3).mean()
df['vol_acceleration'] = df['vol_expansion'] - df['vol_expansion'].shift(1)
df['high_vol_days'] = (df['vol_expansion'] > 1.2).rolling(5).mean()  # Mean, not sum
df['vol_regime_shift'] = (df['vol_expansion'] > df['vol_expansion'].rolling(20).quantile(0.8)).astype(int)
df['intraday_vol'] = (df['High'] - df['Low']) / df['Close']
df['intraday_vol_expansion'] = df['intraday_vol'] / df['intraday_vol'].rolling(10).mean()

# TOP 3 ENTROPY FEATURES (based on consistent importance rankings)
print("Engineering top 3 entropy features...")

# 1. tight_candles - consistently ranked #2-4 across folds
pct_band = 0.01
df['tight_candles'] = ((df['High'] - df['Low']) / df['Close'] < pct_band).rolling(10).sum() / 10

# 2. return_entropy - consistently top 5
def rolling_entropy(series, window):
    def entropy(x):
        if len(x) < 3:
            return 0
        hist, _ = np.histogram(x, bins=5, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))
    return series.rolling(window).apply(entropy, raw=False)

df['return_entropy'] = rolling_entropy(df['return_1d'], 10)

# 3. vol_of_vol - volatility of volatility persistence
df['vol_of_vol'] = df['vol_persistence'].rolling(10).std()

df.dropna(inplace=True)

print(f"Feature engineering complete")
print(f"Dataset shape: {df.shape}")

# Target
LOOKFORWARD_DAYS = 5
df['future_max_vol'] = df['vol_expansion'].rolling(LOOKFORWARD_DAYS).max().shift(-LOOKFORWARD_DAYS + 1)
df.dropna(subset=['future_max_vol'], inplace=True)

# OPTIMIZED FEATURE SET: 11 FEATURES
feature_cols = [
    # Original 8
    'vol_expansion',
    'vol_expansion_3d',
    'vol_persistence',
    'vol_acceleration',
    'high_vol_days',
    'vol_regime_shift',
    'intraday_vol',
    'intraday_vol_expansion',
    # Top 3 Entropy
    'tight_candles',
    'return_entropy',
    'vol_of_vol'
]

print(f"OPTIMIZED FEATURE SET: {len(feature_cols)} features")
print("\nOriginal volatility features (8):")
for feat in feature_cols[:8]:
    print(f"  - {feat}")
print("\nTop entropy features (3):")
for feat in feature_cols[8:]:
    print(f"  - {feat}")

# TIME SERIES CROSS-VALIDATION

X = df[feature_cols].copy()

tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
class_results = {0: [], 1: [], 2: [], 3: []}

print("TIME SERIES CROSS-VALIDATION")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    
    print(f"FOLD {fold+1}")
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Calculate thresholds ONLY on training set
    train_vol = df_train['future_max_vol']
    thresholds = train_vol.quantile([0.6, 0.8, 0.95]).values
    
    print(f"Thresholds: {thresholds}")
    
    # Assign class labels
    conditions = [
        df_train['future_max_vol'] <= thresholds[0],
        (df_train['future_max_vol'] > thresholds[0]) & (df_train['future_max_vol'] <= thresholds[1]),
        (df_train['future_max_vol'] > thresholds[1]) & (df_train['future_max_vol'] <= thresholds[2]),
        df_train['future_max_vol'] > thresholds[2]
    ]
    df_train['vol_regime_target'] = np.select(conditions, [0, 1, 2, 3], default=0)
    
    conditions = [
        df_test['future_max_vol'] <= thresholds[0],
        (df_test['future_max_vol'] > thresholds[0]) & (df_test['future_max_vol'] <= thresholds[1]),
        (df_test['future_max_vol'] > thresholds[1]) & (df_test['future_max_vol'] <= thresholds[2]),
        df_test['future_max_vol'] > thresholds[2]
    ]
    df_test['vol_regime_target'] = np.select(conditions, [0, 1, 2, 3], default=0)
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_train = df_train['vol_regime_target'].astype(int)
    y_test = df_test['vol_regime_target'].astype(int)
    
    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True).sort_index())
    
    # Train model
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
    
    # Per-class metrics
    print("\nPer-Class Performance:")
    for class_idx in range(4):
        mask_true = y_test == class_idx
        
        if mask_true.sum() > 0:
            precision = precision_score(y_test == class_idx, y_pred == class_idx, zero_division=0)
            recall = recall_score(y_test == class_idx, y_pred == class_idx, zero_division=0)
            f1 = f1_score(y_test == class_idx, y_pred == class_idx, zero_division=0)
            print(f"  Class {class_idx}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={mask_true.sum()})")
            
            class_results[class_idx].append({
                'fold': fold + 1,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    
    # Binary high-vol detection
    y_test_highvol = y_test.isin([2, 3]).astype(int)
    y_pred_highvol = pd.Series(y_pred).isin([2, 3]).astype(int)
    
    precision = precision_score(y_test_highvol, y_pred_highvol, zero_division=0)
    recall = recall_score(y_test_highvol, y_pred_highvol, zero_division=0)
    f1 = f1_score(y_test_highvol, y_pred_highvol, zero_division=0)
    
    print("\nHigh Vol Regime Detection (Classes 2+3):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    # Feature importances
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    fold_results.append({
        'fold': fold + 1,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# COMPARISON SUMMARY

print("OPTIMIZED 11-FEATURE MODEL RESULTS")

results_df = pd.DataFrame(fold_results)
print(f"\nHigh-Vol Detection (Classes 2+3):")
print(f"  Mean F1:        {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}")
print(f"  Mean Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
print(f"  Mean Recall:    {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")

print(f"\nPer-Class Performance Across All Folds:")
for class_idx in range(4):
    if len(class_results[class_idx]) > 0:
        class_df = pd.DataFrame(class_results[class_idx])
        print(f"\n  Class {class_idx}:")
        print(f"    Mean F1:        {class_df['f1'].mean():.3f} ± {class_df['f1'].std():.3f}")
        print(f"    Mean Precision: {class_df['precision'].mean():.3f} ± {class_df['precision'].std():.3f}")
        print(f"    Mean Recall:    {class_df['recall'].mean():.3f} ± {class_df['recall'].std():.3f}")

# Comparison table
print(f"\n{'='*60}")
print("COMPARISON: Run 2 (8 feat) vs Optimized (11 feat)")
print("\n                          Run 2    |  Optimized (11 feat)")
print(f"High-Vol F1:              0.711    |  {results_df['f1'].mean():.3f}")
print(f"Class 0 F1:               ~0.85    |  {pd.DataFrame(class_results[0])['f1'].mean():.3f}")
print(f"Class 1 F1:               ~0.15    |  {pd.DataFrame(class_results[1])['f1'].mean():.3f}")
print(f"Class 2 F1:               ~0.35    |  {pd.DataFrame(class_results[2])['f1'].mean():.3f}")
print(f"Class 3 F1:               ~0.40    |  {pd.DataFrame(class_results[3])['f1'].mean():.3f}")

# Determine verdict
highvol_f1 = results_df['f1'].mean()
class1_f1 = pd.DataFrame(class_results[1])['f1'].mean()

print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")

if highvol_f1 >= 0.60 and class1_f1 >= 0.20:
    print("SUCCESS! Found the sweet spot:")
    print(f"   - High-vol F1 = {highvol_f1:.3f} (target: >0.60)")
    print(f"   - Class 1 F1 = {class1_f1:.3f} (target: >0.20)")
    print("\n   This 11-feature model balances Class 1 improvement with overall performance!")
elif highvol_f1 >= 0.60:
    print("PARTIAL SUCCESS:")
    print(f"   - High-vol F1 = {highvol_f1:.3f} (preserved)")
    print(f"   - Class 1 F1 = {class1_f1:.3f} (still low)")
    print("\n   Recommend: Use Run 2 (8 features) as primary model")
elif class1_f1 >= 0.20:
    print("PARTIAL SUCCESS:")
    print(f"   - High-vol F1 = {highvol_f1:.3f} (degraded)")
    print(f"   - Class 1 F1 = {class1_f1:.3f} (improved)")
    print("\n   Trade-off: Class 1 improvement still costs too much overall performance")
else:
    print("NO IMPROVEMENT:")
    print(f"   - High-vol F1 = {highvol_f1:.3f}")
    print(f"   - Class 1 F1 = {class1_f1:.3f}")

# Save summary
summary_path = os.path.join(VIZ_OUTPUT, '3entropy8base_summary.txt')
with open(summary_path, 'w') as f:
    f.write("11-FEATURE MODEL RESULTS\n")
    
    f.write("FEATURE SET:\n")
    f.write("Original volatility features (8):\n")
    for feat in feature_cols[:8]:
        f.write(f"  - {feat}\n")
    f.write("\nTop entropy features (3):\n")
    for feat in feature_cols[8:]:
        f.write(f"  - {feat}\n")
    
    f.write("RESULTS\n")
    
    f.write("High-Vol Detection (Classes 2+3):\n")
    f.write(f"  Mean F1:        {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}\n")
    f.write(f"  Mean Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}\n")
    f.write(f"  Mean Recall:    {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}\n\n")
    
    f.write("Per-Class Performance:\n")
    for class_idx in range(4):
        if len(class_results[class_idx]) > 0:
            class_df = pd.DataFrame(class_results[class_idx])
            f.write(f"\n  Class {class_idx}:\n")
            f.write(f"    Mean F1:        {class_df['f1'].mean():.3f} ± {class_df['f1'].std():.3f}\n")
            f.write(f"    Mean Precision: {class_df['precision'].mean():.3f} ± {class_df['precision'].std():.3f}\n")
            f.write(f"    Mean Recall:    {class_df['recall'].mean():.3f} ± {class_df['recall'].std():.3f}\n")
    
    f.write("COMPARISON\n")
    f.write("                          Run 2    |  Optimized\n")
    f.write(f"High-Vol F1:              0.711    |  {results_df['f1'].mean():.3f}\n")
    f.write(f"Class 1 F1:               ~0.15    |  {pd.DataFrame(class_results[1])['f1'].mean():.3f}\n")

print(f"\nSaved: optimized_11feat_summary.txt")
print(f"\nResults saved to: {VIZ_OUTPUT}")
print("OPTIMIZED MODEL TESTING COMPLETE!")
