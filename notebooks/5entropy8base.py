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

print("RUN 2 + CONDITIONAL ENTROPY FEATURES")
print("\nHypothesis: Entropy features need to be GATED by volatility level")
print("  - Tight candles + LOW vol = Class 1 (sideways)")
print("  - Tight candles + HIGH vol = Class 0 (compression)")

# Load data
print(f"\nLoading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

cols_to_clean = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)
df.dropna(inplace=True)

print(f"Data loaded: {len(df)} rows")

# FEATURE ENGINEERING
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Basic features
df['return_1d'] = df['Close'].pct_change()
df['volatility_5d'] = df['return_1d'].rolling(5).std()
df['volatility_10d'] = df['return_1d'].rolling(10).std()

# RUN 2 ORIGINAL FEATURES (8)
print("\n1. Original Run 2 features (8)...")
df['vol_expansion'] = df['volatility_5d'] / df['volatility_10d']
df['vol_expansion_3d'] = df['return_1d'].rolling(3).std() / df['volatility_5d']
df['vol_persistence'] = df['vol_expansion'].rolling(3).mean()
df['vol_acceleration'] = df['vol_expansion'] - df['vol_expansion'].shift(1)
df['high_vol_days'] = (df['vol_expansion'] > 1.2).rolling(5).mean()
df['vol_regime_shift'] = (df['vol_expansion'] > df['vol_expansion'].rolling(20).quantile(0.8)).astype(int)
df['intraday_vol'] = (df['High'] - df['Low']) / df['Close']
df['intraday_vol_expansion'] = df['intraday_vol'] / df['intraday_vol'].rolling(10).mean()

# ENTROPY RAW FEATURES (for interaction)
print("2. Entropy raw features...")
pct_band = 0.01
df['tight_candles'] = ((df['High'] - df['Low']) / df['Close'] < pct_band).rolling(10).sum() / 10

df['return_sign'] = np.sign(df['return_1d'])
df['direction_changes'] = (df['return_sign'] != df['return_sign'].shift(1)).rolling(10).sum()
df['trendlessness'] = df['direction_changes'] / 10

# CONDITIONAL/INTERACTION FEATURES (NEW)
print("3. Conditional entropy features (gated by volatility)...")

# Class 1 signal: Tight candles + LOW volatility expansion
df['class1_signal'] = ((df['tight_candles'] > 0.5) & 
                       (df['vol_expansion'] < 1.1)).astype(float)

# Class 1 confirmation: Trendless + NOT regime shift
df['class1_trendless'] = ((df['trendlessness'] > 0.5) & 
                          (df['vol_regime_shift'] == 0)).astype(float)

# Compression signal: Tight range + LOW vol expansion (Class 0)
df['compression_tight'] = ((df['tight_candles'] > 0.4) & 
                           (df['vol_expansion'] < 1.0)).astype(float)

# Breakout signal: High intraday vol + HIGH vol expansion (Class 3)
df['breakout_signal'] = ((df['intraday_vol_expansion'] > 1.3) & 
                         (df['vol_expansion'] > 1.2)).astype(float)

# Interaction: vol_expansion * tight_candles
# Low value = sideways drift, High value = volatile compression
df['vol_tight_interaction'] = df['vol_expansion'] * df['tight_candles']

# Clean up
df.drop(['return_sign', 'direction_changes'], axis=1, inplace=True)
df.dropna(inplace=True)

print(f"✓ Feature engineering complete")
print(f"Dataset shape: {df.shape}")

# Target
LOOKFORWARD_DAYS = 5
df['future_max_vol'] = df['vol_expansion'].rolling(LOOKFORWARD_DAYS).max().shift(-LOOKFORWARD_DAYS + 1)
df.dropna(subset=['future_max_vol'], inplace=True)

# FEATURE SETS
# Original 8
original_features = [
    'vol_expansion',
    'vol_expansion_3d',
    'vol_persistence',
    'vol_acceleration',
    'high_vol_days',
    'vol_regime_shift',
    'intraday_vol',
    'intraday_vol_expansion'
]

# Conditional entropy features (5)
conditional_features = [
    'class1_signal',
    'class1_trendless',
    'compression_tight',
    'breakout_signal',
    'vol_tight_interaction'
]

# Combined: 13 features
feature_cols = original_features + conditional_features

print(f"FEATURE SET: {len(feature_cols)} features")
print("\nOriginal (8):")
for feat in original_features:
    print(f"  - {feat}")
print("\nConditional Entropy (5):")
for feat in conditional_features:
    print(f"  - {feat}")

# TIME SERIES CROSS-VALIDATION
X = df[feature_cols].copy()

tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
class_results = {0: [], 1: [], 2: [], 3: []}

print("TIME SERIES CROSS-VALIDATION (4 CLASSES)")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}")
    print(f"{'='*60}")
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Calculate thresholds on training set
    train_vol = df_train['future_max_vol']
    thresholds = train_vol.quantile([0.6, 0.8, 0.95]).values
    
    print(f"Thresholds: {thresholds}")
    
    # Assign 4-class labels
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
    
    # High-vol detection
    y_test_highvol = y_test.isin([2, 3]).astype(int)
    y_pred_highvol = pd.Series(y_pred).isin([2, 3]).astype(int)
    
    precision = precision_score(y_test_highvol, y_pred_highvol, zero_division=0)
    recall = recall_score(y_test_highvol, y_pred_highvol, zero_division=0)
    f1 = f1_score(y_test_highvol, y_pred_highvol, zero_division=0)
    
    print(f"\nHigh Vol Regime Detection (Classes 2+3):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    # Feature importances
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
        'type': ['Original' if f in original_features else 'Conditional' for f in feature_cols]
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Check if conditional features are being used
    conditional_importance = importance_df[importance_df['type'] == 'Conditional']
    if len(conditional_importance) > 0:
        print(f"\nConditional Feature Rankings:")
        for idx, row in conditional_importance.iterrows():
            rank = importance_df.index.get_loc(idx) + 1
            print(f"  #{rank}: {row['feature']} (importance: {row['importance']:.4f})")
    
    fold_results.append({
        'fold': fold + 1,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# FINAL COMPARISON
print("FINAL RESULTS")

results_df = pd.DataFrame(fold_results)
print(f"\nHigh-Vol Detection (Classes 2+3):")
print(f"  Mean F1:        {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}")
print(f"  Mean Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
print(f"  Mean Recall:    {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")

print(f"\nPer-Class Performance:")
for class_idx in range(4):
    if len(class_results[class_idx]) > 0:
        class_df = pd.DataFrame(class_results[class_idx])
        print(f"\n  Class {class_idx}:")
        print(f"    Mean F1:        {class_df['f1'].mean():.3f} ± {class_df['f1'].std():.3f}")
        print(f"    Mean Precision: {class_df['precision'].mean():.3f} ± {class_df['precision'].std():.3f}")
        print(f"    Mean Recall:    {class_df['recall'].mean():.3f} ± {class_df['recall'].std():.3f}")

print("COMPARISON")

highvol_f1 = results_df['f1'].mean()
class1_f1 = pd.DataFrame(class_results[1])['f1'].mean()

print(f"\nRun 2 (8 features):")
print(f"  High-Vol F1: 0.711")
print(f"  Class 1 F1:  ~0.15")

print(f"\nRun 2 + Conditional Features (13 features):")
print(f"  High-Vol F1: {highvol_f1:.3f}")
print(f"  Class 1 F1:  {class1_f1:.3f}")

print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")

if highvol_f1 >= 0.68 and class1_f1 >= 0.20:
    print("SUCCESS! Found the solution:")
    print(f"   - High-vol F1 ≥ 0.68: {highvol_f1:.3f}")
    print(f"   - Class 1 F1 ≥ 0.20:  {class1_f1:.3f}")
    print("\n   Conditional features successfully balance performance!")
    print("   RECOMMENDATION: Use this as final model")
elif highvol_f1 >= 0.68:
    print(f"High-vol preserved ({highvol_f1:.3f}) but Class 1 still weak ({class1_f1:.3f})")
    print("   Conditional features didn't help Class 1 enough")
    print("   RECOMMENDATION: Use Run 2 (8 features) as final")
else:
    print(f"Degraded performance:")
    print(f"   - High-vol F1: {highvol_f1:.3f} (target: ≥0.68)")
    print(f"   - Class 1 F1:  {class1_f1:.3f}")
    print("\n   Conditional features hurt overall performance")

# Save summary
summary_path = os.path.join(VIZ_OUTPUT, '5entropy8base_summary.txt')
with open(summary_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("RUN 2 + CONDITIONAL ENTROPY FEATURES\n")
    f.write("="*60 + "\n\n")
    
    f.write("HYPOTHESIS:\n")
    f.write("Entropy features need to be conditional on volatility level\n\n")
    
    f.write("FEATURES ADDED (5):\n")
    for feat in conditional_features:
        f.write(f"  - {feat}\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("RESULTS\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"High-Vol F1: {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}\n")
    f.write(f"Class 1 F1:  {class1_f1:.3f}\n")
    
    f.write("\nComparison to Run 2:\n")
    f.write(f"  Run 2 High-Vol F1:     0.711\n")
    f.write(f"  Conditional High-Vol:  {highvol_f1:.3f}\n")
    f.write(f"  Change:                {highvol_f1 - 0.711:+.3f}\n")

print(f"\nSaved: conditional_features_summary.txt")
print(f"\nResults saved to: {VIZ_OUTPUT}")
print("\n" + "="*60)
print("CONDITIONAL FEATURES TEST COMPLETE!")
