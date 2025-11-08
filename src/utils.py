"""
Utility Functions for SPY Volatility Regime Detection
======================================================
Helper functions for data loading, feature engineering, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def load_spy_data(filepath='../data/spy_clean.csv'):
    """
    Load and clean SPY OHLCV data.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame : Cleaned SPY data with datetime index
    """
    # Try different path configurations
    if not os.path.exists(filepath):
        filepath = 'data/spy_clean.csv'
    if not os.path.exists(filepath):
        filepath = 'spy_clean.csv'
    
    df = pd.read_csv(filepath)
    
    # Clean numeric columns
    cols_to_clean = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    
    return df


def engineer_features(df, lookforward_days=5):
    """
    Engineer volatility features for regime detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV columns
    lookforward_days : int
        Number of days to look forward for target
        
    Returns:
    --------
    pd.DataFrame : DataFrame with engineered features
    list : Feature column names
    """
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
    
    # Target
    df['future_max_vol'] = df['vol_expansion'].shift(-1).rolling(lookforward_days).max()
    df.dropna(subset=['future_max_vol'], inplace=True)
    
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
    
    return df, feature_cols


def assign_regime_classes(df, thresholds):
    """
    Assign 4-class regime labels based on thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'future_max_vol' column
    thresholds : array-like
        Three threshold values [60th, 80th, 95th percentile]
        
    Returns:
    --------
    np.ndarray : Class labels (0, 1, 2, 3)
    """
    conditions = [
        df['future_max_vol'] <= thresholds[0],
        (df['future_max_vol'] > thresholds[0]) & (df['future_max_vol'] <= thresholds[1]),
        (df['future_max_vol'] > thresholds[1]) & (df['future_max_vol'] <= thresholds[2]),
        df['future_max_vol'] > thresholds[2]
    ]
    return np.select(conditions, [0, 1, 2, 3], default=0)


def plot_confusion_matrix(y_true, y_pred, classes=['Low', 'Sideways', 'Transition', 'Extreme'], 
                         normalize=False, title='Confusion Matrix', figsize=(8, 6)):
    """
    Plot confusion matrix with optional normalization.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list
        Class names
    normalize : bool
        Whether to normalize by row (true labels)
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    

def plot_feature_importance(model, feature_names, top_n=None, figsize=(10, 6)):
    """
    Plot feature importances from trained model.
    
    Parameters:
    -----------
    model : XGBoost model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to plot (None = all)
    figsize : tuple
        Figure size
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    if top_n:
        importance_df = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()


def plot_regime_distribution(df, figsize=(12, 6)):
    """
    Plot distribution of volatility regimes over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Date and vol_regime_target columns
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Time series plot
    colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
    for regime in range(4):
        mask = df['vol_regime_target'] == regime
        ax1.scatter(df[mask]['Date'], df[mask]['Close'], 
                   c=colors[regime], s=10, alpha=0.5, 
                   label=f'Class {regime}')
    ax1.set_ylabel('SPY Close Price')
    ax1.set_title('Volatility Regimes Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Distribution bar chart
    regime_counts = df['vol_regime_target'].value_counts().sort_index()
    ax2.bar(regime_counts.index, regime_counts.values, 
           color=[colors[i] for i in regime_counts.index])
    ax2.set_xlabel('Regime Class')
    ax2.set_ylabel('Count')
    ax2.set_title('Regime Distribution')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['Low Vol', 'Sideways', 'Transition', 'Extreme'])
    
    plt.tight_layout()


def print_fold_summary(fold, y_train, y_test, y_pred, thresholds):
    """
    Print summary statistics for a CV fold.
    
    Parameters:
    -----------
    fold : int
        Fold number
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    y_pred : array-like
        Predicted labels
    thresholds : array-like
        Threshold values used
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")
    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")
    print(f"Thresholds: {thresholds}")
    
    print(f"\nTrain distribution:")
    for cls in range(4):
        count = (y_train == cls).sum()
        pct = 100 * count / len(y_train)
        print(f"  Class {cls}: {count:3d} ({pct:4.1f}%)")
    
    print(f"\nTest distribution:")
    for cls in range(4):
        count = (y_test == cls).sum()
        pct = 100 * count / len(y_test)
        print(f"  Class {cls}: {count:3d} ({pct:4.1f}%)")


# Class name mappings
CLASS_NAMES = {
    0: 'Low Volatility (Compression)',
    1: 'Sideways Drift',
    2: 'Transitional',
    3: 'Extreme Volatility'
}

CLASS_DESCRIPTIONS = {
    0: 'Market compression - low realized volatility, tight ranges',
    1: 'Sideways drift - lack of directional commitment, choppy action',
    2: 'Transitional - building momentum, increasing volatility',
    3: 'Extreme - tail events, market stress, high realized volatility'
}