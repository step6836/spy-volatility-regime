# SPY Volatility Regime Detection

**Multi-class XGBoost classifier for detecting volatility regimes in SPY with rigorous time series validation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

This project implements a production-ready volatility regime detector for SPY (S&P 500 ETF) that identifies high-volatility periods **5 days in advance** with **F1 = 0.654**, enabling proactive risk management for portfolio allocation.

**Key Insight:** Entropy-based features (trendlessness, momentum collapse) improve detection of sideways regimes (Class 1) by ~2x but inherently trade off high-volatility detection due to overlapping feature space.

---

## Performance

| Metric                   | Value              |
|---------------------------|------------------|
| **High-Volatility F1**    | **0.654 ± 0.045** |
| Precision                 | 0.698            |
| Recall                    | 0.620            |
| Low-Volatility F1         | 0.840            |
| Dataset                   | SPY 2019-2024    |

**Per-Class Performance:**

- **Class 0 (Compression):** F1 = 0.840 - Model excels at identifying low-volatility periods
- **Class 1 (Sideways Drift):** F1 = 0.358 - Challenging regime with temporal structure
- **Class 2 (Transitional):** F1 = 0.374 - Building momentum periods
- **Class 3 (Extreme):** F1 = 0.350 - Rare tail events (~5% of data)

---

## Research Contribution

### Problem Discovery
SHAP analysis revealed Class 1 (sideways drift) exhibited weak, non-directional signals distinct from magnitude-based volatility regimes, suggesting different feature requirements.

### Hypothesis Testing
Engineered 9 entropy-based features (tight candles, return entropy, momentum collapse indicators) to capture "indecision" vs "volatility magnitude."

### Key Findings
- **Feature Space Conflict:** Entropy features improved Class 1 detection (F1: 0.15 → 0.36, +140%) but degraded high-volatility detection (F1: 0.711 → 0.518, -27%).  
- **Cohen's d Analysis:**  
  - `return_entropy`: d = 0.15 (overlap with Classes 2/3)  
  - `vol_of_vol`: d = 0.12 (no separation)  
  - `tight_candles`: d = 1.06 (only useful feature)  

**Tested Approaches (6 iterations):**

| Approach                       | F1 Score |
|--------------------------------|----------|
| All entropy features (+9)       | 0.435    |
| Top 3 entropy features          | 0.557    |
| Best entropy feature only       | 0.511    |
| Two-stage hierarchical model    | 0.463    |
| 3-class restructure             | 0.379    |
| Conditional/gated features      | 0.518    |

**Conclusion:** Selected **8-feature model** (F1 = 0.654) for production. Entropy features inherently trade off sideways vs. high-volatility detection.

---

## Technical Architecture

### Model
- **Algorithm:** XGBoost (multi-class classification)
- **Classes:** 4 volatility regimes (compression, sideways, transitional, extreme)
- **Prediction Window:** 5-day lookahead
- **Validation:** TimeSeriesSplit (5-fold) with fold-specific threshold calculation

### Features (8 Total)

**Core Volatility Dynamics:**
- `vol_expansion`: 5d/10d volatility ratio (primary signal)
- `vol_expansion_3d`: Short-term expansion context
- `vol_persistence`: 3-day rolling mean of expansion (trend strength)
- `vol_acceleration`: Day-over-day expansion change (momentum)

**Regime Indicators:**
- `high_vol_days`: Proportion of days with expansion > 1.2 (clustering)
- `vol_regime_shift`: Binary flag for 80th percentile breach (regime change)

**Intraday Signals:**
- `intraday_vol`: High-low range relative to close (single-bar volatility)
- `intraday_vol_expansion`: Intraday vol vs 10-day average (expansion context)

### Leakage Prevention
- Thresholds calculated **per-fold on training data only**
- No future information used in feature engineering
- Proper time series split (no shuffling)

---

## Project Structure
```
spy-volatility-regime/
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── .gitignore 
├── LICENSE # Project license
│
├── data/ # Cleaned data
│ ├── README.md # Data description
│ └── spy_clean.csv # SPY OHLCV data (2019-2024)
│
├── src/ # Source code
│ ├── regime_detection.py # Main model implementation
│ └── utils.py # Helper functions
│
├── notebooks/ # Exploratory notebooks
│ ├── 3entropy8base.py # Entropy feature exploration + base
│ └── 5entropy8base.py # Top entropy feature + base
│
├── results/ # Model outputs / summaries
│ ├── 3entropy8base_summary.txt
│ ├── 5entropy8base_summary.txt
│ └── regime_detection_summary.txt
│
└── visualizations/ # Plots and visual analysis
├── shap_class_*.png # SHAP force plots per class
├── feature_correlation_heatmap.png
└── entropy_overlap_diagnosis.png # Cohen's d analysis
```
---

## Getting Started

### Prerequisites
```bash
pip install pandas numpy xgboost scikit-learn matplotlib shap
```

### Quick Start
```python
python regimedetection.py
```

### Expected Output
```
FOLD 1-5 Results:
  High-Vol F1: 0.654 ± 0.045
  Class 0 F1:  0.840
  Class 1 F1:  0.358
  Class 2 F1:  0.374
  Class 3 F1:  0.350
```

---

## Use Cases

**Portfolio Risk Management:**
- Reduce equity exposure 5 days before high-volatility periods
- Increase cash allocations during compression (Class 0)
- Rebalance during regime transitions (Class 2)

**Options Trading:**
- Long volatility strategies ahead of Class 2/3 regimes
- Short volatility during Class 0 compression

**Backtesting:**
- Filter trading signals by volatility regime
- Adjust position sizing based on predicted regime

---

## Model Interpretability

### SHAP Insights

**Class 0 (Low Volatility):**
- Strong negative signal from `vol_expansion` (< 1.0)
- Clean, magnitude-based separation
- Low `vol_regime_shift` reinforces stability

**Class 1 (Sideways Drift):**
- Weak, muddy signals across all features
- No clear directional pattern (explains low F1)
- Temporal structure not captured by volatility features

**Class 2 (Transitional):**
- Moderate `vol_expansion` (1.2-1.4)
- `vol_acceleration` shows building momentum
- `intraday_vol_expansion` captures increasing ranges

**Class 3 (Extreme):**
- Very high `vol_expansion` (> 1.4)
- Strong `vol_regime_shift` signal
- Rare events (5% of data) → low recall expected

---

## Benchmark Comparison

| Approach | F1 Score | Notes |
| Naive Threshold (VIX > 20) | 0.45-0.55 | Simple baseline |
| GARCH Models | 0.55-0.65 | Statistical approach |
| **This Model (XGBoost)** | **0.654** | 8 engineered features |
| Academic Research | 0.60-0.75 | Published papers (often 10+ years data) |
| Deep Learning (LSTM) | 0.70-0.80 | Requires 10+ years data |

**Industry Context:** Most production regime detectors achieve F1 = 0.60-0.70 on 5-10 years of data.

---

## Experimental Results

### Entropy Feature Exploration

**Motivation:** Class 1 (sideways drift) showed weak SHAP signals, suggesting need for temporal/entropy features rather than volatility magnitude.

**Features Tested:**
- `tight_candles`: Proportion of days with range < 1%
- `return_entropy`: Shannon entropy of return distribution
- `trendlessness`: Frequency of direction changes
- `momentum_collapse`: Declining trend strength
- `vol_of_vol`: Volatility of volatility persistence
- 4 additional entropy indicators

**Results Summary:**

| Model | High-Vol F1 | Class 1 F1 | Change |
| Baseline (8 feat) | 0.654 | 0.358 | - |
| +9 entropy feat | 0.435 | 0.304 | -33% / -15% |
| +3 top entropy | 0.557 | 0.329 | -15% / -8% |
| +1 best entropy | 0.511 | 0.366 | -22% / +2% |

**Diagnostic Analysis:**
- Misclassification pattern: 47% of Class 2/3 samples pushed to Class 0 (not Class 1)
- Cohen's d showed entropy features overlap between regimes
- Conditional gating (entropy × volatility level) still degraded performance by 27%

**Decision:** Retained 8-feature baseline model as production version.

---

## Lessons Learned

1. **Proper validation is critical** - Fold-specific threshold calculation prevents 10-15% optimistic bias
2. **Feature engineering trade-offs are real** - Not all improvements are additive
3. **Interpretability guides iteration** - SHAP analysis identified which classes needed attention
4. **Know when to stop** - Tested 6 approaches; all showed same trade-off pattern
5. **Research findings matter** - Negative results are valuable (feature space conflicts documented)

---

## Future Work

**Potential Improvements:**
- Incorporate macroeconomic features (VIX, Fed rates, yield curve)
- Extend data to 10+ years (more Class 3 samples)
- Test ensemble methods (XGBoost + LightGBM + CatBoost)
- Explore more elegant hierarchical models (binary high/low → sub-classification)
- Alternative target: Predict volatility percentile (regression) vs regime (classification)

**Not Recommended:**
- Aggressive hyperparameter tuning (marginal gains, risks overfitting)
- Adding more entropy features (established trade-off pattern)
- Deep learning without 10+ years data (insufficient samples)

---

## Technologies Used

- **Python 3.8+**
- **XGBoost 2.0+** - Gradient boosting for classification
- **scikit-learn** - TimeSeriesSplit, metrics
- **SHAP** - Model interpretability and feature analysis
- **pandas/numpy** - Data manipulation
- **matplotlib** - Visualization

---

## Author

**Stephanie H.**
- Psychology Major (Neuroscience Emphasis) → Data Science
- [https://www.linkedin.com/in/stephaniehur/](#) | [https://github.com/step6836](#) 

---

## License

MIT License - feel free to use for educational purposes with attribution.

---

## Acknowledgments

- Data: Yahoo Finance (SPY OHLCV)
- Inspiration: Academic research on financial regime detection
- Validation methodology: Best practices from Kaggle time series competitions
