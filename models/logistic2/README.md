# NFL Elastic-Net Logistic Regression Model

Advanced logistic regression model for predicting NFL game outcomes using elastic-net regularization and comprehensive feature engineering.

## Overview

This model is designed to provide strong performance on NFL game predictions through:

1. **Elastic-Net Regularization**: Combines L1 and L2 penalties for optimal bias-variance tradeoff
2. **Elo Rating System**: Maintains team strength ratings that update after each game
3. **EPA-Based Metrics**: Leverage Expected Points Added for offensive/defensive efficiency
4. **Success Rate Features**: Play-level success metrics
5. **Rest Days**: Account for days between games
6. **Proper Time-Based Splits**: Prevents data leakage in backtests

## Features

The model uses differential features (home - away) including:

### Core Features
- **Elo Differential**: Difference in Elo ratings with home field advantage
- **Win Percentage Differential**: Historical win rate differences
- **Point Differential**: Average points scored/allowed differences

### Offensive Features
- **EPA per Play**: Expected Points Added efficiency
- **Success Rate**: Percentage of successful plays (EPA > 0)
- **Passing Metrics**: Completion %, yards/attempt, TD rate, INT rate
- **Rushing Metrics**: Yards/carry, TD rate

### Defensive Features
- **Defensive EPA**: EPA allowed per play
- **Defensive Success Rate**: Opponent success rate
- **Defensive Passing/Rushing**: Yards allowed metrics

### Game Context Features
- **Rest Differential**: Days of rest difference between teams
- **Divisional Game**: Indicator for divisional matchups
- **Pace**: Plays per game

### Optional Features
- **Market Spread**: Vegas line (if `use_market_spread=True`)

## Model Architecture

```
Elastic-Net Logistic Regression
├── Penalty: α * [L1_ratio * ||w||₁ + (1 - L1_ratio) * ||w||₂²]
├── Cross-Validation: 5-fold CV to select best (C, L1_ratio)
├── Scoring: Brier Score (optimizes calibration)
└── Solver: SAGA (handles elastic-net efficiently)
```

## Usage

### Training and Testing on 2024 Season (Weeks 6-18)

```python
from logistic_regression import NFLElasticNetLogisticRegression

# Initialize model
model = NFLElasticNetLogisticRegression(
    train_seasons=[2022, 2023],
    test_season=2024,
    l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    cv_folds=5,
    random_state=42
)

# Load data
model.load_data()

# Train model
model.train()

# Generate comprehensive report
model.generate_report(
    output_dir='results',
    start_week=6,
    end_week=18
)
```

### Quick Test Script

```bash
cd models/logistic2
python s24_test.py
```

### Making Predictions

```python
# Predict a specific game
prediction = model.predict_game(
    home_team='KC',
    away_team='BUF',
    season=2024,
    week=10
)

print(f"Predicted Winner: {prediction['predicted_winner']}")
print(f"Home Win Prob: {prediction['home_win_probability']:.1%}")
print(f"Confidence: {prediction['confidence']:.1%}")
```

## Output Files

The model generates several output files in the `results/` directory:

- **elasticnet_2024_game_results.csv**: Game-by-game predictions and probabilities
- **elasticnet_2024_weekly_results.csv**: Weekly accuracy summary
- **elasticnet_2024_summary.csv**: Overall model performance metrics
- **calibration_curve.png**: Calibration plot (predicted vs actual probabilities)
- **weekly_performance.png**: Week-by-week accuracy visualization

## Performance Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Percentage of correct predictions
- **Log Loss**: Probabilistic loss (lower is better)
- **Brier Score**: Calibration metric (lower is better, perfect = 0)
- **ROC AUC**: Area under ROC curve

## Key Advantages

1. **Strong Regularization**: Elastic-net prevents overfitting on small/medium datasets
2. **Well-Calibrated**: Optimizes Brier score for reliable probability estimates
3. **Feature Engineering**: Incorporates domain knowledge (Elo, EPA, rest)
4. **Leakage-Proof**: Proper time-based splits ensure no future data leakage
5. **Interpretable**: Linear model with clear feature importance

## Hyperparameters

### Regularization
- **C**: Inverse regularization strength (selected by CV)
- **L1_ratio**: Mix of L1/L2 penalty (0 = pure Ridge, 1 = pure Lasso)
  - Tested values: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

### Elo System
- **initial_elo**: Starting rating for teams (default: 1500)
- **elo_k**: K-factor for rating updates (default: 20)
- **home_advantage**: Elo points for home field (default: 65)

## Comparison to Other Models

### vs. Simple Logistic Regression
- ✓ Better generalization through regularization
- ✓ Automatic feature selection via L1 penalty
- ✓ Handles multicollinearity better

### vs. Random Forest / Neural Networks
- ✓ More interpretable (linear coefficients)
- ✓ Better calibrated out-of-the-box
- ✓ Less prone to overfitting on limited data
- ✓ Faster training and prediction

### vs. Poisson-Elo
- ✓ More flexible feature incorporation
- ✓ Better handles non-scoring factors
- ✗ Doesn't directly predict scores (only win/loss)

## Data Requirements

- **nflreadpy**: For loading NFL play-by-play and schedule data
- **Seasons**: 2022-2024 data (expandable)
- **Game Types**: Regular season games only

## Implementation Notes

1. **Elo Updates**: Ratings are updated chronologically during dataset preparation
2. **Feature Scaling**: StandardScaler ensures all features have similar magnitudes
3. **Missing Data**: Filled with 0 (reasonable for differential features)
4. **Cross-Validation**: Stratified K-fold maintains class balance

## Future Enhancements

Potential improvements:
- [ ] Weather data integration
- [ ] QB injury/status tracking
- [ ] Travel distance features
- [ ] Market spread as feature (optional)
- [ ] Player-level injury reports
- [ ] Advanced EPA metrics (situational)
- [ ] Time-decayed Elo ratings

## References

- Elastic-Net Regularization: Zou & Hastie (2005)
- Elo Rating System: Arpad Elo (1960s)
- EPA Metrics: Burke (2010s), nflfastR project
- Calibration: Brier (1950)

## License

See project root LICENSE file.

