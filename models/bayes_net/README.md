# Bayesian Network Model for NFL Betting

A probabilistic graphical model implementation for predicting NFL game outcomes using injury data, weather conditions, and temporal factors.

## Overview

The Bayesian network model uses a hierarchical structure to capture the relationships between different factors that influence game outcomes. It focuses on pre-game actionable data to avoid circular logic with spreads or post-game statistics.

## Model Architecture

```
Game Outcome (Home Win)
├── Injury Factors
│   ├── Away Team Key Injuries (QB/RB/WR/TE)
│   └── Home Team Key Injuries (QB/RB/WR/TE)
├── Weather Conditions
│   ├── Wind Speed (affects passing)
│   └── Temperature (affects strategy)
├── Temporal Factors
│   ├── Season Phase (early/mid/late)
│   └── Rest Day Advantages
└── Game Context
    └── Divisional Rivalry
```

## Key Features

- **Discretized Features**: Converts continuous variables to categorical for better probability estimation
- **Conditional Probabilities**: Learned from historical game data
- **Injury Impact Modeling**: Focuses on key position injuries (QB/RB/WR/TE)
- **Weather Integration**: Accounts for wind and temperature effects
- **Temporal Analysis**: Considers season progression and rest advantages

## Performance

### Standard Training Results
- **Accuracy**: 75% on test data (81/108 correct predictions)
- **Training Data**: 544 games from 2023-2024 seasons
- **Key Predictor**: Away team key injuries (0.087 correlation, p=0.042)

### Experimental Results: Weeks 1-5 vs Weeks 6-18

**Training Setup:**
- **Training Data**: 78 games from Weeks 1-5, 2024
- **Test Data**: 194 games from Weeks 6-18, 2024

**Results:**
- **Training Accuracy**: 85.9% (67/78 correct)
- **Test Accuracy**: 47.4% (92/194 correct)
- **Overfitting Gap**: 38.5% difference
- **Baseline**: 53.6% home win rate in test period

**Week-by-Week Performance (Weeks 6-18):**
| Week | Accuracy | Notes |
|------|----------|-------|
| W6   | 42.9%    | Poor start to test period |
| W7   | 66.7%    | Best performing week |
| W8   | 43.8%    | Below baseline |
| W9   | 33.3%    | Worst performing week |
| W10  | 64.3%    | Strong recovery |
| W11  | 50.0%    | Exactly at baseline |
| W12  | 61.5%    | Good performance |
| W13  | 50.0%    | Back to baseline |
| W14  | 30.8%    | Very poor performance |
| W15  | 68.8%    | Excellent performance |
| W16  | 18.8%    | Catastrophic failure |
| W17  | 43.8%    | Below baseline |
| W18  | 43.8%    | Below baseline |

**Key Findings:**
1. **Severe Overfitting**: Model memorized early season patterns but failed to generalize
2. **Extreme Variability**: Performance ranges from 18.8% to 68.8% across weeks
3. **Insufficient Training Data**: 78 games too small for complex Bayesian network
4. **Late Season Collapse**: Performance degrades significantly in final weeks (W16-W18)

**Research Implications:**
- Demonstrates critical need for sufficient training data
- Shows need for regularization to prevent overfitting
- Highlights week-to-week variability in NFL patterns
- Validates need for ensemble approaches or simpler models
- Late season performance suggests model cannot adapt to changing dynamics

### Additional Experiment: Decade Training (2013-2023 → 2024)
Training on 2,863 games from 2013-2023 and testing on 2024 showed **reduced overfitting** (18.9% gap vs 38.5%) but still achieved only **45.6% accuracy**, confirming that data insufficiency was a major factor but the Bayesian Network approach has fundamental limitations for NFL prediction.

## Usage

### Training the Model

#### Standard Training (All Available Data)
```bash
python nfl_bayesian_network.py
```

#### Experimental Training (Weeks 1-5 vs Weeks 6-18)
```bash
python train_weeks_1_5_test_weeks_6_18.py
```
This trains on Weeks 1-5 of 2024 and tests on Weeks 6-18, demonstrating overfitting issues.

### Making Predictions
```bash
# Command line interface
python cli_predictor.py KC BUF 8 --home-injuries 1 --away-injuries 4 --wind 10

# Interactive mode (use cli_predictor.py with arguments)
python cli_predictor.py KC BUF 8 --home-injuries 1 --away-injuries 4 --wind 10
```

### Python Integration
```python
from nfl_bayesian_network import NFLBayesianNetwork, GameFeatures

# Load model
bn = NFLBayesianNetwork()
bn.load_model('nfl_bayesian_model.pkl')

# Create features
features = GameFeatures(
    home_key_injuries=1,
    away_key_injuries=4,
    home_total_injuries=2,
    away_total_injuries=8,
    wind_speed=10,
    temperature=65,
    week=8,
    is_divisional=True,
    home_rest_days=10,
    away_rest_days=7
)

# Predict
prediction = bn.predict(features)
print(f"Home win probability: {prediction.home_win_probability:.1%}")
```

## Input Parameters

### Required
- `home_team`: Home team abbreviation (e.g., "KC")
- `away_team`: Away team abbreviation (e.g., "BUF") 
- `week`: Week number (1-18)

### Optional (with defaults)
- `home_injuries`: Home team key injuries (QB/RB/WR/TE) [default: 0]
- `away_injuries`: Away team key injuries (QB/RB/WR/TE) [default: 0]
- `wind`: Wind speed in mph [default: 5]
- `temp`: Temperature in Fahrenheit [default: 70]
- `divisional`: Is divisional game [default: False]
- `home_rest`: Home team rest days [default: 7]
- `away_rest`: Away team rest days [default: 7]

## Output Interpretation

### Win Probabilities
- **>65%**: Strong bet recommendation
- **55-65%**: Moderate bet recommendation  
- **45-55%**: No clear edge - avoid betting
- **<35%**: Strong bet against

### Confidence Levels
- **High**: Multiple factors align, strong prediction
- **Medium**: Some factors favor one team
- **Low**: Factors are balanced or conflicting

### Betting Recommendations
- **STRONG HOME/AWAY BET**: High confidence recommendation
- **MODERATE HOME/AWAY BET**: Slight edge recommendation
- **NO CLEAR EDGE**: Avoid betting on this game

## Example Output

```
============================================================
NFL GAME PREDICTION
============================================================
BUF @ KC (Week 8)

Home Team Win Probability: 65.2%
Away Team Win Probability: 34.8%

Confidence Level: HIGH

Key Factors:
  1. Away team has 3 more key injuries
  2. Home team has significant rest advantage
  3. High wind (15 mph) affects passing game

Betting Recommendation:
   STRONG HOME BET - High confidence in home team win
============================================================
```

## Correlation Analysis Results

Based on analysis of 544 NFL games, the most predictive factors are:

| Variable | Correlation | Significance | Notes |
|----------|-------------|--------------|-------|
| `away_key_injuries` | 0.087 | p=0.042 | **Most actionable** |
| `away_injured_players` | 0.067 | p=0.121 | Total injury count |
| `home_key_injuries` | 0.063 | p=0.140 | Home team injuries |
| `wind` | 0.053 | p=0.341 | Weather impact |
| `week` | 0.040 | p=0.348 | Season progression |

## Files

- `nfl_bayesian_network.py` - Main Bayesian network implementation
- `correlation_tests.py` - Comprehensive correlation analysis
- `cli_predictor.py` - Command line prediction interface
- `train_weeks_1_5_test_weeks_6_18.py` - Experimental training script (overfitting analysis)

## Running Correlation Analysis

```bash
python correlation_tests.py
```

This will generate:
- Interactive HTML visualizations with scrolling
- CSV files with correlation results
- Summary reports of predictive factors

## Model Strengths

- Good at capturing injury impacts and weather effects
- Uses only pre-game actionable data
- Provides confidence levels and key factors
- Avoids circular logic with spreads

## Model Limitations

- Simplified network structure
- Limited to categorical features
- Requires manual injury data input
- Trained on limited dataset (544 games)

## Future Improvements

- More sophisticated network structures
- Additional data sources (DVOA, PFF grades)
- Real-time injury data integration
- Ensemble methods with other models
