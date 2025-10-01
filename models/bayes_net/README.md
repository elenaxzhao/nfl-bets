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

- **Accuracy**: 75% on test data (81/108 correct predictions)
- **Training Data**: 544 games from 2023-2024 seasons
- **Key Predictor**: Away team key injuries (0.087 correlation, p=0.042)

## Usage

### Training the Model
```bash
python nfl_bayesian_network.py
```

### Making Predictions
```bash
# Command line interface
python cli_predictor.py KC BUF 8 --home-injuries 1 --away-injuries 4 --wind 10

# Interactive mode
python simple_predictor.py
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
- `simple_predictor.py` - Interactive prediction interface

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
