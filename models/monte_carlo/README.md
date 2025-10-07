# NFL Monte Carlo Simulation Model

A comprehensive Monte Carlo simulation model for predicting NFL game outcomes, including win probabilities, expected point spreads, and total scores.

## Overview

This model uses historical team performance data to build probability distributions for team scoring. It then runs thousands of simulations to estimate game outcomes, providing robust predictions that account for variance and uncertainty.

## Features

- **Win Probability Predictions**: Estimates the probability of each team winning
- **Expected Scores**: Predicts expected points for both teams with confidence intervals
- **Spread Predictions**: Estimates point spreads with uncertainty bounds
- **Total Points**: Predicts over/under totals
- **Betting Insights**: Provides probabilities for common betting scenarios
- **Model Evaluation**: Built-in functions to evaluate prediction accuracy

## Installation

Ensure you have the required dependencies:

```bash
pip install -r ../../requirements.txt
```

## Quick Start

```python
from monte_carlo_model import NFLMonteCarloModel

# Initialize model with 3 seasons of training data (recommended)
model = NFLMonteCarloModel(
    train_seasons=[2022, 2023, 2024],
    test_season=2025,
    n_simulations=10000
)

# Load data and build distributions
model.load_data()
model.build_team_distributions()

# Predict a single game
prediction = model.predict_game('KC', 'BUF')
model.print_prediction(prediction)

# Predict all games in a week
week_predictions = model.predict_week(week=1, season=2025)
```

## Training Data Recommendations

### How Much Data?

**Recommended: 3-5 seasons**

- **Too Few (1-2 seasons)**: May lack statistical power, especially for teams with significant roster changes
- **Optimal (3-5 seasons)**: Best balance between data volume and relevance
- **Too Many (6+ seasons)**: May include outdated team dynamics, different coaching staffs, and rule changes

### Why This Range?

1. **Statistical Significance**: 3+ seasons provide ~48+ games per team, sufficient for stable distribution estimates
2. **Relevance**: Recent seasons reflect current team composition, coaching, and league meta
3. **Variance Capture**: Multiple seasons capture both team consistency and year-to-year variance
4. **Roster Stability**: Most NFL rosters turn over significantly beyond 5 years

### Recency Weighting

The model includes optional recency weighting (`recency_weight` parameter):
- Set to 0.0 for equal weighting of all games
- Set to 0.3 (default) for 30% more weight on recent games
- Set higher for more aggressive recency bias

## Model Parameters

### Core Parameters

- `train_seasons`: List of seasons for training (e.g., [2022, 2023, 2024])
- `test_season`: Season to make predictions for
- `n_simulations`: Number of Monte Carlo simulations (default: 10,000)
  - More simulations = more stable predictions but slower
  - 10,000 provides good balance
  - 5,000 suitable for batch processing
  - 20,000+ for high-stakes predictions

- `home_field_advantage`: Points added to home team (default: 2.5)
  - NFL average is ~2.5 points
  - Can adjust based on specific venue analysis

- `recency_weight`: Weight for recent games (default: 0.3)
  - 0.0 = all games weighted equally
  - 0.5 = strong recency bias

## Output Metrics

### Win Probabilities
- `home_win_probability`: P(home team wins)
- `away_win_probability`: P(away team wins)
- `tie_probability`: P(tie) - very low in NFL

### Score Predictions
- `expected_home_score`: Mean predicted home score
- `expected_away_score`: Mean predicted away score
- Confidence intervals (95%) for both scores

### Spread & Totals
- `expected_spread`: Home team spread (positive = home favored)
- `expected_total`: Total points (over/under)
- Confidence intervals for both

### Betting Insights
- `prob_home_covers_3`: P(home covers -3 spread)
- `prob_home_covers_7`: P(home covers -7 spread)
- `prob_over_45`: P(total > 45)
- `prob_over_50`: P(total > 50)

## Model Performance

Based on testing with 3 seasons of training data (2022-2024):

- **Win Prediction Accuracy**: ~60-70%
- **Average Spread Error**: ~10-14 points
- **Average Score Error**: ~8-12 points per team

Performance varies by:
- Time in season (better later as more data accumulates)
- Team stability (better for consistent teams)
- Matchup type (divisional games harder to predict)

## Examples

See `example_usage.py` for comprehensive examples including:

1. Single game predictions with visualizations
2. Full week predictions
3. Model evaluation on historical games
4. Multi-week performance analysis
5. Training data size comparison

Run examples:
```bash
python example_usage.py
```

## Methodology

### 1. Data Collection
- Loads play-by-play and schedule data using `nflreadpy`
- Calculates game-level statistics for each team
- Tracks offensive and defensive performance

### 2. Distribution Building
- Creates probability distributions for each team's scoring
- Separates home and away performance
- Accounts for opponent strength (offensive vs. defensive)
- Applies optional recency weighting

### 3. Monte Carlo Simulation
- For each game, simulates N outcomes (default: 10,000)
- Samples from team distributions accounting for:
  - Team's offensive strength
  - Opponent's defensive strength
  - Home field advantage
  - Historical variance

### 4. Aggregation
- Aggregates simulations to compute:
  - Win probabilities
  - Expected scores
  - Confidence intervals
  - Betting probabilities

## Advantages of Monte Carlo Approach

1. **Handles Uncertainty**: Explicitly models variance in team performance
2. **No Strict Assumptions**: Doesn't require specific distribution assumptions
3. **Flexible**: Easy to incorporate new factors (injuries, weather, etc.)
4. **Interpretable**: Provides intuitive confidence intervals
5. **Robust**: Less sensitive to outliers than point estimates

## Comparison to Other Models

| Model Type | Strengths | Weaknesses |
|------------|-----------|------------|
| **Logistic Regression** | Fast, interpretable coefficients | Binary outcomes only, rigid assumptions |
| **Poisson-ELO** | Dynamic ratings, accounts for strength | Assumes Poisson distribution, complex updating |
| **Monte Carlo** | Uncertainty quantification, flexible | Computationally intensive, requires tuning |

## Future Enhancements

Potential improvements:
- Weather data integration
- Injury/roster adjustments
- Rest days and travel distance
- Referee tendencies
- Player-level modeling
- In-season distribution updating
- Correlation modeling (division rivals, etc.)

## Citation

If using this model for research or publication:

```
NFL Monte Carlo Simulation Model
NFL Bets Analysis Project
2024-2025
```

## License

See LICENSE file in repository root.

## Support

For questions or issues, please open an issue in the GitHub repository.

