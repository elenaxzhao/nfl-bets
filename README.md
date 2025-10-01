# NFLBets

A research repository testing different machine learning models to find the most effective approach for NFL sports betting prediction.

## Overview

**Current Phase: Model Research** - This project is currently focused on researching and comparing different modeling approaches to NFL game prediction. We're testing various algorithms and data sources to identify which methods provide the best betting edge against sportsbooks.

**Future Goal: Prediction System** - Once the research phase identifies the most effective models, we plan to build a comprehensive prediction system.

Our research focuses on **pre-game actionable factors** like injuries, weather, and rest days while avoiding circular logic with spreads or post-game data.

## Models Implemented

### Bayesian Network Model
- **Approach**: Probabilistic graphical model using injury, weather, and temporal factors
- **Performance**: 75% accuracy on test data
- **Documentation**: See [models/bayes_net/README.md](models/bayes_net/README.md) for detailed usage

### Poisson ELO Model
- **Approach**: Elo rating system with Poisson distribution for scoring
- **Performance**: Currently in development
- **Key Features**: Team strength ratings, scoring probability estimation

### Logistic Regression Model
- **Approach**: Statistical model using team performance differentials from first 5 games
- **Features**: Win percentage, points for/against, passing/rushing efficiency, turnover rates
- **Performance**: Uses team stats up to Week 5 to predict Week 6+ outcomes
- **Documentation**: See [models/logistic/](models/logistic/) for implementation details

## Research Focus Areas

- **Correlation Analysis** of all NFL data variables to identify predictive factors
- **Injury Impact Modeling** - Key position injuries (QB/RB/WR/TE)
- **Weather Factor Integration** - Wind speed and temperature effects
- **Temporal Analysis** - Season phase and rest day advantages
- **Model Comparison** - Evaluating different approaches for betting edge

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd nfl-bets
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Models

#### Bayesian Network Model
```bash
# See detailed documentation
cat models/bayes_net/README.md

# Train and test the Bayesian network
python models/bayes_net/nfl_bayesian_network.py
```

#### Poisson ELO Model
```bash
# Run ELO model analysis
python models/poisson_elo/poisson_elo_model.py
```

#### Correlation Analysis
```bash
# Run comprehensive correlation analysis
python models/bayes_net/correlation_tests.py
```

## Key Research Findings

### Most Predictive Factors (from correlation analysis)
- **Away team key injuries**: 0.087 correlation (p=0.042) - Most actionable factor
- **Weather conditions**: Wind speed affects passing game strategy
- **Rest advantages**: Teams with more rest days perform better
- **Season progression**: Early vs late season performance differences

*Detailed correlation analysis results available in [models/bayes_net/README.md](models/bayes_net/README.md)*

## Current Model Performance

### Bayesian Network
- **Accuracy**: 75% on test data (81/108 correct predictions)
- **Features**: Injury data, weather, temporal factors
- **Strengths**: Good at capturing injury impacts and weather effects
- **Limitations**: Simplified network structure, limited to categorical features

### Poisson ELO
- **Status**: In development
- **Approach**: Team rating system with scoring probability
- **Expected Strengths**: Dynamic team strength adjustment
- **Research Focus**: Optimal rating update parameters

### Logistic Regression
- **Status**: Complete implementation with team performance features
- **Approach**: Uses team stats differentials (win %, points, efficiency metrics)
- **Features**: 12 key performance indicators from first 5 games
- **Research Value**: Demonstrates effectiveness of team performance metrics

## Model Architecture Overview

Each model uses different approaches to capture game outcome patterns:

- **Bayesian Network**: Hierarchical probabilistic structure with injury/weather factors
- **Poisson ELO**: Team rating system with scoring probability estimation  
- **Logistic Regression**: Linear combination of predictive features

*Detailed architecture information available in individual model documentation*

## Usage Examples

### Running Individual Models
```bash
# Bayesian Network
python models/bayes_net/nfl_bayesian_network.py

# Logistic Regression
python models/logistic/logistic_regression.py

# Poisson ELO 
python models/poisson_elo/poisson_elo_model.py

# Correlation analysis
python models/bayes_net/correlation_tests.py
```

### Getting Started
Each model has its own documentation with detailed usage instructions:
- **Bayesian Network**: [models/bayes_net/README.md](models/bayes_net/README.md)
- **Poisson ELO**: [models/poisson_elo/README.md](models/poisson_elo/README.md) (when available)

## Data Sources
- Trained on 2023-2024 data (544 games)

- **NFL Schedules**: Game results and basic info
- **Injury Reports**: Player injury status by week
- **Weather Data**: Wind speed and temperature
- **Team Rosters**: Player positions and availability

All data sourced from [nflreadpy](https://github.com/nflverse/nflreadpy) - the Python port of nflreadr.

## Project Structure

```
nfl-bets/
├── models/
│   ├── bayes_net/
│   │   ├── nfl_bayesian_network.py      # Main Bayesian network
│   │   ├── correlation_tests.py         # Correlation analysis
│   │   ├── cli_predictor.py            # Command line interface
│   │   └── simple_predictor.py         # Interactive interface
│   ├── logistic/
│   │   ├── logistic_regression.py      # Main logistic regression model
│   │   └── example_usage.py            # Usage examples
│   └── poisson_elo/
│       ├── poisson_elo_model.py        # Main ELO rating system
│       └── s24_test.py                 # Test implementation
├── requirements.txt                    # Dependencies
├── README.md                          # This file
└── .gitignore                         # Git ignore rules
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [nflreadpy](https://github.com/nflverse/nflreadpy) for NFL data access
- [nflverse](https://www.nflverse.com/) for comprehensive NFL data
- Academic sports analytics research for methodological inspiration
- Open source machine learning community for tools and frameworks

---

**Currently in research phase - focusing on model development and comparison**