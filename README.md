# NBA Bets

Machine learning models for NBA game prediction and sports betting analysis.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Models

- **Bayesian Network** - Probabilistic model using injuries, weather, and temporal factors
- **Logistic Regression** - Statistical model using team performance metrics
- **Monte Carlo** - Simulation-based prediction model
- **Poisson ELO** - ELO rating system with scoring probability estimation
- **Recurrent Neural Network** - Deep learning model for sequence-based predictions

## Project Structure

```
_models/
├── bayes_net/
├── logistic/
├── monte_carlo/
├── poisson_elo/
└── recurrent_neural_net/
```

Each model directory contains implementation code and documentation. See individual README files for detailed usage instructions.

## License

MIT License - see LICENSE file for details.

