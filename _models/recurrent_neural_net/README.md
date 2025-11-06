# NFL Game Prediction using Recurrent Neural Networks

A PyTorch-based sequence model for predicting NFL game outcomes using Siamese encoders and dual prediction heads.

## Overview

This model predicts:
1. **Home win probability** (binary classification)
2. **Expected points** for home and away teams (regression)

### Key Features

- **Siamese GRU/LSTM encoders** that process each team's recent game history
- **Time-aware rolling origin evaluation** with strict no look-ahead guarantees
- **Team embeddings** for capturing team-specific characteristics
- **Dual prediction heads** for win probability and score prediction
- **Recency weighting** with exponential decay for recent games
- **Mixed precision training** (AMP) for faster training on CUDA
- **Early stopping** with validation-based monitoring
- **Calibration support** (Platt scaling, isotonic regression)

## Architecture

```
Input: Home Team Sequence [K games × features] + Away Team Sequence [K games × features]
    ↓
Team Encoder (GRU/LSTM) → Home Vector [hidden_dim]
    ↓
Team Encoder (shared weights) → Away Vector [hidden_dim]
    ↓
Concatenate: [home_vec, away_vec, home_embed, away_embed, matchup_features]
    ↓
    ├─→ Win Head (MLP) → P(home win)
    └─→ Score Head (MLP) → (μ_home, μ_away)
```

## Installation

```bash
# Install dependencies
pip install -r ../../requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

## Configuration

All settings are in `config.yaml`:

### Data Settings
- `seasons`: List of seasons to load
- `cache_dir`: Directory for caching parquet files
- `use_cache`: Whether to use cached data

### Model Architecture
- `encoder_type`: `'gru'` or `'lstm'`
- `hidden_dim`: RNN hidden dimension (default: 128)
- `num_layers`: Number of RNN layers (default: 2)
- `team_embedding_dim`: Team embedding size (default: 32)
- `dropout`: Dropout rate (default: 0.3)

### Training
- `learning_rate`: Adam learning rate (default: 0.001)
- `batch_size`: Training batch size (default: 32)
- `max_epochs`: Maximum training epochs (default: 100)
- `early_stopping_patience`: Epochs to wait for improvement (default: 10)
- `use_amp`: Mixed precision training (default: true)
- `gradient_clip_value`: Gradient clipping threshold (default: 1.0)

### Loss Function
- `lambda_bce`: Win prediction loss weight (default: 1.0)
- `lambda_score`: Score prediction loss weight (default: 0.5)
- `score_loss_type`: `'mae'`, `'mse'`, `'poisson'`, or `'negative_binomial'`

### Sequence Configuration
- `max_history_length`: Maximum games per team (K, default: 10)
- `min_history_length`: Minimum games required (default: 1)

## Usage

### Quick Start

```bash
# Run 2024 season evaluation (weeks 6-18)
python s24_test.py
```

This will:
1. Load data from nflreadpy (cached to `data/`)
2. Compute features (EPA, success rates, Elo, etc.)
3. For each test week:
   - Train on prior seasons + weeks 1 to (w-1)
   - Predict week w
   - Evaluate predictions
4. Save results to `results/`

### Individual Components

```python
# Data loading
from data_loader import NFLDataLoader

loader = NFLDataLoader(seasons=[2023, 2024], cache_dir="data/")
pbp, games = loader.load_data()

# Feature engineering
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(use_epa=True, use_elo=True)
features = engineer.compute_features_for_games(games, pbp)

# Sequence building
from sequence_builder import SequenceBuilder

builder = SequenceBuilder(max_history_length=10)
home_seq, away_seq = builder.build_matchup_sequences(
    "KC", "BUF", features, up_to_date=(2024, 10)
)

# Model creation
from model import NFLGamePredictor

model = NFLGamePredictor(
    feature_dim=14,
    hidden_dim=128,
    encoder_type='gru'
)

# Training
from trainer import Trainer, TrainingConfig

config = TrainingConfig(learning_rate=0.001, max_epochs=50)
trainer = Trainer(model, config)
trainer.train(train_loader, val_loader)

# Evaluation
from metrics import MetricsCalculator

metrics = MetricsCalculator.compute_all_metrics(
    y_true_win, y_pred_win_prob,
    y_true_home_score, y_true_away_score,
    y_pred_home_score, y_pred_away_score
)
```

## Features

### Per-Game Features (from play-by-play)

- **EPA (Expected Points Added)**
  - Offensive EPA per play
  - Defensive EPA per play
  
- **Success Rates**
  - Offensive success rate (EPA > 0)
  - Defensive success rate
  
- **Turnover Rates**
  - Turnovers per play (offense)
  - Opponent turnovers forced (defense)
  
- **Pace**
  - Plays per game
  - Drives per game
  
- **Elo Rating**
  - Updated sequentially using actual game results
  - K-factor: 20.0, initial: 1500.0

- **Scores**
  - Points scored
  - Points allowed

### Matchup Features

- Home field advantage flag
- Divisional game indicator
- Week (normalized 0-1)
- Home-minus-away feature differences

## Data Pipeline

### Time-Aware Splitting

The model uses strict time-aware splits:

```
Train: All completed games from prior seasons + weeks 1 to (w-1) of test season
Test: Week w of test season
```

No data from week w or later is used during training for week w predictions.

### Sequence Construction

For each team, we build a sequence of their last K games (configurable):

1. Sort games by (season, week)
2. Take last K games before the target game
3. Pad if < K games (zero padding at beginning)
4. Create attention mask (1 for valid, 0 for padding)

### Recency Weighting

Exponential decay weights for recent games:

```
w(t) = 2^(-t/half_life)
```

where `t` is games ago. Can be applied as:
- **Sample weights** (in loss function)
- **Attention weights** (in RNN encoder)

## Evaluation Metrics

### Win Prediction
- **Accuracy**: Correct predictions / total games
- **Log Loss**: Cross-entropy loss
- **Brier Score**: Mean squared error of probabilities
- **Calibration Curve**: Reliability diagram

### Score Prediction
- **MAE**: Mean absolute error for home/away scores
- **RMSE**: Root mean squared error
- **Spread MAE**: MAE for (home_score - away_score)

## Results

Results are saved to `results/`:

- `rnn_2024_predictions.csv`: Game-by-game predictions
- `rnn_2024_weekly_metrics.csv`: Metrics per week
- `rnn_2024_summary.csv`: Overall summary statistics

## File Structure

```
recurrent_neural_net/
├── config.yaml              # Configuration
├── data_loader.py           # Data loading and caching
├── feature_engineering.py   # Feature computation
├── sequence_builder.py      # Sequence construction with padding/masking
├── model.py                 # PyTorch model architecture
├── trainer.py               # Training loop with AMP, early stopping
├── metrics.py               # Evaluation metrics and calibration
├── s24_test.py             # Main evaluation script
├── README.md               # This file
├── data/                   # Cached parquet files (auto-created)
├── results/                # Evaluation results (auto-created)
└── checkpoints/            # Model checkpoints (optional)
```

## Advanced Usage

### Custom Training

```python
# Modify config.yaml or pass custom config
config = {
    'model': {
        'encoder_type': 'lstm',  # Use LSTM instead of GRU
        'hidden_dim': 256,       # Larger hidden dimension
        'num_layers': 3,         # Deeper network
        'bidirectional': True    # Bidirectional RNN
    },
    'training': {
        'learning_rate': 0.0005,
        'lambda_bce': 2.0,       # More weight on win prediction
        'lambda_score': 1.0
    }
}
```

### Calibration

```python
from metrics import Calibrator

# Train calibrator on validation set
calibrator = Calibrator(method='isotonic')
calibrator.fit(y_val_true, y_val_pred_prob)

# Apply to test predictions
y_test_calibrated = calibrator.transform(y_test_pred_prob)
```

### Custom Features

```python
# Add custom features in feature_engineering.py
class CustomFeatureEngineer(FeatureEngineer):
    def compute_game_features(self, game_row, pbp_data, for_team, is_home):
        features = super().compute_game_features(game_row, pbp_data, for_team, is_home)
        
        # Add custom feature
        features['my_custom_metric'] = ...
        
        return features
```

## Reproducibility

The model ensures reproducibility through:
- **Deterministic seeds** (PyTorch, NumPy, random)
- **Deterministic CUDA operations** (when enabled)
- **Time-ordered data splits** (consistent validation sets)
- **Cached data** (identical data across runs)

Set in `config.yaml`:
```yaml
training:
  seed: 42
  deterministic: true
```

## Performance Tips

### GPU Acceleration
- Set `device: "cuda"` in config (auto-detected)
- Enable AMP: `use_amp: true`
- Increase batch size for better GPU utilization

### Training Speed
- Use cached data: `use_cache: true`
- Reduce `max_epochs` or increase `early_stopping_patience`
- Use smaller `max_history_length` (fewer sequence steps)

### Memory
- Reduce `batch_size`
- Reduce `hidden_dim` or `num_layers`
- Use `bidirectional: false`

## Comparison to Other Models

This RNN model complements the existing models:

| Model | Type | Strengths |
|-------|------|-----------|
| **Monte Carlo** | Simulation | Fast, interpretable, good for live betting |
| **Logistic Regression** | Linear | Simple baseline, fast inference |
| **RNN (this)** | Deep Learning | Captures temporal patterns, team embeddings |

## Citation

If you use this code, please reference:
- **nflreadpy**: NFL play-by-play data
- **PyTorch**: Deep learning framework
- Architecture inspired by Siamese networks for sequence comparison

## License

See root LICENSE file.

## Contributing

Improvements welcome:
- Additional features (weather, injuries, etc.)
- Alternative architectures (Transformers, TCN)
- Hyperparameter tuning
- Better calibration methods

## Contact

For questions or issues, please open an issue on GitHub.

