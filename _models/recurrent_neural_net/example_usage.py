#!/usr/bin/env python3
"""
Example Usage
=============

Demonstrates how to use individual components of the RNN model.
"""

import yaml
import torch
import pandas as pd


def example_1_data_loading():
    """Example 1: Load and cache NFL data."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Data Loading")
    print("="*70)
    
    from data_loader import NFLDataLoader, TimeAwareSplitter
    
    # Load data for multiple seasons
    loader = NFLDataLoader(
        seasons=[2023, 2024],
        cache_dir="data/",
        use_cache=True
    )
    
    pbp, games = loader.load_data()
    
    print(f"\nLoaded:")
    print(f"  Play-by-play records: {len(pbp):,}")
    print(f"  Games: {len(games):,}")
    print(f"  Seasons: {sorted(games['season'].unique())}")
    print(f"  Teams: {len(loader.get_all_teams())}")
    
    # Time-aware split
    train_games, test_games = TimeAwareSplitter.get_train_test_split(
        games, test_season=2024, test_week=10
    )
    
    print(f"\nTime-aware split for 2024 Week 10:")
    print(f"  Training games: {len(train_games)}")
    print(f"  Test games: {len(test_games)}")
    
    return pbp, games


def example_2_feature_engineering(pbp, games):
    """Example 2: Compute features from play-by-play data."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Feature Engineering")
    print("="*70)
    
    from feature_engineering import FeatureEngineer
    
    # Initialize engineer
    engineer = FeatureEngineer(
        use_epa=True,
        use_success_rate=True,
        use_turnover_rate=True,
        use_pace=True,
        use_drives=True,
        use_elo=True
    )
    
    # Compute features for a subset of games
    sample_games = games[
        (games['season'] == 2024) &
        (games['week'] <= 5) &
        (games['game_type'] == 'REG') &
        (games['home_score'].notna())
    ].head(10)
    
    print(f"\nComputing features for {len(sample_games)} games...")
    
    team_features = engineer.compute_features_for_games(sample_games, pbp)
    team_features = engineer.update_elo_ratings(team_features)
    
    feature_columns = engineer.get_feature_names()
    
    print(f"✓ Computed {len(feature_columns)} features per game")
    print(f"\nFeatures: {feature_columns}")
    
    # Show sample features for one team
    sample_team = team_features['team'].iloc[0]
    sample_row = team_features[team_features['team'] == sample_team].iloc[0]
    
    print(f"\nSample features for {sample_team}:")
    for col in feature_columns[:5]:
        print(f"  {col}: {sample_row[col]:.3f}")
    
    return team_features, feature_columns


def example_3_sequence_building(team_features, feature_columns):
    """Example 3: Build sequences with padding and masking."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Sequence Building")
    print("="*70)
    
    from sequence_builder import SequenceBuilder, RecencyWeightCalculator
    import numpy as np
    
    # Initialize builder
    builder = SequenceBuilder(
        max_history_length=10,
        min_history_length=1,
        pad_value=0.0,
        feature_columns=feature_columns
    )
    
    # Build sequence for a team
    team = team_features['team'].iloc[0]
    
    sequence = builder.build_team_sequence(
        team=team,
        team_features=team_features,
        up_to_date=(2024, 10)
    )
    
    if sequence:
        print(f"\nSequence for {team}:")
        print(f"  Features shape: {sequence.features.shape}")
        print(f"  Actual length: {sequence.seq_length} games")
        print(f"  Padding: {10 - sequence.seq_length} steps")
        print(f"  Mask: {sequence.mask}")
        
        # Compute recency weights
        seq_lengths = np.array([sequence.seq_length])
        weights = RecencyWeightCalculator.compute_weights(
            seq_lengths, max_length=10, half_life=8.0
        )
        
        print(f"\nRecency weights (half-life=8):")
        print(f"  {weights[0]}")
        print(f"  Sum: {weights[0].sum():.4f}")
    
    # Build matchup sequences
    home_team = team_features['team'].iloc[0]
    away_team = team_features['team'].iloc[1]
    
    home_seq, away_seq = builder.build_matchup_sequences(
        home_team, away_team, team_features,
        up_to_date=(2024, 10)
    )
    
    print(f"\nMatchup: {away_team} @ {home_team}")
    if home_seq and away_seq:
        print(f"  {home_team} history: {home_seq.seq_length} games")
        print(f"  {away_team} history: {away_seq.seq_length} games")


def example_4_model_creation():
    """Example 4: Create and inspect model architecture."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Model Architecture")
    print("="*70)
    
    from model import NFLGamePredictor
    
    # Create model
    model = NFLGamePredictor(
        feature_dim=14,
        team_embedding_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        encoder_type='gru',
        bidirectional=False,
        win_head_hidden=[64, 32],
        score_head_hidden=[64, 32],
        num_teams=32
    )
    
    print(f"\nModel architecture:")
    print(f"  Encoder type: GRU")
    print(f"  Hidden dimension: 128")
    print(f"  Number of layers: 2")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    feature_dim = 14
    
    home_sequences = torch.randn(batch_size, seq_len, feature_dim)
    away_sequences = torch.randn(batch_size, seq_len, feature_dim)
    home_masks = torch.ones(batch_size, seq_len)
    away_masks = torch.ones(batch_size, seq_len)
    
    # Simulate some padding
    home_masks[0, :3] = 0
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            home_sequences=home_sequences,
            away_sequences=away_sequences,
            home_masks=home_masks,
            away_masks=away_masks
        )
    
    print(f"\nForward pass (batch_size={batch_size}):")
    print(f"  Win probabilities: {outputs['win_probs'].numpy()}")
    print(f"  Predicted home scores: {outputs['home_scores'].numpy()}")
    print(f"  Predicted away scores: {outputs['away_scores'].numpy()}")
    
    return model


def example_5_training_config():
    """Example 5: Configure training."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Training Configuration")
    print("="*70)
    
    from trainer import TrainingConfig
    
    # Create configuration
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        max_epochs=100,
        weight_decay=0.0001,
        gradient_clip_value=1.0,
        use_amp=True,
        early_stopping_patience=10,
        early_stopping_metric="val_log_loss",
        early_stopping_mode="min",
        recency_half_life=8.0,
        apply_recency_as="sample_weights",
        lambda_bce=1.0,
        lambda_score=0.5,
        score_loss_type="mae"
    )
    
    print(f"\nTraining configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print(f"  Use AMP: {config.use_amp}")
    print(f"  Gradient clipping: {config.gradient_clip_value}")
    print(f"\nLoss weights:")
    print(f"  Win prediction (BCE): {config.lambda_bce}")
    print(f"  Score prediction (MAE): {config.lambda_score}")
    print(f"\nRecency weighting:")
    print(f"  Half-life: {config.recency_half_life} games")
    print(f"  Apply as: {config.apply_recency_as}")


def example_6_metrics():
    """Example 6: Compute evaluation metrics."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Evaluation Metrics")
    print("="*70)
    
    from metrics import MetricsCalculator
    import numpy as np
    
    # Simulate predictions
    np.random.seed(42)
    n_games = 50
    
    y_true_win = np.random.randint(0, 2, n_games)
    y_pred_win_prob = np.random.beta(2, 2, n_games)
    
    # Correlate predictions with truth
    y_pred_win_prob = 0.7 * y_true_win + 0.3 * y_pred_win_prob
    y_pred_win_prob = np.clip(y_pred_win_prob, 0, 1)
    
    y_true_home_score = np.random.poisson(24, n_games).astype(float)
    y_true_away_score = np.random.poisson(21, n_games).astype(float)
    
    y_pred_home_score = y_true_home_score + np.random.normal(0, 4, n_games)
    y_pred_away_score = y_true_away_score + np.random.normal(0, 4, n_games)
    
    y_pred_home_score = np.maximum(0, y_pred_home_score)
    y_pred_away_score = np.maximum(0, y_pred_away_score)
    
    # Compute metrics
    metrics = MetricsCalculator.compute_all_metrics(
        y_true_win=y_true_win,
        y_pred_win_prob=y_pred_win_prob,
        y_true_home_score=y_true_home_score,
        y_true_away_score=y_true_away_score,
        y_pred_home_score=y_pred_home_score,
        y_pred_away_score=y_pred_away_score
    )
    
    print(f"\nMetrics on {n_games} games:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Score MAE: {metrics['mae_total']:.2f} points")
    print(f"  Score RMSE: {metrics['rmse_total']:.2f} points")
    print(f"  Spread MAE: {metrics['spread_mae']:.2f} points")


def main():
    """Run all examples."""
    print("\n" + "🏈"*35)
    print("RNN MODEL - EXAMPLE USAGE")
    print("🏈"*35)
    
    # Example 1: Data loading
    pbp, games = example_1_data_loading()
    
    # Example 2: Feature engineering
    team_features, feature_columns = example_2_feature_engineering(pbp, games)
    
    # Example 3: Sequence building
    example_3_sequence_building(team_features, feature_columns)
    
    # Example 4: Model creation
    model = example_4_model_creation()
    
    # Example 5: Training configuration
    example_5_training_config()
    
    # Example 6: Metrics
    example_6_metrics()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review config.yaml to customize settings")
    print("  2. Run s24_test.py for full 2024 season evaluation")
    print("  3. Check results/ directory for outputs")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

