#!/usr/bin/env python3
"""
Sequence Builder Module
========================

Builds per-team sequences of the last K games with padding and masking.
Ensures strict time-ordering and no look-ahead bias.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TeamSequence:
    """Container for a team's game sequence."""
    features: np.ndarray  # Shape: (seq_len, n_features)
    mask: np.ndarray      # Shape: (seq_len,) - 1 for valid, 0 for padding
    seq_length: int       # Actual sequence length (before padding)
    team: str
    

class SequenceBuilder:
    """
    Builds sequences of recent games for each team.
    Handles variable-length histories with padding and masking.
    """
    
    def __init__(
        self,
        max_history_length: int = 10,
        min_history_length: int = 1,
        pad_value: float = 0.0,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize sequence builder.
        
        Args:
            max_history_length: Maximum number of games in sequence (K)
            min_history_length: Minimum games required
            pad_value: Value to use for padding
            feature_columns: List of feature column names to use
        """
        self.max_history_length = max_history_length
        self.min_history_length = min_history_length
        self.pad_value = pad_value
        self.feature_columns = feature_columns
        
    def build_team_sequence(
        self,
        team: str,
        team_features: pd.DataFrame,
        up_to_date: Tuple[int, int] = None
    ) -> Optional[TeamSequence]:
        """
        Build sequence of recent games for a team.
        
        Args:
            team: Team abbreviation
            team_features: DataFrame with features for this team's games
            up_to_date: (season, week) tuple - only use games before this date
            
        Returns:
            TeamSequence or None if insufficient data
        """
        # Filter to this team
        team_data = team_features[team_features['team'] == team].copy()
        
        # Filter by date if specified (no look-ahead)
        if up_to_date is not None:
            season, week = up_to_date
            team_data = team_data[
                (team_data['season'] < season) |
                ((team_data['season'] == season) & (team_data['week'] < week))
            ]
        
        # Sort by time
        team_data = team_data.sort_values(['season', 'week'])
        
        # Check if enough data
        if len(team_data) < self.min_history_length:
            return None
        
        # Take last K games
        recent_games = team_data.tail(self.max_history_length)
        
        # Extract features
        if self.feature_columns is not None:
            features = recent_games[self.feature_columns].values
        else:
            # Use all numeric columns except metadata
            exclude_cols = ['game_id', 'season', 'week', 'team', 'opponent']
            feature_cols = [c for c in recent_games.columns if c not in exclude_cols]
            features = recent_games[feature_cols].values
        
        # Get actual sequence length
        seq_len = len(features)
        
        # Pad if necessary
        if seq_len < self.max_history_length:
            n_pad = self.max_history_length - seq_len
            padding = np.full((n_pad, features.shape[1]), self.pad_value)
            features = np.vstack([padding, features])  # Pad at the beginning
        
        # Create mask (1 for valid, 0 for padding)
        mask = np.zeros(self.max_history_length)
        mask[-seq_len:] = 1  # Mark actual data as valid
        
        return TeamSequence(
            features=features.astype(np.float32),
            mask=mask.astype(np.float32),
            seq_length=seq_len,
            team=team
        )
    
    def build_matchup_sequences(
        self,
        home_team: str,
        away_team: str,
        team_features: pd.DataFrame,
        up_to_date: Tuple[int, int]
    ) -> Tuple[Optional[TeamSequence], Optional[TeamSequence]]:
        """
        Build sequences for both teams in a matchup.
        
        Args:
            home_team: Home team
            away_team: Away team
            team_features: DataFrame with features for all teams
            up_to_date: (season, week) - only use games before this
            
        Returns:
            Tuple of (home_sequence, away_sequence)
        """
        home_seq = self.build_team_sequence(home_team, team_features, up_to_date)
        away_seq = self.build_team_sequence(away_team, team_features, up_to_date)
        
        return home_seq, away_seq
    
    def build_dataset_for_week(
        self,
        games: pd.DataFrame,
        team_features: pd.DataFrame,
        season: int,
        week: int,
        include_targets: bool = True
    ) -> List[Dict]:
        """
        Build dataset for all games in a specific week.
        
        Args:
            games: DataFrame of games for this week
            team_features: DataFrame with historical team features
            season: Season
            week: Week
            include_targets: Whether to include target variables
            
        Returns:
            List of dictionaries with sequences and targets
        """
        dataset = []
        
        for _, game in games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Build sequences (only using data before this week)
            home_seq, away_seq = self.build_matchup_sequences(
                home_team, away_team, team_features, up_to_date=(season, week)
            )
            
            # Skip if either team has insufficient data
            if home_seq is None or away_seq is None:
                continue
            
            # Prepare data point
            data_point = {
                'game_id': game['game_id'],
                'season': season,
                'week': week,
                'home_team': home_team,
                'away_team': away_team,
                'home_sequence': home_seq.features,
                'home_mask': home_seq.mask,
                'away_sequence': away_seq.features,
                'away_mask': away_seq.mask,
                'is_divisional': 0.0,  # TODO: Add divisional lookup
            }
            
            # Add targets if available
            if include_targets:
                if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
                    home_score = float(game['home_score'])
                    away_score = float(game['away_score'])
                    
                    data_point['home_score'] = home_score
                    data_point['away_score'] = away_score
                    data_point['home_win'] = 1.0 if home_score > away_score else 0.0
                else:
                    # Skip games without targets during training
                    if include_targets:
                        continue
            
            dataset.append(data_point)
        
        return dataset
    
    def collate_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data points into tensors.
        
        Args:
            batch: List of data points
            
        Returns:
            Dictionary of tensors
        """
        # Stack sequences
        home_sequences = torch.FloatTensor(np.stack([b['home_sequence'] for b in batch]))
        away_sequences = torch.FloatTensor(np.stack([b['away_sequence'] for b in batch]))
        
        home_masks = torch.FloatTensor(np.stack([b['home_mask'] for b in batch]))
        away_masks = torch.FloatTensor(np.stack([b['away_mask'] for b in batch]))
        
        # Matchup features
        is_divisional = torch.FloatTensor([b['is_divisional'] for b in batch])
        
        batch_dict = {
            'home_sequences': home_sequences,
            'away_sequences': away_sequences,
            'home_masks': home_masks,
            'away_masks': away_masks,
            'is_divisional': is_divisional,
        }
        
        # Add targets if available
        if 'home_win' in batch[0]:
            batch_dict['home_win'] = torch.FloatTensor([b['home_win'] for b in batch])
            batch_dict['home_score'] = torch.FloatTensor([b['home_score'] for b in batch])
            batch_dict['away_score'] = torch.FloatTensor([b['away_score'] for b in batch])
        
        # Add metadata
        batch_dict['game_ids'] = [b['game_id'] for b in batch]
        batch_dict['home_teams'] = [b['home_team'] for b in batch]
        batch_dict['away_teams'] = [b['away_team'] for b in batch]
        
        return batch_dict


class RecencyWeightCalculator:
    """
    Calculates recency weights for sequence steps using exponential decay.
    """
    
    @staticmethod
    def compute_weights(
        seq_lengths: np.ndarray,
        max_length: int,
        half_life: float = 8.0
    ) -> np.ndarray:
        """
        Compute exponential decay weights for sequences.
        
        Args:
            seq_lengths: Actual lengths of sequences (batch_size,)
            max_length: Maximum sequence length
            half_life: Half-life for exponential decay (in games)
            
        Returns:
            Weights array (batch_size, max_length)
        """
        batch_size = len(seq_lengths)
        weights = np.zeros((batch_size, max_length))
        
        for i, seq_len in enumerate(seq_lengths):
            if seq_len == 0:
                continue
            
            # Compute time indices (0 = oldest, seq_len-1 = most recent)
            time_indices = np.arange(seq_len)
            
            # Exponential decay: w(t) = 2^(-t/half_life)
            # More recent games get higher weights
            time_from_present = seq_len - 1 - time_indices
            decay_weights = 2.0 ** (-time_from_present / half_life)
            
            # Normalize
            decay_weights = decay_weights / decay_weights.sum()
            
            # Place in the correct position (accounting for padding)
            start_idx = max_length - seq_len
            weights[i, start_idx:] = decay_weights
        
        return weights.astype(np.float32)


def main():
    """Test sequence building."""
    from data_loader import NFLDataLoader
    from feature_engineering import FeatureEngineer
    
    print("\n" + "="*70)
    print("SEQUENCE BUILDER - TEST")
    print("="*70)
    
    # Load data
    loader = NFLDataLoader(seasons=[2023, 2024], cache_dir="data/", use_cache=True)
    pbp, games = loader.load_data()
    
    # Compute features
    engineer = FeatureEngineer()
    
    train_games = games[
        (games['season'] == 2023) &
        (games['game_type'] == 'REG') &
        (games['home_score'].notna())
    ]
    
    print(f"\nComputing features for {len(train_games)} games...")
    team_features = engineer.compute_features_for_games(train_games, pbp)
    team_features = engineer.update_elo_ratings(team_features)
    
    # Initialize sequence builder
    feature_cols = engineer.get_feature_names()
    builder = SequenceBuilder(
        max_history_length=10,
        min_history_length=1,
        feature_columns=feature_cols
    )
    
    print(f"\nBuilding sequences with max_length={builder.max_history_length}")
    print(f"Using features: {feature_cols}")
    
    # Test building sequences for a team
    test_team = 'KC'
    seq = builder.build_team_sequence(
        test_team, 
        team_features, 
        up_to_date=(2023, 10)
    )
    
    if seq:
        print(f"\n{test_team} sequence:")
        print(f"  Shape: {seq.features.shape}")
        print(f"  Actual length: {seq.seq_length}")
        print(f"  Mask: {seq.mask}")
        print(f"  First game features: {seq.features[-seq.seq_length][:5]}...")
    
    # Build dataset for a week
    test_games = games[
        (games['season'] == 2024) &
        (games['week'] == 6) &
        (games['game_type'] == 'REG')
    ]
    
    print(f"\n\nBuilding dataset for 2024 Week 6 ({len(test_games)} games)...")
    
    # Need all features up to week 6
    all_features = engineer.compute_features_for_games(
        games[(games['season'].isin([2023, 2024])) & (games['home_score'].notna())],
        pbp
    )
    all_features = engineer.update_elo_ratings(all_features)
    
    dataset = builder.build_dataset_for_week(
        test_games, all_features, season=2024, week=6, include_targets=True
    )
    
    print(f"✓ Built dataset with {len(dataset)} matchups")
    
    # Test collation
    if len(dataset) >= 2:
        batch = builder.collate_batch(dataset[:2])
        print(f"\nBatch shapes:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")
    
    # Test recency weights
    print("\n\nTesting recency weights:")
    seq_lengths = np.array([5, 8, 10])
    weights = RecencyWeightCalculator.compute_weights(
        seq_lengths, max_length=10, half_life=8.0
    )
    
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights for seq_len=10:\n{weights[2]}")
    print(f"Sum: {weights[2].sum():.4f}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

