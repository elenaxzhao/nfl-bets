#!/usr/bin/env python3
"""
Data Loading and Caching Module
================================

Handles loading NFL data from nflreadpy and caching to parquet/CSV.
Ensures strict time-aware data handling (no look-ahead).
"""

import os
import pandas as pd
import numpy as np
import nflreadpy as nfl
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class NFLDataLoader:
    """
    Loads and caches NFL play-by-play and schedule data.
    """
    
    def __init__(self, seasons: List[int], cache_dir: str = "data/", use_cache: bool = True):
        """
        Initialize data loader.
        
        Args:
            seasons: List of seasons to load
            cache_dir: Directory for caching data
            use_cache: Whether to use cached data if available
        """
        self.seasons = seasons
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        
        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.pbp_data = None
        self.games_data = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load play-by-play and schedule data.
        
        Returns:
            Tuple of (pbp_data, games_data)
        """
        pbp_cache_path = self.cache_dir / f"pbp_{'_'.join(map(str, self.seasons))}.parquet"
        games_cache_path = self.cache_dir / f"games_{'_'.join(map(str, self.seasons))}.parquet"
        
        # Try to load from cache
        if self.use_cache and pbp_cache_path.exists() and games_cache_path.exists():
            print(f"Loading data from cache...")
            self.pbp_data = pd.read_parquet(pbp_cache_path)
            self.games_data = pd.read_parquet(games_cache_path)
            print(f"✓ Loaded {len(self.pbp_data):,} plays and {len(self.games_data):,} games from cache")
        else:
            print(f"Loading data from nflreadpy for seasons: {self.seasons}")
            
            # Load from nflreadpy
            self.pbp_data = nfl.load_pbp(self.seasons).to_pandas()
            self.games_data = nfl.load_schedules(self.seasons).to_pandas()
            
            print(f"✓ Loaded {len(self.pbp_data):,} plays and {len(self.games_data):,} games")
            
            # Cache the data
            if self.use_cache:
                print(f"Caching data to {self.cache_dir}...")
                self.pbp_data.to_parquet(pbp_cache_path, index=False)
                self.games_data.to_parquet(games_cache_path, index=False)
                print("✓ Data cached")
        
        return self.pbp_data, self.games_data
    
    def get_games_up_to_week(
        self,
        season: int,
        week: int,
        include_current_week: bool = False
    ) -> pd.DataFrame:
        """
        Get games up to (but not including) a specific week.
        This ensures no look-ahead bias.
        
        Args:
            season: Season to filter
            week: Week number
            include_current_week: Whether to include the current week
            
        Returns:
            DataFrame of games
        """
        if self.games_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if include_current_week:
            games = self.games_data[
                (self.games_data['season'] == season) &
                (self.games_data['week'] <= week) &
                (self.games_data['game_type'] == 'REG')
            ]
        else:
            games = self.games_data[
                (self.games_data['season'] == season) &
                (self.games_data['week'] < week) &
                (self.games_data['game_type'] == 'REG')
            ]
        
        return games.copy()
    
    def get_games_for_week(self, season: int, week: int) -> pd.DataFrame:
        """
        Get games for a specific week.
        
        Args:
            season: Season
            week: Week number
            
        Returns:
            DataFrame of games
        """
        if self.games_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        games = self.games_data[
            (self.games_data['season'] == season) &
            (self.games_data['week'] == week) &
            (self.games_data['game_type'] == 'REG')
        ]
        
        return games.copy()
    
    def get_pbp_for_game(self, game_id: str) -> pd.DataFrame:
        """
        Get play-by-play data for a specific game.
        
        Args:
            game_id: Game ID
            
        Returns:
            DataFrame of plays
        """
        if self.pbp_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.pbp_data[self.pbp_data['game_id'] == game_id].copy()
    
    def get_all_teams(self) -> List[str]:
        """
        Get list of all unique teams.
        
        Returns:
            List of team abbreviations
        """
        if self.games_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        home_teams = self.games_data['home_team'].dropna().unique()
        away_teams = self.games_data['away_team'].dropna().unique()
        
        return sorted(set(home_teams) | set(away_teams))


class TimeAwareSplitter:
    """
    Handles strict time-aware train/test splits for rolling origin evaluation.
    """
    
    @staticmethod
    def get_train_test_split(
        games_data: pd.DataFrame,
        test_season: int,
        test_week: int,
        include_prior_seasons: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split with no look-ahead.
        
        Train: All games from prior seasons + games from test season before test_week
        Test: Games from test_week of test_season
        
        Args:
            games_data: DataFrame of all games
            test_season: Season to test on
            test_week: Week to test on
            include_prior_seasons: Whether to include all prior seasons in training
            
        Returns:
            Tuple of (train_games, test_games)
        """
        # Test set: games from test_week of test_season
        test_games = games_data[
            (games_data['season'] == test_season) &
            (games_data['week'] == test_week) &
            (games_data['game_type'] == 'REG')
        ].copy()
        
        # Train set: prior seasons + earlier weeks of test season
        if include_prior_seasons:
            train_games = games_data[
                (
                    (games_data['season'] < test_season) |
                    (
                        (games_data['season'] == test_season) &
                        (games_data['week'] < test_week)
                    )
                ) &
                (games_data['game_type'] == 'REG') &
                (games_data['home_score'].notna())  # Only completed games
            ].copy()
        else:
            train_games = games_data[
                (games_data['season'] == test_season) &
                (games_data['week'] < test_week) &
                (games_data['game_type'] == 'REG') &
                (games_data['home_score'].notna())
            ].copy()
        
        return train_games, test_games
    
    @staticmethod
    def create_validation_split(
        train_games: pd.DataFrame,
        val_fraction: float = 0.15,
        time_ordered: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create validation split from training data.
        
        Args:
            train_games: Training games
            val_fraction: Fraction to use for validation
            time_ordered: If True, use most recent games for validation
            
        Returns:
            Tuple of (train_games, val_games)
        """
        if time_ordered:
            # Sort by season and week
            train_games = train_games.sort_values(['season', 'week'])
            
            # Take most recent games for validation
            n_val = int(len(train_games) * val_fraction)
            val_games = train_games.tail(n_val).copy()
            train_games = train_games.head(len(train_games) - n_val).copy()
        else:
            # Random split
            val_games = train_games.sample(frac=val_fraction, random_state=42)
            train_games = train_games.drop(val_games.index)
        
        return train_games, val_games


def main():
    """Test data loading."""
    print("\n" + "="*70)
    print("NFL DATA LOADER - TEST")
    print("="*70)
    
    # Initialize loader
    loader = NFLDataLoader(seasons=[2023, 2024], cache_dir="data/", use_cache=True)
    
    # Load data
    pbp, games = loader.load_data()
    
    print(f"\nData Summary:")
    print(f"  Total plays: {len(pbp):,}")
    print(f"  Total games: {len(games):,}")
    print(f"  Seasons: {sorted(games['season'].unique())}")
    print(f"  Teams: {len(loader.get_all_teams())}")
    
    # Test time-aware split
    print("\n" + "="*70)
    print("TIME-AWARE SPLIT TEST")
    print("="*70)
    
    train, test = TimeAwareSplitter.get_train_test_split(
        games, test_season=2024, test_week=6
    )
    
    print(f"\nSplit for 2024 Week 6:")
    print(f"  Training games: {len(train)}")
    print(f"  Test games: {len(test)}")
    print(f"  Train season range: {train['season'].min()} - {train['season'].max()}")
    print(f"  Train week range (2024): {train[train['season']==2024]['week'].min() if len(train[train['season']==2024]) > 0 else 'N/A'} - "
          f"{train[train['season']==2024]['week'].max() if len(train[train['season']==2024]) > 0 else 'N/A'}")
    
    # Create validation split
    train_final, val = TimeAwareSplitter.create_validation_split(train, val_fraction=0.15)
    
    print(f"\nValidation Split:")
    print(f"  Final training games: {len(train_final)}")
    print(f"  Validation games: {len(val)}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

