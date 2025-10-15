#!/usr/bin/env python3
"""
Feature Engineering Module
===========================

Computes per-game features from play-by-play data, including:
- Offensive/Defensive EPA and success rates
- Pace (plays per game, drives per game)
- Turnover rates
- Elo ratings (optional)
- Home/away flags, divisional indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Computes features for NFL games from play-by-play data.
    """
    
    def __init__(
        self,
        use_epa: bool = True,
        use_success_rate: bool = True,
        use_turnover_rate: bool = True,
        use_pace: bool = True,
        use_drives: bool = True,
        use_elo: bool = True,
        initial_elo: float = 1500.0,
        elo_k: float = 20.0,
    ):
        """
        Initialize feature engineer.
        
        Args:
            use_epa: Compute EPA features
            use_success_rate: Compute success rate features
            use_turnover_rate: Compute turnover rate features
            use_pace: Compute pace features
            use_drives: Compute drive features
            use_elo: Compute Elo ratings
            initial_elo: Initial Elo rating for new teams
            elo_k: Elo K-factor for updates
        """
        self.use_epa = use_epa
        self.use_success_rate = use_success_rate
        self.use_turnover_rate = use_turnover_rate
        self.use_pace = use_pace
        self.use_drives = use_drives
        self.use_elo = use_elo
        
        # Elo settings
        self.initial_elo = initial_elo
        self.elo_k = elo_k
        self.elo_ratings = {}  # Team -> current Elo
        
    def compute_game_features(
        self,
        game_row: pd.Series,
        pbp_data: pd.DataFrame,
        for_team: str,
        is_home: bool
    ) -> Dict[str, float]:
        """
        Compute features for a single game from a team's perspective.
        
        Args:
            game_row: Row from games dataframe
            pbp_data: Play-by-play data for all games
            for_team: Team to compute features for
            is_home: Whether this team is home
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Get plays for this game
        game_pbp = pbp_data[pbp_data['game_id'] == game_row['game_id']]
        
        if len(game_pbp) == 0:
            return self._get_default_features()
        
        # Determine team and opponent
        if is_home:
            team = game_row['home_team']
            opp = game_row['away_team']
            team_score = game_row.get('home_score', 0)
            opp_score = game_row.get('away_score', 0)
        else:
            team = game_row['away_team']
            opp = game_row['home_team']
            team_score = game_row.get('away_score', 0)
            opp_score = game_row.get('home_score', 0)
        
        # Get team's offensive plays
        team_offense = game_pbp[game_pbp['posteam'] == team]
        team_defense = game_pbp[game_pbp['defteam'] == team]
        
        # EPA features
        if self.use_epa and 'epa' in game_pbp.columns:
            features['off_epa_per_play'] = team_offense['epa'].mean() if len(team_offense) > 0 else 0.0
            features['def_epa_per_play'] = team_defense['epa'].mean() if len(team_defense) > 0 else 0.0
        else:
            features['off_epa_per_play'] = 0.0
            features['def_epa_per_play'] = 0.0
        
        # Success rate (EPA > 0)
        if self.use_success_rate and 'epa' in game_pbp.columns:
            features['off_success_rate'] = (team_offense['epa'] > 0).mean() if len(team_offense) > 0 else 0.5
            features['def_success_rate'] = (team_defense['epa'] > 0).mean() if len(team_defense) > 0 else 0.5
        else:
            features['off_success_rate'] = 0.5
            features['def_success_rate'] = 0.5
        
        # Turnover rates
        if self.use_turnover_rate:
            # Interceptions + fumbles lost
            if 'interception' in game_pbp.columns and 'fumble_lost' in game_pbp.columns:
                team_tos = (team_offense['interception'].fillna(0) + team_offense['fumble_lost'].fillna(0)).sum()
                opp_tos = (team_defense['interception'].fillna(0) + team_defense['fumble_lost'].fillna(0)).sum()
                
                n_plays_off = len(team_offense)
                n_plays_def = len(team_defense)
                
                features['turnover_rate'] = team_tos / n_plays_off if n_plays_off > 0 else 0.0
                features['opp_turnover_rate'] = opp_tos / n_plays_def if n_plays_def > 0 else 0.0
            else:
                features['turnover_rate'] = 0.0
                features['opp_turnover_rate'] = 0.0
        else:
            features['turnover_rate'] = 0.0
            features['opp_turnover_rate'] = 0.0
        
        # Pace features
        if self.use_pace:
            features['plays_per_game'] = len(team_offense)
        else:
            features['plays_per_game'] = 0.0
        
        # Drive features
        if self.use_drives and 'drive' in game_pbp.columns:
            n_drives = game_pbp[game_pbp['posteam'] == team]['drive'].nunique()
            features['drives_per_game'] = n_drives
        else:
            features['drives_per_game'] = 0.0
        
        # Points scored/allowed
        features['points_scored'] = team_score
        features['points_allowed'] = opp_score
        
        # Win/loss
        features['won'] = 1.0 if team_score > opp_score else 0.0
        
        # Home/away
        features['is_home'] = 1.0 if is_home else 0.0
        
        return features
    
    def compute_features_for_games(
        self,
        games: pd.DataFrame,
        pbp_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute features for multiple games.
        
        Args:
            games: DataFrame of games
            pbp_data: Play-by-play data
            
        Returns:
            DataFrame with features per team per game
        """
        features_list = []
        
        for _, game in games.iterrows():
            game_id = game['game_id']
            season = game['season']
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Compute features for home team
            home_features = self.compute_game_features(
                game, pbp_data, home_team, is_home=True
            )
            home_features.update({
                'game_id': game_id,
                'season': season,
                'week': week,
                'team': home_team,
                'opponent': away_team,
            })
            features_list.append(home_features)
            
            # Compute features for away team
            away_features = self.compute_game_features(
                game, pbp_data, away_team, is_home=False
            )
            away_features.update({
                'game_id': game_id,
                'season': season,
                'week': week,
                'team': away_team,
                'opponent': home_team,
            })
            features_list.append(away_features)
        
        return pd.DataFrame(features_list)
    
    def update_elo_ratings(self, games_features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and add Elo ratings to game features.
        
        Elo is updated sequentially in time order.
        
        Args:
            games_features: DataFrame with game features
            
        Returns:
            DataFrame with Elo ratings added
        """
        if not self.use_elo:
            games_features['elo'] = self.initial_elo
            return games_features
        
        # Sort by time
        games_features = games_features.sort_values(['season', 'week', 'game_id']).copy()
        
        # Initialize Elo ratings
        teams = games_features['team'].unique()
        for team in teams:
            if team not in self.elo_ratings:
                self.elo_ratings[team] = self.initial_elo
        
        # Add Elo column
        elo_before_game = []
        
        # Process each game
        processed_games = set()
        
        for idx, row in games_features.iterrows():
            game_id = row['game_id']
            team = row['team']
            opp = row['opponent']
            won = row['won']
            
            # Record Elo before game
            elo_before_game.append(self.elo_ratings.get(team, self.initial_elo))
            
            # Update Elo (only once per game)
            if game_id not in processed_games:
                team_elo = self.elo_ratings.get(team, self.initial_elo)
                opp_elo = self.elo_ratings.get(opp, self.initial_elo)
                
                # Expected score
                expected = 1 / (1 + 10 ** ((opp_elo - team_elo) / 400))
                
                # Update
                team_new_elo = team_elo + self.elo_k * (won - expected)
                opp_new_elo = opp_elo + self.elo_k * ((1 - won) - (1 - expected))
                
                self.elo_ratings[team] = team_new_elo
                self.elo_ratings[opp] = opp_new_elo
                
                processed_games.add(game_id)
        
        games_features['elo'] = elo_before_game
        
        return games_features
    
    def compute_matchup_features(
        self,
        home_team: str,
        away_team: str,
        home_recent_features: pd.DataFrame,
        away_recent_features: pd.DataFrame,
        is_divisional: bool = False,
        week: int = None
    ) -> Dict[str, float]:
        """
        Compute matchup-specific features (home_minus_away diffs).
        
        Args:
            home_team: Home team
            away_team: Away team
            home_recent_features: Recent game features for home team
            away_recent_features: Recent game features for away team
            is_divisional: Whether this is a divisional game
            week: Current week
            
        Returns:
            Dictionary of matchup features
        """
        matchup = {}
        
        # Compute averages of recent features
        feature_cols = [
            'off_epa_per_play', 'def_epa_per_play',
            'off_success_rate', 'def_success_rate',
            'turnover_rate', 'opp_turnover_rate',
            'plays_per_game', 'drives_per_game',
            'points_scored', 'points_allowed',
        ]
        
        if self.use_elo:
            feature_cols.append('elo')
        
        for col in feature_cols:
            if col in home_recent_features.columns and col in away_recent_features.columns:
                home_avg = home_recent_features[col].mean()
                away_avg = away_recent_features[col].mean()
                
                # Compute difference
                matchup[f'{col}_diff'] = home_avg - away_avg
            else:
                matchup[f'{col}_diff'] = 0.0
        
        # Add flags
        matchup['is_home'] = 1.0
        matchup['is_divisional'] = 1.0 if is_divisional else 0.0
        
        # Add week (normalized)
        if week is not None:
            matchup['week_normalized'] = week / 18.0
        else:
            matchup['week_normalized'] = 0.0
        
        return matchup
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when data is unavailable."""
        return {
            'off_epa_per_play': 0.0,
            'def_epa_per_play': 0.0,
            'off_success_rate': 0.5,
            'def_success_rate': 0.5,
            'turnover_rate': 0.0,
            'opp_turnover_rate': 0.0,
            'plays_per_game': 0.0,
            'drives_per_game': 0.0,
            'points_scored': 0.0,
            'points_allowed': 0.0,
            'won': 0.0,
            'is_home': 0.0,
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        feature_names = []
        
        if self.use_epa:
            feature_names.extend(['off_epa_per_play', 'def_epa_per_play'])
        
        if self.use_success_rate:
            feature_names.extend(['off_success_rate', 'def_success_rate'])
        
        if self.use_turnover_rate:
            feature_names.extend(['turnover_rate', 'opp_turnover_rate'])
        
        if self.use_pace:
            feature_names.append('plays_per_game')
        
        if self.use_drives:
            feature_names.append('drives_per_game')
        
        if self.use_elo:
            feature_names.append('elo')
        
        # Always include
        feature_names.extend([
            'points_scored', 'points_allowed', 'won', 'is_home'
        ])
        
        return feature_names


def main():
    """Test feature engineering."""
    from data_loader import NFLDataLoader
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING - TEST")
    print("="*70)
    
    # Load data
    loader = NFLDataLoader(seasons=[2023, 2024], cache_dir="data/", use_cache=True)
    pbp, games = loader.load_data()
    
    # Initialize feature engineer
    engineer = FeatureEngineer(
        use_epa=True,
        use_success_rate=True,
        use_turnover_rate=True,
        use_pace=True,
        use_drives=True,
        use_elo=True
    )
    
    # Get a subset of games
    test_games = games[
        (games['season'] == 2024) &
        (games['week'] <= 5) &
        (games['game_type'] == 'REG') &
        (games['home_score'].notna())
    ].head(20)
    
    print(f"\nComputing features for {len(test_games)} games...")
    
    # Compute features
    features_df = engineer.compute_features_for_games(test_games, pbp)
    
    print(f"✓ Computed features for {len(features_df)} team-game records")
    
    # Update Elo
    features_df = engineer.update_elo_ratings(features_df)
    
    print(f"\nFeature columns: {features_df.columns.tolist()}")
    print(f"\nSample features:")
    print(features_df[['team', 'season', 'week', 'off_epa_per_play', 'def_epa_per_play', 
                       'points_scored', 'points_allowed', 'elo']].head(10))
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

