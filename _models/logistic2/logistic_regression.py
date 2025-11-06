#!/usr/bin/env python3
"""
NFL Logistic Regression Model with Elastic-Net Regularization
==============================================================

Advanced logistic regression model for NFL game prediction featuring:
- Elastic-Net regularization (L1 + L2) for optimal bias-variance tradeoff
- Elo rating system as a feature prior
- Comprehensive feature engineering (EPA, success rates, rest days)
- Proper time-based train/test split to prevent data leakage
- Calibration metrics (Brier score, log loss)
- Support for market spread as a feature (optional)

This model is designed to be hard to beat with well-engineered features
and careful regularization.

Author: NFL Bets Analysis
Date: October 2025
"""

import nflreadpy as nfl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class NFLElasticNetLogisticRegression:
    """
    Advanced logistic regression model with elastic-net regularization for NFL predictions.
    
    Features:
    - Elastic-Net regularization (combines L1 and L2 penalties)
    - Elo rating system
    - EPA-based metrics
    - Success rates
    - Rest days
    - Home field advantage
    - Divisional game indicators
    - Optional market spread integration
    """
    
    def __init__(
        self,
        train_seasons: List[int] = [2022, 2023],
        test_season: int = 2024,
        l1_ratios: List[float] = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv_folds: int = 5,
        random_state: int = 42,
        initial_elo: float = 1500.0,
        elo_k: float = 20.0,
        use_market_spread: bool = False
    ):
        """
        Initialize the NFL Elastic-Net Logistic Regression model.
        
        Args:
            train_seasons: List of seasons to use for training
            test_season: Season to use for testing
            l1_ratios: List of L1 penalty ratios to try (0=L2 only, 1=L1 only)
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            initial_elo: Initial Elo rating for new teams
            elo_k: Elo K-factor for rating updates
            use_market_spread: Whether to include market spread as feature
        """
        self.train_seasons = train_seasons
        self.test_season = test_season
        self.l1_ratios = l1_ratios
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_market_spread = use_market_spread
        
        # Elo system parameters
        self.initial_elo = initial_elo
        self.elo_k = elo_k
        self.elo_ratings = {}
        
        # Data storage
        self.pbp_data = None
        self.games_data = None
        
        # Model components
        self.model = LogisticRegressionCV(
            Cs=10,  # Number of regularization strengths to try
            cv=cv_folds,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=l1_ratios,
            random_state=random_state,
            max_iter=5000,
            scoring='neg_brier_score',  # Optimize for calibration
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Results storage
        self.train_data = None
        self.test_data = None
        
        print(f"Initialized NFL Elastic-Net Logistic Regression Model")
        print(f"Training seasons: {train_seasons}")
        print(f"Test season: {test_season}")
        print(f"L1 ratios: {l1_ratios}")
        print(f"CV folds: {cv_folds}")
    
    def load_data(self):
        """Load play-by-play and schedule data for all seasons."""
        print("\n" + "="*70)
        print("LOADING NFL DATA")
        print("="*70)
        
        all_seasons = self.train_seasons + [self.test_season]
        
        print(f"Loading data for seasons: {all_seasons}")
        
        # Load play-by-play data
        print("Loading play-by-play data...")
        self.pbp_data = nfl.load_pbp(all_seasons).to_pandas()
        
        # Load schedule data
        print("Loading schedule data...")
        self.games_data = nfl.load_schedules(all_seasons).to_pandas()
        
        print(f"\nData loaded successfully:")
        print(f"  - Play-by-play records: {len(self.pbp_data):,}")
        print(f"  - Games: {len(self.games_data):,}")
        
        # Filter to regular season only
        self.games_data = self.games_data[
            self.games_data['game_type'] == 'REG'
        ].copy()
        
        print(f"  - Regular season games: {len(self.games_data):,}")
        
        # Convert gameday to datetime
        self.games_data['gameday'] = pd.to_datetime(self.games_data['gameday'])
        
    def update_elo(self, home_team: str, away_team: str, home_score: float, away_score: float):
        """
        Update Elo ratings based on game result.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_score: Home team's final score
            away_score: Away team's final score
        """
        # Get current ratings
        home_elo = self.elo_ratings.get(home_team, self.initial_elo)
        away_elo = self.elo_ratings.get(away_team, self.initial_elo)
        
        # Expected score (with home field advantage)
        home_advantage = 65  # Elo points for home field
        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))
        
        # Actual result
        if home_score > away_score:
            actual = 1.0
        elif home_score < away_score:
            actual = 0.0
        else:
            actual = 0.5
        
        # Margin of victory multiplier
        score_diff = abs(home_score - away_score)
        mov_multiplier = np.log(max(1, score_diff)) + 1
        
        # Update ratings
        rating_change = self.elo_k * mov_multiplier * (actual - expected_home)
        
        self.elo_ratings[home_team] = home_elo + rating_change
        self.elo_ratings[away_team] = away_elo - rating_change
        
        return home_elo, away_elo
    
    def calculate_team_stats(self, season: int, week: int) -> pd.DataFrame:
        """
        Calculate cumulative team statistics up to (but not including) a specific week.
        
        This ensures we only use historical data available before the game,
        preventing data leakage.
        
        Args:
            season: Season year
            week: Week number (stats calculated for weeks < week)
            
        Returns:
            DataFrame with team statistics
        """
        # Filter games up to the specified week
        if season in self.train_seasons:
            # For training seasons, use all prior training data
            prior_seasons = [s for s in self.train_seasons if s < season]
            games_mask = (
                (self.games_data['season'].isin(prior_seasons)) |
                ((self.games_data['season'] == season) & (self.games_data['week'] < week))
            )
        else:
            # For test season, use all training data + current season up to current week
            games_mask = (
                (self.games_data['season'].isin(self.train_seasons)) |
                ((self.games_data['season'] == season) & (self.games_data['week'] < week))
            )
        
        games_filtered = self.games_data[games_mask].copy()
        
        if len(games_filtered) == 0:
            return pd.DataFrame()
        
        # Get corresponding play-by-play data
        game_ids = games_filtered['game_id'].unique()
        pbp_filtered = self.pbp_data[self.pbp_data['game_id'].isin(game_ids)].copy()
        
        # Calculate stats for each team
        team_stats = []
        teams = pd.concat([
            games_filtered['home_team'],
            games_filtered['away_team']
        ]).unique()
        
        for team in teams:
            if pd.isna(team):
                continue
            
            # Get team's games
            team_games = games_filtered[
                (games_filtered['home_team'] == team) |
                (games_filtered['away_team'] == team)
            ].copy()
            
            if len(team_games) == 0:
                continue
            
            # Calculate game outcomes
            wins = 0
            points_for = []
            points_against = []
            
            for _, game in team_games.iterrows():
                is_home = game['home_team'] == team
                team_score = game['home_score'] if is_home else game['away_score']
                opp_score = game['away_score'] if is_home else game['home_score']
                
                points_for.append(team_score)
                points_against.append(opp_score)
                
                if team_score > opp_score:
                    wins += 1
            
            # Get team's offensive plays
            off_plays = pbp_filtered[pbp_filtered['posteam'] == team].copy()
            
            # Get team's defensive plays
            def_plays = pbp_filtered[pbp_filtered['defteam'] == team].copy()
            
            # EPA-based metrics
            if 'epa' in off_plays.columns and len(off_plays) > 0:
                off_epa = off_plays['epa'].mean()
                off_success_rate = (off_plays['epa'] > 0).mean()
            else:
                off_epa = 0.0
                off_success_rate = 0.5
            
            if 'epa' in def_plays.columns and len(def_plays) > 0:
                def_epa = def_plays['epa'].mean()
                def_success_rate = (def_plays['epa'] > 0).mean()
            else:
                def_epa = 0.0
                def_success_rate = 0.5
            
            # Passing metrics
            pass_plays = off_plays[off_plays['play_type'] == 'pass']
            pass_attempts = len(pass_plays)
            
            if pass_attempts > 0:
                pass_completions = pass_plays['complete_pass'].sum()
                pass_yards = pass_plays['passing_yards'].sum()
                pass_tds = pass_plays['touchdown'].sum()
                interceptions = pass_plays['interception'].sum()
                
                completion_pct = pass_completions / pass_attempts
                yards_per_attempt = pass_yards / pass_attempts
                pass_td_rate = pass_tds / pass_attempts
                int_rate = interceptions / pass_attempts
            else:
                completion_pct = 0.0
                yards_per_attempt = 0.0
                pass_td_rate = 0.0
                int_rate = 0.0
            
            # Rushing metrics
            run_plays = off_plays[off_plays['play_type'] == 'run']
            rush_attempts = len(run_plays)
            
            if rush_attempts > 0:
                rush_yards = run_plays['rushing_yards'].sum()
                rush_tds = run_plays['touchdown'].sum()
                yards_per_rush = rush_yards / rush_attempts
                rush_td_rate = rush_tds / rush_attempts
            else:
                yards_per_rush = 0.0
                rush_td_rate = 0.0
            
            # Defensive metrics
            def_pass_plays = def_plays[def_plays['play_type'] == 'pass']
            def_rush_plays = def_plays[def_plays['play_type'] == 'run']
            
            if len(def_pass_plays) > 0:
                def_pass_yards = def_pass_plays['passing_yards'].sum()
                def_yards_per_pass = def_pass_yards / len(def_pass_plays)
            else:
                def_yards_per_pass = 0.0
            
            if len(def_rush_plays) > 0:
                def_rush_yards = def_rush_plays['rushing_yards'].sum()
                def_yards_per_rush = def_rush_yards / len(def_rush_plays)
            else:
                def_yards_per_rush = 0.0
            
            # Turnovers
            fumbles_lost = off_plays['fumble_lost'].sum() if 'fumble_lost' in off_plays.columns else 0
            turnovers = (interceptions if pass_attempts > 0 else 0) + fumbles_lost
            
            # Defensive turnovers forced
            def_ints = def_pass_plays['interception'].sum() if 'interception' in def_pass_plays.columns else 0
            def_fumbles = def_plays['fumble_lost'].sum() if 'fumble_lost' in def_plays.columns else 0
            def_turnovers_forced = def_ints + def_fumbles
            
            games_played = len(team_games)
            
            stats = {
                'team': team,
                'season': season,
                'week': week,
                'games_played': games_played,
                
                # Win/Loss stats
                'wins': wins,
                'win_pct': wins / games_played if games_played > 0 else 0,
                
                # Scoring
                'ppg': np.mean(points_for) if points_for else 0,
                'pa_pg': np.mean(points_against) if points_against else 0,
                'point_diff': np.mean(points_for) - np.mean(points_against) if points_for else 0,
                
                # EPA metrics
                'off_epa_per_play': off_epa,
                'def_epa_per_play': def_epa,
                'off_success_rate': off_success_rate,
                'def_success_rate': def_success_rate,
                
                # Passing
                'completion_pct': completion_pct,
                'yards_per_pass_attempt': yards_per_attempt,
                'pass_td_rate': pass_td_rate,
                'int_rate': int_rate,
                
                # Rushing
                'yards_per_rush': yards_per_rush,
                'rush_td_rate': rush_td_rate,
                
                # Defense
                'def_yards_per_pass': def_yards_per_pass,
                'def_yards_per_rush': def_yards_per_rush,
                
                # Turnovers
                'turnovers_pg': turnovers / games_played if games_played > 0 else 0,
                'def_turnovers_forced_pg': def_turnovers_forced / games_played if games_played > 0 else 0,
                
                # Pace
                'plays_per_game': len(off_plays) / games_played if games_played > 0 else 0,
            }
            
            team_stats.append(stats)
        
        return pd.DataFrame(team_stats)
    
    def calculate_rest_days(self, team: str, current_gameday: datetime, 
                           season: int, week: int) -> int:
        """
        Calculate days of rest since last game.
        
        Args:
            team: Team abbreviation
            current_gameday: Date of current game
            season: Current season
            week: Current week
            
        Returns:
            Number of rest days (default to 7 if no prior game)
        """
        # Get prior games for this team
        prior_games = self.games_data[
            ((self.games_data['home_team'] == team) | (self.games_data['away_team'] == team)) &
            ((self.games_data['season'] < season) | 
             ((self.games_data['season'] == season) & (self.games_data['week'] < week))) &
            (self.games_data['gameday'] < current_gameday)
        ].sort_values('gameday', ascending=False)
        
        if len(prior_games) > 0:
            last_game_date = prior_games.iloc[0]['gameday']
            rest_days = (current_gameday - last_game_date).days
            return min(rest_days, 14)  # Cap at 14 days
        else:
            return 7  # Default rest
    
    def create_game_features(self, home_stats: Dict, away_stats: Dict,
                            home_elo: float, away_elo: float,
                            home_rest: int, away_rest: int,
                            is_divisional: bool = False,
                            spread: Optional[float] = None) -> Dict:
        """
        Create feature vector by comparing home and away team statistics.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            home_elo: Home team Elo rating
            away_elo: Away team Elo rating
            home_rest: Home team rest days
            away_rest: Away team rest days
            is_divisional: Whether this is a divisional game
            spread: Market spread (if available)
            
        Returns:
            Dictionary of differential features
        """
        # Skip non-statistical columns
        skip_cols = ['team', 'season', 'week', 'games_played', 'wins']
        
        features = {}
        
        # Statistical differentials
        for key in home_stats.keys():
            if key in skip_cols:
                continue
            
            # Create differential feature (home - away)
            feature_name = f'{key}_diff'
            features[feature_name] = home_stats[key] - away_stats[key]
        
        # Elo differential
        features['elo_diff'] = home_elo - away_elo
        
        # Rest differential
        features['rest_diff'] = home_rest - away_rest
        features['rest_advantage'] = 1.0 if home_rest > away_rest else 0.0
        
        # Divisional game indicator
        features['is_divisional'] = 1.0 if is_divisional else 0.0
        
        # Market spread (if available)
        if self.use_market_spread and spread is not None:
            features['market_spread'] = spread
        
        return features
    
    def prepare_dataset(self, season: int, weeks: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Prepare dataset for a given season and weeks.
        
        Args:
            season: Season year
            weeks: List of weeks to include (None = all weeks)
            
        Returns:
            DataFrame with features and targets
        """
        games = self.games_data[self.games_data['season'] == season].copy()
        
        if weeks is not None:
            games = games[games['week'].isin(weeks)]
        
        # Only include games with scores (completed games)
        games = games[games['home_score'].notna() & games['away_score'].notna()]
        
        dataset = []
        
        for _, game in games.iterrows():
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']
            gameday = game['gameday']
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Get team stats up to this week
            team_stats = self.calculate_team_stats(season, week)
            
            if len(team_stats) == 0:
                continue
            
            home_stats_df = team_stats[team_stats['team'] == home_team]
            away_stats_df = team_stats[team_stats['team'] == away_team]
            
            if len(home_stats_df) == 0 or len(away_stats_df) == 0:
                continue
            
            home_stats = home_stats_df.iloc[0].to_dict()
            away_stats = away_stats_df.iloc[0].to_dict()
            
            # Get Elo ratings (before game update)
            home_elo = self.elo_ratings.get(home_team, self.initial_elo)
            away_elo = self.elo_ratings.get(away_team, self.initial_elo)
            
            # Calculate rest days
            home_rest = self.calculate_rest_days(home_team, gameday, season, week)
            away_rest = self.calculate_rest_days(away_team, gameday, season, week)
            
            # Check if divisional game
            # Note: Would need division mapping for accurate detection
            # For now, use simple heuristic or leave as False
            is_divisional = False
            
            # Get market spread if available
            spread = game.get('spread_line', None) if self.use_market_spread else None
            
            # Create features
            features = self.create_game_features(
                home_stats, away_stats,
                home_elo, away_elo,
                home_rest, away_rest,
                is_divisional, spread
            )
            
            # Update Elo ratings for future games
            self.update_elo(home_team, away_team, game['home_score'], game['away_score'])
            
            # Add target (1 = home win, 0 = away win)
            target = 1 if game['home_score'] > game['away_score'] else 0
            
            # Add metadata
            features['target'] = target
            features['game_id'] = game['game_id']
            features['season'] = season
            features['week'] = week
            features['home_team'] = home_team
            features['away_team'] = away_team
            features['home_score'] = game['home_score']
            features['away_score'] = game['away_score']
            features['actual_spread'] = game['home_score'] - game['away_score']
            
            dataset.append(features)
        
        return pd.DataFrame(dataset)
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from all training seasons.
        
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        print("\n" + "="*70)
        print("PREPARING TRAINING DATA")
        print("="*70)
        
        # Reset Elo ratings for training
        self.elo_ratings = {}
        
        all_data = []
        
        for season in self.train_seasons:
            print(f"\nProcessing {season} season...")
            season_data = self.prepare_dataset(season)
            all_data.append(season_data)
            print(f"  - Games processed: {len(season_data)}")
        
        # Combine all training data
        train_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\nTotal training games: {len(train_df)}")
        print(f"Home wins: {train_df['target'].sum()} ({train_df['target'].mean()*100:.1f}%)")
        print(f"Away wins: {len(train_df) - train_df['target'].sum()} ({(1-train_df['target'].mean())*100:.1f}%)")
        
        # Separate features and target
        metadata_cols = ['target', 'game_id', 'season', 'week', 'home_team',
                        'away_team', 'home_score', 'away_score', 'actual_spread']
        self.feature_columns = [col for col in train_df.columns if col not in metadata_cols]
        
        X = train_df[self.feature_columns]
        y = train_df['target']
        
        print(f"\nFeature count: {len(self.feature_columns)}")
        
        # Store training data for later analysis
        self.train_data = train_df
        
        return X, y
    
    def prepare_test_data(self, start_week: int = 6, end_week: int = 18) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare test data from test season.
        
        Args:
            start_week: First week to test on
            end_week: Last week to test on
        
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        print("\n" + "="*70)
        print("PREPARING TEST DATA")
        print("="*70)
        
        print(f"\nProcessing {self.test_season} season (weeks {start_week}-{end_week})...")
        
        # Get weeks to test
        weeks = list(range(start_week, end_week + 1))
        
        test_df = self.prepare_dataset(self.test_season, weeks=weeks)
        
        print(f"Total test games: {len(test_df)}")
        if len(test_df) > 0:
            print(f"Home wins: {test_df['target'].sum()} ({test_df['target'].mean()*100:.1f}%)")
            print(f"Away wins: {len(test_df) - test_df['target'].sum()} ({(1-test_df['target'].mean())*100:.1f}%)")
        
        X = test_df[self.feature_columns]
        y = test_df['target']
        
        # Store test data for later analysis
        self.test_data = test_df
        
        return X, y
    
    def train(self):
        """Train the elastic-net logistic regression model."""
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        # Prepare training data
        X_train, y_train = self.prepare_training_data()
        
        # Handle missing values
        X_train = X_train.fillna(0)
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model with cross-validation
        print("Training elastic-net logistic regression with cross-validation...")
        print(f"  - Testing {len(self.l1_ratios)} L1 ratios")
        print(f"  - {self.cv_folds}-fold cross-validation")
        print(f"  - Optimizing for Brier score (calibration)")
        
        self.model.fit(X_train_scaled, y_train)
        
        print("\n✓ Model training complete!")
        print(f"\nBest parameters:")
        print(f"  - C (inverse regularization): {self.model.C_[0]:.4f}")
        print(f"  - L1 ratio: {self.model.l1_ratio_[0]:.4f}")
        
        # Show feature importance
        self.show_feature_importance()
    
    def show_feature_importance(self, top_n: int = 20):
        """
        Display the most important features based on coefficient magnitude.
        
        Args:
            top_n: Number of top features to display
        """
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_[0]
        })
        
        feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        print("\n" + "="*70)
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print("="*70)
        print("\nPositive coefficient → favors home team win")
        print("Negative coefficient → favors away team win\n")
        
        for i, row in feature_importance.head(top_n).iterrows():
            direction = "+" if row['coefficient'] > 0 else ""
            print(f"{row['feature']:40s} {direction}{row['coefficient']:8.4f}")
    
    def evaluate(self, X, y, dataset_name: str = "Test") -> Dict:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*70)
        print(f"EVALUATING {dataset_name.upper()} PERFORMANCE")
        print("="*70)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        logloss = log_loss(y, y_pred_proba)
        brier = brier_score_loss(y, y_pred_proba)
        
        try:
            auc = roc_auc_score(y, y_pred_proba)
        except:
            auc = None
        
        # Print results
        print(f"\n{'Metric':<30s} {'Value':>10s}")
        print("-" * 42)
        print(f"{'Accuracy':<30s} {accuracy:>10.4f}")
        print(f"{'Log Loss':<30s} {logloss:>10.4f}")
        print(f"{'Brier Score':<30s} {brier:>10.4f}")
        if auc is not None:
            print(f"{'ROC AUC':<30s} {auc:>10.4f}")
        
        # Classification report
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print(classification_report(y, y_pred,
                                   target_names=['Away Win', 'Home Win'],
                                   digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print("-"*70)
        print("CONFUSION MATRIX")
        print("-"*70)
        print(f"                  Predicted")
        print(f"               Away    Home")
        print(f"Actual Away    {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"       Home    {cm[1,0]:4d}    {cm[1,1]:4d}")
        
        # Store results
        results = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'brier_score': brier,
            'auc': auc,
            'y_true': y,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }
        
        return results
    
    def predict_game(self, home_team: str, away_team: str,
                    season: int, week: int) -> Dict:
        """
        Predict the outcome of a specific game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number
            
        Returns:
            Dictionary with prediction details
        """
        # Get team stats up to this week
        team_stats = self.calculate_team_stats(season, week)
        
        home_stats_df = team_stats[team_stats['team'] == home_team]
        away_stats_df = team_stats[team_stats['team'] == away_team]
        
        if len(home_stats_df) == 0 or len(away_stats_df) == 0:
            raise ValueError(f"Insufficient data for {home_team} vs {away_team}")
        
        home_stats = home_stats_df.iloc[0].to_dict()
        away_stats = away_stats_df.iloc[0].to_dict()
        
        # Get Elo ratings
        home_elo = self.elo_ratings.get(home_team, self.initial_elo)
        away_elo = self.elo_ratings.get(away_team, self.initial_elo)
        
        # Estimate rest days (default to 7 for predictions)
        home_rest = 7
        away_rest = 7
        
        # Create features
        features = self.create_game_features(
            home_stats, away_stats,
            home_elo, away_elo,
            home_rest, away_rest,
            is_divisional=False,
            spread=None
        )
        
        # Convert to DataFrame
        X = pd.DataFrame([features])[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        win_prob = self.model.predict_proba(X_scaled)[0][1]
        predicted_winner = home_team if win_prob > 0.5 else away_team
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'season': season,
            'week': week,
            'home_win_probability': win_prob,
            'away_win_probability': 1 - win_prob,
            'predicted_winner': predicted_winner,
            'confidence': abs(win_prob - 0.5) * 2,
            'home_elo': home_elo,
            'away_elo': away_elo
        }
    
    def plot_calibration_curve(self, results: Dict, save_path: Optional[str] = None):
        """Plot calibration curve for model predictions."""
        y_true = results['y_true']
        y_pred_proba = results['y_pred_proba']
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
               label='Model', linewidth=2, markersize=8)
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curve\n(Home Team Win Probability)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nCalibration curve saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_weekly_performance(self, save_path: Optional[str] = None):
        """Plot week-by-week prediction accuracy for test season."""
        if self.test_data is None:
            print("No test data available. Run evaluate on test set first.")
            return
        
        # Get predictions for test data
        X_test = self.test_data[self.feature_columns].fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Add predictions to test data
        test_with_pred = self.test_data.copy()
        test_with_pred['predicted'] = y_pred
        test_with_pred['correct'] = (test_with_pred['target'] == test_with_pred['predicted']).astype(int)
        
        # Calculate weekly accuracy
        weekly_stats = test_with_pred.groupby('week').agg({
            'correct': ['sum', 'count', 'mean']
        }).reset_index()
        weekly_stats.columns = ['week', 'correct', 'total', 'accuracy']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(weekly_stats['week'], weekly_stats['accuracy'],
              alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=2,
                  label='Random Baseline', alpha=0.7)
        ax.axhline(weekly_stats['accuracy'].mean(), color='green',
                  linestyle='--', linewidth=2, label='Overall Average', alpha=0.7)
        
        # Add count labels on bars
        for _, row in weekly_stats.iterrows():
            ax.text(row['week'], row['accuracy'] + 0.02,
                   f"{row['correct']:.0f}/{row['total']:.0f}",
                   ha='center', fontsize=9)
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{self.test_season} Season: Week-by-Week Prediction Accuracy',
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Weekly performance plot saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, output_dir: str = 'results', start_week: int = 6, end_week: int = 18):
        """
        Generate comprehensive evaluation report with visualizations.
        
        Args:
            output_dir: Directory to save results
            start_week: First week to test on
            end_week: Last week to test on
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*70)
        
        # Evaluate training set
        X_train, y_train = self.prepare_training_data()
        train_results = self.evaluate(X_train, y_train, "Training")
        
        # Evaluate test set
        X_test, y_test = self.prepare_test_data(start_week, end_week)
        
        if len(X_test) > 0:
            test_results = self.evaluate(X_test, y_test, "Test")
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            
            self.plot_calibration_curve(
                test_results,
                save_path=f'{output_dir}/calibration_curve.png'
            )
            
            self.plot_weekly_performance(
                save_path=f'{output_dir}/weekly_performance.png'
            )
            
            # Save detailed results to CSV
            test_results_df = self.test_data.copy()
            X_test_filled = self.test_data[self.feature_columns].fillna(0)
            X_test_scaled = self.scaler.transform(X_test_filled)
            
            test_results_df['predicted_winner'] = self.model.predict(X_test_scaled)
            test_results_df['home_win_prob'] = self.model.predict_proba(X_test_scaled)[:, 1]
            test_results_df['correct'] = (test_results_df['target'] == test_results_df['predicted_winner']).astype(int)
            
            results_path = f'{output_dir}/elasticnet_2024_game_results.csv'
            test_results_df.to_csv(results_path, index=False)
            print(f"Detailed predictions saved to: {results_path}")
            
            # Save weekly summary
            weekly_summary = test_results_df.groupby('week').agg({
                'correct': ['sum', 'count', 'mean']
            }).reset_index()
            weekly_summary.columns = ['week', 'correct', 'total', 'accuracy']
            weekly_path = f'{output_dir}/elasticnet_2024_weekly_results.csv'
            weekly_summary.to_csv(weekly_path, index=False)
            print(f"Weekly summary saved to: {weekly_path}")
            
            # Save summary statistics
            summary = {
                'train_accuracy': train_results['accuracy'],
                'train_log_loss': train_results['log_loss'],
                'train_brier_score': train_results['brier_score'],
                'test_accuracy': test_results['accuracy'],
                'test_log_loss': test_results['log_loss'],
                'test_brier_score': test_results['brier_score'],
                'test_auc': test_results['auc'],
                'train_seasons': str(self.train_seasons),
                'test_season': self.test_season,
                'test_weeks': f'{start_week}-{end_week}',
                'n_features': len(self.feature_columns),
                'n_train_games': len(y_train),
                'n_test_games': len(y_test),
                'best_C': self.model.C_[0],
                'best_l1_ratio': self.model.l1_ratio_[0]
            }
            
            summary_df = pd.DataFrame([summary])
            summary_path = f'{output_dir}/elasticnet_2024_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"Model summary saved to: {summary_path}")
        else:
            print("No test data available for evaluation.")
        
        print("\n" + "="*70)
        print("✓ REPORT GENERATION COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {output_dir}/")


def main():
    """Main function to run the complete analysis."""
    
    # Initialize model
    model = NFLElasticNetLogisticRegression(
        train_seasons=[2022, 2023],
        test_season=2024,
        l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv_folds=5,
        random_state=42,
        use_market_spread=False
    )
    
    # Load data
    model.load_data()
    
    # Train model
    model.train()
    
    # Generate comprehensive report (weeks 6-18)
    model.generate_report(output_dir='results', start_week=6, end_week=18)
    
    # Example: Predict a specific game
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)
    
    try:
        # Predict Week 6, 2024 game (example)
        prediction = model.predict_game('KC', 'BUF', season=2024, week=6)
        
        print(f"\nGame: {prediction['away_team']} @ {prediction['home_team']}")
        print(f"Week {prediction['week']}, {prediction['season']}")
        print(f"\nElo Ratings: {prediction['home_team']} {prediction['home_elo']:.0f}, {prediction['away_team']} {prediction['away_elo']:.0f}")
        print(f"\nPredicted Winner: {prediction['predicted_winner']}")
        print(f"Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"Away Win Probability: {prediction['away_win_probability']:.1%}")
        print(f"Confidence: {prediction['confidence']:.1%}")
    except Exception as e:
        print(f"Could not generate example prediction: {e}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

