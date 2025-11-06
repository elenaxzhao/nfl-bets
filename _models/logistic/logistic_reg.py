"""
NFL Logistic Regression Model for Win/Loss Prediction
======================================================

This module implements a logistic regression model to predict NFL game outcomes
using historical team statistics from 2022-2023 to predict 2024 season games.

Features:
- Train on 2022 and 2023 seasons
- Test on 2024 season with rolling predictions (avoid data leakage)
- Comprehensive team-level statistics
- Multiple evaluation metrics (accuracy, log loss, Brier score, calibration)
- Visualization of results and model performance

Author: NFL Betting Analysis
Date: October 2025
"""

import nflreadpy as nfl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
import warnings
warnings.filterwarnings('ignore')


class NFLLogisticRegression:
    """
    Logistic regression model for predicting NFL game outcomes.
    
    The model predicts win probabilities for each game based on historical
    team-level statistics. Training is performed on 2022-2023 data, with
    testing on 2024 using a rolling time-based approach to prevent data leakage.
    """
    
    def __init__(self, train_seasons: List[int] = [2022, 2023], 
                 test_season: int = 2024,
                 random_state: int = 42):
        """
        Initialize the NFL Logistic Regression model.
        
        Args:
            train_seasons: List of seasons to use for training (default: [2022, 2023])
            test_season: Season to use for testing (default: 2024)
            random_state: Random state for reproducibility
        """
        self.train_seasons = train_seasons
        self.test_season = test_season
        self.random_state = random_state
        
        # Data storage
        self.pbp_data = None
        self.games_data = None
        
        # Model components
        self.model = LogisticRegression(
            random_state=random_state, 
            max_iter=1000,
            solver='lbfgs',
            C=1.0
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Results storage
        self.train_results = None
        self.test_results = None
        
        print(f"Initialized NFL Logistic Regression Model")
        print(f"Training seasons: {train_seasons}")
        print(f"Test season: {test_season}")
    
    def load_data(self):
        """Load play-by-play and schedule data for all seasons."""
        print("\n" + "="*60)
        print("LOADING NFL DATA")
        print("="*60)
        
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
        # Use all training seasons + current season up to current week
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
            
            # Offensive statistics
            pass_plays = off_plays[off_plays['play_type'] == 'pass']
            run_plays = off_plays[off_plays['play_type'] == 'run']
            
            pass_attempts = len(pass_plays)
            pass_completions = pass_plays['complete_pass'].sum()
            pass_yards = pass_plays['passing_yards'].sum()
            pass_tds = pass_plays['touchdown'].sum()
            interceptions = pass_plays['interception'].sum()
            
            rush_attempts = len(run_plays)
            rush_yards = run_plays['rushing_yards'].sum()
            rush_tds = run_plays['touchdown'].sum()
            fumbles_lost = off_plays['fumble_lost'].sum()
            
            # Defensive statistics
            def_pass_plays = def_plays[def_plays['play_type'] == 'pass']
            def_run_plays = def_plays[def_plays['play_type'] == 'run']
            
            def_pass_attempts = len(def_pass_plays)
            def_pass_completions = def_pass_plays['complete_pass'].sum()
            def_pass_yards = def_pass_plays['passing_yards'].sum()
            def_pass_tds = def_pass_plays['touchdown'].sum()
            
            def_rush_attempts = len(def_run_plays)
            def_rush_yards = def_run_plays['rushing_yards'].sum()
            def_rush_tds = def_run_plays['touchdown'].sum()
            
            # Defensive turnovers created
            def_interceptions = def_pass_plays['interception'].sum()
            def_fumbles_recovered = def_plays['fumble_lost'].sum()
            
            # Calculate rates and averages
            games_played = len(team_games)
            
            stats = {
                'team': team,
                'season': season,
                'week': week,
                'games_played': games_played,
                
                # Win/Loss stats
                'wins': wins,
                'losses': games_played - wins,
                'win_pct': wins / games_played if games_played > 0 else 0,
                
                # Scoring
                'ppg': np.mean(points_for) if points_for else 0,
                'pa_pg': np.mean(points_against) if points_against else 0,
                'point_diff': np.mean(points_for) - np.mean(points_against) if points_for else 0,
                
                # Offensive passing
                'pass_attempts_pg': pass_attempts / games_played if games_played > 0 else 0,
                'completion_pct': pass_completions / pass_attempts if pass_attempts > 0 else 0,
                'pass_yards_pg': pass_yards / games_played if games_played > 0 else 0,
                'yards_per_pass_attempt': pass_yards / pass_attempts if pass_attempts > 0 else 0,
                'pass_td_rate': pass_tds / pass_attempts if pass_attempts > 0 else 0,
                'int_rate': interceptions / pass_attempts if pass_attempts > 0 else 0,
                
                # Offensive rushing
                'rush_attempts_pg': rush_attempts / games_played if games_played > 0 else 0,
                'rush_yards_pg': rush_yards / games_played if games_played > 0 else 0,
                'yards_per_rush': rush_yards / rush_attempts if rush_attempts > 0 else 0,
                'rush_td_rate': rush_tds / rush_attempts if rush_attempts > 0 else 0,
                
                # Turnovers
                'turnovers_pg': (interceptions + fumbles_lost) / games_played if games_played > 0 else 0,
                
                # Defensive passing
                'def_pass_attempts_pg': def_pass_attempts / games_played if games_played > 0 else 0,
                'def_completion_pct': def_pass_completions / def_pass_attempts if def_pass_attempts > 0 else 0,
                'def_pass_yards_pg': def_pass_yards / games_played if games_played > 0 else 0,
                'def_yards_per_pass': def_pass_yards / def_pass_attempts if def_pass_attempts > 0 else 0,
                'def_pass_td_rate': def_pass_tds / def_pass_attempts if def_pass_attempts > 0 else 0,
                
                # Defensive rushing
                'def_rush_attempts_pg': def_rush_attempts / games_played if games_played > 0 else 0,
                'def_rush_yards_pg': def_rush_yards / games_played if games_played > 0 else 0,
                'def_yards_per_rush': def_rush_yards / def_rush_attempts if def_rush_attempts > 0 else 0,
                'def_rush_td_rate': def_rush_tds / def_rush_attempts if def_rush_attempts > 0 else 0,
                
                # Defensive turnovers
                'def_turnovers_forced_pg': (def_interceptions + def_fumbles_recovered) / games_played if games_played > 0 else 0,
            }
            
            team_stats.append(stats)
        
        return pd.DataFrame(team_stats)
    
    def create_game_features(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """
        Create feature vector by comparing home and away team statistics.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            Dictionary of differential features
        """
        # Skip non-statistical columns
        skip_cols = ['team', 'season', 'week', 'games_played', 'wins', 'losses']
        
        features = {}
        for key in home_stats.keys():
            if key in skip_cols:
                continue
            
            # Create differential feature (home - away)
            feature_name = f'{key}_diff'
            features[feature_name] = home_stats[key] - away_stats[key]
        
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
        
        dataset = []
        
        for _, game in games.iterrows():
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']
            
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
            
            # Create features
            features = self.create_game_features(home_stats, away_stats)
            
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
        print("\n" + "="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        
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
    
    def prepare_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare test data from test season.
        
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        print("\n" + "="*60)
        print("PREPARING TEST DATA")
        print("="*60)
        
        print(f"\nProcessing {self.test_season} season...")
        test_df = self.prepare_dataset(self.test_season)
        
        print(f"Total test games: {len(test_df)}")
        print(f"Home wins: {test_df['target'].sum()} ({test_df['target'].mean()*100:.1f}%)")
        print(f"Away wins: {len(test_df) - test_df['target'].sum()} ({(1-test_df['target'].mean())*100:.1f}%)")
        
        X = test_df[self.feature_columns]
        y = test_df['target']
        
        # Store test data for later analysis
        self.test_data = test_df
        
        return X, y
    
    def train(self):
        """Train the logistic regression model."""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Prepare training data
        X_train, y_train = self.prepare_training_data()
        
        # Handle missing values
        X_train = X_train.fillna(0)
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print("Training logistic regression...")
        self.model.fit(X_train_scaled, y_train)
        
        print("\n✓ Model training complete!")
        
        # Show feature importance
        self.show_feature_importance()
        
    def show_feature_importance(self, top_n: int = 15):
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
        
        print("\n" + "="*60)
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print("="*60)
        print("\nPositive coefficient → favors home team win")
        print("Negative coefficient → favors away team win\n")
        
        for i, row in feature_importance.head(top_n).iterrows():
            direction = "+" if row['coefficient'] > 0 else ""
            print(f"{row['feature']:35s} {direction}{row['coefficient']:8.4f}")
    
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
        print("\n" + "="*60)
        print(f"EVALUATING {dataset_name.upper()} PERFORMANCE")
        print("="*60)
        
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
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT")
        print("-"*60)
        print(classification_report(y, y_pred, 
                                   target_names=['Away Win', 'Home Win'],
                                   digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print("-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
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
        
        # Create features
        features = self.create_game_features(home_stats, away_stats)
        
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
            'confidence': abs(win_prob - 0.5) * 2
        }
    
    def plot_calibration_curve(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot calibration curve for model predictions.
        
        Args:
            results: Results dictionary from evaluate()
            save_path: Path to save plot (if None, displays plot)
        """
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
    
    def plot_prediction_distribution(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot distribution of prediction probabilities.
        
        Args:
            results: Results dictionary from evaluate()
            save_path: Path to save plot (if None, displays plot)
        """
        y_true = results['y_true']
        y_pred_proba = results['y_pred_proba']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Overall distribution
        ax = axes[0]
        ax.hist(y_pred_proba, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax.set_xlabel('Predicted Home Win Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Predictions', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distributions by actual outcome
        ax = axes[1]
        home_wins = y_pred_proba[y_true == 1]
        away_wins = y_pred_proba[y_true == 0]
        
        ax.hist(home_wins, bins=20, alpha=0.6, label='Actual Home Wins', 
               edgecolor='black', color='green')
        ax.hist(away_wins, bins=20, alpha=0.6, label='Actual Away Wins', 
               edgecolor='black', color='orange')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax.set_xlabel('Predicted Home Win Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Predictions by Actual Outcome', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction distribution saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_weekly_performance(self, save_path: Optional[str] = None):
        """
        Plot week-by-week prediction accuracy for test season.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
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
    
    def generate_report(self, output_dir: str = 'results'):
        """
        Generate comprehensive evaluation report with visualizations.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        # Evaluate training set
        X_train, y_train = self.prepare_training_data()
        train_results = self.evaluate(X_train, y_train, "Training")
        
        # Evaluate test set
        X_test, y_test = self.prepare_test_data()
        test_results = self.evaluate(X_test, y_test, "Test")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        self.plot_calibration_curve(
            test_results, 
            save_path=f'{output_dir}/calibration_curve.png'
        )
        
        self.plot_prediction_distribution(
            test_results,
            save_path=f'{output_dir}/prediction_distribution.png'
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
        
        results_path = f'{output_dir}/detailed_predictions_{self.test_season}.csv'
        test_results_df.to_csv(results_path, index=False)
        print(f"Detailed predictions saved to: {results_path}")
        
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
            'n_features': len(self.feature_columns),
            'n_train_games': len(y_train),
            'n_test_games': len(y_test)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = f'{output_dir}/model_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Model summary saved to: {summary_path}")
        
        print("\n" + "="*60)
        print("✓ REPORT GENERATION COMPLETE")
        print("="*60)
        print(f"\nResults saved to: {output_dir}/")


def main():
    """Main function to run the complete analysis."""
    
    # Initialize model
    model = NFLLogisticRegression(
        train_seasons=[2022, 2023],
        test_season=2024,
        random_state=42
    )
    
    # Load data
    model.load_data()
    
    # Train model
    model.train()
    
    # Generate comprehensive report
    model.generate_report(output_dir='results')
    
    # Example: Predict a specific game
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    try:
        # Predict Week 5, 2024 game (example)
        prediction = model.predict_game('KC', 'BUF', season=2024, week=5)
        
        print(f"\nGame: {prediction['away_team']} @ {prediction['home_team']}")
        print(f"Week {prediction['week']}, {prediction['season']}")
        print(f"\nPredicted Winner: {prediction['predicted_winner']}")
        print(f"Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"Away Win Probability: {prediction['away_win_probability']:.1%}")
        print(f"Confidence: {prediction['confidence']:.1%}")
    except Exception as e:
        print(f"Could not generate example prediction: {e}")
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

