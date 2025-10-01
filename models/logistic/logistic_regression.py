import nflreadpy as nfl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class NFLGamePredictor:
    """
    A logistic regression model to predict NFL game outcomes and point spreads
    using team statistics from the first 5 games of the season.
    """
    
    def __init__(self, train_seasons: List[int] = [2023], test_season: int = 2024, test_start_week: int = 10):
        self.train_seasons = train_seasons
        self.test_season = test_season
        self.test_start_week = test_start_week
        self.pbp_data = None
        self.games_data = None
        self.team_stats = None
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self):
        """Load play-by-play and games data for specified seasons."""
        print("Loading NFL data...")
        
        # Load data for training seasons and test season
        all_seasons = self.train_seasons + [self.test_season]
        
        # Load play-by-play data
        self.pbp_data = nfl.load_pbp(all_seasons).to_pandas()
        
        # Load games data
        self.games_data = nfl.load_schedules(all_seasons).to_pandas()
        
        print(f"Loaded {len(self.pbp_data)} plays and {len(self.games_data)} games")
        print(f"Training seasons: {self.train_seasons}")
        print(f"Test season: {self.test_season} (starting week {self.test_start_week})")
        
    def calculate_team_stats(self, week: int) -> pd.DataFrame:
        """
        Calculate team statistics up to a specific week.
        
        Args:
            week: The week up to which to calculate stats
            
        Returns:
            DataFrame with team statistics
        """
        # Filter games up to the specified week
        games_up_to_week = self.games_data[
            (self.games_data['week'] <= week) & 
            (self.games_data['game_type'] == 'REG')
        ].copy()
        
        # Filter play-by-play data for these games
        game_ids = games_up_to_week['game_id'].unique()
        pbp_filtered = self.pbp_data[self.pbp_data['game_id'].isin(game_ids)].copy()
        
        team_stats = []
        
        for team in games_up_to_week['home_team'].unique():
            if pd.isna(team):
                continue
                
            # Get team's games
            team_games = games_up_to_week[
                (games_up_to_week['home_team'] == team) | 
                (games_up_to_week['away_team'] == team)
            ]
            
            if len(team_games) == 0:
                continue
                
            # Get team's plays
            team_plays = pbp_filtered[
                (pbp_filtered['posteam'] == team) | 
                (pbp_filtered['defteam'] == team)
            ]
            
            # Calculate offensive stats
            offensive_plays = team_plays[team_plays['posteam'] == team]
            defensive_plays = team_plays[team_plays['defteam'] == team]
            
            # Basic stats
            total_games = len(team_games)
            wins = 0
            total_points_for = 0
            total_points_against = 0
            
            for _, game in team_games.iterrows():
                if game['home_team'] == team:
                    team_score = game['home_score']
                    opp_score = game['away_score']
                else:
                    team_score = game['away_score']
                    opp_score = game['home_score']
                
                total_points_for += team_score
                total_points_against += opp_score
                
                if team_score > opp_score:
                    wins += 1
            
            # Offensive statistics
            pass_attempts = len(offensive_plays[offensive_plays['play_type'] == 'pass'])
            pass_completions = len(offensive_plays[
                (offensive_plays['play_type'] == 'pass') & 
                (offensive_plays['complete_pass'] == 1)
            ])
            pass_yards = offensive_plays[offensive_plays['play_type'] == 'pass']['passing_yards'].sum()
            rush_attempts = len(offensive_plays[offensive_plays['play_type'] == 'run'])
            rush_yards = offensive_plays[offensive_plays['play_type'] == 'run']['rushing_yards'].sum()
            turnovers = len(offensive_plays[offensive_plays['interception'] == 1]) + \
                       len(offensive_plays[offensive_plays['fumble_lost'] == 1])
            
            # Defensive statistics
            def_pass_attempts = len(defensive_plays[defensive_plays['play_type'] == 'pass'])
            def_pass_completions = len(defensive_plays[
                (defensive_plays['play_type'] == 'pass') & 
                (defensive_plays['complete_pass'] == 1)
            ])
            def_pass_yards = defensive_plays[defensive_plays['play_type'] == 'pass']['passing_yards'].sum()
            def_rush_attempts = len(defensive_plays[defensive_plays['play_type'] == 'run'])
            def_rush_yards = defensive_plays[defensive_plays['play_type'] == 'run']['rushing_yards'].sum()
            def_turnovers = len(defensive_plays[defensive_plays['interception'] == 1]) + \
                           len(defensive_plays[defensive_plays['fumble_lost'] == 1])
            
            # Calculate rates and averages
            win_pct = wins / total_games if total_games > 0 else 0
            points_for_avg = total_points_for / total_games if total_games > 0 else 0
            points_against_avg = total_points_against / total_games if total_games > 0 else 0
            completion_pct = pass_completions / pass_attempts if pass_attempts > 0 else 0
            yards_per_pass = pass_yards / pass_attempts if pass_attempts > 0 else 0
            yards_per_rush = rush_yards / rush_attempts if rush_attempts > 0 else 0
            turnover_rate = turnovers / total_games if total_games > 0 else 0
            
            # Defensive rates
            def_completion_pct = def_pass_completions / def_pass_attempts if def_pass_attempts > 0 else 0
            def_yards_per_pass = def_pass_yards / def_pass_attempts if def_pass_attempts > 0 else 0
            def_yards_per_rush = def_rush_yards / def_rush_attempts if def_rush_attempts > 0 else 0
            def_turnover_rate = def_turnovers / total_games if total_games > 0 else 0
            
            team_stats.append({
                'team': team,
                'week': week,
                'games_played': total_games,
                'win_pct': win_pct,
                'points_for_avg': points_for_avg,
                'points_against_avg': points_against_avg,
                'point_differential': points_for_avg - points_against_avg,
                'pass_attempts_avg': pass_attempts / total_games if total_games > 0 else 0,
                'completion_pct': completion_pct,
                'yards_per_pass': yards_per_pass,
                'rush_attempts_avg': rush_attempts / total_games if total_games > 0 else 0,
                'yards_per_rush': yards_per_rush,
                'turnover_rate': turnover_rate,
                'def_completion_pct': def_completion_pct,
                'def_yards_per_pass': def_yards_per_pass,
                'def_yards_per_rush': def_yards_per_rush,
                'def_turnover_rate': def_turnover_rate,
            })
        
        return pd.DataFrame(team_stats)
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data using full training seasons and early test season.
        
        Returns:
            Tuple of (features, target, point_spreads)
        """
        print("Preparing training data...")
        
        training_data = []
        
        # Use all games from training seasons
        for season in self.train_seasons:
            season_games = self.games_data[
                (self.games_data['season'] == season) & 
                (self.games_data['game_type'] == 'REG')
            ].copy()
            
            for _, game in season_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']
                
                if pd.isna(home_team) or pd.isna(away_team):
                    continue
                
                # Calculate team stats up to the current week
                team_stats = self.calculate_team_stats_for_game(game)
                
                if team_stats is None:
                    continue
                
                home_stats, away_stats = team_stats
                
                # Create feature vector (home team stats - away team stats)
                features = self.create_feature_vector(home_stats, away_stats)
                
                # Target: 1 if home team wins, 0 if away team wins
                home_score = game['home_score']
                away_score = game['away_score']
                target = 1 if home_score > away_score else 0
                
                # Add point spread
                features['point_spread'] = home_score - away_score
                
                training_data.append({**features, 'target': target, 'game_id': game['game_id']})
        
        # Add early 2024 games (weeks 1-9) for additional training data
        early_2024_games = self.games_data[
            (self.games_data['season'] == self.test_season) & 
            (self.games_data['week'] < self.test_start_week) &
            (self.games_data['game_type'] == 'REG')
        ].copy()
        
        for _, game in early_2024_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Calculate team stats up to the current week
            team_stats = self.calculate_team_stats_for_game(game)
            
            if team_stats is None:
                continue
            
            home_stats, away_stats = team_stats
            
            # Create feature vector (home team stats - away team stats)
            features = self.create_feature_vector(home_stats, away_stats)
            
            # Target: 1 if home team wins, 0 if away team wins
            home_score = game['home_score']
            away_score = game['away_score']
            target = 1 if home_score > away_score else 0
            
            # Add point spread
            features['point_spread'] = home_score - away_score
            
            training_data.append({**features, 'target': target, 'game_id': game['game_id']})
        
        training_df = pd.DataFrame(training_data)
        
        if len(training_df) == 0:
            raise ValueError("No training data available")
        
        # Separate features and target
        self.feature_columns = [col for col in training_df.columns if col not in ['target', 'game_id', 'point_spread']]
        X = training_df[self.feature_columns]
        y = training_df['target']
        
        print(f"Training data shape: {X.shape}")
        print(f"Features: {self.feature_columns}")
        
        return X, y, training_df['point_spread']
    
    def calculate_team_stats_for_game(self, game) -> Tuple[Dict, Dict]:
        """
        Calculate team stats for a specific game up to that point in the season.
        
        Args:
            game: Game row from games_data
            
        Returns:
            Tuple of (home_stats, away_stats) or None if stats not available
        """
        home_team = game['home_team']
        away_team = game['away_team']
        season = game['season']
        week = game['week']
        
        # Get team stats up to the current week
        team_stats = self.calculate_team_stats(week)
        
        home_stats = team_stats[team_stats['team'] == home_team]
        away_stats = team_stats[team_stats['team'] == away_team]
        
        if len(home_stats) == 0 or len(away_stats) == 0:
            return None
        
        return home_stats.iloc[0].to_dict(), away_stats.iloc[0].to_dict()
    
    def create_feature_vector(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """
        Create feature vector from home and away team stats.
        
        Args:
            home_stats: Home team statistics dictionary
            away_stats: Away team statistics dictionary
            
        Returns:
            Dictionary of features
        """
        return {
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
            'points_for_diff': home_stats['points_for_avg'] - away_stats['points_for_avg'],
            'points_against_diff': home_stats['points_against_avg'] - away_stats['points_against_avg'],
            'point_differential_diff': home_stats['point_differential'] - away_stats['point_differential'],
            'completion_pct_diff': home_stats['completion_pct'] - away_stats['completion_pct'],
            'yards_per_pass_diff': home_stats['yards_per_pass'] - away_stats['yards_per_pass'],
            'yards_per_rush_diff': home_stats['yards_per_rush'] - away_stats['yards_per_rush'],
            'turnover_rate_diff': home_stats['turnover_rate'] - away_stats['turnover_rate'],
            'def_completion_pct_diff': home_stats['def_completion_pct'] - away_stats['def_completion_pct'],
            'def_yards_per_pass_diff': home_stats['def_yards_per_pass'] - away_stats['def_yards_per_pass'],
            'def_yards_per_rush_diff': home_stats['def_yards_per_rush'] - away_stats['def_yards_per_rush'],
            'def_turnover_rate_diff': home_stats['def_turnover_rate'] - away_stats['def_turnover_rate'],
        }
    
    def train_model(self):
        """Train the logistic regression model."""
        print("Training logistic regression model...")
        
        X, y, point_spreads = self.prepare_training_data()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Store point spreads for spread prediction
        self.point_spreads = point_spreads
        
        print("Model training completed!")
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nFeature Importance (by coefficient magnitude):")
        print(feature_importance)
        
        return X_scaled, y
    
    def prepare_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare test data using late 2024 games.
        
        Returns:
            Tuple of (features, target, point_spreads)
        """
        print("Preparing test data...")
        
        # Get late 2024 games (starting from test_start_week)
        test_games = self.games_data[
            (self.games_data['season'] == self.test_season) & 
            (self.games_data['week'] >= self.test_start_week) &
            (self.games_data['game_type'] == 'REG')
        ].copy()
        
        test_data = []
        
        for _, game in test_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Calculate team stats up to the current week
            team_stats = self.calculate_team_stats_for_game(game)
            
            if team_stats is None:
                continue
            
            home_stats, away_stats = team_stats
            
            # Create feature vector (home team stats - away team stats)
            features = self.create_feature_vector(home_stats, away_stats)
            
            # Target: 1 if home team wins, 0 if away team wins
            home_score = game['home_score']
            away_score = game['away_score']
            target = 1 if home_score > away_score else 0
            
            # Add point spread
            features['point_spread'] = home_score - away_score
            
            test_data.append({**features, 'target': target, 'game_id': game['game_id']})
        
        test_df = pd.DataFrame(test_data)
        
        if len(test_df) == 0:
            raise ValueError("No test data available")
        
        # Separate features and target
        X = test_df[self.feature_columns]
        y = test_df['target']
        
        print(f"Test data shape: {X.shape}")
        
        return X, y, test_df['point_spread']
    
    def predict_game(self, home_team: str, away_team: str, week: int = 5) -> Dict:
        """
        Predict the outcome of a specific game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            week: Week to use for team stats (default 5)
            
        Returns:
            Dictionary with prediction results
        """
        # Get team stats
        team_stats = self.calculate_team_stats(week)
        
        home_stats = team_stats[team_stats['team'] == home_team]
        away_stats = team_stats[team_stats['team'] == away_team]
        
        if len(home_stats) == 0 or len(away_stats) == 0:
            raise ValueError(f"Team stats not found for {home_team} or {away_team}")
        
        home_stats = home_stats.iloc[0]
        away_stats = away_stats.iloc[0]
        
        # Create feature vector
        features = {
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
            'points_for_diff': home_stats['points_for_avg'] - away_stats['points_for_avg'],
            'points_against_diff': home_stats['points_against_avg'] - away_stats['points_against_avg'],
            'point_differential_diff': home_stats['point_differential'] - away_stats['point_differential'],
            'completion_pct_diff': home_stats['completion_pct'] - away_stats['completion_pct'],
            'yards_per_pass_diff': home_stats['yards_per_pass'] - away_stats['yards_per_pass'],
            'yards_per_rush_diff': home_stats['yards_per_rush'] - away_stats['yards_per_rush'],
            'turnover_rate_diff': home_stats['turnover_rate'] - away_stats['turnover_rate'],
            'def_completion_pct_diff': home_stats['def_completion_pct'] - away_stats['def_completion_pct'],
            'def_yards_per_pass_diff': home_stats['def_yards_per_pass'] - away_stats['def_yards_per_pass'],
            'def_yards_per_rush_diff': home_stats['def_yards_per_rush'] - away_stats['def_yards_per_rush'],
            'def_turnover_rate_diff': home_stats['def_turnover_rate'] - away_stats['def_turnover_rate'],
        }
        
        # Convert to DataFrame and scale
        X = pd.DataFrame([features])[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        win_probability = self.model.predict_proba(X_scaled)[0][1]
        predicted_winner = home_team if win_probability > 0.5 else away_team
        
        # Estimate point spread (simplified approach)
        # Use historical point spreads and win probability to estimate
        avg_spread = np.mean(self.point_spreads)
        estimated_spread = avg_spread * (win_probability - 0.5) * 2
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'home_win_probability': win_probability,
            'away_win_probability': 1 - win_probability,
            'estimated_point_spread': estimated_spread,
            'confidence': abs(win_probability - 0.5) * 2  # 0 to 1 scale
        }
    
    def evaluate_model(self):
        """Evaluate the model performance on test data."""
        print("Evaluating model performance on test data...")
        
        X, y, _ = self.prepare_test_data()
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        
        print(f"\nModel Performance on Test Data:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Away Win', 'Home Win']))
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Away Win', 'Home Win'],
                   yticklabels=['Away Win', 'Home Win'])
        plt.title('Confusion Matrix - Test Data')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        return accuracy
    
    def predict_late_season_games(self, season: int = None, week: int = None):
        """Predict games for a specific week in a season."""
        if season is None:
            season = self.test_season
        if week is None:
            week = self.test_start_week
            
        print(f"Predicting Week {week} games for {season}...")
        
        # Get games for the specified week
        target_games = self.games_data[
            (self.games_data['week'] == week) & 
            (self.games_data['season'] == season) &
            (self.games_data['game_type'] == 'REG')
        ]
        
        predictions = []
        
        for _, game in target_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            try:
                prediction = self.predict_game(home_team, away_team, week)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting {away_team} @ {home_team}: {e}")
        
        return predictions

def main():
    """Main function to demonstrate the NFL Game Predictor."""
    # Initialize predictor with 2023 training data and 2024 test data (starting week 10)
    predictor = NFLGamePredictor(train_seasons=[2023], test_season=2024, test_start_week=10)
    
    # Load data
    predictor.load_data()
    
    # Train model
    predictor.train_model()
    
    # Evaluate model on test data
    accuracy = predictor.evaluate_model()
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # You can change these teams to any valid NFL team abbreviations
    try:
        example_prediction = predictor.predict_game('KC', 'BUF', week=10)
        
        print(f"\nPrediction for {example_prediction['away_team']} @ {example_prediction['home_team']}:")
        print(f"Predicted Winner: {example_prediction['predicted_winner']}")
        print(f"Home Win Probability: {example_prediction['home_win_probability']:.3f}")
        print(f"Away Win Probability: {example_prediction['away_win_probability']:.3f}")
        print(f"Estimated Point Spread: {example_prediction['estimated_point_spread']:.1f}")
        print(f"Confidence: {example_prediction['confidence']:.3f}")
    except Exception as e:
        print(f"Error making example prediction: {e}")
    
    # Predict late season games for 2024
    print("\n" + "="*50)
    print("LATE SEASON PREDICTIONS FOR 2024")
    print("="*50)
    
    try:
        late_season_predictions = predictor.predict_late_season_games(2024, 10)
        
        if late_season_predictions:
            print(f"\nFound {len(late_season_predictions)} Week 10 games:")
            for i, pred in enumerate(late_season_predictions, 1):
                print(f"\nGame {i}: {pred['away_team']} @ {pred['home_team']}")
                print(f"  Predicted Winner: {pred['predicted_winner']}")
                print(f"  Confidence: {pred['confidence']:.3f}")
                print(f"  Estimated Spread: {pred['estimated_point_spread']:.1f}")
        else:
            print("No Week 10 games found for 2024")
    except Exception as e:
        print(f"Error predicting late season games: {e}")

if __name__ == "__main__":
    main()
