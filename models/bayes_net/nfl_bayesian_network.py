"""
NFL Bayesian Network for Win Prediction
Uses injury data, weather, and temporal features to predict game outcomes
"""

import nflreadpy as nfl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from enum import Enum
import pickle
import os

warnings.filterwarnings('ignore')

class TeamStrength(Enum):
    """Team strength categories"""
    WEAK = "weak"
    AVERAGE = "average"
    STRONG = "strong"

class InjuryLevel(Enum):
    """Injury severity levels"""
    LOW = "low"      # 0-2 key injuries
    MEDIUM = "medium"  # 3-4 key injuries
    HIGH = "high"    # 5+ key injuries

class WeatherCondition(Enum):
    """Weather condition categories"""
    GOOD = "good"      # Low wind, moderate temp
    MODERATE = "moderate"  # Some wind or extreme temp
    POOR = "poor"      # High wind or very extreme temp

class SeasonPhase(Enum):
    """Season phase categories"""
    EARLY = "early"    # Weeks 1-4
    MID = "mid"       # Weeks 5-12
    LATE = "late"     # Weeks 13-18

@dataclass
class GameFeatures:
    """Features for a single game"""
    home_key_injuries: int
    away_key_injuries: int
    home_total_injuries: int
    away_total_injuries: int
    wind_speed: float
    temperature: float
    week: int
    is_divisional: bool
    home_rest_days: int
    away_rest_days: int

@dataclass
class PredictionResult:
    """Result of Bayesian network prediction"""
    home_win_probability: float
    away_win_probability: float
    confidence: str
    key_factors: List[str]
    betting_recommendation: str

class NFLBayesianNetwork:
    """
    Bayesian Network for NFL game outcome prediction
    """
    
    def __init__(self):
        self.network_structure = None
        self.conditional_probabilities = {}
        self.prior_probabilities = {}
        self.is_trained = False
        
    def load_training_data(self, seasons: List[int] = None) -> pd.DataFrame:
        """Load and preprocess training data"""
        if seasons is None:
            seasons = [2023, 2024]
            
        print(f"Loading training data for seasons: {seasons}")
        
        # Load schedules
        schedules = nfl.load_schedules(seasons).to_pandas()
        reg_games = schedules[schedules['game_type'] == 'REG'].copy()
        
        # Load injury data
        injuries = nfl.load_injuries(seasons).to_pandas()
        
        # Create features
        game_features = []
        
        for _, game in reg_games.iterrows():
            features = self._extract_game_features(game, injuries)
            if features:
                game_features.append(features)
        
        self.training_data = pd.DataFrame(game_features)
        print(f"Loaded {len(self.training_data)} games for training")
        
        return self.training_data
    
    def _extract_game_features(self, game: pd.Series, injuries: pd.DataFrame) -> Optional[Dict]:
        """Extract features for a single game"""
        try:
            game_id = game['game_id']
            season = game['season']
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Extract injury data
            week_injuries = injuries[
                (injuries['season'] == season) & 
                (injuries['week'] == week)
            ]
            
            home_injuries = week_injuries[week_injuries['team'] == home_team]
            away_injuries = week_injuries[week_injuries['team'] == away_team]
            
            # Count key position injuries (QB, RB, WR, TE)
            key_positions = ['QB', 'RB', 'WR', 'TE']
            
            home_key_injuries = len(home_injuries[home_injuries['position'].isin(key_positions)])
            away_key_injuries = len(away_injuries[away_injuries['position'].isin(key_positions)])
            
            home_total_injuries = len(home_injuries)
            away_total_injuries = len(away_injuries)
            
            # Weather data
            wind_speed = game.get('wind', 0) if pd.notna(game.get('wind')) else 0
            temperature = game.get('temp', 70) if pd.notna(game.get('temp')) else 70
            
            # Game context
            is_divisional = game.get('div_game', False)
            home_rest_days = game.get('home_rest', 7)
            away_rest_days = game.get('away_rest', 7)
            
            # Outcome
            home_win = 1 if game['home_score'] > game['away_score'] else 0
            
            return {
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'home_key_injuries': home_key_injuries,
                'away_key_injuries': away_key_injuries,
                'home_total_injuries': home_total_injuries,
                'away_total_injuries': away_total_injuries,
                'wind_speed': wind_speed,
                'temperature': temperature,
                'week': week,
                'is_divisional': is_divisional,
                'home_rest_days': home_rest_days,
                'away_rest_days': away_rest_days,
                'home_win': home_win
            }
            
        except Exception as e:
            print(f"Error extracting features for game {game.get('game_id', 'unknown')}: {e}")
            return None
    
    def _discretize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert continuous features to discrete categories"""
        df_discrete = df.copy()
        
        # Discretize injury levels
        df_discrete['home_injury_level'] = pd.cut(
            df['home_key_injuries'], 
            bins=[-1, 2, 4, 100], 
            labels=['low', 'medium', 'high']
        )
        df_discrete['away_injury_level'] = pd.cut(
            df['away_key_injuries'], 
            bins=[-1, 2, 4, 100], 
            labels=['low', 'medium', 'high']
        )
        
        # Discretize weather conditions
        df_discrete['weather_condition'] = pd.cut(
            df['wind_speed'], 
            bins=[-1, 5, 15, 100], 
            labels=['good', 'moderate', 'poor']
        )
        
        # Discretize season phase
        df_discrete['season_phase'] = pd.cut(
            df['week'], 
            bins=[0, 4, 12, 18], 
            labels=['early', 'mid', 'late']
        )
        
        # Discretize rest days
        df_discrete['home_rest_advantage'] = (df['home_rest_days'] > df['away_rest_days']).astype(int)
        
        return df_discrete
    
    def train(self, training_data: pd.DataFrame = None):
        """Train the Bayesian network"""
        if training_data is None:
            training_data = self.load_training_data()
        
        print("Training Bayesian Network...")
        
        # Discretize features
        df_discrete = self._discretize_features(training_data)
        
        # Define network structure (simplified for this example)
        # In a full implementation, you'd use structure learning algorithms
        self.network_structure = {
            'home_win': ['home_injury_level', 'away_injury_level', 'weather_condition', 
                        'season_phase', 'is_divisional', 'home_rest_advantage'],
            'home_injury_level': [],
            'away_injury_level': [],
            'weather_condition': [],
            'season_phase': [],
            'is_divisional': [],
            'home_rest_advantage': []
        }
        
        # Learn conditional probabilities
        self._learn_conditional_probabilities(df_discrete)
        
        # Learn prior probabilities
        self._learn_prior_probabilities(df_discrete)
        
        self.is_trained = True
        print("Training complete!")
    
    def _learn_prior_probabilities(self, df: pd.DataFrame):
        """Learn prior probabilities for root nodes"""
        print("Learning prior probabilities...")
        
        for node in ['home_injury_level', 'away_injury_level', 'weather_condition', 
                    'season_phase', 'is_divisional', 'home_rest_advantage']:
            if node in df.columns:
                self.prior_probabilities[node] = df[node].value_counts(normalize=True).to_dict()
    
    def _learn_conditional_probabilities(self, df: pd.DataFrame):
        """Learn conditional probabilities for child nodes"""
        print("Learning conditional probabilities...")
        
        # Learn P(home_win | parents)
        parents = self.network_structure['home_win']
        
        # Create all combinations of parent values
        parent_combinations = []
        for parent in parents:
            if parent in df.columns:
                parent_combinations.append(df[parent].unique())
        
        if parent_combinations:
            import itertools
            all_combinations = list(itertools.product(*parent_combinations))
            
            for combo in all_combinations:
                # Create mask for this combination
                mask = pd.Series([True] * len(df))
                for i, parent in enumerate(parents):
                    if parent in df.columns:
                        mask &= (df[parent] == combo[i])
                
                if mask.sum() > 0:  # If we have data for this combination
                    combo_key = tuple(combo)
                    self.conditional_probabilities[combo_key] = {
                        1: (df[mask]['home_win'] == 1).mean(),  # P(home_win=1 | combo)
                        0: (df[mask]['home_win'] == 0).mean()   # P(home_win=0 | combo)
                    }
    
    def predict(self, features: GameFeatures) -> PredictionResult:
        """Make a prediction for a game"""
        if not self.is_trained:
            raise ValueError("Network must be trained before making predictions")
        
        # Discretize input features
        home_injury_level = self._discretize_injury_level(features.home_key_injuries)
        away_injury_level = self._discretize_injury_level(features.away_key_injuries)
        weather_condition = self._discretize_weather(features.wind_speed)
        season_phase = self._discretize_season_phase(features.week)
        home_rest_advantage = 1 if features.home_rest_days > features.away_rest_days else 0
        
        # Find matching probability
        combo_key = (home_injury_level, away_injury_level, weather_condition, 
                    season_phase, features.is_divisional, home_rest_advantage)
        
        if combo_key in self.conditional_probabilities:
            probs = self.conditional_probabilities[combo_key]
            home_win_prob = probs[1]
            away_win_prob = probs[0]
        else:
            # Use prior probabilities if exact combination not found
            home_win_prob = 0.5  # Default to 50-50
            away_win_prob = 0.5
        
        # Determine confidence and key factors
        confidence, key_factors = self._analyze_prediction(features, home_win_prob)
        
        # Generate betting recommendation
        betting_rec = self._generate_betting_recommendation(home_win_prob, features)
        
        return PredictionResult(
            home_win_probability=home_win_prob,
            away_win_probability=away_win_prob,
            confidence=confidence,
            key_factors=key_factors,
            betting_recommendation=betting_rec
        )
    
    def _discretize_injury_level(self, injury_count: int) -> str:
        """Convert injury count to level"""
        if injury_count <= 2:
            return 'low'
        elif injury_count <= 4:
            return 'medium'
        else:
            return 'high'
    
    def _discretize_weather(self, wind_speed: float) -> str:
        """Convert wind speed to condition"""
        if wind_speed <= 5:
            return 'good'
        elif wind_speed <= 15:
            return 'moderate'
        else:
            return 'poor'
    
    def _discretize_season_phase(self, week: int) -> str:
        """Convert week to season phase"""
        if week <= 4:
            return 'early'
        elif week <= 12:
            return 'mid'
        else:
            return 'late'
    
    def _analyze_prediction(self, features: GameFeatures, home_win_prob: float) -> Tuple[str, List[str]]:
        """Analyze the prediction to determine confidence and key factors"""
        confidence = "low"
        key_factors = []
        
        # Check injury advantage
        injury_diff = features.away_key_injuries - features.home_key_injuries
        if injury_diff > 2:
            key_factors.append(f"Away team has {injury_diff} more key injuries")
            confidence = "medium" if confidence == "low" else "high"
        elif injury_diff < -2:
            key_factors.append(f"Home team has {-injury_diff} more key injuries")
            confidence = "medium" if confidence == "low" else "high"
        
        # Check weather impact
        if features.wind_speed > 15:
            key_factors.append(f"High wind ({features.wind_speed} mph) affects passing game")
            confidence = "medium" if confidence == "low" else "high"
        
        # Check rest advantage
        if features.home_rest_days > features.away_rest_days + 2:
            key_factors.append(f"Home team has significant rest advantage")
            confidence = "medium" if confidence == "low" else "high"
        
        # Check divisional game
        if features.is_divisional:
            key_factors.append("Divisional rivalry game")
        
        # Check prediction strength
        if abs(home_win_prob - 0.5) > 0.3:
            confidence = "high"
        
        return confidence, key_factors
    
    def _generate_betting_recommendation(self, home_win_prob: float, features: GameFeatures) -> str:
        """Generate betting recommendation based on prediction"""
        if home_win_prob > 0.65:
            return "STRONG HOME BET - High confidence in home team win"
        elif home_win_prob > 0.55:
            return "MODERATE HOME BET - Slight edge to home team"
        elif home_win_prob < 0.35:
            return "STRONG AWAY BET - High confidence in away team win"
        elif home_win_prob < 0.45:
            return "MODERATE AWAY BET - Slight edge to away team"
        else:
            return "NO CLEAR EDGE - Avoid betting on this game"
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'network_structure': self.network_structure,
            'conditional_probabilities': self.conditional_probabilities,
            'prior_probabilities': self.prior_probabilities,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.network_structure = model_data['network_structure']
        self.conditional_probabilities = model_data['conditional_probabilities']
        self.prior_probabilities = model_data['prior_probabilities']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def evaluate_model(self, test_data: pd.DataFrame = None) -> Dict:
        """Evaluate model performance"""
        if test_data is None:
            # Use last 20% of training data as test
            test_data = self.training_data.tail(int(len(self.training_data) * 0.2))
        
        correct_predictions = 0
        total_predictions = 0
        
        for _, game in test_data.iterrows():
            features = GameFeatures(
                home_key_injuries=game['home_key_injuries'],
                away_key_injuries=game['away_key_injuries'],
                home_total_injuries=game['home_total_injuries'],
                away_total_injuries=game['away_total_injuries'],
                wind_speed=game['wind_speed'],
                temperature=game['temperature'],
                week=game['week'],
                is_divisional=game['is_divisional'],
                home_rest_days=game['home_rest_days'],
                away_rest_days=game['away_rest_days']
            )
            
            prediction = self.predict(features)
            actual_winner = "home" if game['home_win'] == 1 else "away"
            predicted_winner = "home" if prediction.home_win_probability > 0.5 else "away"
            
            if actual_winner == predicted_winner:
                correct_predictions += 1
            
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }

def main():
    """Example usage of the Bayesian Network"""
    print("NFL Bayesian Network Example")
    print("=" * 40)
    
    # Create and train the network
    bn = NFLBayesianNetwork()
    bn.train()
    
    # Example prediction
    example_game = GameFeatures(
        home_key_injuries=1,      # Home team has 1 key injury
        away_key_injuries=4,      # Away team has 4 key injuries (significant disadvantage)
        home_total_injuries=5,
        away_total_injuries=12,
        wind_speed=8,             # Moderate wind
        temperature=65,           # Good temperature
        week=8,                   # Mid-season
        is_divisional=False,      # Not a divisional game
        home_rest_days=10,        # Home team has 10 days rest
        away_rest_days=7          # Away team has 7 days rest
    )
    
    prediction = bn.predict(example_game)
    
    print(f"\nGame Prediction:")
    print(f"Home Win Probability: {prediction.home_win_probability:.3f}")
    print(f"Away Win Probability: {prediction.away_win_probability:.3f}")
    print(f"Confidence: {prediction.confidence}")
    print(f"Key Factors: {', '.join(prediction.key_factors)}")
    print(f"Betting Recommendation: {prediction.betting_recommendation}")
    
    # Evaluate model
    evaluation = bn.evaluate_model()
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {evaluation['accuracy']:.3f}")
    print(f"Correct Predictions: {evaluation['correct_predictions']}/{evaluation['total_predictions']}")
    
    # Save model
    bn.save_model('nfl_bayesian_model.pkl')
    
    return bn

if __name__ == "__main__":
    model = main()
