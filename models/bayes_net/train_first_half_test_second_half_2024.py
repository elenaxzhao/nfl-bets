"""
Train Bayesian Network on 2023 data and test on 2024 data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nfl_bayesian_network import NFLBayesianNetwork, GameFeatures, PredictionResult
import pandas as pd
import numpy as np

def train_first_half_test_second_half_2024():
    """Train on first half of 2024 data and test on second half of 2024 data"""
    print("NFL Bayesian Network - Train First Half 2024, Test Second Half 2024")
    print("=" * 60)
    
    # Create network instance
    bn = NFLBayesianNetwork()
    
    # Load 2024 data
    print("\n1. Loading 2024 season data...")
    full_2024_data = bn.load_training_data(seasons=[2024])
    print(f"Loaded {len(full_2024_data)} total games from 2024")
    
    # Split into first half (Weeks 1-9) and second half (Weeks 10-18)
    training_data = full_2024_data[full_2024_data['week'] <= 9].copy()
    test_data = full_2024_data[full_2024_data['week'] >= 10].copy()
    
    print(f"Training data (Weeks 1-9): {len(training_data)} games")
    print(f"Test data (Weeks 10-18): {len(test_data)} games")
    
    # Train the model
    print("\n2. Training Bayesian Network on first half of 2024...")
    bn.train(training_data)
    
    # Test the model on second half of 2024 data
    print("\n3. Testing model on second half of 2024...")
    evaluation = bn.evaluate_model(test_data)
    
    print(f"\n" + "="*60)
    print("RESULTS - FIRST HALF 2024 TRAINED, SECOND HALF 2024 TESTED")
    print("="*60)
    print(f"Training Data (Weeks 1-9, 2024): {len(training_data)} games")
    print(f"Test Data (Weeks 10-18, 2024): {len(test_data)} games")
    print(f"Accuracy on second half 2024: {evaluation['accuracy']:.3f} ({evaluation['accuracy']*100:.1f}%)")
    print(f"Correct Predictions: {evaluation['correct_predictions']}/{evaluation['total_predictions']}")
    
    # Show some example predictions
    print(f"\n4. Example second half 2024 predictions:")
    print("-" * 40)
    
    # Show first 5 test games (from second half)
    for i, (_, game) in enumerate(test_data.head(5).iterrows()):
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
        
        prediction = bn.predict(features)
        actual_winner = "HOME" if game['home_win'] == 1 else "AWAY"
        predicted_winner = "HOME" if prediction.home_win_probability > 0.5 else "AWAY"
        correct = "✓" if actual_winner == predicted_winner else "✗"
        
        print(f"Game {i+1}: {game['away_team']} @ {game['home_team']} (Week {game['week']})")
        print(f"  Predicted: {predicted_winner} ({prediction.home_win_probability:.3f})")
        print(f"  Actual:    {actual_winner} {correct}")
        print(f"  Key Factors: {', '.join(prediction.key_factors[:2])}")  # Show top 2 factors
        print()
    
    # Save the trained model
    print("\n5. Saving trained model...")
    bn.save_model('nfl_bayesian_model_first_half_2024.pkl')
    print("Model saved as 'nfl_bayesian_model_first_half_2024.pkl'")
    
    # Additional analysis
    print(f"\n6. Additional Analysis:")
    print("-" * 30)
    
    # Analyze by week
    test_data['predicted_home_win'] = test_data.apply(lambda x: 
        bn.predict(GameFeatures(
            home_key_injuries=x['home_key_injuries'],
            away_key_injuries=x['away_key_injuries'],
            home_total_injuries=x['home_total_injuries'],
            away_total_injuries=x['away_total_injuries'],
            wind_speed=x['wind_speed'],
            temperature=x['temperature'],
            week=x['week'],
            is_divisional=x['is_divisional'],
            home_rest_days=x['home_rest_days'],
            away_rest_days=x['away_rest_days']
        )).home_win_probability > 0.5, axis=1)
    
    test_data['correct'] = (test_data['home_win'] == test_data['predicted_home_win'])
    
    # Accuracy by week
    weekly_accuracy = test_data.groupby('week')['correct'].mean().sort_index()
    print("Accuracy by Week:")
    for week, acc in weekly_accuracy.items():
        print(f"  Week {week:2d}: {acc:.3f} ({acc*100:.1f}%)")
    
    # Overall home team performance
    home_win_rate = test_data['home_win'].mean()
    training_home_win_rate = training_data['home_win'].mean()
    print(f"\nTraining Data (Weeks 1-9) Home Win Rate: {training_home_win_rate:.3f} ({training_home_win_rate*100:.1f}%)")
    print(f"Test Data (Weeks 10-18) Home Win Rate: {home_win_rate:.3f} ({home_win_rate*100:.1f}%)")
    
    # Compare training vs test performance
    training_eval = bn.evaluate_model(training_data)
    print(f"\nTraining Accuracy (Weeks 1-9): {training_eval['accuracy']:.3f} ({training_eval['accuracy']*100:.1f}%)")
    print(f"Test Accuracy (Weeks 10-18): {evaluation['accuracy']:.3f} ({evaluation['accuracy']*100:.1f}%)")
    print(f"Overfitting Check: {training_eval['accuracy'] - evaluation['accuracy']:.3f} difference")
    
    return bn, evaluation

if __name__ == "__main__":
    model, results = train_first_half_test_second_half_2024()
