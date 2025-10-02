"""
Train Bayesian Network on a decade of data (2013-2023) and test on 2024 season
This addresses the data insufficiency issues identified in previous experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nfl_bayesian_network import NFLBayesianNetwork, GameFeatures, PredictionResult

def train_decade_test_2024():
    """Train on 2013-2023 data and test on 2024 data"""
    print("NFL Bayesian Network - Train 2013-2023, Test 2024")
    print("=" * 60)
    
    # Create network instance
    bn = NFLBayesianNetwork()
    
    # Load training data (2013-2023)
    print("\n1. Loading training data (2013-2023)...")
    training_data = bn.load_training_data(seasons=list(range(2013, 2024)))
    print(f"Loaded {len(training_data)} games from 2013-2023")
    
    # Load test data (2024)
    print("\n2. Loading test data (2024)...")
    test_data = bn.load_training_data(seasons=[2024])
    print(f"Loaded {len(test_data)} games from 2024")
    
    # Train the model
    print("\n3. Training Bayesian Network on decade of data...")
    bn.train(training_data)
    
    # Test the model on 2024 data
    print("\n4. Testing model on 2024 season...")
    evaluation = bn.evaluate_model(test_data)
    
    print(f"\n" + "="*60)
    print("RESULTS - DECADE TRAINED (2013-2023), 2024 TESTED")
    print("="*60)
    print(f"Training Data (2013-2023): {len(training_data)} games")
    print(f"Test Data (2024): {len(test_data)} games")
    print(f"Accuracy on 2024: {evaluation['accuracy']:.3f} ({evaluation['accuracy']*100:.1f}%)")
    print(f"Correct Predictions: {evaluation['correct_predictions']}/{evaluation['total_predictions']}")
    
    # Show some example predictions
    print(f"\n5. Example 2024 predictions:")
    print("-" * 40)
    
    # Show first 5 test games
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
    print("6. Saving trained model...")
    bn.save_model('nfl_bayesian_model_decade_2013_2023.pkl')
    print("Model saved as 'nfl_bayesian_model_decade_2013_2023.pkl'")
    
    # Additional analysis
    print(f"\n7. Additional Analysis:")
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
    print("Accuracy by Week (2024):")
    for week, acc in weekly_accuracy.items():
        print(f"  Week {week:2d}: {acc:.3f} ({acc*100:.1f}%)")
    
    # Overall home team performance
    home_win_rate = test_data['home_win'].mean()
    training_home_win_rate = training_data['home_win'].mean()
    print(f"\nTraining Data (2013-2023) Home Win Rate: {training_home_win_rate:.3f} ({training_home_win_rate*100:.1f}%)")
    print(f"Test Data (2024) Home Win Rate: {home_win_rate:.3f} ({home_win_rate*100:.1f}%)")
    
    # Compare training vs test performance
    training_eval = bn.evaluate_model(training_data.tail(1000))  # Sample last 1000 games to avoid memory issues
    print(f"\nTraining Accuracy (Sample of 2013-2023): {training_eval['accuracy']:.3f} ({training_eval['accuracy']*100:.1f}%)")
    print(f"Test Accuracy (2024): {evaluation['accuracy']:.3f} ({evaluation['accuracy']*100:.1f}%)")
    print(f"Overfitting Check: {training_eval['accuracy'] - evaluation['accuracy']:.3f} difference")
    
    # Data quality analysis
    print(f"\n8. Data Quality Analysis:")
    print("-" * 30)
    print(f"Training data spans: {training_data['season'].min()} - {training_data['season'].max()}")
    print(f"Average games per season: {len(training_data) / (training_data['season'].max() - training_data['season'].min() + 1):.1f}")
    print(f"Training data injury coverage:")
    print(f"  Games with injury data: {(training_data['home_key_injuries'] + training_data['away_key_injuries'] > 0).mean():.3f}")
    print(f"  Average home key injuries: {training_data['home_key_injuries'].mean():.2f}")
    print(f"  Average away key injuries: {training_data['away_key_injuries'].mean():.2f}")
    
    return bn, evaluation

if __name__ == "__main__":
    model, results = train_decade_test_2024()
