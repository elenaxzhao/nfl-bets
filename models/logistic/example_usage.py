"""
Example usage of the NFL Game Predictor

This script demonstrates how to use the NFLGamePredictor class to:
1. Train a model on the first 5 games of the season
2. Predict the outcome of the 6th game
3. Estimate point spreads
"""

from logistic_regression import NFLGamePredictor

def main():
    print("NFL Game Predictor - Example Usage")
    print("=" * 40)
    
    # Initialize the predictor with recent seasons
    predictor = NFLGamePredictor(seasons=[2020, 2021, 2022, 2023, 2024])
    
    # Load NFL data
    print("\n1. Loading NFL data...")
    predictor.load_data()
    
    # Train the model using first 5 weeks
    print("\n2. Training model on first 5 weeks...")
    predictor.train_model(week=5)
    
    # Evaluate model performance
    print("\n3. Evaluating model performance...")
    accuracy = predictor.evaluate_model()
    
    # Make predictions for specific games
    print("\n4. Making predictions for specific games...")
    
    # Example 1: Chiefs vs Bills
    print("\nExample 1: Kansas City Chiefs vs Buffalo Bills")
    try:
        prediction = predictor.predict_game('KC', 'BUF', week=5)
        print(f"  {prediction['away_team']} @ {prediction['home_team']}")
        print(f"  Predicted Winner: {prediction['predicted_winner']}")
        print(f"  Home Win Probability: {prediction['home_win_probability']:.3f}")
        print(f"  Away Win Probability: {prediction['away_win_probability']:.3f}")
        print(f"  Estimated Point Spread: {prediction['estimated_point_spread']:.1f}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Example 2: Different teams
    print("\nExample 2: Dallas Cowboys vs Philadelphia Eagles")
    try:
        prediction = predictor.predict_game('DAL', 'PHI', week=5)
        print(f"  {prediction['away_team']} @ {prediction['home_team']}")
        print(f"  Predicted Winner: {prediction['predicted_winner']}")
        print(f"  Home Win Probability: {prediction['home_win_probability']:.3f}")
        print(f"  Away Win Probability: {prediction['away_win_probability']:.3f}")
        print(f"  Estimated Point Spread: {prediction['estimated_point_spread']:.1f}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Predict all Week 6 games for 2024
    print("\n5. Predicting all Week 6 games for 2024...")
    try:
        week_6_predictions = predictor.predict_week_6_games(2024)
        
        if week_6_predictions:
            print(f"\nFound {len(week_6_predictions)} Week 6 games:")
            for i, pred in enumerate(week_6_predictions, 1):
                print(f"\nGame {i}: {pred['away_team']} @ {pred['home_team']}")
                print(f"  Predicted Winner: {pred['predicted_winner']}")
                print(f"  Confidence: {pred['confidence']:.3f}")
                print(f"  Estimated Spread: {pred['estimated_point_spread']:.1f}")
        else:
            print("No Week 6 games found for 2024")
    except Exception as e:
        print(f"Error predicting Week 6 games: {e}")
    
    print("\n" + "=" * 40)
    print("Example completed!")
    print("\nTo use this model with your own teams:")
    print("1. Use valid NFL team abbreviations (e.g., 'KC', 'BUF', 'DAL', 'PHI')")
    print("2. Call predictor.predict_game(home_team, away_team, week=5)")
    print("3. The model uses stats from the first 5 weeks to predict Week 6 outcomes")

if __name__ == "__main__":
    main()
