"""
Simple NFL Prediction Tool - Easy to Use
"""

from nfl_bayesian_network import NFLBayesianNetwork, GameFeatures

def predict_game():
    """Simple function to predict a game"""
    
    # Load the trained model
    print("Loading Bayesian Network model...")
    try:
        bn = NFLBayesianNetwork()
        bn.load_model('nfl_bayesian_model.pkl')
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Training new model...")
        bn = NFLBayesianNetwork()
        bn.train()
        bn.save_model('nfl_bayesian_model.pkl')
        print("✓ Model trained and saved!")
    
    print("\n" + "="*60)
    print("NFL GAME PREDICTOR")
    print("="*60)
    
    # Get game information
    print("\nEnter game details:")
    home_team = input("Home team (e.g., KC): ").upper().strip()
    away_team = input("Away team (e.g., BUF): ").upper().strip()
    week = int(input("Week number: "))
    
    print(f"\nInjury Report for {home_team} vs {away_team}:")
    home_injuries = int(input(f"How many key injuries (QB/RB/WR/TE) does {home_team} have? "))
    away_injuries = int(input(f"How many key injuries (QB/RB/WR/TE) does {away_team} have? "))
    
    print(f"\nWeather Conditions:")
    wind = float(input("Wind speed (mph, or press Enter for 5): ") or "5")
    temp = float(input("Temperature (F, or press Enter for 70): ") or "70")
    
    print(f"\nGame Context:")
    divisional = input("Divisional game? (y/n): ").lower().startswith('y')
    home_rest = int(input(f"{home_team} rest days (or press Enter for 7): ") or "7")
    away_rest = int(input(f"{away_team} rest days (or press Enter for 7): ") or "7")
    
    # Create prediction
    features = GameFeatures(
        home_key_injuries=home_injuries,
        away_key_injuries=away_injuries,
        home_total_injuries=home_injuries * 2,
        away_total_injuries=away_injuries * 2,
        wind_speed=wind,
        temperature=temp,
        week=week,
        is_divisional=divisional,
        home_rest_days=home_rest,
        away_rest_days=away_rest
    )
    
    # Make prediction
    prediction = bn.predict(features)
    
    # Display results
    print(f"\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"{away_team} @ {home_team} (Week {week})")
    print(f"")
    print(f"🏠 Home Team Win Probability: {prediction.home_win_probability:.1%}")
    print(f"✈️  Away Team Win Probability: {prediction.away_win_probability:.1%}")
    print(f"")
    print(f"Confidence Level: {prediction.confidence.upper()}")
    print(f"")
    
    if prediction.key_factors:
        print("Key Factors:")
        for i, factor in enumerate(prediction.key_factors, 1):
            print(f"  {i}. {factor}")
        print("")
    
    print("💰 Betting Recommendation:")
    print(f"   {prediction.betting_recommendation}")
    print("="*60)

if __name__ == "__main__":
    predict_game()
