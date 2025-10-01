"""
Simple Interface for NFL Bayesian Network Predictions
"""

from nfl_bayesian_network import NFLBayesianNetwork, GameFeatures
import pandas as pd

def predict_upcoming_games(season: int = 2024, week: int = None):
    """Predict outcomes for upcoming games"""
    
    # Load the trained model
    try:
        bn = NFLBayesianNetwork()
        bn.load_model('nfl_bayesian_model.pkl')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("Model not found. Training new model...")
        bn = NFLBayesianNetwork()
        bn.train()
        bn.save_model('nfl_bayesian_model.pkl')
    
    # Load current season schedules
    print(f"Loading {season} season data...")
    schedules = nfl.load_schedules([season]).to_pandas()
    
    # Filter for the specified week or upcoming games
    if week:
        games = schedules[schedules['week'] == week].copy()
    else:
        # Get upcoming games (games without scores)
        games = schedules[
            (schedules['home_score'].isna()) | 
            (schedules['away_score'].isna())
        ].copy()
    
    # Load injury data for current season
    injuries = nfl.load_injuries([season]).to_pandas()
    
    print(f"\nPredicting {len(games)} games...")
    print("=" * 80)
    
    predictions = []
    
    for _, game in games.iterrows():
        try:
            # Extract features for this game
            features = bn._extract_game_features(game, injuries)
            if not features:
                continue
            
            # Create GameFeatures object
            game_features = GameFeatures(
                home_key_injuries=features['home_key_injuries'],
                away_key_injuries=features['away_key_injuries'],
                home_total_injuries=features['home_total_injuries'],
                away_total_injuries=features['away_total_injuries'],
                wind_speed=features['wind_speed'],
                temperature=features['temperature'],
                week=features['week'],
                is_divisional=features['is_divisional'],
                home_rest_days=features['home_rest_days'],
                away_rest_days=features['away_rest_days']
            )
            
            # Make prediction
            prediction = bn.predict(game_features)
            
            # Store results
            predictions.append({
                'game_id': features['game_id'],
                'home_team': features['home_team'],
                'away_team': features['away_team'],
                'week': features['week'],
                'home_win_prob': prediction.home_win_probability,
                'away_win_prob': prediction.away_win_probability,
                'confidence': prediction.confidence,
                'key_factors': '; '.join(prediction.key_factors),
                'betting_rec': prediction.betting_recommendation,
                'home_injuries': features['home_key_injuries'],
                'away_injuries': features['away_key_injuries'],
                'wind': features['wind_speed']
            })
            
            # Print prediction
            print(f"\n{features['away_team']} @ {features['home_team']} (Week {features['week']})")
            print(f"Home Win: {prediction.home_win_probability:.1%} | Away Win: {prediction.away_win_probability:.1%}")
            print(f"Confidence: {prediction.confidence}")
            if prediction.key_factors:
                print(f"Key Factors: {'; '.join(prediction.key_factors)}")
            print(f"Betting: {prediction.betting_recommendation}")
            
        except Exception as e:
            print(f"Error predicting {game.get('home_team', '?')} vs {game.get('away_team', '?')}: {e}")
            continue
    
    # Create summary DataFrame
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        
        # Sort by betting value (games with clear edges)
        df_predictions['betting_value'] = df_predictions.apply(
            lambda row: abs(row['home_win_prob'] - 0.5) if row['confidence'] in ['medium', 'high'] else 0, 
            axis=1
        )
        df_predictions = df_predictions.sort_values('betting_value', ascending=False)
        
        print(f"\n" + "=" * 80)
        print("BETTING RECOMMENDATIONS SUMMARY")
        print("=" * 80)
        
        for _, pred in df_predictions.iterrows():
            if pred['confidence'] in ['medium', 'high'] and pred['betting_value'] > 0.1:
                print(f"\n🎯 {pred['away_team']} @ {pred['home_team']}")
                print(f"   Home: {pred['home_win_prob']:.1%} | Away: {pred['away_win_prob']:.1%}")
                print(f"   {pred['betting_rec']}")
        
        # Save predictions to CSV
        df_predictions.to_csv('nfl_predictions.csv', index=False)
        print(f"\nPredictions saved to 'nfl_predictions.csv'")
        
        return df_predictions
    
    return None

def predict_specific_game(home_team: str, away_team: str, week: int, season: int = 2024):
    """Predict a specific game with custom injury data"""
    
    # Load model
    bn = NFLBayesianNetwork()
    bn.load_model('nfl_bayesian_model.pkl')
    
    print(f"Predicting: {away_team} @ {home_team} (Week {week}, {season})")
    print("Enter injury information:")
    
    # Get user input for injuries
    home_key_injuries = int(input(f"How many key injuries (QB/RB/WR/TE) does {home_team} have? "))
    away_key_injuries = int(input(f"How many key injuries (QB/RB/WR/TE) does {away_team} have? "))
    
    # Get weather info
    wind_speed = float(input("Wind speed (mph): ") or "5")
    temperature = float(input("Temperature (F): ") or "70")
    
    # Get game context
    is_divisional = input("Divisional game? (y/n): ").lower().startswith('y')
    home_rest = int(input(f"{home_team} rest days: ") or "7")
    away_rest = int(input(f"{away_team} rest days: ") or "7")
    
    # Create features
    features = GameFeatures(
        home_key_injuries=home_key_injuries,
        away_key_injuries=away_key_injuries,
        home_total_injuries=home_key_injuries * 2,  # Estimate
        away_total_injuries=away_key_injuries * 2,  # Estimate
        wind_speed=wind_speed,
        temperature=temperature,
        week=week,
        is_divisional=is_divisional,
        home_rest_days=home_rest,
        away_rest_days=away_rest
    )
    
    # Make prediction
    prediction = bn.predict(features)
    
    print(f"\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Home Win Probability: {prediction.home_win_probability:.1%}")
    print(f"Away Win Probability: {prediction.away_win_probability:.1%}")
    print(f"Confidence Level: {prediction.confidence.upper()}")
    print(f"\nKey Factors:")
    for factor in prediction.key_factors:
        print(f"  • {factor}")
    print(f"\nBetting Recommendation:")
    print(f"  {prediction.betting_recommendation}")
    
    return prediction

if __name__ == "__main__":
    import nflreadpy as nfl
    
    print("NFL Bayesian Network Prediction Interface")
    print("=" * 50)
    
    choice = input("""
Choose an option:
1. Predict upcoming games
2. Predict specific game with custom data
3. Exit

Enter choice (1-3): """)
    
    if choice == "1":
        week = input("Enter week number (or press Enter for all upcoming): ")
        week = int(week) if week else None
        predict_upcoming_games(week=week)
    
    elif choice == "2":
        home_team = input("Home team (e.g., KC): ").upper()
        away_team = input("Away team (e.g., BUF): ").upper()
        week = int(input("Week number: "))
        predict_specific_game(home_team, away_team, week)
    
    else:
        print("Goodbye!")
