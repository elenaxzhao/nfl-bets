#!/usr/bin/env python3
"""
NFL Elastic-Net Logistic Regression - 2024 Season Test Script
==============================================================

Test the elastic-net logistic regression model on 2024 NFL season:
- Trains on 2022-2023 seasons
- Tests on 2024 season weeks 6-18
- Generates comprehensive performance metrics and visualizations

Usage:
    python s24_test.py

Author: NFL Bets Analysis
Date: October 2025
"""

import sys
import os

# Add parent directory to path to import the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logistic_regression import NFLElasticNetLogisticRegression


def main():
    """Main function to test the elastic-net logistic regression model."""
    
    print("="*70)
    print("NFL ELASTIC-NET LOGISTIC REGRESSION")
    print("2024 Season Test (Weeks 6-18)")
    print("="*70)
    
    # Initialize model with optimal parameters
    print("\n[1/4] Initializing model...")
    model = NFLElasticNetLogisticRegression(
        train_seasons=[2022, 2023],
        test_season=2024,
        l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv_folds=5,
        random_state=42,
        initial_elo=1500.0,
        elo_k=20.0,
        use_market_spread=False
    )
    
    print("✓ Model initialized")
    print(f"  - Training on: {model.train_seasons}")
    print(f"  - Testing on: {model.test_season} (weeks 6-18)")
    print(f"  - Regularization: Elastic-Net with CV")
    print(f"  - Features: Elo, EPA, Success Rate, Rest Days, and more")
    
    # Load data
    print("\n[2/4] Loading NFL data...")
    model.load_data()
    print("✓ Data loaded successfully")
    
    # Train model
    print("\n[3/4] Training model on 2022-2023 seasons...")
    model.train()
    print("✓ Model trained successfully")
    
    # Generate comprehensive report for weeks 6-18
    print("\n[4/4] Evaluating on 2024 weeks 6-18 and generating report...")
    model.generate_report(
        output_dir='results',
        start_week=6,
        end_week=18
    )
    
    # Display sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    if model.test_data is not None and len(model.test_data) > 0:
        # Show predictions for a few sample weeks
        sample_weeks = [6, 10, 14]
        
        for week in sample_weeks:
            week_games = model.test_data[model.test_data['week'] == week]
            
            if len(week_games) > 0:
                print(f"\n--- Week {week} Sample ---")
                
                # Get predictions for this week
                X_week = week_games[model.feature_columns].fillna(0)
                X_week_scaled = model.scaler.transform(X_week)
                probs = model.model.predict_proba(X_week_scaled)[:, 1]
                preds = model.model.predict(X_week_scaled)
                
                # Show first 3 games of the week
                for i, (idx, game) in enumerate(week_games.head(3).iterrows()):
                    home_team = game['home_team']
                    away_team = game['away_team']
                    home_score = game['home_score']
                    away_score = game['away_score']
                    home_prob = probs[i]
                    predicted = preds[i]
                    
                    actual_winner = home_team if home_score > away_score else away_team
                    predicted_winner = home_team if predicted == 1 else away_team
                    
                    correct = "✓" if actual_winner == predicted_winner else "✗"
                    
                    print(f"  {correct} {away_team} @ {home_team}")
                    print(f"     Predicted: {predicted_winner} ({home_prob:.1%} home)")
                    print(f"     Actual: {actual_winner} ({away_team} {away_score}, {home_team} {home_score})")
    
    # Final summary
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    if model.test_data is not None and len(model.test_data) > 0:
        # Calculate overall accuracy
        X_test = model.test_data[model.feature_columns].fillna(0)
        X_test_scaled = model.scaler.transform(X_test)
        y_pred = model.model.predict(X_test_scaled)
        y_true = model.test_data['target']
        
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        accuracy = correct / total
        
        print(f"\nOverall Test Performance (Weeks 6-18):")
        print(f"  - Games: {total}")
        print(f"  - Correct: {correct}")
        print(f"  - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"\nResults saved to: results/")
        print(f"  - elasticnet_2024_game_results.csv")
        print(f"  - elasticnet_2024_weekly_results.csv")
        print(f"  - elasticnet_2024_summary.csv")
        print(f"  - calibration_curve.png")
        print(f"  - weekly_performance.png")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

