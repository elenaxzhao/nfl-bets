#!/usr/bin/env python3
"""
NFL ELO Model Rolling Window Testing
====================================

This script implements a rolling window approach to test the Poisson ELO model:
- Train on weeks 1-5, predict week 6
- Train on weeks 1-6, predict week 7
- Continue until end of season
- Measure and compare model performance across different training periods

Author: NFL Bets Analysis
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the parent directory to path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poisson_elo_model import PoissonELOModel

class RollingWindowTester:
    """
    Class to handle rolling window training and testing of the ELO model
    """
    
    def __init__(self, years=[2024]):
        """
        Initialize the rolling window tester
        
        Parameters:
        -----------
        years : list
            Years to load data for
        """
        self.years = years
        self.results = []
        self.model = None
        
    def load_data(self):
        """Load NFL data"""
        print("Loading NFL data...")
        self.model = PoissonELOModel(initial_rating=1500, k_factor=32, home_advantage=25)
        self.model.load_nfl_data(self.years)
        self.model.calculate_team_stats()
        print("✓ Data loaded successfully")
    
    def get_available_weeks(self, game_types=['REG']):
        """
        Get available weeks in the dataset
        
        Parameters:
        -----------
        game_types : list
            Game types to include
            
        Returns:
        --------
        list : Available weeks sorted
        """
        completed_games = self.model.schedule_data[
            (self.model.schedule_data['home_score'].notna()) & 
            (self.model.schedule_data['away_score'].notna())
        ].copy()
        
        if game_types:
            completed_games = completed_games[completed_games['game_type'].isin(game_types)]
        
        weeks = sorted(completed_games['week'].unique())
        return weeks
    
    def get_week_games(self, week, game_types=['REG']):
        """
        Get games for a specific week
        
        Parameters:
        -----------
        week : int
            Week number
        game_types : list
            Game types to include
            
        Returns:
        --------
        pandas.DataFrame : Games for the week
        """
        week_games = self.model.schedule_data[
            (self.model.schedule_data['week'] == week) &
            (self.model.schedule_data['home_score'].notna()) & 
            (self.model.schedule_data['away_score'].notna())
        ].copy()
        
        if game_types:
            week_games = week_games[week_games['game_type'].isin(game_types)]
        
        return week_games
    
    def train_and_predict(self, train_weeks, test_week, game_types=['REG']):
        """
        Train model on specified weeks and predict test week
        
        Parameters:
        -----------
        train_weeks : list
            Weeks to train on
        test_week : int
            Week to predict
        game_types : list
            Game types to include
            
        Returns:
        --------
        dict : Results including accuracy and predictions
        """
        print(f"\n🔄 Training on weeks {train_weeks}, predicting week {test_week}")
        
        # Create a fresh model for this iteration
        model = PoissonELOModel(initial_rating=1500, k_factor=32, home_advantage=25)
        model.load_nfl_data(self.years)
        model.calculate_team_stats()
        
        # Train on specified weeks
        model.train_model(
            start_week=min(train_weeks), 
            end_week=max(train_weeks), 
            game_types=game_types
        )
        
        # Get test week games
        test_games = self.get_week_games(test_week, game_types)
        
        if len(test_games) == 0:
            print(f"No games found for week {test_week}")
            return None
        
        # Make predictions and evaluate
        predictions = []
        correct_predictions = 0
        total_predictions = len(test_games)
        
        for _, game in test_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            home_score = game['home_score']
            away_score = game['away_score']
            
            # Get prediction
            prediction = model.predict_game(home_team, away_team)
            predicted_winner = prediction['predicted_winner']
            
            # Actual winner
            actual_winner = home_team if home_score > away_score else away_team
            
            # Check if prediction is correct
            is_correct = predicted_winner == actual_winner
            
            if is_correct:
                correct_predictions += 1
            
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'predicted_winner': predicted_winner,
                'actual_winner': actual_winner,
                'home_win_prob': prediction['home_win_probability'],
                'away_win_prob': prediction['away_win_probability'],
                'confidence': prediction['confidence'],
                'correct': is_correct
            })
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        result = {
            'train_weeks': train_weeks,
            'test_week': test_week,
            'num_train_games': len(model.schedule_data[
                (model.schedule_data['week'].isin(train_weeks)) &
                (model.schedule_data['home_score'].notna()) & 
                (model.schedule_data['away_score'].notna()) &
                (model.schedule_data['game_type'].isin(game_types))
            ]),
            'num_test_games': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'predictions': predictions,
            'top_5_teams': model.get_team_rankings().head(5)[['team', 'elo_rating']].to_dict('records')
        }
        
        print(f"✓ Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        return result
    
    def run_rolling_window_test(self, start_week=1, min_train_weeks=5, game_types=['REG']):
        """
        Run rolling window test from start_week to end of season
        
        Parameters:
        -----------
        start_week : int
            Starting week for training
        min_train_weeks : int
            Minimum number of weeks to train on
        game_types : list
            Game types to include
        """
        print("🏈 Starting Rolling Window Test")
        print("=" * 50)
        
        # Get available weeks
        available_weeks = self.get_available_weeks(game_types)
        print(f"Available weeks: {available_weeks}")
        
        # Determine test range
        max_week = max(available_weeks)
        print(f"Testing from week {start_week + min_train_weeks} to week {max_week}")
        
        results = []
        
        # Rolling window: train on weeks 1-N, predict week N+1
        for test_week in range(start_week + min_train_weeks, max_week + 1):
            if test_week not in available_weeks:
                print(f"Week {test_week} not available, skipping...")
                continue
            
            # Train on all weeks from start_week to test_week-1
            train_weeks = list(range(start_week, test_week))
            
            result = self.train_and_predict(train_weeks, test_week, game_types)
            
            if result:
                results.append(result)
        
        self.results = results
        return results
    
    def analyze_results(self):
        """Analyze and display results"""
        if not self.results:
            print("No results to analyze!")
            return
        
        print("\n" + "=" * 50)
        print("ROLLING WINDOW TEST RESULTS")
        print("=" * 50)
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                'train_weeks': f"{r['train_weeks'][0]}-{r['train_weeks'][-1]}",
                'test_week': r['test_week'],
                'train_games': r['num_train_games'],
                'test_games': r['num_test_games'],
                'accuracy': r['accuracy'],
                'correct': r['correct_predictions']
            }
            for r in self.results
        ])
        
        print(results_df.to_string(index=False))
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"Average Accuracy: {results_df['accuracy'].mean():.3f}")
        print(f"Best Accuracy: {results_df['accuracy'].max():.3f} (Week {results_df.loc[results_df['accuracy'].idxmax(), 'test_week']})")
        print(f"Worst Accuracy: {results_df['accuracy'].min():.3f} (Week {results_df.loc[results_df['accuracy'].idxmin(), 'test_week']})")
        print(f"Standard Deviation: {results_df['accuracy'].std():.3f}")
        
        return results_df
    
    def visualize_results(self):
        """Create visualizations of the results"""
        if not self.results:
            print("No results to visualize!")
            return
        
        results_df = pd.DataFrame([
            {
                'test_week': r['test_week'],
                'train_games': r['num_train_games'],
                'accuracy': r['accuracy'],
                'train_weeks_count': len(r['train_weeks'])
            }
            for r in self.results
        ])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy over time
        ax1.plot(results_df['test_week'], results_df['accuracy'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Test Week')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy vs Test Week')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=results_df['accuracy'].mean(), color='r', linestyle='--', alpha=0.7, label=f'Average: {results_df["accuracy"].mean():.3f}')
        ax1.legend()
        
        # Training games vs accuracy
        ax2.scatter(results_df['train_games'], results_df['accuracy'], s=100, alpha=0.7)
        ax2.set_xlabel('Number of Training Games')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Data Size vs Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(results_df['train_games'], results_df['accuracy'], 1)
        p = np.poly1d(z)
        ax2.plot(results_df['train_games'], p(results_df['train_games']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.show()
    
    def show_detailed_predictions(self, week=None):
        """Show detailed predictions for a specific week or all weeks"""
        if not self.results:
            print("No results to show!")
            return
        
        if week:
            # Show specific week
            week_results = [r for r in self.results if r['test_week'] == week]
            if not week_results:
                print(f"No results found for week {week}")
                return
            results_to_show = week_results
        else:
            # Show all weeks
            results_to_show = self.results
        
        for result in results_to_show:
            print(f"\n{'='*60}")
            print(f"WEEK {result['test_week']} PREDICTIONS")
            print(f"Trained on weeks: {result['train_weeks'][0]}-{result['train_weeks'][-1]}")
            print(f"{'='*60}")
            
            for pred in result['predictions']:
                status = "✓" if pred['correct'] else "✗"
                print(f"{status} {pred['home_team']} vs {pred['away_team']}")
                print(f"   Predicted: {pred['predicted_winner']} ({pred['confidence']:.1%} confidence)")
                print(f"   Actual: {pred['actual_winner']} ({pred['home_score']}-{pred['away_score']})")
                print(f"   Probabilities: {pred['home_team']} {pred['home_win_prob']:.1%} vs {pred['away_team']} {pred['away_win_prob']:.1%}")
                print()

def main():
    """Main function to run the rolling window test"""
    print("🏈 NFL ELO Rolling Window Test")
    print("=" * 50)
    
    # Initialize tester
    tester = RollingWindowTester(years=[2024])
    
    # Load data
    tester.load_data()
    
    # Run rolling window test
    # Train on weeks 1-5, predict week 6, then weeks 1-6, predict week 7, etc.
    results = tester.run_rolling_window_test(
        start_week=1, 
        min_train_weeks=5, 
        game_types=['REG']  # Regular season only
    )
    
    # Analyze results
    results_df = tester.analyze_results()
    
    # Show detailed predictions for a few weeks
    print(f"\n{'='*60}")
    print("DETAILED PREDICTIONS (Sample Weeks)")
    print(f"{'='*60}")
    
    # Show predictions for weeks 6, 10, and 14 as examples
    for week in [6, 10, 14]:
        tester.show_detailed_predictions(week)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    tester.visualize_results()
    
    print(f"\n{'='*50}")
    print("Rolling window test complete!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
