#!/usr/bin/env python3
"""
Evaluation Metrics Module
==========================

Computes metrics for model evaluation:
- Accuracy, log loss, Brier score for win probability
- MAE, RMSE for scores
- Spread MAE
- Calibration curves
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Calculates various evaluation metrics.
    """
    
    @staticmethod
    def compute_all_metrics(
        y_true_win: np.ndarray,
        y_pred_win_prob: np.ndarray,
        y_true_home_score: np.ndarray,
        y_true_away_score: np.ndarray,
        y_pred_home_score: np.ndarray,
        y_pred_away_score: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            y_true_win: True home wins (0/1)
            y_pred_win_prob: Predicted home win probabilities
            y_true_home_score: True home scores
            y_true_away_score: True away scores
            y_pred_home_score: Predicted home scores
            y_pred_away_score: Predicted away scores
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Win prediction metrics
        y_pred_win = (y_pred_win_prob > 0.5).astype(int)
        
        metrics['accuracy'] = accuracy_score(y_true_win, y_pred_win)
        
        # Log loss (cross-entropy)
        # Clip probabilities to avoid log(0)
        y_pred_win_prob_clipped = np.clip(y_pred_win_prob, 1e-7, 1 - 1e-7)
        metrics['log_loss'] = log_loss(y_true_win, y_pred_win_prob_clipped)
        
        # Brier score
        metrics['brier_score'] = brier_score_loss(y_true_win, y_pred_win_prob)
        
        # Score prediction metrics
        metrics['mae_home'] = mean_absolute_error(y_true_home_score, y_pred_home_score)
        metrics['mae_away'] = mean_absolute_error(y_true_away_score, y_pred_away_score)
        metrics['mae_total'] = (metrics['mae_home'] + metrics['mae_away']) / 2
        
        metrics['rmse_home'] = np.sqrt(mean_squared_error(y_true_home_score, y_pred_home_score))
        metrics['rmse_away'] = np.sqrt(mean_squared_error(y_true_away_score, y_pred_away_score))
        metrics['rmse_total'] = (metrics['rmse_home'] + metrics['rmse_away']) / 2
        
        # Spread MAE
        true_spread = y_true_home_score - y_true_away_score
        pred_spread = y_pred_home_score - y_pred_away_score
        metrics['spread_mae'] = mean_absolute_error(true_spread, pred_spread)
        
        return metrics
    
    @staticmethod
    def compute_calibration(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve.
        
        Args:
            y_true: True labels (0/1)
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            (fraction_of_positives, mean_predicted_value)
        """
        return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    @staticmethod
    def compute_weekly_metrics(
        predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute metrics grouped by week.
        
        Args:
            predictions: DataFrame with columns: week, y_true_win, y_pred_win_prob, etc.
            
        Returns:
            DataFrame with metrics per week
        """
        weekly_metrics = []
        
        for week, week_df in predictions.groupby('week'):
            if len(week_df) == 0:
                continue
            
            metrics = MetricsCalculator.compute_all_metrics(
                y_true_win=week_df['y_true_win'].values,
                y_pred_win_prob=week_df['y_pred_win_prob'].values,
                y_true_home_score=week_df['y_true_home_score'].values,
                y_true_away_score=week_df['y_true_away_score'].values,
                y_pred_home_score=week_df['y_pred_home_score'].values,
                y_pred_away_score=week_df['y_pred_away_score'].values
            )
            
            metrics['week'] = week
            metrics['n_games'] = len(week_df)
            weekly_metrics.append(metrics)
        
        return pd.DataFrame(weekly_metrics)


class Calibrator:
    """
    Calibrates predicted probabilities using Platt scaling or isotonic regression.
    """
    
    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: 'platt' or 'isotonic'
        """
        self.method = method
        self.calibrator = None
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Fit calibrator.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        elif self.method == 'platt':
            # Platt scaling = logistic regression on predicted probabilities
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration.
        
        Args:
            y_prob: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'isotonic':
            return self.calibrator.transform(y_prob)
        else:  # platt
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]


class ResultsLogger:
    """
    Logs and saves evaluation results.
    """
    
    def __init__(self, results_dir: str = "results/"):
        """
        Initialize logger.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        
        import os
        os.makedirs(results_dir, exist_ok=True)
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Print metrics to console.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for printing
        """
        print(f"\n{prefix}Metrics:")
        print("─" * 50)
        
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.1%}")
        if 'log_loss' in metrics:
            print(f"  Log Loss: {metrics['log_loss']:.4f}")
        if 'brier_score' in metrics:
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        print()
        
        if 'mae_total' in metrics:
            print(f"  Score MAE (total): {metrics['mae_total']:.2f} points")
        if 'mae_home' in metrics:
            print(f"    Home: {metrics['mae_home']:.2f} points")
        if 'mae_away' in metrics:
            print(f"    Away: {metrics['mae_away']:.2f} points")
        
        print()
        
        if 'rmse_total' in metrics:
            print(f"  Score RMSE (total): {metrics['rmse_total']:.2f} points")
        if 'spread_mae' in metrics:
            print(f"  Spread MAE: {metrics['spread_mae']:.2f} points")
        
        print("─" * 50)
    
    def save_predictions(
        self,
        predictions: pd.DataFrame,
        filename: str
    ):
        """
        Save predictions to CSV.
        
        Args:
            predictions: DataFrame with predictions
            filename: Output filename
        """
        import os
        filepath = os.path.join(self.results_dir, filename)
        predictions.to_csv(filepath, index=False)
        print(f"✓ Predictions saved to {filepath}")
    
    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str
    ):
        """
        Save metrics to CSV.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        import os
        df = pd.DataFrame([metrics])
        filepath = os.path.join(self.results_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Metrics saved to {filepath}")
    
    def save_weekly_metrics(
        self,
        weekly_metrics: pd.DataFrame,
        filename: str
    ):
        """
        Save weekly metrics to CSV.
        
        Args:
            weekly_metrics: DataFrame with weekly metrics
            filename: Output filename
        """
        import os
        filepath = os.path.join(self.results_dir, filename)
        weekly_metrics.to_csv(filepath, index=False)
        print(f"✓ Weekly metrics saved to {filepath}")


def main():
    """Test metrics calculation."""
    print("\n" + "="*70)
    print("METRICS - TEST")
    print("="*70)
    
    # Create dummy predictions
    np.random.seed(42)
    n_games = 100
    
    # Simulate somewhat realistic predictions
    y_true_win = np.random.randint(0, 2, n_games)
    y_pred_win_prob = np.random.beta(2, 2, n_games)  # Between 0 and 1
    
    # Make predictions somewhat correlated with truth
    y_pred_win_prob = 0.7 * y_true_win + 0.3 * y_pred_win_prob
    y_pred_win_prob = np.clip(y_pred_win_prob, 0, 1)
    
    # Scores
    y_true_home_score = np.random.poisson(24, n_games).astype(float)
    y_true_away_score = np.random.poisson(21, n_games).astype(float)
    
    # Add noise to predictions
    y_pred_home_score = y_true_home_score + np.random.normal(0, 3, n_games)
    y_pred_away_score = y_true_away_score + np.random.normal(0, 3, n_games)
    
    y_pred_home_score = np.maximum(0, y_pred_home_score)
    y_pred_away_score = np.maximum(0, y_pred_away_score)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = MetricsCalculator.compute_all_metrics(
        y_true_win=y_true_win,
        y_pred_win_prob=y_pred_win_prob,
        y_true_home_score=y_true_home_score,
        y_true_away_score=y_true_away_score,
        y_pred_home_score=y_pred_home_score,
        y_pred_away_score=y_pred_away_score
    )
    
    logger = ResultsLogger(results_dir="results/")
    logger.log_metrics(metrics, prefix="Test ")
    
    # Test calibration
    print("\n\nTesting calibration...")
    frac_pos, mean_pred = MetricsCalculator.compute_calibration(
        y_true_win, y_pred_win_prob, n_bins=5
    )
    
    print("\nCalibration curve (5 bins):")
    print("  Predicted  |  Actual")
    print("─" * 30)
    for pred, actual in zip(mean_pred, frac_pos):
        print(f"  {pred:.3f}     |  {actual:.3f}")
    
    # Test calibrator
    print("\n\nTesting calibrator...")
    calibrator = Calibrator(method='isotonic')
    
    # Split for calibration
    n_train = 70
    calibrator.fit(y_true_win[:n_train], y_pred_win_prob[:n_train])
    
    y_calibrated = calibrator.transform(y_pred_win_prob[n_train:])
    
    print(f"Original probs (sample): {y_pred_win_prob[n_train:n_train+5]}")
    print(f"Calibrated probs (sample): {y_calibrated[:5]}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

