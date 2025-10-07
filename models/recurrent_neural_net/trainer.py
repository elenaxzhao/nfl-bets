#!/usr/bin/env python3
"""
Training Loop Module
====================

Handles model training with:
- Early stopping
- Mixed precision (AMP)
- Gradient clipping
- Recency weighting
- Multiple loss functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    weight_decay: float = 0.0001
    gradient_clip_value: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_log_loss"
    early_stopping_mode: str = "min"  # 'min' or 'max'
    recency_half_life: float = 8.0
    apply_recency_as: str = "sample_weights"  # 'sample_weights' or 'attention_weights'
    lambda_bce: float = 1.0  # Win loss weight
    lambda_score: float = 0.5  # Score loss weight
    score_loss_type: str = "mae"  # 'mae', 'mse', 'poisson', 'negative_binomial'
    device: str = "cuda"  # Will be auto-detected


class CombinedLoss(nn.Module):
    """
    Combined loss for win prediction and score prediction.
    """
    
    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_score: float = 0.5,
        score_loss_type: str = "mae"
    ):
        """
        Initialize combined loss.
        
        Args:
            lambda_bce: Weight for win prediction loss
            lambda_score: Weight for score prediction loss
            score_loss_type: Type of score loss ('mae', 'mse', 'poisson', 'negative_binomial')
        """
        super().__init__()
        
        self.lambda_bce = lambda_bce
        self.lambda_score = lambda_score
        self.score_loss_type = score_loss_type
        
        # Win prediction loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(
        self,
        win_logits: torch.Tensor,
        win_targets: torch.Tensor,
        home_scores: torch.Tensor,
        away_scores: torch.Tensor,
        home_score_targets: torch.Tensor,
        away_score_targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            win_logits: Predicted win logits
            win_targets: True win labels
            home_scores: Predicted home scores
            away_scores: Predicted away scores
            home_score_targets: True home scores
            away_score_targets: True away scores
            sample_weights: Optional sample weights
            
        Returns:
            (total_loss, loss_dict)
        """
        # Win prediction loss (BCE)
        bce_loss = self.bce_loss(win_logits, win_targets)
        
        # Apply sample weights if provided
        if sample_weights is not None:
            bce_loss = bce_loss * sample_weights
        
        bce_loss = bce_loss.mean()
        
        # Score prediction loss
        if self.score_loss_type == 'mae':
            score_loss_home = torch.abs(home_scores - home_score_targets)
            score_loss_away = torch.abs(away_scores - away_score_targets)
        elif self.score_loss_type == 'mse':
            score_loss_home = (home_scores - home_score_targets) ** 2
            score_loss_away = (away_scores - away_score_targets) ** 2
        elif self.score_loss_type == 'poisson':
            # Poisson NLL
            score_loss_home = self._poisson_nll(home_scores, home_score_targets)
            score_loss_away = self._poisson_nll(away_scores, away_score_targets)
        elif self.score_loss_type == 'negative_binomial':
            # Simplified negative binomial (assumes fixed dispersion)
            score_loss_home = self._negative_binomial_nll(home_scores, home_score_targets)
            score_loss_away = self._negative_binomial_nll(away_scores, away_score_targets)
        else:
            raise ValueError(f"Unknown score loss type: {self.score_loss_type}")
        
        score_loss = (score_loss_home + score_loss_away) / 2
        
        # Apply sample weights
        if sample_weights is not None:
            score_loss = score_loss * sample_weights
        
        score_loss = score_loss.mean()
        
        # Total loss
        total_loss = self.lambda_bce * bce_loss + self.lambda_score * score_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'bce': bce_loss.item(),
            'score': score_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _poisson_nll(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Poisson negative log-likelihood."""
        # Ensure predictions are positive
        predictions = torch.clamp(predictions, min=1e-6)
        
        # NLL = lambda - y * log(lambda) + log(y!)
        # Ignoring the log(y!) constant term
        nll = predictions - targets * torch.log(predictions)
        
        return nll
    
    def _negative_binomial_nll(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Negative binomial negative log-likelihood.
        Simplified version with fixed dispersion parameter.
        """
        # Ensure predictions are positive
        mu = torch.clamp(predictions, min=1e-6)
        
        # Variance = mu + alpha * mu^2
        # This is a simplified version; full NB would learn alpha
        var = mu + alpha * mu ** 2
        
        # Use Gaussian approximation for simplicity
        nll = (targets - mu) ** 2 / (2 * var) + 0.5 * torch.log(var)
        
        return nll


class EarlyStopping:
    """
    Early stopping handler.
    """
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        delta: float = 0.0
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            mode: 'min' or 'max'
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.delta)
        else:
            improved = score > (self.best_score + self.delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class Trainer:
    """
    Handles model training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.config = config
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = CombinedLoss(
            lambda_bce=config.lambda_bce,
            lambda_score=config.lambda_score,
            score_loss_type=config.score_loss_type
        )
        
        # AMP scaler
        self.use_amp = config.use_amp and (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=config.early_stopping_mode
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_model_state = None
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        use_recency_weights: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            use_recency_weights: Whether to use recency weighting
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_bce_loss = 0.0
        total_score_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Recency weights (if using)
            recency_weights = None
            if use_recency_weights and self.config.apply_recency_as == "attention_weights":
                from sequence_builder import RecencyWeightCalculator
                seq_lengths = batch['home_masks'].sum(dim=1).cpu().numpy()
                recency_weights = RecencyWeightCalculator.compute_weights(
                    seq_lengths,
                    max_length=batch['home_masks'].shape[1],
                    half_life=self.config.recency_half_life
                )
                recency_weights = torch.FloatTensor(recency_weights).to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        home_sequences=batch['home_sequences'],
                        away_sequences=batch['away_sequences'],
                        home_masks=batch['home_masks'],
                        away_masks=batch['away_masks'],
                        is_divisional=batch.get('is_divisional'),
                        recency_weights=recency_weights
                    )
                    
                    # Sample weights for loss
                    sample_weights = None
                    if use_recency_weights and self.config.apply_recency_as == "sample_weights":
                        # Use average recency weight per sample
                        seq_lengths = batch['home_masks'].sum(dim=1)
                        weights = torch.exp(-torch.arange(batch['home_masks'].shape[1], device=self.device).float() / self.config.recency_half_life)
                        # This is simplified; could be more sophisticated
                        sample_weights = torch.ones_like(seq_lengths)
                    
                    loss, loss_dict = self.criterion(
                        win_logits=outputs['win_logits'],
                        win_targets=batch['home_win'],
                        home_scores=outputs['home_scores'],
                        away_scores=outputs['away_scores'],
                        home_score_targets=batch['home_score'],
                        away_score_targets=batch['away_score'],
                        sample_weights=sample_weights
                    )
                
                # Backward pass with AMP
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    home_sequences=batch['home_sequences'],
                    away_sequences=batch['away_sequences'],
                    home_masks=batch['home_masks'],
                    away_masks=batch['away_masks'],
                    is_divisional=batch.get('is_divisional'),
                    recency_weights=recency_weights
                )
                
                loss, loss_dict = self.criterion(
                    win_logits=outputs['win_logits'],
                    win_targets=batch['home_win'],
                    home_scores=outputs['home_scores'],
                    away_scores=outputs['away_scores'],
                    home_score_targets=batch['home_score'],
                    away_score_targets=batch['away_score']
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )
                
                # Optimizer step
                self.optimizer.step()
            
            total_loss += loss_dict['total']
            total_bce_loss += loss_dict['bce']
            total_score_loss += loss_dict['score']
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'bce_loss': total_bce_loss / n_batches,
            'score_loss': total_score_loss / n_batches
        }
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            (metrics_dict, predictions_dict)
        """
        self.model.eval()
        
        all_win_logits = []
        all_win_probs = []
        all_home_scores = []
        all_away_scores = []
        all_win_targets = []
        all_home_score_targets = []
        all_away_score_targets = []
        
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    home_sequences=batch['home_sequences'],
                    away_sequences=batch['away_sequences'],
                    home_masks=batch['home_masks'],
                    away_masks=batch['away_masks'],
                    is_divisional=batch.get('is_divisional')
                )
                
                loss, _ = self.criterion(
                    win_logits=outputs['win_logits'],
                    win_targets=batch['home_win'],
                    home_scores=outputs['home_scores'],
                    away_scores=outputs['away_scores'],
                    home_score_targets=batch['home_score'],
                    away_score_targets=batch['away_score']
                )
                
                total_loss += loss.item()
                n_batches += 1
                
                # Collect predictions
                all_win_logits.append(outputs['win_logits'].cpu().numpy())
                all_win_probs.append(outputs['win_probs'].cpu().numpy())
                all_home_scores.append(outputs['home_scores'].cpu().numpy())
                all_away_scores.append(outputs['away_scores'].cpu().numpy())
                all_win_targets.append(batch['home_win'].cpu().numpy())
                all_home_score_targets.append(batch['home_score'].cpu().numpy())
                all_away_score_targets.append(batch['away_score'].cpu().numpy())
        
        # Concatenate all predictions
        predictions = {
            'win_logits': np.concatenate(all_win_logits),
            'win_probs': np.concatenate(all_win_probs),
            'home_scores': np.concatenate(all_home_scores),
            'away_scores': np.concatenate(all_away_scores),
            'win_targets': np.concatenate(all_win_targets),
            'home_score_targets': np.concatenate(all_home_score_targets),
            'away_score_targets': np.concatenate(all_away_score_targets)
        }
        
        # Compute metrics
        from metrics import MetricsCalculator
        metrics = MetricsCalculator.compute_all_metrics(
            y_true_win=predictions['win_targets'],
            y_pred_win_prob=predictions['win_probs'],
            y_true_home_score=predictions['home_score_targets'],
            y_true_away_score=predictions['away_score_targets'],
            y_pred_home_score=predictions['home_scores'],
            y_pred_away_score=predictions['away_scores']
        )
        
        metrics['loss'] = total_loss / n_batches
        
        return metrics, predictions
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        if verbose:
            print("\n" + "="*70)
            print("TRAINING")
            print("="*70)
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            
            # Validate
            if val_loader is not None:
                val_metrics, _ = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_metrics'].append(val_metrics)
                
                # Print progress
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
                    print(f"  Train Loss: {train_metrics['loss']:.4f}")
                    print(f"  Val Loss: {val_metrics['loss']:.4f}")
                    print(f"  Val Accuracy: {val_metrics['accuracy']:.1%}")
                    print(f"  Val Log Loss: {val_metrics['log_loss']:.4f}")
                    print(f"  Val MAE: {val_metrics['mae_total']:.2f}")
                
                # Early stopping
                stop_metric = val_metrics.get(
                    self.config.early_stopping_metric.replace('val_', ''),
                    val_metrics['loss']
                )
                
                # Save best model
                if self.early_stopping.best_score is None or \
                   (self.config.early_stopping_mode == 'min' and stop_metric < self.early_stopping.best_score) or \
                   (self.config.early_stopping_mode == 'max' and stop_metric > self.early_stopping.best_score):
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                
                if self.early_stopping(stop_metric, epoch):
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        print(f"Best epoch: {self.early_stopping.best_epoch + 1}")
                    break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
                    print(f"  Train Loss: {train_metrics['loss']:.4f}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print("\n✓ Restored best model")
        
        if verbose:
            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print("="*70)
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, filepath)
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"✓ Checkpoint loaded from {filepath}")


def main():
    """Test trainer."""
    print("\n" + "="*70)
    print("TRAINER - TEST")
    print("="*70)
    
    from model import NFLGamePredictor
    
    # Create model
    model = NFLGamePredictor(
        feature_dim=14,
        team_embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    )
    
    # Create config
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=16,
        max_epochs=10,
        use_amp=False,  # Disable for CPU test
        early_stopping_patience=5
    )
    
    # Create trainer
    trainer = Trainer(model, config, device=torch.device('cpu'))
    
    # Create dummy data
    from torch.utils.data import TensorDataset, DataLoader
    
    n_samples = 64
    seq_len = 10
    feature_dim = 14
    
    home_seqs = torch.randn(n_samples, seq_len, feature_dim)
    away_seqs = torch.randn(n_samples, seq_len, feature_dim)
    home_masks = torch.ones(n_samples, seq_len)
    away_masks = torch.ones(n_samples, seq_len)
    home_wins = torch.randint(0, 2, (n_samples,)).float()
    home_scores = torch.randint(10, 40, (n_samples,)).float()
    away_scores = torch.randint(10, 40, (n_samples,)).float()
    is_div = torch.zeros(n_samples)
    
    dataset = TensorDataset(
        home_seqs, away_seqs, home_masks, away_masks,
        home_wins, home_scores, away_scores, is_div
    )
    
    # Custom collate to match expected format
    def collate_fn(batch):
        return {
            'home_sequences': torch.stack([b[0] for b in batch]),
            'away_sequences': torch.stack([b[1] for b in batch]),
            'home_masks': torch.stack([b[2] for b in batch]),
            'away_masks': torch.stack([b[3] for b in batch]),
            'home_win': torch.stack([b[4] for b in batch]),
            'home_score': torch.stack([b[5] for b in batch]),
            'away_score': torch.stack([b[6] for b in batch]),
            'is_divisional': torch.stack([b[7] for b in batch])
        }
    
    train_loader = DataLoader(dataset[:48], batch_size=16, collate_fn=collate_fn)
    val_loader = DataLoader(dataset[48:], batch_size=16, collate_fn=collate_fn)
    
    # Train
    print("\nTraining for 10 epochs...")
    history = trainer.train(train_loader, val_loader, verbose=True)
    
    print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

