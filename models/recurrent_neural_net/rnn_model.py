#!/usr/bin/env python3
"""
RNN Model Architecture
======================

Siamese encoders (GRU/LSTM) for home and away teams.
Dual prediction heads for win probability and scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class TeamEncoder(nn.Module):
    """
    Encodes a team's recent game sequence using GRU or LSTM.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        encoder_type: str = 'gru',
        bidirectional: bool = False
    ):
        """
        Initialize team encoder.
        
        Args:
            input_dim: Feature dimension per game
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            dropout: Dropout rate
            encoder_type: 'gru' or 'lstm'
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        
        # Select RNN type
        if encoder_type == 'gru':
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif encoder_type == 'lstm':
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
    def forward(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor,
        recency_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode team sequences.
        
        Args:
            sequences: (batch_size, seq_len, input_dim)
            masks: (batch_size, seq_len) - 1 for valid, 0 for padding
            recency_weights: Optional (batch_size, seq_len) weights for attention
            
        Returns:
            Team encoding: (batch_size, output_dim)
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Pack padded sequences for efficiency
        seq_lengths = masks.sum(dim=1).cpu().long()
        
        # Avoid zero-length sequences
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences,
            seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # Run RNN
        if self.encoder_type == 'gru':
            packed_output, hidden = self.rnn(packed)
        else:  # LSTM
            packed_output, (hidden, cell) = self.rnn(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get final state
        if self.bidirectional:
            # Concatenate forward and backward final states
            # hidden: (num_layers * 2, batch, hidden_dim)
            forward_hidden = hidden[-2, :, :]  # Last forward layer
            backward_hidden = hidden[-1, :, :]  # Last backward layer
            final_state = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # Just take last layer
            final_state = hidden[-1, :, :]
        
        # Apply recency-weighted attention if provided
        if recency_weights is not None:
            # Compute attention over outputs
            # Expand weights: (batch, seq_len, 1)
            weights = recency_weights.unsqueeze(-1)
            
            # Mask out padding
            weights = weights * masks.unsqueeze(-1)
            
            # Normalize
            weights_sum = weights.sum(dim=1, keepdim=True)
            weights = weights / (weights_sum + 1e-8)
            
            # Weighted sum of outputs
            weighted_output = (output * weights).sum(dim=1)
            
            # Combine with final state
            final_state = (final_state + weighted_output) / 2
        
        return final_state


class WinPredictionHead(nn.Module):
    """
    MLP head for predicting win probability.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.3):
        """
        Initialize win prediction head.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer outputs logit
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            Logits: (batch_size, 1)
        """
        return self.network(x)


class ScorePredictionHead(nn.Module):
    """
    MLP head for predicting team scores.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.3):
        """
        Initialize score prediction head.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer outputs 2 values (home_score, away_score)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            (home_scores, away_scores): Each (batch_size,)
        """
        output = self.network(x)
        
        # Apply ReLU to ensure non-negative scores
        output = F.relu(output)
        
        home_scores = output[:, 0]
        away_scores = output[:, 1]
        
        return home_scores, away_scores


class NFLGamePredictor(nn.Module):
    """
    Complete model for NFL game prediction using Siamese encoders.
    """
    
    def __init__(
        self,
        feature_dim: int,
        team_embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        encoder_type: str = 'gru',
        bidirectional: bool = False,
        win_head_hidden: List[int] = [64, 32],
        score_head_hidden: List[int] = [64, 32],
        num_teams: int = 32,
    ):
        """
        Initialize complete model.
        
        Args:
            feature_dim: Per-game feature dimension
            team_embedding_dim: Team embedding dimension
            hidden_dim: RNN hidden dimension
            num_layers: Number of RNN layers
            dropout: Dropout rate
            encoder_type: 'gru' or 'lstm'
            bidirectional: Bidirectional RNN
            win_head_hidden: Hidden dims for win head
            score_head_hidden: Hidden dims for score head
            num_teams: Number of teams (for embeddings)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.team_embedding_dim = team_embedding_dim
        
        # Team embeddings (optional - can be looked up by team ID)
        self.team_embeddings = nn.Embedding(num_teams, team_embedding_dim)
        
        # Siamese encoder (shared weights for both teams)
        self.encoder = TeamEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            encoder_type=encoder_type,
            bidirectional=bidirectional
        )
        
        encoder_output_dim = self.encoder.output_dim
        
        # Combined representation dimension
        # [home_encoding, away_encoding, matchup_features, home_embedding, away_embedding]
        combined_dim = encoder_output_dim * 2 + team_embedding_dim * 2 + 2  # +2 for is_home, is_divisional
        
        # Prediction heads
        self.win_head = WinPredictionHead(
            input_dim=combined_dim,
            hidden_dims=win_head_hidden,
            dropout=dropout
        )
        
        self.score_head = ScorePredictionHead(
            input_dim=combined_dim,
            hidden_dims=score_head_hidden,
            dropout=dropout
        )
        
    def forward(
        self,
        home_sequences: torch.Tensor,
        away_sequences: torch.Tensor,
        home_masks: torch.Tensor,
        away_masks: torch.Tensor,
        home_team_ids: Optional[torch.Tensor] = None,
        away_team_ids: Optional[torch.Tensor] = None,
        is_divisional: Optional[torch.Tensor] = None,
        recency_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            home_sequences: (batch, seq_len, feature_dim)
            away_sequences: (batch, seq_len, feature_dim)
            home_masks: (batch, seq_len)
            away_masks: (batch, seq_len)
            home_team_ids: Optional (batch,) team IDs
            away_team_ids: Optional (batch,) team IDs
            is_divisional: Optional (batch,) divisional flags
            recency_weights: Optional (batch, seq_len) recency weights
            
        Returns:
            Dictionary with predictions
        """
        batch_size = home_sequences.shape[0]
        
        # Encode sequences
        home_encoding = self.encoder(home_sequences, home_masks, recency_weights)
        away_encoding = self.encoder(away_sequences, away_masks, recency_weights)
        
        # Get team embeddings
        if home_team_ids is not None and away_team_ids is not None:
            home_embed = self.team_embeddings(home_team_ids)
            away_embed = self.team_embeddings(away_team_ids)
        else:
            # Default to zero embeddings
            home_embed = torch.zeros(batch_size, self.team_embedding_dim, device=home_sequences.device)
            away_embed = torch.zeros(batch_size, self.team_embedding_dim, device=away_sequences.device)
        
        # Matchup features
        is_home_flag = torch.ones(batch_size, 1, device=home_sequences.device)
        
        if is_divisional is not None:
            is_div_flag = is_divisional.unsqueeze(1)
        else:
            is_div_flag = torch.zeros(batch_size, 1, device=home_sequences.device)
        
        # Combine all features
        combined = torch.cat([
            home_encoding,
            away_encoding,
            home_embed,
            away_embed,
            is_home_flag,
            is_div_flag
        ], dim=1)
        
        # Win probability
        win_logits = self.win_head(combined).squeeze(-1)
        win_probs = torch.sigmoid(win_logits)
        
        # Score predictions
        home_scores, away_scores = self.score_head(combined)
        
        return {
            'win_logits': win_logits,
            'win_probs': win_probs,
            'home_scores': home_scores,
            'away_scores': away_scores
        }


def create_model_from_config(config: Dict, num_teams: int = 32) -> NFLGamePredictor:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Configuration dict
        num_teams: Number of teams
        
    Returns:
        NFLGamePredictor model
    """
    model_config = config.get('model', {})
    
    # Infer feature dimension from feature config
    # This will be set properly during training
    feature_dim = model_config.get('feature_dim', 16)
    
    model = NFLGamePredictor(
        feature_dim=feature_dim,
        team_embedding_dim=model_config.get('team_embedding_dim', 32),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.3),
        encoder_type=model_config.get('encoder_type', 'gru'),
        bidirectional=model_config.get('bidirectional', False),
        win_head_hidden=model_config.get('win_head_hidden', [64, 32]),
        score_head_hidden=model_config.get('score_head_hidden', [64, 32]),
        num_teams=num_teams
    )
    
    return model


def main():
    """Test model architecture."""
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE - TEST")
    print("="*70)
    
    # Create model
    model = NFLGamePredictor(
        feature_dim=14,  # Example: 14 features per game
        team_embedding_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        encoder_type='gru',
        bidirectional=False,
        num_teams=32
    )
    
    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy input
    batch_size = 4
    seq_len = 10
    feature_dim = 14
    
    home_sequences = torch.randn(batch_size, seq_len, feature_dim)
    away_sequences = torch.randn(batch_size, seq_len, feature_dim)
    home_masks = torch.ones(batch_size, seq_len)
    away_masks = torch.ones(batch_size, seq_len)
    
    # Mask some positions
    home_masks[0, :3] = 0  # First sample has 3 padding steps
    away_masks[1, :5] = 0  # Second sample has 5 padding steps
    
    home_team_ids = torch.randint(0, 32, (batch_size,))
    away_team_ids = torch.randint(0, 32, (batch_size,))
    is_divisional = torch.randint(0, 2, (batch_size,)).float()
    
    print(f"\nInput shapes:")
    print(f"  home_sequences: {home_sequences.shape}")
    print(f"  away_sequences: {away_sequences.shape}")
    print(f"  home_masks: {home_masks.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            home_sequences=home_sequences,
            away_sequences=away_sequences,
            home_masks=home_masks,
            away_masks=away_masks,
            home_team_ids=home_team_ids,
            away_team_ids=away_team_ids,
            is_divisional=is_divisional
        )
    
    print(f"\nOutput shapes:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    print(f"\nSample predictions:")
    print(f"  Win probabilities: {outputs['win_probs'][:3].numpy()}")
    print(f"  Home scores: {outputs['home_scores'][:3].numpy()}")
    print(f"  Away scores: {outputs['away_scores'][:3].numpy()}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

