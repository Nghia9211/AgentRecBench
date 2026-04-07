"""
moe_fusion/gating_network.py
─────────────────────────────
MoE Gating Network: g(u, i) = softmax(MLP(x_{u,i}))

MLP nhỏ nhận feature vector [s_seq, s_gcn, s_sem] (đã normalize),
output 3 weights [g1, g2, g3] qua softmax.

Khi chưa train: dùng default_weights (uniform 1/3, 1/3, 1/3).
Sau khi train bằng train_gating.py: load từ checkpoint.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import GatingConfig, DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# MLP Model
# ─────────────────────────────────────────────────────────────────────────────

class GatingMLP(nn.Module):
    """
    MLP nhỏ để học gating weights từ (s_seq, s_gcn, s_sem).

    Architecture: Linear → ReLU → Dropout → Linear → ... → Linear → Softmax
    Input:  (batch, 3)  — [s_seq, s_gcn, s_sem] đã normalize
    Output: (batch, 3)  — [g1, g2, g3] softmax weights
    """

    def __init__(self, cfg: GatingConfig = None):
        super().__init__()
        cfg = cfg or DEFAULT_CONFIG.gating
        self.cfg = cfg  

        layers = []
        in_dim = cfg.input_dim
        for h_dim in cfg.hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, cfg.input_dim))  # output = num_signals
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3) float tensor
        Returns: (batch, 3) softmax weights
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# GatingNetwork wrapper (train + inference)
# ─────────────────────────────────────────────────────────────────────────────

class GatingNetwork:
    """
    Wrapper quản lý GatingMLP: load, save, inference.

    Inference flow:
        features = [s_seq_norm, s_gcn_norm, s_sem_norm]  # per (u, i)
        g1, g2, g3 = gating.predict(features)
        s0 = g1 * s_seq + g2 * s_gcn + g3 * s_sem
    """

    def __init__(
        self,
        cfg:             GatingConfig  = None,
        model_path:      str           = None,
        device:          torch.device  = None,
    ):
        self.cfg    = cfg or DEFAULT_CONFIG.gating
        self.device = device or torch.device("cpu")
        self.model  = GatingMLP(self.cfg).to(self.device)
        self.trained = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            print("[GatingNetwork] No checkpoint found — using default weights "
                  f"{self.cfg.default_weights}")

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────

    def predict(
        self,
        signal_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Predict gating weights cho tất cả items trong signal_scores.

        Args:
            signal_scores: Dict[item_name → {'seq': float, 'gcn': float, 'sem': float}]

        Returns:
            Dict[item_name → (g1, g2, g3)]
        """
        if not signal_scores:
            return {}

        items = list(signal_scores.keys())
        raw_features = np.array([
            [signal_scores[it]['seq'],
             signal_scores[it]['gcn'],
             signal_scores[it]['sem']]
            for it in items
        ], dtype=np.float32)

        # Normalize features per column (seq, gcn, sem riêng biệt)
        norm_features = self._normalize_features(raw_features)

        if not self.trained:
            # Dùng default uniform weights
            w = self.cfg.default_weights
            return {it: (w[0], w[1], w[2]) for it in items}

        x = torch.tensor(norm_features, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            weights = self.model(x).cpu().numpy()  # (n_items, 3)

        return {
            it: (float(weights[i, 0]), float(weights[i, 1]), float(weights[i, 2]))
            for i, it in enumerate(items)
        }

    def predict_single(
        self,
        s_seq: float,
        s_gcn: float,
        s_sem: float,
    ) -> Tuple[float, float, float]:
        """
        Predict gating weights cho 1 item.
        Tiện cho testing/debugging.
        """
        if not self.trained:
            w = self.cfg.default_weights
            return (w[0], w[1], w[2])

        x = torch.tensor([[s_seq, s_gcn, s_sem]],
                          dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            w = self.model(x).cpu().numpy()[0]
        return (float(w[0]), float(w[1]), float(w[2]))

    # ─────────────────────────────────────────────────────────────────────
    # Feature normalization
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        result = features.copy()
        for col in range(features.shape[1]):
            col_data = features[:, col]
            mean_v, std_v = col_data.mean(), col_data.std()
            if std_v > 0:
                result[:, col] = (col_data - mean_v) / std_v
            else:
                result[:, col] = 0.0
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'cfg':              self.cfg,
        }, path)
        print(f"[GatingNetwork] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.trained = True
        print(f"[GatingNetwork] Loaded from {path} — using trained weights")
