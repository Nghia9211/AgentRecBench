import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import GatingConfig, DEFAULT_CONFIG

class GatingMLP(nn.Module):
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
        layers.append(nn.Linear(in_dim, 3)) 
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

class GatingNetwork:
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
        
        self.feat_mean = None
        self.feat_std  = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            print(f"[GatingNetwork] No checkpoint found — using default weights {self.cfg.default_weights}")

    def predict(
        self,
        signal_scores: Dict[str, Dict[str, float]],
        len_seq: int = 0, 
    ) -> Dict[str, Tuple[float, float, float]]:
        if not signal_scores: return {}

        items = list(signal_scores.keys())
        norm_len = min(len_seq / 50.0, 1.0) 

        raw_features = np.array([
            [signal_scores[it]['seq'],
             signal_scores[it]['gcn'],
             signal_scores[it]['sem'],
             norm_len] 
            for it in items
        ], dtype=np.float32)

        if not self.trained:
            if len_seq <= 5: return {it: (0.2, 0.4, 0.4) for it in items}
            return {it: self.cfg.default_weights for it in items}

        if self.feat_mean is not None and self.feat_std is not None:
            features_to_infer = (raw_features - self.feat_mean) / self.feat_std
        else:
            features_to_infer = raw_features

        x = torch.tensor(features_to_infer, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            weights = self.model(x).cpu().numpy()

        return {
            it: (float(weights[i, 0]), float(weights[i, 1]), float(weights[i, 2]))
            for i, it in enumerate(items)
        }

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
        
        if 'norm_min' in ckpt and 'norm_max' in ckpt:
            self.feat_mean = np.array(ckpt['norm_min'], dtype=np.float32)
            self.feat_std  = np.array(ckpt['norm_max'], dtype=np.float32)
            print(f"[GatingNetwork] Loaded normalization params: Mean={self.feat_mean}, Std={self.feat_std}")