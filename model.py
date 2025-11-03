import math
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1),
        )
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x)
        x_sel = x * weights
        return self.proj(x_sel)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.fc1(x))
        x = self.dropout(self.fc2(x))
        gate = torch.sigmoid(self.gate(x))
        x = gate * x + (1 - gate) * residual
        return self.norm(x)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_ff: int = 256, dropout: float = 0.1,
                 max_pos_len: int = 1000):
        super().__init__()
        self.varsel = VariableSelectionNetwork(in_dim, d_model)
        self.register_buffer("pos_encoding", self._build_positional_encoding(d_model, max_pos_len))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion_gate = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _build_positional_encoding(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        h = self.varsel(x)
        h = h + self.pos_encoding[:, :seq_len, :].to(x.device)
        h = self.encoder(h)
        h = self.fusion_gate(h)
        return self.head(h[:, -1, :]).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: torch.Tensor = None,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone())
        else:
            self.pos_weight = None
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(logits)
        pos_weight = None if self.pos_weight is None else self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCELossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.05, pos_weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pos_weight = None if self.pos_weight is None else self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def compute_class_stats(labels: np.ndarray) -> Tuple[float, float]:
    pos_rate = float(np.mean(labels))
    neg_rate = 1.0 - pos_rate
    pos_weight = neg_rate / max(pos_rate, 1e-6) if pos_rate < 1.0 else 1.0
    return pos_rate, pos_weight


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              optimizer: torch.optim.Optimizer = None, grad_clip: float = None,
              device: torch.device = torch.device("cpu")) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)

    losses = []
    probs = []
    labels = []
    rets = []

    for batch in loader:
        if len(batch) == 3:
            xb, yb, rb = batch
        else:
            xb, yb = batch
            rb = torch.zeros_like(yb)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0).to(device)
        yb = yb.to(device)
        rb = rb.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        losses.append(loss.item())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(yb.detach().cpu().numpy().tolist())
        rets.extend(rb.detach().cpu().numpy().tolist())

    return float(np.mean(losses)), np.asarray(probs), np.asarray(labels), np.asarray(rets)


def trade_statistics(prob: np.ndarray, ret: np.ndarray, thr: float, cost: float = 0.0) -> dict:
    prob = np.asarray(prob)
    ret = np.asarray(ret)
    mask = ~np.isnan(prob) & ~np.isnan(ret)
    prob = prob[mask]
    ret = ret[mask]
    signal = np.where(prob >= thr, 1.0, -1.0)
    trade_penalty = np.abs(np.diff(signal, prepend=0.0)) * cost
    pnl = signal * ret - trade_penalty
    mean = float(np.nanmean(pnl)) if len(pnl) else float("nan")
    std = float(np.nanstd(pnl)) if len(pnl) else float("nan")
    sharpe = mean / std if std and std > 1e-8 else float("nan")
    hit_rate = float(np.mean(np.sign(ret) == np.sign(signal))) if len(pnl) else float("nan")
    return dict(mean=mean, std=std, sharpe=sharpe, hit_rate=hit_rate)


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidates: Iterable[float],
    metric: str,
    y_ret: np.ndarray = None,
    cost: float = 0.0,
) -> float:
    if len(np.unique(y_true)) <= 1:
        print("Validation labels contain a single class; keep threshold=0.500")
        return 0.5

    candidates = list(candidates)
    metric = (metric or "accuracy").lower()

    def score_fn(y_t: np.ndarray, y_pred: np.ndarray, thr: float) -> float:
        if metric == "accuracy":
            return accuracy_score(y_t, y_pred)
        if metric in {"balanced_accuracy", "balanced"}:
            return balanced_accuracy_score(y_t, y_pred)
        if metric == "f1":
            return f1_score(y_t, y_pred, zero_division=0)
        if metric == "sharpe":
            if y_ret is None:
                raise ValueError("Sharpe-based threshold selection requires return series.")
            stats = trade_statistics(y_prob, y_ret, thr, cost)
            sharpe = stats["sharpe"]
            return sharpe if np.isfinite(sharpe) else float("-inf")
        raise ValueError(f"Unsupported threshold metric: {metric}")

    scores = []
    for thr in candidates:
        preds = (y_prob >= thr).astype(int)
        try:
            score = score_fn(y_true, preds, thr)
        except ValueError:
            score = float("-inf")
        scores.append(score)

    if not scores or max(scores) == float("-inf"):
        print("Failed to compute threshold metric; defaulting to 0.500.")
        return 0.5

    best_idx = int(np.argmax(scores))
    best_thr = float(candidates[best_idx])
    print(f"Selected val threshold={best_thr:.3f} via {metric} (score={scores[best_idx]:.3f})")
    return best_thr


def safe_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    y_pred = (y_prob >= thr).astype(int)

    metrics = dict(acc=float("nan"), bal=float("nan"), f1=float("nan"), mcc=float("nan"), roc=float("nan"))
    if len(y_true):
        metrics["acc"] = accuracy_score(y_true, y_pred)
        if len(np.unique(y_true)) > 1:
            metrics["bal"] = balanced_accuracy_score(y_true, y_pred)
            metrics["f1"] = f1_score(y_true, y_pred)
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
            metrics["roc"] = roc_auc_score(y_true, y_prob)
    return metrics
