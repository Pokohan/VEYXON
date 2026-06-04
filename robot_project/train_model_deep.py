"""
train_model_deep.py — Entraînement LSTM (PyTorch)

Architecture :
  • Entrée  : séquence de SEQ_LEN frames × 126 features (positions + vitesses)
  • LSTM    : 2 couches, hidden=128, dropout=0.3
  • Sortie  : 4 classes (Vide / Faible / Optimal / Trop fort)

Avantage vs MLP/RandomForest :
  Le modèle voit le MOUVEMENT de fermeture de la main sur N frames,
  pas seulement une pose instantanée. Beaucoup plus robuste en conditions réelles.

Compatible avec robot_autonome.py :
  Sauvegardé via joblib, expose predict() et predict_proba() comme sklearn.
"""
from typing import Union
import os, logging, joblib, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import (
    CLEAN_CSV, MERGED_CSV, MODEL_PATH, LOG_DIR,
    TEST_SIZE, RANDOM_STATE, PRESSURE_CLASSES,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR,"train_deep.log")), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Hyperparamètres ───────────────────────────────────────────────────────────
SEQ_LEN    = 20      # nb de frames par séquence
INPUT_DIM  = 126     # 63 positions + 63 vitesses
HIDDEN_DIM = 128     # neurones par couche LSTM
NUM_LAYERS = 2       # couches LSTM empilées
DROPOUT    = 0.3
NUM_CLASSES= 4
BATCH_SIZE = 64
EPOCHS     = 60
LR         = 1e-3
PATIENCE   = 10      # early stopping

LM_COLS  = [f"lm{i}_{a}" for i in range(21) for a in ("x","y","z")]
VEL_COLS = [f"lm{i}_v{a}" for i in range(21) for a in ("x","y","z")]
DEEP_MODEL_PATH = os.path.join(os.path.dirname(MODEL_PATH), "pressure_model_lstm.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset PyTorch : découpe le CSV en séquences glissantes
# ══════════════════════════════════════════════════════════════════════════════
class HandSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.sequences = []
        self.labels    = []
        # Fenêtre glissante : chaque séquence = SEQ_LEN frames consécutives
        for i in range(len(X) - seq_len + 1):
            self.sequences.append(X[i : i + seq_len])
            # Classe de la dernière frame = label de la séquence
            self.labels.append(y[i + seq_len - 1])

        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels    = torch.tensor(np.array(self.labels),    dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Architecture LSTM
# ══════════════════════════════════════════════════════════════════════════════
class HandLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x : (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out    = out[:, -1, :]     # on prend uniquement la dernière frame
        out    = self.dropout(out)
        return self.head(out)      # logits (batch, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper sklearn-compatible pour robot_autonome.py
# ══════════════════════════════════════════════════════════════════════════════
class LSTMWrapper:
    """
    Expose predict() et predict_proba() comme un modèle sklearn.
    robot_autonome.py n'a pas besoin de savoir que c'est du PyTorch.
    """
    def __init__(self, model: HandLSTM, scaler: StandardScaler, seq_len: int):
        self.model   = model
        self.scaler  = scaler
        self.seq_len = seq_len
        self._buffer: list = []    # fenêtre glissante en inférence temps réel

    def _prepare(self, X_single: np.ndarray) -> Union[torch.Tensor, None]:
        """
        Accumule les frames dans un buffer glissant.
        Retourne None si le buffer n'est pas encore plein.
        """
        scaled = self.scaler.transform(X_single.reshape(1, -1))[0]
        self._buffer.append(scaled)
        if len(self._buffer) > self.seq_len:
            self._buffer.pop(0)
        if len(self._buffer) < self.seq_len:
            return None
        seq = np.array(self._buffer)
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = self._prepare(X)
            if t is None:
                return np.array([0])
            logits = self.model(t)
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = self._prepare(X)
            if t is None:
                # Buffer pas encore plein → incertitude maximale (uniforme)
                return np.array([[0.25, 0.25, 0.25, 0.25]])
            logits = self.model(t)
            proba  = torch.softmax(logits, dim=1).numpy()
            return proba

    def reset_buffer(self):
        """À appeler quand la main disparaît (grace period expirée)."""
        self._buffer.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ══════════════════════════════════════════════════════════════════════════════
def load_data(csv_path: str) -> tuple:
    for candidate in [csv_path, MERGED_CSV]:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            log.info("Dataset : %s (%d lignes)", candidate, len(df))
            break
    else:
        raise FileNotFoundError("Aucun dataset. Lancez pipeline.py d'abord.")

    if "pressure_class" not in df.columns:
        if "pressure" in df.columns:
            df["pressure_class"] = pd.cut(
                df["pressure"], bins=[-1,0,2,7,10], labels=[0,1,2,3]
            ).astype(int)
        else:
            raise ValueError("Colonne cible introuvable.")

    avail = [c for c in LM_COLS + VEL_COLS if c in df.columns]
    X = df[avail].fillna(0).values
    y = df["pressure_class"].values
    log.info("Features utilisées : %d", len(avail))
    return X, y


def augment_train(X: np.ndarray, y: np.ndarray) -> tuple:
    """Augmentation spatiale sur le train uniquement (jamais sur le test)."""
    n_lm = 63
    parts = [X]
    labels = [y]
    for _ in range(2):
        Xn = X.copy()
        Xn += np.random.normal(0, 0.004, Xn.shape)
        scale = np.random.uniform(0.88, 1.12, (len(X), 1))
        Xn[:, :n_lm] *= scale
        tx = np.random.uniform(-0.05, 0.05, (len(X), 1))
        ty = np.random.uniform(-0.05, 0.05, (len(X), 1))
        Xn[:, list(range(0, n_lm, 3))] += tx
        Xn[:, list(range(1, n_lm, 3))] += ty
        parts.append(Xn); labels.append(y)
    return np.vstack(parts), np.concatenate(labels)


def plot_history(train_losses, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(val_losses,   label="Val loss")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(val_accs, label="Val accuracy", color="green")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    out = os.path.join(LOG_DIR, "lstm_training.png")
    plt.savefig(out, dpi=120); plt.close()
    log.info("Courbe d'entraînement → %s", out)


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(PRESSURE_CLASSES.values()))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matrice de confusion — LSTM")
    plt.tight_layout()
    out = os.path.join(LOG_DIR, "confusion_lstm.png")
    plt.savefig(out, dpi=120); plt.close()
    log.info("Confusion → %s", out)


# ══════════════════════════════════════════════════════════════════════════════
# Boucle d'entraînement
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilité
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════
def train(csv_path=CLEAN_CSV, model_out=DEEP_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)

    X, y = load_data(csv_path)

    # 1. Split AVANT augmentation (évite le data leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # 2. Augmentation sur train uniquement
    X_tr_aug, y_tr_aug = augment_train(X_tr, y_tr)
    log.info("Train original=%d | augmenté=%d | Test=%d", len(X_tr), len(X_tr_aug), len(X_te))

    # 3. Normalisation (fit sur train uniquement)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_aug)
    X_te_scaled = scaler.transform(X_te)

    # 4. Datasets et DataLoaders
    train_ds = HandSequenceDataset(X_tr_scaled, y_tr_aug, SEQ_LEN)
    test_ds  = HandSequenceDataset(X_te_scaled, y_te,     SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log.info("Séquences train=%d | test=%d", len(train_ds), len(test_ds))

    # 5. Modèle, loss, optimiseur
    model     = HandLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    log.info("Paramètres du modèle : %d", sum(p.numel() for p in model.parameters()))

    # 6. Boucle d'entraînement avec early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc               = train_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, preds, labels = eval_epoch(model, test_dl,  criterion, device)
        scheduler.step(va_loss)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        log.info("Epoch %2d/%d | tr_loss=%.4f tr_acc=%.3f | va_loss=%.4f va_acc=%.3f | %.1fs",
                 epoch, EPOCHS, tr_loss, tr_acc, va_loss, va_acc, time.time()-t0)

        # Early stopping
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "_best_lstm.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping à l'epoch %d", epoch)
                break

    # 7. Charger le meilleur checkpoint
    model.load_state_dict(torch.load(os.path.join(LOG_DIR, "_best_lstm.pt")))

    # 8. Rapport final
    _, final_acc, final_preds, final_labels = eval_epoch(model, test_dl, criterion, device)
    log.info("\n★ Accuracy finale : %.4f", final_acc)
    print("\n" + classification_report(final_labels, final_preds,
          target_names=list(PRESSURE_CLASSES.values())))

    plot_history(train_losses, val_losses, val_accs)
    plot_confusion(final_labels, final_preds)

    # 9. Sauvegarde wrapper sklearn-compatible
    wrapper = LSTMWrapper(model, scaler, SEQ_LEN)
    joblib.dump(wrapper, model_out)
    log.info("Modèle LSTM sauvegardé → %s", model_out)
    log.info("Pour l'utiliser dans robot_autonome.py, changez MODEL_PATH dans config.py")


if __name__ == "__main__":
    train()
