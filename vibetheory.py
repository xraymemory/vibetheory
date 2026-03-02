"""
vibetheory.py — A Mathematical Model of Vibe
=============================================

Based on Peli Grietzer's "A Theory of Vibe" (Glass Bead, 2017) and his
December 2025 update to the theory.

Theory of Vibe '17 (VAE model):
  An autoencoder trained on a domain learns a compressed generative vocabulary
  whose "canon" (the set of all inputs with zero reconstruction error) is the
  idealized aesthetic unity of that domain — its vibe. A vibe is "an abstractum
  inseparable from its concreta." The meaning of an item is its position in a
  global latent space of V-possible items.

Theory of Vibe '25 (diffusion/manifold model):
  VAEs fail on "surface-diverse but vibe-coherent" datasets. Diffusion models
  learn an implicit lower-dimensional manifold in input space without a global
  Euclidean chart. This manifold is semantically meaningful *locally*: through
  local tangent spaces, local PCA, and neighborhood-relative feature vectors.
  Vibe '25: a vibe enables the construction of "meaningful bases of comparison
  in neighborhoods of V-possible items" — it is an enabling condition for
  determinate cognition that itself resists global determinate expression.

This module implements both models:

  Mode 'vae' (2017):
    Text -> TF-IDF -> bottleneck autoencoder -> global latent space -> canon
    -> global aesthetic unity, comparability, and reconstruction error metrics.

  Mode 'diffusion' (2025):
    Text -> TF-IDF -> score-based denoising network learns implicit manifold
    -> local tangent spaces via Jacobian of the learned score function
    -> neighborhood comparability in tangent-projected space
    -> manifold metrics: on-manifold fraction, local comparability,
       tangent spectrum, anomaly detection via score norms.
    NOTE: All equations are our operationalization of Grietzer's
    conceptual framework. Grietzer '25 contains no formal math.

Usage:
    python vibetheory.py "path/to/text.txt"
    python vibetheory.py --mode diffusion "path/to/text.txt"
    python vibetheory.py --text "Paste text directly here..."
    python vibetheory.py --compare "path1.txt" "path2.txt"
    python vibetheory.py --demo
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# 1. TEXT PREPROCESSING — converting text into "phenomena" (sliding windows)
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-literary noise."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^\w\s\'".,;:!?\-—()]', "", text)
    return text.strip()


def extract_sentences(text: str) -> list[str]:
    """Split text into sentence-level units (the atomic 'phenomena')."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw if len(s.strip().split()) >= 3]
    return sentences


def create_windows(
    sentences: list[str], window_size: int = 3, stride: int = 1
) -> list[str]:
    """
    Create overlapping windows of sentences.

    Each window is a "phenomenon" in Grietzer's sense — a local textual
    unit that the autoencoder will learn to reconstruct. The window
    captures both local texture and micro-structural relationships.
    """
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
    windows = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        window = " ".join(sentences[i : i + window_size])
        windows.append(window)
    return windows


def text_to_phenomena(text: str, window_size: int = 3, stride: int = 1) -> list[str]:
    """Full pipeline: raw text -> list of textual phenomena."""
    text = clean_text(text)
    sentences = extract_sentences(text)
    if len(sentences) < window_size:
        return sentences if sentences else [text]
    return create_windows(sentences, window_size, stride)


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION — TF-IDF vectorization of phenomena
# ---------------------------------------------------------------------------


@dataclass
class PhenomenaMatrix:
    """The input-space representation of a text's phenomena."""

    phenomena: list[str]
    vectors: np.ndarray  # shape (n_phenomena, n_features)
    feature_names: list[str]
    scaler: StandardScaler
    vectorizer: TfidfVectorizer


def vectorize_phenomena(
    phenomena: list[str],
    max_features: int = 512,
    min_df: int = 1,
    max_df: float = 0.95,
) -> PhenomenaMatrix:
    """
    Convert textual phenomena into TF-IDF vectors.

    This is the "input space" — the high-dimensional representation of
    worldly phenomena before compression into the lower-dimensional
    "feature space" (the respects of variation).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(phenomena).toarray().astype(np.float32)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(tfidf_matrix)

    return PhenomenaMatrix(
        phenomena=phenomena,
        vectors=scaled,
        feature_names=list(vectorizer.get_feature_names_out()),
        scaler=scaler,
        vectorizer=vectorizer,
    )


# ===========================================================================
#
#  MODEL A: THEORY OF VIBE '17 — VAE / BOTTLENECK AUTOENCODER
#
# ===========================================================================

# ---------------------------------------------------------------------------
# 3A. THE LITERARY AUTOENCODER — learning the "respects of variation"
# ---------------------------------------------------------------------------


class LiteraryAutoencoder(nn.Module):
    """
    A neural autoencoder whose bottleneck learns Grietzer's
    "respects of variation" — the limited generative vocabulary
    that defines the vibe.

    Architecture:
        input_dim -> encoder hidden -> latent_dim -> decoder hidden -> input_dim

    The latent_dim is deliberately small relative to input_dim,
    forcing the network to discover the "handful of respects of
    variation" that best compress the training phenomena.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Feature function (encoder): phenomena -> summary
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh(),  # Bound the "feature values" to [-1, 1]
        )

        # Decoder function: summary -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Feature function: produce 'summaries' (latent codes)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder function: reconstruct from summaries."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full autoencoding: x -> z -> x_hat."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def train_autoencoder(
    data: np.ndarray,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 50,
    verbose: bool = True,
) -> tuple[LiteraryAutoencoder, list[float]]:
    """
    Train the literary autoencoder on the phenomena matrix.

    Returns the trained model and the loss history.
    """
    input_dim = data.shape[1]
    n_samples = data.shape[0]

    # Scale latent dim relative to data size, but respect the user's choice
    effective_latent = min(latent_dim, max(4, n_samples // 4), input_dim // 4)
    effective_hidden = min(hidden_dim, input_dim)

    model = LiteraryAutoencoder(input_dim, effective_latent, effective_hidden)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, n_samples),
        shuffle=True,
        drop_last=False,
    )

    loss_history = []
    best_loss = float("inf")
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            x_hat, z = model(batch)
            recon_loss = nn.functional.mse_loss(x_hat, batch)
            sparse_loss = 0.01 * z.abs().mean()
            loss = recon_loss + sparse_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += recon_loss.item() * batch.size(0)

        scheduler.step()
        epoch_loss /= n_samples
        loss_history.append(epoch_loss)

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, loss_history


# ---------------------------------------------------------------------------
# 4A. CANON DERIVATION & RECONSTRUCTION ERROR (VAE '17)
# ---------------------------------------------------------------------------


@dataclass
class CanonAnalysis:
    """
    Results of deriving the autoencoder's canon from the text.

    Per Grietzer '17: "The canon is the set of all the objects that a given
    trained autoencoder can imagine or conceive of whole, without
    approximation or simplification."
    """

    reconstruction_errors: np.ndarray
    latent_codes: np.ndarray
    reconstructions: np.ndarray
    canon_mask: np.ndarray
    canon_threshold: float
    canon_indices: list[int]
    non_canon_indices: list[int]


def derive_canon(
    model: LiteraryAutoencoder,
    data: np.ndarray,
    canon_percentile: float = 25.0,
) -> CanonAnalysis:
    """
    Derive the canon: the subset of phenomena the autoencoder reconstructs
    with minimal error.
    """
    tensor_data = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        reconstructions, latent_codes = model(tensor_data)

    recon_np = reconstructions.numpy()
    latent_np = latent_codes.numpy()

    errors = np.sqrt(np.sum((data - recon_np) ** 2, axis=1))
    threshold = np.percentile(errors, canon_percentile)
    canon_mask = errors <= threshold

    return CanonAnalysis(
        reconstruction_errors=errors,
        latent_codes=latent_np,
        reconstructions=recon_np,
        canon_mask=canon_mask,
        canon_threshold=threshold,
        canon_indices=list(np.where(canon_mask)[0]),
        non_canon_indices=list(np.where(~canon_mask)[0]),
    )


# ---------------------------------------------------------------------------
# 5A. VIBE METRICS — VAE '17 (global)
# ---------------------------------------------------------------------------


@dataclass
class VibeMetrics:
    """
    Quantitative measures of the text's vibe (Theory of Vibe '17).

    These are GLOBAL metrics: every item gets meaning from its position
    in a single Euclidean latent chart.
    """

    # Core metrics
    vibe_density: float
    aesthetic_unity: float
    mean_reconstruction_error: float
    canon_coherence: float
    complexity_ratio: float

    # Feature analysis
    feature_activations: np.ndarray
    dominant_features: list[int]
    feature_variance: np.ndarray

    # Comparability (section 18): "sameness of difference"
    canon_comparability: float
    full_comparability: float

    # Excess of reality
    excess_of_reality: float
    vibe_strength: float


def compute_vibe_metrics(
    canon: CanonAnalysis,
    data: np.ndarray,
    model: LiteraryAutoencoder,
) -> VibeMetrics:
    """Compute the full suite of VAE '17 vibe metrics."""
    latent = canon.latent_codes
    errors = canon.reconstruction_errors
    mask = canon.canon_mask

    n_total = len(errors)
    n_canon = mask.sum()

    vibe_density = n_canon / n_total if n_total > 0 else 0.0

    # Aesthetic unity: global latent-space cosine similarity within canon
    if n_canon > 1:
        canon_latent = latent[mask]
        norms = np.linalg.norm(canon_latent, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = canon_latent / norms
        cos_sim = normed @ normed.T
        triu_idx = np.triu_indices(n_canon, k=1)
        aesthetic_unity = float(np.mean(cos_sim[triu_idx]))
    else:
        aesthetic_unity = 1.0

    if n_total > 1:
        norms_all = np.linalg.norm(latent, axis=1, keepdims=True)
        norms_all = np.clip(norms_all, 1e-8, None)
        normed_all = latent / norms_all
        cos_sim_all = normed_all @ normed_all.T
        triu_all = np.triu_indices(n_total, k=1)
        full_comparability = float(np.mean(cos_sim_all[triu_all]))
    else:
        full_comparability = 1.0

    mean_error = float(np.mean(errors))
    non_canon_errors = errors[~mask]
    excess_of_reality = (
        float(np.mean(non_canon_errors)) if len(non_canon_errors) > 0 else 0.0
    )

    canon_coherence = aesthetic_unity
    vibe_strength = (
        (canon_coherence - full_comparability) if full_comparability != 0 else 0.0
    )

    complexity_ratio = model.latent_dim / model.input_dim

    feature_activations = (
        np.mean(np.abs(latent[mask]), axis=0)
        if n_canon > 0
        else np.zeros(latent.shape[1])
    )
    feature_variance = (
        np.var(latent[mask], axis=0) if n_canon > 0 else np.zeros(latent.shape[1])
    )
    dominant = list(np.argsort(-feature_activations))

    return VibeMetrics(
        vibe_density=vibe_density,
        aesthetic_unity=aesthetic_unity,
        mean_reconstruction_error=mean_error,
        canon_coherence=canon_coherence,
        complexity_ratio=complexity_ratio,
        feature_activations=feature_activations,
        dominant_features=dominant,
        feature_variance=feature_variance,
        canon_comparability=aesthetic_unity,
        full_comparability=full_comparability,
        excess_of_reality=excess_of_reality,
        vibe_strength=vibe_strength,
    )


# ===========================================================================
#
#  MODEL B: THEORY OF VIBE '25 — DIFFUSION / IMPLICIT MANIFOLD
#
#  Grietzer '25: "Diffusion models learn an implicit lower-dimensional
#  manifold in input space but don't learn a lower-dimensional Euclidean
#  chart for it... you can use the internals of diffusion models to learn
#  local lower-dimensional Euclidean charts covering the implicit
#  lower-dimensional manifold in input space, and extract good 'local'
#  semantic feature vectors for the neighborhood of each data point."
#
#  Instead of global meaning (position in a single chart), items get
#  meaning locally: through "meaningful bases of comparison in
#  neighborhoods of V-possible items."
#
# ===========================================================================

# ---------------------------------------------------------------------------
# 3B. SCORE NETWORK — learns the score function (gradient of log-density)
# ---------------------------------------------------------------------------


class ScoreNetwork(nn.Module):
    """
    A time-conditional score network that learns the score function
    (gradient of log probability density) of the data distribution.

    In a diffusion model, the score function implicitly defines a
    lower-dimensional manifold: data concentrates where the score is
    small, and the score's Jacobian encodes local geometric structure.

    Grietzer '25 argues (conceptually, without equations) that diffusion
    models learn "an implicit lower-dimensional manifold in input space"
    and that meaning is extracted locally. This network is our
    operationalization of that idea using standard denoising score
    matching (Song & Ermon, 2019).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, time_embed_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim

        # Time embedding: sinusoidal encoding of noise level
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Score prediction network: (x, t) -> score = grad log p_t(x)
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the score at noise level t.

        Args:
            x: noisy data, shape (batch, input_dim)
            t: noise level, shape (batch, 1)

        Returns:
            Predicted score (gradient of log density), shape (batch, input_dim)
        """
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


def train_score_network(
    data: np.ndarray,
    hidden_dim: int = 128,
    epochs: int = 600,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 60,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    verbose: bool = True,
) -> tuple[ScoreNetwork, list[float]]:
    """
    Train the score network via denoising score matching.

    The training objective is: for noise level sigma, given x_0 ~ data and
    x_t = x_0 + sigma * eps (eps ~ N(0,I)), predict the score
    grad_x log p_sigma(x_t) = -(x_t - x_0) / sigma^2.

    This is equivalent to the standard denoising score matching loss.
    """
    input_dim = data.shape[1]
    n_samples = data.shape[0]
    effective_hidden = min(hidden_dim, input_dim * 2)

    model = ScoreNetwork(input_dim, effective_hidden)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, n_samples),
        shuffle=True,
        drop_last=False,
    )

    loss_history = []
    best_loss = float("inf")
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            bs = batch.shape[0]

            # Sample noise levels uniformly in log space
            log_sigma = torch.rand(bs, 1) * (
                math.log(sigma_max) - math.log(sigma_min)
            ) + math.log(sigma_min)
            sigma = log_sigma.exp()

            # Add noise: x_t = x_0 + sigma * eps
            eps = torch.randn_like(batch)
            x_noisy = batch + sigma * eps

            # Target score: -(x_t - x_0) / sigma^2 = -eps / sigma
            target_score = -eps / sigma

            # Predict score
            pred_score = model(x_noisy, sigma)

            # Weighted score matching loss (weight by sigma^2 for stability)
            loss = (sigma**2 * (pred_score - target_score) ** 2).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * bs

        scheduler.step()
        epoch_loss /= n_samples
        loss_history.append(epoch_loss)

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, loss_history


# ---------------------------------------------------------------------------
# 4B. MANIFOLD ANALYSIS — local tangent spaces, local PCA, neighborhoods
# ---------------------------------------------------------------------------


@dataclass
class LocalNeighborhood:
    """
    Local semantic structure around a single phenomenon.

    Per Grietzer '25: meaning is local — each data point gets local
    semantic structure through its neighborhood, not a position in a
    global chart. The tangent space computation here is our own
    operationalization (not Grietzer's — his essay contains no equations).
    """

    index: int
    # Local tangent space from score Jacobian (our construction; see note in README)
    tangent_vectors: np.ndarray  # principal tangent directions (k, input_dim)
    tangent_singular_values: np.ndarray  # strength of each tangent direction
    # Score-based anomaly measure (our construction)
    score_norm: float  # ||score(x, sigma_min)|| — low = on-manifold
    # Neighborhood structure
    neighbor_indices: np.ndarray  # indices of k nearest neighbors
    neighbor_distances: np.ndarray  # distances to neighbors
    # Local comparability: cosine similarity in local tangent projection
    local_comparability: float  # mean pairwise similarity among neighbors


@dataclass
class ManifoldAnalysis:
    """
    Full manifold analysis of the text (Theory of Vibe '25).

    Instead of a single global canon, we have:
    - Per-phenomenon local neighborhoods with local tangent spaces
    - Anomaly scores (how well each phenomenon fits the manifold)
    - Local comparability (how meaningful local comparisons are)

    NOTE: The tangent space computation and anomaly detection are our
    operationalization of Grietzer's conceptual framework. Grietzer '25
    contains no equations — see README Section 5 for provenance notes.
    """

    # Global manifold properties
    mean_score_norm: float  # mean anomaly score
    score_norms: np.ndarray  # per-phenomenon score norms

    # Manifold membership (analogous to canon, but softer)
    on_manifold_mask: np.ndarray  # boolean: True if on the learned manifold
    on_manifold_threshold: float  # score norm threshold

    # Local neighborhoods
    neighborhoods: list[LocalNeighborhood]

    # Aggregate local metrics
    mean_local_comparability: float  # mean local comparability across all points
    local_comparability_std: float  # how much local comparability varies

    # Feature structure
    global_tangent_spectrum: np.ndarray  # singular values of global tangent structure


def compute_score_jacobian(
    model: ScoreNetwork,
    x: torch.Tensor,
    sigma: float = 0.05,
) -> np.ndarray:
    """
    Compute the Jacobian of the score function at a data point.

    NOTE ON PROVENANCE: This is our own construction. Park et al. '23
    compute the Jacobian of an intermediate U-Net activation (encoder
    Jacobian for a pullback metric), not the score Jacobian. We use the
    score Jacobian because our small MLP score network has no intermediate
    activations to extract. The score Jacobian (= Hessian of log p) does
    encode local geometric information, but is a different mathematical
    object from Park et al.'s encoder Jacobian.

    Wang et al. '24 show (under MoLRG assumptions) that the DAE Jacobian
    is low-rank, with rank matching intrinsic dimension. We apply this
    intuition to the score Jacobian as an approximation.
    """
    x_in = x.clone().requires_grad_(True)
    sigma_t = torch.full((x_in.shape[0], 1), sigma)

    # Compute Jacobian via batched autograd
    score = model(x_in, sigma_t)
    d = x_in.shape[1]

    jacobian_rows = []
    for j in range(d):
        grad_outputs = torch.zeros_like(score)
        grad_outputs[:, j] = 1.0
        grad = torch.autograd.grad(
            score,
            x_in,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=(j < d - 1),
        )[0]
        jacobian_rows.append(grad.detach().numpy())

    # Shape: (d, batch, d) -> for single point: (d, d)
    jacobian = np.stack(jacobian_rows, axis=0)
    if jacobian.ndim == 3:
        jacobian = jacobian[:, 0, :]  # take first (only) batch element

    return jacobian


def analyze_manifold(
    model: ScoreNetwork,
    data: np.ndarray,
    k_neighbors: int = 5,
    sigma_probe: float = 0.05,
    n_tangent_dims: int = 8,
    manifold_percentile: float = 75.0,
    verbose: bool = True,
) -> ManifoldAnalysis:
    """
    Perform the full manifold analysis (Theory of Vibe '25).

    For each phenomenon:
      1. Compute score norm (anomaly measure: high = off-manifold)
      2. Compute Jacobian of score -> SVD -> local tangent space
      3. Find k nearest neighbors in input space
      4. Project neighbors into local tangent space -> local comparability
    """
    n_samples = data.shape[0]
    input_dim = data.shape[1]
    k_neighbors = min(k_neighbors, n_samples - 1)
    n_tangent_dims = min(n_tangent_dims, input_dim)

    # --- Score norms (anomaly detection) ---
    if verbose:
        print("    Computing score norms (anomaly detection)...")
    tensor_data = torch.tensor(data, dtype=torch.float32)
    sigma_t = torch.full((n_samples, 1), sigma_probe)
    with torch.no_grad():
        scores = model(tensor_data, sigma_t).numpy()
    score_norms = np.linalg.norm(scores, axis=1)

    # On-manifold threshold
    on_manifold_threshold = np.percentile(score_norms, manifold_percentile)
    on_manifold_mask = score_norms <= on_manifold_threshold

    # --- Nearest neighbors ---
    if verbose:
        print("    Building neighborhood graph...")
    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn_model.fit(data)
    distances, indices = nn_model.kneighbors(data)
    # Remove self-neighbor
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    # --- Local tangent spaces via score Jacobian ---
    if verbose:
        print("    Computing local tangent spaces (score Jacobian)...")

    neighborhoods = []
    all_local_comparabilities = []
    all_tangent_svs = []

    for i in range(n_samples):
        x_i = tensor_data[i : i + 1]

        # Compute Jacobian of score at this point
        jacobian = compute_score_jacobian(model, x_i, sigma=sigma_probe)

        # SVD of Jacobian -> tangent structure
        U, S, Vt = np.linalg.svd(jacobian, full_matrices=False)
        S = np.abs(S)

        # Top tangent vectors and their singular values
        top_tangent = Vt[:n_tangent_dims]
        top_svs = S[:n_tangent_dims]
        all_tangent_svs.append(top_svs)

        # --- Local comparability ---
        # Project this point's neighbors into the local tangent space
        # and measure how comparable they are in that space.
        # Per Grietzer '25: a vibe "enables the construction of meaningful
        # bases of comparison in neighborhoods of V-possible items."
        nbrs = neighbor_indices[i]
        nbr_vecs = data[nbrs] - data[i]  # relative vectors

        # Project into tangent space (using all n_tangent_dims directions)
        if n_tangent_dims > 0 and len(nbrs) > 1:
            tangent_basis = top_tangent
            projected = nbr_vecs @ tangent_basis.T  # (k, n_tangent_dims)

            # Cosine similarity among projected neighbors
            proj_norms = np.linalg.norm(projected, axis=1, keepdims=True)
            proj_norms = np.clip(proj_norms, 1e-8, None)
            proj_normed = projected / proj_norms

            if len(proj_normed) > 1:
                cos_sim = proj_normed @ proj_normed.T
                triu = np.triu_indices(len(proj_normed), k=1)
                local_comp = float(np.mean(np.abs(cos_sim[triu])))
            else:
                local_comp = 1.0
        else:
            local_comp = 0.0

        all_local_comparabilities.append(local_comp)

        neighborhoods.append(
            LocalNeighborhood(
                index=i,
                tangent_vectors=top_tangent,
                tangent_singular_values=top_svs,
                score_norm=float(score_norms[i]),
                neighbor_indices=nbrs,
                neighbor_distances=neighbor_distances[i],
                local_comparability=local_comp,
            )
        )

    # --- Aggregate ---
    local_comps = np.array(all_local_comparabilities)

    # Global tangent spectrum: average singular values across all points
    max_sv_len = max(len(sv) for sv in all_tangent_svs)
    padded_svs = np.zeros((len(all_tangent_svs), max_sv_len))
    for i, sv in enumerate(all_tangent_svs):
        padded_svs[i, : len(sv)] = sv
    global_tangent_spectrum = np.mean(padded_svs, axis=0)

    return ManifoldAnalysis(
        mean_score_norm=float(np.mean(score_norms)),
        score_norms=score_norms,
        on_manifold_mask=on_manifold_mask,
        on_manifold_threshold=on_manifold_threshold,
        neighborhoods=neighborhoods,
        mean_local_comparability=float(np.mean(local_comps)),
        local_comparability_std=float(np.std(local_comps)),
        global_tangent_spectrum=global_tangent_spectrum,
    )


# ---------------------------------------------------------------------------
# 5B. VIBE METRICS — DIFFUSION '25 (local)
# ---------------------------------------------------------------------------


@dataclass
class VibeMetrics25:
    """
    Quantitative measures of the text's vibe (Theory of Vibe '25).

    These are LOCAL metrics: meaning comes from "meaningful bases of
    comparison in neighborhoods of V-possible items," not from a
    global Euclidean chart.

    Per Grietzer '25: "Instead of a vibe V directly grounding meaningful
    global comparisons between any two V-possible items, V enables the
    construction of meaningful bases of comparison in neighborhoods of
    V-possible items."

    NOTE: All metrics here are our operationalization. Grietzer '25
    contains no equations or formal metric definitions.
    """

    # Vibe as manifold membership
    on_manifold_fraction: float  # fraction of phenomena on the manifold
    mean_anomaly_score: float  # mean score norm (lower = more on-manifold)

    # Local meaning (the '25 innovation)
    mean_local_comparability: float  # how meaningful local comparisons are
    local_comparability_variability: float  # uniformity of local meaning

    # Tangent spectrum (generative vocabulary, but local)
    tangent_spectrum: np.ndarray  # average singular values of local tangent spaces

    # Global comparison (for reference)
    global_comparability: float  # classic pairwise cosine sim in input space


def compute_vibe_metrics_25(
    manifold: ManifoldAnalysis,
    data: np.ndarray,
) -> VibeMetrics25:
    """Compute the full suite of Theory of Vibe '25 metrics."""
    n_total = data.shape[0]

    # On-manifold fraction (analog of vibe density)
    on_manifold_fraction = float(manifold.on_manifold_mask.sum()) / n_total

    # Local comparability
    mean_local_comp = manifold.mean_local_comparability
    local_comp_var = manifold.local_comparability_std

    # Global comparability (in input space, for reference)
    if n_total > 1:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = data / norms
        cos_sim = normed @ normed.T
        triu = np.triu_indices(n_total, k=1)
        global_comp = float(np.mean(cos_sim[triu]))
    else:
        global_comp = 1.0

    # Tangent spectrum
    spectrum = manifold.global_tangent_spectrum

    return VibeMetrics25(
        on_manifold_fraction=on_manifold_fraction,
        mean_anomaly_score=manifold.mean_score_norm,
        mean_local_comparability=mean_local_comp,
        local_comparability_variability=local_comp_var,
        tangent_spectrum=spectrum,
        global_comparability=global_comp,
    )


# ===========================================================================
#
#  REPORTS & VISUALIZATION (both models)
#
# ===========================================================================

# ---------------------------------------------------------------------------
# 6A. VIBE REPORT — VAE '17
# ---------------------------------------------------------------------------


def format_vibe_report_17(
    metrics: VibeMetrics,
    canon: CanonAnalysis,
    phenomena: PhenomenaMatrix,
    title: str = "Unknown Text",
) -> str:
    """Generate a human-readable vibe report (Theory of Vibe '17)."""
    n_total = len(canon.reconstruction_errors)
    n_canon = len(canon.canon_indices)
    n_latent = canon.latent_codes.shape[1]

    lines = []
    lines.append("=" * 70)
    lines.append(f"  VIBE ANALYSIS: {title}")
    lines.append("=" * 70)
    lines.append("  Theory of Vibe '17 (VAE / global latent chart)")
    lines.append("  Based on Grietzer, 'A Theory of Vibe' (Glass Bead, 2017)")
    lines.append("-" * 70)

    lines.append("")
    lines.append("  MODEL STRUCTURE")
    lines.append(f"    Phenomena (textual windows):      {n_total}")
    lines.append(f"    Input dimensions (TF-IDF):        {phenomena.vectors.shape[1]}")
    lines.append(f"    Latent dimensions (respects of variation): {n_latent}")
    lines.append(
        f"    Compression ratio:                {metrics.complexity_ratio:.4f}"
    )

    lines.append("")
    lines.append("  VIBE METRICS (global)")
    lines.append(f"    Vibe Density:                     {metrics.vibe_density:.3f}")
    lines.append(f"      (fraction of text in the canon — the 'dense vibe')")
    lines.append(
        f"    Aesthetic Unity (canon):           {metrics.aesthetic_unity:.3f}"
    )
    lines.append(f"      (inter-comparability of canonical phenomena)")
    lines.append(
        f"    Full-Text Comparability:           {metrics.full_comparability:.3f}"
    )
    lines.append(f"      (inter-comparability of all phenomena)")
    lines.append(f"    Vibe Strength:                    {metrics.vibe_strength:+.3f}")
    lines.append(f"      (canon coherence minus full-text comparability)")
    lines.append(
        f"    Mean Reconstruction Error:        {metrics.mean_reconstruction_error:.4f}"
    )
    lines.append(
        f"    Excess of Reality:                {metrics.excess_of_reality:.4f}"
    )
    lines.append(f"      (mean error on non-canonical phenomena)")

    lines.append("")
    lines.append("  RESPECTS OF VARIATION (latent feature signature)")
    lines.append(f"    Active latent dimensions: {n_latent}")
    top_n = min(8, n_latent)
    lines.append(f"    Top {top_n} features by mean activation:")
    for idx in metrics.dominant_features[:top_n]:
        act = metrics.feature_activations[idx]
        var = metrics.feature_variance[idx]
        lines.append(
            f"      Feature {idx:2d}:  activation={act:.3f}  variance={var:.3f}"
        )

    lines.append("")
    lines.append("  CANON (most vibe-conforming phenomena)")
    lines.append(f"    Canon size: {n_canon} / {n_total} phenomena")
    lines.append(f"    Canon threshold (max error): {canon.canon_threshold:.4f}")
    n_show = min(5, n_canon)
    if n_show > 0:
        lines.append(f"    Sample canonical phenomena (best {n_show}):")
        sorted_canon = sorted(
            canon.canon_indices, key=lambda i: canon.reconstruction_errors[i]
        )
        for rank, idx in enumerate(sorted_canon[:n_show]):
            err = canon.reconstruction_errors[idx]
            snippet = phenomena.phenomena[idx][:120]
            lines.append(f"      [{rank + 1}] (err={err:.4f}) {snippet}...")

    lines.append("")
    lines.append("  EXCESS OF REALITY (least vibe-conforming phenomena)")
    n_excess_show = min(3, len(canon.non_canon_indices))
    if n_excess_show > 0:
        sorted_non = sorted(
            canon.non_canon_indices, key=lambda i: -canon.reconstruction_errors[i]
        )
        lines.append(f"    Highest-error phenomena (top {n_excess_show}):")
        for rank, idx in enumerate(sorted_non[:n_excess_show]):
            err = canon.reconstruction_errors[idx]
            snippet = phenomena.phenomena[idx][:120]
            lines.append(f"      [{rank + 1}] (err={err:.4f}) {snippet}...")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  INTERPRETATION (per Grietzer '17)")
    lines.append("")

    if metrics.aesthetic_unity > 0.5:
        lines.append("  HIGH AESTHETIC UNITY: This text has a strong, coherent vibe.")
        lines.append(
            "  Its phenomena are tightly bound by a limited generative vocabulary."
        )
        lines.append(
            "  Like a 64k Intro: 'individually complex but collectively simple.'"
        )
    elif metrics.aesthetic_unity > 0.2:
        lines.append("  MODERATE AESTHETIC UNITY: The text has a discernible vibe,")
        lines.append("  but with meaningful variation. Some phenomena exceed the")
        lines.append("  generative vocabulary — a productive tension between")
        lines.append("  structure and 'excess of material reality.'")
    else:
        lines.append("  LOW AESTHETIC UNITY: The text's phenomena are diverse and")
        lines.append("  resist compression into a limited generative vocabulary.")
        lines.append("  This may indicate polyvalence, collage, or an expansive")
        lines.append("  'lifeworld' that resists schematization.")

    if metrics.vibe_strength > 0.1:
        lines.append("")
        lines.append("  STRONG VIBE DIFFERENTIATION: The canon is notably more")
        lines.append("  coherent than the text as a whole — a 'dense vibe' emerges")
        lines.append("  that idealizes the text's looser aesthetic unity.")
    elif metrics.vibe_strength < -0.05:
        lines.append("")
        lines.append("  DIFFUSE VIBE: The canon is not much more coherent than")
        lines.append("  the full text — the vibe is ambient and distributed")
        lines.append("  rather than concentrated in a core set of phenomena.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6B. VIBE REPORT — DIFFUSION '25
# ---------------------------------------------------------------------------


def format_vibe_report_25(
    metrics: VibeMetrics25,
    manifold: ManifoldAnalysis,
    phenomena: PhenomenaMatrix,
    title: str = "Unknown Text",
) -> str:
    """Generate a human-readable vibe report (Theory of Vibe '25)."""
    n_total = len(phenomena.phenomena)

    lines = []
    lines.append("=" * 70)
    lines.append(f"  VIBE ANALYSIS: {title}")
    lines.append("=" * 70)
    lines.append("  Theory of Vibe '25 (diffusion / implicit manifold)")
    lines.append("  Based on Grietzer, 'Theory of Vibe' update (Dec 2025)")
    lines.append("  NOTE: All metrics below are our operationalization of")
    lines.append("  Grietzer's conceptual framework (his essay has no equations).")
    lines.append("-" * 70)

    lines.append("")
    lines.append("  IMPLICIT MANIFOLD STRUCTURE")
    lines.append(f"    Phenomena (textual windows):      {n_total}")
    lines.append(f"    Input dimensions (TF-IDF):        {phenomena.vectors.shape[1]}")

    lines.append("")
    lines.append("  MANIFOLD MEMBERSHIP (analog of 'canon')")
    lines.append(
        f"    On-manifold fraction:             {metrics.on_manifold_fraction:.3f}"
    )
    lines.append(f"      (fraction of phenomena lying on the learned manifold)")
    lines.append(
        f"    Mean anomaly score:               {metrics.mean_anomaly_score:.4f}"
    )
    lines.append(f"      (lower = more on-manifold; score norm at sigma_min)")

    lines.append("")
    lines.append("  LOCAL MEANING (the '25 innovation)")
    lines.append(
        f"    Mean local comparability:         {metrics.mean_local_comparability:.3f}"
    )
    lines.append(f"      (how meaningful comparisons are within neighborhoods)")
    lines.append(
        f"    Local comparability variability:  {metrics.local_comparability_variability:.3f}"
    )
    lines.append(f"      (how uniform local meaning is across the manifold)")
    lines.append(
        f"    Global comparability:             {metrics.global_comparability:.3f}"
    )
    lines.append(f"      (pairwise similarity in raw input space, for reference)")

    lines.append("")
    lines.append("  TANGENT SPECTRUM (local generative vocabulary)")
    n_show_spectrum = min(8, len(metrics.tangent_spectrum))
    lines.append(
        f"    Top {n_show_spectrum} tangent singular values (mean across phenomena):"
    )
    for i in range(n_show_spectrum):
        sv = metrics.tangent_spectrum[i]
        lines.append(f"      Tangent dir {i:2d}:  {sv:.4f}")

    # Show on-manifold and off-manifold phenomena
    lines.append("")
    lines.append("  ON-MANIFOLD PHENOMENA (most vibe-conforming)")
    on_indices = np.where(manifold.on_manifold_mask)[0]
    sorted_on = sorted(on_indices, key=lambda i: manifold.score_norms[i])
    n_show = min(5, len(sorted_on))
    if n_show > 0:
        for rank, idx in enumerate(sorted_on[:n_show]):
            score = manifold.score_norms[idx]
            nbhd = manifold.neighborhoods[idx]
            snippet = phenomena.phenomena[idx][:100]
            lines.append(
                f"    [{rank + 1}] (score={score:.3f}, "
                f"local_comp={nbhd.local_comparability:.3f})"
            )
            lines.append(f"        {snippet}...")

    lines.append("")
    lines.append("  OFF-MANIFOLD PHENOMENA (highest anomaly)")
    off_indices = np.where(~manifold.on_manifold_mask)[0]
    sorted_off = sorted(off_indices, key=lambda i: -manifold.score_norms[i])
    n_show_off = min(3, len(sorted_off))
    if n_show_off > 0:
        for rank, idx in enumerate(sorted_off[:n_show_off]):
            score = manifold.score_norms[idx]
            snippet = phenomena.phenomena[idx][:100]
            lines.append(f"    [{rank + 1}] (score={score:.3f}) {snippet}...")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  INTERPRETATION (per Grietzer '25)")
    lines.append("")

    # Manifold quality
    if metrics.on_manifold_fraction > 0.7:
        lines.append("  STRONG MANIFOLD: Most phenomena lie on the learned manifold.")
        lines.append("  The text has a coherent implicit structure — a 'surface' in")
        lines.append("  phenomenon-space that organizes its diversity.")
    elif metrics.on_manifold_fraction > 0.4:
        lines.append("  PARTIAL MANIFOLD: The text has an implicit structure, but with")
        lines.append(
            "  significant phenomena that escape it — productive 'excess of reality.'"
        )
    else:
        lines.append("  WEAK MANIFOLD: The phenomena resist organization onto a single")
        lines.append(
            "  implicit manifold. The text may be polyvalent or surface-diverse"
        )
        lines.append("  beyond the manifold's capacity.")

    # Local comparability interpretation
    lines.append("")
    if metrics.mean_local_comparability > 0.4:
        lines.append("  HIGH LOCAL COMPARABILITY: Neighborhoods on the manifold")
        lines.append("  are semantically structured. Per Grietzer '25, the vibe")
        lines.append("  enables 'meaningful bases of comparison in neighborhoods")
        lines.append("  of V-possible items.'")
    elif metrics.mean_local_comparability > 0.2:
        lines.append("  MODERATE LOCAL COMPARABILITY: Some neighborhood structure")
        lines.append("  is present. The vibe provides partial local semantics.")
    else:
        lines.append("  LOW LOCAL COMPARABILITY: Neighborhoods lack strong")
        lines.append("  internal structure. Local comparison may not be meaningful.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VISUALIZATION (both models)
# ---------------------------------------------------------------------------


def plot_vibe_17(
    metrics: VibeMetrics,
    canon: CanonAnalysis,
    phenomena: PhenomenaMatrix,
    title: str = "Vibe Analysis",
    save_path: Optional[str] = None,
):
    """Visualize the vibe (VAE '17 model)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="darkgrid", palette="muted")
    except ImportError:
        print("  [matplotlib/seaborn not available — skipping plots]")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Vibe '17: {title}", fontsize=14, fontweight="bold")

    # 1. Latent space (PCA projection)
    ax = axes[0, 0]
    latent = canon.latent_codes
    if latent.shape[1] > 2:
        pca = PCA(n_components=2)
        projected = pca.fit_transform(latent)
    else:
        projected = latent[:, :2]
    canon_pts = projected[canon.canon_mask]
    non_canon_pts = projected[~canon.canon_mask]
    ax.scatter(
        non_canon_pts[:, 0],
        non_canon_pts[:, 1],
        c="gray",
        alpha=0.4,
        s=20,
        label="Non-canon",
    )
    ax.scatter(
        canon_pts[:, 0], canon_pts[:, 1], c="crimson", alpha=0.7, s=30, label="Canon"
    )
    ax.set_title("Global Latent Space (respects of variation)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)

    # 2. Reconstruction error distribution
    ax = axes[0, 1]
    errors = canon.reconstruction_errors
    ax.hist(
        errors[canon.canon_mask],
        bins=20,
        alpha=0.7,
        color="crimson",
        label="Canon",
        density=True,
    )
    ax.hist(
        errors[~canon.canon_mask],
        bins=20,
        alpha=0.5,
        color="gray",
        label="Non-canon",
        density=True,
    )
    ax.axvline(
        canon.canon_threshold,
        color="black",
        linestyle="--",
        label=f"Threshold={canon.canon_threshold:.3f}",
    )
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Error (L2)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # 3. Feature activation signature
    ax = axes[1, 0]
    x_feat = np.arange(len(metrics.feature_activations))
    ax.bar(x_feat, metrics.feature_activations, color="steelblue", alpha=0.8)
    ax.set_title("Feature Activation Signature")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Mean |activation|")

    # 4. Summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"Vibe Density:          {metrics.vibe_density:.3f}\n"
        f"Aesthetic Unity:       {metrics.aesthetic_unity:.3f}\n"
        f"Full Comparability:   {metrics.full_comparability:.3f}\n"
        f"Vibe Strength:         {metrics.vibe_strength:+.3f}\n"
        f"Complexity Ratio:     {metrics.complexity_ratio:.4f}\n"
        f"Mean Recon Error:    {metrics.mean_reconstruction_error:.4f}\n"
        f"Excess of Reality:    {metrics.excess_of_reality:.4f}\n"
        f"\nPhenomena: {len(errors)}\n"
        f"Canon size: {canon.canon_mask.sum()}\n"
        f"Latent dims: {canon.latent_codes.shape[1]}"
    )
    ax.text(
        0.1,
        0.5,
        summary,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Vibe '17 Summary")

    plt.tight_layout()
    out_path = save_path or "vibe_analysis_17.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out_path}")
    plt.close()


def plot_vibe_25(
    metrics: VibeMetrics25,
    manifold: ManifoldAnalysis,
    phenomena: PhenomenaMatrix,
    title: str = "Vibe Analysis",
    save_path: Optional[str] = None,
):
    """Visualize the vibe (Diffusion '25 model)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="darkgrid", palette="muted")
    except ImportError:
        print("  [matplotlib/seaborn not available — skipping plots]")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Vibe '25: {title}", fontsize=14, fontweight="bold")

    # 1. Input space colored by anomaly score
    ax = axes[0, 0]
    data = phenomena.vectors
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        projected = pca.fit_transform(data)
    else:
        projected = data[:, :2]
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=manifold.score_norms,
        cmap="RdYlGn_r",
        alpha=0.7,
        s=30,
        edgecolors="black",
        linewidths=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="Score norm (anomaly)")
    # Mark on-manifold points
    on_pts = projected[manifold.on_manifold_mask]
    ax.scatter(
        on_pts[:, 0],
        on_pts[:, 1],
        facecolors="none",
        edgecolors="blue",
        s=60,
        linewidths=1.5,
        label="On-manifold",
    )
    ax.set_title("Implicit Manifold (score norm = anomaly)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)

    # 2. Score norm distribution
    ax = axes[0, 1]
    on_scores = manifold.score_norms[manifold.on_manifold_mask]
    off_scores = manifold.score_norms[~manifold.on_manifold_mask]
    ax.hist(
        on_scores,
        bins=15,
        alpha=0.7,
        color="steelblue",
        label="On-manifold",
        density=True,
    )
    if len(off_scores) > 0:
        ax.hist(
            off_scores,
            bins=15,
            alpha=0.5,
            color="coral",
            label="Off-manifold",
            density=True,
        )
    ax.axvline(
        manifold.on_manifold_threshold,
        color="black",
        linestyle="--",
        label=f"Threshold={manifold.on_manifold_threshold:.3f}",
    )
    ax.set_title("Score Norm Distribution (anomaly)")
    ax.set_xlabel("Score norm")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # 3. Tangent spectrum
    ax = axes[1, 0]
    spectrum = metrics.tangent_spectrum
    x_spec = np.arange(len(spectrum))
    ax.bar(x_spec, spectrum, color="darkorange", alpha=0.8)
    ax.set_title("Tangent Spectrum (local generative vocabulary)")
    ax.set_xlabel("Tangent direction")
    ax.set_ylabel("Mean singular value")

    # 4. Local comparability distribution
    ax = axes[1, 1]
    local_comps = [n.local_comparability for n in manifold.neighborhoods]
    ax.hist(local_comps, bins=15, alpha=0.7, color="seagreen", density=True)
    ax.axvline(
        metrics.mean_local_comparability,
        color="black",
        linestyle="--",
        label=f"Mean={metrics.mean_local_comparability:.3f}",
    )
    ax.axvline(
        abs(metrics.global_comparability),
        color="red",
        linestyle=":",
        label=f"|Global|={abs(metrics.global_comparability):.3f}",
    )
    ax.set_title("Local Comparability Distribution")
    ax.set_xlabel("Local comparability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = save_path or "vibe_analysis_25.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out_path}")
    plt.close()


# ===========================================================================
#
#  COMPARISON (both models)
#
# ===========================================================================


@dataclass
class VibeComparison:
    """Compare the vibes of two texts."""

    title_a: str
    title_b: str
    metrics_a: VibeMetrics
    metrics_b: VibeMetrics
    canon_a: CanonAnalysis
    canon_b: CanonAnalysis
    cross_reconstruction_error_a_on_b: float
    cross_reconstruction_error_b_on_a: float
    vibe_similarity: float


def compare_vibes(
    text_a: str,
    text_b: str,
    title_a: str = "Text A",
    title_b: str = "Text B",
    latent_dim: int = 16,
    verbose: bool = True,
) -> VibeComparison:
    """
    Compare the vibes of two texts by cross-reconstruction (VAE '17).
    """
    if verbose:
        print(f"\n  Analyzing '{title_a}'...")
    vibe_a = analyze_vibe(
        text_a,
        title=title_a,
        latent_dim=latent_dim,
        verbose=verbose,
        plot=False,
        mode="vae",
    )

    if verbose:
        print(f"\n  Analyzing '{title_b}'...")
    vibe_b = analyze_vibe(
        text_b,
        title=title_b,
        latent_dim=latent_dim,
        verbose=verbose,
        plot=False,
        mode="vae",
    )

    # Cross-reconstruction
    phenom_b_in_a_space = (
        vibe_a["phenomena"]
        .vectorizer.transform(vibe_b["phenomena"].phenomena)
        .toarray()
        .astype(np.float32)
    )
    phenom_b_in_a_space = vibe_a["phenomena"].scaler.transform(phenom_b_in_a_space)

    phenom_a_in_b_space = (
        vibe_b["phenomena"]
        .vectorizer.transform(vibe_a["phenomena"].phenomena)
        .toarray()
        .astype(np.float32)
    )
    phenom_a_in_b_space = vibe_b["phenomena"].scaler.transform(phenom_a_in_b_space)

    with torch.no_grad():
        recon_b_by_a, _ = vibe_a["model"](
            torch.tensor(phenom_b_in_a_space, dtype=torch.float32)
        )
        recon_a_by_b, _ = vibe_b["model"](
            torch.tensor(phenom_a_in_b_space, dtype=torch.float32)
        )

    cross_err_a_on_b = float(
        np.sqrt(np.mean((phenom_b_in_a_space - recon_b_by_a.numpy()) ** 2))
    )
    cross_err_b_on_a = float(
        np.sqrt(np.mean((phenom_a_in_b_space - recon_a_by_b.numpy()) ** 2))
    )

    self_err_a = vibe_a["metrics"].mean_reconstruction_error
    self_err_b = vibe_b["metrics"].mean_reconstruction_error
    excess_a = (cross_err_b_on_a - self_err_a) / max(self_err_a, 1e-8)
    excess_b = (cross_err_a_on_b - self_err_b) / max(self_err_b, 1e-8)
    mean_excess = (max(0, excess_a) + max(0, excess_b)) / 2
    vibe_similarity = float(math.exp(-mean_excess))

    return VibeComparison(
        title_a=title_a,
        title_b=title_b,
        metrics_a=vibe_a["metrics"],
        metrics_b=vibe_b["metrics"],
        canon_a=vibe_a["canon"],
        canon_b=vibe_b["canon"],
        cross_reconstruction_error_a_on_b=cross_err_a_on_b,
        cross_reconstruction_error_b_on_a=cross_err_b_on_a,
        vibe_similarity=vibe_similarity,
    )


def format_comparison_report(comp: VibeComparison) -> str:
    """Format a comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  VIBE COMPARISON")
    lines.append("=" * 70)
    lines.append(f"  '{comp.title_a}' vs '{comp.title_b}'")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"  {'Metric':<35} {'A':>12} {'B':>12}")
    lines.append(f"  {'-' * 35} {'-' * 12} {'-' * 12}")
    lines.append(
        f"  {'Aesthetic Unity':<35} "
        f"{comp.metrics_a.aesthetic_unity:>12.3f} "
        f"{comp.metrics_b.aesthetic_unity:>12.3f}"
    )
    lines.append(
        f"  {'Vibe Density':<35} "
        f"{comp.metrics_a.vibe_density:>12.3f} "
        f"{comp.metrics_b.vibe_density:>12.3f}"
    )
    lines.append(
        f"  {'Vibe Strength':<35} "
        f"{comp.metrics_a.vibe_strength:>12.3f} "
        f"{comp.metrics_b.vibe_strength:>12.3f}"
    )
    lines.append(
        f"  {'Mean Reconstruction Error':<35} "
        f"{comp.metrics_a.mean_reconstruction_error:>12.4f} "
        f"{comp.metrics_b.mean_reconstruction_error:>12.4f}"
    )
    lines.append(
        f"  {'Excess of Reality':<35} "
        f"{comp.metrics_a.excess_of_reality:>12.4f} "
        f"{comp.metrics_b.excess_of_reality:>12.4f}"
    )
    lines.append("")
    lines.append("  CROSS-RECONSTRUCTION")
    lines.append(
        f"    A's model reconstructing B:  "
        f"error = {comp.cross_reconstruction_error_a_on_b:.4f}"
    )
    lines.append(
        f"    B's model reconstructing A:  "
        f"error = {comp.cross_reconstruction_error_b_on_a:.4f}"
    )
    lines.append(f"    Vibe Similarity:             {comp.vibe_similarity:.3f}")
    lines.append("")

    if comp.vibe_similarity > 0.7:
        lines.append("  INTERPRETATION: These texts share a strong vibe affinity.")
        lines.append(
            "  Each model can substantially reconstruct the other's phenomena,"
        )
        lines.append("  suggesting overlapping 'respects of variation.'")
    elif comp.vibe_similarity > 0.4:
        lines.append("  INTERPRETATION: These texts share a partial vibe affinity.")
        lines.append("  Some structural overlap in generative vocabulary, but")
        lines.append("  significant phenomena in each are alien to the other's vibe.")
    else:
        lines.append("  INTERPRETATION: These texts have distinct vibes.")
        lines.append("  Each model struggles to reconstruct the other's phenomena —")
        lines.append("  their 'respects of variation' are largely disjoint.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ===========================================================================
#
#  MAIN ANALYSIS PIPELINE
#
# ===========================================================================


def analyze_vibe(
    text: str,
    title: str = "Unknown Text",
    mode: str = "vae",
    window_size: int = 3,
    stride: int = 1,
    max_features: int = 512,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    epochs: int = 500,
    canon_percentile: float = 25.0,
    verbose: bool = True,
    plot: bool = True,
) -> dict:
    """
    Full vibe analysis pipeline.

    Args:
        mode: 'vae' for Theory of Vibe '17, 'diffusion' for '25.

    Returns a dict with all intermediate results.
    """
    mode_label = "VAE '17" if mode == "vae" else "Diffusion '25"
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  VIBE THEORY ({mode_label}) — Analyzing: {title}")
        print(f"{'=' * 70}")

    # Step 1: Text -> phenomena
    if verbose:
        print("\n  [1/5] Extracting phenomena (textual windows)...")
    phenomena_list = text_to_phenomena(text, window_size, stride)
    if verbose:
        print(f"         Found {len(phenomena_list)} phenomena")

    if len(phenomena_list) < 5:
        print("  WARNING: Very few phenomena extracted. Results may be unreliable.")
        print("  Consider providing a longer text for meaningful vibe analysis.")

    # Step 2: Vectorize
    if verbose:
        print("  [2/5] Vectorizing phenomena (TF-IDF)...")
    effective_max = min(max_features, len(phenomena_list) * 3)
    phenomena = vectorize_phenomena(phenomena_list, max_features=effective_max)
    if verbose:
        print(f"         Matrix shape: {phenomena.vectors.shape}")

    if mode == "vae":
        # ---- Theory of Vibe '17 ----
        if verbose:
            print("  [3/5] Training literary autoencoder (VAE '17)...")
        model, loss_history = train_autoencoder(
            phenomena.vectors,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            epochs=epochs,
            verbose=verbose,
        )
        if verbose:
            print(f"         Final loss: {loss_history[-1]:.6f}")
            print(f"         Effective latent dim: {model.latent_dim}")

        if verbose:
            print("  [4/5] Deriving canon...")
        canon = derive_canon(
            model, phenomena.vectors, canon_percentile=canon_percentile
        )
        if verbose:
            print(
                f"         Canon size: "
                f"{len(canon.canon_indices)} / {len(phenomena_list)}"
            )

        if verbose:
            print("  [5/5] Computing vibe metrics (VAE '17)...")
        metrics = compute_vibe_metrics(canon, phenomena.vectors, model)

        report = format_vibe_report_17(metrics, canon, phenomena, title=title)
        if verbose:
            print(report)

        if plot:
            plot_vibe_17(metrics, canon, phenomena, title=title)

        return {
            "model": model,
            "phenomena": phenomena,
            "canon": canon,
            "metrics": metrics,
            "loss_history": loss_history,
            "report": report,
            "mode": "vae",
        }

    else:
        # ---- Theory of Vibe '25 ----
        if verbose:
            print("  [3/5] Training score network (Diffusion '25)...")
        model, loss_history = train_score_network(
            phenomena.vectors,
            hidden_dim=hidden_dim,
            epochs=epochs,
            verbose=verbose,
        )
        if verbose:
            print(f"         Final loss: {loss_history[-1]:.6f}")

        if verbose:
            print("  [4/5] Analyzing implicit manifold...")
        manifold = analyze_manifold(
            model,
            phenomena.vectors,
            manifold_percentile=100.0 - canon_percentile,
            verbose=verbose,
        )
        if verbose:
            print(
                f"         On-manifold: "
                f"{manifold.on_manifold_mask.sum()} / {len(phenomena_list)}"
            )

        if verbose:
            print("  [5/5] Computing vibe metrics (Diffusion '25)...")
        metrics = compute_vibe_metrics_25(manifold, phenomena.vectors)

        report = format_vibe_report_25(metrics, manifold, phenomena, title=title)
        if verbose:
            print(report)

        if plot:
            plot_vibe_25(metrics, manifold, phenomena, title=title)

        return {
            "model": model,
            "phenomena": phenomena,
            "manifold": manifold,
            "metrics": metrics,
            "loss_history": loss_history,
            "report": report,
            "mode": "diffusion",
        }


# ===========================================================================
#
#  CLI
#
# ===========================================================================

SAMPLE_KAFKA = """
Someone must have slandered Josef K., for one morning, without having done anything truly wrong, he was arrested. His landlady's cook, who brought him his breakfast every morning at about eight o'clock, did not come this time. That had never happened before. K. waited a while longer, watching from his pillow the old woman who lived opposite and who was observing him with a curiosity unusual even for her, but then, feeling both put out and hungry, he rang the bell. At once there was a knock at the door and a man he had never seen before in these lodgings entered. He was slim yet solidly built, and he wore a closely fitting black suit, which was furnished with all sorts of pleats, pockets, buckles, and buttons and a belt, like a traveler's outfit, and consequently seemed eminently practical, though one could not be quite sure what actual purpose it served. "Who are you?" asked K., half raising himself in bed. But the man ignored the question, as though his presence required no explanation, and simply said: "You rang?" "Anna should bring me my breakfast," said K., and then silently, by means of close observation, he tried to determine who the man actually was. But the man did not submit to K.'s gaze for very long, and instead turned to the door and opened it slightly in order to say to someone who was evidently standing just behind it: "He says Anna should bring him his breakfast." There was a little laughter in the next room; from the sound of it, it was not clear whether several people were involved. Although the strange man could not have learned anything from it that he did not already know, he said to K., as though making a report: "It can't be done." "This is news to me," said K., springing out of bed and quickly pulling on his trousers. "I want to see who the people next door are, and how Frau Grubach can account for this disturbance." It immediately occurred to him, of course, that he should not have said this out loud and that by doing so he in a way acknowledged the stranger's right to supervise, but it did not seem important at the moment. The stranger understood it that way all the same, for he said: "Wouldn't you rather stay here?" "I have no wish to stay here, nor to be addressed by you, until you've told me who you are." "I meant well," said the stranger, and opened the door of his own accord.
In the next room, which K. entered more slowly than he had intended, things looked at first glance almost exactly as they had the evening before. It was Frau Grubach's living room; in this room, crammed with furniture, covers, china, and photographs, there was perhaps a little more space than usual today; but one did not notice that at first, especially since the main change consisted of the presence of a man who was sitting by the open window with a book, from which he now looked up. "You should have stayed in your room! Didn't Franz tell you that?" "What is it you want, then?" said K., looking back and forth between this new acquaintance and the one called Franz, who had remained standing in the doorway. Through the open window he could again see the old woman, who had moved to the window directly opposite with a truly senile curiosity, so that she could go on seeing everything. "I want to see Frau Grubach--" said K., and made a movement as though tearing himself free from the two men, who were, however, standing some distance away from him, and started to go. "No," said the man by the window, throwing his book down on a little table and getting up. "You can't leave, you are being held." "So it appears," said K. "And why?" he asked. "We are not authorized to tell you that. Go to your room and wait. Proceedings have been initiated against you, and you will be informed of everything in due course."
"""

SAMPLE_STEIN = """
A kind in glass and a cousin, a spectacle and nothing strange a single hurt color and an arrangement in a system to pointing. All this and not ordinary, not unordered in not resembling. The difference is spreading. A carafe that is a blind glass. A kind in glass and a cousin, a spectacle and nothing strange a single hurt color and an arrangement in a system to pointing. Nothing elegant and nothing extraordinary. A long dress and a single thing ordinary, an edging of hair and not all new but nearly a new thing. A candle burnt, a table covered, and a reason for that is in the length of time. Next to this is a table covered and has nothing in it but a space for resting. A case some certain there is a right that has length. This which shows it that it stops a long way is well. There can be no doubt that this is real and yet it is enough for three or four or even more. To handle it is with care and with a use of the thing that is there and yet it is used. A use that is the result of the handling that is there is not any different in the way that it is different. It is not the same thing to have it happen more than once and yet it is the same thing as something.
Glazed glitter is a place where nothing is nothing nothing and yet it is something that is there and here. Nothing is nothing and yet it is this that is true and real. Glazed glitter and the light that comes from outside is not a question of placing in a corner. It is not even a question of placing. The light is there and the arrangement and yet the place is not the same. A sign of no vacation and a time for resting and yet the time that is there is not the same as the time that passes. A color in that, a single color, a large piece of is that is there is there and yet it is not the same. A single color that is the color and yet it is not the same as the color that is there.
"""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Vibe Theory: Mathematical vibe analysis based on Grietzer (2017, 2025)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python vibetheory.py novel.txt                    # VAE '17 (default)
          python vibetheory.py --mode diffusion novel.txt   # Diffusion '25
          python vibetheory.py --text "Some text..."
          python vibetheory.py --compare kafka.txt stein.txt
          python vibetheory.py --demo
        """),
    )
    parser.add_argument("file", nargs="?", help="Path to text file to analyze")
    parser.add_argument("--text", "-t", help="Analyze text provided directly")
    parser.add_argument(
        "--compare",
        "-c",
        nargs=2,
        metavar="FILE",
        help="Compare vibes of two text files",
    )
    parser.add_argument(
        "--demo",
        "-d",
        action="store_true",
        help="Run demo with built-in Kafka and Stein samples",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["vae", "diffusion"],
        default="vae",
        help="Model: 'vae' (Theory of Vibe '17) or 'diffusion' ('25). Default: vae",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Latent dimension size (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Training epochs (default: 500)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sentence window size for phenomena (default: 3)",
    )
    parser.add_argument(
        "--canon-percentile",
        type=float,
        default=25.0,
        help="Percentile threshold for canon/manifold (default: 25)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress training output",
    )

    args = parser.parse_args()

    if args.demo:
        print("\n  Running demo: Kafka vs Stein\n")

        for mode in ["vae", "diffusion"]:
            print(f"\n  {'=' * 60}")
            print(
                f"  MODE: {'Theory of Vibe 17 (VAE)' if mode == 'vae' else 'Theory of Vibe 25 (Diffusion)'}"
            )
            print(f"  {'=' * 60}")

            analyze_vibe(
                SAMPLE_KAFKA,
                title="Kafka - The Trial",
                mode=mode,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                verbose=not args.quiet,
                plot=not args.no_plot,
            )
            analyze_vibe(
                SAMPLE_STEIN,
                title="Stein - Tender Buttons",
                mode=mode,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                verbose=not args.quiet,
                plot=not args.no_plot,
            )

        # Comparison (VAE only for cross-reconstruction)
        comp = compare_vibes(
            SAMPLE_KAFKA,
            SAMPLE_STEIN,
            title_a="Kafka - The Trial",
            title_b="Stein - Tender Buttons",
            latent_dim=args.latent_dim,
            verbose=not args.quiet,
        )
        print(format_comparison_report(comp))
        return

    if args.compare:
        path_a, path_b = args.compare
        text_a = Path(path_a).read_text(encoding="utf-8")
        text_b = Path(path_b).read_text(encoding="utf-8")
        comp = compare_vibes(
            text_a,
            text_b,
            title_a=Path(path_a).stem,
            title_b=Path(path_b).stem,
            latent_dim=args.latent_dim,
            verbose=not args.quiet,
        )
        print(format_comparison_report(comp))
        return

    if args.text:
        text = args.text
        title = "Direct Input"
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file '{args.file}' not found")
            sys.exit(1)
        text = path.read_text(encoding="utf-8")
        title = path.stem
    else:
        parser.print_help()
        sys.exit(0)

    analyze_vibe(
        text,
        title=title,
        mode=args.mode,
        window_size=args.window_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        canon_percentile=args.canon_percentile,
        verbose=not args.quiet,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
