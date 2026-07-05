#!/usr/bin/env python3
"""
Reproduce the figures in the README from brain2's actual mechanism: real concept
embeddings from Ollama -> cosine coupling -> Kuramoto phase dynamics. Renders in the
Cerebras Lab palette (ink ground, amber accent, phase = hue).

    being/venv/bin/python docs/make_figures.py
"""
import json, os, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

OLLAMA_EMB = "http://localhost:11434/api/embed"
EMB_MODEL = "nomic-embed-text"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

INK = "#0a0b0f"; PANEL = "#12141b"; AMBER = "#f5b544"; TEXT = "#e9ebf2"; MUTED = "#8a90a2"; GRID = "#20242f"

CONCEPTS = [
    "silence","noise","music","voice","word","language",
    "light","dark","shadow","fire","water","stone",
    "fear","joy","grief","love","anger","calm",
    "mother","child","stranger","crowd","alone","touch",
    "time","memory","future","death","birth","dream",
    "hunger","body","breath","pain","warmth","cold",
]

def embed(words):
    body = json.dumps({"model": EMB_MODEL, "input": words}).encode()
    req = urllib.request.Request(OLLAMA_EMB, data=body, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=120)
    return np.array(json.load(r)["embeddings"], dtype=float)

def coupling_from_embeddings(E, thresh=0.55):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    C = E @ E.T                       # cosine similarity
    np.fill_diagonal(C, 0.0)
    # near concepts couple (excite), everything below threshold is inert.
    # keeps coupling sparse and clustered, so coherence is partial, not global.
    A = np.where(C > thresh, C, 0.0)
    return A

def order_parameter(phase):
    return np.abs(np.mean(np.exp(1j * phase)))

def simulate(A, steps=900, scramble_at=430, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    deg = A.sum(axis=1) + 1e-9
    phase = rng.uniform(0, 2*np.pi, n)
    freq = rng.uniform(-0.4, 0.4, n) * 0.02   # narrow spread so coupling can win
    Kstr = 0.0
    R = np.zeros(steps)
    snaps = {}
    for t in range(steps):
        Kstr = min(1.0, Kstr + 0.006)
        if t == scramble_at:
            phase = rng.uniform(0, 2*np.pi, n)   # the --scramble control: binding destroyed
            Kstr = 0.0
        # Kuramoto on the similarity graph: dtheta_i = w_i + (K/deg_i) * sum_j A_ij sin(theta_j - theta_i)
        interaction = (A * np.sin(phase[None, :] - phase[:, None])).sum(axis=1) / deg
        phase = (phase + freq + Kstr * 0.8 * interaction) % (2*np.pi)
        R[t] = order_parameter(phase)
        if t in (scramble_at - 1, scramble_at + 1, steps - 1):
            snaps[t] = phase.copy()
    return R, snaps, scramble_at

def fig_order(R, scramble_at):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
    ax.plot(R, color=AMBER, lw=2)
    ax.axvline(scramble_at, color=MUTED, lw=1, ls="--")
    ax.annotate("--scramble\nbinding destroyed", (scramble_at, 0.92), color=MUTED,
                fontsize=9, ha="center", family="monospace")
    ax.set_ylim(0, 1); ax.set_xlim(0, len(R))
    ax.set_ylabel("order parameter  R", color=TEXT, fontsize=11)
    ax.set_xlabel("tick", color=MUTED, fontsize=10, family="monospace")
    ax.set_title("Coherence rises, a scramble pulse resets it, it re-forms",
                 color=TEXT, fontsize=12, loc="left", pad=12)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=MUTED); ax.grid(color=GRID, lw=.6, alpha=.6)
    fig.tight_layout()
    out = os.path.join(ASSETS, "order-parameter.png")
    fig.savefig(out, facecolor=INK); plt.close(fig); return out

def fig_field(phase, title, fname):
    n = len(phase)
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 1, (n, 2))
    hue = (phase % (2*np.pi)) / (2*np.pi)
    rgb = hsv_to_rgb(np.stack([hue, np.full(n, .68), np.full(n, .95)], axis=1))
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=150)
    fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
    # amber links between in-phase, near neighbours
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(*(xy[i]-xy[j]))
            if d < 0.28:
                sync = np.cos(phase[i]-phase[j])
                if sync > 0.6:
                    ax.plot([xy[i,0], xy[j,0]], [xy[i,1], xy[j,1]],
                            color=AMBER, lw=0.7, alpha=(sync-0.6)*1.1*(1-d/0.28))
    ax.scatter(xy[:,0], xy[:,1], c=rgb, s=90, edgecolors="none", zorder=3)
    ax.set_title(title, color=TEXT, fontsize=12, family="monospace", loc="left", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(GRID)
    fig.tight_layout()
    out = os.path.join(ASSETS, fname)
    fig.savefig(out, facecolor=INK); plt.close(fig); return out

if __name__ == "__main__":
    print("embedding %d concepts via Ollama..." % len(CONCEPTS))
    E = embed(CONCEPTS)
    K = coupling_from_embeddings(E)
    R, snaps, sc = simulate(K)
    print("wrote", fig_order(R, sc))
    print("wrote", fig_field(snaps[sc-1], "bound: in-phase groups", "phase-field-bound.png"))
    print("wrote", fig_field(snaps[sc+1], "scrambled: binding killed", "phase-field-scrambled.png"))
