"""Stage 4: Neural network as a soft-decision decoder for Hamming(7,4).

A small MLP (7 -> 64 -> 64 -> 16) is trained with PyTorch at Eb/N0 = 4 dB to
map the 7 received soft values to one of the 16 messages. It is compared
with (a) classical hard decision + syndrome decoding and (b) optimal
maximum-likelihood decoding (closest codeword), 10^6 data bits per point.

Usage: python 04_ml_decoder.py [--no-show]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import erfc

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
args = parser.parse_args()
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(0)
rng = np.random.default_rng(42)
R = 4 / 7

# Hamming(7,4) wie in Etappe 3
G = np.array([[1,0,0,0,1,1,0],
              [0,1,0,0,1,0,1],
              [0,0,1,0,0,1,1],
              [0,0,0,1,1,1,1]])
H = np.array([[1,1,0,1,1,0,0],
              [1,0,1,1,0,1,0],
              [0,1,1,1,0,0,1]])
lut = -np.ones(8, dtype=int)
for j in range(7):
    s = H[:, j]
    lut[s[0]*4 + s[1]*2 + s[2]] = j

# Alle 16 möglichen Nachrichten und ihre Codewörter
all_msgs = np.array([[int(b) for b in f"{i:04b}"] for i in range(16)])
codebook = 2 * (all_msgs @ G % 2) - 1                  # BPSK-Codewörter

def channel(idx, ebn0_db):
    ebn0 = 10 ** (ebn0_db / 10)
    sigma = np.sqrt(1 / (2 * R * ebn0))
    tx = codebook[idx]
    return tx + sigma * rng.standard_normal(tx.shape)

# ---------- 1. Neuronales Netz trainieren ----------
model = nn.Sequential(
    nn.Linear(7, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 16)                                   # 16 Klassen = 16 Nachrichten
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for step in range(3000):
    idx = rng.integers(0, 16, 1024)
    x = torch.tensor(channel(idx, 4.0), dtype=torch.float32)   # Training bei 4 dB
    y = torch.tensor(idx)
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"Schritt {step}: Loss {loss.item():.4f}")

# ---------- 2. Decoder vergleichen ----------
ebn0_db = np.arange(0, 9)
n_test = 250_000
ber_hard, ber_opt, ber_nn = [], [], []

for e in ebn0_db:
    idx = rng.integers(0, 16, n_test)
    msgs = all_msgs[idx]
    rx = channel(idx, e)

    # a) Klassisch: hart entscheiden + Syndrom
    hard = (rx > 0).astype(int)
    syn = hard @ H.T % 2
    pos = lut[syn[:, 0]*4 + syn[:, 1]*2 + syn[:, 2]]
    k = np.where(pos >= 0)[0]
    hard[k, pos[k]] ^= 1
    ber_hard.append(np.mean(msgs != hard[:, :4]))

    # b) Optimal: ähnlichstes Codewort (Maximum Likelihood)
    best = np.argmax(rx @ codebook.T, axis=1)
    ber_opt.append(np.mean(msgs != all_msgs[best]))

    # c) Neuronales Netz
    with torch.no_grad():
        pred = model(torch.tensor(rx, dtype=torch.float32)).argmax(1).numpy()
    ber_nn.append(np.mean(msgs != all_msgs[pred]))

# ---------- 3. Plot ----------
def plot(ber, style, label):
    ber = np.array(ber)
    m = ber > 0
    plt.semilogy(ebn0_db[m], ber[m], style, label=label)

plt.semilogy(ebn0_db, 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10))), "--", color="gray", label="Uncoded (theory)")
plot(ber_hard, "o-", "Hamming, classical (hard decision)")
plot(ber_opt, "s-", "Hamming, optimal (maximum likelihood)")
plot(ber_nn, "x--", "Hamming, neural network")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("Bit error rate (BER)")
plt.title("Neural network decoder vs. classical decoding")
plt.grid(True, which="both")
plt.legend()
plt.savefig(FIG_DIR / "ml_decoder_ber.png", dpi=150)
if not args.no_show:
    plt.show()
