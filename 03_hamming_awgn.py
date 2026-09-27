"""Stage 3: Channel coding with Hamming(7,4) over BPSK/AWGN.

Encodes 250,000 blocks of 4 data bits (10^6 data bits) per Eb/N0 point,
transmits them with BPSK (noise scaled by the code rate R = 4/7 so that the
energy per data bit is fair), hard-decides and corrects single-bit errors
via syndrome lookup. The coded BER is compared with uncoded BPSK theory.

Usage: python 03_hamming_awgn.py [--no-show]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
args = parser.parse_args()
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n_blocks = 250_000                     # 250.000 x 4 = 1 Mio. Datenbits
ebn0_db = np.arange(0, 11)
R = 4 / 7                              # Coderate

# Hamming(7,4): Generator- und Prüfmatrix
G = np.array([[1,0,0,0,1,1,0],
              [0,1,0,0,1,0,1],
              [0,0,1,0,0,1,1],
              [0,0,0,1,1,1,1]])
H = np.array([[1,1,0,1,1,0,0],
              [1,0,1,1,0,1,0],
              [0,1,1,1,0,0,1]])

# Syndrom -> Position des fehlerhaften Bits
lut = -np.ones(8, dtype=int)
for j in range(7):
    s = H[:, j]
    lut[s[0]*4 + s[1]*2 + s[2]] = j

ber_coded = []
for e in ebn0_db:
    ebn0 = 10 ** (e / 10)
    msg = rng.integers(0, 2, (n_blocks, 4))
    codeword = msg @ G % 2                          # 1. Kodieren
    tx = 2 * codeword - 1                           # 2. BPSK
    sigma = np.sqrt(1 / (2 * R * ebn0))             # Energie pro Datenbit fair verteilt
    rx = tx + sigma * rng.standard_normal(tx.shape) # 3. AWGN-Kanal
    hard = (rx > 0).astype(int)                     # 4. Entscheidung
    syn = hard @ H.T % 2                            # 5. Syndrom berechnen
    pos = lut[syn[:, 0]*4 + syn[:, 1]*2 + syn[:, 2]]
    idx = np.where(pos >= 0)[0]
    hard[idx, pos[idx]] ^= 1                        # 6. Fehler korrigieren
    ber_coded.append(np.mean(msg != hard[:, :4]))

ber_uncoded = 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10)))

plt.semilogy(ebn0_db, ber_uncoded, "-", label="Uncoded (theory)")
ber_coded = np.array(ber_coded)
mask = ber_coded > 0                   # Punkte ohne Fehler (BER = 0) weglassen
plt.semilogy(ebn0_db[mask], ber_coded[mask], "o-", label="Hamming(7,4) simulation")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("Bit error rate (BER)")
plt.title("Channel coding: Hamming(7,4) vs. uncoded")
plt.grid(True, which="both")
plt.legend()
plt.savefig(FIG_DIR / "hamming_ber.png", dpi=150)
if not args.no_show:
    plt.show()
