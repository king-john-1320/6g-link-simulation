"""Stage 1: BPSK over an AWGN channel.

Simulates 10^6 random bits per Eb/N0 point (0-10 dB), maps them to BPSK
symbols, adds white Gaussian noise and compares the simulated bit error
rate with the theoretical BER 0.5 * erfc(sqrt(Eb/N0)).

Usage: python 01_bpsk_awgn.py [--no-show]
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
n_bits = 1_000_000
ebn0_db = np.arange(0, 11)   # Signal-Rausch-Verhältnis in dB

ber_sim = []
for e in ebn0_db:
    bits = rng.integers(0, 2, n_bits)          # 1. Zufällige Bits
    symbols = 2 * bits - 1                      # 2. BPSK: 0 -> -1, 1 -> +1
    ebn0 = 10 ** (e / 10)
    sigma = np.sqrt(1 / (2 * ebn0))
    received = symbols + sigma * rng.standard_normal(n_bits)  # 3. AWGN-Kanal
    bits_hat = (received > 0).astype(int)       # 4. Entscheidung
    ber_sim.append(np.mean(bits != bits_hat))   # 5. Bitfehlerrate

ber_theory = 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10)))

ber_sim = np.array(ber_sim)
mask = ber_sim > 0                              # Punkte ohne Fehler (BER = 0) weglassen
plt.semilogy(ebn0_db[mask], ber_sim[mask], "o", label="Simulation")
plt.semilogy(ebn0_db, ber_theory, "-", label="Theory")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("Bit error rate (BER)")
plt.title("BPSK over AWGN channel")
plt.grid(True, which="both")
plt.legend()
plt.savefig(FIG_DIR / "bpsk_ber.png", dpi=150)
if not args.no_show:
    plt.show()
