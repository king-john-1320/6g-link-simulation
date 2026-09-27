"""Stage 2: QPSK over an AWGN channel.

Gray-mapped QPSK (2 bits per complex symbol), 10^6 bits per Eb/N0 point
(0-10 dB). Left: simulated BER vs. theory (identical to BPSK per Eb/N0).
Right: constellation of received symbols at Eb/N0 = 8 dB.

Usage: python 02_qpsk_awgn.py [--no-show]
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
ebn0_db = np.arange(0, 11)

def qpsk_transmit(ebn0):
    bits = rng.integers(0, 2, n_bits)
    b = bits.reshape(-1, 2)                                   # je 2 Bits -> 1 Symbol
    symbols = ((2*b[:, 0] - 1) + 1j*(2*b[:, 1] - 1)) / np.sqrt(2)
    sigma = np.sqrt(1 / (4 * ebn0))                           # Rauschen pro Dimension
    noise = sigma * (rng.standard_normal(len(symbols)) + 1j*rng.standard_normal(len(symbols)))
    received = symbols + noise
    bits_hat = np.column_stack((received.real > 0, received.imag > 0)).astype(int).ravel()
    return bits, bits_hat, received

ber_sim = []
for e in ebn0_db:
    bits, bits_hat, _ = qpsk_transmit(10 ** (e / 10))
    ber_sim.append(np.mean(bits != bits_hat))

ber_theory = 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ber_sim = np.array(ber_sim)
mask = ber_sim > 0                                            # Punkte ohne Fehler (BER = 0) weglassen
ax1.semilogy(ebn0_db[mask], ber_sim[mask], "o", label="QPSK simulation")
ax1.semilogy(ebn0_db, ber_theory, "-", label="Theory")
ax1.set_xlabel("Eb/N0 [dB]")
ax1.set_ylabel("Bit error rate (BER)")
ax1.set_title("QPSK over AWGN channel")
ax1.grid(True, which="both")
ax1.legend()

_, _, received = qpsk_transmit(10 ** (8 / 10))                # Konstellation bei 8 dB
ax2.scatter(received[:3000].real, received[:3000].imag, s=2, alpha=0.4)
ax2.axhline(0, color="gray"); ax2.axvline(0, color="gray")
ax2.set_title("Received symbols (Eb/N0 = 8 dB)")
ax2.set_xlabel("In-phase (I)"); ax2.set_ylabel("Quadrature (Q)")
ax2.set_aspect("equal")

plt.tight_layout()
plt.savefig(FIG_DIR / "qpsk_ber.png", dpi=150)
if not args.no_show:
    plt.show()
