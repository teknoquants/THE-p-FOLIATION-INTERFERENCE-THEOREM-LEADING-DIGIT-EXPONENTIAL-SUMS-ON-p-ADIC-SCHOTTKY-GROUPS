"""Section 2.3 -- Leaves of the valuation foliation.

Claims: L_m = {delta : val det gamma_delta = m}; under uniform t the leaf L_{nt} is exactly the
set C_n of cyclically reduced words of length n; N_n = |C_n| = Tr(B_1^n) = (2g-1)^n + g + (g-1)(-1)^n;
with non-uniform t a leaf mixes word lengths (a valuation clock is then needed).

Run:  python validate_2_3_leaves.py
"""
import sys
from fractions import Fraction as Fr
import numpy as np
from qpf_core import Instance, Report, good_instance, vp, det, mmul, ident

def word_det_val(I, wd):
    M = ident()
    for L in wd:
        M = mmul(M, I.nabla(L))
    return vp(det(M), I.p)

def hashimoto(I):
    idx = {L: k for k, L in enumerate(I.letters)}
    B = np.zeros((I.d, I.d))
    for L1 in I.letters:
        for L2 in I.letters:
            if I.allowed(L1, L2):
                B[idx[L1], idx[L2]] = 1
    return B

def run():
    R = Report("2.3  Leaves of the valuation foliation")
    NMAX = 6
    for g, p, t in [(2, 7, 1), (2, 7, 2), (3, 11, 1)]:
        gens = [(2 * i, 2 * i + 1) for i in range(g)]
        I = Instance(p, gens, t=t)
        cyc = {n: list(I.cyclic_words(n)) for n in range(1, NMAX + 1)}
        # exact determinant valuations
        vals = {n: {wd: word_det_val(I, wd) for wd in cyc[n]} for n in range(1, 5)}
        ok_leaf = all(v == n * t for n in vals for v in vals[n].values())
        R.check(f"[g={g},p={p},t={t}] val det gamma_delta = n t exactly (exact rationals, n<=4)", ok_leaf)
        # leaf L_{nt} among all cyclic words of length <= NMAX equals C_n
        all_words = [wd for n in cyc for wd in cyc[n]]
        ok_eq = all({wd for wd in all_words if len(wd) * t == n * t} == set(cyc[n]) for n in cyc)
        R.check(f"[g={g},p={p},t={t}] leaf L_(nt) = C_n for n = 1..{NMAX}", ok_eq)
        # counting formula
        B = hashimoto(I)
        ok_cnt = True
        for n in range(1, NMAX + 1):
            formula = (2 * g - 1) ** n + g + (g - 1) * (-1) ** n
            tr = int(round(np.trace(np.linalg.matrix_power(B, n))))
            ok_cnt &= len(cyc[n]) == formula == tr
        R.check(f"[g={g},p={p},t={t}] N_n = |C_n| = Tr(B_1^n) = (2g-1)^n + g + (g-1)(-1)^n, n = 1..{NMAX}", ok_cnt)
        # flag probability used in 5.3
        d = I.d
        ratios = [len(cyc[n]) / (d * (d - 1) ** (n - 1)) for n in range(1, NMAX + 1)]
        R.check(f"[g={g},p={p},t={t}] flag probability N_n / (d(d-1)^(n-1)) bounded below (min {min(ratios):.3f})",
                min(ratios) >= (d - 2) / (d - 1) - 1e-12)

    # non-uniform contraction: leaves mix lengths
    J = Instance(7, [(0, 1), (2, 3)], t=[1, 2])
    leaf = {}
    for n in range(1, 5):
        for wd in J.cyclic_words(n):
            leaf.setdefault(word_det_val(J, wd), set()).add(n)
    mixed = {m: sorted(ns) for m, ns in leaf.items() if len(ns) > 1}
    R.check("Non-uniform t=(1,2): some leaf L_m contains words of different lengths (clock register needed)",
            len(mixed) > 0, f"e.g. L_4 holds lengths {mixed.get(4)}")
    ok_sum = all(word_det_val(J, wd) == sum(J.tL(L) for L in wd) for n in range(1, 4) for wd in J.cyclic_words(n))
    R.check("Non-uniform t: val det gamma_delta = sum of t over letters", ok_sum)
    return R.summary()

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
