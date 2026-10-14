"""Section 2.2 -- Theorem 1 (leading-digit product formula).

Claim: for a cyclically reduced word delta = (e_1..e_n) under Axioms I-IV,
    r(delta) = prod_k w(e_k, e_{k+1}) * (1 + O(p^-t)),   uniformly in n,
valuations add exactly, and the leading digit of r(delta) is prod_k a(e_k, e_{k+1}) mod p.
Long words are handled exactly in Z/p^K (all entries are p-integral in good position).

Run:  python validate_2_2_theorem1.py
"""
import random, sys
from qpf_core import (Instance, Report, good_instance, bad_instance, dominant_eigenvalue_mod,
                      vp_int_mod)

LENGTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

def check_instance(R, I, label, rng, trials=60, order="left-to-right"):
    p, K = I.p, I.K
    m = p ** K
    tmin = min(I.t)
    worst_rel = K
    ok_digit = ok_val = ok_gap = True
    per_n = {}
    mismatches = 0
    total = 0
    for n in LENGTHS:
        mn = K
        for _ in range(trials):
            wd = I.random_cyclic_word(n, rng)
            M = I.word_matrix_mod(wd) if order == "left-to-right" else I.word_matrix_mod_reversed(wd)
            T = (M[0][0] + M[1][1]) % m
            if T % p == 0:                       # trace not a unit: formula cannot hold
                mismatches += 1; total += 1; mn = 0; continue
            W = I.cyclic_weight_mod(wd)
            if W % p == 0:                       # product of weights not a unit (Axiom IV fails)
                mismatches += 1; total += 1; mn = 0; continue
            lam, T, D = dominant_eigenvalue_mod(M, p, K)
            rel = vp_int_mod(lam * pow(W, -1, m) - 1, p, K)
            mn = min(mn, rel)
            total += 1
            if lam % p != I.cyclic_digit(wd):
                mismatches += 1
            sum_t = sum(I.tL(L) for L in wd)
            ok_val &= vp_int_mod(D, p, K) == min(sum_t, K)
            lam2 = (T - lam) % m                  # second eigenvalue
            ok_gap &= vp_int_mod(lam2, p, K) >= min(sum_t, K)
        per_n[n] = mn
        worst_rel = min(worst_rel, mn)
    return worst_rel, per_n, mismatches, total, ok_val, ok_gap

def run():
    R = Report("2.2  Theorem 1: leading-digit product formula")
    rng = random.Random(7)
    configs = [("p=7, t=1", good_instance(7, 1)),
               ("p=7, t=2", good_instance(7, 2)),
               ("p=11, t=1", good_instance(11, 1)),
               ("p=13, t=3", good_instance(13, 3)),
               ("p=7, mixed t=(1,3)", Instance(7, [(0, 1), (2, 3)], t=[1, 3])),
               ("p=11, rank 3, t=1", Instance(11, [(0, 1), (2, 3), (4, 5)], t=1))]
    for label, I in configs:
        tmin = min(I.t)
        worst, per_n, mism, tot, ok_val, ok_gap = check_instance(R, I, label, rng)
        R.check(f"[{label}] r(delta) = prod w * (1 + O(p^-t)): min v_p(r/prod w - 1) = {worst} >= t_min = {tmin}",
                worst >= tmin)
        uniform = min(per_n.values()) == min(per_n[n] for n in LENGTHS[-4:])
        R.check(f"[{label}] uniform in n (n = 1..{LENGTHS[-1]}): no decay of the error exponent with length",
                worst >= tmin and uniform,
                "per-n min: " + ", ".join(f"{n}:{per_n[n]}" for n in (1, 4, 16, 64, 128)))
        R.check(f"[{label}] leading digit of r(delta) = prod a(e_k,e_k+1) mod p ({tot} words)", mism == 0)
        R.check(f"[{label}] valuations add: v_p(det) = sum of t over letters", ok_val)
        R.check(f"[{label}] second eigenvalue: v_p(lambda_2) >= sum t (lambda_1 = trace to O(p^-t))", ok_gap)

    # ---- matrix order: the white paper prints nabla_{e_n} ... nabla_{e_1}
    I = good_instance(7, 1)
    worst, per_n, mism, tot, _, _ = check_instance(R, I, "order", rng, trials=60, order="right-to-left")
    R.check("Order as printed (nabla_en ... nabla_e1) gives the same r(delta) as nabla_e1 ... nabla_en",
            mism == 0, f"{mism}/{tot} disagree (2x2 trace-reversal identity: tr(w) = tr(reverse w), same det)")
    # exact equality on an asymmetric instance
    J = Instance(7, [(38, 0), (6, 79)], t=1)
    same = True
    for n in (3, 5, 8, 13, 21):
        for _ in range(30):
            wd = J.random_cyclic_word(n, rng)
            a, _, _ = dominant_eigenvalue_mod(J.word_matrix_mod(wd), 7, J.K)
            b, _, _ = dominant_eigenvalue_mod(J.word_matrix_mod_reversed(wd), 7, J.K)
            same &= a == b
    R.check("Word reversal leaves r(delta) unchanged exactly mod p^K (asymmetric instance)", same)

    # ---- negative control: bad position
    B = bad_instance(7, 1)
    worst, per_n, mism, tot, _, _ = check_instance(R, B, "bad", rng, trials=40)
    R.check("Negative control: without Axiom IV the formula fails on some words", worst < 1 or mism > 0,
            f"min rel. valuation {worst}, digit mismatches {mism}/{tot}")
    return R.summary()

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
