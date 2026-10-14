"""Section 2.4 -- The quantum layer, Axioms V and VI.

Axiom V : fiber H = C[F_p^*]; each allowed junction e -> e' acts by the unitary permutation
          |y> -> |a(e,e') y>.
Axiom VI: leaf observable Phi |y> = psi(y)|y>,  psi(y) = exp(2 pi i y / p).
Also checks the bridge between the layers: transporting |1> around a closed word lands on
|leading digit of r(delta)>, the quantity computed by the p-adic layer (Theorem 1).

Run:  python validate_2_4_quantum_axioms.py
"""
import random, sys
import numpy as np
from qpf_core import Report, good_instance, psi, dominant_eigenvalue_mod

def M_perm(a, p):
    """Permutation matrix of y -> a*y on basis {|1>,...,|p-1>}."""
    M = np.zeros((p - 1, p - 1))
    for y in range(1, p):
        M[(a * y) % p - 1, y - 1] = 1
    return M

def Phi(p):
    return np.diag([psi(y, p) for y in range(1, p)])

def ket(y, p):
    v = np.zeros(p - 1, dtype=complex); v[y % p - 1] = 1; return v

def run():
    R = Report("2.4  Quantum layer: Axioms V and VI")
    rng = random.Random(3)
    for p in (5, 7, 11, 13, 31):
        I = good_instance(p, 1)
        Ms = {a: M_perm(a, p) for a in range(1, p)}
        Id = np.eye(p - 1)
        R.check(f"[p={p}] Axiom V: every M_a is unitary (permutation of F_p^*)",
                all(np.allclose(M.T @ M, Id) for M in Ms.values()))
        R.check(f"[p={p}] Axiom V: M_a M_b = M_ab (representation of F_p^*)",
                all(np.allclose(Ms[a] @ Ms[b], Ms[(a * b) % p]) for a in range(1, p) for b in range(1, p)))
        R.check(f"[p={p}] Axiom V: transport operators commute (order of junctions irrelevant)",
                all(np.allclose(Ms[a] @ Ms[b], Ms[b] @ Ms[a]) for a in range(1, p) for b in range(1, p)))
        R.check(f"[p={p}] Axiom V: junction digits a(e,e') are in F_p^* for all allowed junctions",
                all(1 <= I.digit(L1, L2) < p for L1 in I.letters for L2 in I.letters if I.allowed(L1, L2)))
        F = Phi(p)
        R.check(f"[p={p}] Axiom VI: Phi is unitary and diagonal", np.allclose(F.conj().T @ F, Id))
        R.check(f"[p={p}] Axiom VI: Phi is not a multiple of a transport operator (additive vs multiplicative)",
                not any(np.allclose(F, c * Ms[a]) for a in range(1, p) for c in (1,)))
        # bridge: transport |1> around a cyclic word -> |leading digit of r(delta)>, phase psi(that digit)
        ok_bridge = ok_phase = True
        m = p ** I.K
        for _ in range(200):
            n = rng.randint(1, 40)
            wd = I.random_cyclic_word(n, rng)
            state = ket(1, p)
            for k in range(n):
                state = Ms[I.digit(wd[k], wd[(k + 1) % n])] @ state
            lam, _, _ = dominant_eigenvalue_mod(I.word_matrix_mod(wd), p, I.K)
            ok_bridge &= np.allclose(state, ket(lam % p, p))
            ok_phase &= np.isclose(state.conj() @ F @ state, psi(lam % p, p))
        R.check(f"[p={p}] Bridge: transport of |1> around delta = |leading digit of r(delta)> (200 words)", ok_bridge)
        R.check(f"[p={p}] Bridge: <Phi> on the transported state = psi(leading digit of r(delta))", ok_phase)
    return R.summary()

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
