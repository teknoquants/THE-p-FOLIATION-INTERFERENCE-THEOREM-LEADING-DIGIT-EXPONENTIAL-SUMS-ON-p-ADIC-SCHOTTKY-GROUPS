"""Theorem 2 (Qp-Foliation Interference Theorem), parts (1)-(4) and Hypothesis (H).

S_n = sum_{delta in C_n} psi(abar(delta)),  abar(delta) = prod_k a(e_k, e_{k+1}) mod p.

(1) S_n = sum_chi G(chibar)/(p-1) Tr(B_chi^n); coefficient moduli sqrt(p)/(p-1) and 1/(p-1).
(2) Hankel-rank lower bound p-1 for phi = psi o abar; chi o abar factors (rank 1).
(3) Statevector simulation of the quantum-layer circuit (word prep, flag, digit transport,
    Hadamard test) reproduces S_n/N_n; amplitude-estimation premise (Grover rotation) holds.
(4) Explicit transfer matrix of dimension d(p-1) computes S_n; the minimal dimension grows like p;
    classical sampling error scales as N^-1/2 (Theta(1/eps^2) samples).

Run:  python validate_theorem2.py
"""
import math, random, sys, itertools
import numpy as np
from qpf_core import (Report, good_instance, psi, chi, gauss_sum, dlog_table)

# ------------------------------------------------------------------ helpers
def S_direct(I, n):
    return sum(psi(I.cyclic_digit(wd), I.p) for wd in I.cyclic_words(n))

def B_chi(I, j, tab):
    idx = {L: k for k, L in enumerate(I.letters)}
    B = np.zeros((I.d, I.d), dtype=complex)
    for L1 in I.letters:
        for L2 in I.letters:
            if I.allowed(L1, L2):
                B[idx[L1], idx[L2]] = chi(j, I.digit(L1, L2), I.p, tab)
    return B

def S_sectors(I, n, tab):
    p = I.p
    tot = 0
    for j in range(p - 1):
        G = gauss_sum((-j) % (p - 1), p, tab)          # G(chibar_j)
        tot += G / (p - 1) * np.trace(np.linalg.matrix_power(B_chi(I, j, tab), n))
    return tot

def open_digit(I, wd):
    """Product of internal junction digits of an open non-backtracking word (None if it backtracks)."""
    x = 1
    for k in range(len(wd) - 1):
        if not I.allowed(wd[k], wd[k + 1]):
            return None
        x = x * I.digit(wd[k], wd[k + 1]) % I.p
    return x

def nb_words_upto(I, L):
    out = [()]
    for n in range(1, L + 1):
        out += list(I.reduced_words(n))
    return out

def num_rank(M, tol=1e-8):
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0

# ------------------------------------------------------------------ part (1)
def part1(R):
    for p in (7, 11):
        I = good_instance(p, 1); tab = dlog_table(p)
        ok = True; worst = 0
        for n in range(1, 8):
            a, b = S_direct(I, n), S_sectors(I, n, tab)
            worst = max(worst, abs(a - b)); ok &= abs(a - b) < 1e-7 * max(1, abs(a))
        R.check(f"[p={p}] (1) S_n = sum_chi G(chibar)/(p-1) Tr(B_chi^n) for n = 1..7", ok, f"max |diff| = {worst:.2e}")
        mods = [abs(gauss_sum((-j) % (p - 1), p, tab)) / (p - 1) for j in range(p - 1)]
        R.check(f"[p={p}] (1) coefficient moduli: 1/(p-1) for chi = 1, sqrt(p)/(p-1) otherwise, none zero",
                abs(mods[0] - 1 / (p - 1)) < 1e-12 and all(abs(m - math.sqrt(p) / (p - 1)) < 1e-12 for m in mods[1:]))
        # each sector factors: chi(abar) = prod chi(a(e_k,e_k+1)) -- check on all words n <= 5
        okf = True
        for n in range(1, 6):
            for wd in I.cyclic_words(n):
                for j in range(p - 1):
                    lhs = chi(j, I.cyclic_digit(wd), p, tab)
                    rhs = np.prod([chi(j, I.digit(wd[k], wd[(k + 1) % n]), p, tab) for k in range(n)])
                    okf &= abs(lhs - rhs) < 1e-9
        R.check(f"[p={p}] (1) every sector factors along the walk: chi(abar) = prod chi(a(e_k,e_k+1))", okf)
        # psi does not factor: find a word where psi(abar) != prod psi(a(...))
        nf = any(abs(psi(I.cyclic_digit(wd), p) - np.prod([psi(I.digit(wd[k], wd[(k + 1) % len(wd)]), p)
                 for k in range(len(wd))])) > 1e-6 for wd in I.cyclic_words(3))
        R.check(f"[p={p}] (1) the additive character does not factor: psi(abar) != prod psi(a(...)) on some word", nf)

# ------------------------------------------------------------------ part (2)
def part2(R):
    for p in (7, 11, 13):
        I = good_instance(p, 1); tab = dlog_table(p)
        estar = I.letters[0]
        eprime = next(L for L in I.letters if I.allowed(estar, L))
        c = I.digit(estar, eprime)
        LMAX = 6
        words = nb_words_upto(I, LMAX)
        pref, suff = {}, {}
        for wd in words:
            if wd and wd[-1] == estar:
                x = open_digit(I, wd); pref.setdefault(x, wd)
            if wd and wd[0] == eprime:
                y = open_digit(I, wd); suff.setdefault(y, wd)
        full = set(range(1, p))
        R.check(f"[p={p}] Hypothesis (H): prefix digits ending at e* cover F_p^* (lengths <= {LMAX})",
                set(pref) == full, f"{len(set(pref) & full)}/{p-1}")
        R.check(f"[p={p}] Hypothesis (H): suffix digits starting at e' cover F_p^*", set(suff) == full)
        if set(pref) != full or set(suff) != full:
            continue
        xs = sorted(full)
        K = np.array([[psi(open_digit(I, pref[x] + suff[y]), p) for y in xs] for x in xs])
        Kformula = np.array([[psi(c * x * y, p) for y in xs] for x in xs])
        R.check(f"[p={p}] (2) Hankel block phi(uv) = psi(c a(u) a(v))", np.allclose(K, Kformula))
        R.check(f"[p={p}] (2) rank of the block K = p - 1 = {p-1}", num_rank(K) == p - 1, f"rank {num_rank(K)}")
        Kc = np.array([[chi(1, open_digit(I, pref[x] + suff[y]), p, tab) for y in xs] for x in xs])
        R.check(f"[p={p}] (2) contrast: chi o abar block has rank 1 (factors)", num_rank(Kc) == 1)
        # full Hankel matrix over all non-backtracking prefixes/suffixes up to length 4
        W = nb_words_upto(I, 4)
        H = np.zeros((len(W), len(W)), dtype=complex)
        for i, u in enumerate(W):
            for j, v in enumerate(W):
                x = open_digit(I, u + v)
                H[i, j] = 0 if x is None else psi(x, p)
        rk = num_rank(H)
        R.check(f"[p={p}] (2) full Hankel rank >= p - 1 (lower bound used in the proof)", rk >= p - 1, f"rank {rk}")
        R.check(f"[p={p}] (2) full Hankel rank <= d(p-1) + 1 (explicit automaton, incl. start state)",
                rk <= I.d * (p - 1) + 1, f"rank {rk}, d(p-1)+1 = {I.d*(p-1)+1}")
        R.check(f"[p={p}] (2) as printed: Hankel rank EXACTLY p - 1", rk == p - 1,
                f"observed {rk}; correct statement: p-1 <= rank <= d(p-1)+1")

# ------------------------------------------------------------------ part (3): circuit simulation
class Circuit:
    """Registers: word (d^n letter strings), flag (2), digit (p-1), ancilla (2)."""
    def __init__(self, I, n):
        self.I, self.n, self.p, self.d = I, n, I.p, I.d
        self.strings = list(itertools.product(range(I.d), repeat=n))
        self.sidx = {s: k for k, s in enumerate(self.strings)}
        self.shape = (2, len(self.strings), 2, self.p - 1)      # (anc, word, flag, digit)
        red = [tuple(I.letters.index(L) for L in wd) for wd in I.reduced_words(n)]
        s = np.zeros(len(self.strings)); s[[self.sidx[r] for r in red]] = 1 / math.sqrt(len(red))
        e0 = np.zeros(len(self.strings)); e0[0] = 1
        x = e0 - s; self.hh = x / np.linalg.norm(x)              # Householder: maps e0 <-> s
        self.n_red = len(red)
        self.info = {}
        for k, st in enumerate(self.strings):
            wd = tuple(I.letters[i] for i in st)
            reduced = all(I.allowed(wd[i], wd[i + 1]) for i in range(n - 1))
            flag = reduced and I.allowed(wd[-1], wd[0])
            dig = 1
            if reduced:
                for i in range(n - 1):
                    dig = dig * I.digit(wd[i], wd[i + 1]) % self.p
            close = I.digit(wd[-1], wd[0]) if flag else 1
            self.info[k] = (reduced, flag, dig, close)

    def prep(self, v):          # Householder on word register (self-inverse)
        return v - 2 * np.einsum('w,awfy->afy', self.hh, v)[:, None, :, :] * self.hh[None, :, None, None]

    def flag(self, v):          # flag ^= f(word); self-inverse
        out = v.copy()
        for k, (red, f, _, _) in self.info.items():
            if f:
                out[:, k, 0, :], out[:, k, 1, :] = v[:, k, 1, :], v[:, k, 0, :]
        return out

    def transport(self, v, inverse=False):
        out = np.zeros_like(v)
        for k, (red, f, dig, close) in self.info.items():
            for fl in (0, 1):
                mult = dig * (close if fl == 1 else 1) % self.p if red else 1
                if inverse:
                    mult = pow(mult, -1, self.p)
                for y in range(1, self.p):
                    out[:, k, fl, (mult * y) % self.p - 1] += v[:, k, fl, y - 1]
        return out

    def had(self, v):
        return np.stack([(v[0] + v[1]) / math.sqrt(2), (v[0] - v[1]) / math.sqrt(2)])

    def cphase(self, v, inverse=False):  # controlled on anc=1: Phi on digit if flag=1
        out = v.copy()
        ph = np.array([psi(y, self.p) for y in range(1, self.p)])
        if inverse:
            ph = ph.conj()
        out[1, :, 1, :] = v[1, :, 1, :] * ph[None, :]
        return out

    def A(self, v):
        v = self.prep(v); v = self.flag(v); v = self.transport(v)
        v = self.had(v); v = self.cphase(v); v = self.had(v)
        return v

    def A_dag(self, v):
        v = self.had(v); v = self.cphase(v, inverse=True); v = self.had(v)
        v = self.transport(v, inverse=True); v = self.flag(v); v = self.prep(v)
        return v

    def zero(self):
        v = np.zeros(self.shape, dtype=complex); v[0, 0, 0, 0] = 1; return v

def part3(R):
    for p, n in ((7, 3), (7, 4), (11, 3)):
        I = good_instance(p, 1)
        C = Circuit(I, n)
        # unitarity of A: A_dag(A(v)) = v and norm preserved on random vectors
        rng = np.random.default_rng(0)
        v = rng.normal(size=C.shape) + 1j * rng.normal(size=C.shape); v /= np.linalg.norm(v)
        R.check(f"[p={p},n={n}] (3) circuit A is unitary (A^dag A v = v, norm preserved)",
                np.allclose(C.A_dag(C.A(v)), v) and np.isclose(np.linalg.norm(C.A(v)), 1))
        out = C.A(C.zero())
        P0 = np.sum(abs(out[0]) ** 2)
        reV = 2 * P0 - 1
        Pf = sum(1 for k, inf in C.info.items() if inf[1]) / C.n_red
        Nn = len(list(I.cyclic_words(n)))
        target = S_direct(I, n) / Nn
        est_re = (reV - (1 - Pf)) / Pf
        R.check(f"[p={p},n={n}] (3) flag probability = N_n / (d(d-1)^(n-1)) = {Pf:.4f}",
                abs(Pf - Nn / (I.d * (I.d - 1) ** (n - 1))) < 1e-12)
        R.check(f"[p={p},n={n}] (3) Hadamard test returns Re(S_n/N_n) = {target.real:+.6f}",
                abs(est_re - target.real) < 1e-10, f"circuit {est_re:+.6f}")
        # imaginary part: S^dag on ancilla before the final H  <=> evaluate <Phi> directly from the state
        mid = C.transport(C.flag(C.prep(C.zero())))
        ph = np.array([psi(y, p) for y in range(1, p)])
        exp_flag = np.sum(abs(mid[0, :, 1, :]) ** 2 * ph[None, :])
        R.check(f"[p={p},n={n}] (3) full complex value: <V> on flagged branch / P_f = S_n/N_n",
                abs(exp_flag / Pf - target) < 1e-10)
        # amplitude-estimation premise: Grover iterate rotates by 2 theta in the good/bad plane
        a = P0; theta = math.asin(math.sqrt(a))
        def Q(w):
            w = w.copy(); w[0] *= -1                                  # S_good (anc = 0 is "good")
            w = C.A_dag(w); w[0, 0, 0, 0] *= -1; w = -C.A(w)          # -A S_0 A^dag
            return w
        ok = True; w = out
        for k in range(1, 25):
            w = Q(w)
            ok &= abs(np.sum(abs(w[0]) ** 2) - math.sin((2 * k + 1) * theta) ** 2) < 1e-9
        R.check(f"[p={p},n={n}] (3) Grover iterate: P_good(Q^k A|0>) = sin^2((2k+1) theta), k = 1..24 "
                f"(premise of O(1/eps) amplitude estimation)", ok)
        qubits = n * math.ceil(math.log2(I.d)) + math.ceil(math.log2(p))
        R.note(f"p={p}, n={n}: quantum layer uses n*ceil(log2 d) + ceil(log2 p) = {qubits} data qubits (+ flag, ancilla)")

# ------------------------------------------------------------------ part (4)
def transfer_matrix(I):
    """Explicit d(p-1)-dimensional transfer matrix on (edge, digit)."""
    p, d = I.p, I.d
    T = np.zeros((d * (p - 1), d * (p - 1)))
    for i, L1 in enumerate(I.letters):
        for j, L2 in enumerate(I.letters):
            if I.allowed(L1, L2):
                a = I.digit(L1, L2)
                for y in range(1, p):
                    T[j * (p - 1) + (a * y) % p - 1, i * (p - 1) + y - 1] = 1
    return T

def part4(R):
    for p in (7, 11):
        I = good_instance(p, 1)
        T = transfer_matrix(I)
        ok = True
        for n in range(1, 8):
            Tn = np.linalg.matrix_power(T, n)
            val = sum(psi(y, p) * Tn[i * (p - 1) + y - 1, i * (p - 1)] for i in range(I.d) for y in range(1, p))
            ok &= abs(val - S_direct(I, n)) < 1e-7
        R.check(f"[p={p}] (4) explicit transfer matrix of dimension d(p-1) = {I.d*(p-1)} computes S_n (n=1..7)", ok)
    # minimal dimension grows linearly in p: block rank = p-1 for several primes
    ranks = {}
    for p in (5, 7, 11, 13, 17, 19, 23):
        xs = range(1, p); c = 3 % p or 1
        K = np.array([[psi(c * x * y, p) for y in xs] for x in xs])
        ranks[p] = num_rank(K)
    R.check("(4) transfer-matrix lower bound grows like p: rank K = p - 1 for p = 5..23",
            all(ranks[p] == p - 1 for p in ranks), str(ranks))
    # classical sampling: RMSE ~ N^-1/2
    I = good_instance(11, 1); n = 8
    target = S_direct(I, n) / len(list(I.cyclic_words(n)))
    rng = random.Random(5)
    Ns, rmse = [100, 400, 1600, 6400], []
    for N in Ns:
        errs = []
        for _ in range(40):
            est = sum(psi(I.cyclic_digit(I.random_cyclic_word(n, rng)), 11) for _ in range(N)) / N
            errs.append(abs(est - target) ** 2)
        rmse.append(math.sqrt(sum(errs) / len(errs)))
    slope = np.polyfit(np.log(Ns), np.log(rmse), 1)[0]
    R.check("(4) classical sampling error scales as N^-1/2, i.e. Theta(1/eps^2) samples",
            abs(slope + 0.5) < 0.12, f"fitted slope {slope:.3f}; RMSE " + ", ".join(f"{e:.4f}" for e in rmse))

def run():
    R = Report("Theorem 2: Qp-Foliation Interference Theorem")
    part1(R); part2(R); part3(R); part4(R)
    return R.summary()

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
