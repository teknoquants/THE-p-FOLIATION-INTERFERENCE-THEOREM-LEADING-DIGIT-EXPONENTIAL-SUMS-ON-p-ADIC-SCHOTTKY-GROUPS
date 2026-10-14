"""Shared core for the Qp-Foliation validation suite.

Exact arithmetic only for p-adic statements:
  * Fractions (exact rationals, which embed in Q_p) for axiom checks;
  * integers mod p^K for long words (all matrix entries lie in Z_p in good position).
Complex floating point is used only for the quantum layer and character sums.
"""
from fractions import Fraction as Fr
import cmath, itertools, random

# ---------------------------------------------------------------- p-adic basics
def vp(x, p):
    """p-adic valuation of a rational (inf for 0)."""
    x = Fr(x)
    if x == 0:
        return float('inf')
    v, n, d = 0, x.numerator, x.denominator
    while n % p == 0:
        n //= p; v += 1
    while d % p == 0:
        d //= p; v -= 1
    return v

def ac(x, p):
    """Leading p-adic digit (angular component) of a nonzero rational: x = p^v * u, returns u mod p."""
    x = Fr(x)
    v = vp(x, p)
    u = x / Fr(p) ** v
    return (u.numerator * pow(u.denominator, -1, p)) % p

def to_mod(x, p, K):
    """Reduce a p-integral rational into Z/p^K."""
    x = Fr(x)
    assert vp(x, p) >= 0, "not p-integral"
    m = p ** K
    return (x.numerator * pow(x.denominator, -1, m)) % m

def vp_int_mod(x, p, K):
    """Valuation of an element of Z/p^K (returns K if zero mod p^K)."""
    x %= p ** K
    if x == 0:
        return K
    v = 0
    while x % p == 0:
        x //= p; v += 1
    return v

def sup_norm_val(M, p):
    """Valuation of the sup-norm of a matrix/vector of rationals (= min entry valuation)."""
    flat = [e for row in M for e in row] if isinstance(M[0], (list, tuple)) else list(M)
    return min(vp(e, p) for e in flat)

# ---------------------------------------------------------------- 2x2 matrices
def mmul(A, B, mod=None):
    C = [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
         [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
    if mod:
        C = [[c % mod for c in r] for r in C]
    return C

def det(A):
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]

def inv(A):
    d = Fr(det(A))
    return [[A[1][1]/d, -A[0][1]/d], [-A[1][0]/d, A[0][0]/d]]

def ident(mod=None):
    return [[1, 0], [0, 1]] if mod else [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]

def mobius(A, z):
    """Action on P^1(Q_p); z = None means infinity."""
    if z is None:
        return None if A[1][0] == 0 else Fr(A[0][0]) / A[1][0]
    den = A[1][0]*z + A[1][1]
    return None if den == 0 else (A[0][0]*z + A[0][1]) / den

# ---------------------------------------------------------------- Schottky instance
class Instance:
    """A rank-g p-adic Schottky group given by fixed points and contraction exponents.

    gens: list of (a, r) = (attracting, repelling) fixed points, rationals in Z_p.
    t:    list of contraction exponents t_i >= 1 (multiplier mu_i = p^t_i), or one int.
    Letters are (i, s) with s = +1 (g_i) or -1 (g_i^{-1}).
    """
    def __init__(self, p, gens, t=1, K=60):
        self.p, self.K = p, K
        self.gens = [(Fr(a), Fr(r)) for a, r in gens]
        self.g = len(gens)
        self.t = [t] * self.g if isinstance(t, int) else list(t)
        self.letters = [(i, s) for i in range(self.g) for s in (+1, -1)]
        self.d = len(self.letters)

    def a(self, L):
        i, s = L; a, r = self.gens[i]
        return a if s > 0 else r

    def r(self, L):
        i, s = L; a, r = self.gens[i]
        return r if s > 0 else a

    def mu(self, L):
        return Fr(self.p) ** self.t[L[0]]

    def tL(self, L):
        return self.t[L[0]]

    @staticmethod
    def bar(L):
        return (L[0], -L[1])

    # Normalized connection: nabla_e = P diag(1, mu) P^{-1}, P = [[a, r], [1, 1]].
    def nabla(self, L):
        a, r, q = self.a(L), self.r(L), self.mu(L)
        P = [[a, r], [Fr(1), Fr(1)]]
        D = [[Fr(1), Fr(0)], [Fr(0), q]]
        return mmul(mmul(P, D), inv(P))

    def nabla_mod(self, L):
        cache = self.__dict__.setdefault('_ncache', {})
        if L not in cache:
            cache[L] = [[to_mod(x, self.p, self.K) for x in row] for row in self.nabla(L)]
        return cache[L]

    def u(self, L):
        return [self.a(L), Fr(1)]

    def v(self, L):
        a, r = self.a(L), self.r(L)
        return [Fr(1) / (a - r), -r / (a - r)]

    def w(self, L1, L2):
        """Junction weight w(e, e') = v_e^T u_e' = (a_e' - r_e)/(a_e - r_e)."""
        return (self.a(L2) - self.r(L1)) / (self.a(L1) - self.r(L1))

    def digit(self, L1, L2):
        key = (L1, L2)
        cache = self.__dict__.setdefault('_dcache', {})
        if key not in cache:
            cache[key] = ac(self.w(L1, L2), self.p)
        return cache[key]

    def allowed(self, L1, L2):
        return L2 != self.bar(L1)

    # -------- words
    def reduced_words(self, n):
        def rec(prefix):
            if len(prefix) == n:
                yield tuple(prefix); return
            for L in self.letters:
                if not prefix or self.allowed(prefix[-1], L):
                    prefix.append(L); yield from rec(prefix); prefix.pop()
        yield from rec([])

    def cyclic_words(self, n):
        for wd in self.reduced_words(n):
            if self.allowed(wd[-1], wd[0]):
                yield wd

    def random_cyclic_word(self, n, rng):
        while True:
            wd = [rng.choice(self.letters)]
            for _ in range(n - 1):
                wd.append(rng.choice([L for L in self.letters if self.allowed(wd[-1], L)]))
            if self.allowed(wd[-1], wd[0]):
                return tuple(wd)

    def word_matrix_mod(self, wd):
        """Matrix of gamma_delta = g_{e1} g_{e2} ... g_{en} (product left to right), mod p^K."""
        m = self.p ** self.K
        M = ident(mod=True)
        for L in wd:
            M = mmul(M, self.nabla_mod(L), mod=m)
        return M

    def word_matrix_mod_reversed(self, wd):
        """Product nabla_{en} ... nabla_{e1} (the order printed in the white paper's Theorem 1)."""
        return self.word_matrix_mod(tuple(reversed(wd)))

    def cyclic_digit(self, wd):
        x = 1
        n = len(wd)
        for k in range(n):
            x = (x * self.digit(wd[k], wd[(k + 1) % n])) % self.p
        return x

    def cyclic_weight_mod(self, wd):
        m = self.p ** self.K
        x = 1
        n = len(wd)
        for k in range(n):
            x = (x * to_mod(self.w(wd[k], wd[(k + 1) % n]), self.p, self.K)) % m
        return x

def dominant_eigenvalue_mod(M, p, K):
    """Hensel/Newton root of x^2 - T x + D near x = T in Z/p^K (requires T a unit, v_p(D) >= 1)."""
    m = p ** K
    T = (M[0][0] + M[1][1]) % m
    D = (M[0][0]*M[1][1] - M[0][1]*M[1][0]) % m
    assert T % p != 0, "trace not a unit"
    x = T
    for _ in range(K.bit_length() + 2):
        f = (x * x - T * x + D) % m
        fp = (2 * x - T) % m
        x = (x - f * pow(fp, -1, m)) % m
    assert (x * x - T * x + D) % m == 0
    return x, T, D

# ---------------------------------------------------------------- characters
def primitive_root(p):
    fac, n, f = set(), p - 1, 2
    while f * f <= n:
        while n % f == 0:
            fac.add(f); n //= f
        f += 1
    if n > 1:
        fac.add(n)
    for gamma in range(2, p):
        if all(pow(gamma, (p - 1) // q, p) != 1 for q in fac):
            return gamma
    raise ValueError

def dlog_table(p):
    gamma = primitive_root(p)
    tab, x = {}, 1
    for k in range(p - 1):
        tab[x] = k; x = x * gamma % p
    return tab

def psi(y, p):
    return cmath.exp(2j * cmath.pi * (y % p) / p)

def chi(j, x, p, tab):
    """Multiplicative character chi_j(x) = exp(2 pi i j dlog(x)/(p-1)); j = 0 is trivial."""
    return cmath.exp(2j * cmath.pi * j * tab[x % p] / (p - 1))

def gauss_sum(j, p, tab):
    return sum(chi(j, z, p, tab) * psi(z, p) for z in range(1, p))

# ---------------------------------------------------------------- reporting
class Report:
    def __init__(self, title):
        self.title, self.rows = title, []
        print("=" * 78); print(title); print("=" * 78)

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        return ok

    def note(self, msg):
        print(f"       note: {msg}")

    def summary(self):
        n_ok = sum(ok for _, ok in self.rows)
        print(f"-- {n_ok}/{len(self.rows)} checks passed\n")
        return n_ok == len(self.rows)

# ---------------------------------------------------------------- standard test instances
def good_instance(p=7, t=1, K=60):
    """Rank-2 instance, four fixed points in distinct residue classes mod p (Remark 2.2)."""
    assert p >= 4
    return Instance(p, [(0, 1), (2, 3)], t=t, K=K)

def bad_instance(p=7, t=1, K=60):
    """Violates good position: a_2 = 1 + p shares a residue class with r_1 = 1."""
    return Instance(p, [(0, 1), (1 + p, 3)], t=t, K=K)
