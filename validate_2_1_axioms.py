"""Section 2.1 -- Definition 2.1 (Qp-Foliation, p-adic layer), Axioms I-IV, Remark 2.2.

Run:  python validate_2_1_axioms.py
"""
import random, sys
from fractions import Fraction as Fr
from qpf_core import (Instance, Report, good_instance, bad_instance, vp, mmul, inv, det,
                      sup_norm_val, mobius)

def rand_padic(p, rng, vmin=-2, vmax=4):
    """Random nonzero rational with prescribed valuation in [vmin, vmax]."""
    v = rng.randint(vmin, vmax)
    while True:
        n, d = rng.randint(1, 10**6), rng.randint(1, 10**6)
        if n % p and d % p:
            return Fr(n, d) * Fr(p) ** v * rng.choice((1, -1))

def run():
    R = Report("2.1  Definition 2.1 and Axioms I-IV (p-adic layer)")
    rng = random.Random(1)

    for p, t in [(7, 1), (7, 2), (11, 1), (13, 3)]:
        I = good_instance(p, t)
        R.note(f"instance p={p}, t={t}, generators (a,r) = {[(str(a), str(r)) for a, r in I.gens]}")

        # ---- Definition 2.1: nabla_e in GL2(Q_p), mu(e) in pZ_p with |mu|_p <= p^-t
        ok_gl = all(det(I.nabla(L)) != 0 for L in I.letters)
        ok_det = all(det(I.nabla(L)) == I.mu(L) for L in I.letters)
        ok_mu = all(vp(I.mu(L), p) == I.tL(L) >= 1 for L in I.letters)
        R.check(f"[p={p},t={t}] Def 2.1: nabla_e in GL2(Q_p), det nabla_e = mu(e)", ok_gl and ok_det)
        R.check(f"[p={p},t={t}] Def 2.1: mu(e) in pZ_p, v_p(mu) = t >= 1", ok_mu)
        ok_eig = all([I.nabla(L)[0][0] * I.a(L) + I.nabla(L)[0][1], I.nabla(L)[1][0] * I.a(L) + I.nabla(L)[1][1]]
                     == [I.a(L), Fr(1)] for L in I.letters)
        R.check(f"[p={p},t={t}] Def 2.1: normalization -- u_e = (a_e,1) has eigenvalue 1", ok_eig)

        # ---- Axiom I: reversal
        lit = all(I.nabla(I.bar(L)) == inv(I.nabla(L)) for L in I.letters)
        proj = all(I.nabla(I.bar(L)) == [[I.mu(L) * x for x in row] for row in inv(I.nabla(L))]
                   for L in I.letters)
        R.check(f"[p={p},t={t}] Axiom I in PGL2: nabla_ebar = mu(e) * nabla_e^-1 (equal projectively)", proj)
        R.check(f"[p={p},t={t}] Axiom I as printed: nabla_ebar = nabla_e^-1 exactly", lit,
                "fails: with the eigenvalue-1 normalization the identity holds only up to the scalar mu(e)")

        # ---- Axiom II: ultrametric sup-norm on V_v = Q_p^2 (and induced operator norm)
        ok2 = True
        for _ in range(400):
            x = [rand_padic(p, rng), rand_padic(p, rng)]
            y = [rand_padic(p, rng), rand_padic(p, rng)]
            s = [x[0] + y[0], x[1] + y[1]]
            vs = sup_norm_val(s, p) if any(s) else float('inf')
            ok2 &= vs >= min(sup_norm_val(x, p), sup_norm_val(y, p))
        R.check(f"[p={p},t={t}] Axiom II: ||x+y|| <= max(||x||,||y||) on 400 random pairs", ok2)
        ok2b = True
        for _ in range(200):
            A = [[rand_padic(p, rng) for _ in range(2)] for _ in range(2)]
            B = [[rand_padic(p, rng) for _ in range(2)] for _ in range(2)]
            AB = mmul(A, B)
            ok2b &= sup_norm_val(AB, p) >= sup_norm_val(A, p) + sup_norm_val(B, p)
        R.check(f"[p={p},t={t}] Axiom II: operator sup-norm submultiplicative ||AB|| <= ||A|| ||B||", ok2b)

        # ---- Axiom III: nabla_e = u v^T + mu R, v^T u = 1, ||R|| <= 1, derived from ping-pong
        ok3a = ok3b = True
        for L in I.letters:
            u, v = I.u(L), I.v(L)
            ok3a &= (v[0] * u[0] + v[1] * u[1]) == 1
            N = I.nabla(L)
            Rm = [[(N[i][j] - u[i] * v[j]) / I.mu(L) for j in range(2)] for i in range(2)]
            ok3b &= sup_norm_val(Rm, p) >= 0
        R.check(f"[p={p},t={t}] Axiom III: v_e^T u_e = 1", ok3a)
        R.check(f"[p={p},t={t}] Axiom III: R_e = (nabla_e - u_e v_e^T)/mu(e) has ||R_e|| <= 1", ok3b)

        # ping-pong: g_e maps P^1 minus D_ebar into D_e  (D_e = ball of radius p^-1 around a_e)
        ok_pp = True
        pts = [None] + [Fr(k) for k in range(-3 * p, 3 * p)] + [Fr(1, p), Fr(1, p * p)] + \
              [rand_padic(p, rng, -2, 3) for _ in range(150)]
        for L in I.letters:
            for z in pts:
                outside = z is None or vp(z - I.r(L), p) < 1
                if outside:
                    gz = mobius(I.nabla(L), z)
                    ok_pp &= gz is not None and vp(gz - I.a(L), p) >= I.tL(L)
        R.check(f"[p={p},t={t}] Axiom III (derivation): ping-pong g_e(P^1 \\ D_ebar) in a_e + p^t Z_p", ok_pp)

        # ---- Axiom IV: good position -- every allowed junction weight is a unit
        ws = [(L1, L2, I.w(L1, L2)) for L1 in I.letters for L2 in I.letters if I.allowed(L1, L2)]
        R.check(f"[p={p},t={t}] Axiom IV: all {len(ws)} allowed junction weights are p-adic units",
                all(vp(w, p) == 0 for _, _, w in ws))
        back = all(I.w(L, I.bar(L)) == 0 for L in I.letters)
        R.check(f"[p={p},t={t}] Axiom IV: backtracking junctions e -> ebar have w = 0 (correctly excluded)", back)

    # ---- negative control: bad position violates Axiom IV
    B = bad_instance(7, 1)
    bad = [(L1, L2) for L1 in B.letters for L2 in B.letters
           if B.allowed(L1, L2) and vp(B.w(L1, L2), 7) != 0]
    R.check("Negative control: bad position (a_2 = 1+p ~ r_1 = 1) violates Axiom IV", len(bad) > 0,
            f"{len(bad)} non-unit junction(s)")

    # ---- Remark 2.2: distinct residues mod p => Axiom IV (random instances, random p-adic lifts)
    ok_r = True
    for _ in range(300):
        p = rng.choice([5, 7, 11, 13, 17])
        g = rng.randint(2, (p) // 2)
        res = rng.sample(range(p), 2 * g)
        gens = [(res[2 * i] + p * rng.randint(-50, 50), res[2 * i + 1] + p * rng.randint(-50, 50))
                for i in range(g)]
        J = Instance(p, gens, t=rng.randint(1, 3))
        ok_r &= all(vp(J.w(L1, L2), p) == 0 for L1 in J.letters for L2 in J.letters if J.allowed(L1, L2))
    R.check("Remark 2.2: 2g fixed points in distinct residue classes => Axiom IV (300 random instances)", ok_r)
    return R.summary()

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
