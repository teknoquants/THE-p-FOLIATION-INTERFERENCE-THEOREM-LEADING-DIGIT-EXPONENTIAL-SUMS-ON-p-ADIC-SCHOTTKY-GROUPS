# Validation suite — The 𝒬ₚ-Foliation Interference Theorem

Python 3.9+ and NumPy. Run everything with:

```
python run_all.py
```

Each script also runs on its own. Total runtime is about 15 seconds.

| Script | Paper section | What it checks |
|---|---|---|
| `qpf_core.py` | — | Shared code: exact p-adic arithmetic (rationals and Z/p^K), Schottky instances, junction weights, digits, characters, Gauss sums |
| `validate_2_1_axioms.py` | 2.1, Def. 2.1, Axioms I–IV, Remark 2.2 | GL₂ and multiplier conditions; reversal; ultrametric norm; rank-one decomposition and ping-pong; good position; 300 random instances for Remark 2.2; negative control |
| `validate_2_2_theorem1.py` | 2.2, Theorem 1 | Product formula to O(p^−t) for words up to length 128, uniform in n; leading digits; additivity of valuations; second-eigenvalue gap; mixed t; rank 3; negative control |
| `validate_2_3_leaves.py` | 2.3 | val det γ_δ = nt; L_{nt} = Cₙ; counting formula for Nₙ; flag probability; non-uniform t mixes word lengths |
| `validate_2_4_quantum_axioms.py` | 2.4, Axioms V–VI | Unitarity and representation property of the digit transport; the phase observable; bridge from the quantum layer to Theorem 1 |
| `validate_theorem2.py` | Section 4–5, Theorem 2 (1)–(4), Hypothesis (H) | Sector decomposition; Gauss-sum moduli; Hankel ranks; statevector simulation of the quantum circuit and of the Grover rotation behind amplitude estimation; explicit transfer matrix; classical sampling scaling |
| `run_log.txt` | — | Output of the run that accompanies this package |

## Result of the accompanying run

181 checks pass and 7 fail. All 7 failures are statements in the white paper as currently printed:

1. **Axiom I (4 failures).** With the normalization "dominant eigenvalue 1", ∇_ē = ∇_e⁻¹ holds only up to a scalar. The identity that holds is ∇_ē = μ(e)·∇_e⁻¹, i.e. equality in PGL₂(ℚₚ). Suggested wording: "∇_ē = ∇_e⁻¹ in PGL₂(ℚₚ) (exactly: ∇_ē = μ(e)∇_e⁻¹)."
2. **Theorem 2(2) "Hankel rank exactly p − 1" (3 failures).** The full Hankel rank is larger: 25, 37 and 38 for p = 7, 11, 13. What holds, and what the proof uses, is: the block K has rank exactly p − 1, so the full Hankel rank is **at least p − 1** and at most d(p − 1) + 1. The upper bound printed in §5.2 should be d(p − 1) + 1 (the extra state is the start state). The same "exactly" appears in the abstract and on the cover.

Neither correction affects the separation: the lower bound p − 1 is what Theorem 2(2) needs.

One potential issue was checked and is **not** an error: Theorem 1 writes the product as ∇_{eₙ}⋯∇_{e₁}, while w(e, e′) = v_eᵀu_{e′} matches ∇_{e₁}⋯∇_{eₙ}. For 2×2 matrices a word and its reverse have the same trace and determinant, so r(δ) is identical either way (verified exactly mod p^K).

## Scope of the checks

- p-adic statements are checked in **exact** arithmetic (rationals, or integers mod p^60), not floating point.
- The quantum layer is checked by exact statevector simulation for small n. The O(1/ε) scaling itself is the amplitude-estimation theorem of Brassard–Høyer–Mosca–Tapp; the suite verifies its instance-specific premise (the Grover iterate rotates by 2θ), not the general theorem.
- Numerical checks on finite instances support the theorems; they do not replace the proofs.
