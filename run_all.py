"""Run the full Qp-Foliation validation suite and print a summary table."""
import importlib, sys, time

MODULES = [("2.1 Definition 2.1, Axioms I-IV", "validate_2_1_axioms"),
           ("2.2 Theorem 1", "validate_2_2_theorem1"),
           ("2.3 Leaves", "validate_2_3_leaves"),
           ("2.4 Quantum layer, Axioms V-VI", "validate_2_4_quantum_axioms"),
           ("Theorem 2", "validate_theorem2")]

if __name__ == "__main__":
    results = []
    for label, mod in MODULES:
        t0 = time.time()
        ok = importlib.import_module(mod).run()
        results.append((label, ok, time.time() - t0))
    print("=" * 78); print("SUMMARY"); print("=" * 78)
    for label, ok, dt in results:
        print(f"{'ALL PASS ' if ok else 'HAS FAILS'}  {label:40s} ({dt:5.1f}s)")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)
