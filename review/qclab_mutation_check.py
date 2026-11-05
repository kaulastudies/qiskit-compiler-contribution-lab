"""Mutation check for qiskit-compiler-contribution-lab (reviewed at commit 1e29941).

Idea: a checker is only useful if a *wrong* solution fails it. Each mutant below
is the instructor reference with one plausible mistake. "KILLED" means the
checker caught it; "SURVIVED" means the checker passed a wrong solution.

These mutants are new; they don't repeat the 8 already in tests/test_cli.py.

Run from the repository root, inside the locked teaching environment:

    uv run python /path/to/qclab_mutation_check.py

Nothing in the repository is modified. Temporary files are deleted afterwards.
"""

import importlib.util
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path.cwd()
REF01 = ROOT / "src/qclab/reference.py"
REF02 = ROOT / "src/qclab/reference02.py"
CASE_TESTS = ROOT / "tests/test_contribution_case.py"

# (lab, id, [(old, new), ...], expectation, why it is a plausible mistake)
MUTANTS = [
    (
        "01",
        "skips-directives",
        [("dag.op_nodes()", "dag.op_nodes(include_directives=False)")],
        "survive?",
        "Learner copies an op_nodes() call that hides barriers; Barrier is still an Instruction.",
    ),
    (
        "01",
        "topological-order",
        [("for node in dag.op_nodes() if", "for node in dag.topological_op_nodes() if")],
        "survive?",
        "Contract says 'order of dag.op_nodes()'; every test circuit is a single wire chain.",
    ),
    (
        "01",
        "empty-means-all",
        [("if operation_types is None:", "if not operation_types:")],
        "killed",
        "Truthiness check treats [] / () / set() like None.",
    ),
    (
        "01",
        "match-by-name",
        [
            (
                "    return [node for node in dag.op_nodes() if isinstance(node.op, types)]",
                "    names = {t.__name__.lower() for t in types}\n"
                "    return [node for node in dag.op_nodes() if node.name in names]",
            )
        ],
        "killed",
        "Compares class names to node names instead of using isinstance.",
    ),
    (
        "01",
        "returns-tuple",
        [
            (
                "return [node for node in dag.op_nodes() if isinstance(node.op, types)]",
                "return tuple(node for node in dag.op_nodes() if isinstance(node.op, types))",
            )
        ],
        "killed",
        "Returns a tuple, not the list the contract promises.",
    ),
    (
        "01",
        "coerces-instances",
        [
            (
                "    if not all(isinstance(operation_type, type) for operation_type in types):\n"
                '        raise TypeError("Every operation_types entry must be a Python class")\n',
                "    types = tuple(t if isinstance(t, type) else type(t) for t in types)\n",
            )
        ],
        "killed",
        "'Helpfully' converts Measure() or 'reset' to a class instead of raising TypeError.",
    ),
    (
        "02",
        "count-ops-recurses",
        [('"operation_counts": counts,', '"operation_counts": dict(dag.count_ops()),')],
        "killed",
        "Uses dag.count_ops(), which looks inside control-flow blocks by default.",
    ),
    (
        "02",
        "two-qubit-ops-helper",
        [
            (
                '"num_two_qubit_gates": two_qubit_gates,',
                '"num_two_qubit_gates": len(dag.two_qubit_ops()),',
            )
        ],
        "killed",
        "Uses dag.two_qubit_ops(), which counts a 2-qubit Instruction that is not a Gate.",
    ),
    (
        "02",
        "skips-directives",
        [("for node in dag.op_nodes():", "for node in dag.op_nodes(include_directives=False):")],
        "killed",
        "Drops barriers from operation_counts.",
    ),
    (
        "02",
        "at-least-two-qubits",
        [("len(node.qargs) == 2", "len(node.qargs) >= 2")],
        "killed",
        "Off-by-arity: counts 3-qubit gates as two-qubit gates.",
    ),
]


def apply(source, edits, label):
    for old, new in edits:
        if old not in source:
            raise SystemExit(f"[{label}] reference text not found, update the mutant: {old!r}")
        source = source.replace(old, new)
    return source


def run_check(lab, solution):
    result = subprocess.run(
        [sys.executable, "-m", "qclab.cli", "check", "--lab", lab, "--solution", str(solution)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    caught = re.findall(r"FAILED \S*::(\w+)", result.stdout)
    return result.returncode, sorted(set(caught)), result.stdout + result.stderr


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Proposed new tests (same style as src/qclab/checks/test_selection.py) ----------


def proposed_directives_are_selected(select_op_nodes):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Barrier, Instruction
    from qiskit.converters import circuit_to_dag

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.barrier(0)
    dag = circuit_to_dag(circuit)
    assert [n.name for n in select_op_nodes(dag, Barrier)] == ["barrier"]
    assert [n.name for n in select_op_nodes(dag, Instruction)] == ["h", "barrier"]
    assert select_op_nodes(dag) == list(dag.op_nodes())


def proposed_order_follows_op_nodes_across_wires(select_op_nodes):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate
    from qiskit.converters import circuit_to_dag

    circuit = QuantumCircuit(2)
    circuit.h(1)
    circuit.h(0)
    dag = circuit_to_dag(circuit)
    assert select_op_nodes(dag, HGate) == list(dag.op_nodes())


PROPOSED_TESTS = [proposed_directives_are_selected, proposed_order_follows_op_nodes_across_wires]

QDRIFT_CONTROL = textwrap.dedent(
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import QDrift


    def test_nonzero_operator_and_time_produce_rotations():
        evolution = PauliEvolutionGate(SparsePauliOp("Z", coeffs=[1.0]), time=1.0)
        assert len(QDrift(seed=7).expand(evolution)) > 0
    """
)

ALWAYS_EMPTY_RUNNER = textwrap.dedent(
    """
    import sys
    import pytest
    from qiskit.synthesis import QDrift

    if sys.argv[1] == "mutant":
        QDrift.expand = lambda self, evolution: []
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[2:]]))
    """
)


def main():
    for path in (REF01, REF02, CASE_TESTS):
        if not path.is_file():
            raise SystemExit(f"Run this from the repository root (missing {path}).")

    import qiskit

    print(f"Python {sys.version.split()[0]}, Qiskit {qiskit.__version__}, repo {ROOT}\n")
    survivors = []
    with tempfile.TemporaryDirectory(prefix="qclab-mutants-") as tmp:
        tmp = Path(tmp)

        # 1. Lab checkers vs new mutants
        print("## 1. Lab checkers vs new mutants\n")
        print("| Lab | Mutant | Expected | Result | Caught by |")
        print("|---|---|---|---|---|")
        sources = {"01": REF01.read_text(), "02": REF02.read_text()}
        mutant_files = {}
        for lab, label, edits, expected, _why in MUTANTS:
            path = tmp / f"lab{lab}_{label}.py"
            path.write_text(apply(sources[lab], edits, label), encoding="utf-8")
            mutant_files[(lab, label)] = path
            code, caught, output = run_check(lab, path)
            if code == 0:
                verdict = "SURVIVED"
                survivors.append(f"Lab {lab} {label}")
            elif code == 1:
                verdict = "killed"
            else:
                verdict = f"ERROR (exit {code})"
                print(textwrap.indent(output[-1500:], "    "))
            print(f"| {lab} | {label} | {expected} | {verdict} | {', '.join(caught) or '-'} |")

        # 2. Contribution case: always-empty QDrift.expand
        print("\n## 2. Contribution case: QDrift.expand always returns []\n")
        runner = tmp / "always_empty_runner.py"
        runner.write_text(ALWAYS_EMPTY_RUNNER, encoding="utf-8")
        control = tmp / "test_qdrift_control.py"
        control.write_text(QDRIFT_CONTROL, encoding="utf-8")
        for mode in ("real", "mutant"):
            for target in (CASE_TESTS, control):
                result = subprocess.run(
                    [sys.executable, str(runner), mode, str(target)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
                print(f"- {mode:6} QDrift, {target.name}: exit {result.returncode} ({tail})")
                if mode == "mutant" and target == CASE_TESTS and result.returncode == 0:
                    survivors.append("contribution case: always-empty expand")

        # 3. Proposed Lab 01 tests: must pass the reference and kill the survivors
        print("\n## 3. Proposed Lab 01 tests\n")
        candidates = {"reference": REF01}
        for key in [("01", "skips-directives"), ("01", "topological-order")]:
            candidates[key[1]] = mutant_files[key]
        for name, path in candidates.items():
            module = load(path, f"cand_{name.replace('-', '_')}")
            for test in PROPOSED_TESTS:
                try:
                    test(module.select_op_nodes)
                    outcome = "pass"
                except AssertionError:
                    outcome = "FAIL"
                except Exception as exc:  # noqa: BLE001
                    outcome = f"error: {type(exc).__name__}: {exc}"
                print(f"- {name:18} {test.__name__}: {outcome}")

        # 4. Witness: do op_nodes() and topological_op_nodes() really disagree?
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dag

        circuit = QuantumCircuit(2)
        circuit.h(1)
        circuit.h(0)
        dag = circuit_to_dag(circuit)

        def wires(nodes):
            return [dag.find_bit(n.qargs[0]).index for n in nodes]

        print("\n## 4. Witness for topological-order\n")
        print(f"- op_nodes() qubit order:            {wires(dag.op_nodes())}")
        print(f"- topological_op_nodes() qubit order: {wires(dag.topological_op_nodes())}")

    print("\n## Summary\n")
    print("Survivors: " + (", ".join(survivors) if survivors else "none"))


if __name__ == "__main__":
    main()
