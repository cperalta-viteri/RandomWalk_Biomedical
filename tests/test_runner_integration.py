"""End-to-end CLI contracts for the independent STRING network modes."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RWR_ROOT = Path(__file__).resolve().parents[1]


def run_mode(tmp_path, mode):
    outdir = tmp_path / mode
    subprocess.run(
        [sys.executable, "run_string_workflow.py", "--mode", mode, "--outdir", str(outdir),
         "--additional", "5", "--label-mode", "all"],
        cwd=RWR_ROOT, check=True, capture_output=True, text=True, timeout=180,
    )
    return outdir


def test_functional_and_physical_cli_runs_are_independent_and_complete(tmp_path):
    functional = run_mode(tmp_path, "functional")
    physical = run_mode(tmp_path, "physical")
    required = {
        "nodes.tsv", "ranks.tsv", "selected_edges.tsv", "membership.tsv", "enrichment.tsv",
        "term_assignments.tsv", "module_display_annotations.tsv", "plot_nodes_cluster.tsv",
        "plot_nodes_enriched_term.tsv", "network_by_cluster.png", "network_by_cluster.svg",
        "network_by_enriched_term.png", "network_by_enriched_term.svg",
        "run_metadata.json", "run_metadata.tsv",
    }
    for mode, outdir in (("functional", functional), ("physical", physical)):
        assert required <= {path.name for path in outdir.iterdir()}
        metadata = json.loads((outdir / "run_metadata.json").read_text())
        assert metadata["mode"] == mode
        selected = pd.read_csv(outdir / "membership.tsv", sep="\t")
        plot = pd.read_csv(outdir / "plot_nodes_cluster.tsv", sep="\t")
        assert len(selected) == len(plot)
        assert plot.is_labelled.all()
        assert plot.label_text.astype(bool).all()
        assert plot[["label_x", "label_y"]].notna().all().all()
        # Coordinates and texts are intentionally exported for all-node-label QA.
        assert plot.label_text.nunique() == len(plot)
    functional_edges = pd.read_csv(functional / "selected_edges.tsv", sep="\t")
    physical_edges = pd.read_csv(physical / "selected_edges.tsv", sep="\t")
    assert not functional_edges.equals(physical_edges)

