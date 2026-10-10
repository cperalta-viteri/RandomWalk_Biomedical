#!/usr/bin/env python3
"""Recompute the public STRING RWR worked example from raw versioned inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import date
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from src.public_workflow import (
    assign_enriched_terms, build_graph, cluster, enrich_gmt, plot_network, rank_scores, run_rwr,
    select_subgraph, validate_seeds,
)

ROOT = Path(__file__).resolve().parent
STRING_CATEGORIES = {"Biological Process (Gene Ontology)", "Reactome Pathways"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(data_dir: Path) -> list[dict]:
    manifest = json.loads((ROOT / "provenance" / "string_v12_5_source_manifest.json").read_text())
    checked = []
    for item in manifest:
        path = data_dir / item["file"]
        observed = sha256(path)
        if observed != item["sha256"]:
            raise ValueError(f"checksum mismatch for {path.name}; obtain the recorded STRING release again")
        checked.append({**item, "access_date": str(date.today()), "verified_sha256": observed})
    return checked


def load_names(data_dir: Path) -> dict:
    info = pd.read_csv(data_dir / "9606.protein.info.v12.5.txt.gz", sep="\t")
    info = info.rename(columns={"#string_protein_id": "node"})
    if info.node.duplicated().any():
        raise ValueError("STRING protein information contains duplicate protein IDs")
    return dict(zip(info.node, info.preferred_name))


def load_seed_table(path: Path, graph: nx.Graph) -> tuple[dict, pd.DataFrame, dict]:
    seeds = pd.read_csv(path, sep="\t")
    required = {"string_protein_id", "weight"}
    if not required.issubset(seeds):
        raise ValueError(f"seed TSV needs {sorted(required)}")
    if seeds.string_protein_id.isna().any() or seeds.weight.isna().any():
        raise ValueError("seed IDs and weights must not be missing")
    if seeds.string_protein_id.duplicated().any():
        raise ValueError("seed TSV has duplicate STRING protein IDs; combine them before running")
    provided = dict(zip(seeds.string_protein_id, seeds.weight))
    matched, audit = validate_seeds(graph, provided)
    seeds["mapped"] = seeds.string_protein_id.isin(graph)
    seeds["positive_weight"] = seeds.weight.astype(float).gt(0)
    if audit["unmatched"]:
        raise ValueError(f"unmapped seed IDs (not dropped): {audit['unmatched']}")
    return matched, seeds, audit


def load_string_edges(data_dir: Path, mode: str, confidence: int) -> pd.DataFrame:
    filename = "9606.protein.links.v12.5.txt.gz" if mode == "functional" else "9606.protein.physical.links.v12.5.txt.gz"
    parts = []
    for chunk in pd.read_csv(data_dir / filename, sep=r"\s+", chunksize=250_000):
        keep = chunk.loc[chunk.combined_score.ge(confidence), ["protein1", "protein2", "combined_score"]]
        parts.append(keep)
    edges = pd.concat(parts, ignore_index=True).rename(columns={"protein1": "source", "protein2": "target"})
    edges["weight"] = edges.combined_score / 1000.0
    return edges[["source", "target", "weight"]]


def load_string_terms(data_dir: Path) -> dict:
    records = []
    for chunk in pd.read_csv(data_dir / "9606.protein.enrichment.terms.v12.5.txt.gz", sep="\t", chunksize=250_000):
        records.append(chunk.loc[chunk.category.isin(STRING_CATEGORIES)])
    annotation = pd.concat(records, ignore_index=True).rename(columns={"#string_protein_id": "node"})
    terms = {}
    for (category, term_id, description), group in annotation.groupby(["category", "term", "description"], sort=True):
        # A separator that cannot collide with the displayed description.
        terms[f"{category}\x1f{term_id}\x1f{description}"] = set(group.node)
    return terms


def annotate_enrichment(membership: dict, background: set, terms: dict) -> pd.DataFrame:
    outputs = []
    for module in sorted(set(membership.values())):
        query = {node for node, assigned in membership.items() if assigned == module}
        result = enrich_gmt(query, background, terms)
        if result.empty:
            continue
        details = result.term.str.split("\x1f", n=2, expand=True)
        result.insert(0, "module", module)
        result.insert(1, "category", details[0])
        result.insert(2, "term_id", details[1])
        result["term"] = details[2]
        outputs.append(result)
    columns = ["module", "category", "term_id", "term", "overlap", "query_size", "term_size", "background_size", "p_value", "fdr_bh", "overlap_genes"]
    return pd.concat(outputs, ignore_index=True)[columns] if outputs else pd.DataFrame(columns=columns)


def run_analysis(mode: str, outdir: Path, seed_file: Path, confidence: int = 700,
                 restart: float = .5, tolerance: float = 1e-12, max_iter: int = 1000,
                 additional: int = 100, algorithm: str = "louvain", resolution: float = 1.0,
                 cluster_seed: int = 42, layout: str = "module_spring", label_mode: str = "all") -> dict:
    """Run exactly one STRING network view. Functional and physical views never mix."""
    if mode not in {"functional", "physical"}:
        raise ValueError("mode must be functional or physical")
    data_dir = ROOT / "data"
    manifest = verify_manifest(data_dir)
    edge_input = load_string_edges(data_dir, mode, confidence)
    graph, graph_audit = build_graph(edge_input, duplicate_policy="max")
    names = load_names(data_dir)
    seeds, seed_table, seed_audit = load_seed_table(seed_file, graph)
    # Preserve seed-only vertices so their mapping is visible even if a future filter isolates one.
    graph.add_nodes_from(seeds)
    graph_audit.update({"final_nodes": graph.number_of_nodes(), "final_edges": graph.number_of_edges(), "final_isolated_nodes": len(list(nx.isolates(graph)))})
    scores, rwr_audit = run_rwr(graph, seeds, restart=restart, tolerance=tolerance, max_iter=max_iter)
    selected, ranking, chosen = select_subgraph(graph, scores, seeds, additional=additional)
    membership = cluster(selected, algorithm=algorithm, seed=cluster_seed, resolution=resolution)
    enrichment = annotate_enrichment(membership, set(graph), load_string_terms(data_dir))
    term_assignments, term_assignment_table, module_annotation_map = assign_enriched_terms(selected, membership, enrichment)
    module_annotations = pd.DataFrame([{"module": module, **record} for module, record in module_annotation_map.items()])

    outdir.mkdir(parents=True, exist_ok=True)
    nodes = rank_scores(scores, names)
    nodes["is_seed"] = nodes.node.isin(seeds)
    nodes["selected"] = nodes.node.isin(chosen)
    nodes["module"] = nodes.node.map(membership)
    selected_nodes = nodes.loc[nodes.selected].copy()
    selected_edges = pd.DataFrame([{"source": source, "target": target, "weight": data["weight"]}
                                   for source, target, data in selected.edges(data=True)])
    membership_table = selected_nodes[["node", "label", "module", "is_seed", "rwr_probability", "rank"]]
    graph_for_export = selected.copy()
    nx.set_node_attributes(graph_for_export, scores, "rwr_probability")
    nx.set_node_attributes(graph_for_export, membership, "module")
    nx.set_node_attributes(graph_for_export, {node: node in seeds for node in selected}, "is_seed")
    nx.set_node_attributes(graph_for_export, {node: names.get(node, str(node)) for node in selected}, "label")
    metadata = {
        "title": f"STRING v12.5 {mode} association network: RWR-selected local graph",
        "mode": mode, "association_semantics": "functional association" if mode == "functional" else "physical association",
        "confidence_threshold": confidence, "edge_weight": "STRING combined_score / 1000; duplicate pairs use maximum score",
        "restart": restart, "tolerance": tolerance, "max_iter": max_iter, "additional_non_seed_nodes": additional,
        "selected_node_rule": "all positive-weight mapped seeds plus top positive-probability non-seeds",
        "cluster_algorithm": algorithm, "cluster_resolution": resolution, "cluster_seed": cluster_seed,
        "layout": layout, "seed_file": str(seed_file), "graph_audit": graph_audit, "seed_audit": seed_audit, "rwr_audit": rwr_audit,
        "python": sys.version, "platform": platform.platform(), "string_manifest": manifest,
    }
    edge_input.to_csv(outdir / "input_edges_after_confidence_filter.tsv", sep="\t", index=False)
    seed_table.to_csv(outdir / "seed_mapping_audit.tsv", sep="\t", index=False)
    nodes.to_csv(outdir / "nodes.tsv", sep="\t", index=False)
    ranking.to_csv(outdir / "ranks.tsv", sep="\t", index=False)
    selected_edges.to_csv(outdir / "selected_edges.tsv", sep="\t", index=False)
    membership_table.to_csv(outdir / "membership.tsv", sep="\t", index=False)
    enrichment.to_csv(outdir / "enrichment.tsv", sep="\t", index=False)
    module_annotations.to_csv(outdir / "module_display_annotations.tsv", sep="\t", index=False)
    term_assignment_table.to_csv(outdir / "term_assignments.tsv", sep="\t", index=False)
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    pd.DataFrame([{"field": key, "value": json.dumps(value, default=str) if isinstance(value, (dict, list)) else value}
                  for key, value in metadata.items()]).to_csv(outdir / "run_metadata.tsv", sep="\t", index=False)
    nx.write_graphml(graph_for_export, outdir / "selected_graph.graphml")
    plot_metadata={"title": metadata["title"], "restart": restart, "algorithm": algorithm}
    cluster_figure, _ = plot_network(selected, scores, membership, output=outdir / "network_by_cluster.png", node_names=names,
                                     seed_nodes=seeds, layout=layout, seed=cluster_seed, label_mode=label_mode,
                                     enrichment=enrichment, metadata=plot_metadata, colour_by="cluster")
    cluster_figure.savefig(outdir / "network_by_cluster.svg", bbox_inches="tight")
    cluster_figure.rwr_plot_nodes.to_csv(outdir / "plot_nodes_cluster.tsv", sep="\t", index=False)
    plt.close(cluster_figure)
    term_figure, _ = plot_network(selected, scores, membership, output=outdir / "network_by_enriched_term.png", node_names=names,
                                  seed_nodes=seeds, layout=layout, seed=cluster_seed, label_mode=label_mode,
                                  enrichment=enrichment, metadata=plot_metadata, colour_by="enriched_term",
                                  term_assignments=term_assignments)
    term_figure.savefig(outdir / "network_by_enriched_term.svg", bbox_inches="tight")
    term_figure.rwr_plot_nodes.to_csv(outdir / "plot_nodes_enriched_term.tsv", sep="\t", index=False)
    plt.close(term_figure)
    return {"graph": graph, "selected": selected, "scores": scores, "ranking": ranking, "membership": membership,
            "enrichment": enrichment, "names": names, "seeds": seeds, "metadata": metadata, "outdir": outdir}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["functional", "physical"], default="functional")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "tutorial_run")
    parser.add_argument("--seed-file", type=Path, default=ROOT / "data" / "string_worked_example_seeds.tsv")
    parser.add_argument("--confidence", type=int, default=700)
    parser.add_argument("--restart", type=float, default=.5)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--additional", type=int, default=100)
    parser.add_argument("--algorithm", choices=["louvain", "leiden"], default="louvain")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--cluster-seed", type=int, default=42)
    parser.add_argument("--layout", choices=["module_spring", "spring", "kamada_kawai", "spectral"], default="module_spring")
    parser.add_argument("--label-mode", choices=["key", "all"], default="all",
                        help="label every selected RWR-node (default), or only seeds plus top neighbours")
    parser.add_argument("--overwrite", action="store_true", help="replace the specified output directory")
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.outdir} is not empty; choose a clean directory or use --overwrite")
        shutil.rmtree(args.outdir)
    run_args = vars(args).copy()
    run_args.pop("overwrite")
    result = run_analysis(**run_args)
    print(result["outdir"])


if __name__ == "__main__":
    main()
