import pandas as pd
import pytest

from src.public_workflow import assign_enriched_terms, build_graph, cluster, enrich_gmt, run_rwr, validate_seeds


def test_undirected_graph_preserves_integer_node_ids_and_combines_duplicates():
    graph, audit = build_graph(pd.DataFrame({
        "source": [1, 2, 2], "target": [2, 1, 3], "weight": [0.2, 0.7, 1.0],
    }), duplicate_policy="max")
    assert set(graph) == {1, 2, 3}
    assert graph[1][2]["weight"] == 0.7
    assert audit["duplicate_rows"] == 2


def test_invalid_weights_and_unmapped_seeds_are_explicit():
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_graph(pd.DataFrame({"source": ["a"], "target": ["b"], "weight": [-1]}))
    graph, _ = build_graph(pd.DataFrame({"source": ["a"], "target": ["b"]}))
    matched, audit = validate_seeds(graph, {"a": 1, "absent": 1, "b": 0})
    assert matched == {"a": 1.0}
    assert audit["unmatched"] == ["absent"]
    assert audit["zero_weight"] == ["b"]


def test_rwr_is_normalized_and_cluster_covers_isolates():
    graph, _ = build_graph(pd.DataFrame({"source": ["a"], "target": ["b"], "weight": [1.0]}))
    graph.add_node("isolate")
    scores, _ = run_rwr(graph, {"a": 1.0})
    assert sum(scores.values()) == pytest.approx(1.0)
    membership = cluster(graph, algorithm="louvain", seed=1, resolution=1.0)
    assert set(membership) == set(graph)


def test_bh_uses_zero_overlap_eligible_terms_and_is_monotone():
    result = enrich_gmt({"a"}, {"a", "b", "c", "d"},
                        {"hit": {"a"}, "zero": {"c"}, "also_zero": {"d"}},
                        min_term_size=1, max_term_size=4)
    assert set(result.term) == {"hit", "zero", "also_zero"}
    assert result.loc[result.term == "zero", "p_value"].item() == 1.0
    ordered = result.sort_values("p_value")
    assert ordered.fdr_bh.is_monotonic_increasing
    assert result.loc[result.term == "hit", "fdr_bh"].item() == pytest.approx(.75)


def test_leiden_covers_isolate_when_available():
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    graph, _ = build_graph(pd.DataFrame({"source": ["a"], "target": ["b"], "weight": [1.0]}))
    graph.add_node("isolate")
    assert set(cluster(graph, algorithm="leiden", seed=1)) == set(graph)


def test_term_assignment_is_explicit_for_nonoverlap_nodes():
    graph, _ = build_graph(pd.DataFrame({"source": ["a"], "target": ["b"], "weight": [1.0]}))
    enrichment = pd.DataFrame([{"module": 1, "category": "GO:BP", "term_id": "GO:1", "term": "Example term",
                                "overlap": 1, "query_size": 2, "term_size": 1, "background_size": 2,
                                "p_value": .01, "fdr_bh": .01, "overlap_genes": "a"}])
    assigned, table, annotations = assign_enriched_terms(graph, {"a": 1, "b": 1}, enrichment, min_overlap=1)
    assert assigned["a"] == "GO:1: Example term"
    assert assigned["b"] == "Unassigned"
    assert annotations[1]["term"] == "Example term"
    assert table.is_assigned.sum() == 1


def test_all_node_labels_export_final_coordinates():
    from src.public_workflow import plot_network
    graph, _ = build_graph(pd.DataFrame({"source": ["a", "a", "b"], "target": ["b", "c", "c"], "weight": [1., 1., 1.]}))
    scores = {"a": .5, "b": .3, "c": .2}
    figure, _ = plot_network(graph, scores, {"a": 1, "b": 1, "c": 1}, node_names={"a": "GENEA", "b": "GENEB", "c": "GENEC"}, label_mode="all")
    data = figure.rwr_plot_nodes
    assert data.is_labelled.all()
    assert data.label_text.tolist() == ["GENEA", "GENEB", "GENEC"]
    assert data[["label_x", "label_y"]].notna().all().all()
    assert len(set(zip(data.label_x.round(8), data.label_y.round(8)))) == len(data)

