"""Build the executable, public-facing notebook from maintained cell content."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "pygments_lexer": "ipython3"}
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# A reproducible STRING random-walk-with-restart workflow

**Goal:** start from a plain gene-symbol seed list and a versioned STRING graph, calculate network proximity with random walk with restart (RWR), partition a prespecified local graph, annotate partitions with supplied gene sets, and make a portable figure.

\`\`\`
Choose graph → download/audit graph → validate seeds → RWR → select local graph
      → Leiden or Louvain → GMT enrichment → explicit plotting inputs → figure
\`\`\`

This is a reusable network-analysis workflow, not evidence of causal biology, target engagement, pathway activation, inhibition, biochemical flux, or mechanism. A high RWR score means proximity to the supplied seed distribution **in this graph**.

**Routes:** read the rendered \`01_string_rwr_workflow.html\`, or execute this notebook from a clean checkout. Expected time on a typical laptop is about 1–3 minutes with the bundled STRING data. Run it from \`RWR/\`; all new files go to \`results/notebook_run/\`.
""")
code("""from pathlib import Path
import shutil, sys
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / 'run_string_workflow.py').exists():
    raise RuntimeError('Start Jupyter from the RWR directory, not from vignettes/.')
sys.path.insert(0, str(ROOT))
from run_string_workflow import (verify_manifest, load_names, load_string_edges, load_seed_table,
                                 load_string_terms, annotate_enrichment)
from src.public_workflow import assign_enriched_terms, build_graph, run_rwr, rank_scores, select_subgraph, cluster, plot_network

DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'notebook_run'
if OUT.exists():
    shutil.rmtree(OUT)  # dedicated generated output; guarantees no stale tables are reused
OUT.mkdir(parents=True)
print(f'Working directory: {ROOT}')
print(f'Fresh output directory: {OUT}')""")

md("""## 1. Choose, load and audit the exact STRING graph

**Choose the graph here.** `STRING_VIEW = 'functional'` selects the STRING functional-association graph: an edge means STRING evidence supports a functional association, not necessarily direct physical contact. `STRING_VIEW = 'physical'` selects the separate STRING physical-association graph: an edge means evidence for physical association. These are distinct input files and analyses; they are never combined.

The table and code below show the selected graph resource before it is loaded. We retain edges with combined score ≥ 700 and transform confidence to `weight = combined_score / 1000`. For duplicate undirected pairs, the maximum confidence is retained. Missing endpoints and invalid, non-finite, or negative weights are rejected.

The bundled manifest is verified here and records the official versioned URL, checksum, and verification date for the files used in this run.
""")
code("""STRING_VIEW = 'functional'  # Choose 'functional' or 'physical' here.
MODE = STRING_VIEW  # short display label used in later cells
CONFIDENCE = 700

manifest = verify_manifest(DATA)
display(pd.DataFrame(manifest)[['file', 'url', 'sha256', 'access_date']])

string_views = pd.DataFrame([
    {'view': 'functional', 'STRING file': '9606.protein.links.v12.5.txt.gz',
     'edge meaning': 'functional association; not necessarily direct physical contact'},
    {'view': 'physical', 'STRING file': '9606.protein.physical.links.v12.5.txt.gz',
     'edge meaning': 'physical association'},
])
selected_view = string_views.query('view == @STRING_VIEW')
if selected_view.empty:
    raise ValueError("STRING_VIEW must be 'functional' or 'physical'")
display(selected_view)

edge_input = load_string_edges(DATA, STRING_VIEW, CONFIDENCE)
graph, graph_audit = build_graph(edge_input, directed=False, duplicate_policy='max')
names = load_names(DATA)
display(edge_input.head())  # actual retained edges from the selected STRING resource
pd.Series(graph_audit | {'selected_STRING_view': STRING_VIEW, 'confidence_threshold': CONFIDENCE}, name='value').to_frame()""")

md("""## 2. Start with a gene-symbol list; create and validate the seed table

Replace only the symbols in `SEED_GENES` with your own list. The notebook maps them to STRING protein IDs, reports missing or ambiguous mappings, stops if either occurs, and creates the required TSV itself. It never silently drops a supplied gene. Equal weights are used here; change `weight` only if you have a justified non-negative weighting scheme.

Seeds are *measured or directly supplied starting points*. Nodes reached by RWR are graph-inferred neighbours, not additional direct evidence.
""")
code("""SEED_GENES = '''
SPARC SRM STC2 PTTG1 FGF1 POU5F1B HMMR COL23A1 DPYSL3 GFPT2 PDGFRB
'''.split()  # Replace this list for your analysis.

protein_info = pd.read_csv(DATA / '9606.protein.info.v12.5.txt.gz', sep='\\t').rename(columns={'#string_protein_id': 'string_protein_id'})
seed_table = protein_info[protein_info.preferred_name.isin(SEED_GENES)][['string_protein_id', 'preferred_name']].rename(columns={'preferred_name': 'gene'})
missing = sorted(set(SEED_GENES) - set(seed_table.gene))
ambiguous = seed_table[seed_table.gene.duplicated(keep=False)].sort_values('gene')
if missing or not ambiguous.empty:
    raise ValueError(f'Mapping stopped. Missing genes: {missing}; ambiguous mappings:\\n{ambiguous}')
seed_table['weight'] = 1.0
seed_table = seed_table.set_index('gene').loc[SEED_GENES].reset_index()
seed_file = OUT / 'seeds_mapped_from_gene_list.tsv'
seed_table.to_csv(seed_file, sep='\\t', index=False)
seeds, seed_audit_table, seed_audit = load_seed_table(seed_file, graph)
graph.add_nodes_from(seeds)
display(pd.DataFrame([seed_audit]))
seed_audit_table""")

md("""## 3. Run RWR and select a local graph

RWR restarts at the normalized seed distribution with probability 0.50. It uses weighted edges, tolerance \`1e-12\`, and at most 1,000 iterations. The prespecified display rule is: **all positive-weight mapped seeds plus the 100 highest-scoring positive-probability non-seeds**. It is a display/annotation subset, not a new biological test.
""")
code("""RESTART, TOLERANCE, MAX_ITER, ADDITIONAL = .5, 1e-12, 1000, 100
scores, rwr_audit = run_rwr(graph, seeds, restart=RESTART, tolerance=TOLERANCE, max_iter=MAX_ITER)
selected, ranking, chosen = select_subgraph(graph, scores, seeds, additional=ADDITIONAL)
ranking['is_input_seed'] = ranking.node.isin(seeds)
ranking['selected_for_local_graph'] = ranking.node.isin(chosen)
display(ranking.head(15))
pd.DataFrame([rwr_audit])""")

md("""The table above makes provenance visible: \`is_input_seed=True\` denotes a supplied seed; otherwise it is network-inferred. RWR probability is not a confidence of targeting and has no effect direction.

## 4. Cluster the selected local graph

Louvain and Leiden are both available topology partitions of exactly the same prespecified selected graph. Choose one **before** inspecting biological output; the notebook runs only that chosen method. Both use edge weights, a fixed seed and resolution 1.0; every selected node, including any isolate, receives exactly one membership. This worked example prespecifies Louvain; change \`ALGORITHM\` to \`leiden\` if that is your prespecified choice.
""")
code("""CLUSTER_SEED, RESOLUTION, ALGORITHM = 42, 1.0, 'louvain'  # or 'leiden'; run one prespecified method
membership = cluster(selected, algorithm=ALGORITHM, seed=CLUSTER_SEED, resolution=RESOLUTION)
nodes_sorted = sorted(selected, key=str)
pd.DataFrame({'node': nodes_sorted, 'gene': [names.get(n, str(n)) for n in nodes_sorted],
              'selected_partition': [membership[n] for n in nodes_sorted],
              'algorithm': ALGORITHM}).head(20)""")

md("""## 5. Enrich modules with a supplied GMT-style collection

A GMT line is \`term<TAB>description<TAB>gene1<TAB>gene2...\`. The generic \`enrich_gmt()\` function accepts a GMT path or \`{term: set(nodes)}\` mapping. This STRING example constructs such a mapping from the versioned STRING GO:BP and Reactome annotation file. The background is the final graph-node universe.

For each module, the function tests every eligible term (size 5–2,000 in the graph universe), including zero-overlap terms, by a one-sided hypergeometric test; then it applies a monotone Benjamini–Hochberg correction across that whole eligible family. Enrichment is descriptive annotation of selected nodes, not proof of pathway activity.
""")
code("""terms = load_string_terms(DATA)
enrichment = annotate_enrichment(membership, set(graph), terms)
enrichment.query('fdr_bh < 0.05').head(15)""")

md("""## 6. Understand the plotting inputs

Before plotting, it helps to separate the biological inputs from the display inputs. The code below makes each one visible so you can inspect or replace it safely:

- **`selected`** is the RWR-selected local graph. It supplies the nodes and induced STRING edges that will be drawn. It does not contain the full human STRING network.
- **`scores`** is the RWR-probability dictionary for all graph nodes. In the plot, a larger node means greater network proximity to the input seeds; it does not mean a stronger biological effect.
- **`membership`** assigns each selected node to one Louvain or Leiden community. It is used to calculate enrichment within communities, but it is not used as node colour in this tutorial.
- **`names`** maps STRING protein IDs to readable gene symbols. It determines the node labels.
- **`seeds`** is the set of mapped input proteins. It determines the black node outline.
- **`term_assignments`** maps each selected node to its displayed enriched-term label or `Unassigned`. It determines node colour: coloured means overlap with the selected community term; grey means no overlap with that term.

The following cell converts those objects into inspectable node/edge tables and dictionaries before the plotting call.
""")
code("""plot_node_table = ranking.loc[ranking.node.isin(chosen), ['node', 'label', 'rwr_probability', 'rank', 'is_input_seed']].copy()
plot_edge_table = pd.DataFrame([(u, v, d['weight']) for u, v, d in selected.edges(data=True)],
                               columns=['source', 'target', 'weight'])
plot_score_dictionary = scores
plot_cluster_membership = membership
plot_display_labels = {node: names.get(node, str(node)) for node in selected}
plot_seed_set = set(seeds)
plot_enriched_term_annotation = enrichment.query('fdr_bh < 0.05')[['module', 'category', 'term_id', 'term', 'fdr_bh']]
display(plot_node_table.head(), plot_edge_table.head(), plot_enriched_term_annotation.head())
print('score dictionary:', len(plot_score_dictionary), 'nodes; cluster membership:', len(plot_cluster_membership), 'nodes')""")

md("""## 7. Plotting graph communities

Spring, Kamada–Kawai and spectral layouts change only drawing coordinates; they do not change RWR, partitioning or enrichment. The default \`module_spring\` is a deterministic, component-aware readability layout. Every protein and every induced edge in the selected local graph is drawn and every selected protein is labelled by default. Label repulsion is applied by \`adjustText\`; \`label_mode='key'\` is available only for a compact alternative.

This notebook uses one colour scheme: enriched-term overlap. A node receives a colour only when it overlaps its module's displayed GO:BP/Reactome term; grey means it does not. Node area is RWR probability and black outline is an input seed. Cluster membership is not a visible colour encoding.
""")
code("""LAYOUT = 'module_spring'  # also: 'spring', 'kamada_kawai', 'spectral'
term_assignments, term_assignment_table, module_annotation_map = assign_enriched_terms(selected, membership, enrichment)
plot_metadata = {'title': f'STRING v12.5 {MODE} network: RWR-selected local graph',
                 'restart': RESTART, 'algorithm': ALGORITHM}
term_figure, _ = plot_network(selected, scores, membership, output=OUT / 'network_by_enriched_term.png', node_names=names,
    seed_nodes=seeds, layout=LAYOUT, seed=CLUSTER_SEED, label_mode='all', enrichment=enrichment,
    metadata=plot_metadata, colour_by='enriched_term', term_assignments=term_assignments)""")

md("""### Independent physical STRING view

Functional association and physical association answer different network questions and are never combined. This cell reruns the exact same prespecified workflow against the physical STRING file and writes a separate, complete output directory.
""")
code("""from run_string_workflow import run_analysis
PHYSICAL_OUT = ROOT / 'results' / 'notebook_physical_run'
if PHYSICAL_OUT.exists():
    shutil.rmtree(PHYSICAL_OUT)
physical_result = run_analysis('physical', PHYSICAL_OUT, seed_file, confidence=CONFIDENCE, restart=RESTART,
    tolerance=TOLERANCE, max_iter=MAX_ITER, additional=ADDITIONAL, algorithm=ALGORITHM,
    resolution=RESOLUTION, cluster_seed=CLUSTER_SEED, layout=LAYOUT, label_mode='all')
physical_term_assignments, _, _ = assign_enriched_terms(physical_result['selected'], physical_result['membership'], physical_result['enrichment'])
physical_figure, _ = plot_network(physical_result['selected'], physical_result['scores'], physical_result['membership'],
    node_names=physical_result['names'], seed_nodes=physical_result['seeds'], layout=LAYOUT, seed=CLUSTER_SEED,
    label_mode='all', enrichment=physical_result['enrichment'], colour_by='enriched_term',
    term_assignments=physical_term_assignments,
    metadata={'title': 'STRING v12.5 physical association network: RWR-selected local graph', 'restart': RESTART, 'algorithm': ALGORITHM})
print('This is an independently rerun physical-association RWR local graph, not a validation or overlay of the functional view.')""")

md("""### Read-only previews

The rendered HTML bundles fresh, separately calculated functional and physical previews. They are illustrative: rerun the preceding cells after changing graph, seeds, or parameters; no saved rank, enrichment, or figure is reused by the workflow.

![Functional STRING RWR-selected local graph; all selected proteins are drawn, node size is RWR probability, node colour is enriched-term overlap, black outline indicates input seed.](../results/functional_run/network_by_enriched_term.png)

![Physical STRING RWR-selected local graph; all selected proteins are drawn independently from functional STRING edges, node size is RWR probability, node colour is enriched-term overlap, black outline indicates input seed.](../results/physical_run/network_by_enriched_term.png)
""")

md("""## 8. Export this notebook run

This notebook—not a command line—is the primary workflow. It writes the exact functional graph, seeds, ranks, local-node table, edges, memberships, enrichment, term assignments, plotting inputs and figures calculated in the cells above. Use the optional CLI notebook only later if you need unattended or batch runs.
""")
code("""functional_nodes = ranking.loc[ranking.node.isin(chosen)].copy()
functional_nodes['gene'] = functional_nodes.node.map(names)
functional_nodes['module'] = functional_nodes.node.map(membership)
functional_edges = pd.DataFrame([(u, v, data['weight']) for u, v, data in selected.edges(data=True)], columns=['source', 'target', 'weight'])
functional_nodes.to_csv(OUT / 'functional_selected_nodes.tsv', sep='\\t', index=False)
functional_edges.to_csv(OUT / 'functional_selected_edges.tsv', sep='\\t', index=False)
ranking.to_csv(OUT / 'functional_ranks.tsv', sep='\\t', index=False)
enrichment.to_csv(OUT / 'functional_enrichment.tsv', sep='\\t', index=False)
term_assignment_table.to_csv(OUT / 'functional_term_assignments.tsv', sep='\\t', index=False)
term_figure.rwr_plot_nodes.to_csv(OUT / 'functional_plot_nodes_enriched_term.tsv', sep='\\t', index=False)
print('Notebook-generated files:')
for path in sorted(OUT.iterdir()): print(path.name)""")

md("""**Interpretation limits:** graph neighbours are inferred from STRING associations; enrichment terms label annotation overlap. Neither establishes activity, inhibition, mutation, flux, direction of effect, or causality. Record the graph release, edge semantics, filtering, seed provenance, parameters and GMT version whenever you compare runs.
""")

nb['cells'] = cells
nb['metadata']['title'] = 'STRING RWR workflow'
nbf.write(nb, ROOT / 'vignettes' / '01_string_rwr_workflow.ipynb')

# Deliberately separate optional automation material from the notebook-first tutorial.
cli = nbf.v4.new_notebook()
cli.metadata.kernelspec = nb.metadata.kernelspec
cli.metadata.language_info = nb.metadata.language_info
cli.cells = [
    nbf.v4.new_markdown_cell("""# Optional: run the STRING RWR workflow from the command line

This is an automation/batch route only. Start with `01_string_rwr_workflow.ipynb`, which maps a gene list, validates the seed table, performs the analysis and explains every output in Jupyter.

The CLI accepts the TSV created by that notebook. Functional and physical STRING associations are independent modes and must never be combined.
"""),
    nbf.v4.new_markdown_cell("""## Inputs

Use the notebook-created `seeds_mapped_from_gene_list.tsv`, containing `string_protein_id`, `gene`, and non-negative `weight`. The runner refuses unmapped IDs and non-empty output directories unless `--overwrite` is explicit.
"""),
    nbf.v4.new_code_cell("""# Run these in a terminal from RWR/; they are shown here only for optional automation.
# python run_string_workflow.py --mode functional --seed-file results/notebook_run/seeds_mapped_from_gene_list.tsv --outdir results/functional_run
# python run_string_workflow.py --mode physical --seed-file results/notebook_run/seeds_mapped_from_gene_list.tsv --outdir results/physical_run
"""),
    nbf.v4.new_markdown_cell("""## Outputs

Each mode writes separate graph/rank/membership/enrichment/plot-input/metadata TSVs and cluster- and enriched-term-coloured PNG/SVG figures. `--algorithm leiden` selects Leiden after installing `igraph` and `leidenalg`; otherwise the default is Louvain. See `INSTALLME.md` for the clean-environment verification.
"""),
]
cli.metadata['title'] = 'Optional STRING RWR command-line automation'
nbf.write(cli, ROOT / 'vignettes' / '02_optional_cli_runner.ipynb')

