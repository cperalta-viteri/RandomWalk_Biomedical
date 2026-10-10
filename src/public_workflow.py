"""Validated, graph-agnostic RWR, clustering, GMT enrichment and plotting."""
from pathlib import Path
import math
from textwrap import shorten
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import matplotlib.patheffects as pe
try:
    from adjustText import adjust_text
except ImportError:  # The core figure remains usable when optional label repulsion is absent.
    def adjust_text(texts, **kwargs):
        return texts, []

ENRICHMENT_COLUMNS=['term','overlap','query_size','term_size','background_size','p_value','fdr_bh','overlap_genes']

def _weights(values, label):
    x=pd.to_numeric(values,errors='raise')
    if not np.isfinite(x).all() or (x<0).any(): raise ValueError(f'{label} must be finite and non-negative')
    return x

def build_graph(edges, directed=False, source='source', target='target', weight='weight', duplicate_policy='sum'):
    """Build a weighted graph and return it with a transparent audit dictionary."""
    if not {source,target}.issubset(edges.columns): raise ValueError(f'edge table needs {source!r} and {target!r}')
    if duplicate_policy not in {'sum','max','error'}: raise ValueError("duplicate_policy must be 'sum', 'max', or 'error'")
    x=edges[[source,target]+([weight] if weight in edges else [])].copy()
    if x[[source,target]].isna().any().any(): raise ValueError('edge endpoints must not be missing')
    if weight not in x: x[weight]=1.
    x[weight]=_weights(x[weight],'edge weights')
    if directed: keys=[source,target]
    else:
        # Canonicalise an undirected pair without changing the caller's node IDs.
        # Sorting on their display form gives a deterministic orientation even for
        # heterogeneous, non-orderable Python identifiers.
        pairs=[(left,right) if str(left)<=str(right) else (right,left)
               for left,right in zip(x[source],x[target])]
        x['_source']=[pair[0] for pair in pairs]; x['_target']=[pair[1] for pair in pairs]; keys=['_source','_target']
    duplicate_rows=int(x.duplicated(keys,keep=False).sum())
    if duplicate_rows and duplicate_policy=='error': raise ValueError('duplicate edges found')
    x=x.groupby(keys,as_index=False,sort=False)[weight].agg('sum' if duplicate_policy=='sum' else 'max')
    if not directed: x=x.rename(columns={'_source':source,'_target':target})
    graph=nx.from_pandas_edgelist(x,source,target,edge_attr=weight,create_using=nx.DiGraph if directed else nx.Graph)
    audit={'directed':directed,'input_rows':int(len(edges)),'duplicate_rows':duplicate_rows,'duplicate_policy':duplicate_policy,'nodes':graph.number_of_nodes(),'edges':graph.number_of_edges(),'components':nx.number_weakly_connected_components(graph) if directed else nx.number_connected_components(graph),'isolated_nodes':len(list(nx.isolates(graph)))}
    return graph,audit

def validate_seeds(graph,seeds):
    """Validate seed weights without coercing graph-node identifier types."""
    if not isinstance(seeds,dict) or not seeds: raise ValueError('seeds must be a non-empty node-to-weight mapping')
    _weights(pd.Series(seeds,dtype=float),'seed weights')
    matched={node:float(value) for node,value in seeds.items() if node in graph and value>0}
    audit={'provided':len(seeds),'matched_positive':len(matched),'unmatched':[node for node in seeds if node not in graph],'zero_weight':[node for node,value in seeds.items() if value==0]}
    if not matched: raise ValueError('no positive seed weights map to graph')
    return matched,audit

def run_rwr(graph,seeds,restart=.5,tolerance=1e-12,max_iter=1000,weight='weight'):
    """Run weighted RWR. Scores are network proximity, not causal evidence."""
    if not math.isfinite(restart) or not 0<restart<=1: raise ValueError('restart must be in (0, 1]')
    if tolerance<=0 or max_iter<1: raise ValueError('invalid convergence settings')
    matched,audit=validate_seeds(graph,seeds); total=sum(matched.values()); p={node:matched.get(node,0.)/total for node in graph}
    scores=nx.pagerank(graph,alpha=1-restart,personalization=p,dangling=p,weight=weight,tol=tolerance,max_iter=max_iter)
    audit.update({'restart':restart,'tolerance':tolerance,'max_iter':max_iter,'weight_attribute':weight}); return scores,audit

def rank_scores(scores,node_names=None):
    names=node_names or {}; x=pd.DataFrame({'node':list(scores),'rwr_probability':list(scores.values())})
    x['label']=x.node.map(names).fillna(x.node.map(str)); return x.sort_values(['rwr_probability','label'],ascending=[False,True],kind='stable').assign(rank=lambda z:range(1,len(z)+1))

def select_subgraph(graph,scores,seed_nodes,additional=100):
    """Keep seed nodes plus top non-seeds with positive probability; never pad zeros."""
    if additional<0: raise ValueError('additional must be non-negative')
    seeds=set(seed_nodes); missing=seeds-set(graph)
    if missing: raise ValueError(f'seed nodes missing from graph: {sorted(missing,key=str)[:5]}')
    ranks=rank_scores(scores); candidates=ranks[(~ranks.node.isin(seeds))&ranks.rwr_probability.gt(0)]; chosen=seeds|set(candidates.head(additional).node)
    return graph.subgraph(chosen).copy(),ranks,chosen

def cluster(graph,algorithm='louvain',seed=42,weight='weight',resolution=1.0):
    """Weighted Louvain/Leiden membership for every node, including isolates."""
    if algorithm not in {'louvain','leiden'}: raise ValueError("algorithm must be 'louvain' or 'leiden'")
    if not math.isfinite(resolution) or resolution<=0: raise ValueError('resolution must be positive')
    original=graph.to_undirected(); nodes=sorted(original,key=str)
    # Both Louvain and force layouts are sensitive to insertion order. Rebuild a
    # canonical graph so the configured seed genuinely makes repeated runs stable.
    g=nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(sorted(original.edges(data=True),key=lambda edge:(str(edge[0]),str(edge[1]))))
    if algorithm=='louvain': groups=nx.community.louvain_communities(g,weight=weight,resolution=resolution,seed=seed)
    else:
        try: import igraph as ig; import leidenalg
        except ImportError as exc: raise ImportError('Install igraph and leidenalg for Leiden') from exc
        index={node:i for i,node in enumerate(nodes)}; edges=[(index[u],index[v]) for u,v in g.edges()]; edge_weights=[float(g[u][v].get(weight,1.)) for u,v in g.edges()]
        igraph_graph=ig.Graph(n=len(nodes),edges=edges,directed=False)
        groups=[{nodes[i] for i in part} for part in leidenalg.find_partition(igraph_graph,leidenalg.RBConfigurationVertexPartition,weights=edge_weights or None,resolution_parameter=resolution,seed=seed)]
    groups=sorted(groups,key=lambda group:(-len(group),min(map(str,group)))); membership={node:i+1 for i,group in enumerate(groups) for node in group}
    if set(membership)!=set(nodes): raise RuntimeError('cluster membership does not cover every node')
    return membership

def read_gmt(path):
    terms={}
    for line in Path(path).read_text().splitlines():
        fields=line.rstrip().split('\t')
        if len(fields)>=3: terms[fields[0]]=set(fields[2:])
    return terms

def _bh(p):
    p=np.asarray(p,dtype=float); order=np.argsort(p,kind='stable'); adjusted=np.empty(len(p)); running=1.
    for rank in range(len(p),0,-1):
        index=order[rank-1]; running=min(running,p[index]*len(p)/rank); adjusted[index]=running
    return adjusted

def enrich_gmt(query,background,gmt,min_term_size=5,max_term_size=2000):
    """Hypergeometric GMT enrichment with monotone BH over all eligible terms."""
    universe=set(background); query=set(query)&universe
    if not universe: raise ValueError('background must not be empty')
    if not query: return pd.DataFrame(columns=ENRICHMENT_COLUMNS)
    terms=read_gmt(gmt) if isinstance(gmt,(str,Path)) else gmt; rows=[]
    for term,members in sorted(terms.items()):
        members=set(members)&universe
        if min_term_size<=len(members)<=max_term_size:
            hits=query&members; k=len(hits)
            rows.append({'term':term,'overlap':k,'query_size':len(query),'term_size':len(members),'background_size':len(universe),'p_value':float(hypergeom.sf(k-1,len(universe),len(members),len(query))) if k else 1.,'overlap_genes':';'.join(map(str,sorted(hits,key=str)))} )
    if not rows: return pd.DataFrame(columns=ENRICHMENT_COLUMNS)
    out=pd.DataFrame(rows).sort_values(['p_value','term'],kind='stable').reset_index(drop=True); out['fdr_bh']=_bh(out.p_value)
    return out[ENRICHMENT_COLUMNS].sort_values(['fdr_bh','p_value','term'],kind='stable').reset_index(drop=True)

def _module_spring_layout(graph, membership, seed):
    """Lay out every selected node within a deterministic, module-aware grid."""
    modules=sorted(set(membership.values())); columns=min(3,max(1,len(modules)))
    positions={}
    for order,module in enumerate(modules):
        nodes=sorted([node for node in graph if membership[node]==module],key=str)
        # Rebuild in sorted order: NetworkX's force layout follows graph insertion
        # order, so a subgraph view alone is not reproducible across equivalent runs.
        local=nx.Graph()
        local.add_nodes_from(nodes)
        local.add_edges_from(sorted(graph.subgraph(nodes).edges(data=True),key=lambda edge:(str(edge[0]),str(edge[1]))))
        local_position=nx.spring_layout(local,weight='weight',seed=seed+order,iterations=300,scale=1)
        radius=.72 if len(nodes)>10 else (.52 if len(nodes)>3 else .34)
        row,column=divmod(order,columns); center=np.array([column*2.35,row*-1.85])
        for node in nodes: positions[node]=center+local_position[node]*np.array([radius,radius*.78])
    return positions


def _repel_label_positions(labels, positions, names, iterations=240):
    """Deterministic light-weight label repulsion when adjustText is unavailable."""
    ordered=sorted(labels,key=str)
    xy=np.array([positions[node] for node in ordered],dtype=float)
    span=np.maximum(np.ptp(xy,axis=0),[1.,1.])
    placed=xy.copy(); placed[:,1]+=span[1]*.018
    widths=np.array([max(3,len(str(names.get(node,node)))) for node in ordered])*span[0]*.0065
    height=span[1]*.028
    for _ in range(iterations):
        moved=np.zeros_like(placed)
        for left in range(len(ordered)):
            for right in range(left+1,len(ordered)):
                delta=placed[right]-placed[left]
                limit_x=(widths[left]+widths[right])/2
                if abs(delta[0])<limit_x and abs(delta[1])<height:
                    direction=delta if np.any(delta) else np.array([1., .5])
                    direction=direction/(np.linalg.norm(direction) or 1.)
                    push=.004*span
                    moved[left]-=direction*push; moved[right]+=direction*push
        placed+=moved
    return {node:placed[index] for index,node in enumerate(ordered)}


def _module_annotations(enrichment, membership, min_overlap=3, fdr_threshold=.05):
    """Select one documented significant annotation per module for the figure panel."""
    if enrichment is None or len(enrichment)==0:
        return {}
    required={'module','term','overlap','query_size','fdr_bh'}
    if not required.issubset(enrichment.columns):
        raise ValueError(f'enrichment needs columns {sorted(required)} for annotation display')
    rows=enrichment[(enrichment.fdr_bh<=fdr_threshold)&(enrichment.overlap>=min_overlap)].copy()
    rows=rows.sort_values(['module','fdr_bh','p_value','term'],kind='stable')
    return {int(row.module):row.to_dict() for _,row in rows.groupby('module',sort=True).head(1).iterrows()}


def assign_enriched_terms(graph, membership, enrichment, min_overlap=3, fdr_threshold=.05):
    """Assign plotted nodes to the displayed term only when they overlap that term.

    The deterministic module annotation rule is lowest BH FDR, then p-value and
    term name, among terms meeting the fixed overlap/FDR requirements. Nodes not
    in that term's overlap are explicitly ``Unassigned`` rather than inheriting
    a pathway label from their neighbours.
    """
    annotations=_module_annotations(enrichment,membership,min_overlap,fdr_threshold)
    assignment={}; rows=[]
    for node in graph:
        module=membership[node]; record=annotations.get(module)
        overlap=set() if record is None else set(str(record.get('overlap_genes','')).split(';'))-{''}
        term='Unassigned' if record is None or str(node) not in overlap else f"{record.get('term_id','')}: {record['term']}"
        assignment[node]=term
        rows.append({'node':node,'module':module,'assigned_term':term,'is_assigned':term!='Unassigned'})
    return assignment,pd.DataFrame(rows),annotations


def plot_network(graph,scores,membership,output=None,node_names=None,seed_nodes=(),layout='module_spring',seed=42,
                 metadata=None,label_mode='key',enrichment=None,min_term_overlap=3,fdr_threshold=.05,
                 colour_by='cluster',term_assignments=None):
    """Plot every selected node with module colours, seed outlines and term annotations.

    ``label_mode='key'`` labels all seeds plus the two top non-seeds per module;
    ``'all'`` labels every plotted node.  The annotated side panel chooses the
    smallest-FDR eligible term for each module by a documented fixed rule.
    """
    if set(graph)!=set(membership): raise ValueError('membership must cover exactly the plotted graph')
    names=node_names or {}; seeds=set(seed_nodes); nodes=sorted(graph,key=str)
    if layout=='module_spring': pos=_module_spring_layout(graph,membership,seed)
    elif layout=='spring': pos=nx.spring_layout(graph,weight='weight',seed=seed,iterations=250)
    elif layout=='kamada_kawai': pos=nx.kamada_kawai_layout(graph,weight='weight')
    elif layout=='spectral': pos=nx.spectral_layout(graph,weight='weight')
    else: raise ValueError("layout must be 'module_spring', 'spring', 'kamada_kawai', or 'spectral'")
    vals=np.array([scores[n] for n in nodes]); sizes=50+450*np.sqrt((vals-vals.min())/max(vals.max()-vals.min(),1e-15))
    if colour_by not in {'cluster','enriched_term'}: raise ValueError("colour_by must be 'cluster' or 'enriched_term'")
    modules=sorted(set(membership.values())); annotations=_module_annotations(enrichment,membership,min_term_overlap,fdr_threshold)
    if colour_by=='cluster':
        node_colour={node:f'Module {membership[node]}' for node in nodes}
    else:
        node_colour=term_assignments or assign_enriched_terms(graph,membership,enrichment,min_term_overlap,fdr_threshold)[0]
        if set(node_colour)!=set(nodes): raise ValueError('term_assignments must cover exactly the plotted graph')
    categories=sorted(set(node_colour.values()),key=lambda x:(x=='Unassigned',str(x)))
    palette=plt.get_cmap('tab20',max(len(categories),1)); colours={category:('#b8c0c8' if category=='Unassigned' else palette(index)) for index,category in enumerate(categories)}
    fig,(ax,panel)=plt.subplots(1,2,figsize=(17,11.5),gridspec_kw={'width_ratios':[3.1,1.35]})
    internal=[(u,v) for u,v in graph.edges if membership[u]==membership[v]]; external=[(u,v) for u,v in graph.edges if membership[u]!=membership[v]]
    nx.draw_networkx_edges(graph,pos,edgelist=external,ax=ax,alpha=.20,width=.55,edge_color='#64748b')
    nx.draw_networkx_edges(graph,pos,edgelist=internal,ax=ax,alpha=.38,width=.65,edge_color='#64748b')
    for category in categories:
        category_nodes=[n for n in nodes if node_colour[n]==category]
        normal=[n for n in category_nodes if n not in seeds]; input_seeds=[n for n in category_nodes if n in seeds]
        nx.draw_networkx_nodes(graph,pos,nodelist=normal,node_size=[sizes[nodes.index(n)] for n in normal],node_color=[colours[category]],edgecolors='white',linewidths=.55,alpha=.94,ax=ax)
        nx.draw_networkx_nodes(graph,pos,nodelist=input_seeds,node_size=[sizes[nodes.index(n)] for n in input_seeds],node_color=[colours[category]],edgecolors='#111827',linewidths=1.7,alpha=.98,ax=ax)
    if label_mode not in {'key','all'}: raise ValueError("label_mode must be 'key' or 'all'")
    labels=set(nodes) if label_mode=='all' else set(seeds)
    if label_mode=='key':
        for module in modules:
            candidates=sorted([n for n in nodes if membership[n]==module and n not in seeds],key=lambda n:(-scores[n],str(n)))
            labels.update(candidates[:2])
    label_positions=_repel_label_positions(labels,pos,names) if labels else {}
    texts=[]; text_nodes=[]
    for node in sorted(labels,key=str):
        x,y=label_positions[node]; is_seed=node in seeds
        texts.append(ax.text(x,y+.045,names.get(node,str(node)),fontsize=8 if is_seed else 6.7,
                             fontweight='bold' if is_seed else 'normal',color='#122232' if is_seed else '#3e4b58',
                             ha='center',va='bottom',zorder=8,path_effects=[pe.withStroke(linewidth=2.4,foreground='white')]))
        text_nodes.append(node)
    if texts: adjust_text(texts,ax=ax,expand=(1.08,1.15),force_text=(.35,.5),force_static=(.15,.2),time_lim=4,
                          arrowprops=dict(arrowstyle='-',color='#74808b',lw=.4))
    final_label_positions={node:text.get_position() for node,text in zip(text_nodes,texts)}
    meta=metadata or {}; ax.set_title(meta.get('title','RWR-selected graph'),loc='left',fontweight='bold',fontsize=14)
    ax.text(.01,.01,f"{len(nodes)} selected proteins; {graph.number_of_edges()} selected edges; colour={colour_by}; layout={layout}; restart={meta.get('restart','not supplied')}; {meta.get('algorithm','cluster')} seed={seed}",transform=ax.transAxes,fontsize=8,color='#425466')
    ax.axis('off'); panel.axis('off')
    handles=[Line2D([0],[0],marker='o',color='w',label='Input seed (black outline)',markerfacecolor='white',markeredgecolor='#111827',markersize=8),Line2D([0],[0],marker='o',color='w',label='Node area: RWR probability',markerfacecolor='#94a3b8',markersize=9)]
    panel.legend(handles=handles,loc='upper left',frameon=False,fontsize=8)
    if colour_by=='cluster':
        panel.set_title('Clusters and enrichment',loc='left',fontweight='bold',fontsize=12)
        for index,module in enumerate(modules):
            y=.85-index*(.76/max(len(modules),1)); record=annotations.get(module)
            colour=colours[f'Module {module}']; term='No displayed GO:BP/Reactome term' if record is None else record['term']
            category='' if record is None else record.get('category','')
            support='' if record is None else f"{int(record['overlap'])}/{int(record['query_size'])} overlap; FDR {record['fdr_bh']:.1e}"
            module_seeds=', '.join(names.get(n,str(n)) for n in nodes if membership[n]==module and n in seeds) or 'none'
            panel.add_patch(Circle((.028,y),.014,color=colour,transform=panel.transAxes,clip_on=False))
            term_caption='No displayed GO:BP/Reactome term' if record is None else f"{record.get('term_id','')}: {shorten(str(term), width=50, placeholder='…')}"
            panel.text(.06,y,f"Module {module}: {term_caption}",transform=panel.transAxes,fontsize=8.3,fontweight='bold',va='center')
            panel.text(.06,y-.035,f"{category} | Seeds: {module_seeds}\n{support}",transform=panel.transAxes,fontsize=7.1,color='#425466',va='top')
        footer='Colours identify topology modules. Terms are one descriptive annotation per module: lowest BH FDR among terms with sufficient overlap.'
    else:
        panel.set_title('Enriched-term overlap',loc='left',fontweight='bold',fontsize=12)
        term_records=[]
        for module,record in annotations.items():
            term_key=f"{record.get('term_id','')}: {record.get('term','')}"
            if term_key in categories: term_records.append((term_key,module,record))
        items=[(category,None,None) for category in categories if category=='Unassigned']+term_records
        for index,(term_key,module,record) in enumerate(items):
            y=.85-index*(.76/max(len(items),1)); colour=colours[term_key]
            panel.add_patch(Circle((.028,y),.014,color=colour,transform=panel.transAxes,clip_on=False))
            if term_key=='Unassigned':
                panel.text(.06,y,'Unassigned',transform=panel.transAxes,fontsize=8.3,fontweight='bold',va='center')
                panel.text(.06,y-.035,'Selected protein does not overlap its module\'s displayed term.',transform=panel.transAxes,fontsize=7.1,color='#425466',va='top')
            else:
                caption=f"{record.get('term_id','')}: {shorten(str(record['term']), width=50, placeholder='…')}"
                support=f"{record.get('category','')} | {int(record['overlap'])}/{int(record['query_size'])} overlap; FDR {record['fdr_bh']:.1e}"
                panel.text(.06,y,caption,transform=panel.transAxes,fontsize=8.3,fontweight='bold',va='center')
                panel.text(.06,y-.035,support,transform=panel.transAxes,fontsize=7.1,color='#425466',va='top')
        footer='Colours identify enriched-term overlap only; they do not encode topology clusters. The displayed term is selected per module by lowest eligible BH FDR.'
    panel.text(.01,.01,footer,transform=panel.transAxes,fontsize=7.2,color='#425466',wrap=True)
    fig.tight_layout()
    figure_nodes=pd.DataFrame({'node':nodes,'x':[pos[n][0] for n in nodes],'y':[pos[n][1] for n in nodes],
                               'rwr_probability':[scores[n] for n in nodes],'node_size':[sizes[nodes.index(n)] for n in nodes],
                               'module':[membership[n] for n in nodes],'colour_category':[node_colour[n] for n in nodes],
                               'is_seed':[n in seeds for n in nodes],'is_labelled':[n in labels for n in nodes],
                               'label_text':[names.get(n,str(n)) if n in labels else '' for n in nodes],
                               'label_x':[final_label_positions.get(n,(np.nan,np.nan))[0] for n in nodes],
                               'label_y':[final_label_positions.get(n,(np.nan,np.nan))[1] for n in nodes]})
    fig.rwr_plot_nodes=figure_nodes
    fig.rwr_module_annotations=pd.DataFrame([{'module':module,**record} for module,record in annotations.items()])
    if output: fig.savefig(output,dpi=240,bbox_inches='tight')
    return fig,ax
