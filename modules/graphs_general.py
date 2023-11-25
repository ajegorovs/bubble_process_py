
from re import A
import networkx as nx, cv2, numpy as np, itertools, copy, pickle
from matplotlib import pyplot as plt
from collections import defaultdict
from bubble_params import (centroid_area_cmomzz)
from misc import (cyclicColor, order_segment_levels, prep_combs_clusters_from_nodes)

#G = None
def extractNeighborsNext(graph, node, time_from_node_function):
    neighbors = list(graph.neighbors(node))
    return [n for n in neighbors if time_from_node_function(n) > time_from_node_function(node)]

def extractNeighborsPrevious(graph, node, time_from_node_function):
    neighbors = list(graph.neighbors(node))
    return [n for n in neighbors if time_from_node_function(n) < time_from_node_function(node)]

# define parameter retrieval functions for nodes.
# in main code these are redined into G_parameter(node) by specifying reference graph G/G2
# node time can be retrieved like 'seg_t_start = lambda node, graph : node[0]'
# but holding parameters on a graph is more universal.
# by defining these more general functions i want to get rid of specific string keys that have to typed in manually
seg_t_start     = lambda node, graph : graph.nodes[node]['t_start'   ]
seg_t_end       = lambda node, graph : graph.nodes[node]['t_end'     ]
seg_n_from      = lambda node, graph : graph.nodes[node]['node_start']    
seg_n_to        = lambda node, graph : graph.nodes[node]['node_end'  ] 
seg_edge_d      = lambda edge, graph : graph.edges[edge]['dist'      ]

node_time       = lambda node, graph : graph.nodes[node]['time'      ]
node_area       = lambda node, graph : graph.nodes[node]['area'      ]
node_centroid   = lambda node, graph : graph.nodes[node]['centroid'  ]
node_owner      = lambda node, graph : graph.nodes[node]['owner'     ]

def node_owner_set(node, graph, owner):
    graph.nodes[node]["owner"] = owner

def graph_extract_paths(G, min_length = 2, f_sort = lambda x: (x[0], x[1:])):
    solo_edge_forw      = set()
    one_2_one_edges     = set()
    for t_from in G.nodes():
        t_successors = list(G.successors(t_from))
        if len(t_successors)  == 1:  
            solo_edge_forw.add((t_from, t_successors[0]))

    for t_from, t_to in solo_edge_forw:
        t_predecessors = list(G.predecessors(t_to))
        if len(t_predecessors) == 1:
            one_2_one_edges.add((t_from, t_to))

    G_chain = nx.Graph()
    G_chain.add_edges_from(one_2_one_edges)

    return sorted(
                    [
                        sorted(c, key = f_sort) for c in nx.connected_components(G_chain) 
                        if len(c) > min_length
                    ]
                , key = f_sort)

def graph_extract_paths_old(H,f):
    nodeCopy = list(H.nodes()).copy()
    segments2 = {a:[] for a in nodeCopy}
    resolved = []
    skipped = []
    for node in nodeCopy:
        goForward = True if node not in resolved else False
        nextNode = node
        prevNode = None
        while goForward == True:
            
            nextNodes = list(H.successors(nextNode))
            prevNodes = list(H.predecessors(nextNode))

            # find if next node exists and its single
            soloNext    = True if len(nextNodes) == 1 else False
            soloPrev    = True if len(prevNodes) == 1 else False # or prevNode is None
            soloPrev2   = True if soloPrev and (prevNode is None or prevNodes[0] == prevNode) else False

            # if looking one step ahead, starting node can have one back and/or forward connection to split/merge
            # this would still mean that its a chain and next/prev node will be included.
            # to fix this, check if next/prev are merges/splits
            # find if next is not merge:
            nextNotMerge = False
            if soloNext:
                nextPrevNodes = list(H.predecessors(nextNodes[0]))
                if len(nextPrevNodes) == 1: 
                    nextNotMerge = True

            nextNotSplit = False
            if soloNext:
                nextNextNodes = list(H.successors(nextNodes[0]))
                if len(nextNextNodes) <= 1:   # if it ends, it does not split. (len = 0)
                    nextNotSplit = True

            prevNotSplit = False
            if soloPrev2:
                prevNextNodes = list(H.successors(prevNodes[0]))
                if len(prevNextNodes) == 1:
                    prevNotSplit = True


            saveNode = False
            # test if it is a chain start point:
            # if prev is a split, implies only one prevNode 
            if prevNode is None:                # starting node
                if len(prevNodes) == 0:         # if no previos node =  possible chain start
                    if nextNotMerge:            # if it does not change into merge, it is good
                        saveNode = True
                    else:
                        skipped.append(node)
                        goForward = False
                elif not prevNotSplit:
                    if nextNotMerge:
                        saveNode = True
                    else: 
                        skipped.append(node)
                        goForward = False
                else:
                    skipped.append(node)
                    goForward = False
            else:
            # check if its an endpoint
                # dead end = zero forward neigbors
                if len(nextNodes) == 0:
                    saveNode = True
                    goForward = False
                # end of chain =  merge of forward neigbor
                elif not nextNotMerge:
                    saveNode = True
                    goForward = False
                elif not nextNotSplit:
                    saveNode = True
                
                # check if it is part of a chain
                elif nextNotMerge:
                    saveNode = True


            if saveNode:
                segments2[node].append(nextNode)
                resolved.append(nextNode)
                prevNode = nextNode
                if goForward :
                    nextNode = nextNodes[0]

    
    
    return segments2, skipped

def find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = False, G = None, edges = None, only_subIDs = False, max_paths = 20000):
    # extract all possible paths from set of nodes nodes_start to set of nodes nodes_end
    # give edges and set construct_graph = True to create a graph, alternatively supply graph G
    fail = None
    #many_paths = False
    if construct_graph:
        G = nx.DiGraph()
        G.add_edges_from(edges)

    all_paths = []
    fail_all = {t:False for t in nodes_start}
    for start_node in nodes_start:
        for end_node in nodes_end:
            G.add_nodes_from([start_node,end_node]) # for safety. they might not be in. nx.has_path will throw an error
            if nx.has_path(G, source = start_node, target = end_node):    # otherwise it will fail to find path.
                paths = []
                for path in nx.all_simple_paths(G, source = start_node, target = end_node):
                    paths.append(path)
                    if len(paths) >= max_paths:
                        fail = 'to_many_paths'
                        break
                if only_subIDs:
                    paths = [[tuple(subIDs) for t,*subIDs in node] for node in paths]
                all_paths.extend(paths)
            else: fail_all[start_node] = True
    if all(fail_all.values()): fail = 'no_path'       # if all paths failed seq = []      
    return all_paths, fail

def comb_product_to_graph_edges(choices_list, thresh_func):
    # to construct all possible branches from list of choices use itertools.product(*choices_list)
    # what if some edges in branch violate a criterion? that branch should not be considered.
    # go though
    output_edges = []
    choice_from = choices_list[0]                               # start with choice 1
    for i in list(range(1, len(choices_list))):                  
        choice_to = choices_list[i]                             # take next choice
        conns = list(itertools.product(choice_from,choice_to))  # find all connections from prev to next choices
        conns_viable = [conn for conn in conns if thresh_func(conn)] # check constr.
        output_edges += conns_viable                            # save viable edges
        choice_from = set([elem_2 for _,elem_2 in conns_viable])# basically refine choice_to

    return output_edges, choices_list[0], choice_from

def extract_graph_connected_components(graph, sort_function = lambda x: x): 
    # extract all conneted component= clusters from graph. to extract unique clusters,
    # all options have to be sorted to drop identical. sorting can be done by sort_function.
    # for nodes with names integers, used lambda x: x, for names as tuples use lambda x: (x[0], *x[1:])
    # where node name is (timeX, A, B, C,..), it will sort  Time first, then by min(A,B,C), etc
    connected_components_all = [list(nx.node_connected_component(graph, key)) for key in graph.nodes()]
    connected_components_all = [sorted(sub, key = sort_function) for sub in connected_components_all] 
    connected_components_unique = []
    [connected_components_unique.append(x) for x in connected_components_all if x not in connected_components_unique]
    return connected_components_unique

#def extract_graph_connected_components_autograph(edges, sort_function_in = lambda x: x, sort_function_out = lambda x: x): 
#    graph = nx.Graph()
#    graph.add_edges_from(edges)
#    return  sorted([sorted(c, key = sort_function_out) for c in nx.connected_components(graph)] , key = sort_function_in)
    

def extract_clusters_from_edges(edges, nodes= [], sort_f_in = lambda x: x, sort_f_out = lambda x: x[0]): 
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return  sorted([sorted(c, key = sort_f_in) for c in nx.connected_components(graph)] , key = sort_f_out)
    

def find_segment_connectivity_isolated(graph, t_min, t_max, add_nodes, active_segments, source_segment, source_node = None, target_nodes = None, return_cc = False):
    # test connectivity of source segments to nearby segments (within certain time interval - [t_max, t_min]
    # to avoid paths that go though other segments, active at that time, temporary subgraph is created
    # from which all but one test target is removed.

    # t_min, t_max & add_nodes for main subgraph; active_segments, target_nodes dict for targets; source_segment, source_node for pathing.
    # NOTE: cc_all sorting is created for nodes (time,ID1,...)
    G_time      = lambda node   : node_time(   node  , graph)
    G_owner     = lambda node   : node_owner(  node  , graph)
    
    nodes_keep_main =   [node for node in graph.nodes() if t_min <= G_time(node) <= t_max]    # isolate main graph segment
    nodes_keep_main +=  add_nodes                                                                          # add extra nodes, if needed
    subgraph_main   =   graph.subgraph(nodes_keep_main)

    if source_node is None:  # automatically find source node if not specified
        for node in nodes_keep_main:
            if G_owner(node) == source_segment:
                source_node = node
                break

    # setup target segments. either known from target_nodes dict or from all time period active (active_segments)
    if type(target_nodes) == dict   : test_segments = set(target_nodes.keys())  - {source_segment}
    else                            : test_segments = set(active_segments)      - {source_segment}
    
    connected_edges = []
    out_cc = defaultdict(list)
    for test_segment in test_segments:                              # go though all active segments except original source
        segments_keep_sub = (None, source_segment, test_segment)    # no owner, source or test segments are selected for subgraph
        nodes_keep_sub    = [node for node in subgraph_main.nodes() if subgraph_main.nodes[node]["owner"] in segments_keep_sub]
        subgraph_test     = subgraph_main.subgraph(nodes_keep_sub)  # subgraph with deleted irrelevant segments

        # find target node from known dict, find it from owner segment or its specified explicitly
        if type(target_nodes) == dict and test_segment in target_nodes:   
            test_target_node = target_nodes[test_segment]
        elif target_nodes is None:
            for node in nodes_keep_sub:
                if G_owner(node) == test_segment:
                    test_target_node = node
                    break
        else: test_target_node = target_nodes
            
        hasPath = nx.has_path(subgraph_test.to_undirected(), source = source_node, target = test_target_node)

        if hasPath:
            connected_edges.append((source_segment,test_segment))
            if return_cc:
                nodes_from  = [node for node in nodes_keep_sub if G_owner(node) == source_segment]
                nodes_to    = [node for node in nodes_keep_sub if G_owner(node) == test_segment]

                node_from_last  = max(nodes_from    , key = lambda x: G_time(x))
                node_to_first    = min(nodes_to     , key = lambda x: G_time(x))

                sub_test_nodes = [node for node in nodes_keep_sub if G_time(node_from_last) < G_time(node) < G_time(node_to_first)]
                sub_test_nodes += [node_from_last, node_to_first]

                sub_subgraph_test = subgraph_test.subgraph(sub_test_nodes) 
                cc_all = extract_graph_connected_components(sub_subgraph_test.to_undirected(),  lambda x: (x[0], *x[1:]))
                cc_sol = [cc for cc in cc_all if source_node in cc]
                if len(cc_sol) == 0:cc_sol = [None]
                out_cc[(source_segment,test_segment)] = cc_sol[0]

    if return_cc: return connected_edges, out_cc
    else: return connected_edges


def drawH(H, paths, node_positions, fixed_nodes = [], show = True, suptitle = "drawH", figsize = ( 10,5), node_size = 30, edge_width_path = 3, edge_width = 1, font_size = 7):
    colors = {i:np.array(cyclicColor(i))/255 for i in paths}
    colors = {i:np.array([R,G,B]) for i,[B,G,R] in colors.items()}
    colors_edges2 = {}
    width2 = {}
    # set colors to different chains. iteratively sets default color until finds match. slow but whatever
    for u, v in H.edges():
        for i, path in paths.items():
            if u in path and v in path:

                colors_edges2[(u,v)] = colors[i]
                width2[(u,v)] = edge_width_path
                break
            else:

                colors_edges2[(u,v)] = np.array((0.5,0.5,0.5))
                width2[(u,v)] = edge_width

    nx.set_node_attributes(H, node_positions, 'pos')

    pos         = nx.get_node_attributes(H, 'pos')
    weight = {t_edge:0 for t_edge in H.edges()}
    #pos = custom_spring_layout(H, pos=node_positions, fixed=fixed_nodes)
    label_offset= 0.05
    lable_pos   = {k: (v[0], v[1] + (-1)**(k[0]%2) * label_offset) for k, v in pos.items()}
    labels      = {node: f"{node}" for node in H.nodes()}
    edge_width = list(width2.values())
    edge_color = list(colors_edges2.values())
    if show: 
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(suptitle, fontsize=10)
        nx.draw_networkx_edges( H, pos, alpha=0.7, width = edge_width, edge_color = edge_color)
        nx.draw_networkx_nodes( H, pos, node_size = node_size, node_color='lightblue', alpha=1)
        #label_options = {"ec": "k", "fc": "white", "alpha": 0.3}
        nx.draw_networkx_labels(H, pos = lable_pos, labels=labels, font_size=font_size)#, bbox=label_options
        #nx.draw(H, pos, with_labels=False, node_size=50, node_color='lightblue',font_size=6,
        #        font_color='black', edge_color=list(colors_edges2.values()), width = list(width2.values()))
        plt.show()
    
    return edge_width, edge_color, node_size
def check_overlap(a, b):
    """
    https://stackoverflow.com/questions/2953967/built-in-function-for-computing-overlap-in-python
    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])

def ranges_overlap(range1, range2):
    return range1.stop > range2.start and range2.stop > range1.start

def for_graph_plots(G, segs = [], optimize_pos = True, show = True, suptitle = "drawH", node_size = 30, edge_width_path = 3, edge_width = 1, font_size = 7):
    import time as time_lib
    from graphs_node_position_spring_lj_model_iterator import (prep_matrices, integration, draw_plot)
    #if len(segs) == 0:
    #    segments3 = graph_extract_paths(G) # default min_length
    #else:
    #    segments3 = [s for s in segs if len(s) > 0 ]
    segments3 = segs
    #paths = {i:vals for i,vals in enumerate(segments3)}
    paths = {i:vals for i,  vals in enumerate(segments3) if len(vals) > 0}
    t_pos = {i:0    for i,  vals in enumerate(segments3) if len(vals) > 0}
    # check overlap between segments time-wise: extract overlapping clusters

    G_time  = lambda node   : node_time(   node  , G)
    G_owner = lambda node   : node_owner(  node  , G)
   

    time_intervals = {i:range(G_time(nodes[0]), G_time(nodes[-1]) + 1) for i, nodes in paths.items()}

    # check all pairs of segments for overlap. keep overlapping
    pairs_overlap = [(a,b) for (a,b) in itertools.combinations(paths.keys(), 2) 
                     if ranges_overlap(time_intervals[a],time_intervals[b])]
    # extract clusters of overlapping segments. different clusters dont overlap. 
    # by 'jumping' from one segment in cluster to other you can reach form min to max cluster time
    # cluster elements are sorted by internal index and clusters themself are sorted by first element.
    # it should inherit proper order with which they were found on graph.
    overlapping_clusters = extract_clusters_from_edges(edges = pairs_overlap, nodes = paths)
    # determine maximal number of layers in all clusters
    max_layers = max([len(subIDs) for subIDs in overlapping_clusters])
    # create empty layers for all clusters. global max layers for every cluster storage
    positions = {c:{layer:[] for layer in range(max_layers)} for c,_ in enumerate(overlapping_clusters)}
    # distribute cluster elements on layers in ladder fashion from lowest to highest
    for cluster_ID, subIDs in enumerate(overlapping_clusters):
        for layer, subID in enumerate(subIDs):
            positions[cluster_ID][layer].append(subID)
    # we have 
    # positions = { 
    #                       cluster_01: 
    #                                   {
    #                                       layer_01:   [segment_ID_01], 
    #                                       layer_02:   [segment_ID_02],
    #                                       ...
    #                                    }, 
    #                       cluster_02:{},
    #                       ...
    #                       }
    #positions0 = copy.deepcopy(positions)
    
    for clusterID, layer_dict in positions.items():
        if max_layers <= 1:         continue        # 1 layer means theres nothing to reshuffle. maybe break instead of continue
        num_IDs_layer = lambda layer: len(layer_dict[layer])
        last_ID_layer = lambda layer: layer_dict[layer][-1]
        for layer, seg_subIDs in layer_dict.items():# initially each layer has 1 element (or zero). have to drop them down.
            if layer == 0:              continue    # dont do anything with first layer.
            if len(seg_subIDs) == 0:    break       # first empty, then rest are also empty.
            ID = seg_subIDs[0]
            interval = time_intervals[ID]        
            # start walking layers below. take last ID on that layer and check overlap. take first case w/o overlap
            first_match = next(
                (l for l in range(2,layer) if num_IDs_layer(l)                                          == 0 or 
                                            not ranges_overlap(interval, time_intervals[last_ID_layer(l)])    
                 )
                , None)

            if first_match is not None:
                positions[clusterID][first_match].append(ID)# add to end of layer with free spot
                positions[clusterID][layer].remove(ID)
                
    
    # move verticaly cluster layers to center of the graph:
    # have to get new layer counts
    num_layers_cluster = {}
    # position holds values: {cluster_ID: {layer_1:list_sub_IDs, layer_2:...},...}
    for clusterID, layer_dict in positions.items():
        num_IDs_layer = lambda layer: len(layer_dict[layer])
        # check how many layers are left after collapse. count up to first empty.
        max_layers_cluster = next((l for l in range(max_layers) if num_IDs_layer(l) == 0 ), max_layers)
        num_layers_cluster[clusterID] = max_layers_cluster
        
    max_layers_2 = max(num_layers_cluster.values())
    # move clusters vertically individually
    
    for clusterID, layer_dict in positions.items():
        offset = np.floor(0.5*(max_layers_2 - num_layers_cluster[clusterID]))
        for layer, seg_subIDs in layer_dict.items():
            for ID in seg_subIDs:
                t_pos[ID] = int(layer + offset) 
    
    #isolate stray nodes
    stray_nodes = [n for n in G.nodes() if G_owner(n) in (-1, None) ]
    # get times at which stray nodes are present
    stray_times = sorted(set([G_time(n) for n in stray_nodes]))
    # sort strat nodes to time related bins
    stray_nodes_time_dict   = {t:[] for t in stray_times}
    stat_nodes_time_dict    = {t:[] for t in stray_times}
    #nodes_count_per_time    = {t:0 for t in stray_times}
    for n in G.nodes():
        t = G_time(n)
        if t in stray_times:
            if G_owner(n) in (-1, None):
                stray_nodes_time_dict[t].append(n)
            else:
                stat_nodes_time_dict[t].append(n)

            #nodes_count_per_time[t] += 1

    # segment nodes will be fixed in place, remember their layer
    node_layer_pos = {}
    for time, nodes in stat_nodes_time_dict.items():
        for node in nodes:
            ID = G_owner(node)
            node_layer_pos[node] = t_pos[ID]


    # stray nodes should be initialized horizontally near the middle of a graph

    for time in stray_times:
        # stationary node reserve layer positions
        pos_node_d  = {node_layer_pos[n]:n  for n in stat_nodes_time_dict[time]}
        # define central point near the middle of the graph
        init_pos    = int(0.5*max_layers_2)
        # prepare stack of nodes that has to be redistributed on vertical layers
        node_stack  = stray_nodes_time_dict[time].copy()
        # redistribution happens in oscillating fashion around central point with increasing offset.
        offset = 0
        end_loop = False
        while True:
            if end_loop: break
            for sign in [1,-1]:

                if init_pos + sign*offset not in pos_node_d:

                    pos_node_d[init_pos + sign*offset] = node_stack[0]
                    node_stack = node_stack[1:]

                if len(node_stack) == 0: 
                    end_loop = True   # break while loop
                    break             # break sign loop

            offset += 1
        # hold to inverse relations node:pos dict
        for pos, node in pos_node_d.items():
            node_layer_pos[node] = pos 
    
    # get segment nodes. these are segment achors - end points and segment nodes that exist along stray nodes at common times.
    seg_nodes = list(set([s[0] for s in paths.values()] + [s[-1] for s in paths.values()] + sum(stat_nodes_time_dict.values(), [])))
    # all fields will hold to values in same order as nodes_all
    nodes_all = stray_nodes + seg_nodes

    positions = np.zeros((len(nodes_all), 2), float)

    for k, node in enumerate(stray_nodes):
        positions[k]                    = [ G_time(node),   node_layer_pos[node]    ]

    for k, node in enumerate(seg_nodes):
        positions[k + len(stray_nodes)] = [ G_time(node),   t_pos[G_owner(node)]    ]

    if optimize_pos:
        # perform stray node position optimization by emulating physical forces between nodes: springs/electrostatics/custom
        positions_t_OG = positions[:,0].copy()
        # time pair edges act as springs, pulling nodes together (only vertically in this case)
        spring_edges = []
        for node in stray_nodes:
            spring_edges.extend(    [(node,n) for n in G.successors(    node)]  )
            spring_edges.extend(    [(node,n) for n in G.predecessors(  node)]  )
        # nodes in vertical (time) slices should repel to counteract spring forces
        edges_vert = []
        for node in stray_nodes:
            time = G_time(node)
            neighbors_vert = [n for n in stray_nodes_time_dict[time] + stat_nodes_time_dict[time] if n != node]
            edges_vert.extend(      [(node,n) for n in neighbors_vert]          )

        #with open('spring_iter_data.pickle', 'wb') as handle:
        #        pickle.dump([stray_nodes,seg_nodes,positions,positions_t_OG, spring_edges, edges_vert], handle)
    
        save_path = 'particle_movement'

        node_enum = {n:i for i,n in enumerate(nodes_all)} # node -> order in nodes_all

        ids_set     = range(len(stray_nodes),len(stray_nodes) + len(seg_nodes))

        positions[:len(stray_nodes),0] += np.random.uniform(-0.15, 0.15, len(stray_nodes))  # perturb horisontally stray nodes

        velocities      = np.zeros_like(positions, float)   # ordered same as nodes_all
        forces          = np.zeros_like(positions, float)
    

        edges_spring    = [(node_enum[a],node_enum[b]) for a,b in spring_edges] # connections between nodes ->
        edges_repel     = [(node_enum[a],node_enum[b]) for a,b in edges_vert]   # -> connections of indicies of nodes_all

        num_particles   = len(positions)
        (m_spr, m_rep) = prep_matrices(edges_spring, edges_repel, num_particles)

        start_time  = time_lib.time()
        print('Optimizing stray node positions...')
        #fig, ax = plt.subplots()
        ax = None
        start       = 0
        num         = 2000 

        k_s_r_t     = [2    ,   50  , 0.1]

        start_num_dt= [start,   num , 0.01]

        positions   =   integration(positions, velocities, forces, *m_spr, *m_rep, positions_t_OG,                                                 
                                *k_s_r_t, *start_num_dt, k_t_ramp = [], k_s_ramp = [], k_r_ramp = [], 
                                doPlot = False, ax = ax, ids_set = ids_set, path = save_path)

        #fig, ax = plt.subplots()
        #x_min,x_max = min(positions[:,0]) - 1, max(positions[:,0]) + 1
        #x_min,x_max = 369,386
        #y_min,y_max = min(positions[:,1]) - 1, max(positions[:,1]) + 1
    
        #draw_plot(ax, positions, ids_set, edges_spring, edges_repel, x_min, x_max, y_min, y_max, -1)
        #plt.show()

        start       += num    # start of new counter
        num         = 2000

        k_t_ramp    = [50   ,200    , int(3/4*num)  ]
        k_s_r_t     = [2    ,50     , 0.1           ]

        start_num_dt= [start, num   , 0.01           ]

        positions   =   integration(positions, velocities, forces, *m_spr, *m_rep, positions_t_OG,                                                 
                                    *k_s_r_t, *start_num_dt, k_t_ramp = k_t_ramp, k_s_ramp = [], k_r_ramp = [], 
                                    doPlot = False, ax = ax, ids_set = ids_set, path = save_path)

        #draw_plot(ax, positions, ids_set, x_min, x_max, y_min, y_max, -1)
        #plt.show()
        start       += num    # start of new counter
        num         = 2000

        k_t_ramp    = [200  , 300   , int(3/4*num)  ]
        k_r_ramp    = [0.1  , 0.2   , int(num/2)    ]
        k_s_r_t     = [1    , 10    , 0.2           ]

        start_num_dt= [start, num   , 0.01           ]

        positions   =   integration(positions, velocities, forces, *m_spr, *m_rep, positions_t_OG, 
                                    *k_s_r_t, *start_num_dt, k_t_ramp = k_t_ramp, k_s_ramp = [], k_r_ramp = k_r_ramp, 
                                    doPlot = False, ax = ax, ids_set = ids_set, path = save_path)
     
    
        print(f'\nOptimizing stray node positions...done. time elsapsed: {(time_lib.time() - start_time):.3f} s')
        positions[:, 0] = np.round(positions[:, 0], decimals=0) # clamp to nearest x integer


    positions_d = {}
    for k, node in enumerate(stray_nodes):
        positions_d[node] = positions[k]

    all_nodes_pos = {}
    for node in G.nodes():
        segID = G_owner(node)
        if segID not in (-1, None):
            all_nodes_pos[node] = [ G_time(node), t_pos[segID]     ]
        else:
            all_nodes_pos[node] = positions_d[node]#[ G_time(node),   ] 

    edge_width, edge_color, node_size = drawH(G.to_undirected(), paths, all_nodes_pos, show = show, suptitle = suptitle, node_size = node_size, edge_width_path = edge_width_path, edge_width = edge_width, font_size = font_size)


    return all_nodes_pos, edge_width, edge_color, node_size



def graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments, t_segments_new, t_segments_keep, ref_node = None):
    # segments active during time period of interest
    t_segments_active_all = set(sum([vals for t,vals in lr_time_active_segments.items() if t_start <= t <= t_end],[])) 
    # which segments (and their nodes) to drop from subgraph
    t_segments_drop = [t for t in t_segments_active_all if t not in t_segments_keep]

    t_nodes_drop    = sum([t_segments_new[tID] for tID in t_segments_drop],[])
    t_nodes_keep    = [node for node in G.nodes() if t_start <= node[0] <= t_end and node not in t_nodes_drop]
            
    t_subgraph = G.subgraph(t_nodes_keep)
    connected_components_unique = extract_graph_connected_components(t_subgraph.to_undirected(),  lambda x: (x[0], *x[1:]))
    if ref_node == None:
        return connected_components_unique
    else:
        sol = [t_cc for t_cc in connected_components_unique if ref_node in t_cc] # merge first node
        assert len(sol) == 1, "inspect path relates to multiple clusters, did not expect it ever to occur"
        return sol[0]


def set_custom_node_parameters(graph, contour_data, nodes_list, owner, calc_hull = 1):
    # set params for graph nodes. if calc_hull = 0, then contour_data is hull dict
    # otherwise contour_data is contour dict
    for node in nodes_list:
            
        if calc_hull:
            t_time, *t_subIDs = node
            t_hull  = cv2.convexHull(np.vstack([contour_data[t_time][subID] for subID in t_subIDs]))
        else:
            t_time, t_ID      = node
            t_hull = contour_data[t_time][t_ID]

        t_centroid, t_area, t_mom_z     = centroid_area_cmomzz(t_hull)

        graph.nodes[node]["time"]       = int(t_time)
        graph.nodes[node]["centroid"]   = t_centroid
        graph.nodes[node]["area"]       = t_area
        graph.nodes[node]["moment_z"]   = t_mom_z
        graph.nodes[node]["owner"]      = owner

    return 

def G2_set_parameters(graph, graph_seg, segment, owner):#, time_func):

    node_start, node_end  = segment[0], segment[-1]

    time_start, time_end  = node_time(node_start, graph), node_time(node_end, graph)
    
    graph_seg.add_node(owner) 
    
    graph_seg.nodes()[owner]['node_start'   ]   =   node_start
    graph_seg.nodes()[owner]['node_end'     ]   =   node_end

    graph_seg.nodes()[owner]["t_start"      ]   =   time_start
    graph_seg.nodes()[owner]["t_end"        ]   =   time_end

    return 

def graph_check_paths(graph_node, graph_seg, time_DT_max, report):

    # method to find connected segments. connection can be established only stray nodes = paths though other segments are not considered
    # connections are checked in forward direction with segments within specific maximum DT time interval (its cheap and fast)
    # depending on graph and segments there might be a lot of connections to check
    # connections are determined by searching paths between segments which are using only stray nodes (on an isolated subgraph)
    # additional method is implemented which drops path checks that are apriori impossible, which can be determined after a short test.
    # test works on principle: 'you need stray nodes to find pathways instead of consructing subgraph and look for paths, see if 
    # there are time steps between segments in which there are no stray nodes. paths cannot be constructed over these holes.
    # see code_ideas/search_for_holes_before_for_path.py and image for idea/implementation

    G_time      = lambda node   : node_time(   node  , graph_node)
    G_owner     = lambda node   : node_owner(  node  , graph_node)
                                                     
    G2_t_start  = lambda node   : seg_t_start( node  , graph_seg )
    G2_t_end    = lambda node   : seg_t_end(   node  , graph_seg )
    G2_n_start  = lambda node   : seg_n_from(  node  , graph_seg )
    G2_n_end    = lambda node   : seg_n_to(    node  , graph_seg )

    available_stray_node_times  = set()
    for node in graph_node.nodes():
        if G_owner(node) in (None, -1): available_stray_node_times.add(G_time(node))

    times_start_all         = np.array([G2_t_start(node) for node in graph_seg.nodes()])
    segment_relevant_IDs    = np.array([node for node in graph_seg.nodes()])
    
    for ID_from in segment_relevant_IDs:   

        successors_old = set(graph_seg.successors(ID_from))  # want to drop old false edges, if present.

        time_from    = G2_t_end(ID_from)    #graph_seg.nodes[ID_from]["t_end"]
        time_diffs   = times_start_all - time_from

        index_pass_DT = np.where((1 <= time_diffs) & (time_diffs <= time_DT_max))[0]
        IDs_DT_pass   = segment_relevant_IDs[index_pass_DT]

        node_from = G2_n_end(ID_from)       #segments_all[ID_from][-1]
        successors_resolved = set()
        for ID_to in IDs_DT_pass:
            node_to = G2_n_start(ID_to)     #segments_all[ID_to][0]
            time_to = G2_t_start(ID_to)   #G_time(node_to)
            edge    = (ID_from,ID_to)
            #  return time of first hole or none = either pass or neighbor segment is right next to it
            first_hole_time = next((t for t in (k for k in range(time_from +1 , time_to )) if t not in available_stray_node_times), None) 

            #if not has_holes_pass_dict[edge]:   # pass  = no holes ; not True = False -> dont do if-else. pass = no holes = time
            if first_hole_time is None:
                nodes_keep    = [node for node in graph_node.nodes() if time_from <= G_time(node) <= time_to and G_owner(node) in (-1, None)] 
                nodes_keep.extend([node_from,node_to])
                subgraph = graph_node.subgraph(nodes_keep)
                hasPath = nx.has_path(subgraph, source = node_from, target = node_to)
                if hasPath:
                    graph_seg.add_edge(ID_from, ID_to, dist = time_to - time_from + 1)
                    successors_resolved.add(ID_to)
            else:
                report[edge] = first_hole_time
        successors_drop = successors_old - successors_resolved  # which old to remove
        graph_seg.remove_edges_from([(ID_from,ID) for ID in successors_drop])

    return graph_seg

def get_event_types_from_segment_graph(graph):
    # method for analyzing which type of event graph edge (between 
    # segments/nodes) is in based on connectivity between neighbor segments.
    # e.g merge event conists of multiple branches (nodes) which have solo 
    # successors et merge segment (node) has multiple predecessors
 
    for edge in graph.edges():
        graph.edges[edge]["in_events"] = set()

    get_in_events   = lambda edge: graph.edges[edge]["in_events"]
    get_to_state    = lambda node: graph.nodes[node]["state_to"]
    get_from_state  = lambda node: graph.nodes[node]["state_from"]

    # determine type of connections (edges) between nodes.
    #  it depends on number of successors and predecessors
    for seg_ID in graph.nodes(): 
        seg_successors = list(graph.successors(seg_ID))
        if len(seg_successors) == 0 :
            graph.nodes[seg_ID]["state_to"] = 'end'
        elif len(seg_successors) == 1: 
            graph.nodes[seg_ID]["state_to"] = 'solo'
            graph.edges[(seg_ID,seg_successors[0])]["in_events"].add('solo')
        else:
            graph.nodes[seg_ID]["state_to"] = 'split'
            [graph.edges[(seg_ID,t)]["in_events"].add('split') for t in seg_successors]

        seg_predecessors = list(graph.predecessors(seg_ID))
        if len(seg_predecessors) == 0 :
            graph.nodes[seg_ID]["state_from"] = 'start'
        elif len(seg_predecessors) == 1: 
            graph.nodes[seg_ID]["state_from"] = 'solo'
            graph.edges[(seg_predecessors[0],seg_ID)]["in_events"].add('solo')
        else:
            graph.nodes[seg_ID]["state_from"] = 'merge'
            [graph.edges[(t,seg_ID)]["in_events"].add('merge') for t in seg_predecessors]

    connections_together = {'solo':[], 'merge':{}, 'split':{}, 'mixed':{}}
    # edge is visited twice as a successor for one node and as a predecessor for another node. 
    # depending which role edge playes in both cases we can deduce what type of an event it is.
    
    # if both roles  result in 'solo' type edge, its a part of segement chain
    connections_together['solo'] = [conn for conn in graph.edges() if get_in_events(conn) == {'solo'}] #graph.edges[conn]["in_events"]

    # for classic merges (or splits) situation is different. from point of view of a branch, it 
    # is coming into (going out of)  single merge (split) node, so this connection is 'solo'. same edge from 
    # point of view of a merge (split) node is 'merge' ('split'), due to many predecessors (successors)
    # some edges may act as both split and merge branches, these are 'mixed' type events
    type_split = {'split','solo'}
    type_merge = {'merge','solo'}
    type_mixed = {'split','merge'} 
    edges_mixed = set()
    for seg_ID in graph.nodes():
        # solo state is ruled out in 'connections_solo'
        if get_to_state(seg_ID) == 'split':      #graph.nodes[seg_ID]["state_to"] 
            # check forward connections and check their types
            edge_types  = {(seg_ID, t): get_in_events((seg_ID,t)) for t in graph.successors(seg_ID)} #graph.edges[(seg_ID,t)]["in_events"]
            type_pass   = [edge_type == type_split for edge_type in edge_types.values()]
            if all(type_pass):  
                # all branches are classic split
                connections_together['split'][seg_ID] = [t for _, t in edge_types.keys()]
            else:
                # check if branches are of mixed type.
                [edges_mixed.add(edge) for edge, edge_type in edge_types.items() if type_mixed.issubset(edge_type)]

        if get_from_state(seg_ID) == 'merge': #graph.nodes[seg_ID]["state_from"]

            edge_types  = {(t, seg_ID): get_in_events((t, seg_ID)) for t in graph.predecessors(seg_ID)} #graph.edges[(t, seg_ID)]["in_events"]

            type_pass   = [edge_type == type_merge for edge_type in edge_types.values()]
            if all(type_pass):
                connections_together['merge'][seg_ID] = [t for t, _ in edge_types.keys()]
            else:
                [edges_mixed.add(edge) for edge, edge_type in edge_types.items() if type_mixed.issubset(edge_type)]
     
    # try to isolate event containing mixed branch. to avoid connectendess to far neighbor segments, which is possible 
    # from edge of event, try to find event nodes from gathering predecessors of successors of 'from' mixed edge node
    # and similarly, but inverse, for 'to' of mixed event. event may hold multiple mixed nodes, but hopefully clusters are the same
    # IT IS A WEAK assumption. But it forces path to cross this event, and connections are made in a way
    # to not use other segments, only stray nodes, which are, most likely, a part of this local event.
    ID_clusters = set()
    for seg_from,seg_to in edges_mixed:
        # start a cluster associated with this mixed edge nodes
        t_node_cluster = {seg_from, seg_to}
        # add connecteness sweep results to this cluster
        for t in graph.successors(seg_from):
            t_node_cluster.update(list(graph.predecessors(t)))
        for t in graph.predecessors(seg_to):
            t_node_cluster.update(list(graph.successors(t)))
        # add this cluster to set of other clusters, frozenset is required to set of sets, its hashable.
        ID_clusters.add(frozenset(t_node_cluster))
    
    # mixed events also have different sub-cases.  but all are later partially solved by extrapolating 'in' branches.
    # clean mixed event happens when multiple branches interact and come out multiple
    # during this event out branches are only connected with incoming branches
    #
    # dirty mixed event is when there is an internal event, but ~one branch goes around it.
    # otherwise you would be able to isolate and process this internal event separately.
    #
    # clean event = no internal events, incoming branches go to their successors (outgoing branches)
    #               predecessors of outgoing branches are incoming branches.
    # dirty event = incoming branches go to outgoing, but also to internal event branches.
    #               inc branch successors --> their predecessors =  not only incoming, but also internal branches.
    # events are isolated on the graph, so only incoming and internal branches have successors,
    # and similarly, only outgoing and internal branches have predecessors
    # internal branches are, simply, an intersection of those two.
    for k, t_cluster in enumerate(ID_clusters):

        subgraph = graph.subgraph(t_cluster)

        successors_all = set()
        predecessors_all = set()

        successors    = {t:list(subgraph.successors(t))     for t in t_cluster}
        predecessors  = {t:list(subgraph.predecessors(t))   for t in t_cluster}

        branches_out  = tuple(sorted([t for t in successors     if len(successors[t]  ) == 0]))
        branches_in   = tuple(sorted([t for t in predecessors   if len(predecessors[t]) == 0]))

        [successors_all.update(t)       for t in successors.values()  ]
        [predecessors_all.update(t)     for t in predecessors.values()]

        intersection = successors_all.intersection(predecessors_all)
        if len(intersection) > 0:
            branches_in = tuple(sorted(list(branches_in) + list(intersection)))  # it will be extrapolated
        connections_together['mixed'][branches_in] = branches_out

    # rename in_events into simpler type instead of which events branches are in.
    # EDIT: not that simple. mixed connections have to be determined by itertools.product(branch_in,branch_out)
    #for *edge, in_events in graph.edges(data='in_events'):
    #    if in_events    == {'solo'}:
    #        graph.edges[edge]["in_events"] = 'solo'
    #    elif in_events  == type_split:
    #        graph.edges[edge]["in_events"] = 'split'
    #    elif in_events  == type_merge:
    #        graph.edges[edge]["in_events"] = 'merge'
    #    else:
    #        graph.edges[edge]["in_events"] = 'mixed'

    return graph, connections_together

if 1 == -1:
    
    def graph_extract_paths_backup_unidir(H,f):
        nodeCopy = list(H.nodes()).copy()
        segments2 = {a:[] for a in nodeCopy}
        resolved = []
        skipped = []
        for node in nodeCopy:
            goForward = True if node not in resolved else False
            nextNode = node
            prevNode = None
            while goForward == True:
                neighbors = list(H.neighbors(nextNode))
                nextNodes = [a for a in neighbors if f(a) > f(nextNode)]
                prevNodes = [a for a in neighbors if f(a) < f(nextNode)]
                # find if next node exists and its single
                soloNext    = True if len(nextNodes) == 1 else False
                soloPrev    = True if len(prevNodes) == 1 else False # or prevNode is None
                soloPrev2   = True if soloPrev and (prevNode is None or prevNodes[0] == prevNode) else False

                # if looking one step ahead, starting node can have one back and/or forward connection to split/merge
                # this would still mean that its a chain and next/prev node will be included.
                # to fix this, check if next/prev are merges/splits
                # find if next is not merge:
                nextNotMerge = False
                if soloNext:
                    nextNeighbors = list(H.neighbors(nextNodes[0]))
                    nextPrevNodes = [a for a in nextNeighbors if f(a) < f(nextNodes[0])]
                    if len(nextPrevNodes) == 1: 
                        nextNotMerge = True

                nextNotSplit = False
                if soloNext:
                    nextNeighbors = list(H.neighbors(nextNodes[0]))
                    nextNextNodes = [a for a in nextNeighbors if f(a) > f(nextNodes[0])]
                    if len(nextNextNodes) <= 1:   # if it ends, it does not split. (len = 0)
                        nextNotSplit = True

                prevNotSplit = False
                if soloPrev2:
                    prevNeighbors = list(H.neighbors(prevNodes[0]))
                    prevNextNodes = [a for a in prevNeighbors if f(a) > f(prevNodes[0])]
                    if len(prevNextNodes) == 1:
                        prevNotSplit = True


                saveNode = False
                # test if it is a chain start point:
                # if prev is a split, implies only one prevNode 
                if prevNode is None:                # starting node
                    if len(prevNodes) == 0:         # if no previos node =  possible chain start
                        if nextNotMerge:            # if it does not change into merge, it is good
                            saveNode = True
                        else:
                            skipped.append(node)
                            goForward = False
                    elif not prevNotSplit:
                        if nextNotMerge:
                            saveNode = True
                        else: 
                            skipped.append(node)
                            goForward = False
                    else:
                        skipped.append(node)
                        goForward = False
                else:
                # check if its an endpoint
                    # dead end = zero forward neigbors
                    if len(nextNodes) == 0:
                        saveNode = True
                        goForward = False
                    # end of chain =  merge of forward neigbor
                    elif not nextNotMerge:
                        saveNode = True
                        goForward = False
                    elif not nextNotSplit:
                        saveNode = True
                
                    # check if it is part of a chain
                    elif nextNotMerge:
                        saveNode = True


                if saveNode:
                    segments2[node].append(nextNode)
                    resolved.append(nextNode)
                    prevNode = nextNode
                    if goForward :
                        nextNode = nextNodes[0]

    
    
        return segments2, skipped