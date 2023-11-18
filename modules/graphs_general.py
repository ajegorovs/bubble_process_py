
from re import A
import networkx as nx, cv2, numpy as np, itertools, copy
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

    nodes_keep_main =   [node for node in graph.nodes() if t_min <= graph.nodes[node]["time"] <= t_max]    # isolate main graph segment
    nodes_keep_main +=  add_nodes                                                                          # add extra nodes, if needed
    subgraph_main   =   graph.subgraph(nodes_keep_main) 

    if source_node is None:  # automatically find source node if not specified
        for t_node in nodes_keep_main:
            if subgraph_main.nodes[t_node]["owner"] == source_segment:
                source_node = t_node
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
                if subgraph_test.nodes[node]["owner"] == test_segment:
                    test_target_node = node
                    break
        else: test_target_node = target_nodes
            
        hasPath = nx.has_path(subgraph_test.to_undirected(), source = source_node, target = test_target_node)

        if hasPath:
            connected_edges.append((source_segment,test_segment))
            if return_cc:
                nodes_from  = [node for node in nodes_keep_sub if subgraph_test.nodes[node]["owner"] == source_segment]
                nodes_to    = [node for node in nodes_keep_sub if subgraph_test.nodes[node]["owner"] == test_segment]

                node_from_last  = max(nodes_from, key = lambda x: x[0])
                node_to_firt    = min(nodes_to  , key = lambda x: x[0])

                sub_test_nodes = [node for node in nodes_keep_sub if node_from_last[0] < node[0] < node_to_firt[0]]
                sub_test_nodes += [node_from_last, node_to_firt]

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

def for_graph_plots(G, segs = [], show = True, suptitle = "drawH", node_size = 30, edge_width_path = 3, edge_width = 1, font_size = 7):

    if len(segs) == 0:
        segments3 = graph_extract_paths(G) # default min_length
    else:
        segments3 = [s for s in segs if len(s) > 0 ]
        #segments3 = {t:[] for t in G.nodes()}
        #for t_seg in segs:
        #    if len(t_seg) > 0:
        #        segments3[t_seg[0]] = t_seg

    # Draw extracted segments with bold lines and different color.
    #segments3 = [a for _,a in segments3.items() if len(a) > 0]
    #segments3 = list(sorted(segments3, key=lambda x: x[0][0]))
    paths = {i:vals for i,vals in enumerate(segments3)}

    # check overlap between segments time-wise: extract overlapping clusters
    # 
    G_time  = lambda node, graph = G : graph.nodes[node]['time']
    G_owner = lambda node, graph = G : graph.nodes[node]['owner']
    # extract (time_from,to_to) of each segment
    #time_intervals = {i:(G_time(nodes[0]), G_time(nodes[-1])) for i, nodes in paths.items()}
    # check all pairs of segments for overlap. keep overlapping
    #pairs_overlap = [(a,b) for (a,b) in itertools.combinations(paths.keys(), 2) 
    #                 if check_overlap(time_intervals[a],time_intervals[b]) >= 0]

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

    for clusterID, layer_dict in positions.items():
        num_IDs_layer = lambda layer: len(layer_dict[layer])
        # check how many layers are left after collapse. count up to first empty.
        max_layers_cluster = next((l for l in range(max_layers) if num_IDs_layer(l) == 0 ), max_layers)
        num_layers_cluster[clusterID] = max_layers_cluster
        
    max_layers_2 = max(num_layers_cluster.values())
    # move clusters vertically individually
    t_pos = {i:0 for i,_ in enumerate(segments3)}
    for clusterID, layer_dict in positions.items():
        offset = np.floor(0.5*(max_layers_2 - num_layers_cluster[clusterID]))
        for layer, seg_subIDs in layer_dict.items():
            for ID in seg_subIDs:
                t_pos[ID] = int(layer + offset) 
    #t_pos = order_segment_levels(segments3, debug = 0)

    #isolate stray nodes
    stray_nodes = [n for n in G.nodes() if G_owner(n) is None]
    # get times at which stray nodes are present
    stray_times = sorted(set([G_time(n) for n in stray_nodes]))
    # sort strat nodes to time related bins
    stray_times_nodes       = {t:[] for t in stray_times}
    stray_times_seg_nodes   = {t:[] for t in stray_times}
    for n in G.nodes():
        if G_time(n) in stray_times:
            if G_owner(n) is None:
                stray_times_nodes[G_time(n)].append(n)
            else:
                stray_times_seg_nodes[G_time(n)].append(n)

    # segment nodes will be fixed in place, remember their layer
    node_layer_pos = {}
    for time, nodes in stray_times_seg_nodes.items():
        for node in nodes:
            ID = G_owner(node)
            node_layer_pos[node] = t_pos[ID]

    # find maximal number of nodes for all time slices.
    max_stray_layers = 0
    num_layers_strays = {}
    for time in stray_times:
        num_tot                 = len(stray_times_nodes[time]) + len(stray_times_seg_nodes[time])
        num_layers_strays[time] = num_tot
        max_stray_layers        = max(max_stray_layers, num_tot)
    
    for time, num_layers in num_layers_strays.items():
        stat_node_pos = [node_layer_pos[n] for n in stray_times_seg_nodes[time]]
        unresolved_strays = stray_times_nodes[time]
        #num_free_nodes = num_layers - len(stat_node_pos)

        for i in range(0,100000):
            if i not in stat_node_pos:
                if len(unresolved_strays) == 0: break
                node_layer_pos[unresolved_strays[0]] = i
                del unresolved_strays[0]

    # nodes at each time step slice should repel each other. get edges between node pairs to calc forces.
    stray_repel_pairs = {}
    for time in stray_times:
        nodes_tot = stray_times_nodes[time] + stray_times_seg_nodes[time]
        stray_repel_pairs[time] = list(itertools.combinations(nodes_tot, 2))
    
    # initialize stray node positions

    all_nodes_pos = {}# {tID:[tID[0],0] for tID in G.nodes()}
    for node in G.nodes():
        segID = G_owner(node)
        if segID is not None:
            all_nodes_pos[node] = [ G_time(node), t_pos[segID]          ]
        else:
            all_nodes_pos[node] = [ G_time(node), node_layer_pos[node]  ] 

    a = 1
    #all_segments_nodes = sorted(sum(segments3,[]), key = lambda x: [x[0],x[1]])
    #all_nodes_pos = {tID:[tID[0],0] for tID in G.nodes()} # initialize all with true horizontal position.
    #fixed_nodes_pos = {tID:[tID[0],0] for tID in G.nodes()}

    #G = G.to_undirected()
    edge_width, edge_color, node_size = drawH(G.to_undirected(), paths, all_nodes_pos, show = show, suptitle = suptitle, node_size = node_size, edge_width_path = edge_width_path, edge_width = edge_width, font_size = font_size)
    #drawH(G, paths, all_nodes_pos, fixed_nodes = all_segments_nodes)
    #drawH(G, paths, node_positions = all_nodes_pos, fixed_nodes = all_segments_nodes)

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


def graph_check_paths(graph_node, graph_seg, graph_seg_new, segment_relevant_IDs, times_start_all, time_DT_max, segments_all, report):

    # method to find connected segments. connection can be established only stray nodes = paths though other segments are not considered
    # connections are checked in forward direction with segments within specific maximum DT time interval (its cheap and fast)
    # depending on graph and segments there might be a lot of connections to check
    # connections are determined by searching paths between segments which are using only stray nodes (on an isolated subgraph)
    # additional method is implemented which drops path checks that are apriori impossible, which can be determined after a short test.
    # test works on principle: 'you need stray nodes to find pathways instead of consructing subgraph and look for paths, see if 
    # there are time steps between segments in which there are no stray nodes. paths cannot be constructed over these holes.
    G_time      = lambda node, graph = graph_node : graph.nodes[node]['time']
    G_owner     = lambda node, graph = graph_node : graph.nodes[node]['owner']

    available_stray_node_times  = set()
    for node in graph_node.nodes():
        if G_owner(node) == None: available_stray_node_times.add(G_time(node))

    for ID_from in segment_relevant_IDs:
        
        if graph_seg != graph_seg_new:
            graph_seg_new.add_node(ID_from, **dict(graph_seg.nodes[ID_from])) #copy old parameters

        #graph_seg_new.add_node(ID_from)
        #graph_seg_new.nodes()[ID_from]["t_start"   ] = G_time(segments_all[ID_from][0   ])
        #graph_seg_new.nodes()[ID_from]["t_end"     ] = G_time(segments_all[ID_from][-1  ])
        #graph_seg_new.nodes()[ID_from]["node_start"] = segments_all[ID_from][0   ]
        #graph_seg_new.nodes()[ID_from]["node_end"  ] = segments_all[ID_from][-1  ]

        time_from    = graph_seg.nodes[ID_from]["t_end"]
        time_diffs   = times_start_all - time_from

        index_pass_DT = np.where((1 <= time_diffs) & (time_diffs <= time_DT_max))[0]
        IDs_DT_pass   = segment_relevant_IDs[index_pass_DT]

        node_from = segments_all[ID_from][-1]
        time_from = G_time(node_from)

        for ID_to in IDs_DT_pass:
            node_to = segments_all[ID_to][0]
            time_to = G_time(node_to)
            edge    = (ID_from,ID_to)
            #  return time of first hole or none = either pass or neighbor segment is right next to it
            first_hole_time = next((t for t in (k for k in range(time_from +1 , time_to )) if t not in available_stray_node_times), None) 
            #has_holes =  first_hole_time if first_hole_time is not None else False
            #has_holes_pass_dict = {edge:first_hole_time} if first_hole_time is not None else {edge:False} 
            
            #if not has_holes_pass_dict[edge]:   # pass  = no holes ; not True = False -> dont do if-else. pass = no holes = time
            if first_hole_time is None:
                nodes_keep    = [node for node in graph_node.nodes() if time_from <= G_time(node) <= time_to and G_owner(node) is None] 
                nodes_keep.extend([node_from,node_to])
                subgraph = graph_node.subgraph(nodes_keep)
                hasPath = nx.has_path(subgraph, source = node_from, target = node_to)
                if hasPath:
                    graph_seg_new.add_edge(ID_from, ID_to, dist = time_to - time_from + 1)
            else:
                report[edge] = first_hole_time

    return graph_seg_new

def get_event_types_from_segment_graph(graph_input):

    # create a directed graph copy of an input graph.
    graph = nx.DiGraph()

    for seg_ID in graph_input.nodes():                                  # copy node parametes
        graph.add_node(seg_ID, **dict(graph_input.nodes[seg_ID]))
    for seg_from, seg_to in graph_input.edges():                        # copy edges parametes
        graph.add_edge(seg_from, seg_to, **dict(graph_input.edges[(seg_from, seg_to)]))
        graph.edges[(seg_from,seg_to)]["in_events"] = set()

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
    # edge is visited twice as a successor and as a predecessor. type of edge will depend on context

    # if both tests result in 'solo' type edge, its a part of segement chain
    connections_together['solo'] = [t_conn for t_conn in graph.edges() if graph.edges[t_conn]["in_events"] == {'solo'}]

    # for classic merges and splits situation is different. from point of view a branch, it 
    # is coming/going in only into merge/split node, so this connection is 'solo'. same edge from 
    # point of view merge/split node is 'merge'/'split' respectivelly, due to many predecessors/successors
    
    type_split = {'split','solo'}
    type_merge = {'merge','solo'}
    # some edges may act as both split and merge branches, these are mixed type events
    type_mixed = {'split','merge'}
    edges_mixed = set()
    for seg_ID in graph.nodes():
        # solo state are ruled out in 'connections_solo'

        if graph.nodes[seg_ID]["state_to"] == 'split':      
            # check forward connections and check their types
            edge_types  = {(seg_ID,t): graph.edges[(seg_ID,t)]["in_events"] for t in graph.successors(seg_ID)}
            type_pass   = [edge_type == type_split for edge_type in edge_types.values()]
            if all(type_pass):  
                # all branches are classic split
                connections_together['split'][seg_ID] = [t for _, t in edge_types.keys()]
            else:
                # check if branches are of mixed type.
                [edges_mixed.add(edge) for edge,edge_type in edge_types.items() if type_mixed.issubset(edge_type)]

        if graph.nodes[seg_ID]["state_from"] == 'merge':
            edge_types  = {(t, seg_ID): graph.edges[(t, seg_ID)]["in_events"] for t in graph.predecessors(seg_ID)}
            type_pass   = [edge_type == type_merge for edge_type in edge_types.values()]
            if all(type_pass):
                connections_together['merge'][seg_ID] = [t for t,_ in edge_types.keys()]
            else:
                [edges_mixed.add(edge) for edge,edge_type in edge_types.items() if type_mixed.issubset(edge_type)]
     
    # try to isolate event containing mixed branch. to avoid connectendess to far neighbor segments, which is possible 
    # from edge of event, try to find event nodes from gathering predecessors of successors of 'from' mixed edge node
    # and similarly, but inverse, for 'to' of mixed event. event may hold multiple mixed nodes, but hopefully clusters are the same
    # IT IS A WEAK assumption. But it forces path to cross this event, and connections are made in a way
    # to not use other segments, only stray nodes, which are, most likely, a part of this local event.
    ID_clusters = set()
    for seg_from,seg_to in edges_mixed:
        # start a cluster associated with this mixed edge nodes
        t_node_cluster = {seg_from,seg_to}
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
        successors    = {t:list(subgraph.successors(t)) for t in t_cluster}
        predecessors  = {t:list(subgraph.predecessors(t)) for t in t_cluster}
        branches_out  = tuple(sorted([t for t in successors      if len(successors[t]  ) == 0]))
        branches_in   = tuple(sorted([t for t in predecessors    if len(predecessors[t]) == 0]))
        [successors_all.update(t)     for t in successors.values()  ]
        [predecessors_all.update(t)   for t in predecessors.values()]
        intersection = successors_all.intersection(predecessors_all)
        if len(intersection) > 0:
            branches_in = tuple(sorted(list(branches_in) + list(intersection)))  # it will be extrapolated
        connections_together['mixed'][branches_in] = branches_out

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