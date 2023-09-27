
import networkx as nx, cv2, numpy as np
from matplotlib import pyplot as plt
from bubble_params import (centroid_area_cmomzz)
from misc import (cyclicColor, order_segment_levels, prep_combs_clusters_from_nodes)

def extractNeighborsNext(graph, node, time_from_node_function):
    neighbors = list(graph.neighbors(node))
    return [n for n in neighbors if time_from_node_function(n) > time_from_node_function(node)]

def extractNeighborsPrevious(graph, node, time_from_node_function):
    neighbors = list(graph.neighbors(node))
    return [n for n in neighbors if time_from_node_function(n) < time_from_node_function(node)]

def graph_extract_paths(H,f):
    nodeCopy = list(H.nodes()).copy()
    segments2 = {a:[] for a in nodeCopy}
    resolved = []
    skipped = []
    for node in nodeCopy:
        goForward = True if node not in resolved else False
        nextNode = node
        prevNode = None
        while goForward == True:
            #neighbors = list(H.neighbors(nextNode))
            #nextNodes = [a for a in neighbors if f(a) > f(nextNode)]
                
            #prevNodes = [a for a in neighbors if f(a) < f(nextNode)]
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
                #nextNeighbors = list(H.neighbors(nextNodes[0]))
                #nextPrevNodes = [a for a in nextNeighbors if f(a) < f(nextNodes[0])]
                nextPrevNodes = list(H.predecessors(nextNodes[0]))
                if len(nextPrevNodes) == 1: 
                    nextNotMerge = True

            nextNotSplit = False
            if soloNext:
                #nextNeighbors = list(H.neighbors(nextNodes[0]))
                #nextNextNodes = [a for a in nextNeighbors if f(a) > f(nextNodes[0])]
                nextNextNodes = list(H.successors(nextNodes[0]))
                if len(nextNextNodes) <= 1:   # if it ends, it does not split. (len = 0)
                    nextNotSplit = True

            prevNotSplit = False
            if soloPrev2:
                #prevNeighbors = list(H.neighbors(prevNodes[0]))
                #prevNextNodes = [a for a in prevNeighbors if f(a) > f(prevNodes[0])]
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
def find_segment_connectivity_isolated(graph, t_min, t_max, add_nodes, active_segments, source_segment, source_node = None, target_nodes = None):
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

    return connected_edges


def drawH(H, paths, node_positions, fixed_nodes = []):
    colors = {i:np.array(cyclicColor(i))/255 for i in paths}
    colors = {i:np.array([R,G,B]) for i,[B,G,R] in colors.items()}
    colors_edges2 = {}
    width2 = {}
    # set colors to different chains. iteratively sets default color until finds match. slow but whatever
    for u, v in H.edges():
        for i, path in paths.items():
            if u in path and v in path:

                colors_edges2[(u,v)] = colors[i]
                width2[(u,v)] = 3
                break
            else:

                colors_edges2[(u,v)] = np.array((0.5,0.5,0.5))
                width2[(u,v)] = 1

    nx.set_node_attributes(H, node_positions, 'pos')

    pos         = nx.get_node_attributes(H, 'pos')
    weight = {t_edge:0 for t_edge in H.edges()}
    #pos = custom_spring_layout(H, pos=node_positions, fixed=fixed_nodes)
    label_offset= 0.05
    lable_pos   = {k: (v[0], v[1] + (-1)**(k[0]%2) * label_offset) for k, v in pos.items()}
    labels      = {node: f"{node}" for node in H.nodes()}

    fig, ax = plt.subplots(figsize=( 10,5))

    nx.draw_networkx_edges( H, pos, alpha=0.7, width = list(width2.values()), edge_color = list(colors_edges2.values()))
    nx.draw_networkx_nodes( H, pos, node_size = 30, node_color='lightblue', alpha=1)
    #label_options = {"ec": "k", "fc": "white", "alpha": 0.3}
    nx.draw_networkx_labels(H, pos = lable_pos, labels=labels, font_size=7)#, bbox=label_options
    
    
    #nx.draw(H, pos, with_labels=False, node_size=50, node_color='lightblue',font_size=6,
    #        font_color='black', edge_color=list(colors_edges2.values()), width = list(width2.values()))
    plt.show()

def for_graph_plots(G, segs = []):
    if len(segs) == 0:
        segments3, skipped = graph_extract_paths(G,lambda x : x[0])
    else:
        segments3 = {t:[] for t in G.nodes()}
        for t_seg in segs:
            if len(t_seg) > 0:
                segments3[t_seg[0]] = t_seg

    # Draw extracted segments with bold lines and different color.
    segments3 = [a for _,a in segments3.items() if len(a) > 0]
    segments3 = list(sorted(segments3, key=lambda x: x[0][0]))
    paths = {i:vals for i,vals in enumerate(segments3)}

 
    t_pos = order_segment_levels(segments3, debug = 0)

    all_segments_nodes = sorted(sum(segments3,[]), key = lambda x: [x[0],x[1]])
    all_nodes_pos = {tID:[tID[0],0] for tID in G.nodes()} # initialize all with true horizontal position.
    #fixed_nodes_pos = {tID:[tID[0],0] for tID in G.nodes()}

    for t_k, t_segment in enumerate(segments3): #modify only path nodes position, rest is default
        for t_node in t_segment:
            all_nodes_pos[t_node][1] = 0.28*t_pos[t_k]
            #all_nodes_pos[t_node] = (t_node[0],0.28*t_pos[t_k])

    not_segment_nodes = [t_node for t_node in G.nodes() if t_node not in all_segments_nodes]
    # not_segment_nodes contain non-segment nodes, thus they are solo ID nodes: (time,ID)
    not_segment_nodes_clusters = prep_combs_clusters_from_nodes(not_segment_nodes)
    for t_time, t_subIDs in not_segment_nodes_clusters.items():
        for t_k, t_subID in enumerate(t_subIDs):
            t_node = tuple([t_time,t_subID])
            all_nodes_pos[t_node][1] = 0.28*(t_k - 0.5)
    G = G.to_undirected()
    drawH(G, paths, all_nodes_pos)
    #drawH(G, paths, all_nodes_pos, fixed_nodes = all_segments_nodes)
    #drawH(G, paths, node_positions = all_nodes_pos, fixed_nodes = all_segments_nodes)

    return None



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