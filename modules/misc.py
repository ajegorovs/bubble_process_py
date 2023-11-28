

import numpy as np, time as time_lib, copy, networkx as nx, cv2
import datetime, itertools
from collections import deque
from collections import defaultdict


colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
np.random.seed(2);np.random.shuffle(colorList);np.random.seed()

def cyclicColor(index):
    return colorList[index % len(colorList)].tolist()


def closest_point(point, array):
    diff = array- point
    distance = np.einsum('ij,ij->i', diff, diff)
    return np.argmin(distance), distance


# https://stackoverflow.com/questions/72109163/how-to-find-nearest-points-between-two-contours-in-opencv 
def closes_point_contours(c1,c2, step = 2):
    # initialize some variables
    min_dist = 4294967294
    chosen_point_c2 = None
    chosen_point_c1 = None
    # iterate through each point in contour c1
    for point in c1[::step]:
        t = point[0][0], point[0][1] 
        index, dist = closest_point(t, c2[:,0]) 
        if dist[index] < min_dist :
            min_dist = dist[index]
            chosen_point_c2 = c2[index]
            chosen_point_c1 = t
    d_vec = tuple(np.diff((chosen_point_c1,chosen_point_c2[0]),axis=0)[0])
    d_mag = np.int32(np.sqrt(min_dist))
    return d_vec, d_mag , tuple(chosen_point_c1), tuple(chosen_point_c2[0])
    # # draw the two points and save
    # cv2.circle(img,(chosen_point_c1), 4, (0,255,255), -1)
    # cv2.circle(img,tuple(chosen_point_c2[0]), 4, (0,255,255), -1)


def start_timer():
    global start_time
    start_time = time_lib.time()

def stop_timer():
    elapsed_time = time_lib.time() - start_time
    return elapsed_time


def timeHMS():
    return datetime.datetime.now().strftime("%H-%M-%S")


def modBR(BR,side):
    x,y,w,h  = BR
    return [x - int(max((side-w)/2,0)), y - int(max((side-h)/2,0)), max(side,w), max(side,h)]


def rect2contour(rect):
    x,y,w,h = rect
    return np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],int).reshape(-1,1,2)

def combs_different_lengths(elements_list):
    #resort = lambda list_of_tuple: sorted(list_of_tuple, key=lambda x: (x[0], x[1:]))
    return sum([list(itertools.combinations(elements_list, r)) for r in range(1,len(elements_list)+1)],[])

def unique_sort_list(arg, sort_function = lambda x: x):
    return sorted(list(set(arg)), key = sort_function)


def sort_len_diff_f(edge):
    # i want to sort edges so most massive come first
    # in addition give prio to least len diff. e.g ((1,2,3),(4,)) worse than ((1,2),(3,4))
    # choices2 = [[(1,), (2,), (1, 2)], [(4,), (5,), (4, 5)]]
    # edges = list(itertools.product(*choices2))
    # --> edges = [ ((1,),(4,)),..., ((1, 2),(4, 5))]
    node1_length = len(edge[0])
    node2_length = len(edge[1])
    total_length = node1_length + node2_length
    length_difference = abs(node1_length - node2_length)
    return (total_length, -length_difference)  

def disperse_nodes_to_times(nodes, sort = False):
    t_perms = defaultdict(list)
    for t, *t_subIDs in nodes:
        t_perms[t].extend(t_subIDs)

    if not sort:
        return dict(t_perms)
    else:
        return {t:sorted(t_perms[t]) for t in sorted(t_perms)}

# DUPLICATE FROM MAIN
# sort function for composite node of type (time, ID1, ID2,...): [(1,2),(2,1),(1,1,9)] -> [(1,1,9),(1,2),(2,1)]
sort_comp_n_fn = lambda x: (x[0], x[1:]) 

def disperse_composite_nodes_into_solo_nodes(composite_node_list, sort = False, sort_fn = sort_comp_n_fn):
    output = set()
    for time, *subIDs in composite_node_list:
        output.update([(time,subID) for subID in subIDs])

    if not sort:
        return list(output)
    else:
        return sorted(output, key = sort_fn)


def find_key_by_value(my_dict, value_to_find):
    for key, value in my_dict.items():
        if value == value_to_find:
            return key
    # If the value is not found, you can return a default value or raise an exception.
    # In this example, I'm returning None.
    return None

def itertools_product_length(choices):
    # product takes L = [[a1], [a2, b2], [a3]] and produces [[a1,a2,a3],[a1,b2,a3]]
    # number of sequences is their individual choice product 1*2*1 = 2
    sequences_length = 1
    for sublist in choices:
        sequences_length *= len(sublist)
    return sequences_length

class CircularBuffer:
    def __init__(self, size, initial_data=None):
        self.size   = size
        self.buffer = deque(maxlen=size)
        if initial_data is not None:
            self.extend(initial_data)

    def append(self, element):
        if isinstance(element, list):
            for item in element:
                self.buffer.append(item)
        else:
            self.buffer.append(element)

    def extend(self, list):
        self.buffer.extend(list)

    def get_data(self):
        return np.array(self.buffer)

class CircularBufferReverse:
    def __init__(self, size, initial_data=None):
        self.size = size
        self.buffer = deque(maxlen=size)
        if initial_data is not None:
            self.extend(initial_data)

    def append(self, element):
        if isinstance(element, list):
            for item in element:
                self.buffer.appendleft(item)
        else:
            self.buffer.appendleft(element)

    def extend(self, lst):
        for item in reversed(lst):
            self.buffer.appendleft(item)

    def get_data(self):
        return np.array(self.buffer)

#class AutoCreateDict:
#    # lr_init_perm_precomputed remake, so dict does not need initialization.
#    def __init__(self):
#        self.data = {}

#    def __getitem__(self, key):
#        if key not in self.data:
#            self.data[key] = AutoCreateDict()
#        return self.data[key]

#    def __setitem__(self, key, value):
#        self.data[key] = value
        
#    def __contains__(self, key):
#        return key in self.data

#    def keys(self):
#        return self.data.keys()

#    def values(self):
#        return self.data.values()

#    def items(self):
#        return self.data.items()

#    def __repr__(self):
#        return repr(self.data)

def find_common_intervals(data):
    element_ids = set()
    active_times = defaultdict(set)

    for time, elements in data.items():
        for element in elements:
            element_ids.add(element)
            active_times[element].add(time)

    common_times_dict = defaultdict(list)

    for pair in itertools.combinations(element_ids, 2):
        element_1, element_2 = pair
        common_times = active_times[element_1].intersection(active_times[element_2])
        
        if common_times:
            common_times_dict[pair] = sorted(common_times)

    return common_times_dict


def find_final_master(slave_master_relations, start_with):
    """
    say you have a pool of people [1,2,3,4], some of them are masters, some slaves
    if given relations of type slave:master i.e 'slave_master_relations' = {1: 1, 2: 1, 3: 2, 4: 4}
    which describes that slave is owned by a master. you see that 3 is a slave of 2
    and 2 is a slave of 1. that means that 3 is a slave of 1.
    goal if this code is to get final relation of who is has the final master of 'start_with'.
    """
    current = start_with
    while current in slave_master_relations and slave_master_relations[current] != current:
        current = slave_master_relations[current]
    return current

def find_final_master_all(slave_master_relations):
    """
    extension of 'find_final_master()' which does it for every entry in a relations dictionary.
    """
    relations_new = {}
    for slave in slave_master_relations:
        master = find_final_master(slave_master_relations, slave)
        relations_new[slave] = master

    return relations_new

def zp_process(edges, node_segments, contours_dict, inheritance_dict):
    """
    WHAT: find segment connections that dont have stray nodes in-between
    WHY:  very much likely that its the same trajectory. interruption is very brief. 
    WHY:  connection has a stray node/s and edges, which have confused chain extraction method.
    HOW:  best i can do is to 'absorb' stray nodes into solo contour nodes by forming a composite node.
    """
    from graphs_general import (set_custom_node_parameters, G2_set_parameters, get_connected_components)
    from graphs_general import (G, G2, G2_t_start, G2_t_end, G_time, G_owner)

    #inheritance_dict = {i: i for i,v in enumerate(node_segments) if len(v) > 0}

    # analyze zero path cases:
    for fr, to in edges:
        fr = inheritance_dict[fr]         # in case there is a sequence of ZP events one should account for inheritance of IDs
        time_buffer = 5
        # take a certain buffer zone about ZP event. its width should not exceed sizes of segments on both sides
        # which is closer to event from left  : start of  left segment    or end      - buffer size?
        time_from   = max(G2_t_start(fr), G2_t_end(fr) - time_buffer    )   # test: say, time_buffer = 1e20 -> time_from = G2_t_start(fr)   # (1)
        # which is closer to event from right : end   of  right segment   or start    + buffer size?
        time_to     = min(G2_t_end(to)  , G2_t_start(to) + time_buffer  )
        # NOTE: at least one node should be left in order to reconnect recovered segment back. looks like current approach does it. by (1) and (2)
        
        #nodes_keep              = []
        #nodes_stray             = [node for node in G.nodes() if time_from < G_time(node) < time_to and G_owner(node) is None] 
        #nodes_extra_segments    = [n for n in node_segments[fr] if G_time(n) > time_from] + [n for n in node_segments[to] if G_time(n) < time_to]   # (2)
        
        #nodes_keep.extend(nodes_stray)
        #nodes_keep.extend(nodes_extra_segments)   
        
        #clusters    = nx.connected_components(G.subgraph(nodes_keep).to_undirected())
        #sol         = next((cluster for cluster in clusters if nodes_extra_segments[0] in cluster), None)
        #assert sol is not None, 'cannot find connected components'

        nodes_extra_segments    = [n for n in node_segments[fr] if G_time(n) > time_from] + [n for n in node_segments[to] if G_time(n) < time_to]
        sol, nodes_stray = get_connected_components(time_from, time_to, nodes_extra_segments, nodes_extra_segments[0], edges_extra = [])

        nodes_composite = [(t,) + tuple(IDs) for t, IDs in disperse_nodes_to_times(sol, sort = True).items()] 

        node_segments[fr]   =       [n for n in node_segments[fr] if G_time(n) <= time_from]                                                           
        node_segments[fr].extend(   nodes_composite)                                                     
        node_segments[fr].extend(   [n for n in node_segments[to] if G_time(n) >= time_to]    )
        node_segments[to]   = []
         
        G.remove_nodes_from(nodes_stray)           # stray nodes are absorbed into composite nodes. they have to go from G
        
        G.remove_nodes_from(nodes_extra_segments)  # old parts of segments have to go. they will be replaced by composite nodes.

        G.add_nodes_from(nodes_composite)
        set_custom_node_parameters( contours_dict, node_segments[fr], fr, calc_hull = 1)

        # have to reconnect reconstructed interval on G. get nodes on interval edges

        ref_left    = next((n for n in node_segments[fr] if G_time(n) == time_from))
        ref_right   = next((n for n in node_segments[fr] if G_time(n) == time_to))

        # generate edges by staggered zip method.

        nodes_temp = [ref_left] + nodes_composite + [ref_right]

        edges = [(a, b) for (a, b) in zip(nodes_temp[:-1], nodes_temp[1:])]

        G.add_edges_from(edges)

        edges_next = [(fr, i) for i in G2.successors(fr)]
       
        G2_set_parameters(node_segments[fr], fr, edges = edges_next)
            
        G2.remove_node(to)
            
        inheritance_dict[to] = fr

        """ 
        some comments on inheritance: 
        if there is chain of zp (or 121) events -> A->B->C, where A,B,C are segments
        if you resolve (A,B) before (B,C), segment B will seize to exist because it will be absorbed into A.
        so you have to resolve (A,C) instea of (B,C) in order to use most recent state of graphs.
        thats why you track if 'from' segment was absorbed/inherited by someone.
        although inheritances holds incorrect for global scope: if (B,C) is resolved first, then inheritance
        states that SLAVE:MASTER = C:B. if in global scope you are interested in connection (C,X),
        C will redirect only one step away, to B. while real target should be A !!!
        so its ok for this specific stage resolution but not to be used for global scope
        """
        return None

def f121_disperse_stray_nodes(edges):
    """
    WHAT: 121 segments are connected via stay (non-segment) nodes. extract them and reformat into  {TIME1:[*SUBIDS_AT_TIME1],...}
    WHY:  at each time step bubble may be any combination of SUBIDS_AT_TIMEX. most likely whole SUBIDS_AT_TIMEX. but possibly not
    """
    from graphs_general import (G,  G2_n_to, G2_n_from, G_time, G_owner)

    output_dict = {}
    for fr, to in edges:
            
        node_from, node_to = G2_n_to(fr)        , G2_n_from(to)
        time_from, time_to = G_time(node_from)  , G_time(node_to)

        # isolate stray nodes on graph at time interval between two connected segments
        
        nodes_keep    = [node for node in G.nodes() if time_from < G_time(node) < time_to and G_owner(node) is None] 
        nodes_keep.extend([node_from,node_to])
            
        clusters    = nx.connected_components(G.subgraph(nodes_keep).to_undirected())
        sol         = next((cluster for cluster in clusters if node_from in cluster), None)
        assert sol is not None, 'cannot find connected components'
        
        output_dict[(fr,to)] = disperse_nodes_to_times(sol, sort = True)

    return output_dict

def f121_interpolate_holes(chains, node_segments):
    """
    WHAT: we have chains of segments that represent solo bubble. its at least 2 segments (1 hole). interpolate data
    WHY:  if there are more than 2 segments, we will have more history and interpolation will be of better quality
    HOW:  scipy interpolate 
    """
    from interpolation import (interpolateMiddle2D_2, interpolateMiddle1D_2)
    from graphs_general import (G2_t_start, G2_t_end, G_time, G_area, G_centroid)

    edges_relevant      = []
    interpolation_data  = {}

    for subIDs in chains:

        # prepare data: resolved  time steps, centroids, areas G2
        temp_nodes_all  = sum(      [node_segments[i]   for i   in subIDs       ], [])
        #temp_nodes_all  = sorted(temp_nodes_all, key = G_time)
        known_times     = np.array( [G_time(n)      for n   in temp_nodes_all   ])
        known_areas     = np.array( [G_area(n)      for n   in temp_nodes_all   ])
        known_centroids = np.array( [G_centroid(n)  for n   in temp_nodes_all   ])
                
        # generate time steps in holes for interpolation
        edges_times_dict = {}
        for (fr, to) in zip(subIDs[:-1], subIDs[1:]):
            edges_times_dict[(fr, to)]  = range(G2_t_end(fr) + 1, G2_t_start(to), 1)

        edges_relevant.extend(edges_times_dict.keys())
        times_missing = []; [times_missing.extend(times) for times in edges_times_dict.values()]
        
        # interpolate composite (long) parameters 
        t_interpolation_centroids_0 = interpolateMiddle2D_2(known_times, known_centroids, times_missing, s = 15         , debug = 0 , aspect = 'equal', title = subIDs)
        t_interpolation_areas_0     = interpolateMiddle1D_2(known_times, known_areas    , times_missing, rescale = True , s = 15    , debug = 0, aspect = 'auto', title = subIDs)
        # form dict = {time:centroid} for convinience
        t_interpolation_centroids_1 = {t: c for t, c in zip(times_missing,t_interpolation_centroids_0)}
        t_interpolation_areas_1     = {t: c for t, c in zip(times_missing,t_interpolation_areas_0)    }
        # save data with t_conns keys
        for edge, times in edges_times_dict.items():
            interpolation_data[edge] = {}
            centroids   = np.array([c    for t, c    in t_interpolation_centroids_1.items()  if t in times])
            areas       = np.array([a    for t, a    in t_interpolation_areas_1.items()      if t in times])

            interpolation_data[edge]['centroids'] = centroids
            interpolation_data[edge]['areas'    ] = areas
            interpolation_data[edge]['times'    ] = times

    return edges_relevant, interpolation_data

def f121_calc_permutations( disperesed_dict):
    """            
    WHAT: generate different permutation of subIDs for each time step.
    WHY:  Bubble may be any combination of contour subIDs at a given time. should consider all combs as solution
    HOW:  itertools combinations of varying lenghts
    """
    permutations_dict           = {}
    
    for edge, time_IDs_dict in disperesed_dict.items():
        permutations_dict[edge] = {}
        for time, IDs in time_IDs_dict.items():
            permutations_dict[edge][time] = combs_different_lengths(IDs)

    return permutations_dict

def f121_precompute_params(times_perms_dict_cases, contours):
    """
    WHY: these will be reused alot in next steps, store beforehand
    """
    from bubble_params  import centroid_area_cmomzz

    areas, centroids, moms_z = {}, {}, {}
            
    for conn, times_perms_dict in times_perms_dict_cases.items():

        areas[     conn]  = {t:{} for t in times_perms_dict}
        centroids[ conn]  = {t:{} for t in times_perms_dict}
        moms_z[    conn]  = {t:{} for t in times_perms_dict}

        for time, permutations in times_perms_dict.items():
            for IDs in permutations:
                hull = cv2.convexHull(np.vstack([contours[time][subID] for subID in IDs]))
                centroid, area, mom_z           = centroid_area_cmomzz(hull)
                areas[      conn][time][IDs]    = area
                centroids[  conn][time][IDs]    = centroid
                moms_z[     conn][time][IDs]    = mom_z

    return areas, centroids, moms_z


def f121_get_evolutions(permutations_dict, permutations_dict_areas, sort_len_diff_f, max_paths):

    """
     WHAT: using subID permutations at missing time construct choice tree. 
     WHY:  each branch represents a bubbles contour ID evolution through unresolved intervals.
     HOW:  either though itertools product or, if number of branches is big, dont consider
     HOW:  branches where area changes more than a set threshold. 
     -----------------------------------------------------------------------------------------------
     NOTE: number of branches (evolutions) from start to finish may be very large, depending on 
     NOTE: number of time steps and contour permutation count at each step. 
     NOTE: it can be calculated as 't_branches_count' = len(choices_t1)*len(choices_t2)*...
     case 0)   if count is small, then use combinatorics product function.
     case 1a)  if its large, dont consider evoltions where bubble changes area rapidly from one step to next.
     case 1b)  limit number of branches retrieved by 't_max_paths'. we can leverage known assumption to
               get better branches in this batch. assumption- bubbles most likely include all 
               subcontours at each time step. Transitions from large cluster to large cluster has a priority.
               if we create a graph with large to large cluster edges first, pathfinding method
               will use them to generate first evolutions. so you just have to build graph in specific order.
     case 2)   optical splits have 121 chains in inter-segment space. these chains are not atomized into
               solo nodes, but can be included into evolutions as a whole. i.e if 2 internal chains are
               present, you can create paths which include 1st, 2nd on both branches simultaneously.
     """
    from graphs_general import (comb_product_to_graph_edges, find_paths_from_to_multi)
    permutations_dict_cases = {}
    permutations_dict_times = {}
    lr_drop_huge_perms = []

    for edge, times_IDs_dict in permutations_dict.items():

        func = lambda edge_test : edge_crit_func(edge, edge_test, permutations_dict_areas, 2)
        
        values            = list(times_IDs_dict.values())

        times             = list(times_IDs_dict.keys())

        branches_count    = itertools_product_length(values) # large number of paths is expected

        if branches_count >= max_paths:
            # 1a) keep only tranitions that dont change area very much
            choices                       = [[(t,) + p for p in perms] for t, perms in zip(times,values)]

            edges, nodes_start, nodes_end   = comb_product_to_graph_edges(choices, func)

            if len(nodes_end) == 0: # finding paths will fail, since no target node

                nodes_end.add(choices[-1][0])   # add last, might also want edges to that node, since they have failed

                edges.extend(list(itertools.product(*choices[-2:])))
                
            # 1b) resort edges for pathfining graph: sort by large to large first subIDs first: ((1,2),(3,4)) -> ((1,2),(3,))
            # 1b) for same size sort by cluster size uniformity e.g ((1,2),(3,4)) -> ((1,2,3),(4,)) 
            sorted_edges    = sorted(edges, key = sort_len_diff_f, reverse=True) 

            sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, graph = None, edges = sorted_edges, only_subIDs = True, max_paths = max_paths - 1)

            # 'fail' can be either because 't_max_paths' is reached or there is no path from source to target
            if fail == 'to_many_paths':    # add trivial solution where evolution is transition between max element per time step number clusters 
                
                seq_0 = list(itertools.product(*[[t[-1]] for t in values] ))

                if seq_0[0] not in sequences: sequences = seq_0 + sequences # but first check. in case of max paths it should be on top anyway.

            elif fail == 'no_path': # increase rel area change threshold

                func = lambda edge_test : edge_crit_func(edge, edge_test, permutations_dict_areas, 5)

                edges, nodes_start, nodes_end   = comb_product_to_graph_edges(choices, func)

                sorted_edges                    = sorted(edges, key=sort_len_diff_f, reverse=True) 

                sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, graph = None, edges = sorted_edges, only_subIDs = True, max_paths = max_paths - 1)

        # 0) -> t_branches_count < t_max_paths. use product
        else:
            sequences = list(itertools.product(*values))
                
        if len(sequences) == 0: # len = 0 because second pass of rel_area_thresh has failed.
            sequences   = []    # since there is no path, dont solve this conn
            times       = []
            lr_drop_huge_perms.append(edge)

        permutations_dict_cases[edge] = sequences
        permutations_dict_times[edge] = times

    return permutations_dict_cases, permutations_dict_times, lr_drop_huge_perms


def unique_active_segments(start, end, segments_active_dict):
    # filter dict by time : {t_min:[*subIDs_min]..., t_max:[*subIDs_max]}; gather [[*subIDs_min],...,[*subIDs_max]]
    # flatten using sum(_,[]) and gather uniques by set()
    return set(sum([vals for t,vals in segments_active_dict.items() if start <= t <= end],[]))

def lr_reindex_masters(relations, connections, remove_solo_ID = 0):
    output = None
    if type(connections) == list:
        output = [tuple(sorted([relations[fr], relations[to]])) for fr,to in connections]
        if remove_solo_ID:
            return [(a,b) for a,b in output if a != b]
        else: 
            return output
    elif type(connections) == tuple:
        return tuple(sorted([relations[connections[0]], relations[connections[1]]]))
    else: 
        raise ValueError("wrong input type. list of tuple or single tuple expected")
# given list of nodes of tupe (time,*subIDs), gather all subIDs for each unique time and sort
def prep_combs_clusters_from_nodes(t_nodes):
    t_times = sorted(set([t_node[0] for t_node in t_nodes]))
    t_return = {t_time:[] for t_time in t_times}
    for t_time,*t_subIDs in t_nodes:
        t_return[t_time] += [min(t_subIDs)] # <<<<<<<< in case of many IDs inside, take only minimal, as representative >><<< its not really used currently!!!
    return {t_time:sorted(t_subIDs) for t_time,t_subIDs in t_return.items()}

   
#rel_area_thresh = 2        
def edge_crit_func(conn,edge,lr_big121s_perms_areas, rel_area_thresh):
    time1,*subIDs1 = edge[0]; time2,*subIDs2 = edge[1]
                
    area1 = lr_big121s_perms_areas[conn][time2][tuple(subIDs2)] 
    area2 = lr_big121s_perms_areas[conn][time1][tuple(subIDs1)]
    rel_change = abs(area2-area1)/max(area2, 0.0001)
    out = True if rel_change < rel_area_thresh else False
    return out

def split_into_bins(L, num_bins):
    n = len(L)
    
    if num_bins == 1:
        yield [L]#([list(t)] for t in combs_different_lengths(L))  # Sort the elements before yielding
        return
    
    for r in range(1, n - num_bins + 2):
        for combination in itertools.combinations(L, r):
            remaining = [x for x in L if x not in combination]
            for rest in split_into_bins(remaining, num_bins - 1):
                yield [sorted(list(combination))] + rest  # Sort the combination sublist before yielding
# have to redefine here because i get circular import error
#def extract_graph_connected_components(graph, sort_function = lambda x: x): 
#    # extract all conneted component= clusters from graph. to extract unique clusters,
#    # all options have to be sorted to drop identical. sorting can be done by sort_function.
#    # for nodes with names integers, used lambda x: x, for names as tuples use lambda x: (x[0], *x[1:])
#    # where node name is (timeX, A, B, C,..), it will sort  Time first, then by min(A,B,C), etc
#    connected_components_all = [list(nx.node_connected_component(graph, key)) for key in graph.nodes()]
#    connected_components_all = [sorted(sub, key = sort_function) for sub in connected_components_all] 
#    connected_components_unique = []
#    [connected_components_unique.append(x) for x in connected_components_all if x not in connected_components_unique]
#    return connected_components_unique



rescale_linear = lambda x,minX,maxX: (x-minX)/(maxX - minX)
def two_crit_many_branches(setA,setB,numObjects):
    # input setA = { crit combination option 1: [branch 1 crit A value 1, branch 2 ...]}. similarly for setB.
    # e.g setA is a displacement crit for multiple objects. we want to find case where all objects have lowest total crit.
    # naturally, you would get a "crit combination option X" where sum([branch 1 crit A value 1, branch 2 ...]) is the least.
    # when other crit is introduced - setB, you would want to do a weighted sum of branch values of A and B.
    # means you have to rescale each "branch 1 crit A value X" to min-max of all options of branch 1s values X.

    # NOTE: if only 2 options and they are cross- mixed, if none option is simultaneously better 
    # than other, take setB min as a sol
    minMaxA = []
    minMaxB = []
                
    # collect all crit values for object with index i. find min and max that object shows.
    for i in range(numObjects): 
        temp1 = []
        temp2 = []
        for t,vals in setA.items():
            temp1.append(vals[i])
        for t,vals in setB.items():
            temp2.append(vals[i])
                    
        minMaxA.append((min(temp1),max(temp1)))
        minMaxB.append((min(temp2),max(temp2)))
                
    # combine setA and setB comb options into single mean & rescaled value.
    weightedSum = {}
    for t in setA:
        weightedSum[t] = []
        for i in range(numObjects):
            weightedSum[t].append(0.5*(rescale_linear(setA[t][i], *minMaxA[i]) + rescale_linear(setB[t][i],*minMaxB[i])))
    weightedSumMean= {t:np.mean(vals) for t, vals in weightedSum.items()}
    minID = min(weightedSumMean, key = weightedSumMean.get)

    if len(setA) == 2: # with 2 options, with cross crit pass, rescaling breaks and weights become 0.5 and 0.5
        if np.diff(list(weightedSumMean.values()))[0] == 0.0:
            maxB = {t:max(vals) for t,vals in setB.items()}
            minID = min(maxB, key = maxB.get)
            weightedSumMean = {minID:-1}
            #A,B = list(setA.keys())
            #mixedA = True if np.any(setA[A] > setA[B]) else False  
            #mixedB = True if np.any(setB[A] > setB[B]) else False
            #if mixedA and mixedB:                                  # logic does not work for 1 object.
            #    maxB = {t:max(vals) for t,vals in setB.items()}
            #    minID = min(maxB, key = maxB.get)
            #    weightedSumMean = {minID:-1}
            #else:
            #    assert 1 == -1, 'havent seen this one yet'
                            
    return minID, weightedSumMean 

def dfs_pred(graph, node, time_lim, node_set):
    node_set.add(node)
    predecessors = list(graph.predecessors(node))
    for predecessor in predecessors:
        if predecessor not in node_set and graph.nodes[predecessor]['time'] > time_lim:
            dfs_pred(graph, predecessor, time_lim, node_set)

def dfs_succ(graph, node, time_lim, node_set):
    node_set.add(node)
    successors = list(graph.successors(node))
    for successor in successors:
        if successor not in node_set and graph.nodes[successor]['time'] < time_lim:
            dfs_succ(graph, successor, time_lim, node_set)

def old_conn_2_new(conn,transform):
    return (transform[conn[0]], transform[conn[1]])

def lr_evel_perm_interp_data(t_conns, lr_permutation_cases,lr_121_interpolation_centroids,lr_permutation_times,
        lr_permutation_centroids_precomputed,lr_permutation_areas_precomputed,lr_permutation_mom_z_precomputed):
        
    t_sols_c        = {t_conn:[] for t_conn in lr_permutation_cases}
    t_sols_c_i      = {t_conn:[] for t_conn in lr_permutation_cases}
    t_sols_a        = {t_conn:[] for t_conn in lr_permutation_cases}
    t_sols_m        = {t_conn:[] for t_conn in lr_permutation_cases}

    for t_conn in t_conns:

        # pre define arrays for storing centroids, area and moments
        num_time_steps  = len(lr_permutation_times[t_conn])
        num_perms       = len(lr_permutation_cases[t_conn])

        t_c_all_traj    = np.zeros((num_perms, num_time_steps, 2))
        t_areas         = np.zeros((num_perms, num_time_steps))
        t_moments       = np.zeros((num_perms, num_time_steps))

        # fill storage with values depending on permutations
        for i, t_perms in enumerate(lr_permutation_cases[t_conn]):
            for j, (t_time, t_perm) in enumerate(zip(lr_permutation_times[t_conn], t_perms)):
                t_c_all_traj[  i][j]    = lr_permutation_centroids_precomputed[   t_conn][t_time][t_perm]
                t_areas[       i][j]    = lr_permutation_areas_precomputed[       t_conn][t_time][t_perm]
                t_moments[     i][j]    = lr_permutation_mom_z_precomputed[       t_conn][t_time][t_perm]

        # get difference between evolution and expected (interpolated) trajectory. 
        # 't_c_all_traj' includes end points, interpolation not, take them out
        # calc dispalcement norm at each time step 
        t_c_interp                  = lr_121_interpolation_centroids[t_conn]
        t_c_inter_traj_diff_norms   = np.linalg.norm(t_c_all_traj[:, 1:-1] - t_c_interp, axis=2)
        # calc mean of displ for each evolution
        t_c_i_traj_d_norms_means    = np.mean(t_c_inter_traj_diff_norms, axis=1)
        t_c_i_mean_min              = np.argmin(t_c_i_traj_d_norms_means)

        # get displacements for all trajectories, calculate displacment norms
        t_c_diffs = np.diff(       t_c_all_traj , axis = 1)
        t_c_norms = np.linalg.norm(t_c_diffs    , axis = 2)
        
        t_c_means       = np.mean(  t_c_norms   , axis = 1)
        t_c_stdevs      = np.std(   t_c_norms   , axis = 1)

        t_c_mean_min    = np.argmin(t_c_means   )
        t_c_stdevs_min  = np.argmin(t_c_stdevs  )


        # same with areas
        t_a_d_abs   = np.abs(np.diff(   t_areas     , axis = 1)) #np.array([np.abs(t) for t in t_a_diffs])
        t_a_means   = np.mean(          t_a_d_abs   , axis = 1)
        t_a_stdevs  = np.std(           t_a_d_abs   , axis = 1)

        t_a_mean_min    = np.argmin(t_a_means)
        t_a_stdevs_min  = np.argmin(t_a_stdevs)

        # same with moments
        t_m_d_abs   = np.abs(np.diff(   t_moments, axis = 1))# np.array([np.abs(t) for t in t_m_diffs])
        t_m_means   = np.mean(          t_m_d_abs, axis = 1)
        t_m_stdevs  = np.std(           t_m_d_abs, axis = 1)

        t_m_mean_min    = np.argmin(t_m_means)
        t_m_stdevs_min  = np.argmin(t_m_stdevs)

        # save cases with least mean and stdev
        t_sols_c[   t_conn].extend( [t_c_mean_min   , t_c_stdevs_min]   )
        t_sols_c_i[ t_conn].extend( [t_c_i_mean_min                 ]   )
        t_sols_a[   t_conn].extend( [t_a_mean_min   , t_a_stdevs_min]   )
        t_sols_m[   t_conn].extend( [t_m_mean_min   , t_m_stdevs_min]   )

    return t_sols_c, t_sols_c_i, t_sols_a, t_sols_m

def lr_weighted_sols(t_conns, weights, t_sols, lr_permutation_cases):
    lr_weighted_solutions = {t_conn:{} for t_conn in lr_permutation_cases}
    lr_weight_c, lr_weight_c_i,lr_weight_a, lr_weight_m = weights
    # set normalized weights for methods
    lr_weights = np.array(weights)
    lr_weight_c, lr_weight_c_i,lr_weight_a, lr_weight_m = lr_weights / np.sum(lr_weights)

    t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = t_sols
    for t_conn in t_conns:  # gather all best solutions
        t_all_sols = []
        t_all_sols += t_sols_c[t_conn]
        t_all_sols += t_sols_c_i[t_conn]
        t_all_sols += t_sols_a[t_conn]
        t_all_sols += t_sols_m[t_conn]

        t_all_sols_unique = sorted(list(set(t_all_sols)))  # take unique
     
        lr_weighted_solutions[t_conn] = {tID:0 for tID in t_all_sols_unique}
        for tID in t_all_sols_unique:                       # go though solutions if unique ID is present there, add bonus reward
            if tID in t_sols_c[   t_conn]:                  # if unique ID is in all sols, it will have a max reward of 1
                lr_weighted_solutions[t_conn][tID] += lr_weight_c
            if tID in t_sols_c_i[   t_conn]:
                lr_weighted_solutions[t_conn][tID] += lr_weight_c_i
            if tID in t_sols_a[   t_conn]:
                lr_weighted_solutions[t_conn][tID] += lr_weight_a
            if tID in t_sols_m[   t_conn]:
                lr_weighted_solutions[t_conn][tID] += lr_weight_m
        # normalize weights for IDs
        r_total = np.sum([t_weight for t_weight in lr_weighted_solutions[t_conn].values()])
        lr_weighted_solutions[t_conn] = {tID:round(t_val/r_total,3) for tID,t_val in lr_weighted_solutions[t_conn].items()}

 
    
    lr_weighted_solutions_max = {t_conn:0 for t_conn in t_conns}
    lr_weighted_solutions_accumulate_problems = {}
    for t_conn in t_conns:                     # determing winning solution by max reward
        t_weight_max = max(lr_weighted_solutions[t_conn].values())
        t_keys_max = [tID for tID, t_weight in lr_weighted_solutions[t_conn].items() if t_weight == t_weight_max]
        if len(t_keys_max) == 1:                            # only one has max
            lr_weighted_solutions_max[t_conn] = t_keys_max[0]
        else:                                               # multiple have even max reward.
            # >>>>>>>>>>VERY CUSTOM: take sol with max elements in total <<<<<<<<<<<<<<
            t_combs = [lr_permutation_cases[t_conn][tID] for tID in t_keys_max]
            t_combs_lens = [np.sum([len(t_perm) for t_perm in t_path]) for t_path in t_combs]
            t_sol = np.argmax(t_combs_lens) # picks first if there are same lengths
            lr_weighted_solutions_max[t_conn] = t_keys_max[t_sol]
            t_count = t_combs_lens.count(max(t_combs_lens))
            if t_count > 1: lr_weighted_solutions_accumulate_problems[t_conn] = t_combs_lens # holds poistion of t_keys_max, != all keys
            a = 1
    return lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems

#def perms_with_branches(t_to_branches,t_segments_new,t_times_contours, return_nodes = False):
#    # if branch segments are present their sequences will be continuous and present fully
#    # means soolution will consist of combination of branches + other free nodes.
            
#    # take combination of branches
#    t_branches_perms = combs_different_lengths(t_to_branches)
#    #t_branches_perms = sum([list(itertools.combinations(t_to_branches, r)) for r in range(1,len(t_to_branches)+1)],[])

#    #t_values_drop = t_values.copy()
#    t_times_contours_drop = copy.deepcopy(t_times_contours)
#    t_contour_combs_perms = {}
#    t_branch_comb_variants = []
#    # drop all branches from choices. i will construct more different pools of contour combinations, where branches are frozen
#    for tID in t_to_branches: 
#        for t, (t_time, *t_subIDs) in enumerate(t_segments_new[tID]):
#            for t_subID in t_subIDs:
#                t_times_contours_drop[t_time].remove(t_subID)
#    # pre compute non-frozen node combinations
#    for t_time,t_contours in t_times_contours_drop.items():
#        t_perms = combs_different_lengths(t_contours)
#        #t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
#        t_contour_combs_perms[t_time] = t_perms
#    # prepare combination of branches. intermediate combinations should be gathered and frozen
#    t_branch_nodes = defaultdict(list) # for return_nodes
#    for t, t_branch_IDs in enumerate(t_branches_perms):
#        t_branch_comb_variants.append(copy.deepcopy(t_contour_combs_perms))    # copy a primer.
#        t_temp = {}                                                            # this buffer will gather multiple branches and their ID together
#        for t_branch_ID in t_branch_IDs:
#            for t_time, *t_subIDs in t_segments_new[t_branch_ID]:
#                if t_time not in t_temp: t_temp[t_time] = []
#                t_temp[t_time] += list(t_subIDs)                              # fill buffer
#        for t_time, t_subIDs in t_temp.items():
#            t_branch_comb_variants[t][t_time] += [tuple(t_subIDs)]            # walk buffer and add frozen combs to primer
#            t_branch_nodes[t_time].append(tuple(t_subIDs))

#    #aa2 = [itertools_product_length(t_choices.values()) for t_choices in t_branch_comb_variants]
#    # do N different variants and combine them together. it should be much shorter, tested on case where 138k combinations with 2 branches were reduced to 2.8k combinations
#    out = sum([list(itertools.product(*t_choices.values())) for t_choices in t_branch_comb_variants],[])
#    if return_nodes:
#        out2 = {t_time:[] for t_time in t_times_contours.keys()}
#        for t_time, t_combs in t_contour_combs_perms.items():
#            if len(t_combs) > 0:
#                for t_comb in t_combs:
#                    out2[t_time].append(t_comb)
#        for t_time, t_combs in t_branch_nodes.items():
#            for t_comb in t_combs:
#                out2[t_time].append(t_comb)
#        return out, out2
#    else:
#        return out


def save_connections_two_ways(node_segments, sols_dict, segment_from,  segment_to, ID_remap, contours_dict):
    """
    NOTE: G_time and such are not implemented
    *** is used to modify graphs and data storage with info about resolved connections between segments ***
    ----------------------------------------------------------------------------------------------------
    (1) drop all edges from last known node going into intermediate solution.
    (2) same as (1), but with other side
    (3) in case when extension (merge for example) is stopped earlier, and last intermediate node is 
    (3) composite, this node has to inherit forward edges of each solo node it consists of
    (4) reconstruct path using chain of nodes
    (5) i have to drop disperesed nodes so edges are also dropped. but also parameters will be dropped
    (5) if resolved combs may stay solo. their parameters should be copied on top of clean nodes
    (6) drop disperse nodes (this way unwanted edges are also dropped) and reconstruct only useful edges
    ----------------------------------------------------------------------------------------------------
    from sols_dict {time1:[ID1,ID2,..],..} regenerate composite nodes if there are, but also
    disperse into original solo nodes, which are a part of event space (node paths between segments)
    """
    from graphs_general import (set_custom_node_parameters, G2_set_parameters, G_owner_set)
    from graphs_general import (G, G2, G2_n_from, G2_n_to, G_time)

    graph_nodes = G
    graph_segments = G2
    

    

    nodes_solo, nodes_composite = [],[]
    for time, subIDs in sols_dict.items():                     
        for subID in subIDs:                                
            nodes_solo.append((time,subID))               
                        
        nodes_composite.append((time,) + tuple(subIDs))   # tuple([time] + list(subIDs))

    segment_from_new        = ID_remap[segment_from]      # get reference of master of this segment
    ID_remap[segment_to]    = segment_from_new            # add let master inherit slave of this segment

    node_from_last          = G2_n_to(segment_from_new)#node_segments[segment_from_new][-1]         # (1)
    from_successors         = list(graph_nodes.successors(node_from_last))                          # (1)
    from_successors_edges   = [(node_from_last, node) for node in  from_successors]                 # (1)
    graph_nodes.remove_edges_from(from_successors_edges)                                            # (1)

    node_to_first           = G2_n_from(segment_to)#node_segments[segment_to][0]                    # (2)
    to_predecessors         = list(graph_nodes.predecessors(node_to_first))                         # (2)
    to_predecessors_edges   = [(node, node_to_first) for node in  to_predecessors]                  # (2)
    graph_nodes.remove_edges_from(to_predecessors_edges)                                            # (2)

    #time_check      = lambda node: G_time(node_from_last) < G_time( node)    < G_time(node_to_first)  
    time_check_c    = lambda node: G_time(node_from_last) <         node[0] < G_time(node_to_first) 

    nodes_solo      = [node for node in nodes_solo      if time_check_c(node) ]
    nodes_composite = [node for node in nodes_composite if time_check_c(node) ]

    node_chain      = [node_from_last] + nodes_composite + [node_to_first]                          # (4)
    edges_sequence  = [(x, y) for x, y in zip(node_chain[:-1], node_chain[1:])]                     # (4)

    for node in node_segments[segment_to]:
        #graph_nodes.nodes[node]["owner"] = segment_from_new
        G_owner_set(node,segment_from_new) 

    node_segments[segment_from_new] += nodes_composite
    node_segments[segment_from_new] += node_segments[segment_to]
    node_segments[segment_to]       = []
    if segment_from_new != segment_from: node_segments[segment_from] = []

    segment_successors  = graph_segments.successors(segment_to) 
    t_edges             = [(segment_from_new,successor) for successor in segment_successors]
    #graph_segments.remove_nodes_from([segment_to])
    #graph_segments.add_edges_from(t_edges)
    
    #graph_segments.nodes()[segment_from_new]["t_end"    ] = node_segments[segment_from_new][-1][0]
    #graph_segments.nodes()[segment_from_new]["node_end" ] = node_segments[segment_from_new][-1]
            
    t_common_nodes      = set(nodes_solo).intersection(set(nodes_composite))                        # (5)
    t_composite_nodes   = set(nodes_composite) - set(nodes_solo)                                    # (5)
    t_node_params       = {t:dict(graph_nodes.nodes[t]) for t in t_common_nodes}                    # (5)
        
    graph_nodes.remove_nodes_from(nodes_solo)                                                       # (6)
    graph_nodes.add_edges_from(edges_sequence)                                                      # (6)

    for t,t_params in t_node_params.items():    
        graph_nodes.add_node(t, **t_params)
        G_owner_set(t,segment_from_new) 
        #graph_nodes.nodes[t]["owner"] = segment_from_new
            
    set_custom_node_parameters(contours_dict, t_composite_nodes, segment_from_new, calc_hull = 1)    
    G2_set_parameters(node_segments[segment_from_new], segment_from_new, edges = t_edges, remove_nodes = [segment_to])
    
def save_connections_merges(node_segments, sols_dict, segment_from,  segment_to, ID_remap, contours_dict):
    # *** is used to modify graphs and data storage with info about resolved extensions of merge branches (from left to merge node) ***
    # ref save_connections_two_ways() for docs. (2) absent, (3) is new
    from graphs_general import (set_custom_node_parameters, G2_set_parameters, G_owner_set)
    from graphs_general import (G, G2, G2_n_to, G_time)

    graph_nodes = G
    graph_segments = G2
    #G2_n_to    = lambda node: seg_n_to(    node, graph_segments)
    #G_time      = lambda node: node_time(   node, graph_nodes   )
    #G_owner_set = lambda node, owner: node_owner_set(node, graph_nodes, owner)

    nodes_solo, nodes_composite = [],[]

    for time,subIDs in sols_dict.items():                     
        for subID in subIDs:                                
            nodes_solo.append((time,subID))               
                        
        nodes_composite.append((time,) + tuple(subIDs))  

    segment_from_new        = ID_remap[segment_from]

    node_from_last          = G2_n_to(segment_from_new)#node_segments[segment_from_new][-1]        # (1)
    from_successors         = graph_nodes.successors(node_from_last)                                # (1)
    from_successors_edges   = [(node_from_last, node) for node in  from_successors]                 # (1)
    graph_nodes.remove_edges_from(from_successors_edges)                                            # (1)

    sols_node_last          = nodes_composite[-1]                                                   # (3)
    sols_nodes_disperesed   = disperse_composite_nodes_into_solo_nodes([sols_node_last])            # (3)
    sols_last_edges         = set()                                                                 # (3)
    for node in sols_nodes_disperesed:                                                              # (3)
        sols_last_edges.update([(sols_node_last,t) for t in graph_nodes.successors(node)])          # (3)
           
    time_test       = G_time(node_from_last)
    nodes_solo      = [node for node in nodes_solo      if time_test < node[0]]
    nodes_composite = [node for node in nodes_composite if time_test < node[0]]

    node_chain      = [node_from_last] + nodes_composite                                            # (4)
    edges_sequence  = [(x, y) for x, y in zip(node_chain[:-1], node_chain[1:])]                     # (4)
    edges_sequence  += list(sols_last_edges)                                                        # (4)

    node_segments[segment_from_new] += nodes_composite
 
    #graph_segments.nodes()[segment_from_new]["t_end"    ] = node_segments[segment_from_new][-1][0]
    #graph_segments.nodes()[segment_from_new]["node_end" ] = node_segments[segment_from_new][-1]
     
    t_common_nodes      = set(nodes_solo).intersection(set(nodes_composite))                        # (5)
    t_composite_nodes   = set(nodes_composite) - set(nodes_solo)                                    # (5)
    t_node_params       = {t:dict(graph_nodes.nodes[t]) for t in t_common_nodes}                    # (5)
        
    graph_nodes.remove_nodes_from(nodes_solo)                                                       # (6)
    graph_nodes.add_edges_from(edges_sequence)                                                      # (6)

    for t,t_params in t_node_params.items():    
        graph_nodes.add_node(t, **t_params)
        G_owner_set(t, segment_from_new) 
        #graph_nodes.nodes[t]["owner"] = segment_from_new
    set_custom_node_parameters(contours_dict, t_composite_nodes, segment_from_new, calc_hull = 1)  
    G2_set_parameters(node_segments[segment_from_new], segment_from_new)

    
def save_connections_splits(node_segments, sols_dict, segment_from,  segment_to,  ID_remap, contours_dict):
    # *** is used to modify graphs and data storage with info about resolved extensions of split branches (from right to split node) ***
    # ref save_connections_two_ways() for docs. (1) absent, (3) is new
    # >> maybe i have to sort nodes_composite instead of [::-1] <<< later
    from graphs_general import (set_custom_node_parameters, G2_set_parameters, G_owner_set)
    from graphs_general import (G, G2, G2_n_from, G_time)
    graph_nodes = G
    graph_segments = G2
    #G2_n_from  = lambda node: seg_n_from(  node, graph_segments)
    #G_time      = lambda node: node_time(   node, graph_nodes   )
    #G_owner_set = lambda node, owner: node_owner_set(node, graph_nodes, owner)

    nodes_solo, nodes_composite = [],[]

    for time,subIDs in sols_dict.items():                     
        for subID in subIDs:                                
            nodes_solo.append((time,subID))               
                        
        nodes_composite.append((time,) + tuple(subIDs))  

    #segment_from_new        = ID_remap[segment_from] # should not be used, since splits dont reach it

    node_to_first           = G2_n_from(segment_to)#node_segments[segment_to][0]                   # (2)
    to_predecessors         = graph_nodes.predecessors(node_to_first)                               # (2)
    to_predecessors_edges   = [(node, node_to_first) for node in  to_predecessors]                  # (2)
    graph_nodes.remove_edges_from(to_predecessors_edges)                                            # (2)

    sols_node_first = nodes_composite[-1]                                                           # (3)
    sols_nodes_disperesed = disperse_composite_nodes_into_solo_nodes([sols_node_first])             # (3)
    sols_first_edges = set()                                                                        # (3)
    for node in sols_nodes_disperesed:                                                              # (3)
        sols_first_edges.update([(t,sols_node_first) for t in graph_nodes.predecessors(node)])      # (3)
                    
    time_test = G_time(node_to_first)
    nodes_solo      = [node for node in nodes_solo      if  node[0] < time_test]
    nodes_composite = [node for node in nodes_composite if  node[0] < time_test]

    node_chain = nodes_composite[::-1] + [node_to_first]                                            # (4)
    edges_sequence = [(x, y) for x, y in zip(node_chain[:-1], node_chain[1:])]                      # (4)
    edges_sequence += list(sols_first_edges)                                                        # (4)


    node_segments[segment_to] = nodes_composite[::-1] + node_segments[segment_to]

    #graph_segments.nodes()[segment_to]["t_start"    ] = node_segments[segment_to][0][0]    #nodes_composite[0][0] # should have been nodes_composite[::-1][0]
    #graph_segments.nodes()[segment_to]["node_start" ] = node_segments[segment_to][0]       # nodes_composite[0]
     
    t_common_nodes = set(nodes_solo).intersection(set(nodes_composite))                             # (5)
    t_composite_nodes = set(nodes_composite) - set(nodes_solo)                                      # (5)
    t_node_params = {t:dict(graph_nodes.nodes[t]) for t in t_common_nodes}                          # (5)
        
    graph_nodes.remove_nodes_from(nodes_solo)                                                       # (6)
    graph_nodes.add_edges_from(edges_sequence)                                                      # (6)

    for t,t_params in t_node_params.items():    
            graph_nodes.add_node(t, **t_params)
            G_owner_set(t, segment_to)
            #graph_nodes.nodes[t]["owner"] = segment_to
    set_custom_node_parameters(contours_dict, t_composite_nodes, segment_to, calc_hull = 1)  # owner changed
    G2_set_parameters(node_segments[segment_to], segment_to)



# ===============================================================================================
# ====  FOR BRANCH EXTENSION, SEE IF THERE ARE CONTESTED NODES AND REDISTRIBUTE THEM ====
# ===============================================================================================
# WHY: extensions are performed effectively in parallel, without coupling between processes.
# WHY: thats why some recovered paths may overlap. and overlap has to be removed.
# HOW: nodes that are in multiple extensions are called contested. Since each contested node can have
# HOW: only one owner, we construct all possible choices for this kind of node redistribution.
# HOW: additionally, we cannot strip node from path that will be left with no nodes at any time step.
# NOTE: only case where no contested nodes are found and only one possible redistribution is scripted. <<<
def conflicts_stage_1(owner_dict):
    # find contested nodes by counting their owners
    node_owners = defaultdict(list)
    for owner, times_subIDs in owner_dict.items():
        # {owner:nodes,...}, where nodes = {time:*subIDs,...} = {483: (3,), 484: (4,),..}
        nodes =  [(time, subID) for time, subIDs in times_subIDs.items() for subID in subIDs]
        for node in nodes:
            node_owners[node].extend([owner])

    return {node: owners for node,owners in node_owners.items() if len(owners) > 1}

def conflicts_stage_2(contested_dict):
    # prep data for redistribution schemes. get options of node owners: node1:owners1 -> 
    # -> (owner) choices for node 1 = ([node1, owner11],[node1, owner12],..)
    # node_owner_choices -> [choices for node 1, choices for node 2,...]
    node_owner_choices = [] 
    for key,values in contested_dict.items():
        node_owner_choices.append(list(itertools.product([key],values)))
    # for fair node distribution, only one owner choice from each "choices for node X" is possible
    # so we construct combinations of single choices for each "choices for node X"
    # which is well described by permutation product (a branching choice tree).
    return list(itertools.product(*node_owner_choices))

def conflicts_stage_3(node_destribution_options,contested_node_owners_dict, owner_dict):
    # if this choice of node redistribution is correct, i have to delete contested nodes from
    # alternative owners. if some of alternative owners are left with no nodes, its incorrect choice
    variants_possible = []
    for nodes_owners in node_destribution_options: # examine one redistribution variant
        t_skip = False
        for node, owner in nodes_owners:           # examine one particular node
            time, *subIDs = node                   # determine alternative owners
            owners_others = [t for t in contested_node_owners_dict[node] if t != owner]
            for owner_other in owners_others:
                leftover_subIDs = set(owner_dict[owner_other][time]) - set(subIDs)
                if  len(leftover_subIDs) == 0:     # check subIDs after removal of contested subIDs
                    t_skip = True
                    break                          # stop examining alternative owners. case failed
            if t_skip: break                       # stop examining nodes of case
        if not t_skip:
            variants_possible.append(nodes_owners)
    return variants_possible


#if 1 == -1:
#    # check if there are contested nodes in all extrapolated paths
#    t_duplicates = conflicts_stage_1(t_extrapolate_sol_comb)
#    assert len(t_duplicates) == 0, 'havent tested after addition of split and mixed extension code 28.09.23'
#    if len(t_duplicates) > 0:
#        # retrieve viable ways of redistribute contested nodes
#        variants_all        = conflicts_stage_2(t_duplicates)
#        variants_possible   = conflicts_stage_3(variants_all,t_duplicates, t_extrapolate_sol_comb)
#        #if there is only one solution by default take it as answer
#        if len(variants_possible) == 1:  
#            t_choice_evol = variants_possible[0]
#        else:
#            # method is not yet constructed, it should be based on criterium minimization for all variants
#            # current, trivial solution, is to pick solution at random. at least there is no overlap.
#            assert -1 == 0, 'multiple variants of node redistribution'
#            t_choice_evol = variants_possible[0]
         
#        # redistribute nodes for best solution.
#        for t_node,t_conn in t_choice_evol:
#            tID                 = t_conn[1]     # not correct w.r.t different states <<<
#            t_time, *t_subIDs   = t_node
#            t_delete_conns      = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
#            for t_delete_conn in t_delete_conns:
#                t_temp = t_extrapolate_sol_comb[t_delete_conn][t_time]
#                t_temp = [t for t in t_temp if t not in t_subIDs]
#                t_extrapolate_sol_comb[t_delete_conn][t_time] = t_temp
#            t_conns_relevant = [t_c for t_c in t_extrapolate_sol_comb if t_c[1] == tID]
#            lr_conn_merges_good.update(t_conns_relevant) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< not correct anymore

#    t_all_problematic_conns = list(set(sum(list(t_duplicates.values()),[])))
#    t_all_problematic_conns_to = [a[1] for a in t_all_problematic_conns]
#    for tID,t_state in ms_branch_extend_IDs: # here lies problem with splits, and possibly with mixed cases!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        if tID not in t_all_problematic_conns_to:
#            if t_state      == 'merge':
#                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
#            elif t_state    == 'split':
#                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[0] == tID]
#            else:
#                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
#            for t_conn in t_conns_relevant:
#                lr_conn_merges_good.add((t_conn,t_state))
#            #lr_conn_merges_good[(tID,t_state)].update(t_conns_relevant)
#            #lr_conn_merges_good.update((tID,t_state))
    
#        print('branches are resolved without conflict, or conflict resolved by redistribution of nodes')


# not used =====================
def rotRect(rect):
    x,y,w,h = rect
    return (tuple((int(x+w/2),int(y+h/2))), tuple((int(w),int(h))), 0)

def set_ith_elements_multilist(index, entries, *lists):
    for lst, entry in zip(lists, entries):
        lst[index] = entry
    return


#def set_ith_elements_multilist_at_depth(indices, entry, *nested_lists):
#    # The function set_elements_at_depth allows you to modify elements in
#    # nested lists at a specific depth using provided indices.
#    # DOES the following- from:
#    # t_segments_121_centroids[t_from_new][t_node]   = t_centroid
#    # t_segments_121_areas[    t_from_new][t_node]   = t_area
#    # t_segments_121_mom_z[    t_from_new][t_node]   = t_mom_z
#    # to:
#    # set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
        
#    current_lists = nested_lists
#    for idx in indices[:-1]:
#        current_lists = [lst[idx] for lst in current_lists]
#    for lst, val in zip(current_lists, entry):
#        lst[indices[-1]] = val
#    return

#def combine_dictionaries_multi(t_from_new, t_to, *dictionaries):
#    # DOES the following- from:
#    #t_segments_121_centroids[   t_from_new] = {**t_segments_121_centroids[   t_from_new],  **t_segments_121_centroids[   t_to]}
#    #t_segments_121_areas[       t_from_new] = {**t_segments_121_areas[       t_from_new],  **t_segments_121_areas[       t_to]}
#    #t_segments_121_mom_z[       t_from_new] = {**t_segments_121_mom_z[       t_from_new],  **t_segments_121_mom_z[       t_to]}
#    # to
#    #combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
#    for dictionary in dictionaries:
#        dictionary[t_from_new] = {**dictionary[t_from_new], **dictionary[t_to]}
#    return

#def lr_init_perm_precomputed(possible_permutation_dict, initialize_value):
#    return {t_conn: {t_time: 
#                            {t_perm:initialize_value for t_perm in t_perms}
#                        for t_time,t_perms in t_times_perms.items()}
#            for t_conn,t_times_perms in possible_permutation_dict.items()}



