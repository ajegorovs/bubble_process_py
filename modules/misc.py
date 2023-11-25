

import numpy as np, time as time_lib, copy, networkx as nx
import datetime, itertools
from collections import deque
from collections import defaultdict

#from graphs_general import (extract_graph_connected_components) # gives an error of circular import


colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
np.random.seed(2);np.random.shuffle(colorList);np.random.seed()

def cyclicColor(index):
    return colorList[index % len(colorList)].tolist()

def check_overlap(segments):
    # given list of segments designated as [seg_1, seg_2,..] = [(start_1,end_1),...]
    # return which seg_ at least partially intsects with other segment, assuming continous space between start and end. otherwise use set intersection.
    overlaps = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            seg1 = segments[i]
            seg2 = segments[j]
            
            if seg1[0] <= seg2[1] and seg1[1] >= seg2[0]:
                overlaps.append((i, j))
    return overlaps


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
    resort = lambda list_of_tuple: sorted(list_of_tuple, key=lambda x: (x[0], x[1:]))
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

class AutoCreateDict:
    # lr_init_perm_precomputed remake, so dict does not need initialization.
    def __init__(self):
        self.data = {}

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = AutoCreateDict()
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        
    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __repr__(self):
        return repr(self.data)

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

def order_segment_levels(t_segments, debug = 0):
    from graphs_general import (extract_graph_connected_components)
    #[needs : check_overlap, cyclicColor, networkx as nx, plt, np]

    # want to draw bunch of segments horizontally, but they overlay and have to be stacked vertically
    # if generally they overlay locally, then they can be vertically compressed, if one of prev segments has ended

    # SEGMENTS are list of sequences of nodes like paths: [path1, path2, ..] and path = (node1, node2,..)
    # NODE holds its time in name with other paremeter, such as contour IDS: node = (time, ID1, ID2) = (time,*subIDS)
    # lambda functions are used to specify which element is time. e.g time = (lambda x: x[0])(time, ID1, ID2) 

    # initialize positions as zeroes
    t_pos = {t:0 for t in range(len(t_segments))}
    # transform segments to start-end times
    start_end = [(seg[0][0],seg[-1][0]) for seg in t_segments]
    # check which segments overlap any way
    overlap = check_overlap(start_end)
    # form a graph to extract overlapping clusters
    HH = nx.Graph()
    t_all_nodes = list(range(len(t_segments)))
    HH.add_nodes_from(t_all_nodes)
    for g in t_all_nodes:
        HH.nodes()[g]["t_start"]    = t_segments[g][0][0]
        HH.nodes()[g]["t_end"]      = t_segments[g][-1][0]
    HH.add_edges_from(overlap)

    # extract connected components =  chain of overlapping clusters
    t_cc = extract_graph_connected_components(HH, lambda x: x)

        
    for t_set in t_cc:
        
        # sort by starting position to establish hierarchy, secondary sort by end position
        t_set = sorted(t_set, key = lambda x: [HH.nodes[x]["t_start"],HH.nodes[x]["t_end"]])

        # initialize odered subcluster positions as ladder. worst case scenaro it stays a ladder
        for t_k, t_from in enumerate(t_set):
            t_pos[t_from] = t_k

            
            
        # overwrite_nodes contains nodes that gave their position to other segment. so they are out of donor pool.
        overwrite_nodes = []

        for t_k,t_from in enumerate(t_set):
            # take all which have start earlier than you:

                # first all nodes that have started earlier or at same time, except donors
            t_from_all_prev             = [tID for tID in t_set if HH.nodes[tID]["t_start"] <= HH.nodes[t_from]["t_start"] and tID not in overwrite_nodes]

                # then check which of them terminate earler than node start
            t_non_overlapping_all_prev  =  [tID for tID in t_from_all_prev if HH.nodes[tID]["t_end"] < HH.nodes[t_from]["t_start"]]

            # t_non_overlapping_all_prev are available spot donors

            if len(t_non_overlapping_all_prev)>0:
                # get their positions and take lowest. lowest crit is viable, but visually not the best
                t_places        = {t_node:t_pos[t_node] for t_node in t_non_overlapping_all_prev}
                minKey = min(t_places, key = t_places.get)

                # accept donated position and add donor to exclusion list
                t_pos[t_from]   = t_places[minKey]
                overwrite_nodes.append(minKey)

    # EDIT NOT SHOWN IN DEBUG: Collapse on uninhabited layers.
    a = 1
   
    vertical_positions = set(t_pos.values()) 
    if len(vertical_positions) > 0:
        empty_positions = set(range(max(vertical_positions) + 1)) - vertical_positions
        if len(empty_positions)>0:
            position_mapping = {}
            sorted_IDs = sorted(t_pos,key=t_pos.get)
            t_pos2 = {ID:t_pos[ID] for ID in sorted_IDs if t_pos[ID] > min(empty_positions)}
            for i, position in t_pos2.items():
                if len(empty_positions) > 0 and position > min(empty_positions) and position not in position_mapping:
                    # Find the first available empty position
                    new_position = min(empty_positions)
                    position_mapping[position] = new_position
                    t_pos[i] = new_position
                    empty_positions.remove(new_position)

    if debug:

        fig, axes = plt.subplots(1, 1, figsize=( 10,5), sharex=True, sharey=True)

        for t_from, pos in t_pos.items():
            t_start = HH.nodes[t_from]["t_start"]
            t_stop  = HH.nodes[t_from]["t_end"]
            x   = [t_start,t_stop]
            y0  = [t_from,t_from]
            y   = [pos,pos]
            axes.plot(x, y0 ,'-c',c=np.array(cyclicColor(t_from))/255, label = t_from)
            plt.plot(x, y, linestyle='dotted', c=np.array(cyclicColor(t_from))/255)

            

    return t_pos

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

def getNodePos2(dic0, S = 20):
    dups, cnts = np.unique([a for a in dic0.values()], return_counts = True)
    # relate times to 'local IDs', which are also y-positions or y-indexes
    dic = {a:np.arange(b) for a,b in zip(dups,cnts)} # each time -> arange(numDups)
    # give duplicates different y-offset 0,1,2,..
    dic2 = {t:{s:k for s,k in zip(c,[tID for tID, t_time in dic0.items() if t_time == t])} for t,c in dic.items()}
    node_positions = {}
    # offset scaled by S
    #S = 20
    for t,c in dic.items():
        # scale and later offset y-pos by mid-value
        d = c*S
        meanD = np.mean(d)
        for c2,key in dic2[t].items():
            if len(dic2[t]) == 1:
                dy = np.random.randint(low=-3, high=3)
            else: dy = 0
            # form dict in form key: position. time is x-pos. y is modified by order.
            node_positions[key] = (t,c2*S-meanD + dy)

    return node_positions

def getNodePos(test):
    dups, cnts = np.unique([a[0] for a in test], return_counts = True)
    # relate times to 'local IDs', which are also y-positions or y-indexes
    dic = {a:np.arange(b) for a,b in zip(dups,cnts)}
    # give duplicates different y-offset 0,1,2,..
    dic2 = {t:{s:k for s,k in zip(c,[a for a in test if a[0] == t])} for t,c in dic.items()}
    node_positions = {}
    # offset scaled by S
    S = 20
    for t,c in dic.items():
        # scale and later offset y-pos by mid-value
        d = c*S
        meanD = np.mean(d)
        for c2,key in dic2[t].items():
            # form dict in form key: position. time is x-pos. y is modified by order.
            node_positions[key] = (t,c2*S-meanD)
    return node_positions



segments2 = []
def segment_conn_end_start_points(connections, segment_list = segments2, nodes = 0):
    if connections is not None:
        if type(connections) == tuple:
            start,end = connections
            if nodes == 1:
                return (segment_list[start][-1],    segment_list[end][0])
            else:
                return (segment_list[start][-1][0], segment_list[end][0][0])
        if type(connections) == list:
            if nodes == 1:
                return [(segment_list[start][-1],   segment_list[end][0]) for start,end in connections]
            else:
                return [(segment_list[start][-1][0],segment_list[end][0][0]) for start,end in connections]
        else:
            return None


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

def perms_with_branches(t_to_branches,t_segments_new,t_times_contours, return_nodes = False):
    # if branch segments are present their sequences will be continuous and present fully
    # means soolution will consist of combination of branches + other free nodes.
            
    # take combination of branches
    t_branches_perms = combs_different_lengths(t_to_branches)
    #t_branches_perms = sum([list(itertools.combinations(t_to_branches, r)) for r in range(1,len(t_to_branches)+1)],[])

    #t_values_drop = t_values.copy()
    t_times_contours_drop = copy.deepcopy(t_times_contours)
    t_contour_combs_perms = {}
    t_branch_comb_variants = []
    # drop all branches from choices. i will construct more different pools of contour combinations, where branches are frozen
    for tID in t_to_branches: 
        for t, (t_time, *t_subIDs) in enumerate(t_segments_new[tID]):
            for t_subID in t_subIDs:
                t_times_contours_drop[t_time].remove(t_subID)
    # pre compute non-frozen node combinations
    for t_time,t_contours in t_times_contours_drop.items():
        t_perms = combs_different_lengths(t_contours)
        #t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
        t_contour_combs_perms[t_time] = t_perms
    # prepare combination of branches. intermediate combinations should be gathered and frozen
    t_branch_nodes = defaultdict(list) # for return_nodes
    for t, t_branch_IDs in enumerate(t_branches_perms):
        t_branch_comb_variants.append(copy.deepcopy(t_contour_combs_perms))    # copy a primer.
        t_temp = {}                                                            # this buffer will gather multiple branches and their ID together
        for t_branch_ID in t_branch_IDs:
            for t_time, *t_subIDs in t_segments_new[t_branch_ID]:
                if t_time not in t_temp: t_temp[t_time] = []
                t_temp[t_time] += list(t_subIDs)                              # fill buffer
        for t_time, t_subIDs in t_temp.items():
            t_branch_comb_variants[t][t_time] += [tuple(t_subIDs)]            # walk buffer and add frozen combs to primer
            t_branch_nodes[t_time].append(tuple(t_subIDs))

    #aa2 = [itertools_product_length(t_choices.values()) for t_choices in t_branch_comb_variants]
    # do N different variants and combine them together. it should be much shorter, tested on case where 138k combinations with 2 branches were reduced to 2.8k combinations
    out = sum([list(itertools.product(*t_choices.values())) for t_choices in t_branch_comb_variants],[])
    if return_nodes:
        out2 = {t_time:[] for t_time in t_times_contours.keys()}
        for t_time, t_combs in t_contour_combs_perms.items():
            if len(t_combs) > 0:
                for t_comb in t_combs:
                    out2[t_time].append(t_comb)
        for t_time, t_combs in t_branch_nodes.items():
            for t_comb in t_combs:
                out2[t_time].append(t_comb)
        return out, out2
    else:
        return out


def save_connections_two_ways(node_segments, sols_dict, segment_from,  segment_to, graph_nodes, graph_segments, ID_remap, contours_dict):
    from graphs_general import (set_custom_node_parameters, G2_set_parameters, seg_n_from, seg_n_to, node_time, node_owner_set)
    
    G2_n_start  = lambda node: seg_n_from(  node, graph_segments)
    G2_n_end    = lambda node: seg_n_to(    node, graph_segments)
    G_time      = lambda node: node_time(   node, graph_nodes   )
    G_owner_set = lambda node, owner: node_owner_set(node, graph_nodes, owner)

    # NOTE: G_time and such are not implemented
    # *** is used to modify graphs and data storage with info about resolved connections between segments ***
    # ----------------------------------------------------------------------------------------------------
    # (1) drop all edges from last known node going into intermediate solution.
    # (2) same as (1), but with other side
    # (3) in case when extension (merge for example) is stopped earlier, and last intermediate node is 
    # (3) composite, this node has to inherit forward edges of each solo node it consists of
    # (4) reconstruct path using chain of nodes
    # (5) i have to drop disperesed nodes so edges are also dropped. but also parameters will be dropped
    # (5) if resolved combs may stay solo. their parameters should be copied on top of clean nodes
    # (6) drop disperse nodes (this way unwanted edges are also dropped) and reconstruct only useful edges
    # ----------------------------------------------------------------------------------------------------
    # from sols_dict {time1:[ID1,ID2,..],..} regenerate composite nodes if there are, but also
    # disperse into original solo nodes, which are a part of event space (node paths between segments)

    nodes_solo, nodes_composite = [],[]
    for time, subIDs in sols_dict.items():                     
        for subID in subIDs:                                
            nodes_solo.append((time,subID))               
                        
        nodes_composite.append((time,) + tuple(subIDs))   # tuple([time] + list(subIDs))

    segment_from_new        = ID_remap[segment_from]      # get reference of master of this segment
    ID_remap[segment_to]    = segment_from_new            # add let master inherit slave of this segment

    node_from_last          = G2_n_end(segment_from_new)#node_segments[segment_from_new][-1]        # (1)
    from_successors         = list(graph_nodes.successors(node_from_last))                          # (1)
    from_successors_edges   = [(node_from_last, node) for node in  from_successors]                 # (1)
    graph_nodes.remove_edges_from(from_successors_edges)                                            # (1)

    node_to_first           = G2_n_start(segment_to)#node_segments[segment_to][0]                   # (2)
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

    segment_successors  = graph_segments.successors(segment_to) #extractNeighborsNext(graph_segments, segment_to, lambda x: graph_segments.nodes[x]["t_start"])
    t_edges             = [(segment_from_new,successor) for successor in segment_successors]
    graph_segments.remove_nodes_from([segment_to])
    graph_segments.add_edges_from(t_edges)
    
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
            
    set_custom_node_parameters(graph_nodes, contours_dict, t_composite_nodes, segment_from_new, calc_hull = 1)    
    G2_set_parameters(graph_nodes, graph_segments, node_segments[segment_from_new], segment_from_new)
    
def save_connections_merges(node_segments, sols_dict, segment_from,  segment_to, graph_nodes, graph_segments, ID_remap, contours_dict):
    # *** is used to modify graphs and data storage with info about resolved extensions of merge branches (from left to merge node) ***
    # ref save_connections_two_ways() for docs. (2) absent, (3) is new
    from graphs_general import (set_custom_node_parameters, G2_set_parameters, seg_n_to, node_time, node_owner_set)

    G2_n_end    = lambda node: seg_n_to(    node, graph_segments)
    G_time      = lambda node: node_time(   node, graph_nodes   )
    G_owner_set = lambda node, owner: node_owner_set(node, graph_nodes, owner)

    nodes_solo, nodes_composite = [],[]

    for time,subIDs in sols_dict.items():                     
        for subID in subIDs:                                
            nodes_solo.append((time,subID))               
                        
        nodes_composite.append((time,) + tuple(subIDs))  

    segment_from_new        = ID_remap[segment_from]

    node_from_last          = G2_n_end(segment_from_new)#node_segments[segment_from_new][-1]        # (1)
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
    set_custom_node_parameters(graph_nodes, contours_dict, t_composite_nodes, segment_from_new, calc_hull = 1)  
    G2_set_parameters(graph_nodes, graph_segments, node_segments[segment_from_new], segment_from_new)

    
def save_connections_splits(node_segments, sols_dict, segment_from,  segment_to, graph_nodes, graph_segments, ID_remap, contours_dict):
    # *** is used to modify graphs and data storage with info about resolved extensions of split branches (from right to split node) ***
    # ref save_connections_two_ways() for docs. (1) absent, (3) is new
    # >> maybe i have to sort nodes_composite instead of [::-1] <<< later
    from graphs_general import (set_custom_node_parameters, G2_set_parameters, seg_n_from, node_time, node_owner_set)

    G2_n_start  = lambda node: seg_n_from(  node, graph_segments)
    G_time      = lambda node: node_time(   node, graph_nodes   )
    G_owner_set = lambda node, owner: node_owner_set(node, graph_nodes, owner)

    nodes_solo, nodes_composite = [],[]

    for time,subIDs in sols_dict.items():                     
        for subID in subIDs:                                
            nodes_solo.append((time,subID))               
                        
        nodes_composite.append((time,) + tuple(subIDs))  

    #segment_from_new        = ID_remap[segment_from] # should not be used, since splits dont reach it

    node_to_first           = G2_n_start(segment_to)#node_segments[segment_to][0]                   # (2)
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
    set_custom_node_parameters(graph_nodes, contours_dict, t_composite_nodes, segment_to, calc_hull = 1)  # owner changed
    G2_set_parameters(graph_nodes, graph_segments, node_segments[segment_to], segment_to)



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


def set_ith_elements_multilist_at_depth(indices, entry, *nested_lists):
    # The function set_elements_at_depth allows you to modify elements in
    # nested lists at a specific depth using provided indices.
    # DOES the following- from:
    # t_segments_121_centroids[t_from_new][t_node]   = t_centroid
    # t_segments_121_areas[    t_from_new][t_node]   = t_area
    # t_segments_121_mom_z[    t_from_new][t_node]   = t_mom_z
    # to:
    # set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
        
    current_lists = nested_lists
    for idx in indices[:-1]:
        current_lists = [lst[idx] for lst in current_lists]
    for lst, val in zip(current_lists, entry):
        lst[indices[-1]] = val
    return

def combine_dictionaries_multi(t_from_new, t_to, *dictionaries):
    # DOES the following- from:
    #t_segments_121_centroids[   t_from_new] = {**t_segments_121_centroids[   t_from_new],  **t_segments_121_centroids[   t_to]}
    #t_segments_121_areas[       t_from_new] = {**t_segments_121_areas[       t_from_new],  **t_segments_121_areas[       t_to]}
    #t_segments_121_mom_z[       t_from_new] = {**t_segments_121_mom_z[       t_from_new],  **t_segments_121_mom_z[       t_to]}
    # to
    #combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    for dictionary in dictionaries:
        dictionary[t_from_new] = {**dictionary[t_from_new], **dictionary[t_to]}
    return

def lr_init_perm_precomputed(possible_permutation_dict, initialize_value):
    return {t_conn: {t_time: 
                            {t_perm:initialize_value for t_perm in t_perms}
                        for t_time,t_perms in t_times_perms.items()}
            for t_conn,t_times_perms in possible_permutation_dict.items()}

def itertools_product_length(choices):
    # product takes L = [[a1], [a2, b2], [a3]] and produces [[a1,a2,a3],[a1,b2,a3]]
    # number of sequences is their individual choice product 1*2*1 = 2
    sequences_length = 1
    for sublist in choices:
        sequences_length *= len(sublist)
    return sequences_length
