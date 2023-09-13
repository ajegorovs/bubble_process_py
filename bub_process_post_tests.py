import enum, copy
from collections import defaultdict
from tracemalloc import start
import numpy as np, itertools, networkx as nx, sys, time as time_lib
import cv2, os, glob, datetime, re, pickle#, multiprocessing
# import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import interpolate
# import from custom sub-folders are defined bit lower
#from imageFunctionsP2 import (overlappingRotatedRectangles,graphUniqueComponents)
# functions below
if 1 == 1:
    def convertGray2RGB(image):
        if len(image.shape) == 2:
            return cv2.cvtColor(image.copy(),cv2.COLOR_GRAY2RGB)
        else:
            return image
    mapXY = (np.load('./mapx.npy'), np.load('./mapy.npy'))
    def undistort(image):
        return cv2.remap(image,mapXY[0],mapXY[1],cv2.INTER_LINEAR)


    def timeHMS():
        return datetime.datetime.now().strftime("%H-%M-%S")

    colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
    np.random.seed(2);np.random.shuffle(colorList);np.random.seed()

    def cyclicColor(index):
        return colorList[index % len(colorList)].tolist()


    def modBR(BR,side):
        x,y,w,h  = BR
        return [x - int(max((side-w)/2,0)), y - int(max((side-h)/2,0)), max(side,w), max(side,h)]

    def rotRect(rect):
        x,y,w,h = rect
        return (tuple((int(x+w/2),int(y+h/2))), tuple((int(w),int(h))), 0)

    def rect2contour(rect):
        x,y,w,h = rect
        return np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],int).reshape(-1,1,2)

    def compositeRectArea(rects, rectsParamsArr):
        if len(rects) > 1:
            # find region where all rectangles live. working smaller region should be faster.
            minX, minY = 100000, 100000                 # Initialize big low and work down to small
            maxX, maxY = 0, 0                           # Initialize small high  and work up to big

            for ID in rects:
                x,y,w,h = rectsParamsArr[ID]
                minX = min(minX,x)
                minY = min(minY,y)
                maxX = max(maxX,x+w)
                maxY = max(maxY,y+h)

            width = maxX - minX                         # Get composite width
            height = maxY - minY                        # Get composite height
            blank = np.zeros((height,width),np.uint8)   # create a smaller canvas
            for ID in rects:                            # draw masks with offset to composite edge
                x,y,w,h = rectsParamsArr[ID]
                cntr = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],int).reshape(-1,1,2)
                cv2.drawContours( blank, [cntr], -1, 255, -1, offset = (-minX,-minY))
         
            return int(np.count_nonzero(blank))         # count pixels = area

        else:
            x,y,w,h = rectsParamsArr[rects[0]]
            return int(w*h)
    
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

    def centroid_area(contour):
        m = cv2.moments(contour)
        area = int(m['m00'])
        cx0, cy0 = m['m10'], m['m01']
        centroid = np.array([cx0,cy0])/area
        return  centroid, area
    
    # ive tested c_mom_zz in centroid_area_cmomzz. it looks correct from definition m_zz = sum_x,y[r^2] = sum_x,y[(x-x0)^2 + (y-y0)^2]
    # which is the same as sum_x,y[(x-x0)^2] + sum_x,y[(y-y0)^2] = mu(2,0) + mu(0,2), and from tests on images.
    def centroid_area_cmomzz(contour):
        m = cv2.moments(contour)
        area = int(m['m00'])
        cx0, cy0 = m['m10'], m['m01']
        centroid = np.array([cx0,cy0])/area
        c_mom_xx, c_mom_yy = m['mu20'], m['mu02']
        c_mom_zz = int((c_mom_xx + c_mom_yy))
        return  centroid, area, c_mom_zz

    # given list of nodes of tupe (time,*subIDs), gather all subIDs for each unique time and sort
    def prep_combs_clusters_from_nodes(t_nodes):
        t_times = sorted(set([t_node[0] for t_node in t_nodes]))
        t_return = {t_time:[] for t_time in t_times}
        for t_time,*t_subIDs in t_nodes:
            t_return[t_time] += [min(t_subIDs)] # <<<<<<<< in case of many IDs inside, take only minimal, as representative >><<< its not really used currently!!!
        return {t_time:sorted(t_subIDs) for t_time,t_subIDs in t_return.items()}

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

    def order_segment_levels(t_segments, debug = 0):
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

        # extract connected components =  clusters
        t_cc = extract_graph_connected_components(HH, lambda x: x)

        if debug: fig, axes = plt.subplots(1, 1, figsize=( 10,5), sharex=True, sharey=True)

        for t_set in t_cc:
        
            # sort by starting position to establish hierarchy, secondary sort by end position
            t_set = sorted(t_set, key = lambda x: [HH.nodes[x]["t_start"],HH.nodes[x]["t_end"]])

            # initialize odered subcluster positions as ladder. worst case scenaro it stays a ladder
            for t_k, t_from in enumerate(t_set):
                t_pos[t_from] = t_k

            if debug: # plot og for reference
                for t_k,t_from in enumerate(t_set):
                    t_start = HH.nodes[t_from]["t_start"]
                    t_stop  = HH.nodes[t_from]["t_end"]
                    x = [t_start,t_stop]
                    y = [t_pos[t_from],t_pos[t_from]]
                    axes.plot(x, y ,'-c',c=np.array(cyclicColor(t_k))/255, label = t_from)
            
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

                    if debug:
                        t_start = HH.nodes[t_from]["t_start"]
                        t_stop  = HH.nodes[t_from]["t_end"]
                        x = [t_start,t_stop]
                        y = [t_pos[t_from],t_pos[t_from]]
                        plt.plot(x, y ,linestyle='dotted',c=np.array(cyclicColor(t_k))/255)

        if debug:
            plt.legend(ncol=len(t_segments))
            plt.show()
        return t_pos
    
    def for_graph_plots(G):
        segments3, skipped = graph_extract_paths(G,lambda x : x[0])

        # Draw extracted segments with bold lines and different color.
        segments3 = [a for _,a in segments3.items() if len(a) > 0]
        segments3 = list(sorted(segments3, key=lambda x: x[0][0]))
        paths = {i:vals for i,vals in enumerate(segments3)}

 
        t_pos = order_segment_levels(segments3, debug = 0)

        all_segments_nodes = sorted(sum(segments3,[]), key = lambda x: [x[0],x[1]])
        all_nodes_pos = {tID:(0,0) for tID in G.nodes()}

        for t_k, t_segment in enumerate(segments3):
            for t_node in t_segment:
                all_nodes_pos[t_node] = (t_node[0],0.28*t_pos[t_k])

        not_segment_nodes = [t_node for t_node in G.nodes() if t_node not in all_segments_nodes]
        # not_segment_nodes contain non-segment nodes, thus they are solo ID nodes: (time,ID)
        not_segment_nodes_clusters = prep_combs_clusters_from_nodes(not_segment_nodes)
        for t_time, t_subIDs in not_segment_nodes_clusters.items():
            for t_k, t_subID in enumerate(t_subIDs):
                t_node = tuple([t_time,t_subID])
                all_nodes_pos[t_node] = (t_time,0.28*(t_k - 0.5))
        G = G.to_undirected()
        drawH(G, paths, all_nodes_pos)

        return None

    def interpolate_trajectory(trajectory, time_parameters, which_times, s = 10, k = 1, debug = 0, axes = 0, title = 'title', aspect = 'equal'):
    
        spline_object, _ = interpolate.splprep([*trajectory.T] , u=time_parameters, s=s,k=k) 
        interpolation_values = np.array(interpolate.splev(which_times, spline_object,ext=0))
    
        if debug == 1:
            newPlot = False
            if axes == 0:
                newPlot = True
                fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
                axes.plot(*trajectory.T , '-o')
            if min(which_times)> max(time_parameters):     # extrapolation
                t_min = max(time_parameters); t_max = max(which_times)
            else:
                t_min, t_max = min(time_parameters), max(time_parameters)
            #t_min, t_max = min(time_parameters), max(max(which_times), max(time_parameters)) # if which_times in  time_parameters, plot across time_parameters, else from min(time_parameters) to max(which_times)
            interpolation_values_d = np.array(interpolate.splev(np.arange(t_min, t_max + 0.1, 0.1), spline_object,ext=0))
            axes.plot(*interpolation_values_d, linestyle='dotted', c='orange')
            axes.scatter(*interpolation_values, c='red',s = 100)

            if newPlot:
                axes.set_aspect(aspect)
                axes.set_title(title)
                plt.show()
        return interpolation_values.T


    from collections import deque

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

    def extrapolate_find_k_s(trajectory, time, t_k_s_combs, h_interp_len_max, h_start_point_index = 0, debug = 0, debug_show_num_best = 3):
        trajectory_length   = trajectory.shape[0]

        do_work_next_n          = trajectory_length - h_interp_len_max 

        errors_tot_all  = {}
        errors_sol_all   = {}
        errors_sol_diff_norms_all   = {}
        # track previous error value for each k and track wheter iterator for set k should skip rest s.
        t_last_error_k  = {k:0.0 for k in k_all}
        t_stop_k        = {k:0 for k in k_all}
        # MAYBE redo so same trajectory part is ran with different k and s parameters instead of change parts for one comb of k,s
        # EDIT, it does not let you break early if no change is detected. trade off? idk
        for t_comb in t_k_s_combs:
            k  = t_comb[0]; s = t_comb[1]
            if t_stop_k[k] == 1: continue 
            t_errors        = {}
            t_sols          = {}
        
            t_traj_buff     = CircularBuffer(h_interp_len_max,  trajectory[h_start_point_index:h_start_point_index + h_interp_len_max])
            t_time_buff     = CircularBuffer(h_interp_len_max,  time[      h_start_point_index:h_start_point_index + h_interp_len_max])
        
            t_indicies = np.arange(h_start_point_index, h_start_point_index + do_work_next_n , 1)
            for t_start_index in t_indicies:# 
                h_num_points_available = min(h_interp_len_max, trajectory.shape[0] - t_start_index)           # but past start, there are only total-start available
                if t_start_index + h_num_points_available == trajectory_length: break
                t_predict_index = t_start_index + h_num_points_available                                  # some mumbo jumbo with indicies, but its correct

                t_real_val      = [trajectory[   t_predict_index]]
                t_predict_time  = [time[         t_predict_index]]

                t_sol = interpolate_trajectory(t_traj_buff.get_data(), t_time_buff.get_data(), which_times = t_predict_time ,s = s, k = k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')
                t_sols[t_predict_index]         = t_sol[0]
                t_errors[t_predict_index]       = np.linalg.norm(np.diff(np.concatenate((t_sol,t_real_val)), axis = 0), axis = 1)[0] 

                t_traj_buff.append(trajectory[  t_predict_index])
                t_time_buff.append(time[        t_predict_index])

            if len(t_indicies)>0:                   # short trajectory passes without iterations, so skip block inside
                t_errors_tot                        = round(np.sum(list(t_errors.values()))/len(t_errors), 3)
                errors_sol_all[t_comb]              = t_sols
                errors_sol_diff_norms_all[t_comb]   = t_errors
    
                if t_last_error_k[k] == t_errors_tot:
                    t_stop_k[k] = 1;#print(f'stop: k = {k} at s = {s}, err = {t_errors_tot}')
                else:
                    t_last_error_k[k] = t_errors_tot
                    errors_tot_all[t_comb] = t_errors_tot
            else:
                errors_tot_all[t_comb] = -1
                errors_sol_diff_norms_all[t_comb] = {}

        t_OG_comb_sol = min(errors_tot_all, key=errors_tot_all.get)

        if debug:
            (fig_k_s, axes) = plt.subplots(debug_show_num_best, 1, figsize=(12, 8), sharex=True, sharey=True)
            if debug_show_num_best == 1: axes = [axes]

            t_temp = errors_tot_all.copy()
            for ax in axes:

                t_comb_sol = min(t_temp, key=t_temp.get)
                t_traj_sol = np.array(list(errors_sol_all[t_comb_sol].values()))

                ax.plot(*trajectory.T , '-o', c='black')
                ax.scatter(*t_traj_sol.T , c='red')

                ax.set_title(f's = {t_comb_sol[1]}; k = {t_comb_sol[0]}; error= {errors_tot_all[t_comb_sol]:.2f}')
                ax.set_aspect('equal')
                t_temp.pop(t_comb_sol,None)

            plt.figure(fig_k_s.number)
            plt.show()
        return t_OG_comb_sol, errors_sol_diff_norms_all


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


    def start_timer():
        global start_time
        start_time = time_lib.time()

    def stop_timer():
        elapsed_time = time_lib.time() - start_time
        return elapsed_time

    def drawH(H, paths, node_positions):
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
        a = 1

    #drawH(H, paths, node_positions)

    def extractNeighborsNext(graph, node, time_from_node_function):
        neighbors = list(graph.neighbors(node))
        return [n for n in neighbors if time_from_node_function(n) > time_from_node_function(node)]

    def extractNeighborsPrevious(graph, node, time_from_node_function):
        neighbors = list(graph.neighbors(node))
        return [n for n in neighbors if time_from_node_function(n) < time_from_node_function(node)]

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

    def interpolateMiddle2D(t_times_prev,t_times_next,t_traj_prev, t_traj_next, t_inter_times, s = 15, debug = 0, aspect = 'equal', title = "title"):

        t_traj_concat   = np.concatenate((t_traj_prev,t_traj_next))
        t_times         = t_times_prev + t_times_next
        x, y            = t_traj_concat.T
        spline, _       = interpolate.splprep([x, y], u=t_times, s=s,k=1)
        IEpolation      = np.array(interpolate.splev(t_inter_times, spline,ext=0))
        if debug:
            t2D              = np.arange(t_times[0],t_times[-1], 0.1)
            IEpolationD      = np.array(interpolate.splev(t2D, spline,ext=0))
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes.plot(*t_traj_prev.T, '-o', c= 'black')
            axes.plot(*t_traj_next.T, '-o', c= 'black')
            axes.plot(*IEpolationD, linestyle='dotted', c='orange')
            axes.plot(*IEpolation, 'o', c= 'red')
            axes.set_title(title)
            axes.set_aspect(aspect)
            axes.legend(prop={'size': 6})
            plt.show()
        return IEpolation.T
    def interpolateMiddle2D_2(t_times,t_traj_concat,t_inter_times, s = 15, k = 1, debug = 0, aspect = 'equal', title = "title"):

        #t_traj_concat   = np.concatenate((t_traj_prev,t_traj_next)) # Nx2 array
        #t_times         = t_times_prev + t_times_next               # list of N integers
        x, y            = t_traj_concat.T                           # two N lists
        spline, _       = interpolate.splprep([x, y], u=t_times, s=s,k=k)
        IEpolation      = np.array(interpolate.splev(t_inter_times, spline,ext=0)) # t_inter_times list of M elements
        if debug:
            t2D              = np.arange(t_times[0],t_times[-1], 0.1)
            IEpolationD      = np.array(interpolate.splev(t2D, spline,ext=0))
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes.scatter(*t_traj_concat.T, c= 'black')
            axes.plot(*IEpolationD, linestyle='dotted', c='orange')
            axes.plot(*IEpolation, 'o', c= 'red')
            axes.set_title(title)
            axes.set_aspect(aspect)
            axes.legend(prop={'size': 6})
            plt.show()
        return IEpolation.T                                          # IEpolation 2xM array, _.T Mx2 array

    #def interpolateMiddle1D(t_conn, t_property_dict, segments2, t_inter_times, rescale = True, histLen = 5, s = 15, debug = 0, aspect = 'equal'):
    def interpolateMiddle1D(t_times_prev,t_times_next,t_traj_prev, t_traj_next, t_inter_times, rescale = True, s = 15, debug = 0, aspect = 'equal', title = "title"):

        t_traj_concat   = np.concatenate((t_traj_prev,t_traj_next))
        t_times         = t_times_prev + t_times_next

        if rescale:
            minVal, maxVal = min(t_traj_concat), max(t_traj_concat)
            K = max(t_times) - min(t_times)
            # scale so min-max values are same width as min-max time width
            t_traj_concat_rescaled = (t_traj_concat - minVal) * (K / (maxVal - minVal))
            spl = interpolate.splrep(t_times, t_traj_concat_rescaled, k = 1, s = s)
            IEpolation = interpolate.splev(t_inter_times, spl)
            # scale back result
            IEpolation = IEpolation / K * (maxVal - minVal) + minVal

            if debug:
                t2D              = np.arange(t_times[0],t_times[-1], 0.1)
                IEpolationD      = interpolate.splev(t2D, spl)
                IEpolationD      = IEpolationD / K * (maxVal - minVal) + minVal 

        else:
            spl = interpolate.splrep(t_times, t_traj_concat, k = 1, s = s)
            IEpolation = interpolate.splev(t_inter_times, spl)

        if debug:
            if not rescale:
                t2D              = np.arange(t_times[0],t_times[-1], 0.1)
                IEpolationD      = interpolate.splev(t2D, spl)
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes.plot(t_times_prev, t_traj_prev, '-o', c= 'black')
            axes.plot(t_times_next, t_traj_next, '-o', c= 'black')
            axes.plot(t2D, IEpolationD, linestyle='dotted', c='orange')
            axes.plot(t_inter_times, IEpolation, 'o', c= 'red')
            axes.set_title(title)
            axes.set_aspect(aspect)
            axes.legend(prop={'size': 6})
            plt.show()
        return IEpolation.T

    def interpolateMiddle1D_2(t_times,t_traj_concat, t_inter_times, rescale = True, s = 15, k = 1, debug = 0, aspect = 'equal', title = "title"):

        if rescale:
            minVal, maxVal = min(t_traj_concat), max(t_traj_concat)
            K = max(t_times) - min(t_times)
            # scale so min-max values are same width as min-max time width
            t_traj_concat_rescaled = (t_traj_concat - minVal) * (K / (maxVal - minVal))
            spl = interpolate.splrep(t_times, t_traj_concat_rescaled, k = k, s = s)
            IEpolation = interpolate.splev(t_inter_times, spl)
            # scale back result
            IEpolation = IEpolation / K * (maxVal - minVal) + minVal

            if debug:   # not really correct. wrote it for OG version with one data hole.
                t2D              = np.arange(t_times[0],t_times[-1], 0.1)
                IEpolationD      = interpolate.splev(t2D, spl)
                IEpolationD      = IEpolationD / K * (maxVal - minVal) + minVal 

        else:
            spl = interpolate.splrep(t_times, t_traj_concat, k = k, s = s)
            IEpolation = interpolate.splev(t_inter_times, spl)

        if debug:
            if not rescale:
                t2D              = np.arange(t_times[0],t_times[-1], 0.1)
                IEpolationD      = interpolate.splev(t2D, spl)
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes.scatter(t_times, t_traj_concat, c= 'black')
            axes.plot(t2D, IEpolationD, linestyle='dotted', c='orange')
            axes.plot(t_inter_times, IEpolation, 'o', c= 'red')
            axes.set_title(title)
            axes.set_aspect(aspect)
            axes.legend(prop={'size': 6})
            plt.show()
        return IEpolation.T




    def lr_evel_perm_interp_data(lr_permutation_cases,lr_121_interpolation_centroids,lr_permutation_times,
            lr_permutation_centroids_precomputed,lr_permutation_areas_precomputed,lr_permutation_mom_z_precomputed):
        
        t_all_traj      = {t_conn:[] for t_conn in lr_permutation_cases}
        t_all_areas     = {t_conn:[] for t_conn in lr_permutation_cases}
        t_all_moms      = {t_conn:[] for t_conn in lr_permutation_cases}
        t_sols_c        = {t_conn:[] for t_conn in lr_permutation_cases}
        t_sols_c_i      = {t_conn:[] for t_conn in lr_permutation_cases}
        t_sols_a        = {t_conn:[] for t_conn in lr_permutation_cases}
        t_sols_m        = {t_conn:[] for t_conn in lr_permutation_cases}

        for t_conn in lr_permutation_cases:
            t_c_interp = lr_121_interpolation_centroids[t_conn]
            for t,t_perms in enumerate(lr_permutation_cases[t_conn]):
                # dump sequences of areas and centroids for each possible trajectory
                t_temp_c = []
                t_temp_a = []     
                t_temp_m = [] 
                for t_time,t_perm in zip(lr_permutation_times[t_conn],t_perms):
                    t_temp_c.append(lr_permutation_centroids_precomputed[t_conn][t_time][t_perm])
                    t_temp_a.append(lr_permutation_areas_precomputed[t_conn][t_time][t_perm])
                    t_temp_m.append(lr_permutation_mom_z_precomputed[t_conn][t_time][t_perm])
                t_all_traj[t_conn].append(np.array(t_temp_c).reshape(-1,2))
                t_all_areas[t_conn].append(np.array(t_temp_a))
                t_all_moms[t_conn].append(np.array(t_temp_m))

            #
            t_c_inter_traj =  [t[1:-1]       for t in t_all_traj[t_conn]]
            t_c_inter_traj_diff =  [t - t_c_interp      for t in t_c_inter_traj]
            t_c_inter_traj_diff_norms = [np.linalg.norm(t, axis=1)  for t in t_c_inter_traj_diff]
            t_c_i_traj_d_norms_means  = [np.mean(t) for t in t_c_inter_traj_diff_norms]
            t_c_i_mean_min            = np.argmin(t_c_i_traj_d_norms_means)

            # check displacement norms, norm means and stdevs
            t_c_diffs = [np.diff(t,axis = 0)        for t in t_all_traj[t_conn]]
            t_c_norms = [np.linalg.norm(t, axis=1)  for t in t_c_diffs]
            t_c_means = np.mean(t_c_norms, axis=1)
            t_c_stdevs= np.std( t_c_norms, axis=1)

            t_c_mean_min    = np.argmin(t_c_means)
            t_c_stdevs_min  = np.argmin(t_c_stdevs)


            # same with areas
            t_areas = np.array(t_all_areas[t_conn])
            t_a_diffs = np.diff(t_areas, axis=1)
            t_a_d_abs = np.array([np.abs(t) for t in t_a_diffs])
            t_a_d_a_sum = np.sum(t_a_d_abs, axis = 1)
            t_a_means = np.mean(t_a_d_abs, axis=1)
            t_a_stdevs= np.std( t_a_d_abs, axis=1)

            t_a_mean_min    = np.argmin(t_a_means)
            t_a_stdevs_min  = np.argmin(t_a_stdevs)

            # same with moments
            t_moments = np.array(t_all_moms[t_conn])
            t_m_diffs = np.diff(t_moments, axis=1)
            t_m_d_abs = np.array([np.abs(t) for t in t_m_diffs])
            t_m_d_a_sum = np.sum(t_m_d_abs, axis = 1)
            t_m_means = np.mean(t_m_d_abs, axis=1)
            t_m_stdevs= np.std( t_m_d_abs, axis=1)

            t_m_mean_min    = np.argmin(t_m_means)
            t_m_stdevs_min  = np.argmin(t_m_stdevs)

            # save cases with least mean and stdev
            t_sols_c[t_conn] += [t_c_mean_min,t_c_stdevs_min]
            t_sols_c_i[t_conn] += [t_c_i_mean_min]
            t_sols_a[t_conn] += [t_a_mean_min,t_a_stdevs_min]
            t_sols_m[t_conn] += [t_m_mean_min,t_m_stdevs_min]
        return t_sols_c, t_sols_c_i, t_sols_a, t_sols_m


    def itertools_product_length(choices):
        # product takes L = [[a1], [a2, b2], [a3]] and produces [[a1,a2,a3],[a1,b2,a3]]
        # number of sequences is their individual choice product 1*2*1 = 2
        sequences_length = 1
        for sublist in choices:
            sequences_length *= len(sublist)
        return sequences_length

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

    def lr_weighted_sols(weights, t_sols, lr_permutation_cases):
        lr_weighted_solutions = {t_conn:{} for t_conn in lr_permutation_cases}
        lr_weight_c, lr_weight_c_i,lr_weight_a, lr_weight_m = weights
        # set normalized weights for methods
        lr_weights = np.array(weights)
        lr_weight_c, lr_weight_c_i,lr_weight_a, lr_weight_m = lr_weights / np.sum(lr_weights)

        t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = t_sols
        for t_conn in lr_permutation_cases:
            t_all_sols = []
            t_all_sols += t_sols_c[t_conn]
            t_all_sols += t_sols_c_i[t_conn]
            t_all_sols += t_sols_a[t_conn]
            t_all_sols += t_sols_m[t_conn]

            t_all_sols_unique = sorted(list(set(t_sols_c[t_conn] + t_sols_c_i[t_conn] + t_sols_a[t_conn] + t_sols_m[t_conn])))
     
            lr_weighted_solutions[t_conn] = {tID:0 for tID in t_all_sols_unique}
            for tID in t_all_sols_unique:
                if tID in t_sols_c[   t_conn]:
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

 
    
        lr_weighted_solutions_max = {t_conn:0 for t_conn in lr_permutation_cases}
        lr_weighted_solutions_accumulate_problems = {}
        for t_conn in lr_permutation_cases:
            t_weight_max = max(lr_weighted_solutions[t_conn].values())
            t_keys_max = [tID for tID, t_weight in lr_weighted_solutions[t_conn].items() if t_weight == t_weight_max]
            if len(t_keys_max) == 1:
                lr_weighted_solutions_max[t_conn] = t_keys_max[0]
            else:
                # >>>>>>>>>>VERY CUSTOM: take sol with max elements in total <<<<<<<<<<<<<<
                t_combs = [lr_permutation_cases[t_conn][tID] for tID in t_keys_max]
                t_combs_lens = [np.sum([len(t_perm) for t_perm in t_path]) for t_path in t_combs]
                t_sol = np.argmax(t_combs_lens) # picks first if there are same lengths
                lr_weighted_solutions_max[t_conn] = t_keys_max[t_sol]
                t_count = t_combs_lens.count(max(t_combs_lens))
                if t_count > 1: lr_weighted_solutions_accumulate_problems[t_conn] = t_combs_lens # holds poistion of t_keys_max, != all keys
                a = 1
        return lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems

    def unique_sort_list(arg, sort_function = lambda x: x):
        return sorted(list(set(arg)), key = sort_function)

    def perms_with_branches(t_to_branches,t_segments_new,t_times_contours):
        # if branch segments are present their sequences will be continuous and present fully
        # means soolution will consist of combination of branches + other free nodes.
            
        # take combination of branches
        t_branches_perms = sum([list(itertools.combinations(t_to_branches, r)) for r in range(1,len(t_to_branches)+1)],[])

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
            t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
            t_contour_combs_perms[t_time] = t_perms
        # prepare combination of branches. intermediate combinations should be gathered and frozen
        for t, t_branch_IDs in enumerate(t_branches_perms):
            t_branch_comb_variants.append(copy.deepcopy(t_contour_combs_perms))    # copy a primer.
            t_temp = {}                                                            # this buffer will gather multiple branches and their ID together
            for t_branch_ID in t_branch_IDs:
                for t_time, *t_subIDs in t_segments_new[t_branch_ID]:
                    if t_time not in t_temp: t_temp[t_time] = []
                    t_temp[t_time] += list(t_subIDs)                              # fill buffer
            for t_time, t_subIDs in t_temp.items():
                t_branch_comb_variants[t][t_time] += [tuple(t_subIDs)]            # walk buffer and add frozen combs to primer

        #aa2 = [itertools_product_length(t_choices.values()) for t_choices in t_branch_comb_variants]
        # do N different variants and combine them together. it should be much shorted, tested on case where 138k combinations with 2 branches were reduced to 2.8k combinations
        return sum([list(itertools.product(*t_choices.values())) for t_choices in t_branch_comb_variants],[])

    def disperse_nodes_to_times(nodes):
        t_perms = defaultdict(list)
        for t, *t_subIDs in nodes:
            t_perms[t].extend(t_subIDs)
        return dict(t_perms) 

    def graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments, t_segments_new, t_segments_keep, ref_node = None):
        t_segments_active_all = set(sum([vals for t,vals in lr_time_active_segments.items() if t_start <= t <= t_end],[]))
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
# =========== BUILD OUTPUT FOLDERS =============//
inputOutsideRoot            = 1                                                  # bmp images inside root, then input folder hierarchy will
mainInputImageFolder        = r'.\inputFolder'                                   # be created with final inputImageFolder, else custom. NOT USED?!?
inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\100 sccm' #

mainOutputFolder            = r'.\post_tests'                        # these are main themed folders, sub-projects go inside.
if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)
sys.path.append( os.path.join(mainOutputFolder,'modules'))
#from post_tests.modules.cropGUI import cropGUI
from cropGUI import cropGUI
from graphs_brects import (overlappingRotatedRectangles, graphUniqueComponents)
from graph_visualisation_01 import (draw_graph_with_height)


mainOutputSubFolders =  ['HFS 200 mT Series 4','sccm100-meanFix', "00001-05000"]
for folderName in mainOutputSubFolders:     
    mainOutputFolder = os.path.join(mainOutputFolder, folderName)
    if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)

a = outputImagesFolderName, outputStagesFolderName, outputStorageFolderName = ['images', 'stages',  'archives']
b = ['']*len(a)

for i,folderName in enumerate(a):   
    tempFolder = os.path.join(mainOutputFolder, folderName)
    if not os.path.exists(tempFolder): os.mkdir(tempFolder)
    b[i] = tempFolder
imageFolder, stagesFolder, dataArchiveFolder = b



# check crop mask 

cropMaskName = "-".join(mainOutputSubFolders[:2])+'-crop'
cropMaskPath = os.path.join(os.path.join(*mainOutputFolder.split(os.sep)[:-1]), f"{cropMaskName}.png")
cropMaskMissing = True if not os.path.exists(cropMaskPath) else False

meanImagePath   = os.path.join(dataArchiveFolder, "-".join(["mean"]+mainOutputSubFolders)+".npz")
meanImagePathArr= os.path.join(dataArchiveFolder, "-".join(["meanArr"]+mainOutputSubFolders)+".npz")


archivePath             = os.path.join(stagesFolder, "-".join(["croppedImageArr"]+mainOutputSubFolders)+".npz")
binarizedArrPath        = os.path.join(stagesFolder, "-".join(["binarizedImageArr"]+mainOutputSubFolders)+".npz")

dataStart           = 0 #736 #300 #  53+3+5
dataNum             = 500#5005 #130 # 7+5   

assistManually      = 1
assistFramesG       = []    #845,810,1234 2070,2187,1396
assistFrames        = [a - dataStart for a in assistFramesG]
doIntermediateData              = 1                                         # dump backups ?
intermediateDataStepInterval    = 500                                       # dumps latest data field even N steps
readIntermediateData            = 1                                         # load backups ?
startIntermediateDataAtG        = 60700                             # frame stack index ( in X_Data). START FROM NEXT FRAME
startIntermediateDataAt         = startIntermediateDataAtG - dataStart      # to global, which is pseudo global ofc. global to dataStart
# ------------------- this manual if mode  is not 0, 1  or 2



# ========================================================================================================
# ============== Import image files and process them, store in archive or import archive =================
# ========================================================================================================
exportArchive   = 0
useIntermediateData = 1
rotateImageBy   = cv2.ROTATE_180 # -1= no rotation, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180 
startFrom       = 1   #0 or 1 for sub exp                               # offset from ordered list of images- global offset?! yes archive adds images from list as range(startFrom, numImages)
numImages       = 5005 # DONT! intervalStart is what you are after!!!!! # take this many, but it will be updated: min(dataNum,len(imageLinks)-startFrom), if there are less available
postfix         = "-00001-05000"

intervalStart   = 0                         # in ordered list of images start from number intervalStart
intervalStop    = intervalStart + numImages  # and end at number intervalStop
useMeanWindow   = 0                          # averaging intervals will overlap half widths, read more below
N               = 500                        # averaging window width

#--NOTES: startFrom might work incorrectly, use it starting from 0.   
if exportArchive == 1:
    # ===================================================================================================
    # =========================== Get list of paths to image files in working folder ====================
    # ===================================================================================================
    imageLinks = glob.glob(inputImageFolder + "**/*.bmp", recursive=True) 
    if len(imageLinks) == 0:
        input("No files inside directory, copy them and press any key to continue...")
        imageLinks = glob.glob(inputImageFolder + "**/*.bmp", recursive=True) 
    # ===================================================================================================
    # ========================== Splice and sort file names based on criteria ===========================
    # ===================================================================================================
    # here is an example of badly padded data: [.\imt3509,.\img351,.\img3510,.\img3511,...]
    # ------------------------ can filter out integer values out of range -------------------------------
    # ---------------------------------------------------------------------------------------------------
             
    
    extractIntergerFromFileName = lambda x: int(re.findall('\d+', os.path.basename(x))[0])
    imageLinks = list(filter(lambda x: intervalStop > extractIntergerFromFileName(x) > intervalStart , imageLinks))
    # ----------------------- can sort numerically based on integer part---------------------------------
    imageLinks.sort(key=extractIntergerFromFileName)         # and sort alphabetically
    # ===================================================================================================
    # ======== Crop using a mask (draw red rectangle on exportad sample in manual mask folder) ==========
    # ===================================================================================================
    
    if cropMaskMissing:
        # === draw crop rectangle using gui. module only works with img link. so i save it on disk
        # open via GUI, get crop rectangle corners and draw red mask on top and save it. ===
        print(f"No crop mask in {mainOutputFolder} folder!, creating mask : {cropMaskName}.png")
        cv2.imwrite(cropMaskPath, convertGray2RGB(undistort(cv2.imread(imageLinks[0],0))))
        p1,p2 = cropGUI(cropMaskPath)
        cropMask = cv2.imread(cropMaskPath,1)
        cv2.rectangle(cropMask, p1, p2,[0,0,255],-1)
        cv2.imwrite(cropMaskPath,cropMask)

        
    else:
        cropMask = cv2.imread(cropMaskPath,1)
    # ---------------------------- Isolate red rectangle based on its hue ------------------------------
    cropMask = cv2.cvtColor(cropMask, cv2.COLOR_BGR2HSV)

    lower_red = np.array([(0,50,50), (170,50,50)])
    upper_red = np.array([(10,255,255), (180,255,255)])

    manualMask = cv2.inRange(cropMask, lower_red[0], upper_red[0])
    manualMask += cv2.inRange(cropMask, lower_red[1], upper_red[1])

    # --------------------- Extract mask contour-> bounding rectangle (used for crop) ------------------
    contours = cv2.findContours(manualMask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    [X, Y, W, H] = cv2.boundingRect(contours[0])
    # ---------------------------------------------------------------------------------------------------
    # ---------------- Rotate and fix optical distortion (should be prepared previously)-----------------
    # ---------------------------------------------------------------------------------------------------
    print(f"{timeHMS()}: Processing and saving archive data on drive...")

    numImages = min(dataNum,len(imageLinks)-startFrom)
    if rotateImageBy % 2 == 0 and rotateImageBy != -1: W,H = H,W                       # for cv2.XXX rotation commands
    dataArchive = np.zeros((numImages,H,W),np.uint8)
    for i,j in tqdm(enumerate(range(startFrom-1, numImages))):
        if rotateImageBy != -1:
            dataArchive[i]    = cv2.rotate(undistort(cv2.imread (imageLinks[j],0))[Y:Y+H, X:X+W],rotateImageBy)
        else:
            dataArchive[i]    = undistort(cv2.imread (imageLinks[j],0))[Y:Y+H, X:X+W]
    print(f"{timeHMS()}: Processing and saving archive data on drive...saving compressed")
    np.savez_compressed(archivePath,dataArchive)
    #with open(archivePath, 'wb') as handle: 
    #    pickle.dump(dataArchive, handle) 
    
    print(f"{timeHMS()}: Exporting mean image...calculating mean")

    meanImage = np.mean(dataArchive, axis=0)
    print(f"{timeHMS()}: Exporting mean image...saving compressed")
    np.savez_compressed(meanImagePath,meanImage)
    #with open(meanImagePath, 'wb') as handle:
    #    pickle.dump(meanImage, handle)
    
    print(f"{timeHMS()}: Processing and saving archive data on drive... Done!")

elif not os.path.exists(archivePath):
    print(f"{timeHMS()}: No archive detected! Please generate it from project images.")

elif not useIntermediateData:
    print(f"{timeHMS()}: Existing archive found! Importing data...")
    dataArchive = np.load(archivePath)['arr_0']
    #with open(archivePath, 'rb') as handle:
    #    dataArchive = pickle.load(handle)
    print(f"{timeHMS()}: Existing archive found! Importing data... Done!")

    if not os.path.exists(meanImagePath):
        print(f"{timeHMS()}: No mean image found... Calculating mean")

        meanImage = np.mean(dataArchive, axis=0)
        print(f"{timeHMS()}: No mean image found... Saving compressed")
        np.savez_compressed(meanImagePath,meanImage)

        #with open(meanImagePath, 'wb') as handle:
        #    pickle.dump(meanImage, handle)
        print(f"{timeHMS()}: No mean image found... Done")
    else:
        meanImage = np.load(meanImagePath)['arr_0']
        #with open(meanImagePath, 'rb') as handle:
        #    meanImage = pickle.load(handle)
#cv2.imshow(f'mean',meanImage.astype(np.uint8))



# =========================================================================================================
# discrete update moving average with window N, with intervcal overlap of N/2
# [-interval1-]         for first segment: interval [0,N]. switch to next window at i = 3/4*N,
#           |           which is middle of overlap. 
#       [-interval2-]   for second segment: inteval is [i-1/4*N, i+3/4*N]
#                 |     third switch 1/4*N +2*[i-1/4*N, i+3/4*N] and so on. N/2 between switches
if useMeanWindow == 1 and not useIntermediateData:
    meanIndicies = np.arange(0,dataArchive.shape[0],1)                                                       # index all images
    meanWindows = {}                                                                                         # define timesteps at which averaging
    meanWindows[0] = [0,N]                                                                                   # window is switched. eg at 0 use
                                                                                                             # np.mean(archive[0:N])
    meanSwitchPoints = np.array(1/4*N + 1/2*N*np.arange(1, int(len(meanIndicies)/(N/2)), 1), int)            # next switch points, by geom construct
                                                                                                             # 
    for t in meanSwitchPoints:                                                                               # intervals at switch points
        meanWindows[t] = [t-int(1/4*N),min(t+int(3/4*N),max(meanIndicies))]                                  # intervals have an overlap of N/2
    meanWindows[meanSwitchPoints[-1]] = [meanWindows[meanSwitchPoints[-1]][0],max(meanIndicies)]             # modify last to include to the end
    intervalIndecies = {t:i for i,t in enumerate(meanWindows)}                                               # index switch points {i1:0, i2:1, ...}
                                                                                                             # so i1 is zeroth interval
    print(meanWindows)                                                                                       
    print(intervalIndecies)

    if not os.path.exists(meanImagePathArr):
        print(f"{timeHMS()}: Mean window is enabled. No mean image array found. Generating and saving new...")
        masksArr = np.array([np.mean(dataArchive[start:stop], axis=0) for start,stop in meanWindows.values()])   # array of discrete averages

        with open(meanImagePathArr, 'wb') as handle:
            pickle.dump(masksArr, handle)
        print(f"{timeHMS()}: Mean window is enabled. No mean image array found. Generating and saving new... Done")
                                                     

    else:
        print(f"{timeHMS()}: Mean window is enabled. Mean image array found. Importing data...")
        with open(meanImagePathArr, 'rb') as handle:
                masksArr = pickle.load(handle)
        print(f"{timeHMS()}: Mean window is enabled. Mean image array found. Importing data... Done!")

def whichMaskInterval(t,order):                                                                          # as frames go 0,1,..numImgs
    times = np.array(list(order))                                                                        # mean should be taken form the left
    sol = 0                                                                                              # img0:[0,N],img200:[i-a,i+b],...
    for time in times:                                                                                   # so img 199 should use img0 interval
        if time <= t:sol = time                                                                          # EZ sol just to interate and comare 
        else: break                                                                                      # and keep last one that satisfies
                                                                                                         # 
    return order[sol]      

if not useIntermediateData:

    blurMean = cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT)
    if not os.path.exists(binarizedArrPath):
        binarizedMaskArr = dataArchive - blurMean                                                               # substract mean image from stack -> float
        imgH,imgW = blurMean.shape
        thresh0 = 10
        binarizedMaskArr = np.where(binarizedMaskArr < thresh0, 0, 255).astype(np.uint8)                        # binarize stack
        binarizedMaskArr = np.uint8([cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) for img in binarizedMaskArr])    # delete white objects

        print(f"{timeHMS()}: Removing small and edge contours...")
        topFilter, bottomFilter, leftFilter, rightFilter, minArea    = 80, 40, 100, 100, 180
        for i in tqdm(range(binarizedMaskArr.shape[0])):
            contours            = cv2.findContours(binarizedMaskArr[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
            areas               = np.array([int(cv2.contourArea(contour)) for contour in contours])
            boundingRectangles  = np.array([cv2.boundingRect(contour) for contour in contours])
            whereSmall      = np.argwhere(areas < minArea)
            
            # img coords: x increases from left to right; y increases from top to bottom (not usual)
            topCoords       = np.array([y+h for x,y,w,h in boundingRectangles])     # bottom    b-box coords
            bottomCoords    = np.array([y   for x,y,w,h in boundingRectangles])     # top       b-box coords
            leftCoords      = np.array([x+w for x,y,w,h in boundingRectangles])     # right     b-box coords
            rightCoords     = np.array([x   for x,y,w,h in boundingRectangles])     # left      b-box coords

            whereTop    = np.argwhere(topCoords     < topFilter)                    # bottom of b-box is within top band
            whereBottom = np.argwhere(bottomCoords  > (imgH - topFilter))           # top of b-box is within bottom band
            whereLeft   = np.argwhere(leftCoords    < leftFilter)                   # -"-"-
            whereRight  = np.argwhere(rightCoords   > (imgW - rightFilter))         # -"-"-
                                                                             
            whereFailed = np.concatenate((whereSmall, whereTop, whereBottom, whereLeft, whereRight)).flatten()
            whereFailed = np.unique(whereFailed)

            # draw over black (cover) border elements
            [cv2.drawContours(  binarizedMaskArr[i],   contours, j, 0, -1) for j in whereFailed]
                

        print(f"\n{timeHMS()}: Removing small and edge contours... Done")
        print(f"{timeHMS()}: Binarized Array archive not found... Saving")
        np.savez_compressed(binarizedArrPath,binarizedMaskArr)
        print(f"{timeHMS()}: Binarized Array archive not found... Done")
    else:
        print(f"{timeHMS()}: Binarized Array archive located... Loading")
        binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
        print(f"{timeHMS()}: Binarized Array archive located... Done")

    #err             = cv2.morphologyEx(err.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))



    print(f"{timeHMS()}: First Pass: obtaining rough clusters using bounding rectangles...")
    t_range              = range(binarizedMaskArr.shape[0])
    g0_bigBoundingRect   = {t:[] for t in t_range}
    g0_bigBoundingRect2  = {t:{} for t in t_range}
    g0_clusters          = {t:[] for t in t_range}
    g0_clusters2         = {t:[] for t in t_range}
    g0_contours          = {t:[] for t in t_range}
    g0_contours_children = {t:{} for t in t_range}
    g0_contours_hulls    = {t:{} for t in t_range}
    g0_contours_centroids= {t:{} for t in t_range} 
    g0_contours_areas    = {t:{} for t in t_range}
    for i in tqdm(range(binarizedMaskArr.shape[0])):
        # find all local contours
        #contours            = cv2.findContours(binarizedMaskArr[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours, hierarchy = cv2.findContours(binarizedMaskArr[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
        whereParentCs   = np.argwhere(hierarchy[:,:,3]==-1)[:,1] #[:,:,3] = -1 <=> (no owner)
                
        whereChildrenCs = {parent:np.argwhere(hierarchy[:,:,3]==parent)[:,1].tolist() for parent in whereParentCs}
        whereChildrenCs = {parent: children for parent,children in whereChildrenCs.items() if len(children) > 0}
        childrenContours = list(sum(whereChildrenCs.values(),[]))
        areasChildren   = {k:int(cv2.contourArea(contours[k])) for k in childrenContours}
        minChildrenArea = 120
        whereSmallChild = [k for k,area in areasChildren.items() if area <= minChildrenArea]
        whereChildrenCs = {parent: [c for c in children if c not in whereSmallChild] for parent,children in whereChildrenCs.items()}
        whereChildrenCs = {parent: children for parent,children in whereChildrenCs.items() if len(children) > 0}
        
        a = 1
        # get bounding rectangle of each contour
        boundingRectangles  = {pID: cv2.boundingRect(contours[pID]) for pID in whereParentCs}#np.array([cv2.boundingRect(contour) for contour in contours])
        # expand bounding rectangle for small contours to 100x100 box
        brectDict = {pID: modBR(brec,100) for pID,brec in boundingRectangles.items()} #np.array([modBR(brec,100) for brec in boundingRectangles])
        # for a dictionary of bounding rect and local IDs
        #brectDict = {k:brec for k,brec in enumerate(boundingRectangles2)}
        # check overlap of bounding rectangles, hint on clusters
        combosSelf = np.array(overlappingRotatedRectangles(brectDict,brectDict))
        # get cluster of overlapping bounding boxes
        cc_unique  = graphUniqueComponents(list(brectDict.keys()), combosSelf) 
        # create a bounding box of of clusters
        bigBoundingRect     = [cv2.boundingRect(np.vstack([rect2contour(brectDict[k]) for k in comb])) for comb in cc_unique]

        g0_contours[i]          = contours
        g0_contours_children[i] = whereChildrenCs
        g0_clusters[i]          = cc_unique
        g0_bigBoundingRect[i]   = bigBoundingRect
        for k,subElems in enumerate(cc_unique):
            key = tuple([i]+subElems)
            g0_bigBoundingRect2[i][key] = cv2.boundingRect(np.vstack([rect2contour(brectDict[c]) for c in subElems]))
        
        g0_contours_centroids[i]= np.zeros((len(contours),2))
        g0_contours_areas[i]    = np.zeros(len(contours), int)
        g0_contours_hulls[i]    = [None]*len(contours)
        for k,t_contour in enumerate(contours):
            t_hull                        = cv2.convexHull(t_contour)
            t_centroid,t_area             = centroid_area(t_hull)
            g0_contours_hulls[      i][k] = t_hull
            g0_contours_centroids[  i][k] = t_centroid
            g0_contours_areas[      i][k] = int(t_area)
        #img = binarizedMaskArr[i].copy()
        #[cv2.rectangle(img, (x,y), (x+w,y+h), 128, 1) for x,y,w,h in bigBoundingRect]
        #cv2.imshow('a',img)
        #a = 1
    print(f"\n{timeHMS()}: First Pass: obtaining rough clusters using bounding rectangles...Done!")

    print(f"{timeHMS()}: First Pass: forming inter-frame relations for rough clusters...")
    def ID2S(arr, delimiter = '-'):
        return delimiter.join(list(map(str,sorted(arr))))
    debug = 0
    g0_clusterConnections  = {ID:[] for ID in range(binarizedMaskArr.shape[0]-1)}
    g0_clusterConnectionsRepr = g0_clusterConnections.copy()
    g0_pairConnections  = []
    times = np.array(list(g0_bigBoundingRect2.keys()))
    for t in tqdm(times[:-2]):
        # grab this and next frame cluster bounding boxes
        oldBRs = g0_bigBoundingRect2[t]
        newBrs = g0_bigBoundingRect2[t+1]
        # grab all cluster IDs on these frames
        allKeys = list(oldBRs.keys()) + list(newBrs.keys())
        # find overlaps between frames
        combosSelf = overlappingRotatedRectangles(oldBRs,newBrs)                                       
        for conn in combosSelf:
            assert len(conn) == 2, "overlap connects more than 2 elems, not designed for"
            pairCons = list(itertools.combinations(conn, 2))
            pairCons2 = sorted(pairCons, key=lambda x: [a[0] for a in x])
            [g0_pairConnections.append(x) for x in pairCons2]
        cc_unique  = graphUniqueComponents(allKeys, combosSelf)                                       
        g0_clusterConnections[t] = cc_unique
        if t == -1:                                                                                 #  can test 2 frames
            #imgs = [convertGray2RGB(binarizedMaskArr[t].copy()), convertGray2RGB(binarizedMaskArr[t+1].copy())]
            img = convertGray2RGB(np.uint8(np.mean(binarizedMaskArr[t:t+2],axis = 0)))
            rectParamsAll = {**oldBRs,**newBrs}
            for k,comp in enumerate(cc_unique):
                for c in comp:
                    frame = c[0]-t
                    x,y,w,h = rectParamsAll[c]                                                         # connected clusters = same color
                    #cv2.rectangle(imgs[frame], (x,y), (x+w,y+h), cyclicColor(k), 1)
                    cv2.rectangle(img, (x,y), (x+w,y+h), cyclicColor(k), 1)
            #cv2.imshow(f'iter:{t} old',imgs[0])
            #cv2.imshow(f'iter:{t} new',imgs[1])
            cv2.imshow(f'iter:{t}',img)

        a = 1
    cutoff  = 500
    cutoff = min(cutoff,binarizedMaskArr.shape[0])
    g0_clusterConnections_sub = {ID:vals for ID,vals in g0_clusterConnections.items() if ID <= cutoff}
    #g0_splitsMerges = {ID:[a for a in vals if len(a)>2] for ID,vals in g0_clusterConnections_sub.items()}
    #[g0_splitsMerges.pop(ID,None) for ID,vals in g0_splitsMerges.copy().items() if len(vals) == 0]
    #g0_splitsMerges2 = {ID:[[[a,g0_clusters[a][i]] for a,i in subvals] for subvals in vals] for ID, vals in g0_splitsMerges.items()}





    allIDs = sum([list(g0_bigBoundingRect2[t].keys()) for t in g0_bigBoundingRect2],[])

    # form a graph from all IDs and pairwise connections
    H = nx.Graph()
    #H = nx.DiGraph()
    H.add_nodes_from(allIDs)
    H.add_edges_from(g0_pairConnections)
    connected_components_unique = extract_graph_connected_components(H, sort_function = lambda x: (x[0], x[1]))

    if 1 == 1:
        storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
        with open(storeDir, 'wb') as handle:
            pickle.dump(
            [
                g0_bigBoundingRect2, g0_clusters2, g0_contours,g0_contours_hulls, g0_contours_centroids,
                g0_contours_areas, g0_pairConnections, H, connected_components_unique, g0_contours_children
            ], handle) 






print(f"\n{timeHMS()}: Begin loading intermediate data...")
storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
with open(storeDir, 'rb') as handle:
    [
        g0_bigBoundingRect2, g0_clusters2, g0_contours,g0_contours_hulls, g0_contours_centroids,
        g0_contours_areas, g0_pairConnections, H, connected_components_unique, g0_contours_children
    ] = pickle.load(handle)
print(f"\n{timeHMS()}: Begin loading intermediate data...Done!, test sub-case")


# ======================
if 1 == -1:
    segments2 = connected_components_unique
    binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
    imgs = [convertGray2RGB(binarizedMaskArr[k].copy()) for k in range(binarizedMaskArr.shape[0])]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7; thickness = 4;
    for n, case in tqdm(enumerate(segments2)):
        case    = sorted(case, key=lambda x: x[0])
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                cv2.drawContours(  imgs[time],   g0_contours[time], subID, cyclicColor(n), 2)
            #x,y,w,h = cv2.boundingRect(np.vstack([g0_contours[time][ID] for ID in subIDs]))
            x,y,w,h = g0_bigBoundingRect2[time][subCase]
            #x,y,w,h = g0_bigBoundingRect[time][ID]
            cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
            [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                startPos2 = g0_contours[time][subID][-30][0] 
                [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        

    for k,img in enumerate(imgs):
        folder = r"./post_tests/testImgs2/"
        fileName = f"{str(k).zfill(4)}.png"
        cv2.imwrite(os.path.join(folder,fileName) ,img)
        #cv2.imshow('a',img)

t_all_areas = []
for t_areas in g0_contours_areas.values():
    t_all_areas += t_areas.tolist()


g0_area_mean    = np.mean(t_all_areas)
g0_area_std     = np.std(t_all_areas)

if 1 == -1:
    # Plot the histogram
    plt.hist(t_all_areas, bins=70, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.title('Histogram of Animal Heights')

    # Show the plot
   #plt.show()


a = 1

# analyze single strand
doX = 60#30#60#56#20#18#2#84#30#60#30#20#15#2#1#60#84
lessRough_all = connected_components_unique.copy()
test = connected_components_unique[doX]
# find times where multiple elements are connected (split/merge)




node_positions = getNodePos(test)
allNodes = list(H.nodes())
removeNodes = list(range(len(connected_components_unique)))
[allNodes.remove(x) for x in test]

for x in allNodes:
    H.remove_node(x)

H = H.to_directed()
t_H_edges_old = list(H.edges)
H.clear_edges()
t_H_edges_new = [sorted(list(t_edge), key = lambda x: x[0]) for t_edge in t_H_edges_old]# [((262, 4), (263, 4, 7))]
H.add_edges_from(t_H_edges_new)

#nx.draw(H, with_labels=True)
#plt.show()

segments2, skipped = graph_extract_paths(H, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
paths = {i:vals for i,vals in enumerate(segments2)}
# for_graph_plots(H)    # <<<<<<<<<<<<<<

# ===============================================================================================
# ===============================================================================================
# =============== Refine expanded BR overlap subsections with solo contours =====================
# ===============================================================================================
# ===============================================================================================
# less rough relations pass
# drop idea of expanded bounding rectangle.
# 1 contour cluster is the best case.



activeNodes = list(H.nodes())
activeTimes = np.unique([a[0] for a in activeNodes])

# extract all contours active at each time step.
lessRoughIDs = {t:[] for t in activeTimes}
for time, *subIDs in activeNodes:
    lessRoughIDs[time] += subIDs

lessRoughIDs = {time: sorted(list(set(subIDs))) for time, subIDs in lessRoughIDs.items()}

#lessRoughBRs = {}
#for time, subIDs in lessRoughIDs.items():
#    for ID in subIDs:
#        lessRoughBRs[tuple([time, ID])] = cv2.boundingRect(g0_contours[time][ID])

lessRoughBRs = {time: {tuple([time, ID]):cv2.boundingRect(g0_contours[time][ID]) for ID in subIDs} for time, subIDs in lessRoughIDs.items()}

 # grab this and next frame cluster bounding boxes
g0_pairConnections2  = []
for t in tqdm(activeTimes[:-2]):
    oldBRs = lessRoughBRs[t]
    newBrs = lessRoughBRs[t+1]
    # grab all cluster IDs on these frames
    allKeys = list(oldBRs.keys()) + list(newBrs.keys())
    # find overlaps between frames
    combosSelf = overlappingRotatedRectangles(oldBRs,newBrs)                                       
    for conn in combosSelf:
        assert len(conn) == 2, "overlap connects more than 2 elems, not designed for"
        pairCons = list(itertools.combinations(conn, 2))
        pairCons2 = sorted(pairCons, key=lambda x: [a[0] for a in x])
        [g0_pairConnections2.append(x) for x in pairCons2]
    #cc_unique  = graphUniqueComponents(allKeys, combosSelf)                                       
    #g0_clusterConnections[t] = cc_unique

print(f"\n{timeHMS()}: Refining small BR overlap with c-c distance...")
# ===============================================================================================
# ===============================================================================================
# ================ Refine two large close passing bubble pre-merge connection ===================
# ===============================================================================================
# REMARK: to reduce stress future node permutation computation we can detect cases where two
# REMARK: bubbles pass in close proximity. bounding rectangle method connects them due to higher
# REMARK: bounding rectange overlap chance. Characteristics of these cases are: bubble (node) is
# REMARK: connected to future itself and future neighbor. Difference is that distance to itself
# REMARK: is much smaller. Proper distance is closest contour-contour distance (which is slow)

t_temp_dists = {t_edge:0 for t_edge in g0_pairConnections2}
for t_edge in tqdm(g0_pairConnections2):                                                        # examine all edges
    t_node_from , t_node_to   = t_edge                                                           

    t_time_from , t_ID_from   = t_node_from                                                      
    t_time_to   , t_ID_to     = t_node_to                                                        
                                                                                                 
    t_contour_from_area = g0_contours_areas[t_time_from ][t_ID_from ]                           # grab contour are of node_from 
    t_contour_to_area   = g0_contours_areas[t_time_to   ][t_ID_to   ]                           
    if t_contour_from_area >= 0.5*g0_area_mean and t_contour_to_area >= 0.5*g0_area_mean:       # if both are relatively large
        t_contour_from  = g0_contours[t_time_from ][t_ID_from ]                                 # grab contours
        t_contour_to    = g0_contours[t_time_to   ][t_ID_to   ]                                  
        t_temp_dists[t_edge] = closes_point_contours(t_contour_from,t_contour_to, step = 5)[1]  # find closest contour-contour distance
                                                                                                 
t_temp_dists_large  = [t_edge   for t_edge, t_dist      in t_temp_dists.items() if t_dist >= 7] # extract edges with large dist
t_edges_from        = [t_node_1 for t_node_1, _         in t_temp_dists_large]                  # extract 'from' node, it may fork into multiple
t_duplicates_pre    = [t_edge   for t_edge in t_temp_dists if t_edge[0] in t_edges_from]        # find edges that start from possible forking nodes
t_duplicates_edges_pre_from = [t_node_1 for t_node_1, _ in t_duplicates_pre]                    # extract 'from' node, to count fork branches

from collections import Counter                                                                  
t_edges_from_count  = Counter(t_duplicates_edges_pre_from)                                      # count forking edges 
t_edges_from_count_two = [t_node for t_node,t_num in t_edges_from_count.items() if t_num == 2]  # extract cases with 2 connections- future-
                                                                                                # itself and future neighbor
t_edges_combs = {t_node:[] for t_node in t_edges_from_count_two}                                # prep t_from_node:neighbors storage

for t_node_1, t_node_2 in t_temp_dists:                                                         # walk though all edges
    if t_node_1 in t_edges_from_count_two:                                                      # visit relevant
        t_edges_combs[t_node_1] += [t_node_2]                                                   # store neighbors
                                                                                                # filter out cases where big nodes
t_edges_combs_dists =   {t_node:                                                                # grab distances for these edges
                            {t_neighbor:t_temp_dists[(t_node,t_neighbor)] for t_neighbor in t_neighbors} 
                                for t_node, t_neighbors in t_edges_combs.items()}

g0_edges_merge_strong   = [(0,0)]*len(t_edges_combs_dists)                                      # predefine storage (bit faster)
g0_edges_merge_weak     = [(0,0)]*len(t_edges_combs_dists)
for t_k,(t_node_from, t_neighbor_dists) in enumerate(t_edges_combs_dists.items()):

    t_node_dist_small   = min(t_neighbor_dists, key = t_neighbor_dists.get)                     # small distance = most likely connection
    t_node_dist_big     = max(t_neighbor_dists, key = t_neighbor_dists.get)                     # opposite

    g0_edges_merge_strong[  t_k] = (  t_node_from, t_node_dist_small  )                         # these will be asigned as extra property
    g0_edges_merge_weak[    t_k] = (  t_node_from, t_node_dist_big    )                         # for future analysis on graph formation

G_e_m = nx.Graph()
G_e_m.add_edges_from(g0_edges_merge_strong)
g0_edges_merge_strong_clusters = extract_graph_connected_components(G_e_m, lambda x: (x[0],x[1]))
print(f"\n{timeHMS()}: Refining small BR overlap with c-c distance... done")                     

a = 1
#bigBoundingRect     = [cv2.boundingRect(np.vstack([rect2contour(brectDict[k]) for k in comb])) for comb in cc_unique]

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7; thickness = 4;

# for_graph_plots(G)    # <<<<<<<<<<<<<<

# ===============================================================================================
# ===============================================================================================
# =========== Extract solo-to-solo bubble trajectories from less rough graphs ===================
# ===============================================================================================
# REMARK: it is very likely that solo-to-solo (later called 121) is pseudo split-merge, optical effect

allIDs = sum([list(a.keys()) for a in lessRoughBRs.values()],[])
allIDs = sorted(allIDs, key=lambda x: (x[0], x[1]))
#G = nx.Graph()
#G.add_nodes_from(allIDs)
#G.add_edges_from(g0_pairConnections2)

G = nx.DiGraph()
G.add_nodes_from(allIDs)
G.add_edges_from(g0_pairConnections2)
for t_node in G.nodes():
    t_time, t_ID                = t_node
    G.nodes[t_node]["time"]     = int(t_time)
    t_centroid, t_area, t_mom_z = centroid_area_cmomzz(g0_contours_hulls[t_time][t_ID])
    G.nodes[t_node]["centroid"] = t_centroid
    G.nodes[t_node]["area"]     = t_area
    G.nodes[t_node]["moment_z"] = t_mom_z

for t_edge in g0_edges_merge_strong:                                          # assign previously acquired 2 bubble merge 
    G[t_edge[0]][t_edge[1]]['edge_merge_strong'] = True                       # edge of high of high likelihood

for t_edge in g0_edges_merge_weak:
    G[t_edge[0]][t_edge[1]]['edge_merge_strong'] = False

for t_edge in G.edges():
    (t_time_from,t_ID_from),(t_time_to,t_ID_to) = t_edge
    t_area_before   = g0_contours_areas[t_time_from ][t_ID_from ]
    t_area_after    = g0_contours_areas[t_time_to   ][t_ID_to   ]
    t_relative_area_change = np.abs((t_area_after-t_area_before)/t_area_before)

    t_centroids_before  = g0_contours_centroids[t_time_from ][t_ID_from ]
    t_centroids_after   = g0_contours_centroids[t_time_to   ][t_ID_to   ]
    t_edge_distance     = np.linalg.norm(t_centroids_after-t_centroids_before)

    G[t_edge[0]][t_edge[1]]['edge_rel_area_change'  ] = t_relative_area_change
    G[t_edge[0]][t_edge[1]]['edge_distance'         ] = t_edge_distance


distances       = nx.get_edge_attributes(G, 'edge_distance'         )
area_changes    = nx.get_edge_attributes(G, 'edge_rel_area_change'  )
tt = []
tt2 = []
for path in g0_edges_merge_strong_clusters:
    path_distances      = [distances[edge] for edge in zip(path, path[1:])]
    path_area_changes   = [area_changes[edge] for edge in zip(path, path[1:])]
    tt.append(path_distances)
    tt2.append(path_area_changes)
t_edges_strong_2 = []
for t_nodes_path, t_dists, t_areas_rel in zip(g0_edges_merge_strong_clusters,tt,tt2):
    t_dist_mean = np.mean(t_dists)
    t_dist_std  = np.std( t_dists)
    t_dist_low_max          = True if max(t_dists)              <  20    else False
    t_dist_low_dispersion   = True if 2*t_dist_std/t_dist_mean  <= 0.2   else False
    t_area_rel_low_max      = True if max(t_areas_rel)          <  0.2   else False
    if t_dist_low_max and t_dist_low_dispersion and t_area_rel_low_max:
        t_edges = [t_edge for t_edge in zip(t_nodes_path, t_nodes_path[1:])]
        t_edges_strong_2 += t_edges

t_edges_strong_node_start = [t_edge[0] for t_edge in t_edges_strong_2]
t_edges_weak_discard = []
for t_edge in g0_edges_merge_weak:
    if t_edge[0] in t_edges_strong_node_start:
        t_edges_weak_discard.append(t_edge)

G.remove_edges_from(t_edges_weak_discard)
g0_pairConnections2 = [t_edge for t_edge in g0_pairConnections2 if t_edge not in t_edges_weak_discard]
a = 1
node_positions = getNodePos(allIDs)

f = lambda x : x[0]

segments2, skipped = graph_extract_paths(G, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

a = 1

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
paths = {i:vals for i,vals in enumerate(segments2)}

lr_allNodesSegm = sum(segments2,[])
lr_missingNodes = [node for node in allIDs if node not in lr_allNodesSegm] # !! may be same as &skipped !!
assert set(skipped)==set(lr_missingNodes), "set(skipped) is not same as set(lr_missingNodes)"



# ===============================================================================================
# ===============================================================================================
# === find POSSIBLE interval start-end connectedness: start-end exist within set time interval ==
# ===============================================================================================
# ===============================================================================================
# REMARK: it is expected that INTER segment space is limited, so no point searching paths from
# REMARK: one segment to each other in graph. instead inspect certain intervals of set length 
# REMARK: lr_maxDT. Caveat is that inter-space can include smaller segments.. its dealth with after

# get start and end of good segments
lr_start_end    = [[segm[0],segm[-1]] for segm in segments2]
lr_all_start    = np.array([a[0][0] for a in lr_start_end])
lr_all_end      = np.array([a[1][0] for a in lr_start_end])

# get nodes that begin not as a part of a segment = node with no prior neigbors
lr_nodes_other = []
lr_nodes_solo = []
for node in lr_missingNodes:
    #neighbors = list(G.neighbors(node))
    #oldNeigbors = [n for n in neighbors if n[0] < node[0]]
    #newNeigbors = [n for n in neighbors if n[0] > node[0]]
    oldNeigbors = list(G.predecessors(  node))
    newNeigbors = list(G.successors(    node))
    if len(oldNeigbors) == 0 and len(newNeigbors) == 0: lr_nodes_solo.append(node)
    elif len(oldNeigbors) == 0:                         lr_nodes_other.append(node)

# find which segements are separated by small number of time steps
#  might not be connected, but can check later
lr_maxDT = 20

# test non-segments
lr_DTPass_other = {}
for k,node in enumerate(lr_nodes_other):
    timeDiffs = lr_all_start - node[0]
    goodDTs = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
    if len(goodDTs) > 0:
        lr_DTPass_other[k] = goodDTs


# test segments. search for endpoint-to-startpoint DT

lr_DTPass_segm = {}
for k,endTime in enumerate(lr_all_end):
    timeDiffs = lr_all_start - endTime
    goodDTs = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
    if len(goodDTs) > 0:
        lr_DTPass_segm[k] = goodDTs

# ===============================================================================================
# ===============================================================================================
# === find ACTUAL interval start-end connectedness: get all connected paths if there are any ==
# ===============================================================================================
# REMARK: refine previously acquired potential segment connectedness by searching paths between

# check connection between "time-localized" segments
# isolate all nodes on graph that are active in unresolved time between segment existance
# find all paths from one segment end to other start
lr_paths_segm   = {}
for startID,endIDs in lr_DTPass_segm.items():
    startNode = lr_start_end[startID][1]
    startTime = startNode[0]
    for endID in endIDs:
        endNode = lr_start_end[endID][0]
        endTime = lr_all_end[endID]
        #activeNodes = [node for node in allIDs if startTime <= node[0] <= endTime]
        activeNodes = [node for node, time_attr in G.nodes(data='time') if startTime <= time_attr <= endTime]
        subgraph = G.subgraph(activeNodes)

        try:
            shortest_path = list(nx.all_shortest_paths(subgraph, startNode, endNode))
            
        except nx.NetworkXNoPath:
            shortest_path = []
        
        if len(shortest_path)>0: lr_paths_segm[tuple([startID,endID])] = shortest_path

lr_paths_other  = {}
for startID,endIDs in lr_DTPass_other.items():
    startNode = lr_nodes_other[startID]
    startTime = startNode[0]
    for endID in endIDs:
        endNode = lr_start_end[endID][0]
        endTime = lr_all_end[endID]
        #activeNodes = [node for node in allIDs if startTime <= node[0] <= endTime]
        activeNodes = [node for node, time_attr in G.nodes(data='time') if startTime <= time_attr <= endTime]
        subgraph = G.subgraph(activeNodes)
        try:
            shortest_path = list(nx.all_shortest_paths(subgraph, startNode, endNode))
        except nx.NetworkXNoPath:
            shortest_path = []
        
        if len(shortest_path)>0: lr_paths_other[tuple([str(startID),str(endID)])] = shortest_path

lr_paths_segm2 = {a:[b[0][0],b[0][-1]] for a,b in lr_paths_segm.items()}
a = 1

# remember which segment indicies are active at each time step
lr_time_active_segments = {t:[] for t in lessRoughBRs}
for k,t_segment in enumerate(segments2):
    t_times = [a[0] for a in t_segment]
    for t in t_times:
        lr_time_active_segments[t].append(k)

# extract trusted segments.
# high length might indicate high trust

lr_segments_lengths = {}
for k,t_segment in enumerate(segments2):
    lr_segments_lengths[k]  = len(t_segment)


lr_trusted_segment_length = 7
lr_trusted_segments_by_length = []

for k,t_segment in enumerate(segments2):

    lr_segments_lengths[k]  = len(t_segment)

    if len(t_segment) >= lr_trusted_segment_length:
        lr_trusted_segments_by_length.append(k)

# small paths between segments might mean that longer, thus thrusted, segment is present

# store interval lentghs
lr_intervals_lengths = {}

for t_conn, intervals in lr_paths_segm.items():

    lr_intervals_lengths[t_conn] = len(intervals[0])

# extract possible good intervals

lr_trusted_max_interval_length_prio = 5
lr_trusted_max_interval_length      = 3

lr_trusted_interval_test_prio   = []
lr_trusted_interval_test        = []
for t_conn, t_interval_length in lr_intervals_lengths.items():

    if t_interval_length > lr_trusted_max_interval_length_prio: continue

    t_from,t_to = t_conn

    t_good_from = t_from in lr_trusted_segments_by_length
    t_good_to   = t_to in lr_trusted_segments_by_length

    if (t_good_from and t_good_to):
        lr_trusted_interval_test_prio.append(t_conn)
    elif t_interval_length <= lr_trusted_max_interval_length:
        lr_trusted_interval_test.append(t_conn)
  

# check interconnectedness of segments
G2 = nx.Graph()
G2.add_nodes_from(range(len(segments2))) # segment might not interact via edges, so need to add to graph explicitly
# Iterate over the dictionary and add edges with weights
for (node1, node2), weight in lr_intervals_lengths.items():
    G2.add_edge(node1, node2, weight=weight)

for g in G2.nodes():
      G2.nodes()[g]["t_start"] = segments2[g][0][0]
      G2.nodes()[g]["t_end"] = segments2[g][-1][0]
#pos = nx.spring_layout(G2)
#edge_widths = [data['weight'] for _, _, data in G2.edges(data=True)]
#nx.draw(G2, pos, with_labels=True, width=edge_widths)

#plt.show()

# ===============================================================================================
# ===============================================================================================
# === extract closest neighbors of segments ===
# ===============================================================================================
# ===============================================================================================
# REMARK: inspecting closest neighbors allows to elliminate cases with segments inbetween 
# REMARK: connected segments via lr_maxDT. problem is that affects branches of splits/merges
# EDIT 27/07/24 maybe should redo in directed way. should be shorter and faster
if 1 == 1:
    t_nodes = sorted(list(G2.nodes()))
    t_neighbor_sol_all_prev = {tID:{} for tID in t_nodes}
    t_neighbor_sol_all_next = {tID:{} for tID in t_nodes}
    for node in t_nodes:
        # Get the neighboring nodes
        t_neighbors     = list(G2.neighbors(node))
        t_time_start    = G2.nodes()[node]["t_start"]#segments2[node][0][0]
        t_time_end      = G2.nodes()[node]["t_end"]#segments2[node][-1][0]
        t_neighbors_prev = []
        t_neighbors_next = []
        for t_neighbor in t_neighbors:
            t_time_neighbor_start    = G2.nodes()[t_neighbor]["t_start" ]#segments2[t_neighbor][0][0]
            t_time_neighbor_end      = G2.nodes()[t_neighbor]["t_end"   ]#segments2[t_neighbor][-1][0]
            #if t_time_start < t_time_neighbor_start and t_time_end < t_time_neighbor_start:  # EDIT 24.08.23
            #    t_neighbors_next.append(t_neighbor)                                          # EDIT 24.08.23
            #elif t_time_start > t_time_neighbor_end and t_time_end > t_time_neighbor_end:    # EDIT 24.08.23
            #    t_neighbors_prev.append(t_neighbor)                                          # EDIT 24.08.23
            if t_time_end < t_time_neighbor_start:
                t_neighbors_next.append(t_neighbor)
            elif t_time_neighbor_end < t_time_start:
                t_neighbors_prev.append(t_neighbor)
        # check if neighbors are not lost, or path generation is incorrect, like looping back in time
        assert len(t_neighbors) == len(t_neighbors_prev) + len(t_neighbors_next), "missing neighbors, time position assumption is wrong"
    
        t_neighbors_weights_prev = {}
        t_neighbors_weights_next = {}
    
        for t_neighbor in t_neighbors_prev: # back weights are negative
            t_neighbors_weights_prev[t_neighbor] = -1*G2[node][t_neighbor]['weight']
        for t_neighbor in t_neighbors_next:
            t_neighbors_weights_next[t_neighbor] = G2[node][t_neighbor]['weight']
        t_neighbors_time_prev = {tID:segments2[tID][-1][0]  for tID in t_neighbors_weights_prev}
        t_neighbors_time_next = {tID:segments2[tID][0][0]   for tID in t_neighbors_weights_next}
    
        if len(t_neighbors_weights_prev)>0:
            # neg weights, so max, get time of nearset branch t0. get all connections within [t0 - 2, t0] in case of split
            t_val_min = max(t_neighbors_weights_prev.values())
            t_key_min_main = max(t_neighbors_weights_prev, key = t_neighbors_weights_prev.get)
            t_key_main_ref_time = t_neighbors_time_prev[t_key_min_main]
            t_sol = [key for key,t in t_neighbors_time_prev.items() if t_key_main_ref_time - 1 <= t <=  t_key_main_ref_time] #not sure why-+2, changed to -1 
            #t_sol = [key for key, value in t_neighbors_weights_prev.items() if t_val_min - 1 <= value <=  t_val_min]
            for t_node in t_sol:t_neighbor_sol_all_prev[node][t_node] = t_neighbors_weights_prev[t_node]
        
        if len(t_neighbors_weights_next)>0:
            t_val_min = min(t_neighbors_weights_next.values())
            t_key_min_main = min(t_neighbors_weights_next, key = t_neighbors_weights_next.get)
            t_key_main_ref_time = t_neighbors_time_next[t_key_min_main]
            t_sol = [key for key,t in t_neighbors_time_next.items() if t_key_main_ref_time <= t <=  t_key_main_ref_time + 1] # not sure why +2, changed to +1 
            #t_sol = [key for key, value in t_neighbors_weights_next.items() if t_val_min +1 >= value >= t_val_min]          # cause i was overextending minimal segment lenght of 2
            for t_node in t_sol: t_neighbor_sol_all_next[node][t_node] = t_neighbors_weights_next[t_node]
        a = 1

# ===============================================================================================
# ===============================================================================================
# === wipe inter-segement edges from G2 graph and replot only nearest connections ===
# ===============================================================================================
# ===============================================================================================
# prepare plot for segment interconnectedness
G2.remove_edges_from(list(G2.edges()))
for node, t_conn in t_neighbor_sol_all_prev.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2, weight=1/np.abs(weight))
for node, t_conn in t_neighbor_sol_all_next.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2, weight=1/weight)
#for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ===============================================================================================
# === calculate all paths between nearest segments ===
# ===============================================================================================
# ===============================================================================================
# >>>>>>> MAYBE FOR BIG INTER-SEGMENT SAVE ONLY SOLO PATH, THIS WAY YOU CAN KNOW IT CANT BE ANYTHING ELSE <<<<<
#segment_conn_end_start_points((11,20), nodes = 1)
aa = G.is_directed();print(f'G s directed:{aa}')

if 1 == 1:
    lr_close_segments_simple_paths = {}
    lr_close_segments_simple_paths_inter = {}
    tIDs = [tID for tID,t_dic in t_neighbor_sol_all_next.items() if len(t_dic) >0]
    for t_from in tIDs:
        for t_to in t_neighbor_sol_all_next[t_from]:
            t_from_node_last   = segments2[t_from][-1]
            t_to_node_first      = segments2[t_to][0]

            # for directed graph i can disable cutoff, since paths can progress only forwards
            t_from_node_last_time   = t_from_node_last[0]
            t_to_node_first_time    = t_to_node_first[0]
            t_from_to_max_time_steps= t_to_node_first_time - t_from_node_last_time + 1
            hasPath = nx.has_path(G, source=t_from_node_last, target=t_to_node_first)
            # soo.. all_simple_paths method goes into infinite (?) loop when graph is not limited, shortest_simple_paths does not.
            t_start, t_end = t_from_node_last[0], t_to_node_first[0]
            activeNodes = [t_node for t_node, t_time in G.nodes(data='time') if t_start <= t_time <= t_end]
            g_limited = G.subgraph(activeNodes)
            #t_from_to_paths_simple  = list(nx.shortest_simple_paths(G, t_from_node_last, t_to_node_first)) # , cutoff = t_from_to_max_time_steps

            #if t_to == -1:
            #    t_short = nx.shortest_path(g_limited, source=t_from_node_last, target=t_to_node_first, weight=None, method='dijkstra')
            #    t_short_edges = []
            #    for t_node in t_short[1:-1]:
            #        t_short_edges += g_limited.predecessors(t_node)
            #        t_short_edges += g_limited.successors(t_node)
            #    t_short_edges2 = [t_node for t_node in t_short_edges if t_node not in t_short]
            #    a = 1
            t_from_to_paths_simple  = list(nx.all_simple_paths(g_limited, t_from_node_last, t_to_node_first, cutoff = t_from_to_max_time_steps)) 
            t_from_to_paths_nodes_all       = sorted(set(sum(t_from_to_paths_simple,[])),key=lambda x: x[0])
            t_from_to_paths_nodes_all_inter = [t_node for t_node in t_from_to_paths_nodes_all if t_node not in [t_from_node_last,t_to_node_first]]

            lr_close_segments_simple_paths[         tuple([t_from,t_to])]  = t_from_to_paths_simple
            lr_close_segments_simple_paths_inter[   tuple([t_from,t_to])]  = t_from_to_paths_nodes_all_inter
        


    node_positions2 = {}
    t_segments_times = {}
    labels = {}
    for t,segment in enumerate(segments2):
        t_times = [a[0] for a in segment]
        t_segments_times[t] = int(np.mean(t_times))
        labels[t] = f'{t}_{segment[0]}'
    node_positions2 = getNodePos2(t_segments_times, S = 1)


    #for g in G2.nodes():
    #  G2.nodes()[g]["height"] = node_positions2[g][0]
    #draw_graph_with_height(G2,figsize=(5,5), labels=labels)


    #pos = nx.spring_layout(G2, pos = node_positions2, k = 1, iterations = 10)

    #nx.draw(G2, pos=node_positions2, labels=labels)
    #plt.show()
    
# ===============================================================================================
# ====== rudamentary split/merge analysis based on nearest neighbor connectedness symmetry ======
# ===============================================================================================
# REMARK: nearest neighbors elliminates unidirectional connectedness between split/merge branches

# by analyzing nearest neighbors you can see anti-symmetry in case of splits and merges.
# if one branch of merge is closer to merge product, it will have strong connection both ways
# further branches will only have unidirectional connection
if 1==1:
    lr_connections_unidirectional   = []
    lr_connections_forward          = []
    lr_connections_backward         = []
    for t_from, t_forward_conns in t_neighbor_sol_all_next.items():
        for t_to in t_forward_conns.keys():
            lr_connections_forward.append(tuple(sorted([t_from,t_to])))
    for t_to, t_backward_conns in t_neighbor_sol_all_prev.items():
        for t_from in t_backward_conns.keys():
            lr_connections_backward.append(tuple(sorted([t_from,t_to])))

    lr_connections_unidirectional   = sorted(list(set(lr_connections_forward) & set(lr_connections_backward)), key = lambda x: x[0])
    lr_connections_directed         = [t_conn for t_conn in lr_connections_forward + lr_connections_backward if t_conn not in lr_connections_unidirectional]
    lr_connections_directed         = sorted(list(set(lr_connections_directed)), key = lambda x: x[0])

    # lr_connections_directed contain merges/splits, but also lr_connections_unidirectional contain part of splits/merges.
    # unidirectional means last/next segment is connected via unidirectional ege
    t_merge_split_culprit_edges = []
    for t_conn in lr_connections_directed:
        if t_conn in lr_connections_forward:    t_from, t_to = t_conn  
        else:                                   t_to, t_from = t_conn
        # by knowing direction of direactional connection, i can tell that opposite direction connection is absent.
        # there are other connection/s in that opposite (time-wise) directions which are other directional connectsions or unidir
        t_time_to   = segments2[t_to    ][0][0]
        t_time_from = segments2[t_from  ][0][0]
        # if directed connection i time-forward, then unidir connections are from t_to node back in time
        t_forward = True if t_time_to - t_time_from > 0 else False
        if t_forward:
            t_unidir_neighbors = list(t_neighbor_sol_all_prev[t_to].keys())
        else:
            t_unidir_neighbors = list(t_neighbor_sol_all_next[t_to].keys())
        t_unidir_conns = [tuple(sorted([t_to,t])) for t in t_unidir_neighbors]
        t_merge_split_culprit_edges += [t_conn]
        t_merge_split_culprit_edges += t_unidir_conns
        a = 1
        #if t_to in t_neighbor_sol_all_prev[t_from]:
    t_merge_split_culprit_edges = sorted(t_merge_split_culprit_edges, key = lambda x: x[0])


    # simply extract nodes and their neigbors if there are multiple neighbors
    t_merge_split_culprit_edges2= []
    for t_from, t_forward_conns in t_neighbor_sol_all_next.items():
        if len(t_forward_conns)>1:
            for t_to in t_forward_conns.keys():
                t_merge_split_culprit_edges2.append(tuple(sorted([t_from,t_to])))

    for t_to, t_backward_conns in t_neighbor_sol_all_prev.items():
        if len(t_backward_conns)>1:
            for t_from in t_backward_conns.keys():
                t_merge_split_culprit_edges2.append(tuple(sorted([t_from,t_to])))

    t_merge_split_culprit_edges2 = sorted(t_merge_split_culprit_edges2, key = lambda x: x[0])
    
    t_merge_split_culprit_edges_all = sorted(list(set(t_merge_split_culprit_edges + t_merge_split_culprit_edges2)), key = lambda x: x[0])
    t_merge_split_culprit_node_combos = segment_conn_end_start_points(t_merge_split_culprit_edges_all, segment_list = segments2, nodes = 1)

    # find clusters of connected nodes of split/merge events. this way instead of sigment IDs, because one
    # segment may be sandwitched between any of these events and two cluster will be clumped together

    T = nx.Graph()
    T.add_edges_from(t_merge_split_culprit_node_combos)
    connected_components_unique = extract_graph_connected_components(T, sort_function = lambda x: x)
    
    # relate connected node families ^ to segments
    lr_merge_split_node_families = []
    for t_node_cluster in connected_components_unique:
        t_times_active = [t_node[0] for t_node in t_node_cluster]
        t_active_segments = set(sum([lr_time_active_segments[t_time] for t_time in t_times_active],[]))
        t_sol = []
        for t_node in t_node_cluster:
            for t_segment_ID in t_active_segments:
                if t_node in segments2[t_segment_ID]:
                    t_sol.append(t_segment_ID)
                    break
        lr_merge_split_node_families.append(sorted(t_sol))

# ===============================================================================================
# merge/split classification will be here
# ===============================================================================================
# REMARK: if a node has MULTIPLE neighbors from one of side (time-wise)
# REMARK: then split or merge happened, depending which side it is
# REMARK: if cluster has both split and merge of nodes, its classified separately
# REMARK: edges of these 3 events are stored for further analysis
lr_conn_edges_splits                = []
lr_conn_edges_merges                = []
lr_conn_edges_splits_merges_mixed   = []    # not used yet

lr_conn_merges_to_nodes             = []
lr_conn_splits_from_nodes           = []    # not used yet

if 1 == 1:
    
    for t_cluster in lr_merge_split_node_families:

        t_neighbors_prev = {tID:[] for tID in t_cluster}
        t_neighbors_next = {tID:[] for tID in t_cluster}

        for tID in t_cluster:
            t_neighbors_all = [t for t in list(G2.neighbors(tID)) if t in t_cluster]
            t_node_start    = G2.nodes[tID]["t_start"]
            t_node_end      = G2.nodes[tID]["t_end"]
            for t_neighbor in t_neighbors_all:
                t_neighbor_start    = G2.nodes[t_neighbor]["t_start"]
                t_neighbor_end      = G2.nodes[t_neighbor]["t_end"]
                if t_neighbor_start > t_node_end:
                    t_neighbors_next[tID].append(t_neighbor)
                elif t_neighbor_end < t_node_start:
                    t_neighbors_prev[tID].append(t_neighbor)
    
        t_neighbors_prev_large = {tID:t_neighbors for tID,t_neighbors in t_neighbors_prev.items() if len(t_neighbors) > 1}
        t_neighbors_next_large = {tID:t_neighbors for tID,t_neighbors in t_neighbors_next.items() if len(t_neighbors) > 1}
    
        t_edges_merge =  sum([[tuple(sorted([id1,id2])) for id2 in subIDs] for id1,subIDs in t_neighbors_prev_large.items()],[])
        t_edges_split =  sum([[tuple(sorted([id1,id2])) for id2 in subIDs] for id1,subIDs in t_neighbors_next_large.items()],[])
    
        if len(t_neighbors_next_large) == 0 and len(t_neighbors_prev_large) > 0:
            lr_conn_edges_merges += t_edges_merge
        elif len(t_neighbors_prev_large) == 0 and len(t_neighbors_next_large) > 0:
            lr_conn_edges_splits += t_edges_split
        else:
            lr_conn_edges_splits_merges_mixed += (t_edges_merge + t_edges_split)
        a = 1

    lr_conn_merges_to_nodes     = sorted(list(set([t_node[1] for t_node in lr_conn_edges_merges])))
    lr_conn_splits_from_nodes   = sorted(list(set([t_node[0] for t_node in lr_conn_edges_splits])))
    a = 1
#drawH(G, paths, node_positions)
#segment_conn_end_start_points(lr_connections_directed)
#segment_conn_end_start_points(lr_connections_unidirectional)

# ===============================================================================================
# ===============================================================================================
# === extract segment-segment connections that are connected only together (one-to-one; 121) ====
# ===============================================================================================
# REMARK: as discussed before directed connections are part of merges, but in unidirectional
# REMARK: connectios exists an associated (main) connection that makes other a directional
# REMARK: it was a result of performing nearest neighbor refinement.
# REMARK: so non split/merge connection is refinement of unidir connection list with info of merge/split

t_conn_121 = [t_conn for t_conn in lr_connections_unidirectional if t_conn not in t_merge_split_culprit_edges_all]
#aa0  = sorted(   segment_conn_end_start_points(t_conn_121, segment_list = segments2, nodes = 1),   key = lambda x: x[0])


# ===============================================================================================
# separate 121 paths with zero inter length
# ===============================================================================================
# REMARK: these cases occur when segment terminates into split with or merge with a node, not a segment
# REMARK: and 121 separation only concerns itself with segment-segment connecivity
# REMARK: this case might be inside merge/split merge.
t_conn_121_zero_path = []
for t_node in t_conn_121:
    if len(lr_close_segments_simple_paths_inter[t_node]) == 0:
        t_conn_121_zero_path.append(t_node)

t_conn_121 = [t_node for t_node in t_conn_121 if t_node not in t_conn_121_zero_path]

# ===============================================================================================
# reconstruct 121 zero path neighbors
# ===============================================================================================
# REMARK: zero path implies that solo nodes are connected to prev segment end or next segment start
t_conn_121_zero_path_nodes = {t_conn:[] for t_conn in t_conn_121_zero_path}
t_conn_121_zp_contour_combs= {t_conn:{} for t_conn in t_conn_121_zero_path}
for t_conn in t_conn_121_zero_path:
    t_from, t_to    = t_conn
    t_from_node_end = segments2[t_from  ][-1]
    t_to_node_start = segments2[t_to    ][0]
    #t_from_neigbors_next    = extractNeighborsNext(     G, t_from_node_end,  lambda x: x[0])
    #t_to_neigbors_prev      = extractNeighborsPrevious( G, t_to_node_start,  lambda x: x[0])
    t_from_neigbors_next    = list(G.successors(t_from_node_end))
    t_to_neigbors_prev      = list(G.predecessors(t_to_node_start))
    t_conn_121_zero_path_nodes[t_conn] = sorted(set(t_from_neigbors_next+t_to_neigbors_prev),key=lambda x: (x[0], x[1]))
    t_times = sorted(set([t_node[0] for t_node in t_conn_121_zero_path_nodes[t_conn]]))
    t_conn_121_zp_contour_combs[t_conn] = {t_time:[] for t_time in t_times}
    
    for t_time,*t_subIDs in t_conn_121_zero_path_nodes[t_conn]:
        t_conn_121_zp_contour_combs[t_conn][t_time] += t_subIDs

    for t_time in t_conn_121_zp_contour_combs[t_conn]:
         t_conn_121_zp_contour_combs[t_conn][t_time] = sorted(t_conn_121_zp_contour_combs[t_conn][t_time])
a = 1

# ===============================================================================================
# gather segment hulls and centroids
# ===============================================================================================
# REMARK: not doing anything with zero path because it has two or more nodes at one time.
# REMARK: some segments are not in t_all_121_segment_IDs and yet stats are needed later.
# REMARK: they will be appended later, so contents are not strictly 121s
t_all_121_segment_IDs = sorted(list(set(sum([list(a) for a in t_conn_121],[]))))
t_segments_121_centroids    = {tID:{} for tID in range(len(segments2))}
t_segments_121_areas        = {tID:{} for tID in range(len(segments2))}
t_segments_121_mom_z        = {tID:{} for tID in range(len(segments2))}
for tID in t_all_121_segment_IDs:
    t_segment = segments2[tID]
    for t_time,*t_subIDs in t_segment:
        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_node = tuple([t_time] + t_subIDs)
        t_segments_121_centroids[   tID][t_node]   = t_centroid
        t_segments_121_areas[       tID][t_node]   = t_area
        t_segments_121_mom_z[       tID][t_node]   = t_mom_z

            
# ===============================================================================================
# ===== EDIT 10.09.23 lets find joined 121s ========================
# ===============================================================================================
if 1 == 1:
    # codename: big121
    G_seg_view_1 = nx.DiGraph()
    G_seg_view_1.add_edges_from(t_conn_121)

    for g in G_seg_view_1.nodes():
          G_seg_view_1.nodes()[g]["t_start"] = segments2[g][0][0]
          G_seg_view_1.nodes()[g]["t_end"] = segments2[g][-1][0]
    
    # 121s are connected to other segments, if these segments are other 121s, then we can connect 
    # them into bigger one and recover using more data.
    yoo = extract_graph_connected_components(G_seg_view_1.to_undirected(), lambda x: x)


    # big 121s terminate at zero neighbors, real or pseudo merges/splits 
    # both cannot easily tell te difference between real and pseudo. have search for recombination
    # pseudo events means start/end of a big 121 contain only part of bubble and cannot
    # be used to determine history. For the real m/s start/end segments can be used. 
    # how to tell real and pseudo apart? 
    # since pseudo m/s is one bubble:
    # 1) split branches have limited and rather short length before pseudo merge.
    # 2) ! area is recovered after pseudo branch

    # -----------------------------------------------------------------------------------------------
    # PSEUDO MERGE/SPLIT ON EDGES: DETECT BIG 121S WITH M/S
    # -----------------------------------------------------------------------------------------------
    # check left side = part of split
    if 1 == 1:
        t_segment_ID_left   = [t_subIDs[0] for t_subIDs in yoo]
        t_segment_ID_right  = [t_subIDs[-1] for t_subIDs in yoo]
        t_temp_merge    = defaultdict(int)
        t_temp_split    = defaultdict(int)
        t_temp_in_merge = defaultdict(list)
        t_temp_in_split = defaultdict(list)
        # find if left is part of split, and with whom
        for t, t_ID in enumerate(t_segment_ID_left):
            for t_from,t_to in lr_conn_edges_splits:
                if t_to == t_ID:                    # if t_ID is branch of split from t_from
                    t_temp_split[t_ID] = t_from     # save who was mater of left segment-> t_ID:t_from
                    t_temp_in_split[t] = yoo[t]     # index in big 121s: connected components. t_ID in cc.

        # find if right is part of merge, and with whom
        for t, t_ID in enumerate(t_segment_ID_right): # order in big 121s, right segment ID.
            for t_from,t_to in lr_conn_edges_merges:
                if t_from == t_ID: 
                    t_temp_merge[t_ID] = t_to
                    t_temp_in_merge[t] = yoo[t]

        # extract big 121s that are in no need of real/pseudo m/s analysis.
        lr_big121_events_none = defaultdict(list) # big 121s not terminated by splits or merges. ready to resolve.
        # find if elements of big 121s are without m/s or bounded on one or both sides by m/s.
        set1 = set(t_temp_in_merge)
        set2 = set(t_temp_in_split)

        t_only_merge = set1 - set2
        t_only_split = set2 - set1
        t_merge_split = set1.intersection(set2)
        t_merge_split_no = set(range(len(yoo))) - (set1.union(set2))
        assert len(t_merge_split) == 0, "segments bounded on both sides by merge/split. not encountered before, but ez to fix"

        lr_events = {'merge':defaultdict(list), 'split':defaultdict(list)}
    # -----------------------------------------------------------------------------------------------
    # PSEUDO MERGE/SPLIT ON EDGES: DETERMINE IF M/S IS FAKE OR REAL
    # -----------------------------------------------------------------------------------------------

    # test if edge segment is part of real of fake m/s.
    # strip it from big 121s and compare area of prev segment and result of event
    # if area changed drastically then its real m/s, otherwise its same bubble with fake m/s
    # NOTE: currently i take a single end-start nodes. might take more if i test for if they are in m/s
    if 1 == 1:
        t_big121s_merge_conn_fake = []
        t_big121s_split_conn_fake = []

        t_only_merge_fake = []
        t_max_pre_merge_segment_length = 12 # skip if branch is too long. most likely a real branch
        for t in t_only_merge:
            t_big121_subIDs = yoo[t]
            #if t not in t_merge_split:
            t_from      = t_big121_subIDs[-1]   # possible fake merge branch (segment ID).
            t_from_from = t_big121_subIDs[-2]   # segment ID prior to that, most likely OG bubble.
            t_from_to   = t_temp_merge[t_from]  # seg ID of merge.

            if len(segments2[t_from]) >= t_max_pre_merge_segment_length: continue

            t_from_to_node      = segments2[t_from_to   ][0 ]                   # take end-start nodes
            t_from_from_node    = segments2[t_from_from ][-1]

            t_from_to_node_area     = G.nodes[t_from_to_node    ]["area"]       # check their area
            t_from_from_node_area   = G.nodes[t_from_from_node  ]["area"] 
            if np.abs(t_from_to_node_area - t_from_from_node_area) / t_from_to_node_area < 0.35:
                t_only_merge_fake += [t]
                t_big121s_merge_conn_fake.append((t_from,t_from_to))

        t_only_split_fake = []
        for t in t_only_split:
            t_big121_subIDs = yoo[t]
            t_to            = t_big121_subIDs[0]   # possible fake merge branch (segment ID).
            t_to_to         = t_big121_subIDs[1]   # segment ID prior to that, most likely OG bubble.
            t_from_to       = t_temp_split[t_to]  # seg ID of merge.

            if len(segments2[t_to]) >= t_max_pre_merge_segment_length: continue

            t_from_to_node  = segments2[t_from_to   ][-1]                   # take end-start nodes
            t_to_to_node    = segments2[t_to_to     ][0]

            t_from_to_node_area = G.nodes[t_from_to_node]["area"]       # check their area
            t_to_to_node_area   = G.nodes[t_to_to_node  ]["area"] 
            if np.abs(t_from_to_node_area - t_to_to_node_area) / t_from_to_node_area < 0.35:
                t_only_split_fake += [t]
                t_big121s_split_conn_fake.append((t_from_to,t_to))

    # -----------------------------------------------------------------------------------------------
    # PSEUDO MERGE/SPLIT ON EDGES: TRIM BIG 121S AS TO DROP FAKE M/S.
    # CREATE NEW t_conn_121 -> lr_big121s_conn_121
    # -----------------------------------------------------------------------------------------------
    # hopefully fake edges of big 121s are detected, so trim them, if possible and add to future reconstruction.
    if 1 == 1:
        t_big121s_edited = [None]*len(yoo) # by default everything fails. reduced len 2 cc to len 1 stays none = failed.
        for t, t_subIDs in enumerate(yoo):
            if t not in t_only_merge_fake and t not in t_only_split_fake:
                t_big121s_edited[t] = t_subIDs
            elif t in t_only_merge_fake:
                if len(t_subIDs) > 2:
                    t_big121s_edited[t] = t_subIDs[:-1] # drop fake pre-merge branch
            else:
                if len(t_subIDs) > 2:
                    t_big121s_edited[t] = t_subIDs[1:]  # drop fake post-split branch

        # side segments may have been dropped. drop according graph edges.
        lr_big121s_conn_121 = []
        for t_conn in t_conn_121:
            t_from,t_to = t_conn
            for t_subIDs in t_big121s_edited:            
                # check if t_conn fully in connected components
                if t_subIDs != None and t_from in t_subIDs:
                    if t_to in t_subIDs:
                        lr_big121s_conn_121.append(t_conn)
                        break
    # -----------------------------------------------------------------------------------------------
    # BIG 121S INTERPOLATE COMBINED TRAJECTORIES, GIVEN FULL INFO REDISTRIBUTE FOR DATA FOR EACH CONN
    # -----------------------------------------------------------------------------------------------
    # CALCULATE interpolation of holes in data for big 121s
    if 1 == 1:
        lr_big121s_interpolation = defaultdict(dict)
        lr_big121s_interpolation_big = defaultdict(dict) 
        for t,t_subIDs in enumerate(t_big121s_edited):
            if t_subIDs != None:
                t_temp_nodes_all = []
                for t_subID in t_subIDs:
                    t_temp_nodes_all += segments2[t_subID]
                t_temp_times        = [G.nodes[t_node]["time"       ] for t_node in t_temp_nodes_all]
                t_temp_areas        = [G.nodes[t_node]["area"       ] for t_node in t_temp_nodes_all]
                t_temp_centroids    = [G.nodes[t_node]["centroid"   ] for t_node in t_temp_nodes_all]
        
                t_conns_reconstruct = [(x,y) for x,y in zip(t_subIDs[:-1], t_subIDs[1:])]                             # reconstuct t_conns using staggered list approach
                t_conns_times_end_start = [(segments2[x][-1][0], segments2[y][0][0]) for x, y in t_conns_reconstruct] # find end-start times .
                t_conns_times_intermediate = [np.arange(start + 1,end,1) for start,end in t_conns_times_end_start]    # generate intermediate time steps
                t_times_missing_all = np.concatenate(t_conns_times_intermediate)                                      # flatten into one list
        
                a = 1
                # interpolate composite (long) parameters 
                t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, s = 15, debug = 0, aspect = 'equal', title = t_subIDs)
                t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_subIDs)
                # form dict time:centroid for convinience
                t_interpolation_centroids_1 = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_centroids_0)}
                t_interpolation_areas_1     = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_areas_0)}
                # save data with t_conns keys
                t_conns_times_dict          = {x:y for x,y in zip(t_conns_reconstruct,t_conns_times_intermediate)}
                for t_conn,t_times_relevant in t_conns_times_dict.items():

                    t_centroids = [t_centroid for t_time,t_centroid in t_interpolation_centroids_1.items() if t_time in t_times_relevant]
                    t_areas     = [t_area for t_time,t_area in t_interpolation_areas_1.items() if t_time in t_times_relevant]

                    lr_big121s_interpolation[t_conn]['centroids'] = np.array(t_centroids)
                    lr_big121s_interpolation[t_conn]['areas'    ] = t_areas
                    lr_big121s_interpolation[t_conn]['times'    ] = t_times_relevant
                # save whole big 121
                lr_big121s_interpolation_big[tuple(t_subIDs)]['centroids'] = t_interpolation_centroids_1
                lr_big121s_interpolation_big[tuple(t_subIDs)]['areas'    ] = t_interpolation_areas_1

    


    # ===============================================================================================
    # ====  EXTRACT POSSIBLE CONTOUR ELEMENTS FROM PATHS ====
    # ===============================================================================================
    # REMARK: gather all inter-segment nodes and grab slices w.r.t time
    # REMARK: so result is {time_01:[ID1,ID2],time_02:[ID4,ID5],...}
    # REMARK: i expect most of the time solution to be all elements in cluster. except real merges
    if 1 == 1:

        lr_big121s_perms_pre = {}

        for t_conn in lr_big121s_conn_121:
            t_traj = lr_close_segments_simple_paths[t_conn][0]
            t_nodes = {t_time:[] for t_time in [t_node[0] for t_node in t_traj]}
            for t_traj in lr_close_segments_simple_paths[t_conn]:
                for t_time,*t_subIDs in t_traj:
                    t_nodes[t_time] += t_subIDs
            for t_time in t_nodes:
                t_nodes[t_time] = sorted(list(set(t_nodes[t_time])))
            lr_big121s_perms_pre[t_conn] = t_nodes
    
    # ===============================================================================================
    # ====  CONSTRUCT PERMUTATIONS FROM CLUSTER ELEMENT CHOICES ====
    # ===============================================================================================
    # REMARK: grab different combinations of elements in clusters
    #lr_contour_combs_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_contour_combs.items()}
    if 1 == 1:
        lr_big121s_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_big121s_perms_pre.items()}
        for t_conn, t_times_contours in lr_big121s_perms_pre.items():
            for t_time,t_contours in t_times_contours.items():
                t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
                lr_big121s_perms[t_conn][t_time] = t_perms



    # ===============================================================================================
    # ====  PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ====
    # ===============================================================================================
    # REMARK: these will be reused alot in next steps, store them to avoid need of recalculation
    if 1 == 1:    
        lr_big121s_perms_areas      = lr_init_perm_precomputed(lr_big121s_perms,0)
        lr_big121s_perms_centroids  = lr_init_perm_precomputed(lr_big121s_perms,[0,0])
        lr_big121s_perms_mom_z      = lr_init_perm_precomputed(lr_big121s_perms,0)

        #lr_permutation_areas_precomputed        = lr_init_perm_precomputed(lr_contour_combs_perms,0)
        #lr_permutation_centroids_precomputed    = lr_init_perm_precomputed(lr_contour_combs_perms,[0,0])
        #lr_permutation_mom_z_precomputed        = lr_init_perm_precomputed(lr_contour_combs_perms,0)

        for t_conn, t_times_perms in lr_big121s_perms.items():
            for t_time,t_perms in t_times_perms.items():
                for t_perm in t_perms:
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
                    t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                    lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
                    lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
                    lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z

    

        #drawH(G, paths, node_positions)
    # ===============================================================================================
    # ====  CONSTUCT DIFFERENT PATHS FROM UNIQUE CHOICES ====
    # ===============================================================================================
    # REMARK: try different evolutions of inter-segment paths
    if 1 == 1:
        lr_big121s_perms_cases = {t_conn:[] for t_conn in lr_big121s_perms_pre}
        lr_big121s_perms_times = {t_conn:[] for t_conn in lr_big121s_perms_pre}
        #lr_permutation_cases = {t_conn:[] for t_conn in lr_contour_combs}
        #lr_permutation_times = {t_conn:[] for t_conn in lr_contour_combs}
        for t_conn, t_times_perms in lr_big121s_perms.items():
    
            t_values = list(t_times_perms.values())
            t_times = list(t_times_perms.keys())

            sequences = list(itertools.product(*t_values))

            lr_big121s_perms_cases[t_conn] = sequences
            lr_big121s_perms_times[t_conn] = t_times
            #lr_permutation_cases[t_conn] = sequences
            #lr_permutation_times[t_conn] = t_times

    # ===============================================================================================
    # ==== EVALUATE HULL CENTROIDS AND AREAS FOR EACH EVOLUTION, FIND CASES WITH LEAST DEVIATIONS====
    # ===============================================================================================
    # REMARK: evolutions with least path length and are changes should be right ones
    if 1 == 1:
        t_temp_centroids = {t_conn:t_dict['centroids'] for t_conn,t_dict in lr_big121s_interpolation.items()}
        t_args = [lr_big121s_perms_cases,t_temp_centroids,lr_big121s_perms_times,
                lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

        t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)

        t_weights   = [1,1.5,0,1]
        t_sols      = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
        lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_weights, t_sols, lr_big121s_perms_cases )



    # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    C0 = nx.Graph()
    C0.add_edges_from(list(lr_big121s_perms_cases.keys()))
    
    lr_C0_condensed_connections = extract_graph_connected_components(C0, lambda x: x)

    # lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
    t_condensed_connections_all_nodes = sorted(sum(lr_C0_condensed_connections,[])) # neext next
    lr_C0_condensed_connections_relations = {tID: tID for tID in range(len(segments2))} #t_condensed_connections_all_nodes
    for t_subIDs in lr_C0_condensed_connections:
        for t_subID in t_subIDs:
            lr_C0_condensed_connections_relations[t_subID] = min(t_subIDs)

    if  1 == 1:
        t_segments_new = segments2.copy()
        for t_conn in lr_big121s_perms_cases:
            t_from, t_to = t_conn                                     
            t_sol   = lr_weighted_solutions_max[t_conn]               # pick evolution index that won (most likely)
            t_path  = lr_big121s_perms_cases[t_conn][t_sol]             # t_path contains start-end points of segments !!!
            t_times = lr_big121s_perms_times[t_conn]                    # get inter-segment times
            t_nodes_new = []
            for t_time,t_comb in zip(t_times,t_path):                 # create composite nodes
                #for tID in t_comb:
                #    t_nodes_old.append(tuple([t_time, tID]))          # old type of nodes in solution: (time,contourID)     e.g (t1,ID1)
                t_nodes_new.append(tuple([t_time] + list(t_comb)))    # new type of nodes in solution: (time,*clusterIDs)   e.g (t1,ID1,ID2,...)

            t_nodes_all = []
            for t_time,t_comb in lr_big121s_perms_pre[t_conn].items():
                for tID in t_comb:
                    t_nodes_all.append(tuple([t_time, tID]))

            # basically in order to correctly reconnectd sides of inter-segment i have to also delete old edges from side nodes
            # this can be done by deleting them and adding back
            t_nodes_new_sides = [segments2[t_from][-2]] + t_nodes_new + [segments2[t_to][1]]
            # find element which will be deleted, but will be readded later
            t_common_nodes = set(t_nodes_all).intersection(set(t_nodes_new_sides))
            # store their parameters
            t_node_params = {t:dict(G.nodes[t]) for t in t_common_nodes}
            
            G.remove_nodes_from(t_nodes_all)                          # edges will be also dropped
            

            t_pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
    
            G.add_edges_from(t_pairs)
            # restore deleted parameters
            for t,t_params in t_node_params.items():
                G.add_node(t, **t_params)
            # determine t_from masters index, send that segment intermediate nodes and second segment
            # if t_from is its own master, it still works, check it with simple ((0,1),(1,2)) and {0:0,1:0,2:0}
            t_from_new = lr_C0_condensed_connections_relations[t_from]
            t_nodes_intermediate = list(sorted(t_nodes_new, key = lambda x: x[0]))[1:-1]
            t_segments_new[t_from_new] += t_nodes_intermediate
            t_segments_new[t_from_new] += segments2[t_to]

    
            # fill centroid, area and momement of innertia zz missing for intermediate segment
            for t_time,*t_subIDs in t_nodes_intermediate:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
                t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
                t_node = tuple([t_time] + t_subIDs)
                set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
                
                t_node = tuple([t_time] + t_subIDs)
                G.nodes[t_node]["time"]     = int(t_time)
                G.nodes[t_node]["area"]     = t_area
                G.nodes[t_node]["centroid"] = t_centroid
                G.nodes[t_node]["moment_z"] = t_mom_z
    
            # copy data from inherited
    
            combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
            # wipe data if t_from is inherited
            if t_from_new != t_from:
                set_ith_elements_multilist(t_from, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
            # wipe data from t_to anyway
            set_ith_elements_multilist(t_to, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)

        # resort data in case of broken order. i think its easier to sort keys and reconstruct with sorted keys 121 data.
        for t in t_condensed_connections_all_nodes:
            t_segments_new[t]               = list(sorted(t_segments_new[t], key = lambda x: x[0]))
            t_sorted_keys                   = sorted(list(t_segments_121_centroids[t].keys()), key = lambda x: x[0])
            t_segments_121_centroids[   t]  = {tID: t_segments_121_centroids[   t][tID] for tID in t_sorted_keys}
            t_segments_121_areas[       t]  = {tID: t_segments_121_areas[       t][tID] for tID in t_sorted_keys}
            t_segments_121_mom_z[       t]  = {tID: t_segments_121_mom_z[       t][tID] for tID in t_sorted_keys}

    # at this time some segments got condenced into big 121s. right connection might be changed.
    lr_time_active_segments = {t:[] for t in lr_time_active_segments}
    for k,t_segment in enumerate(t_segments_new):
        t_times = [a[0] for a in t_segment]
        for t in t_times:
            lr_time_active_segments[t].append(k)

    # -----------------------------------------------------------------------------------------------
    # prep permutations for merges/splits fake and real
    # -----------------------------------------------------------------------------------------------
    if 1 == 1:
        # -----------------------------------------------------------------------------------------------
        # ------ PREPARE DATA INTO USABLE FORM ----
        # -----------------------------------------------------------------------------------------------
        t_merge_fake_to_ID = unique_sort_list([t_conn[1] for t_conn in t_big121s_merge_conn_fake])                                          # grab master node e.g [10]
        t_split_fake_from_ID = unique_sort_list([t_conn[0] for t_conn in t_big121s_split_conn_fake])                                        # e.g [15]

        t_merge_real_to_ID  = [t_ID for t_ID in set([t_conn[1] for t_conn in lr_conn_edges_merges]) if t_ID not in t_merge_fake_to_ID]      # e.g [13,23]
        t_split_real_from_ID  = [t_ID for t_ID in set([t_conn[0] for t_conn in lr_conn_edges_splits]) if t_ID not in t_split_fake_from_ID]    # e.g []

        t_merge_fake_conns  = [t_conn for t_conn in lr_conn_edges_merges if t_conn[1] in t_merge_fake_to_ID]    # grab all edges with that node. e.g [(8, 10), (9, 10)]
        t_split_fake_conns  = [t_conn for t_conn in lr_conn_edges_splits if t_conn[0] in t_split_fake_from_ID]

        t_merge_real_conns  = [t_conn for t_conn in lr_conn_edges_merges if t_conn not in t_merge_fake_conns]   # extract real from rest of conns. e.g [(15, 16), (15, 17)]
        t_split_real_conns  = [t_conn for t_conn in lr_conn_edges_splits if t_conn not in t_split_fake_conns]
    
        t_merge_fake_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_merges if t_conn[1] == t_ID] for t_ID in t_merge_fake_to_ID}       # e.g {10:[(8, 10), (9, 10)]}
        t_split_fake_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_splits if t_conn[0] == t_ID] for t_ID in t_split_fake_from_ID}     # e.g {15:[(15, 16), (15, 17)]} 

        t_merge_real_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_merges if t_conn[1] == t_ID] for t_ID in t_merge_real_to_ID}
        t_split_real_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_splits if t_conn[0] == t_ID] for t_ID in t_split_real_from_ID}

        # ----------------------------------------------------------------------------------------------
        # ------ GENERATE INFORMATON ABOUT REAL AND FAKE MERGE/SPLIT EVENTS----
        # ----------------------------------------------------------------------------------------------
        # IDEA: for real events gather possible choices for contours for each time step during merge event (t_perms)
        # IDEA: for fake events gather evolutions of contours which integrate fake branches and stray nodes (t_combs)
        # WORKFLOW: take master event ID- either node into which merge happens or node from which spit happens
        # WORKFLOW: real: find branches, gather nodes between branches and master, find all options for time steps
        # WORKFLOW: fake: determine branches and their parents, gather options that go though branch parent- master path
        # NOTE: everything is held in one dictionary, but keys and contents are different
        t_event_start_end_times = {'merge':{},'split':{}}

        func_prev_neighb = lambda x: G2.nodes[x]["t_end"]
        func_next_neighb = lambda x: G2.nodes[x]["t_start"]

         # ------------------------------------- REAL MERGE/SPLIT -------------------------------------
        for t_ID in t_merge_real_to_ID:
            t_predecessors   = extractNeighborsPrevious(G2, t_ID, func_prev_neighb) 
            t_predecessors_times = {t:G2.nodes[t]["t_end"] for t in t_predecessors}
            t_end = G2.nodes[t_ID]["t_start"]
            t_times         = {(t,t_ID):np.arange(t_start, t_end + 1, 1) for t,t_start in t_predecessors_times.items()}
            t_event_start_end_times['merge'][t_ID] = {'t_start':        t_predecessors_times,
                                                      'branches':       t_predecessors,
                                                      't_end':          t_end,
                                                      't_times':        t_times,
                                                      't_perms':        {}}

            # pre-isolate graph segment where all sub-events take place
            t_time_min = min(t_predecessors_times.values())
            t_nodes_keep    = [node for node in G.nodes() if t_time_min <= node[0] <= t_end] # < excluded edges
            t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
            # working with each branch separately
            for t_from,t_to in t_times:
                t_ref_ID = lr_C0_condensed_connections_relations[t_from]
                t_segments_keep = [t_ref_ID, t_to]      # new inherited ID
                t_start = t_predecessors_times[t_from]  # storage uses old ID
                
                if t_end - t_start > 1:
                    t_sol = graph_sub_isolate_connected_components(t_subgraph, t_start, t_end, lr_time_active_segments,
                                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1]) 
                    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                    t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]
                else: t_values = None
                t_event_start_end_times['merge'][t_ID]['t_perms'][(t_from,t_to)] = t_values


        for t_ID in t_split_real_from_ID:
            assert 1 == 1, "never ran t_split_real_from_ID because of lack of data."
            t_successors   = extractNeighborsNext(G2, t_ID, func_next_neighb) 
            t_successors_times = {t:G2.nodes[t]["t_start"] for t in t_successors}
            t_start = G2.nodes[t_ID]["t_end"]
            t_times         = {(t_ID,t):np.arange(t_start, t_end + 1, 1) for t,t_end in t_successors_times.items()}
            t_event_start_end_times['split'][t_ID] = {'t_start' :       t_start,
                                                      'branches':       t_successors,
                                                      't_end'   :       t_successors_times,
                                                      't_times' :       t_times,
                                                      't_perms':        {}}

            # pre-isolate graph segment where all sub-events take place
            t_time_max = max(t_predecessors_times.values())
            t_nodes_keep    = [node for node in G.nodes() if t_start < node[0] < t_time_max]
            t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
            # working with each branch separately
            for t_from,t_to in t_times:
                t_ref_ID = lr_C0_condensed_connections_relations[t_from]
                t_segments_keep = [t_ref_ID, t_to] 
                t_end = t_successors_times[t_from]
                if t_end - t_start > 1:
                    t_sol = graph_sub_isolate_connected_components(t_subgraph, t_start, t_end, lr_time_active_segments,
                                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1])
                    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            
                    t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]
                else: t_values = None
                t_event_start_end_times['split'][t_ID]['t_perms'][(t_from,t_to)] = t_values

        # ------------------------------------- FAKE MERGE/SPLIT -------------------------------------
        for t_ID in t_merge_fake_to_ID:
            t_predecessors   = extractNeighborsPrevious(G2, t_ID, func_prev_neighb) # lambda takes neighbors with lower end time than t_ID. 
            t_pre_predecessors = []
            for t_ID_pre in t_predecessors:
                t_pre_predecessors += extractNeighborsPrevious(G2, t_ID_pre, func_prev_neighb)
            # if fake branches should terminate and only 1 OG should be left. check if this is the case
            assert len(t_pre_predecessors) == 1, "fake branch/es consists of multiple segments, not yet encountered"
            t_ID_pre_pred = t_pre_predecessors[0]
            t_start         = G2.nodes[t_ID_pre_pred]["t_end"]
            t_end           = G2.nodes[t_ID]["t_start"]
            t_times         = {(t_ID_pre_pred, t_ID):np.arange(t_start, t_end + 1, 1)}
            t_event_start_end_times['merge'][t_ID] = {'t_start':        t_start,
                                                      'pre_predecessor':t_ID_pre_pred,
                                                      'branches':       t_predecessors,
                                                      't_end':          t_end,
                                                      't_times' :       t_times}


            t_segments_keep = [t_ID_pre_pred] + t_predecessors + [t_ID]
            
            t_ref_ID = lr_C0_condensed_connections_relations[t_ID_pre_pred]

            t_sol = graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments,
                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1])
            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            seqs = perms_with_branches(t_predecessors,t_segments_new,t_perms) 
            t_event_start_end_times['merge'][t_ID]['t_combs'] = seqs
            a = 1



        for t_ID in t_split_fake_from_ID:
            t_successors    = extractNeighborsNext(G2, t_ID, func_next_neighb) 
            t_post_successors = []
            for t_ID_pre in t_successors:
                t_post_successors += extractNeighborsNext(G2, t_ID_pre, func_next_neighb)
            assert len(t_post_successors), "fake branch/es consists of multiple segments, not yet encountered"
            t_ID_post_succ  = t_post_successors[0]
            t_start         = G2.nodes[t_ID]["t_end"]
            t_end           = G2.nodes[t_ID_post_succ]["t_start"]
            t_times         = {(t_ID, t_ID_post_succ):np.arange(t_start, t_end + 1, 1)}
            t_event_start_end_times['split'][t_ID] = {'t_start':        t_start,
                                                      'post_successor': t_ID_post_succ,
                                                      'branches':       t_successors,
                                                      't_end':          t_end,
                                                      't_times':       t_times}

            t_ref_ID = lr_C0_condensed_connections_relations[t_ID]
            t_segments_keep = [t_ID] + t_successors + [t_ID_post_succ]
            t_segments_keep = [lr_C0_condensed_connections_relations[t] for t in t_segments_keep]
            t_sol = graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments,
                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1])
            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            
            seqs = perms_with_branches(t_successors,t_segments_new,t_perms) 

            t_event_start_end_times['split'][t_ID]['t_combs'] = seqs

        a = 1


    # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
# ===============================================================================================
# ====  Determine interpolation parameters ====
# ===============================================================================================
# we are interested in segments that are not part of psuedo branches (because they will be integrated into path)
# pre merge segments will be extended, post split branches also, but backwards. 
# for pseudo event can take an average paramets for prev and post segments.
t_fake_branches_IDs = []
for t_ID in t_merge_fake_to_ID:
    t_fake_branches_IDs += t_event_start_end_times['merge'][t_ID]['branches']
for t_ID in t_split_fake_from_ID:
    t_fake_branches_IDs += t_event_start_end_times['split'][t_ID]['branches']
# get segments that have possibly inherited other segments and that are not branches
t_segments_IDs_relevant = [t_ID for t_ID,t_traj in enumerate(t_segments_new) if len(t_traj)>0 and t_ID not in t_fake_branches_IDs]

h_interp_len_max = 8          # i want this long history at max
h_interp_len_min = 3      # 
h_start_point_index = 0   # default start from index 0
h_analyze_p_max = 20   # at best analyze this many points, it requires total traj of len h_analyze_p_max + "num pts for interp"
h_analyze_p_min = 5    # 
k_all = (1,2)
s_all = (0,1,5,10,25,50,100,1000,10000)
t_k_s_combs = list(itertools.product(k_all, s_all))

t_segment_k_s       = defaultdict(tuple)
t_segment_k_s_diffs = defaultdict(dict)
for t_ID in t_segments_IDs_relevant:
    trajectory          = np.array([G.nodes[t]["centroid"   ] for t in t_segments_new[t_ID]])
    time                = np.array([G.nodes[t]["time"       ] for t in t_segments_new[t_ID]])
    t_do_k_s_anal = False
    
    if  trajectory.shape[0] > h_analyze_p_max + h_interp_len_max:   # history is large

        h_start_point_index2 = trajectory.shape[0] - h_analyze_p_max - h_interp_len_max

        h_interp_len_max2 = h_interp_len_max
        
        t_do_k_s_anal = True

    elif trajectory.shape[0] > h_analyze_p_min + h_interp_len_min:  # history is smaller, give prio to interp length rather inspected number count
                                                                    # traj [0,1,2,3,4,5], min_p = 2 -> 4 & 5, inter_min_len = 3
        h_start_point_index2 = 0                                    # interp segments: [2,3,4 & 5], [1,2,3 & 4]
                                                                    # but have 1 elem of hist to spare: [1,2,3,4 & 5], [0,1,2,3 & 4]
        h_interp_len_max2 = trajectory.shape[0]  - h_analyze_p_min  # so inter_min_len = len(traj) - min_p = 6 - 2 = 4
        
        t_do_k_s_anal = True

    else:
        h_interp_len_max2 = trajectory.shape[0]                     # traj is very small

    if t_do_k_s_anal:                                               # test differet k and s combinations and gather inerpolation history.
        t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, t_k_s_combs, h_interp_len_max2, h_start_point_index2, debug = 0, debug_show_num_best = 2)
        t_segment_k_s_diffs[t_ID] = errors_sol_diff_norms_all[t_comb_sol]
        t_segment_k_s[      t_ID] = t_comb_sol
    else:                                                           
        t_segment_k_s[t_ID]         = None
        t_segment_k_s_diffs[t_ID]   = None

# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ===============================================================================================
# ====  deal with fake event recovery ====
# ===============================================================================================
# determine segment start and end, find k_s, store info
def get_k_s(t_from_k_s, t_to_k_s, backup = (1,5)):
    if t_from_k_s is None:
        if t_to_k_s is not None:
            return t_to_k_s
        else:
            return backup
    elif t_to_k_s is None:
        return t_from_k_s
    else:
        if t_from_k_s[0] == t_to_k_s[0]:
            return (t_from_k_s[0], min(t_from_k_s[1], t_to_k_s[1]))
        else:
            return min(t_from_k_s, t_to_k_s, key=lambda x: x[0])

t_fake_events_k_s_edge_master = {}
for t_to in t_merge_fake_to_ID:
    t_from =  t_event_start_end_times['merge'][t_to]['pre_predecessor']
    t_to    = lr_C0_condensed_connections_relations[t_to]
    t_from  = lr_C0_condensed_connections_relations[t_from]
    t_from_k_s  = t_segment_k_s[t_from  ]
    t_to_k_s    = t_segment_k_s[t_to    ]
    t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))   # determine which k_s to inherit (lower)
    t_fake_events_k_s_edge_master[t_to] = {'state': 'merge', 'edge': (t_from, t_to), 'k_s':t_k_s_out}
        
for t_from_old in t_split_fake_from_ID:
    t_to = t_event_start_end_times['split'][t_from_old]['post_successor']
    t_to    = lr_C0_condensed_connections_relations[t_to]
    t_from  = lr_C0_condensed_connections_relations[t_from_old]
    t_from_k_s  = t_segment_k_s[t_from  ]
    t_to_k_s    = t_segment_k_s[t_to    ]
    t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))  
    t_fake_events_k_s_edge_master[t_from_old] = {'state': 'split', 'edge': (t_from, t_to), 'k_s':t_k_s_out}

# interpolate
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
a = 1
for t_ID,t_param_dict in t_fake_events_k_s_edge_master.items():
    t_state         = t_param_dict['state'  ] 
    t_from, t_to    = t_param_dict['edge'   ]
    k,s             = t_param_dict['k_s'    ]
    t_subIDs = [t_from, t_to]
    t_start = t_event_start_end_times[t_state][t_ID]['t_start'  ]
    t_end   = t_event_start_end_times[t_state][t_ID]['t_end'    ]
    if 1 == 1:
        t_temp_nodes_all = []
        for t_subID in t_subIDs:
            t_temp_nodes_all += t_segments_new[t_subID]
        t_temp_times        = [G.nodes[t_node]["time"       ] for t_node in t_temp_nodes_all]
        t_temp_areas        = [G.nodes[t_node]["area"       ] for t_node in t_temp_nodes_all]
        t_temp_centroids    = [G.nodes[t_node]["centroid"   ] for t_node in t_temp_nodes_all]
        
        #t_conns_reconstruct = [(x,y) for x,y in zip(t_subIDs[:-1], t_subIDs[1:])]                             # reconstuct t_conns using staggered list approach
        #t_conns_times_end_start = [(segments2[x][-1][0], segments2[y][0][0]) for x, y in t_conns_reconstruct] # find end-start times .
        #t_conns_times_intermediate = [np.arange(start + 1,end,1) for start,end in t_conns_times_end_start]    # generate intermediate time steps
        t_times_missing_all = np.arange(t_start + 1, t_end, 1)
        
        a = 1
        # interpolate composite (long) parameters 
        t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, s = s, k = k, debug = 0, aspect = 'equal', title = t_subIDs)
        t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_subIDs)
        # form dict time:centroid for convinience
        #t_interpolation_centroids_1 = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_centroids_0)}
        #t_interpolation_areas_1     = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_areas_0)}
        # save data with t_conns keys
        #t_conns_times_dict          = {x:y for x,y in zip(t_conns_reconstruct,t_conns_times_intermediate)}
        #for t_conn,t_times_relevant in t_conns_times_dict.items():

        #t_centroids = [t_centroid for t_time,t_centroid in t_interpolation_centroids_1.items() if t_time in t_times_relevant]
        #t_areas     = [t_area for t_time,t_area in t_interpolation_areas_1.items() if t_time in t_times_relevant]
        t_conn = (t_from, t_to)
        lr_big121s_interpolation[t_conn]['centroids'] = np.array(t_interpolation_centroids_0)
        lr_big121s_interpolation[t_conn]['areas'    ] = np.array(t_interpolation_areas_0    )   
        lr_big121s_interpolation[t_conn]['times'    ] = t_times_missing_all
        

a = 1
G_seg_view_2 = nx.DiGraph()
lr_reindex_masters(lr_C0_condensed_connections_relations, t_conn_121, remove_solo_ID = 1)

#t_conn_temp = [(t_from,t_to) for t_to,t_from in lr_C0_condensed_connections_relations.items() if t_from != t_to and len(t_segments_new[t_to]) > 0]
#G_seg_view_2.add_edges_from(t_conn_121)

#for g in G_seg_view_2.nodes():
#        G_seg_view_2.nodes()[g]["t_start"] = segments2[g][0][0]
#        G_seg_view_2.nodes()[g]["t_end"] = segments2[g][-1][0]

if 1 == -1:
    
    # take confirmed merge starting segment and determine k and s values for better extrapolation

    for t_conn in lr_conn_edges_merges: # [(4,13)] lr_conn_edges_merges segment_conn_end_start_points(lr_conn_edges_merges, nodes = 1)
        t_from, t_to        = t_conn
        t_from_time_end = t_segments_new[t_from][-1][0]
        t_to_time_start = t_segments_new[t_to][0][0]
        if t_to_time_start - t_from_time_end == 1:  # nothing to interpolate?
            t_extrapolate_sol_comb[t_conn] = {}     # although not relevant here, i want to store t_conn for next code
            continue
        trajectory          = np.array(list(t_centroids_all[t_from].values()))
        time                = np.array(list(t_centroids_all[t_from].keys()))
        t_do_k_s_anal = False
    
        if  trajectory.shape[0] > h_analyze_p_max + h_interp_len_max:   # history is large

            h_start_point_index2 = trajectory.shape[0] - h_analyze_p_max - h_interp_len_max

            h_interp_len_max2 = h_interp_len_max
        
            t_do_k_s_anal = True

        elif trajectory.shape[0] > h_analyze_p_min + h_interp_len_min:  # history is smaller, give pro to interp length rather inspected number count
                                                                        # traj [0,1,2,3,4,5], min_p = 2 -> 4 & 5, inter_min_len = 3
            h_start_point_index2 = 0                                    # interp segments: [2,3,4 & 5], [1,2,3 & 4]
                                                                        # but have 1 elem of hist to spare: [1,2,3,4 & 5], [0,1,2,3 & 4]
            h_interp_len_max2 = trajectory.shape[0]  - h_analyze_p_min  # so inter_min_len = len(traj) - min_p = 6 - 2 = 4
        
            t_do_k_s_anal = True

        else:
            h_interp_len_max2 = trajectory.shape[0]                     # traj is very small

        if t_do_k_s_anal:                                               # test differet k and s combinations and gather inerpolation history.
            t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, t_k_s_combs, h_interp_len_max2, h_start_point_index2, debug = 0, debug_show_num_best = 2)
            t_k,t_s = t_comb_sol
        else:                                                           # use this with set k = 1, s = 5 to generate some interp history.
            t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, [(1,5)], 2, 0, debug = 0, debug_show_num_best = 2)
            t_k,t_s = t_comb_sol
            errors_sol_diff_norms_all[t_comb_sol] = {-1:5}              # dont have history to compare later..
# ===============================================================================================
# = 2X LARGE Segments: extract one-to-one segment connections of large segments <=> high trust ==
# ===============================================================================================
# lets extract segment-segment (one2one) connections that are both of large length
# most likely a pseudo split-merge (p_split-merge). large length => most likely solo bubble
lr_segment_len_large = 7
lr_conn_one_to_one_large = []
for t_from,t_to in t_conn_121:
    t_segment_length_from   = lr_segments_lengths[t_from]
    t_segment_length_to     = lr_segments_lengths[t_to]
    if t_segment_length_from >= lr_segment_len_large and t_segment_length_to >= lr_segment_len_large:
        lr_conn_one_to_one_large.append(tuple([t_from,t_to]))
lr_conn_one_to_one_large_test = [(segments2[start][-1][0],segments2[end][0][0]) for start,end in lr_conn_one_to_one_large]


# ===============================================================================================
# ======== SHORT SEGMENTS BUT ISOLATED: check if one2one (121) connection is isolated: ==========
# ===============================================================================================
# REMARK: low trust short segment connects to other segment w/o merge/split.
# REMARK: filter such cases where inter-segment space is isolatded- does not branch out ("not hairy xD")
# REMARK: isolated connection is only connected to its own inter-segment path nodes
# REMARK: and not branching out 1 step further. pretty strict condition
# REMARK: idk if this check is needed now. must inspect t_conn_121_other_isolated_not array for fails. <<<<
# take inter-segment simple paths, extract nodes; check neighbors of nodes;
# isolate nonpath nodes; see what isleft. if path had a closed boundary, leftover will be zero.

if 1 == 1:
    lr_conn_one_to_one_other = [t_conn for t_conn in t_conn_121 if t_conn not in lr_conn_one_to_one_large]
    lr_conn_one_to_one_other_test = [(segments2[start][-1][0],segments2[end][0][0]) for start,end in lr_conn_one_to_one_other]

    t_conn_121_other_isolated_not = []
    t_conn_121_other_isolated = []
    for t_from,t_to in lr_conn_one_to_one_other:

        t_from_node_last   = segments2[t_from][-1]
        t_to_node_first      = segments2[t_to][0]

        t_from_node_last_time   = t_from_node_last[0]
        t_to_node_first_time    = t_to_node_first[0]
        t_from_to_max_time_steps= t_to_node_first_time - t_from_node_last_time + 1
        t_from_to_paths_nodes_all_inter = lr_close_segments_simple_paths_inter[   tuple([t_from,t_to])]
        t_all_path_neighbors = []
        for t_node in t_from_to_paths_nodes_all_inter:
            t_all_path_neighbors.append(list(G.successors(t_node))+ list(G.predecessors(t_node)))
        t_all_path_neighbors_node_all = sorted(set(sum(t_all_path_neighbors,[])),key=lambda x: x[0])
        t_nides_not_in_main_path = [t_node for t_node in t_all_path_neighbors_node_all if t_node not in t_from_to_paths_nodes_all_inter + [t_from_node_last,t_to_node_first]]
        if len(t_nides_not_in_main_path):   t_conn_121_other_isolated_not.append(tuple([t_from,t_to]))
        else:                               t_conn_121_other_isolated.append(tuple([t_from,t_to]))

    t_conn_121_other_isolated_not_test  = sorted(   segment_conn_end_start_points(t_conn_121_other_isolated_not,    segment_list = segments2, nodes = 1),   key = lambda x: x[0])
    t_conn_121_other_isolated_test      = sorted(   segment_conn_end_start_points(t_conn_121_other_isolated,        segment_list = segments2, nodes = 1),   key = lambda x: x[0])

#drawH(G, paths, node_positions)
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# =============== CHECK IF SEGMENTS ARE NOT RELATED TO SPLIT MERGE PREV/POST ====================
# ===============================================================================================
# REMARK: example as shows schematically below outlines a problem where pseudo split-merge is not recovered
# REMARK:      /3b\   /5b--6b\
# REMARK: 1--2      4          7--8 
# REMARK:      \3a/    5a--6a/
# REMARK: [1,2]->X->[4,5b,6b] is correct 121 event. although its clear that (4,5a) edge is missing
# REMARK: 
lr_conn_121_other_terminated            = []      # not sandwiched between split/merge
lr_conn_121_other_terminated_inspect    = []      # only one is split/merge
lr_conn_121_other_terminated_failed     = []      # both split/merge. idk what to do here yet.

if 1==1:
    # methodology- find left and right neighbors of connected segments
    # see if neighbors are related to only segments of interest. if multiple- fail
    for t_conn in t_conn_121_other_isolated:
        # segments of interest
        t_from, t_to = t_conn
        # reference start/end times
        t_from_start    = G2.nodes[t_from   ]["t_start"]
        t_to_end        = G2.nodes[t_to     ]["t_end"]
        # segment neighbors
        t_from_neighbors_prev   = [tID for tID in list(G2.neighbors(t_from))    if G2.nodes[tID]["t_end"    ] < t_from_start]
        t_to_neighbors_next     = [tID for tID in list(G2.neighbors(t_to))      if G2.nodes[tID]["t_start"  ] > t_to_end    ]

        t_conn_back = [tuple(sorted([tID,t_from]))  for tID in t_from_neighbors_prev]
        t_conn_forw = [tuple(sorted([t_to,tID]))    for tID in t_to_neighbors_next  ]
        # see if neighbor connection is in list of splits/merges.
        t_prev_was_split    = True if len([t for t in t_conn_back if t in lr_conn_edges_splits]) > 0 else False
        t_next_is_merge     = True if len([t for t in t_conn_forw if t in lr_conn_edges_merges]) > 0 else False

        if not t_prev_was_split and not t_next_is_merge:
            lr_conn_121_other_terminated.append(t_conn)         # if not split and not merge
        elif ((t_prev_was_split and not t_prev_was_split) or  (not t_prev_was_split and t_next_is_merge)):
            lr_conn_121_other_terminated_inspect.append(t_conn) # if only one is split/merge
        else: 
            lr_conn_121_other_terminated_failed.append(t_conn)  # both are splits/merges
    
            # maybe deal with lr_conn_121_other_terminated_inspect here, or during analysis
    a = 1
#drawH(G, paths, node_positions)
# ===============================================================================================
# TEST connections started/terminated by split/merge for area conservation
# consider merge event for segments LEFT->RIGHT->MERGE. True merge branch will conserve area between LEFT-RIGHT
# FAKE merge will split area and then recover original area after merge
# REAL event, on other hand, will preserve area from LEFT->RIGHT. so its easier to analyze. (or can do both?!)
# ===============================================================================================
if 1 == 1:
    for t_conn in lr_conn_121_other_terminated_inspect:
        t_from,t_to = t_conn
        t_from_nodes_part   = segments2[t_from][-4:] 
        t_to_nodes_part = segments2[t_to][:4]
        t_from_areas    = np.array([G.nodes[t_node]["area"] for t_node in t_from_nodes_part])
        t_to_areas      = np.array([G.nodes[t_node]["area"] for t_node in t_to_nodes_part])
        t_from_areas_mean = np.mean(t_from_areas)
        t_to_areas_mean = np.mean(t_to_areas)
        if np.abs(t_to_areas_mean-t_from_areas_mean)/t_from_areas_mean < 0.3:
            lr_conn_121_other_terminated.append(t_conn)
            lr_conn_121_other_terminated_inspect = [t_test for t_test in lr_conn_121_other_terminated_inspect if t_test != t_conn]
        a = 1
        

# ===============================================================================================
# ===============================================================================================
# ============ TEST INTERMERDIATE SEGMENTS FOR LARGE AND CONFIRMED SHORT 121s ===================
# ===============================================================================================
# ===============================================================================================
lr_relevant_conns = lr_conn_one_to_one_large + lr_conn_121_other_terminated
lr_121_interpolation_times      = {t_conn:[] for t_conn in lr_relevant_conns} # missing time steps between segments
lr_121_interpolation_centroids  = {t_conn:[] for t_conn in lr_relevant_conns}
lr_121_interpolation_areas      = {t_conn:[] for t_conn in lr_relevant_conns}
lr_121_interpolation_moment_z   = {t_conn:[] for t_conn in lr_relevant_conns}

if 1 == 1:
    
    # ===============================================================================================
    # ============ interpolate trajectories between segments for further tests ===================
    # ===============================================================================================
    
    # REMARK: we know segments but dont know stuff inbetween. surely we can interpolate middle.
    
    # generate missing time steps
    for t_conn in lr_relevant_conns:
        t_from,t_to     = t_conn
        t_time_prev     = segments2[t_from][-1][0]
        t_time_next     = segments2[t_to][0][0]
        lr_121_interpolation_times[t_conn] = np.arange(t_time_prev+1,t_time_next, 1)

    # prepare history around  inter-segment and interpoalte
    for t_conn in lr_relevant_conns:
        t_from, t_to = t_conn
        histLen = 5
        # prep prev/next history of nodes. it will take at max histLen times, at least 2 (min possible)
        t_hist_prev     = segments2[t_from][-histLen:]
        t_hist_next     = segments2[t_to][:histLen]
        # extract times from history
        t_times_prev    = [t_node[0] for t_node in t_hist_prev]
        t_times_next    = [t_node[0] for t_node in t_hist_next]
        # gather prev/next centroid history
        t_traj_prev_c     = np.array([t_segments_121_centroids[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next_c     = np.array([t_segments_121_centroids[  t_to][t_node] for t_node in t_hist_next])
        # gather prev/next areas history
        t_traj_prev_a     = np.array([t_segments_121_areas[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next_a     = np.array([t_segments_121_areas[  t_to][t_node] for t_node in t_hist_next])
        # gather prev/next moment z history
        t_traj_prev_mz     = np.array([t_segments_121_mom_z[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next_mz     = np.array([t_segments_121_mom_z[  t_to][t_node] for t_node in t_hist_next])
        # take missing time steps
        t_interp_times  = lr_121_interpolation_times[t_conn]
        # interpolate missing values of centroids/area/momZ
        lr_121_interpolation_centroids[t_conn]  = interpolateMiddle2D(t_times_prev,t_times_next,t_traj_prev_c, t_traj_next_c,
                                                                      t_interp_times, s = 15, debug = 0,
                                                                      aspect = 'equal', title = t_conn)

        lr_121_interpolation_areas[t_conn]      = interpolateMiddle1D(t_times_prev,t_times_next,t_traj_prev_a, t_traj_next_a,
                                                                     t_interp_times,  rescale = True,
                                                                     s = 15, debug = 0, aspect = 'auto', title = t_conn)

        lr_121_interpolation_moment_z[t_conn]   = interpolateMiddle1D(t_times_prev,t_times_next,t_traj_prev_mz, t_traj_next_mz,
                                                                     t_interp_times,  rescale = True,
                                                                     s = 15, debug = 0, aspect = 'auto', title = t_conn)
        a = 1
    #for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # ====  EXTRACT POSSIBLE CONTOUR ELEMENTS FROM PATHS ====
    # ===============================================================================================
    # REMARK: gather all inter-segment nodes and grab slices w.r.t time
    # REMARK: so result is {time_01:[ID1,ID2],time_02:[ID4,ID5],...}
    # REMARK: i expect most of the time solution to be all elements in cluster. except real merges

    lr_contour_combs                = {tID:{} for tID in lr_relevant_conns}
    for t_conn in lr_relevant_conns:
        t_traj = lr_close_segments_simple_paths[t_conn][0]
        #t_min, t_max = t_traj[0][0], t_traj[-1][0]
        #t_nodes = {t:[] for t in np.arange(t_min, t_max + 1, 1)}
        t_nodes = {t_time:[] for t_time in [t_node[0] for t_node in t_traj]}
        for t_traj in lr_close_segments_simple_paths[t_conn]:
            for t_time,*t_subIDs in t_traj:
                t_nodes[t_time] += t_subIDs
        for t_time in t_nodes:
            t_nodes[t_time] = sorted(list(set(t_nodes[t_time])))
        lr_contour_combs[t_conn] = t_nodes
    
    a = 1
    # ===============================================================================================
    # ====  CONSTRUCT PERMUTATIONS FROM CLUSTER ELEMENT CHOICES ====
    # ===============================================================================================
    # REMARK: grab different combinations of elements in clusters
    lr_contour_combs_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_contour_combs.items()}
    for t_conn, t_times_contours in lr_contour_combs.items():
        for t_time,t_contours in t_times_contours.items():
            t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
            lr_contour_combs_perms[t_conn][t_time] = t_perms

    # ===============================================================================================
    # ====  PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ====
    # ===============================================================================================
    # REMARK: these will be reused alot in next steps, store them to avoid need of recalculation
    

    lr_permutation_areas_precomputed        = lr_init_perm_precomputed(lr_contour_combs_perms,0)
    lr_permutation_centroids_precomputed    = lr_init_perm_precomputed(lr_contour_combs_perms,[0,0])
    lr_permutation_mom_z_precomputed        = lr_init_perm_precomputed(lr_contour_combs_perms,0)

    for t_conn, t_times_perms in lr_contour_combs_perms.items():
        for t_time,t_perms in t_times_perms.items():
            for t_perm in t_perms:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
                t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                lr_permutation_areas_precomputed[t_conn][t_time][t_perm] = t_area
                lr_permutation_centroids_precomputed[t_conn][t_time][t_perm] = t_centroid
                lr_permutation_mom_z_precomputed[t_conn][t_time][t_perm] = t_mom_z

    #drawH(G, paths, node_positions)
    # ===============================================================================================
    # ====  CONSTUCT DIFFERENT PATHS FROM UNIQUE CHOICES ====
    # ===============================================================================================
    # REMARK: try different evolutions of inter-segment paths
    a = 1
    lr_permutation_cases = {t_conn:[] for t_conn in lr_contour_combs}
    lr_permutation_times = {t_conn:[] for t_conn in lr_contour_combs}
    for t_conn, t_times_perms in lr_contour_combs_perms.items():
    
        t_values = list(t_times_perms.values())
        t_times = list(t_times_perms.keys())

        sequences = list(itertools.product(*t_values))

        lr_permutation_cases[t_conn] = sequences
        lr_permutation_times[t_conn] = t_times

    # ===============================================================================================
    # ==== EVALUATE HULL CENTROIDS AND AREAS FOR EACH EVOLUTION, FIND CASES WITH LEAST DEVIATIONS====
    # ===============================================================================================
    # REMARK: evolutions with least path length and are changes should be right ones

    if 1 == 1:
        

        a = 1

        t_args = [lr_permutation_cases,lr_121_interpolation_centroids,lr_permutation_times,
              lr_permutation_centroids_precomputed,lr_permutation_areas_precomputed,lr_permutation_mom_z_precomputed]

        t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)
    # ===============================================================================================
    # ========== CALCULATED WEIGHTED SOLUTONS =========
    # ===============================================================================================
    # REMARK: give methods (centroid, area,..) different weights and calculate total weights
    # REMARK: of each evolution occurence ID. Then re-eight solution to 0-1.

        
    t_weights = [1,1.5,0,1]
    t_sols = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
    lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_weights, t_sols, lr_permutation_cases )

# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ========== INTEGRATE RESOLVED PATHS INTO GRAPH; REMOVE SECONDARY SOLUTIONS =========
# ===============================================================================================
# REMARK: refactor nodes from solo objects to clusters. remove all previous nodes and replace with new.


 
# ===============
# 1) say lr_relevant_conns -> lr_permutation_cases contain segments that are trusted and self-contained, dont need extra history
# but others might benefit from extra history.
# 2) each resolved t_conn means both segments can be merged into one. but chain of connections can be resolved
# this chain should be condensed into one segment and possibly used to help resolve other, harder cases.
# ===============

C1 = nx.Graph()
C1.add_edges_from(list(lr_permutation_cases.keys()))
    
lr_C1_condensed_connections = extract_graph_connected_components(C1, lambda x: x)

# lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
t_condensed_connections_all_nodes = sorted(sum(lr_C1_condensed_connections,[])) # neext next
lr_C1_condensed_connections_relations = {tID: tID for tID in range(len(segments2))} #t_condensed_connections_all_nodes
for t_subIDs in lr_C1_condensed_connections:
    for t_subID in t_subIDs:
        lr_C1_condensed_connections_relations[t_subID] = min(t_subIDs)

# lr_C1_condensed_connections_relations is such that when given segment ID it provides its master ID, and master points at itself

t_segments_new = segments2.copy()
for t_conn in lr_permutation_cases:
    t_sol   = lr_weighted_solutions_max[t_conn]               # pick evolution index that won (most likely)
    t_path  = lr_permutation_cases[t_conn][t_sol]             # t_path contains start-end points of segments !!!
    t_times = lr_permutation_times[t_conn]                    # get inter-segment times
    #t_nodes_old = []
    t_nodes_new = []
    for t_time,t_comb in zip(t_times,t_path):                 # create composite nodes
        #for tID in t_comb:
        #    t_nodes_old.append(tuple([t_time, tID]))          # old type of nodes in solution: (time,contourID)     e.g (t1,ID1)
        t_nodes_new.append(tuple([t_time] + list(t_comb)))    # new type of nodes in solution: (time,*clusterIDs)   e.g (t1,ID1,ID2,...)

    t_nodes_all = []
    for t_time,t_comb in lr_contour_combs[t_conn].items():
        for tID in t_comb:
            t_nodes_all.append(tuple([t_time, tID]))
    
    # > integrating new paths is not that simple (something to do with prior edges being kept)
    # its easy to remove start-end point nodes, but they will lose connection to segments
    G.remove_nodes_from(t_nodes_all)                          # edges will be also dropped
    # so add extra nodes to make edges with segments, which will create start-end points again.
    t_from, t_to = t_conn                                     
    t_nodes_new_sides = [segments2[t_from][-2]] + t_nodes_new + [segments2[t_to][1]]

    t_pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
    
    G.add_edges_from(t_pairs)

    # determine t_from masters index, send that segment intermediate nodes and second segment
    # if t_from is its own master, it still works, check it with simple ((0,1),(1,2)) and {0:0,1:0,2:0}
    t_from_new = lr_C1_condensed_connections_relations[t_from]
    t_nodes_intermediate = list(sorted(t_nodes_new, key = lambda x: x[0]))[1:-1]
    t_segments_new[t_from_new] += t_nodes_intermediate
    t_segments_new[t_from_new] += segments2[t_to]

    
    # fill centroid, area and momement of innertia zz missing for intermediate segment
    for t_time,*t_subIDs in t_nodes_intermediate:
        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_node = tuple([t_time] + t_subIDs)
        set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    
    # copy data from inherited
    
    combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    # wipe data if t_from is inherited
    if t_from_new != t_from:
        set_ith_elements_multilist(t_from, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
    # wipe data from t_to anyway
    set_ith_elements_multilist(t_to, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)

# resort data in case of broken order. i think its easier to sort keys and reconstruct with sorted keys 121 data.
for t in t_condensed_connections_all_nodes:
    t_segments_new[t]               = list(sorted(t_segments_new[t], key = lambda x: x[0]))
    t_sorted_keys                   = sorted(list(t_segments_121_centroids[t].keys()), key = lambda x: x[0])
    t_segments_121_centroids[   t]  = {tID: t_segments_121_centroids[   t][tID] for tID in t_sorted_keys}
    t_segments_121_areas[       t]  = {tID: t_segments_121_areas[       t][tID] for tID in t_sorted_keys}
    t_segments_121_mom_z[       t]  = {tID: t_segments_121_mom_z[       t][tID] for tID in t_sorted_keys}

# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ========== DEAL WITH ZERO PATH 121s >> SKIP IT FOR NOW, IDK WHAT TO DO << =========
# ===============================================================================================
# REMARK: zero paths might be part of branches, which means segment relations should be refactored

# plan:
# 1) take  start segment, end segment IDs, get history if possible
# since its zero inter path, t_conn_121_zp_contour_combs contains last seg end and next seg start
# combs have same two times as zero length inter segment. even in worst case scenario there is 1 point history
# 2) take history on both sidez and interpolate middle
# 3) check fitness of combinations
# 4) pick best, done!
t_conn_121_zero_path0 = t_conn_121_zp_contour_combs.copy()
t_conn_121_zero_path    = lr_reindex_masters(lr_C1_condensed_connections_relations, t_conn_121_zero_path)
t_conn_121_zp_contour_combs = {lr_reindex_masters(lr_C1_condensed_connections_relations, t):t_vals for t,t_vals in t_conn_121_zp_contour_combs.items()}
for t_conn in t_conn_121_zero_path:
    t_from, t_to = t_conn
    t_traj_prev_c   = t_segments_new[t_from][-4:]
    t_traj_next_c     = t_segments_new[t_to][:4]
    # zero path has only step with multiple nodes, where we have to choose correct
    t_interp_times  = [t for t, t_vals in t_conn_121_zp_contour_combs[t_conn].items() if len(t_vals) > 1]
    t_traj_nodes_prev   = [t_node for t_node in t_traj_prev_c if t_node[0] not in t_interp_times]
    t_traj_nodes_next   = [t_node for t_node in t_traj_next_c if t_node[0] not in t_interp_times]
    
    # ugh! hate to recalculate centroids, but idk where and if ive done it before!!!
    t_traj_prev_c = []
    for t_node in t_traj_nodes_prev:
        t_time, tID = t_node
        t_hull  = cv2.convexHull(g0_contours[t_time][tID])
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_traj_prev_c.append(t_centroid)

    t_traj_next_c = []
    for t_node in t_traj_nodes_next:
        t_time, tID = t_node
        t_hull  = cv2.convexHull(g0_contours[t_time][tID])
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_traj_next_c.append(t_centroid)

    t_times_prev    = [t_node[0] for t_node in t_traj_nodes_prev]
    t_times_next    = [t_node[0] for t_node in t_traj_nodes_next]
    t_interp        = interpolateMiddle2D(t_times_prev,t_times_next,t_traj_prev_c, t_traj_next_c,
                                                                  t_interp_times, s = 15, debug = 0,
                                                                  aspect = 'equal', title = t_conn)
    t_time = t_interp_times[0]
    t_combs = t_conn_121_zp_contour_combs[t_conn][t_time]
    t_perms = sum([list(itertools.combinations(t_combs, r)) for r in range(1,len(t_combs)+1)],[])
    t_choices_c = []
    for t_perm in t_perms:
        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_choices_c.append(t_centroid)
    t_choices_c = np.array(t_choices_c)
    t_diffs = t_choices_c - t_interp[0]
    t_norms = np.linalg.norm(t_diffs, axis = 1)
    t_sol_ID = np.argmin(t_norms)
    t_sol = t_perms[t_sol_ID]

    t_nodes_all = []
    for tID in t_combs:
            t_nodes_all.append(tuple([t_time, tID]))
    
    # remove all single nodes from new cluster node
    G.remove_nodes_from(t_nodes_all)
    t_from_node_last    = t_traj_nodes_prev[-1]
    t_to_node_first     = t_traj_nodes_next[0]
    t_node_new          = tuple([t_time]+list(t_sol))
    t_conns_new = [(t_from_node_last,t_node_new),(t_node_new,t_to_node_first)]
    G.add_edges_from(t_conns_new)

    
    t_from_new = lr_C1_condensed_connections_relations[t_from]
    t_segments_new[t_from_new]  = [t_node for t_node in t_segments_new[t_from_new   ] if t_node not in t_nodes_all]
    t_segments_new[t_to]        = [t_node for t_node in t_segments_new[t_to         ] if t_node not in t_nodes_all]

    t_nodes_intermediate = [t_node_new]
    
    t_segments_new[t_from_new] += t_nodes_intermediate
    t_segments_new[t_from_new] += t_segments_new[t_to]

    # zero paths were not included because end points were uncertain
    if len( t_segments_121_centroids[t_from_new]) == 0:
        t_nodes_intermediate = t_segments_new[t_from_new] + t_nodes_intermediate

    if len( t_segments_121_centroids[t_to]) == 0:
        t_nodes_intermediate = t_nodes_intermediate + t_segments_new[t_to] 


    # fill centroid, area and momement of innertia zz missing for intermediate segment
    for t_time,*t_subIDs in t_nodes_intermediate:
        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_node = tuple([t_time] + t_subIDs)
        set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
        #t_segments_121_mom_z[    t_from_new][t_node]   = t_mom_z
    
    # copy data from inherited
    combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    # wipe data if t_from is inherited
    if t_from_new != t_from:
        set_ith_elements_multilist(t_from, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
    # wipe data from t_to anyway
    set_ith_elements_multilist(t_to, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
    
    a = 1


# relations will be changed here
#lr_C2_condensed_connections_relations = lr_C1_condensed_connections_relations.copy()
C2 = C1.copy()
C2.add_edges_from(t_conn_121_zero_path)
    
lr_C2_condensed_connections = extract_graph_connected_components(C2, lambda x: x)

# lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
t_condensed_connections_all_nodes = sorted(sum(lr_C2_condensed_connections,[])) # neext next
lr_C2_condensed_connections_relations = {tID: tID for tID in range(len(segments2))} #t_condensed_connections_all_nodes
for t_subIDs in lr_C2_condensed_connections:
    for t_subID in t_subIDs:
        lr_C2_condensed_connections_relations[t_subID] = min(t_subIDs)


# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ========= deal with 121s that are partly connected with merging/splitting segments << =========
# ===============================================================================================
# REMARK: 121s in 'lr_conn_121_other_terminated_inspect' are derived from split branch
# REMARK: or result in a merge branch. split->121 might be actually split->merge, same with 
# REMARK: 121->merge. these are ussualy short-lived, so i might combine them and test


# !!!!!!!!!!!!!!!!!!!!!!!!!!I DID NOT TEST MASTER RELATIONS FOR THIS CODE !!!!!!!!!!!!!!!!!!!!!!!



lr_conn_121_other_terminated_inspect    = lr_reindex_masters(lr_C2_condensed_connections_relations, lr_conn_121_other_terminated_inspect)
lr_conn_121_other_terminated_failed     = lr_reindex_masters(lr_C2_condensed_connections_relations, lr_conn_121_other_terminated_inspect)
lr_conn_edges_splits                    = lr_reindex_masters(lr_C2_condensed_connections_relations, lr_conn_edges_splits)
lr_conn_edges_merges                    = lr_reindex_masters(lr_C2_condensed_connections_relations, lr_conn_edges_merges)

#segment_conn_end_start_points(lr_conn_121_other_terminated_inspect, nodes = 1)
lr_inspect_contour_combs                = {t_conn:{} for t_conn in lr_conn_121_other_terminated_inspect}
lr_inspect_c_c_from_to_interp_times     = {t_conn:[] for t_conn in lr_conn_121_other_terminated_inspect}
lr_inspect_121_interpolation_times      = {t_conn:[] for t_conn in lr_conn_121_other_terminated_inspect}
lr_inspect_121_interpolation_from_to    = {t_conn:[] for t_conn in lr_conn_121_other_terminated_inspect}
lr_inspect_121_to_merge_possible_IDs    = {}
lr_inspect_121_to_merge_resolved_IDs    = {}
lr_inspect_121_to_merge_extra_branches  = {}
for t_conn in lr_conn_121_other_terminated_inspect:      # check conn sandwitched by split/merge
    t_from,t_to = t_conn
    spl = [(t_from_from, t_from_to) for t_from_from, t_from_to in lr_conn_edges_splits if t_from == t_from_to] # 121 left   is some other conn right
    mrg = [(t_to_from, t_to_to) for t_to_from, t_to_to in lr_conn_edges_merges if t_to == t_to_from]           # 121 right  is some other conn left

    # test on merge, no spl data yet
    # structure of mrg segment connections is LEFT->RIGHT->MERGE, with parallel branches to RIGHT (t_from->t_to->t_to_merging_IDs)
    if len(mrg)>0:                                       # right segment is one of merge branch
        t_to_merging_IDs = list(set([t[1] for t in mrg]))# and it merges into this
        assert len(t_to_merging_IDs) == 1, "lr_conn_121_other_terminated_inspect merge with multiple, this should not trigger"
        # what are other merge branches besides t_to
        t_to_all_merging_segments = [t_to_from for t_to_from, t_to_to in lr_conn_edges_merges if t_to_to == t_to_merging_IDs[0] and t_to_from != t_to]
        # extract piece of left history
        t_from_nodes_part = t_segments_new[t_from][-4:] # consider with next line, read comment.
        t_from_time_start = t_from_nodes_part[0][0]     # [-4:][0] = take at furthest from back of at least size 4. list([a,b])[-4:][0] = a
        # extract piece of post merge history
        t_to_from_nodes_part = t_segments_new[t_to_merging_IDs[0]][:4]
        t_to_from_time_start = t_to_from_nodes_part[-1][0]

        lr_inspect_c_c_from_to_interp_times[t_conn].append(t_from_nodes_part)
        lr_inspect_c_c_from_to_interp_times[t_conn].append(t_to_from_nodes_part)
        # > lets figure out whole interval between left and merge (including left and other branches)
        # include start-end of neighbor segments similar like before with 121s
        t_internal_interp_time_start    = t_from_nodes_part[-1][0] 
        t_internal_interp_time_end      = t_to_from_nodes_part[0][0] 
        lr_inspect_121_interpolation_times[t_conn] = np.arange(t_internal_interp_time_start, t_internal_interp_time_end + 1, 1)
        lr_inspect_121_interpolation_from_to[t_conn] = [t_from, t_to_merging_IDs[0]]
        # > in order to retrieve all choices isolate relevant nodes within active time period
        activeNodes = [node for node in G.nodes() if t_from_time_start <= node[0] <= t_to_from_time_start]
        subgraph = G.subgraph(activeNodes)
        # extract connected branches
        connected_components_unique = extract_graph_connected_components(subgraph.to_undirected(), sort_function = lambda x: (x[0], x[1]))
        # pick branch with known node
        sol = [t_cc for t_cc in connected_components_unique if t_segments_new[t_from][-1] in t_cc]
        assert len(sol) == 1, "lr_conn_121_other_terminated_inspect inspect path relates to multiple clusters, dont expect it ever to occur"

        # > gather all possible choices for each time step
        t_times = np.arange(t_from_time_start,t_to_from_time_start + 1, 1)
        t_inspect_contour_combs = {t_time:[] for t_time in t_times}
    
        for t_time,*t_subIDs in sol[0]:
            t_inspect_contour_combs[t_time] += t_subIDs

        for t_time in t_inspect_contour_combs:
            t_inspect_contour_combs[t_time] = sorted(t_inspect_contour_combs[t_time])
        
        lr_inspect_contour_combs[t_conn] = t_inspect_contour_combs

        lr_inspect_121_to_merge_possible_IDs[t_conn] = t_to_merging_IDs
        lr_inspect_121_to_merge_extra_branches[t_conn] = t_to_all_merging_segments
    if len(spl) >0:
        assert 1 == -1, "lr_conn_121_other_terminated_inspect case split unexplored" 
    

# prepare possible permutations of clusters
lr_inspect_contour_combs_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_inspect_contour_combs.items()}
for t_conn, t_times_contours in lr_inspect_contour_combs.items():
    for t_time,t_contours in t_times_contours.items():
        t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
        lr_inspect_contour_combs_perms[t_conn][t_time] = t_perms
# predefine storage
lr_inspect_permutation_areas_precomputed        = lr_init_perm_precomputed(lr_inspect_contour_combs_perms, 0    )
lr_inspect_permutation_centroids_precomputed    = lr_init_perm_precomputed(lr_inspect_contour_combs_perms, [0,0])
lr_inspect_permutation_mom_z_precomputed        = lr_init_perm_precomputed(lr_inspect_contour_combs_perms, 0    )
# calculate parameters of each combination
for t_conn, t_times_perms in lr_inspect_contour_combs_perms.items():
    for t_time,t_perms in t_times_perms.items():
        for t_perm in t_perms:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

            lr_inspect_permutation_areas_precomputed[       t_conn][t_time][t_perm]     = t_area
            lr_inspect_permutation_centroids_precomputed[   t_conn][t_time][t_perm]     = t_centroid
            lr_inspect_permutation_mom_z_precomputed[       t_conn][t_time][t_perm]     = t_mom_z

# interpolate middle

lr_inspect_121_interpolation_centroids  = {t_conn:[] for t_conn in lr_inspect_c_c_from_to_interp_times}
lr_inspect_121_interpolation_areas      = {t_conn:[] for t_conn in lr_inspect_c_c_from_to_interp_times}
lr_inspect_121_interpolation_moment_z   = {t_conn:[] for t_conn in lr_inspect_c_c_from_to_interp_times}

# similar to OG
for t_conn, [t_hist_prev,t_hist_next] in lr_inspect_c_c_from_to_interp_times.items():
    t_from, t_to    = lr_inspect_121_interpolation_from_to[t_conn]
    # tID might not be in 121s, which was an ideal case. rough fix is to add missing.
    for tID in lr_inspect_121_interpolation_from_to[t_conn]:
        if len(t_segments_121_centroids[tID]) == 0:
            for t_time,*t_subIDs in t_segments_new[tID]:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
                t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
                t_node = tuple([t_time] + t_subIDs)
                t_segments_121_centroids[   tID][t_node]   = t_centroid
                t_segments_121_areas[       tID][t_node]   = t_area
                t_segments_121_mom_z[       tID][t_node]   = t_mom_z

    t_times_prev    = [t_node[0] for t_node in t_hist_prev]
    t_times_next    = [t_node[0] for t_node in t_hist_next]

    t_traj_prev_c     = np.array([t_segments_121_centroids[t_from][t_node] for t_node in t_hist_prev])
    t_traj_next_c     = np.array([t_segments_121_centroids[  t_to][t_node] for t_node in t_hist_next])

    t_traj_prev_a     = np.array([t_segments_121_areas[t_from][t_node] for t_node in t_hist_prev])
    t_traj_next_a     = np.array([t_segments_121_areas[  t_to][t_node] for t_node in t_hist_next])

    t_traj_prev_mz     = np.array([t_segments_121_mom_z[t_from][t_node] for t_node in t_hist_prev])
    t_traj_next_mz     = np.array([t_segments_121_mom_z[  t_to][t_node] for t_node in t_hist_next])

    t_interp_times  = lr_inspect_121_interpolation_times[t_conn]

    debug = 0
    lr_inspect_121_interpolation_centroids[t_conn]  = interpolateMiddle2D(t_times_prev,t_times_next,t_traj_prev_c, t_traj_next_c,
                                                                  t_interp_times, s = 15, debug = debug,
                                                                  aspect = 'equal', title = t_conn)

    lr_inspect_121_interpolation_areas[t_conn]      = interpolateMiddle1D(t_times_prev,t_times_next,t_traj_prev_a, t_traj_next_a,
                                                                 t_interp_times,  rescale = True, s = 15, debug = debug,
                                                                 aspect = 'auto', title = t_conn)

    lr_inspect_121_interpolation_moment_z[t_conn]   = interpolateMiddle1D(t_times_prev,t_times_next,t_traj_prev_mz, t_traj_next_mz,
                                                                 t_interp_times,  rescale = True, s = 15, debug = debug,
                                                                 aspect = 'auto', title = t_conn)

    a = 1

lr_inspect_permutation_cases = {t_conn:[] for t_conn in lr_inspect_contour_combs_perms}
lr_inspect_permutation_times = {t_conn:[] for t_conn in lr_inspect_contour_combs_perms}
lr_inspect_121_interpolation_times
for t_conn, t_times_perms in lr_inspect_contour_combs_perms.items():
    
    t_values = list([t_val for t_time,t_val in t_times_perms.items() if t_time in lr_inspect_121_interpolation_times[t_conn]])
    t_times = lr_inspect_121_interpolation_times[t_conn]

    sequences = list(itertools.product(*t_values))

    lr_inspect_permutation_cases[t_conn] = sequences
    lr_inspect_permutation_times[t_conn] = t_times

# have to edit centroids, since they are generated bit differently- chop edges. so i can use lr_evel_perm_interp_data()
t_temp_c = {tID:vals[1:-1] for tID, vals in lr_inspect_121_interpolation_centroids.items()}
t_args = [lr_inspect_permutation_cases,t_temp_c,lr_inspect_permutation_times,
          lr_inspect_permutation_centroids_precomputed,lr_inspect_permutation_areas_precomputed,lr_inspect_permutation_mom_z_precomputed]


t_inspect_sols_c, t_inspect_sols_c_i, t_inspect_sols_a, t_inspect_sols_m = lr_evel_perm_interp_data(*t_args) # dropped areas since its almost same as momZ ??? check pls



t_weights = [1,2,0,1]
t_sols = [t_inspect_sols_c, t_inspect_sols_c_i, t_inspect_sols_a, t_inspect_sols_m]
lr_inspect_weighted_solutions_max, lr_inspect_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_weights, t_sols, lr_inspect_permutation_cases )


# heres small detail, from-to from inspect part is a fake connection, its connected to a split or merge
# so real connection is in lr_inspect_121_interpolation_from_to
C3 = C2.copy()
t_temp = [tuple(lr_inspect_121_interpolation_from_to[t]) for t in lr_inspect_permutation_cases]
C3.add_edges_from(t_temp)
    
lr_C3_condensed_connections = extract_graph_connected_components(C3, lambda x: x)
# lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
t_condensed_connections_all_nodes = sorted(sum(lr_C3_condensed_connections,[])) # neext next
lr_C3_condensed_connections_relations = {tID: tID for tID in range(len(t_segments_new))} #t_condensed_connections_all_nodes
for t_subIDs in lr_C3_condensed_connections:
    for t_subID in t_subIDs:
        lr_C3_condensed_connections_relations[t_subID] = min(t_subIDs)

# dont forget redirect fake connection to OG. branches a bit later on saving
for (t_from, t_to_old),(_,t_to) in lr_inspect_121_interpolation_from_to.items():
    lr_C3_condensed_connections_relations[t_to_old] = t_from

for t_conn in lr_inspect_permutation_cases:
    t_sol   = lr_inspect_weighted_solutions_max[t_conn]
    t_path  = lr_inspect_permutation_cases[t_conn][t_sol]             # t_path contains start-end points of segments !!!
    t_times = lr_inspect_permutation_times[t_conn]
    #t_nodes_old = []
    t_nodes_new = []
    for t_time,t_comb in zip(t_times,t_path):
        #for tID in t_comb:
        #    t_nodes_old.append(tuple([t_time, tID]))          # old type of nodes in solution: (time,contourID)     e.g (t1,ID1)
        t_nodes_new.append(tuple([t_time] + list(t_comb)))    # new type of nodes in solution: (time,*clusterIDs)   e.g (t1,ID1,ID2,...)

    t_nodes_all = []
    for t_time,t_comb in {tID:vals for tID,vals in lr_inspect_contour_combs[t_conn].items() if tID in lr_inspect_121_interpolation_times[t_conn]}.items():
        for tID in t_comb:
            t_nodes_all.append(tuple([t_time, tID]))
    
    # its easy to remove start-end point nodes, but they will lose connection to segments
    G.remove_nodes_from(t_nodes_all)
    #[lr_conn_edges_merges.remove(t_edge) for t_edge in 
    # so add extra nodes to make edges with segments, which will create start-end points again.
    t_from, t_to_fake = t_conn

    t_from, t_to = lr_inspect_121_interpolation_from_to[t_conn]    
    t_to_merge_branches = lr_inspect_121_to_merge_extra_branches[t_conn] 

    t_nodes_new_sides = [t_segments_new[t_from][-2]] + t_nodes_new + [t_segments_new[t_to][1]]

    pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
    
    G.add_edges_from(pairs)
    # confirm stored possible = resolved
    lr_inspect_121_to_merge_resolved_IDs[t_conn] = lr_inspect_121_to_merge_possible_IDs[t_conn]

    # =========== have to modify it from OG because its merge/split case, so extra branches
    # determine t_from masters index, send that segment intermediate nodes and second segment
    # if t_from is its own master, it still works, check it with simple ((0,1),(1,2)) and {0:0,1:0,2:0}
    t_from_new = lr_C3_condensed_connections_relations[t_from]
    t_nodes_intermediate = list(sorted(t_nodes_new, key = lambda x: x[0]))[1:-1]
    t_segments_new[t_from_new] += t_nodes_intermediate
    t_segments_new[t_from_new] += t_segments_new[t_to] # edited from OG

    
    # fill centroid, area and momement of innertia zz missing for intermediate segment
    for t_time,*t_subIDs in t_nodes_intermediate:
        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_node = tuple([t_time] + t_subIDs)
        set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
        
    # copy data from inherited
    combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    # wipe data if t_from is inherited
    if t_from_new != t_from:
        set_ith_elements_multilist(t_from, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
    # wipe data from t_to anyway
    set_ith_elements_multilist(t_to, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
    # wipe data from t_to_fake anyway
    set_ith_elements_multilist(t_to_fake, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
    # remove branches: added 25.08.23 because was missing, during fin_ parts of code (final parts, segment extension form free nodes)
    for t_ID_branch in t_to_merge_branches:
        t_segment_branch = copy.deepcopy(t_segments_new[t_ID_branch])
        t_segment_branch_del = [t_node for t_node in t_segment_branch if t_node not in t_nodes_all]
        if len(t_segment_branch_del) == 0:
            print(f"{t_conn} -> {t_to} reconstruction completely purged branch {t_ID_branch}")
        else:
            print(f"{t_conn} -> {t_to} reconstruction partially purged branch {t_ID_branch}, nodes {t_segment_branch_del} are remaining.")
        set_ith_elements_multilist(t_ID_branch, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
        lr_C3_condensed_connections_relations[t_ID_branch] = t_from

a = 1
t_hulls_all = [{time:0 for time,*subIDs in case} for case in t_segments_new]
t_centroids_all = [{time:np.zeros(2, int) for time,*subIDs in case} for case in t_segments_new]
t_centroids_all_redo_IDs = [tID for tID in range(len(t_segments_new)) if tID not in t_condensed_connections_all_nodes]
for tID in tqdm(t_centroids_all_redo_IDs):
    case = t_segments_new[tID]
    for t_k,(t_time,*subIDs) in enumerate(case):
        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t_subID] for t_subID in subIDs]))
        t_hulls_all[tID][t_time] = t_hull
        t_centroid = centroid_area(t_hull)[0]
        t_centroids_all[tID][t_time] = t_centroid

for tID in t_condensed_connections_all_nodes:
    t_centroids_all[tID] = {t_time:t_vals  for (t_time,*t_subIDs), t_vals in t_segments_121_centroids[tID].items()}


if 1 == -1:

    if 1 == 1:
        binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
        imgs = [convertGray2RGB(binarizedMaskArr[k].copy()) for k in range(binarizedMaskArr.shape[0])]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7; thickness = 4;
        for n, case in tqdm(enumerate(t_segments_new)):
            case    = sorted(case, key=lambda x: x[0])
            for k,subCase in enumerate(case):
                t_time,*subIDs = subCase
                for subID in subIDs:
                    cv2.drawContours(  imgs[t_time],   g0_contours[t_time], subID, cyclicColor(n), 2)
                #x,y,w,h = lessRoughBRs[time][subCase]
            
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t_subID] for t_subID in subIDs]))
                x,y,w,h = cv2.boundingRect(t_hull)
            
                cv2.drawContours(  imgs[t_time],  [t_hull], -1, cyclicColor(n), 2)
                #x,y,w,h = g0_bigBoundingRect[time][ID]
                #cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
                [cv2.putText(imgs[t_time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
            for k,subCase in enumerate(case):
                t_time,*subIDs = subCase
                useTimes = [t for t in t_centroids_all[n].keys() if t <= t_time and t > t_time - 10]#
                pts = np.array([t_centroids_all[n][t] for t in useTimes]).reshape(-1, 1, 2)
            
                cv2.polylines(imgs[t_time], [pts] ,0, (255,255,255), 3)
                cv2.polylines(imgs[t_time], [pts] ,0, cyclicColor(n), 2)
                [cv2.circle(imgs[t_time], tuple(p), 3, cyclicColor(n), -1) for [p] in pts]

                for subID in subIDs:
                    startPos2 = g0_contours[t_time][subID][-30][0] 
                    [cv2.putText(imgs[t_time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        
    if 1 == 1:
        #cv2.imshow('a',imgs[time])
        for k,img in enumerate(imgs):
            if k in activeTimes:
                folder = r"./post_tests/testImgs2/"
                fileName = f"{str(k).zfill(4)}.png"
                cv2.imwrite(os.path.join(folder,fileName) ,img)
                #cv2.imshow('a',img)




lr_resolved_one_sided_merges = sum(list(lr_inspect_121_to_merge_resolved_IDs.values()),[])
lr_conn_edges_merges = [t_conn for t_conn in lr_conn_edges_merges if t_conn[1] not in lr_resolved_one_sided_merges]
lr_conn_merges_to_nodes = [tID for tID in lr_conn_merges_to_nodes if tID not in lr_resolved_one_sided_merges]
lr_conn_merges_to_nodes = [lr_C3_condensed_connections_relations[tID] for tID in lr_conn_merges_to_nodes]
lr_conn_edges_merges                    = lr_reindex_masters(lr_C3_condensed_connections_relations, lr_conn_edges_merges)


# ===============================================================================================
# ========= PROCESS MERGES: RESOLVE PRE-MERGE AS CLOSE AS POSSIBLE =========
# ===============================================================================================
# REMARK: merge begins too soon because bounding rectangles begin to overlap
# REMARK: for merges each branch should be resolved forward from pool of combinations
# REMARK: to improve extrapolation ive decided to configure smoothing parameters 
# REMARK: iteratevly based on ability to predict each next step.
# REMARK: overal best prediction parameter wins
print("#========= PROCESS MERGES: RESOLVE PRE-MERGE AS CLOSE AS POSSIBLE =========")

# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# remember which segment indicies are active at each time step
lr_time_active_segments = {t:[] for t in lr_time_active_segments}
for k,t_segment in enumerate(t_segments_new):
    t_times = [a[0] for a in t_segment]
    for t in t_times:
        lr_time_active_segments[t].append(k)


h_interp_len_max = 8          # i want this long history at max
h_interp_len_min = 3      # 
h_start_point_index = 0   # default start from index 0
h_analyze_p_max = 20   # at best analyze this many points, it requires total traj of len h_analyze_p_max + "num pts for interp"
h_analyze_p_min = 5    # 
k_all = (1,2)
s_all = (0,1,5,10,25,50,100,1000,10000)
t_k_s_combs = list(itertools.product(k_all, s_all))
t_extrapolate_sol = {}
t_extrapolate_sol_comb = {}
# take confirmed merge starting segment and determine k and s values for better extrapolation

for t_conn in lr_conn_edges_merges: # [(4,13)] lr_conn_edges_merges segment_conn_end_start_points(lr_conn_edges_merges, nodes = 1)
    t_from, t_to        = t_conn
    t_from_time_end = t_segments_new[t_from][-1][0]
    t_to_time_start = t_segments_new[t_to][0][0]
    if t_to_time_start - t_from_time_end == 1:  # nothing to interpolate?
        t_extrapolate_sol_comb[t_conn] = {}     # although not relevant here, i want to store t_conn for next code
        continue
    trajectory          = np.array(list(t_centroids_all[t_from].values()))
    time                = np.array(list(t_centroids_all[t_from].keys()))
    t_do_k_s_anal = False
    
    if  trajectory.shape[0] > h_analyze_p_max + h_interp_len_max:   # history is large

        h_start_point_index2 = trajectory.shape[0] - h_analyze_p_max - h_interp_len_max

        h_interp_len_max2 = h_interp_len_max
        
        t_do_k_s_anal = True

    elif trajectory.shape[0] > h_analyze_p_min + h_interp_len_min:  # history is smaller, give pro to interp length rather inspected number count
                                                                    # traj [0,1,2,3,4,5], min_p = 2 -> 4 & 5, inter_min_len = 3
        h_start_point_index2 = 0                                    # interp segments: [2,3,4 & 5], [1,2,3 & 4]
                                                                    # but have 1 elem of hist to spare: [1,2,3,4 & 5], [0,1,2,3 & 4]
        h_interp_len_max2 = trajectory.shape[0]  - h_analyze_p_min  # so inter_min_len = len(traj) - min_p = 6 - 2 = 4
        
        t_do_k_s_anal = True

    else:
        h_interp_len_max2 = trajectory.shape[0]                     # traj is very small

    if t_do_k_s_anal:                                               # test differet k and s combinations and gather inerpolation history.
        t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, t_k_s_combs, h_interp_len_max2, h_start_point_index2, debug = 0, debug_show_num_best = 2)
        t_k,t_s = t_comb_sol
    else:                                                           # use this with set k = 1, s = 5 to generate some interp history.
        t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, [(1,5)], 2, 0, debug = 0, debug_show_num_best = 2)
        t_k,t_s = t_comb_sol
        errors_sol_diff_norms_all[t_comb_sol] = {-1:5}              # dont have history to compare later..
        
    
    # extract active segments during inter-time between from_to merge. edges not included
    activeSegments = set(sum([vals for t,vals in lr_time_active_segments.items() if t_from_time_end < t < t_to_time_start],[]))
    active_segm_nodes = sum([t_segments_new[tID] for tID in activeSegments],[])

    activeNodes = [node for node in G.nodes() if t_from_time_end <= node[0] <= t_to_time_start and node not in active_segm_nodes]

    subgraph = G.subgraph(activeNodes)
    
    connected_components_unique = extract_graph_connected_components(subgraph.to_undirected(), lambda x: (x[0],x[1]))

    sol = [t_cc for t_cc in connected_components_unique if t_segments_new[t_from][-1] in t_cc]
    assert len(sol) == 1, "lr_conn_121_other_terminated_inspect inspect path relates to multiple clusters, dont expect it ever to occur"


    sol_combs = prep_combs_clusters_from_nodes(sol[0])
    sol_combs = {t_time:t_comb for t_time,t_comb in sol_combs.items() if t_to_time_start > t_time > t_from_time_end}
    
    # cyclic buffer that stores N entries and pushes old ones out when new value is appended
    N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
    t_last_deltas   = list(errors_sol_diff_norms_all[t_comb_sol].values())[-N:]
    t_norm_buffer   = CircularBuffer(N,                 t_last_deltas                    )
    t_traj_buff     = CircularBuffer(h_interp_len_max2, trajectory[  -h_interp_len_max2:])
    t_time_buff     = CircularBuffer(h_interp_len_max2, time[        -h_interp_len_max2:])
    t_time_next     = time[-1] + 1
    t_extrapolate_sol[t_conn] = {}
    t_extrapolate_sol_comb[t_conn] = {}
    t_times_accumulate_resolved = [t_from_time_end]
    for t_time, t_comb in sol_combs.items():
        # extrapolate traj
        t_extrap = interpolate_trajectory(t_traj_buff.get_data(), t_time_buff.get_data(), which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]

        # get possible permutations
        a = 1
        t_permutations = sum([list(itertools.combinations(t_comb, r)) for r in range(1,len(t_comb)+1)],[])
        t_centroids = []
        t_areas     = []
        for t_permutation in t_permutations:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][tID] for tID in t_permutation]))
            t_centroid, t_area = centroid_area(t_hull)
            t_centroids.append(t_centroid)
            t_areas.append(t_area)
        
        t_diffs = np.array(t_centroids).reshape(-1,2) - t_extrap
        t_diff_norms = np.linalg.norm(t_diffs, axis=1)
        t_where_min = np.argmin(t_diff_norms)
        t_sol_d_norm = t_diff_norms[t_where_min]
        t_known_norms = np.array(t_norm_buffer.get_data())
        t_mean = np.mean(t_known_norms)
        t_std = np.std(t_known_norms)
        if t_diff_norms[t_where_min] < max(t_mean, 5) + 5* t_std:
            t_norm_buffer.append(t_sol_d_norm)
            t_traj_buff.append(t_extrap)
            t_time_buff.append(t_time_next)
            t_time_next  += 1
            t_extrapolate_sol[t_conn][t_time] = t_extrap
            t_extrapolate_sol_comb[t_conn][t_time] = t_permutations[t_where_min]
            #print(f"inter!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
            t_times_accumulate_resolved.append(t_time)
        else:

            a = 1
            #print(f"stdev too much!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
            print(f'segment {t_from}->{t_to} (end: {t_segments_new[t_from][-1]}) extended trajectory to times: {t_times_accumulate_resolved}')
            break

        if t_time == t_to_time_start:
            print(f'segment {t_from}->{t_to} (end: {t_segments_new[t_from][-1]}) extended trajectory to times: {t_times_accumulate_resolved}. Recovered whole inter-segment!')
            


    a = 1
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
# ===========
# check whether  merges share nodes?
# =============
#lr_conn_merges_good = [] # may be redo to dict of tID?
lr_conn_merges_good = {tID:[] for tID in lr_conn_merges_to_nodes}
#lr_conn_merges_good2 = {tID:[] for tID in lr_conn_merges_to_nodes}
if 1 == 1:
    node_properties = defaultdict(list)
    # to find overlapping branches add owners to each node. multiple owners = contested.
    for key, dic in t_extrapolate_sol_comb.items():
        t_nodes =  [(key, value) for key, values in dic.items() for value in values]
        for t_node in t_nodes:
            node_properties[t_node].extend([key])
    t_duplicates = {t_node: t_branches for t_node,t_branches in node_properties.items() if len(t_branches) > 1}
    t_all_problematic_conns = list(set(sum(list(t_duplicates.values()),[])))
    t_all_problematic_conns_to = [a[1] for a in t_all_problematic_conns]
    for tID in lr_conn_merges_to_nodes:
        if tID not in t_all_problematic_conns_to:
            t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
            lr_conn_merges_good[tID] += t_conns_relevant
            #lr_conn_merges_good2[tID] += t_conns_relevant

    variants_possible = []
    if len(t_duplicates) > 0:
        # each contested node is an option. each choice of owner produces a different configuration branch
        variants = [] # generate these options for each contested node
        for t,(key,values) in enumerate(t_duplicates.items()):
            sd  = list(itertools.product([key],values))
            variants.append(sd)
        # generate all possible evoltions via product permutation: choice_1 -> choice_2 -> ...
        variants_all = list(itertools.product(*variants))
        # check if other paths, which will have at least one remaining node after deletion.
        for t_choice_evol in variants_all:
            t_skip = False
            for t_node,t_conn in t_choice_evol:
                t_time, *t_subIDs = t_node
                t_delete_conns = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
                for t_delete_conn in t_delete_conns:
                    if len(t_extrapolate_sol_comb[t_delete_conn][t_time]) == 1:
                        t_skip = True
            if not t_skip:
                variants_possible.append(t_choice_evol)
    if len(variants_possible) == 1: # if there is only one solution by default take it as answer
        t_choice_evol = variants_possible[0]
        for t_node,t_conn in t_choice_evol:
            tID = t_conn[1]
            t_time, *t_subIDs = t_node
            t_delete_conns = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
            for t_delete_conn in t_delete_conns:
                t_temp = t_extrapolate_sol_comb[t_delete_conn][t_time]
                t_temp = [t for t in t_temp if t not in t_subIDs]
                t_extrapolate_sol_comb[t_delete_conn][t_time] = t_temp
            t_conns_relevant = [t_c for t_c in t_extrapolate_sol_comb if t_c[1] == tID]
            lr_conn_merges_good[tID] += t_conns_relevant
            a = 1
            
if 1 == -1:
    for tID in lr_conn_merges_to_nodes:
        # grab t_conns with same merge to node
        t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
        # extract all times of merge branches. observation: times not including from node and incl to node.
        t_times_all = list(sorted(set(sum([list(t_extrapolate_sol_comb[t_conn].keys()) for t_conn in t_conns_relevant],[]))))
        # gather all contour ID combinations that were resolved for all merge branches
        t_combs_of_times = {t_time:[] for t_time in t_times_all}
        for t_conn in t_conns_relevant:
            for t_time, t_comb in t_extrapolate_sol_comb[t_conn].items():
                t_combs_of_times[t_time] += list(t_comb)
        # inspect if there are duplicates in contors =  branches share nodes = bad
        t_combs_of_times_num_copies = {t_time:[] for t_time in t_times_all}
        for t_time, t_comb in t_combs_of_times.items():
            _,t_num_copies = np.unique(t_comb, return_counts = True)
            t_combs_of_times_num_copies[t_time] = t_num_copies
        # save all times which have more than 1 duplicate
        t_times_combs_multi = []
        for t_time, t_copies_num in t_combs_of_times_num_copies.items():
            t_copies_has = True if any([1 if t > 1 else 0 for t in t_copies_num]) else False
            if t_copies_has : t_times_combs_multi.append(t_time)
        if len(t_times_combs_multi) == 0:
            lr_conn_merges_good[tID] += t_conns_relevant
        else:
            # branches share a contour, we have to resolve it.
            # take out contested nodes and try to distribute them to branches, so some condition is minimized.
            raise ValueError(f"merge branches share nodes, not encountered before. t_conns_relevant:{t_conns_relevant}")
        a = 1

a = 1    
# for_graph_plots(G)    #         <<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
# ===========
# modify graph based on extended merge branches
# > i think issue is present, but i use simple solution 1). 
# =============
# REMARK: branches are extended as far as possible. all secondary edges across branches can be purged
# REMARK: after branch ends its back to a wild territory - messy pre-merge connetions between nodes
# REMARK: old solo nodes are removed, new solo-cluster nodes that are forming branches are added.
# REMARK: branches might not be extended the to merge node and may terminate in different times
# ======:
# REMARK: maybe im overthinking, but purging nodes and adding edges in sequence should mess up next branch. 
# REMARK: e.g for a branch that ended earlier, i want to store edges 1-timestep after end, into the wild.
# REMARK: one such edge can be connected to a path of different branch.
# REMARK: next branch, that terminates later, and passes though that node, will be re-initialized
# REMARK: - that will drop all secondary connections along its path
# REMARK: by that logic order of sequence does not matter, and end of branch will be connected
# REMARK: only to free nodes.
# REMARK: i propose two options: 
# REMARK: 1) to do it in one iteration which should be correct if purges are "self-governing".
# REMARK: 2) to gather all nodes and future neighbors, reinitilize nodes (maybe a fault here) and drop
# REMARK: future neighbors that are part of merge branches
# REMARK: 1) is well though through. 2) idea works, its more difficult, some issues like solo/vs cluster node
# REMARK: is not resolved, comparing to 1). USE 1) FOR NOW.

# for_graph_plots(G)             #<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 

# 1)
if 1 == 1:
    for t_to,t_conns in lr_conn_merges_good.items():
        for t_conn in t_conns:
            t_from, t_to = t_conn
        
            t_combs = t_extrapolate_sol_comb[t_conn]

            if len(t_combs) == 0: continue

            t_nodes_all = []                                            # solo nodes will be deleted
            t_nodes_new = []                                            # new composite nodes are formed

            for t_time,t_subIDs in t_combs.items():

                for t_subID in t_subIDs:                                # single object nodes
                    t_nodes_all.append((t_time,t_subID))

                t_nodes_new.append(tuple([t_time] + list(t_subIDs)))    # cluster object nodes
        
            t_nodes_all += [t_segments_new[t_from][-1]]                 # (*) last old may have multiple connections not
                                                                        # covered by best choices in sol-n, so they are will stay.
            # find end edges to reconnect into graph
            # if last node is composite of solo nodes, each might have its edges
            t_time_end, *t_node_subIDs =  t_nodes_new[-1]               # grab subIDs and reconver solo nodes. i do this because
            t_node_last_solo_IDS = [(t_time_end,t_subID) for t_subID in t_node_subIDs] #  its easy to pick [-1], not lookup last time.

            #f = lambda x: x[0]
            t_node_last_solo_IDS_to = list(set(sum([list(G.successors(t_node)) for t_node in t_node_last_solo_IDS],[])))
            #t_node_last_solo_IDS_to = list(set(sum([extractNeighborsNext(G, t_node, f) for t_node in t_node_last_solo_IDS],[])))

            t_edges_next_new = [(t_nodes_new[-1] , t_node) for t_node in t_node_last_solo_IDS_to] # composite to old next neighbors

            G.remove_nodes_from(t_nodes_all)                            # remove all solo nodes and replace them with composite
            
            t_nodes_new_sides = t_segments_new[t_from][-2:] + t_nodes_new#  take nodes from -2 to end of segment, why -2 see (*)

            t_edges_sequence = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
            
            G.add_edges_from(t_edges_sequence)                          # from sequence of nodes create edges -> auto nodes
            G.add_edges_from(t_edges_next_new)                          # add future edges to solo nodes.

            t_segments_new[t_from] += t_nodes_new                       # append resolved to segments.
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 2)
if 1 == -1:
    t_neighbors_next = {}
    for t in lr_conn_merges_good.values(): # wonky structure
        for t_conn in t:
            t_neighbors_next[t_conn] = {}
            #{tID:{t_from:[] for t_from,t_to in t_edges} for tID, t_edges in lr_conn_merges_good.items()}

    t_nodes_remove  = []
    t_edges_add     = []

    for t_to,t_conns in lr_conn_merges_good.items():
        for t_conn in t_conns:
            t_from, t_to = t_conn
        
            t_combs = t_extrapolate_sol_comb[t_conn]
            if len(t_combs) == 0:                                       # connection with no intermediate nodes. no choices
                t_neighbors_next[t_conn][t_segments_new[t_from][-1]] =  [t_segments_new[t_to][0]]
                continue
            t_node_latest = t_segments_new[t_from][-1]
            t_nodes_all = []
            t_nodes_new = []
            for t_time,t_subIDs in t_combs.items():
                for t_subID in t_subIDs:                                # single object nodes
                    t_nodes_all.append((t_time,t_subID))
                t_nodes_new.append(tuple([t_time] + list(t_subIDs)))    # cluster object nodes
        
            t_nodes_all += [t_segments_new[t_from][-1]]
            t_nodes_remove += t_nodes_all
        
            t_neighbors_next[t_conn][t_nodes_new[-1]] =  extractNeighborsNext(G, t_nodes_new[-1],  lambda x: x[0])
        
            t_nodes_new_sides = t_segments_new[t_from][-2:] + t_nodes_new 

            pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
            t_edges_add += pairs
            t_segments_new[t_from] += t_nodes_new

    G.remove_nodes_from(t_nodes_remove)
    G.add_edges_from(t_edges_add)

    # remove next neighbor edges from branchs
    t_branches = {tID:sorted([t_conn[0] for t_conn in t_conns]) for tID,t_conns in lr_conn_merges_good.items()}
    t_edges_good = []
    for t_conn, t_edge_d  in t_neighbors_next.items():
        t_from, t_to = t_conn
        for t_node_from,t_node_to_all in t_edge_d.items():
            for t_node_to in t_node_to_all:
                t_node_not_in_segments = [True if t_node_to not in t_segments_new[tID] else False for tID in t_branches[t_to]]
                if all(t_node_not_in_segments):
                    t_edges_good.append((t_node_from,t_node_to))

    G.add_edges_from(t_edges_good)
a = 1

# ===============================================================================================
# extract unresolved multi-connections. NOT REALLY NEEDED, JUST CHECKING IF NOTHING BIG IS LOST
# ===============================================================================================
print("# extract unresolved multi-connections. NOT REALLY NEEDED, JUST CHECKING IF NOTHING BIG IS LOST")
if 1 == 1:
    # grab base segment IDs = ref to themselfes
    # > get master-> last slave relations. last slave has connections to successors
    t_segments_base_ID = [master for slave,master in lr_C3_condensed_connections_relations.items() if master == slave]
    t_m_to = {master:[] for master in t_segments_base_ID}
    for t_m in t_segments_base_ID:
        t_slaves = [slave for slave,master in lr_C3_condensed_connections_relations.items() if master == t_m]    # pick all slaves
        t_slaves_d = {t_slave:G2.nodes()[t_slave]["t_end"] for t_slave in t_slaves}                              # find each max time
        t_slave_last = max(t_slaves_d, key = t_slaves_d.get)                                                     # pick on with max time
        t_m_end = t_slaves_d[t_slave_last]                                                                       # this is last in inheritance sequence
        t_m_neighbors_next     = [tID for tID in list(G2.neighbors(t_slave_last)) if G2.nodes[tID]["t_start"  ] > t_m_end]
        t_m_neighbors_next2 = [lr_C3_condensed_connections_relations[tID] for tID in t_m_neighbors_next]
        t_m_to[t_m] = t_m_neighbors_next2
        #[t_segments_new[t][0] for t in t_m_neighbors_next2]
        a = 1
    # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # reconstruct solo&solo&.. -> merge, solo -> solo_split&.., mix of merge/splits and compare it to original
    G3 = nx.DiGraph()
    t_edges = [(source, target) for source, targets in t_m_to.items() for target in targets]
    G3.add_edges_from(t_edges)
    t_last_121s     = []
    t_last_merges   = []
    t_last_splits   = []
    t_last_merge_split_mix = []
    for t_node in G3.nodes():
        t_neighbors_next = list(G3.successors(t_node))                                              # grab future neighbors
        if len(t_neighbors_next) == 1 and len(list(G3.predecessors(t_neighbors_next[0]))) == 1:     # x1 future neig and it has x1 past neigh
            t_last_121s     += [(t_node,t_neighbors_next[0])]
        elif len(t_neighbors_next) == 1 and len(list(G3.predecessors(t_neighbors_next[0]))) > 1:    # same be multi past neigh = merge
            t_last_merges   += [(t_neighbor,t_neighbors_next[0]) for t_neighbor in G3.predecessors(t_neighbors_next[0]) if t_neighbor != t_neighbors_next[0]]
        elif len(t_neighbors_next) > 1:                                                             # splits into multiple
            a = 1
            t_predecessors_all = {t_neighbor: list(G3.predecessors(t_neighbor)) for t_neighbor in t_neighbors_next}
            t_pass_one = [1 if len(t_neig) == 1 else 0 for t_neig in t_predecessors_all.values()]               # calc neigh predecessor count
            if all(t_pass_one):                                                                     # if it was only source = split
                t_last_splits += [(t_node,t_neighbor) for t_neighbor in t_neighbors_next if t_neighbor != t_node]
            else:
                #t_last_merge_split_mix += [(t_node,t_neighbor) for t_neighbor in t_predecessors_all if t_neighbor != t_node]
                t_last_merge_split_mix += [(key, value) for key in t_predecessors_all for value in t_predecessors_all[key] if key != value]

    t_last_merges           = list(set([tuple(sorted(list(t_edge))) for t_edge in t_last_merges         ]))
    t_last_splits           = list(set([tuple(sorted(list(t_edge))) for t_edge in t_last_splits         ]))
    t_last_merge_split_mix  = list(set([tuple(sorted(list(t_edge))) for t_edge in t_last_merge_split_mix]))

    t_last_merges           = [t_conn for t_conn in t_last_merges if t_conn not in t_last_merge_split_mix]
    t_last_splits           = [t_conn for t_conn in t_last_splits if t_conn not in t_last_merge_split_mix]

    t_last_merges           = sorted(t_last_merges          , key = lambda x: (x[1],x[0]))
    t_last_splits           = sorted(t_last_splits          , key = lambda x: (x[0],x[1]))
    t_last_merge_split_mix  = sorted(t_last_merge_split_mix , key = lambda x: (x[0],x[1]))

    lr_conn_edges_merges = sorted(lr_conn_edges_merges  , key = lambda x: (x[1],x[0]))
    lr_conn_edges_splits = sorted(lr_conn_edges_splits  , key = lambda x: (x[0],x[1]))

    #assert len(t_last_121s) == 0                    , f"121s are left, examine these cases: {t_last_121s}"
    assert t_last_merges    == lr_conn_edges_merges , "merge reconstruction is different!"
    assert t_last_splits    == lr_conn_edges_splits , "split reconstruction is different!"

t_last_merge_split_mix
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# resolve or extend mixed merge-split cases. copied from pre-merge extension part
# REMARK: t_last_merge_split_mix contains all mixed connections, even false ones.
# REMARK: only 1 try is needed, but it should include furthest connection
# take all master segment- ones with earlier times (a predecessor on directed graph)
print("resolve or extend mixed merge-split cases.")
t_m_s_nodes_start = list(set(sum([list(G3.predecessors(t_node)) for t_node in G3.nodes()],[])))
# isolated connections with that master segment
t_m_s_nodes_neighbors0 = {t_conn:[] for t_conn in t_m_s_nodes_start}
for t_node in t_m_s_nodes_start:
    t_m_s_nodes_neighbors0[t_node] = [(t_from, t_to) for t_from, t_to in t_last_merge_split_mix if t_from == t_node]
t_m_s_nodes_neighbors0 = {t:t_conns for t,t_conns in t_m_s_nodes_neighbors0.items() if len(t_conns) > 0}
# extract edge with furtest time
t_m_s_nodes_neighbors = {}
for t_node in t_m_s_nodes_neighbors0:
    t_m_s_nodes_neighbors[t_node] = max(t_m_s_nodes_neighbors0[t_node], key = lambda x: t_segments_new[x[1]][0])

t_m_s_conns = list(t_m_s_nodes_neighbors.values())
t_extrapolate_sol = {}
t_extrapolate_sol_comb = {}
# take confirmed merge starting segment and determine k and s values for better extrapolation

for t_conn in t_m_s_conns:
    t_from, t_to        = t_conn
    t_from_time_end = t_segments_new[t_from][-1][0]
    t_to_time_start = t_segments_new[t_to][0][0]
    if t_to_time_start - t_from_time_end == 1:  # nothing to interpolate?
        t_extrapolate_sol_comb[t_conn] = {}     # although not relevant here, i want to store t_conn for next code
        continue
    trajectory          = np.array(list(t_centroids_all[t_from].values()))
    time                = np.array(list(t_centroids_all[t_from].keys()))
    t_do_k_s_anal = False
    
    if  trajectory.shape[0] > h_analyze_p_max + h_interp_len_max:   # history is large

        h_start_point_index2 = trajectory.shape[0] - h_analyze_p_max - h_interp_len_max

        h_interp_len_max2 = h_interp_len_max
        
        t_do_k_s_anal = True

    elif trajectory.shape[0] > h_analyze_p_min + h_interp_len_min:  # history is smaller, give pro to interp length rather inspected number count
                                                                    # traj [0,1,2,3,4,5], min_p = 2 -> 4 & 5, inter_min_len = 3
        h_start_point_index2 = 0                                    # interp segments: [2,3,4 & 5], [1,2,3 & 4]
                                                                    # but have 1 elem of hist to spare: [1,2,3,4 & 5], [0,1,2,3 & 4]
        h_interp_len_max2 = trajectory.shape[0]  - h_analyze_p_min  # so inter_min_len = len(traj) - min_p = 6 - 2 = 4
        
        t_do_k_s_anal = True

    else:
        h_interp_len_max2 = trajectory.shape[0]                     # traj is very small

    if t_do_k_s_anal:                                               # test differet k and s combinations and gather inerpolation history.
        t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, t_k_s_combs, h_interp_len_max2, h_start_point_index2, debug = 0, debug_show_num_best = 2)
        t_k,t_s = t_comb_sol
    else:                                                           # use this with set k = 1, s = 5 to generate some interp history.
        t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, [(1,5)], 2, 0, debug = 0, debug_show_num_best = 2)
        t_k,t_s = t_comb_sol
        errors_sol_diff_norms_all[t_comb_sol] = {-1:5}              # dont have history to compare later..
        
    
    # extract active segments during inter-time between from_to merge. edges not included
    activeSegments = set(sum([vals for t,vals in lr_time_active_segments.items() if t_from_time_end < t < t_to_time_start],[]))
    active_segm_nodes = sum([t_segments_new[tID] for tID in activeSegments],[])

    activeNodes = [node for node in G.nodes() if t_from_time_end <= node[0] <= t_to_time_start and node not in active_segm_nodes]

    subgraph = G.subgraph(activeNodes)
    
    connected_components_unique = extract_graph_connected_components(subgraph.to_undirected(), lambda x: (x[0],x[1]))

    sol = [t_cc for t_cc in connected_components_unique if t_segments_new[t_from][-1] in t_cc]
    assert len(sol) == 1, "lr_conn_121_other_terminated_inspect inspect path relates to multiple clusters, dont expect it ever to occur"


    sol_combs = prep_combs_clusters_from_nodes(sol[0])
    sol_combs = {t_time:t_comb for t_time,t_comb in sol_combs.items() if t_time > t_from_time_end}
    
    # cyclic buffer that stores N entries and pushes old ones out when new value is appended
    N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
    t_last_deltas   = list(errors_sol_diff_norms_all[t_comb_sol].values())[-N:]
    t_norm_buffer   = CircularBuffer(N,                 t_last_deltas                    )
    t_traj_buff     = CircularBuffer(h_interp_len_max2, trajectory[  -h_interp_len_max2:])
    t_time_buff     = CircularBuffer(h_interp_len_max2, time[        -h_interp_len_max2:])
    t_time_next     = time[-1] + 1
    t_extrapolate_sol[t_conn] = {}
    t_extrapolate_sol_comb[t_conn] = {}
    t_times_accumulate_resolved = [t_from_time_end]
    for t_time, t_comb in sol_combs.items():
        # extrapolate traj
        t_extrap = interpolate_trajectory(t_traj_buff.get_data(), t_time_buff.get_data(), which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]

        # get possible permutations
        a = 1
        t_permutations = sum([list(itertools.combinations(t_comb, r)) for r in range(1,len(t_comb)+1)],[])
        t_centroids = []
        t_areas     = []
        for t_permutation in t_permutations:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][tID] for tID in t_permutation]))
            t_centroid, t_area = centroid_area(t_hull)
            t_centroids.append(t_centroid)
            t_areas.append(t_area)
        
        t_diffs = np.array(t_centroids).reshape(-1,2) - t_extrap
        t_diff_norms = np.linalg.norm(t_diffs, axis=1)
        t_where_min = np.argmin(t_diff_norms)
        t_sol_d_norm = t_diff_norms[t_where_min]
        t_known_norms = np.array(t_norm_buffer.get_data())
        t_mean = np.mean(t_known_norms)
        t_std = np.std(t_known_norms)
        if t_diff_norms[t_where_min] < max(t_mean, 5) + 5* t_std:
            t_norm_buffer.append(t_sol_d_norm)
            t_traj_buff.append(t_extrap)
            t_time_buff.append(t_time_next)
            t_time_next  += 1
            t_extrapolate_sol[t_conn][t_time] = t_extrap
            t_extrapolate_sol_comb[t_conn][t_time] = t_permutations[t_where_min]
            #print(f"inter!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
            t_times_accumulate_resolved.append(t_time)
        else:

            a = 1
            #print(f"stdev too much!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
            print(f'segment {t_from}->{t_to} (end: {t_segments_new[t_from][-1]}) extended trajectory to times: {t_times_accumulate_resolved}')
            break

        if t_time == t_to_time_start:
            print(f'segment {t_from}->{t_to} (end: {t_segments_new[t_from][-1]}) extended trajectory to times: {t_times_accumulate_resolved}. Recovered whole inter-segment!')
            


    a = 1
# lets check if there are competing solutions. drop each contour ID in a bin for each time.
t_m_s_times = {}
for t_conn,t_dic in t_extrapolate_sol_comb.items():
    for t_time,t_elems in t_dic.items():
        if t_time not in t_m_s_times: t_m_s_times[t_time] = []     # add time to dic if there is none
        t_m_s_times[t_time] += list(t_elems)                       # append elems

# count number of duplicates in a bin- if there is, means it is shared in solution.
from collections import Counter
t_m_s_pass = [sum(count - 1 for count in Counter(t_elems).values()) for t_elems in t_m_s_times.values()]
t_m_s_pass = [True if t_count == 0 else False for t_count in t_m_s_pass]
assert all(t_m_s_pass), "merge split trajectry extension issue: solution has competing elements!"

# ===============================================================================================
# fill resolved merge-split cases, structure copied from lr_permutation_cases.
# REMARK: m-s mixed extension was calculated until furthest neighbor. replace target t_to with proper one
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("fill resolved merge-split cases")
t_m_s_conns_retargeted      = []        # hold all block solutions here
t_m_s_conns_extended_fully  = []        # store cases which fully connect two segments.
for t_conn in t_m_s_conns:
    if len(t_extrapolate_sol_comb[t_conn]) == 0: continue                     # NOT IMPLEMENTED. its zero length intepolation intervals..
    t_from,t_to = t_conn
    t_choices = [t_conns[1] for t_conns in t_m_s_nodes_neighbors0[t_from]]    # these are all options for t_from
    t_choices_traj = {tID:t_segments_new[tID] for tID in t_choices}           # have to grab all nodes for options
    t_time = max(t_extrapolate_sol_comb[t_conn].keys())                     # end time might be different from goal time
    t_to_time = t_segments_new[t_to][0][0]                                    # and extract one node at the end of extensions
    if t_time == t_to_time:
        t_choices_nodes = {tID:[t_node for t_node in t_traj if t_node[0] == t_time][0] for tID,t_traj in t_choices_traj.items()}
        t_sol_node_elem = t_extrapolate_sol_comb[t_conn][t_time][0]               # <<<< taking first element of composite node is incorrect. sols migh be shared.
        t_sol = [tID for tID, t_node in t_choices_nodes.items() if t_sol_node_elem in t_node[1:]] # find matches by filtering
        assert len(t_sol) == 1, "m-s extension does not reach true t_to segment"
    else:
        t_sol = [-1]
    t_sol_new = (t_from,t_sol[0])
    t_m_s_conns_retargeted += [t_sol_new]                              
    t_temp = t_extrapolate_sol_comb[t_conn]
    t_extrapolate_sol_comb.pop(t_conn,None)
    t_extrapolate_sol_comb[t_sol_new] = t_temp
    if t_time == t_to_time:
        t_m_s_conns_extended_fully += [t_sol_new]
   
    a = 1


C4 = C3.copy()
C4.add_edges_from([t_conn for t_conn in t_m_s_conns_retargeted if t_conn in t_m_s_conns_extended_fully])
    
lr_C4_condensed_connections = extract_graph_connected_components(C4, lambda x: x)
# lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
t_condensed_connections_all_nodes = sorted(sum(lr_C4_condensed_connections,[])) # neext next
lr_C4_condensed_connections_relations = {tID: tID for tID in range(len(t_segments_new))} #t_condensed_connections_all_nodes
for t_subIDs in lr_C4_condensed_connections:
    for t_subID in t_subIDs:
        lr_C4_condensed_connections_relations[t_subID] = min(t_subIDs)

# ===============================================================================================
# process connections that connect segments fully (span all inter-segment time space)
# REMARK: copied from similar procedure, added missing node parameters (centroid,hull_area,..)
print("# process connections that connect segments fully (span all inter-segment time space)")
if 1 == 1: 
    # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # changed: t_nodes_new_sides, t_path, t_times, t_nodes_intermediate
    for t_conn in t_m_s_conns_extended_fully:
        t_path  = t_extrapolate_sol_comb[t_conn].values()          
        t_times = t_extrapolate_sol_comb[t_conn].keys()     
        t_from, t_to = t_conn                                     
        t_from_new  = lr_C4_condensed_connections_relations[t_from] # below might be a problem, not present in OG code.ref to cleared array. changed to t_from_new
        # need to prepend last node from t_from segment, could do it in t_extrapolate_sol_comb, but for now do it here
        t_time_prev = min(t_times) - 1
        t_node_prev = [t_node for t_node in t_segments_new[t_from_new] if t_node[0] == t_time_prev]
        t_subIDs= tuple(t_node_prev[0][1:])
        t_times =  [t_time_prev] + list(t_times) 
        t_path  = [t_subIDs] + list(t_path)
        t_nodes_new = []
        for t_time,t_comb in zip(t_times,t_path):                 
            t_nodes_new.append(tuple([t_time] + list(t_comb)))    

        t_nodes_all = []
        for t_time,t_comb in t_extrapolate_sol_comb[t_conn].items():
            for tID in t_comb:
                t_nodes_all.append(tuple([t_time, tID]))
    
        # > integrating new paths is not that simple. Basically LEFT last node could have been connected to multiple nodes and removing only intermediate resolved nodes would not remove extra connection.
        # its easy to remove start-end point nodes, but they will lose connection to segments
        G.remove_nodes_from(t_nodes_all)                          # edges will be also dropped
        # so add extra nodes to make edges with segments, which will create start-end points again.
        t_nodes_new_sides = [t_segments_new[t_from_new][-2]] + t_nodes_new + [t_segments_new[t_to][1]] 

        t_pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
    
        G.add_edges_from(t_pairs)

        # determine t_from masters index, send that segment intermediate nodes and second segment
        # if t_from is its own master, it still works, check it with simple ((0,1),(1,2)) and {0:0,1:0,2:0}
        
        t_nodes_intermediate = list(sorted(t_nodes_new, key = lambda x: x[0]))[1:-1]
        t_segments_new[t_from_new] += t_nodes_intermediate
        t_segments_new[t_from_new] += t_segments_new[t_to]

    
        # fill centroid, area and momement of innertia zz missing for intermediate segment
        for t_time,*t_subIDs in t_nodes_intermediate:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
            t_node = tuple([t_time] + t_subIDs)
            set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
        
        if len(t_segments_121_centroids[   t_to]) == 0: # something was missing
            for t_time,*t_subIDs in t_segments_new[t_to]:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
                t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
                t_node = tuple([t_time] + t_subIDs)
                set_ith_elements_multilist_at_depth([t_to,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    
        # copy data from inherited
        combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
        # wipe data if t_from is inherited
        if t_from_new != t_from:
            set_ith_elements_multilist(t_from, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
        # wipe data from t_to anyway
        set_ith_elements_multilist(t_to, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
        
a = 1
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# process connections that extend partially and dont 121 segments. splits/merges may be present
# REMARK: copied from merge extend. DID NOT CHANGE MUCH! IDK IF IT WORKS IN ALL CASES!
print("# process connections that extend partially and dont 121 segments.")
if 1 == 1: 
    for t_conn in [t_conn for t_conn in t_m_s_conns_retargeted if t_conn not in t_m_s_conns_extended_fully]:
        t_from, t_to = t_conn
        
        t_combs = t_extrapolate_sol_comb[t_conn]

        if len(t_combs) == 0: continue

        t_nodes_all = []                                            # solo nodes will be deleted
        t_nodes_new = []                                            # new composite nodes are formed

        for t_time,t_subIDs in t_combs.items():

            for t_subID in t_subIDs:                                # single object nodes
                t_nodes_all.append((t_time,t_subID))

            t_nodes_new.append(tuple([t_time] + list(t_subIDs)))    # cluster object nodes
        
        t_nodes_all += [t_segments_new[t_from][-1]]                 # (*) last old may have multiple connections not
                                                                    # covered by best choices in sol-n, so they are will stay.
        # find end edges to reconnect into graph
        # if last node is composite of solo nodes, each might have its edges
        t_time_end, *t_node_subIDs =  t_nodes_new[-1]               # grab subIDs and reconver solo nodes. i do this because
        t_node_last_solo_IDS = [(t_time_end,t_subID) for t_subID in t_node_subIDs] #  its easy to pick [-1], not lookup last time.

        #f = lambda x: x[0]
        t_node_last_solo_IDS_to = list(set(sum([list(G.successors(t_node)) for t_node in t_node_last_solo_IDS],[])))
        #t_node_last_solo_IDS_to = list(set(sum([extractNeighborsNext(G, t_node, f) for t_node in t_node_last_solo_IDS],[])))

        t_edges_next_new = [(t_nodes_new[-1] , t_node) for t_node in t_node_last_solo_IDS_to] # composite to old next neighbors

        G.remove_nodes_from(t_nodes_all)                            # remove all solo nodes and replace them with composite
            
        t_nodes_new_sides = t_segments_new[t_from][-2:] + t_nodes_new#  take nodes from -2 to end of segment, why -2 see (*)

        t_edges_sequence = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
            
        G.add_edges_from(t_edges_sequence)                          # from sequence of nodes create edges -> auto nodes
        G.add_edges_from(t_edges_next_new)                          # add future edges to solo nodes.

        t_segments_new[t_from] += t_nodes_new                       # append resolved to segments.

        # add calc

        for t_time,*t_subIDs in t_nodes_new:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
            t_node = tuple([t_time] + t_subIDs)
            t_segments_121_centroids[t_from][t_node]   = t_centroid
            t_segments_121_areas[    t_from][t_node]   = t_area
            t_segments_121_mom_z[    t_from][t_node]   = t_mom_z


# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ===============================================================================================
# try to find fake split-merges
# ===============================================================================================
# REMARK: these are segments sandwitched between  split-merge events or split-121/ 121-merge due to optical artifacts.
print("# try to find fake split-merges")
t_last_splits
t_split_form = list(set([t_from for t_from,_ in t_last_splits]))
t_max_post_split_time = 20   # total time post split detection
t_max_fake_split_time = 10   # max duration of fake split branches
t_segment_start_times = {tID:t_segment[0][0] for tID,t_segment in enumerate(t_segments_new) if len(t_segment)>0}
t_segment_end_times = {tID:t_segment[-1][0] for tID,t_segment in enumerate(t_segments_new) if len(t_segment)>0}
t_fake_split_merge_conns = {}
lr_inspect_contour_combs = {} # redefined
lr_inspect_c_c_from_to_interp_times2 = {}
lr_inspect_121_interpolation_from_to2 = {}
lr_inspect_121_interpolation_times2 = {}
if 1 == 1:
    for t_from in t_split_form:
        t_from_node_last = t_segments_new[t_from][-1]
        t_from_time_end = t_from_node_last[0]
        # get targets that start within next N time steps
        t_rough_targets = [tID for tID,t_to_time in t_segment_start_times.items() if 0 < t_to_time - t_from_time_end < t_max_post_split_time]
        # refine targets to keep only only connected
        t_better_targets = []
        for tID in t_rough_targets:      
             if nx.has_path(G, source=t_from_node_last, target=t_segments_new[tID][-1]):
                 t_better_targets.append(tID)

        t_split_neighbors = [t_t for t_f,t_t in t_last_splits if t_f == t_from] 
        # drop split targets and check whats left, maybe merge segment
        t_temp = {t:t_segment_start_times[t] for t in t_better_targets if t not in t_split_neighbors}
        # real split (without recombination) creates (ussualy) long branches and no merge short term

        ### prune short branches, this leaves possible merge segment: drop those who end in t_max_fake_split_time
        ###t_temp = {t:t_start_time for t,t_start_time in t_temp.items() if t_segment_end_times[t] > t_from_time_end + t_max_fake_split_time}

        if len(t_temp) == 0: continue # terminate if split only into branches larger than t_max_fake_split_time-  real split.
        # t_max_post_split_time might be so big that 121 is included inside. fix it later <<<<<<<<<
        # and drop them as redundand, keep closest pre-121
        #t_temp = {t:t_segment_start_times[t] for t in t_better_targets}
        t_furthest_neighbor = max(t_temp, key = t_temp.get)
        t_to_from_time_start = t_temp[t_furthest_neighbor]
        assert len([tID for tID,t_time in t_temp.items() if t_time == t_to_from_time_start]) == 1, "finding fake split-merges, multiple target segements (same max time)"
        t_conn = (t_from,t_furthest_neighbor)

        t_from_areas_last = list(t_segments_121_areas[t_from].values())[-4:] 
        t_to_areas_first = list(t_segments_121_areas[t_furthest_neighbor].values())[:4]

        t_from_areas_mean   = np.mean(t_from_areas_last)
        t_to_areas_mean     = np.mean(t_to_areas_first)

        if np.abs(t_to_areas_mean-t_from_areas_mean)/t_from_areas_mean < 0.5:
            t_fake_split_merge_conns[t_conn] = t_better_targets

            activeNodes = [node for node in G.nodes() if t_from_time_end <= node[0] <= t_to_from_time_start]
            subgraph = G.subgraph(activeNodes)
            # extract connected branches
            connected_components_unique = extract_graph_connected_components(subgraph.to_undirected(), sort_function = lambda x: (x[0], *x[1:]))
            # pick branch with known node
            sol = [t_cc for t_cc in connected_components_unique if t_segments_new[t_from][-1] in t_cc]
            assert len(sol) == 1, "lr_conn_121_other_terminated_inspect inspect path relates to multiple clusters, dont expect it ever to occur"
            # > gather all possible choices for each time step
            t_times = np.arange(t_from_time_end,t_to_from_time_start + 1, 1)
            lr_inspect_121_interpolation_times2[t_conn] = t_times
            t_inspect_contour_combs = {t_time:[] for t_time in t_times}
    
            for t_time,*t_subIDs in sol[0]:
                t_inspect_contour_combs[t_time] += t_subIDs

            for t_time in t_inspect_contour_combs:
                t_inspect_contour_combs[t_time] = sorted(t_inspect_contour_combs[t_time])
        
            lr_inspect_contour_combs[t_conn] = t_inspect_contour_combs

            t_from_nodes_part = t_segments_new[t_from][-4:] # consider with next line, read comment.
            t_to_from_nodes_part = t_segments_new[t_furthest_neighbor][:4]
            lr_inspect_c_c_from_to_interp_times2[t_conn] = []
            lr_inspect_c_c_from_to_interp_times2[t_conn].append(t_from_nodes_part)
            lr_inspect_c_c_from_to_interp_times2[t_conn].append(t_to_from_nodes_part)
            lr_inspect_121_interpolation_from_to2[t_conn] = t_conn


# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if 1 == 1: # copied from copy of OG
    # prepare possible permutations of clusters
    lr_inspect_contour_combs_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_inspect_contour_combs.items()}
    for t_conn, t_times_contours in lr_inspect_contour_combs.items():
        for t_time,t_contours in t_times_contours.items():
            t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
            lr_inspect_contour_combs_perms[t_conn][t_time] = t_perms
    # predefine storage
    lr_inspect_permutation_areas_precomputed        = lr_init_perm_precomputed(lr_inspect_contour_combs_perms, 0    )
    lr_inspect_permutation_centroids_precomputed    = lr_init_perm_precomputed(lr_inspect_contour_combs_perms, [0,0])
    lr_inspect_permutation_mom_z_precomputed        = lr_init_perm_precomputed(lr_inspect_contour_combs_perms, 0    )
    # calculate parameters of each combination
    for t_conn, t_times_perms in lr_inspect_contour_combs_perms.items():
        for t_time,t_perms in t_times_perms.items():
            for t_perm in t_perms:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
                t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                lr_inspect_permutation_areas_precomputed[       t_conn][t_time][t_perm]     = t_area
                lr_inspect_permutation_centroids_precomputed[   t_conn][t_time][t_perm]     = t_centroid
                lr_inspect_permutation_mom_z_precomputed[       t_conn][t_time][t_perm]     = t_mom_z

    a = 1

if 1 == 1: #keep on stealing
    lr_inspect_121_interpolation_centroids  = {t_conn:[] for t_conn in lr_inspect_c_c_from_to_interp_times}
    lr_inspect_121_interpolation_areas      = {t_conn:[] for t_conn in lr_inspect_c_c_from_to_interp_times}
    lr_inspect_121_interpolation_moment_z   = {t_conn:[] for t_conn in lr_inspect_c_c_from_to_interp_times}

    # similar to OG
    for t_conn, [t_hist_prev,t_hist_next] in lr_inspect_c_c_from_to_interp_times2.items(): #<<< notice 2
        t_from, t_to    = lr_inspect_121_interpolation_from_to2[t_conn]
        # tID might not be in 121s, which was an ideal case. rough fix is to add missing.
        for tID in lr_inspect_121_interpolation_from_to2[t_conn]:
            if len(t_segments_121_centroids[tID]) == 0:
                for t_time,*t_subIDs in t_segments_new[tID]:
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
                    t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
                    t_node = tuple([t_time] + t_subIDs)
                    t_segments_121_centroids[   tID][t_node]   = t_centroid
                    t_segments_121_areas[       tID][t_node]   = t_area
                    t_segments_121_mom_z[       tID][t_node]   = t_mom_z

        t_times_prev    = [t_node[0] for t_node in t_hist_prev]
        t_times_next    = [t_node[0] for t_node in t_hist_next]

        t_traj_prev_c     = np.array([t_segments_121_centroids[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next_c     = np.array([t_segments_121_centroids[  t_to][t_node] for t_node in t_hist_next])

        t_traj_prev_a     = np.array([t_segments_121_areas[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next_a     = np.array([t_segments_121_areas[  t_to][t_node] for t_node in t_hist_next])

        t_traj_prev_mz     = np.array([t_segments_121_mom_z[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next_mz     = np.array([t_segments_121_mom_z[  t_to][t_node] for t_node in t_hist_next])

        t_interp_times  = lr_inspect_121_interpolation_times2[t_conn]

        debug = 0
        lr_inspect_121_interpolation_centroids[t_conn]  = interpolateMiddle2D(t_times_prev,t_times_next,t_traj_prev_c, t_traj_next_c,
                                                                      t_interp_times, s = 15, debug = debug,
                                                                      aspect = 'equal', title = t_conn)

        lr_inspect_121_interpolation_areas[t_conn]      = interpolateMiddle1D(t_times_prev,t_times_next,t_traj_prev_a, t_traj_next_a,
                                                                     t_interp_times,  rescale = True, s = 15, debug = debug,
                                                                     aspect = 'auto', title = t_conn)

        lr_inspect_121_interpolation_moment_z[t_conn]   = interpolateMiddle1D(t_times_prev,t_times_next,t_traj_prev_mz, t_traj_next_mz,
                                                                     t_interp_times,  rescale = True, s = 15, debug = debug,
                                                                     aspect = 'auto', title = t_conn)

        a = 1

if 1 == 1:
    lr_inspect_permutation_cases = {t_conn:[] for t_conn in lr_inspect_contour_combs_perms}
    lr_inspect_permutation_times = {t_conn:[] for t_conn in lr_inspect_contour_combs_perms}
    #lr_inspect_contour_combs_perms_drop = {}
    for t_conn, t_times_perms in lr_inspect_contour_combs_perms.items():
    
        t_values = list([t_val for t_time,t_val in t_times_perms.items() if t_time in lr_inspect_121_interpolation_times2[t_conn]])
        t_times = lr_inspect_121_interpolation_times2[t_conn]

        # calculate number of combinations
        sequences_length = itertools_product_length(t_values)
        
        if sequences_length > 15000:
            t_from, t_to = t_conn
            t_to_all = t_fake_split_merge_conns[t_conn]
            t_to_branches = [tID for tID in t_to_all if tID != t_to] #<
            

            sequences = perms_with_branches(t_to_branches,t_segments_new,t_times_contours)
            if 1 == -11: # old pre-function body
                # if branch segments are present their sequences will be continuous and present fully
                # means soolution will consist of combination of branches + other free nodes.
            
                # take combination of branches
                t_branches_perms = sum([list(itertools.combinations(t_to_branches, r)) for r in range(1,len(t_to_branches)+1)],[])

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
                    t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
                    t_contour_combs_perms[t_time] = t_perms
                # prepare combination of branches. intermediate combinations should be gathered and frozen
                for t, t_branch_IDs in enumerate(t_branches_perms):
                    t_branch_comb_variants.append(copy.deepcopy(t_contour_combs_perms))    # copy a primer.
                    t_temp = {}                                                            # this buffer will gather multiple branches and their ID together
                    for t_branch_ID in t_branch_IDs:
                        for t_time, *t_subIDs in t_segments_new[t_branch_ID]:
                            if t_time not in t_temp: t_temp[t_time] = []
                            t_temp[t_time] += list(t_subIDs)                              # fill buffer
                    for t_time, t_subIDs in t_temp.items():
                        t_branch_comb_variants[t][t_time] += [tuple(t_subIDs)]            # walk buffer and add frozen combs to primer

                #aa2 = [itertools_product_length(t_choices.values()) for t_choices in t_branch_comb_variants]
                # do N different variants and combine them together. it should be much shorted, tested on case where 138k combinations with 2 branches were reduced to 2.8k combinations
                sequences =  sum([list(itertools.product(*t_choices.values())) for t_choices in t_branch_comb_variants],[])
            
            a = 1
            
            
        else:
            sequences = list(itertools.product(*t_values))

        lr_inspect_permutation_cases[t_conn] = sequences
        lr_inspect_permutation_times[t_conn] = t_times
    
    
    # have to edit centroids, since they are generated bit differently- chop edges. so i can use lr_evel_perm_interp_data()
    t_temp_c = {tID:vals[1:-1] for tID, vals in lr_inspect_121_interpolation_centroids.items()}
    t_args = [lr_inspect_permutation_cases,t_temp_c,lr_inspect_permutation_times,
              lr_inspect_permutation_centroids_precomputed,lr_inspect_permutation_areas_precomputed,lr_inspect_permutation_mom_z_precomputed]


    t_inspect_sols_c, t_inspect_sols_c_i, t_inspect_sols_a, t_inspect_sols_m = lr_evel_perm_interp_data(*t_args) # dropped areas since its almost same as momZ ??? check pls



    t_weights = [1,2,0,1]
    t_sols = [t_inspect_sols_c, t_inspect_sols_c_i, t_inspect_sols_a, t_inspect_sols_m]
    lr_inspect_weighted_solutions_max, lr_inspect_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_weights, t_sols, lr_inspect_permutation_cases )

if  1 == 1:
    C5 = C4.copy()
    t_temp = [tuple(lr_inspect_121_interpolation_from_to2[t]) for t in lr_inspect_permutation_cases]
    C5.add_edges_from(t_temp)
    
    lr_C5_condensed_connections = extract_graph_connected_components(C5, lambda x: x)
    # lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
    t_condensed_connections_all_nodes = sorted(sum(lr_C5_condensed_connections,[])) # neext next
    lr_C5_condensed_connections_relations = {tID: tID for tID in range(len(t_segments_new))} #t_condensed_connections_all_nodes
    for t_subIDs in lr_C5_condensed_connections:
        for t_subID in t_subIDs:
            lr_C5_condensed_connections_relations[t_subID] = min(t_subIDs)

    # dont forget redirect fake connection to OG # changed from OG! <<<
    for (t_from, t_to_new),t_all_IDs in t_fake_split_merge_conns.items():
        t_secondary_IDs = [tID for tID in t_all_IDs if tID != t_to_new]
        for tID in t_secondary_IDs:
            lr_C5_condensed_connections_relations[tID] = t_from


# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if 1 == 1:
    for t_conn in lr_inspect_permutation_cases:
        t_sol   = lr_inspect_weighted_solutions_max[t_conn]
        t_path  = lr_inspect_permutation_cases[t_conn][t_sol]             # t_path contains start-end points of segments !!!
        t_times = lr_inspect_permutation_times[t_conn]
        #t_nodes_old = []
        t_nodes_new = []
        for t_time,t_comb in zip(t_times,t_path):
            #for tID in t_comb:
            #    t_nodes_old.append(tuple([t_time, tID]))          # old type of nodes in solution: (time,contourID)     e.g (t1,ID1)
            t_nodes_new.append(tuple([t_time] + list(t_comb)))    # new type of nodes in solution: (time,*clusterIDs)   e.g (t1,ID1,ID2,...)

        t_nodes_all = []
        for t_time,t_comb in {tID:vals for tID,vals in lr_inspect_contour_combs[t_conn].items() if tID in lr_inspect_121_interpolation_times2[t_conn]}.items():
            for tID in t_comb:
                t_nodes_all.append(tuple([t_time, tID]))
    
        # its easy to remove start-end point nodes, but they will lose connection to segments
        G.remove_nodes_from(t_nodes_all)
        #[lr_conn_edges_merges.remove(t_edge) for t_edge in 
        # so add extra nodes to make edges with segments, which will create start-end points again.
        t_from, t_to_fake = t_conn

        t_from, t_to = lr_inspect_121_interpolation_from_to2[t_conn]                                     
        t_nodes_new_sides = [t_segments_new[t_from][-2]] + t_nodes_new + [t_segments_new[t_to][1]]

        pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
    
        G.add_edges_from(pairs)
        # confirm stored possible = resolved
        #lr_inspect_121_to_merge_resolved_IDs[t_conn] = lr_inspect_121_to_merge_possible_IDs[t_conn]

        # =========== have to modify it from OG because its merge/split case, so extra branches
        # determine t_from masters index, send that segment intermediate nodes and second segment
        # if t_from is its own master, it still works, check it with simple ((0,1),(1,2)) and {0:0,1:0,2:0}
        t_from_new = lr_C5_condensed_connections_relations[t_from]
        t_nodes_intermediate = list(sorted(t_nodes_new, key = lambda x: x[0]))[1:-1]
        t_segments_new[t_from_new] += t_nodes_intermediate
        t_segments_new[t_from_new] += t_segments_new[t_to] # edited from OG

    
        # fill centroid, area and momement of innertia zz missing for intermediate segment
        for t_time,*t_subIDs in t_nodes_intermediate:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
            t_node = tuple([t_time] + t_subIDs)
            set_ith_elements_multilist_at_depth([t_from_new,t_node], [t_centroid,t_area,t_mom_z], t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    
        # copy data from inherited
        combine_dictionaries_multi(t_from_new,t_to,t_segments_121_centroids,t_segments_121_areas,t_segments_121_mom_z)
    
        # wipe data if t_from is inherited
        if t_from_new != t_from:
            set_ith_elements_multilist(t_from, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
           
        # wipe data from t_to anyway
        set_ith_elements_multilist(t_to, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)
        # wipe data from t_to_fake anyway
        #set_ith_elements_multilist(t_to_fake, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)

        t_all_IDs = t_fake_split_merge_conns[t_conn]
        t_secondary_IDs = [tID for tID in t_all_IDs if tID != t_to_new]
        for t_to_fake in t_secondary_IDs:
            set_ith_elements_multilist(t_to_fake, [[],{},{},{}], t_segments_new, t_segments_121_centroids, t_segments_121_areas, t_segments_121_mom_z)

# ===============================================================================================
# try to continue segments both ways.
# ===============================================================================================
# !!! reconstruct inter segment connectivity from 
# redoing segment connection from t_segments_new.
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#fin_edges = lr_reindex_masters(lr_C5_condensed_connections_relations, list(G3.edges()), remove_solo_ID = 0)
#fin_base_nodes  = [t for t,t_traj in enumerate(t_segments_new) if len(t_traj) > 0]
# find master segments that have absorbed other (slaves). slaves are zero length in t_segments_new
fin_segments_masters  = [t for t,t_traj in enumerate(t_segments_new) if len(t_traj) > 0]
fin_segments_slaves   = [t for t in range(len(t_segments_new)) if t not in fin_segments_masters]
fin_segments_conn_prev = defaultdict(list)
# masters (leftmost) connect to the other masters slave to the left. via C5 reconnect to master.

for t_node in G.nodes():                    # updating time parameters for graph nodes
    G.nodes[t_node]['time'] = t_node[0]

# creating segement conenctivity abstraction graph
G4 = nx.Graph()                             
G4.add_nodes_from(fin_segments_masters)     # adding masternodes
for t_node in G4.nodes():                   # adding segment start and end point times for easy retreval.
    G4.nodes[t_node]["time_start"]  = t_segments_new[t_node][0][0]
    G4.nodes[t_node]["time_end"]    = t_segments_new[t_node][-1][0]

# gen all possible interconnections
fin_segments_conns_all = list(itertools.combinations(fin_segments_masters, 2)) 
fin_segments_conns_all_dict = defaultdict(list)
for t_from,t_to in fin_segments_conns_all:  # condensing connections into ID:*all_right_side_segments
    fin_segments_conns_all_dict[t_from].append(t_to)  # order emerges itself: ID < min(all_right_side_segments)

# choose only right neighbors that start within time lr_maxDT inteval
fin_segments_conns_refined_DT_dict = defaultdict(dict)
for t_from, t_to_all in fin_segments_conns_all_dict.items():
    t_from_time_end = G4.nodes[t_from]["time_end"]
    t_time_to_maxDT = t_from_time_end + lr_maxDT
    for t_to in t_to_all:
        if t_from_time_end < G4.nodes[t_to]["time_start"] <= t_time_to_maxDT:
            fin_segments_conns_refined_DT_dict[t_from][t_to] = G4.nodes[t_to]["time_start"] - t_from_time_end

# refine those that are connected
#fin_segments_conns_DT_connected_all_subgraphs = defaultdict(list) # store subgraphs here for next step.
fin_segments_conns_DT_connected_all = defaultdict(list)  
for t_from, t_to_dict in fin_segments_conns_refined_DT_dict.items():
    t_to_max_dist_ID      = max(t_to_dict.keys())    # need subraph that includes all times to viable neighbors
    t_to_max_time   = G4.nodes[t_to_max_dist_ID]["time_start"]  # this is end time
    t_from_time_end = G4.nodes[t_from]["time_end"]
    t_active_DT_nodes = [t_node for t_node, t_time in G.nodes(data='time') if t_from_time_end <= t_time <= t_to_max_time]
    g_limited = G.subgraph(t_active_DT_nodes)        # isolated graph
    t_from_end_node = t_segments_new[t_from][-1]
    for t_to in t_to_dict:
        t_to_start_node = t_segments_new[t_to][0]
        hasPath = nx.has_path(g_limited, source = t_from_end_node, target = t_to_start_node)
        if hasPath:
            fin_segments_conns_DT_connected_all[t_from].append(t_to)

# check if each connection can hold without support of others (no segments-bridges inbetween)
# > take each ID of t_to_connected and remove all branches from subgraph that are not ID. check path.
fin_segments_conns_DT_connected = defaultdict(list)
for t_from, t_to_connected in fin_segments_conns_DT_connected_all.items():
    t_to_dict = {t:fin_segments_conns_refined_DT_dict[t_from][t] for t in t_to_connected}
    t_to_max_dist_ID      = max(t_to_dict.keys())    # need subraph that includes all times to viable neighbors
    t_to_max_time   = G4.nodes[t_to_max_dist_ID]["time_start"]  # this is end time
    t_from_time_end = G4.nodes[t_from]["time_end"]
    t_active_DT_nodes = [t_node for t_node, t_time in G.nodes(data='time') if t_from_time_end <= t_time <= t_to_max_time]
    t_from_end_node = t_segments_new[t_from][-1]
    if len(t_to_connected) > 1:  # if ther are choices, check them.
        t_isolate_each_ID = {t:[k for k in t_to_connected if t != k] for t in t_to_connected} #[1,2,3] -> {1:[2,3],2:[1,3],3:[1,2]}
        for k,(t_to,t_to_others) in enumerate(t_isolate_each_ID.items()):
            t_nodes_remove = sum([t_segments_new[t] for t in t_to_others],[]) # all competing segments
            t_active_DT_nodes_choice = [t_node for t_node in t_active_DT_nodes.copy() if t_node not in t_nodes_remove]
            g_limited_choice = G.subgraph(t_active_DT_nodes_choice) # further isolated graph
            t_to_start_node = t_segments_new[t_to][0]
            hasPath = nx.has_path(g_limited_choice, source = t_from_end_node, target = t_to_start_node)
            if hasPath:                                             # has path not involving other segments
                fin_segments_conns_DT_connected[t_from].append(t_to)
    else:   # only choice-> copy into result
        fin_segments_conns_DT_connected[t_from].append(t_to_connected[0])
    
# drop edges that have no time steps inbetween
fin_segments_conns_edges = sum([[(t_from,t_to) for t_to in t_to_all] for t_from,t_to_all in fin_segments_conns_DT_connected.items()],[])
fin_segments_conns_edges_long = [t_conn for t_conn in fin_segments_conns_edges if fin_segments_conns_refined_DT_dict[t_conn[0]][t_conn[1]] > 1]
a = 1
fin_segments_conns_edges
fin_incoming_IDs= defaultdict(list)
# Count connections for each node
for t_from, t_to in fin_segments_conns_edges:
    fin_incoming_IDs[t_to].append(t_from)

a = 1
# i think i should go forward interpolation way. since middle interpolation only works for 121 where post history is known.

# remember which segment indicies are active at each time step
lr_time_active_segments = {t:[] for t in lr_time_active_segments}
for k,t_segment in enumerate(t_segments_new):
    t_times = [a[0] for a in t_segment]
    for t in t_times:
        lr_time_active_segments[t].append(k)

t_extrapolate_sol = {}
t_extrapolate_sol_comb = {}

t_unfinished_extension = defaultdict(dict)
t_finished_extension = defaultdict(list)
# take confirmed merge starting segment and determine k and s values for better extrapolation
# added +1 step that includes target destination to check if t_from reached t_to
for t_conn in fin_segments_conns_edges_long: 
    t_from, t_to        = t_conn
    t_from_time_end = t_segments_new[t_from][-1][0]
    t_to_time_start = t_segments_new[t_to][0][0]
    if t_to_time_start - t_from_time_end == 1:  # nothing to interpolate?
        t_extrapolate_sol_comb[t_conn] = {}     # although not relevant here, i want to store t_conn for next code
        continue
    trajectory          = np.array(list(t_centroids_all[t_from].values()))
    time                = np.array(list(t_centroids_all[t_from].keys()))
    t_do_k_s_anal = False
    
    if  trajectory.shape[0] > h_analyze_p_max + h_interp_len_max:   # history is large

        h_start_point_index2 = trajectory.shape[0] - h_analyze_p_max - h_interp_len_max

        h_interp_len_max2 = h_interp_len_max
        
        t_do_k_s_anal = True

    elif trajectory.shape[0] > h_analyze_p_min + h_interp_len_min:  # history is smaller, give pro to interp length rather inspected number count
                                                                    # traj [0,1,2,3,4,5], min_p = 2 -> 4 & 5, inter_min_len = 3
        h_start_point_index2 = 0                                    # interp segments: [2,3,4 & 5], [1,2,3 & 4]
                                                                    # but have 1 elem of hist to spare: [1,2,3,4 & 5], [0,1,2,3 & 4]
        h_interp_len_max2 = trajectory.shape[0]  - h_analyze_p_min  # so inter_min_len = len(traj) - min_p = 6 - 2 = 4
        
        t_do_k_s_anal = True

    else:
        h_interp_len_max2 = trajectory.shape[0]                     # traj is very small

    if t_do_k_s_anal:                                               # test differet k and s combinations and gather inerpolation history.
        t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, t_k_s_combs, h_interp_len_max2, h_start_point_index2, debug = 0, debug_show_num_best = 2)
        t_k,t_s = t_comb_sol
    else:                                                           # use this with set k = 1, s = 5 to generate some interp history.
        t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, [(1,5)], 2, 0, debug = 0, debug_show_num_best = 2)
        t_k,t_s = t_comb_sol
        errors_sol_diff_norms_all[t_comb_sol] = {-1:5}              # dont have history to compare later..
        
    
    # extract active segments during inter-time between from_to merge. edges not included
    activeSegments = set(sum([vals for t,vals in lr_time_active_segments.items() if t_from_time_end < t < t_to_time_start],[]))
    active_segm_nodes = sum([t_segments_new[tID] for tID in activeSegments],[])

    activeNodes = [node for node in G.nodes() if t_from_time_end <= node[0] <= t_to_time_start and node not in active_segm_nodes]

    subgraph = G.subgraph(activeNodes)
    
    connected_components_unique = extract_graph_connected_components(subgraph.to_undirected(), lambda x: (x[0],x[1]))

    sol = [t_cc for t_cc in connected_components_unique if t_segments_new[t_from][-1] in t_cc]
    assert len(sol) == 1, "lr_conn_121_other_terminated_inspect inspect path relates to multiple clusters, dont expect it ever to occur"


    sol_combs = prep_combs_clusters_from_nodes(sol[0])
    sol_combs = {t_time:t_comb for t_time,t_comb in sol_combs.items() if t_to_time_start >= t_time > t_from_time_end}# changed to >=
    
    # cyclic buffer that stores N entries and pushes old ones out when new value is appended
    N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
    t_last_deltas   = list(errors_sol_diff_norms_all[t_comb_sol].values())[-N:]
    t_norm_buffer   = CircularBuffer(N,                 t_last_deltas                    )
    t_traj_buff     = CircularBuffer(h_interp_len_max2, trajectory[  -h_interp_len_max2:])
    t_time_buff     = CircularBuffer(h_interp_len_max2, time[        -h_interp_len_max2:])
    t_time_next     = time[-1] + 1
    t_extrapolate_sol[t_conn] = {}
    t_extrapolate_sol_comb[t_conn] = {}
    t_times_accumulate_resolved = [t_from_time_end]
    for t_time, t_comb in sol_combs.items():
        t_step_resolved = False
        # extrapolate traj
        t_extrap = interpolate_trajectory(t_traj_buff.get_data(), t_time_buff.get_data(), which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]

        # get possible permutations
        a = 1
        t_permutations = sum([list(itertools.combinations(t_comb, r)) for r in range(1,len(t_comb)+1)],[])
        t_centroids = []
        t_areas     = []
        for t_permutation in t_permutations:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][tID] for tID in t_permutation]))
            t_centroid, t_area = centroid_area(t_hull)
            t_centroids.append(t_centroid)
            t_areas.append(t_area)
        
        t_diffs = np.array(t_centroids).reshape(-1,2) - t_extrap
        t_diff_norms = np.linalg.norm(t_diffs, axis=1)
        t_where_min = np.argmin(t_diff_norms)
        t_sol_d_norm = t_diff_norms[t_where_min]
        t_known_norms = np.array(t_norm_buffer.get_data())
        t_mean = np.mean(t_known_norms)
        t_std = np.std(t_known_norms)
        if t_diff_norms[t_where_min] < max(t_mean, 10) + 5* t_std: #changed 5-> 10
            t_norm_buffer.append(t_sol_d_norm)
            t_traj_buff.append(t_extrap)
            t_time_buff.append(t_time_next)
            t_time_next  += 1
            t_extrapolate_sol[t_conn][t_time] = t_extrap
            t_extrapolate_sol_comb[t_conn][t_time] = t_permutations[t_where_min]
            #print(f"inter!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
            t_times_accumulate_resolved.append(t_time)
            t_step_resolved = True
        else:

            a = 1
            #print(f"stdev too much!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
            print(f'segment {t_from}:{t_segments_new[t_from][-1]}->{t_to}:{t_segments_new[t_to][0]} extended trajectory to times: {t_times_accumulate_resolved}')
            t_unfinished_extension[t_from][t_to] =  t_diff_norms[t_where_min]
            break

        if t_time == t_to_time_start:   # analysis finished at evaluating first node of t_to.
            to_node_subIDs = t_segments_new[t_to][0][1:]
            if t_step_resolved:
                t_extrapolate_sol_comb[t_conn].pop(t_time,None)
                t_finished_extension[t_from].append(t_to)
            #if its not a merge. it should be completed fully. and resolved target should be original target
            if len(fin_incoming_IDs[t_to]) == 1: 
                # but extension could lead to a wrong target
                if len(set(t_permutations[t_where_min]) & set(to_node_subIDs)) == 0:         
                    print(f'segment {t_from}:{t_segments_new[t_from][-1]}->{t_to}:{t_segments_new[t_to][0]} (not a merge) extended trajectory to times: {t_times_accumulate_resolved}. Wrong target! Removing solution!')
                    t_extrapolate_sol_comb[t_conn] = {}
                    t_finished_extension[t_from].remove(t_to)
                elif set(t_permutations[t_where_min]) == set(to_node_subIDs):
                    print(f'segment {t_from}:{t_segments_new[t_from][-1]}->{t_to}:{t_segments_new[t_to][0]} (not a merge) extended trajectory to times: {t_times_accumulate_resolved}. Recovered whole inter-segment!')
                else:
                    print(f'segment {t_from}:{t_segments_new[t_from][-1]}->{t_to}:{t_segments_new[t_to][0]} error!') 
            else: # its a merge
                print(f'segment {t_from}:{t_segments_new[t_from][-1]}->{t_to}:{t_segments_new[t_to][0]} (a merge) extended trajectory to times: {t_times_accumulate_resolved}. Recovered whole inter-segment!')


    a = 1
# some branches to merge might not be finished and failed early. but one of conns is closer to target.
# that closest must inherit the extension.
for t_from,t_to_all in t_unfinished_extension.items():
    t_sol = min(t_to_all, key = t_to_all.get)   # get ID of smallest delta.
    t_sols_other = [tID for tID in t_to_all if tID != t_sol]
    for t_to in t_sols_other:                   # remove other paths.
        t_conn = (t_from,t_to)
        t_extrapolate_sol_comb[t_conn] = {}

assert all([1 if len(t) == 1 else 0 for k,t in t_finished_extension.items()]), "multiple solution from one t_from found!"
a = 1
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===========
# check whether  merges share nodes?
# =============
#lr_conn_merges_good = [] # may be redo to dict of tID?
lr_conn_merges_good = {tID:[] for tID in lr_conn_merges_to_nodes}
#lr_conn_merges_good2 = {tID:[] for tID in lr_conn_merges_to_nodes}
if 1 == 1:
    node_properties = defaultdict(list)
    # to find doverlapping branches add owners to each node. multiple owners = contested.
    for key, dic in t_extrapolate_sol_comb.items():
        t_nodes =  [(key, value) for key, values in dic.items() for value in values]
        for t_node in t_nodes:
            node_properties[t_node].extend([key])
    t_duplicates = {t_node: t_branches for t_node,t_branches in node_properties.items() if len(t_branches) > 1}
    t_all_problematic_conns = list(set(sum(list(t_duplicates.values()),[])))
    t_all_problematic_conns_to = [a[1] for a in t_all_problematic_conns]
    for tID in lr_conn_merges_to_nodes:
        if tID not in t_all_problematic_conns_to:
            t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
            lr_conn_merges_good[tID] += t_conns_relevant
            #lr_conn_merges_good2[tID] += t_conns_relevant

    variants_possible = []
    if len(t_duplicates) > 0:
        # each contested node is an option. each choice of owner produces a different configuration branch
        variants = [] # generate these options for each contested node
        for t,(key,values) in enumerate(t_duplicates.items()):
            sd  = list(itertools.product([key],values))
            variants.append(sd)
        # generate all possible evoltions via product permutation: choice_1 -> choice_2 -> ...
        variants_all = list(itertools.product(*variants))
        # check if other paths, which will have at least one remaining node after deletion.
        for t_choice_evol in variants_all:
            t_skip = False
            for t_node,t_conn in t_choice_evol:
                t_time, *t_subIDs = t_node
                t_delete_conns = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
                for t_delete_conn in t_delete_conns:
                    if len(t_extrapolate_sol_comb[t_delete_conn][t_time]) == 1:
                        t_skip = True
            if not t_skip:
                variants_possible.append(t_choice_evol)
    if len(variants_possible) == 1: # if there is only one solution by default take it as answer
        t_choice_evol = variants_possible[0]
        for t_node,t_conn in t_choice_evol:
            tID = t_conn[1]
            t_time, *t_subIDs = t_node
            t_delete_conns = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
            for t_delete_conn in t_delete_conns:
                t_temp = t_extrapolate_sol_comb[t_delete_conn][t_time]
                t_temp = [t for t in t_temp if t not in t_subIDs]
                t_extrapolate_sol_comb[t_delete_conn][t_time] = t_temp
            t_conns_relevant = [t_c for t_c in t_extrapolate_sol_comb if t_c[1] == tID]
            lr_conn_merges_good[tID] += t_conns_relevant
            a = 1

print("# process connections that extend partially and dont 121 segments. COPY (of a copy?)")
if 1 == 1: 
    for t_conn in t_extrapolate_sol_comb:
        t_from, t_to = t_conn
        
        t_combs = t_extrapolate_sol_comb[t_conn]

        if len(t_combs) == 0: continue

        t_nodes_all = []                                            # solo nodes will be deleted
        t_nodes_new = []                                            # new composite nodes are formed

        for t_time,t_subIDs in t_combs.items():

            for t_subID in t_subIDs:                                # single object nodes
                t_nodes_all.append((t_time,t_subID))

            t_nodes_new.append(tuple([t_time] + list(t_subIDs)))    # cluster object nodes
        
        t_nodes_all += [t_segments_new[t_from][-1]]                 # (*) last old may have multiple connections not
                                                                    # covered by best choices in sol-n, so they are will stay.
        # find end edges to reconnect into graph
        # if last node is composite of solo nodes, each might have its edges
        t_time_end, *t_node_subIDs =  t_nodes_new[-1]               # grab subIDs and reconver solo nodes. i do this because
        t_node_last_solo_IDS = [(t_time_end,t_subID) for t_subID in t_node_subIDs] #  its easy to pick [-1], not lookup last time.

        #f = lambda x: x[0]
        t_node_last_solo_IDS_to = list(set(sum([list(G.successors(t_node)) for t_node in t_node_last_solo_IDS],[])))
        #t_node_last_solo_IDS_to = list(set(sum([extractNeighborsNext(G, t_node, f) for t_node in t_node_last_solo_IDS],[])))

        t_edges_next_new = [(t_nodes_new[-1] , t_node) for t_node in t_node_last_solo_IDS_to] # composite to old next neighbors

        G.remove_nodes_from(t_nodes_all)                            # remove all solo nodes and replace them with composite
            
        t_nodes_new_sides = t_segments_new[t_from][-2:] + t_nodes_new#  take nodes from -2 to end of segment, why -2 see (*)

        t_edges_sequence = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
            
        G.add_edges_from(t_edges_sequence)                          # from sequence of nodes create edges -> auto nodes
        G.add_edges_from(t_edges_next_new)                          # add future edges to solo nodes.

        t_segments_new[t_from] += t_nodes_new                       # append resolved to segments.

        # add calc

        for t_time,*t_subIDs in t_nodes_new:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
            t_node = tuple([t_time] + t_subIDs)
            t_segments_121_centroids[t_from][t_node]   = t_centroid
            t_segments_121_areas[    t_from][t_node]   = t_area
            t_segments_121_mom_z[    t_from][t_node]   = t_mom_z
# ===============================================================================================
# confirm merges and splits here in the future. convex hulls and their area conservation
# ===============================================================================================
# REMARK: merge (fake) might happen via proximity, but its still inside merge/split connections.
# REMARK: for example one of close by bubbles drops out, and it is considered a merge.
# REMARK: real merge/split exhibits center of mass trajectory/ area continuity
# REMARK: failed merge should drop false connections

#lr_conn_edges_merges2 = lr_reindex_masters(lr_C5_condensed_connections_relations, lr_conn_edges_merges)

# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if 1 == 1:

    binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
    imgs = [convertGray2RGB(binarizedMaskArr[k].copy()) for k in range(binarizedMaskArr.shape[0])]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7; thickness = 4;
    for n, case in tqdm(enumerate(t_segments_new)):
        
        
        case    = sorted(case, key=lambda x: x[0])
        t_traj = {t_time:np.zeros(2,int) for t_time,*t in case}
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            t_hull = cv2.convexHull(np.vstack([g0_contours[time][tID] for tID in subIDs]))
            t_centroid, t_area = centroid_area(t_hull)
            t_traj[time] = t_centroid.astype(int)

        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                cv2.drawContours(  imgs[time],   g0_contours[time], subID, cyclicColor(n), 2)

            useTimes = [t for t in t_traj.keys() if time >= t >= time - 10] 
            pts = np.array([t_traj[t] for t in useTimes]).reshape(-1, 1, 2)
            
            cv2.polylines(  imgs[time], [pts] ,0, (255,255,255), 3)
            cv2.polylines(  imgs[time], [pts] ,0, cyclicColor(n), 2)
            [cv2.circle(    imgs[time], tuple(p), 3, cyclicColor(n), -1) for [p] in pts]

            x,y,w,h = cv2.boundingRect(np.vstack([g0_contours[time][ID] for ID in subIDs]))
            #x,y,w,h = g0_bigBoundingRect2[time][subCase]
            #x,y,w,h = g0_bigBoundingRect[time][ID]
            #cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
            #x,y= g0_contours[time]
            [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                startPos2 = g0_contours[time][subID][-30][0] 
                [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        

    for k,img in enumerate(imgs):
        folder = r"./post_tests/testImgs/"
        fileName = f"{str(k).zfill(4)}.png"
        cv2.imwrite(os.path.join(folder,fileName) ,img)
        #cv2.imshow('a',imgs[time])
a = 1
# for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if 1 == -1000:
    if 1 == -1:
        #usefulPoints = startTime

        #interval    = min(interpolatinIntevalLength,usefulPoints)

        #startPoint  = max(0,startTime-interval) 
        #endPoint    = min(startTime,startPoint + interval)
        ##
        #x,y = trajectory[startPoint:endPoint].T
        #t0 = np.arange(startPoint,endPoint,1)

        # plot check solutions
        for t_sol in set(sum(list(t_sols_c.values()) + list(t_sols_c_i.values())  + list(t_sols_a.values()) + list(t_sols_m.values()),[]) ):
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            t_traj = []
            for t,t_perms in  enumerate(lr_inspect_permutation_cases[t_conn][t_sol]):
                t_time  = lr_inspect_permutation_times[t_conn][t]
                #t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t_subID] for t_subID in t_perms]))
                #axes.plot(*t_hull.reshape(-1,2).T, '-',  c='black', linewidth = 0.3)
                #for tID in t_perms:
                #    axes.plot(*g0_contours[t_time][tID].reshape(-1,2).T, '-',  c=np.array(cyclicColor(t))/255, linewidth = 0.5)
                t_traj.append(lr_inspect_permutation_centroids_precomputed[t_conn][t_time][t_perms])
            t_traj = np.array(t_traj)
            axes.plot(*t_traj.T, '-o')
            #axes.plot(*t_traj_next.T, '-o',  c='black')
            #axes.plot(x0,y0, '-o')
        
            #axes.plot(*IEpolation, linestyle='dotted', c='orange')
            axes.set_title(t_sol)
            axes.set_aspect('equal')
            fig.show()

        a = 1
        # ===============================================================================================
        # ========== INTERPOLATE PATHS BETWEEN SEGMENTS AND COMPARE WITH INTER-SEGMENT SOLUTION =========
        # ===============================================================================================
        # REMARK: least path and area devs might not agree, test remaining choices with interpolation




        def interpolateMiddle(t_conn,t_sols_c,t_sols_a,t_segments_121_centroids,t_all_traj, lr_permutation_times, segments2, histLen = 5, s = 15, debug = 0):
            t_from,t_to = t_conn
            t_possible_sols = sorted(list(set(t_sols_c[t_conn] + t_sols_a[t_conn])))
            t_trajectories = {tID:t_all_traj[t_conn][tID] for tID in t_possible_sols}
            t_hist_prev     = segments2[t_from][-histLen:]
            t_hist_next     = segments2[t_to][:histLen]
            t_traj_prev     = np.array([t_segments_121_centroids[t_from][t_node] for t_node in t_hist_prev])
            t_traj_next     = np.array([t_segments_121_centroids[  t_to][t_node] for t_node in t_hist_next])
            t_times_prev    = [t_node[0] for t_node in t_hist_prev]
            t_times_next    = [t_node[0] for t_node in t_hist_next]
            t_traj_concat   = np.concatenate((t_traj_prev,t_traj_next))
            t_times         = t_times_prev + t_times_next
            x, y            = t_traj_concat.T
            spline, _       = interpolate.splprep([x, y], u=t_times, s=s,k=1)

            t2              = np.arange(t_times[0],t_times[-1],0.1)
            IEpolation      = np.array(interpolate.splev(t2, spline,ext=0))
            t_missing       = lr_permutation_times[t_conn][1:-1]
            t_interp_missing= np.array(interpolate.splev(t_missing, spline,ext=0)).T   # notice transpose sempai :3

            t_sol_diffs     = {}
            for t_sol, t_traj in t_trajectories.items():
                t_diffs = t_interp_missing- t_traj[1:-1]
                t_sol_diffs[t_sol] = [np.linalg.norm(t_diff) for t_diff in t_diffs]
        
            if debug:
                fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
                axes.plot(*t_traj_prev.T, '-o', c= 'black')
                axes.plot(*t_traj_next.T, '-o', c= 'black')
        
                axes.plot(*IEpolation, linestyle='dotted', c='orange')
    

                for t_sol, t_traj in t_trajectories.items():
                    axes.plot(*t_traj.T, '-o', label = t_sol)
                axes.set_title(t_conn)
                axes.set_aspect('equal')
                axes.legend(prop={'size': 6})
                plt.show()
            return t_sol_diffs

        # tested this second, first below cssssssssssssss
        sol = []
        for t_conn in t_sols_c:#[(18,19)]
            aa = interpolateMiddle(t_conn,t_sols_c,t_sols_a,t_segments_121_centroids,t_all_traj,lr_permutation_times, segments2, debug = 0)
            sol.append(aa)
        a = 1
    

        for t_conn in t_sols_c:#[(18,19)]
            t_from,t_to = t_conn
            t_possible_sols = sorted(list(set(t_sols_c[t_conn] + t_sols_a[t_conn])))
            t_trajectories = {tID:t_all_traj[t_conn][tID] for tID in t_possible_sols}
            t_hist_prev     = segments2[t_from][-4:-1]
            t_hist_next     = segments2[t_to][1:4]
            t_traj_prev     = np.array([t_segments_121_centroids[t_from][t_node] for t_node in t_hist_prev])
            t_traj_next     = np.array([t_segments_121_centroids[  t_to][t_node] for t_node in t_hist_next])
            t_times_prev    = [t_node[0] for t_node in t_hist_prev]
            t_times_next    = [t_node[0] for t_node in t_hist_next]
            #t_mean_displ = {tID:np.diff(t_traj,axis = 0)       for tID, t_traj in t_trajectories.items()}
            t_mean_displ = {tID:np.mean(np.diff(t_traj,axis = 0) ,axis = 0) for tID, t_traj in t_trajectories.items()}
            for t_sol, t_traj in t_trajectories.items():
                #t_diag = t_mean_displ[t_sol]
                #t_diag_inv = 1/t_diag
                #t_scale_inv = np.diag(t_diag)
                #t_scale = np.diag(t_diag_inv)
                #t_traj_scaled = np.dot(t_traj,t_scale)#t_scale @ t_traj.T #np.matmul(t_scale,)
                #t = np.diff(t_traj_scaled,axis = 0)
                #t_traj_2 = np.dot(t_traj_scaled,t_scale_inv)

                #x,y = t_traj_scaled.T
                x0,y0 = t_traj.T
                t_concat = np.concatenate((t_traj_prev,t_traj,t_traj_next))
                x,y = t_concat.T
                t0 = lr_permutation_times[t_conn]
                t_times = t_times_prev + list(t0) + t_times_next
                spline, _ = interpolate.splprep([x, y], u=t_times, s=10,k=1)

                #t2 = np.arange(t0[0],t0[-1],0.1)
                t2 = np.arange(t_times[0],t_times[-1],0.1)
                IEpolation = np.array(interpolate.splev(t2, spline,ext=0))
    
                fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
        
                for t,t_perms in  enumerate(lr_permutation_cases[t_conn][t_sol]):
                    t_time  = t0[t]
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t_subID] for t_subID in t_perms]))
                    axes.plot(*t_hull.reshape(-1,2).T, '-',  c='black', linewidth = 0.3)
                    for tID in t_perms:
                        axes.plot(*g0_contours[t_time][tID].reshape(-1,2).T, '-',  c=np.array(cyclicColor(t))/255, linewidth = 0.5)
                axes.plot(*t_traj_prev.T, '-o',  c='black')
                axes.plot(*t_traj_next.T, '-o',  c='black')
                axes.plot(x0,y0, '-o')
        
                axes.plot(*IEpolation, linestyle='dotted', c='orange')
                axes.set_title(t_conn)
                axes.set_aspect('equal')
                fig.show()
            a = 1


        for t_conn, test in lr_contour_combs.items():
            yoo     = {t:[] for t in test}
            for t,subIDs in test.items():
                s = sum([list(itertools.combinations(subIDs, r)) for r in range(1,len(subIDs)+1)],[])
                yoo[t] += s
                a = 1
            yoo2 = {t:[] for t in test}
            for t,permList in yoo.items(): 
                for perm in permList:
                    hull = cv2.convexHull(np.vstack([g0_contours[t][c] for c in perm]))
                    yoo2[t].append(cv2.contourArea(hull))
                    #yoo2[t].append(cv2.boundingRect(np.vstack([g0_contours[t][c] for c in perm])) )

            #rectAreas = {t:[] for t in test}
            #maxArea = 0
            #minArea = 9999999999
            #for t,rectList in yoo2.items(): 
            #    for x,y,w,h in rectList:
            #        area = w*h
            #        rectAreas[t].append(area)
            #        maxArea = max(maxArea,area)
            #        minArea = min(minArea,area)
            rectAreas = yoo2
            minTime,maxTime = min(rectAreas), max(rectAreas)

            soloTimes = [a for a,b in rectAreas.items() if len(b) == 1]
            missingTimes = [a for a,b in rectAreas.items() if len(b) > 1]
            from scipy.interpolate import splev, splrep
            x = soloTimes
            y0 = np.array([b[0] for a,b in  rectAreas.items() if a in soloTimes])
            minArea,maxArea = min(y0), max(y0)
            K = maxTime - minTime
            y = (y0 - minArea) * (K / (maxArea - minArea))
            spl = splrep(soloTimes, y, k = 1, s = 0.7)

            x2 = np.linspace(min(rectAreas), max(rectAreas), 30)
            y2 = splev(x2, spl)
            y2 = y2/(K / (maxArea - minArea)) + minArea
            y = y/(K / (maxArea - minArea)) + minArea
            fig = plt.figure()
            for t in missingTimes:
                for rA in rectAreas[t]:
                    plt.scatter([t],[rA], c='b')
            plt.plot(x, y, 'o', x2, y2)
            fig.suptitle(lr_multi_conn_intr[tID][0])
            #plt.show()

        a = 1
        # Draw the graph with constrained x-positions and automatic y-positions
        #nx.draw(G, pos, with_labels=True)

        ##drawH(G, paths, node_positions)

        # check if extracted intervals are "clean", dont contain merge
        # clean means there are nodes attached to all available paths
        # => check neighbors of all nodes in path and see if they are their own neigbors
        # this test is too strict, but allows to isolate pseudo split-merges 
        t_sols = []
        for t_conn in lr_trusted_interval_test_prio:
            a = lr_paths_segm2[t_conn]
            t_from,t_to = t_conn
            t_node_from_last   = segments2[t_from][-1]
            t_node_to_first    = segments2[t_to][0]
            t_neighbors_from_next       = extractNeighborsNext(     G, t_node_from_last,    lambda x: x[0])
            t_neighbors_to_previous     = extractNeighborsPrevious( G, t_node_to_first,     lambda x: x[0])

            t_all_path_nodes = sorted(set(sum(lr_paths_segm[t_conn],[])),key=lambda x: x[0])
    
            t_all_path_nodes = t_all_path_nodes
            t_all_path_neigbors = []
            for t_node in t_all_path_nodes[1:-1]:
                t_all_path_neigbors.append(list(G.neighbors(t_node)))

            t_all_path_neigbors.append(t_neighbors_from_next)
            t_all_path_neigbors.append(t_neighbors_to_previous)

            t_all_path_neigbors = sorted(set(sum(t_all_path_neigbors,[])),key=lambda x: x[0])
            #t_all_path_neigbors = [t_node for t_node in t_all_path_neigbors if t_node not in [t_node_from_last,t_node_to_first]]
            if t_all_path_neigbors != t_all_path_nodes: continue
            t_sols.append(t_conn)

            #t_nodes_all = sum(t_hist_pre.values(),[]) + sum(t_hist_post.values(),[]) +sum(t_paths_choices,[])
            #t_nodes_all = sorted(set(t_nodes_all))
            #t_times_all = [a[0] for a in t_nodes_all]
            #t_times_cIDs = {t:[] for t in t_times_all}
            #for t,cID in t_nodes_all:
            #    t_times_cIDs[t].append(cID)
            #for t in t_times_cIDs:
            #    t_times_cIDs[t] = list(sorted(np.unique(t_times_cIDs[t])))
        a = 1
        #drawH(G, paths, node_positions)
        # ===============================================================================================
        # ===============================================================================================
        # === find split-merge (possibly fake) events; detect split-merges that look like real merges ===
        # ===============================================================================================
        # ===============================================================================================

        # since nodes and edges represent spacial overlap of cluster elements at each time paths between two nodes
        # should (?) contain all elements for permutations for bubble reconstruction from partial reflections

        # !!!!!!!!!!!!!!====================<<<<<<<<<<<<<<<
        # best case scenario of a path is overlap between single contour bubbles.
        # in that case solution is shortest trajectory :)
        # might want to check for area befor and after, in case of merge. overall, i have to do permutations.
        # elements that begin from nothing are problematic. in case split-merge, i should include them into paths, but in reverse.
        # same with split-merge that happend 2+ steps. means bubble splits into two segments and is reconstructed.
        # but also might split into one segment, and other starts from nothing. so merge them in reverse path.

        # 1) check if multiple segments are merged into 1. might be merge, might be split-merge fake or not. area will help.
        # take endpoints of all paths
        lr_paths_endpoints = {**{key:a[0][-1] for key,a in lr_paths_segm.items()},
                              **{key:a[0][-1] for key,a in lr_paths_other.items()}}

        # find common endpoints. inverse dictionary = endpoint:connections
        rev_multidict = {}
        for key, value in lr_paths_endpoints.items():
            rev_multidict.setdefault(value, set()).add(key)
        # if endpoint is associated with multiple connections, extract these connections
        lr_multi_conn = [list(a) for a in rev_multidict.values() if len(a)>1]
        lr_multi_conn_all = sum(lr_multi_conn,[])
        # decide which merge branch is main. expore every merge path
        # path length is not correct metric, since it may be continuation via merge
        lr_multi_conn_win = []
        for t_merges_options in lr_multi_conn:
            # grab time at which segments merge
            t_merge_time = segments2[int(t_merges_options[0][1])][0][0]
            # initialize times for paths which will be compared for 'path length/age' and if set time is legit or intermediate
            t_earliest_time     = {t_ID:0 for t_ID in t_merges_options}
            t_terminated_time   = {t_ID:0 for t_ID in t_merges_options}
            for t_path in t_merges_options:
                # t_path is ID in segments2, take its starting node.
                # lr_paths_other has no history (no prior neighbors)
                if type(t_path[0]) == str:
                    t_earliest_time[t_path] = t_merge_time - 1
                    t_terminated_time[t_path] = 1
                    continue
                t_nodes_path    = segments2[t_path[0]]                       

                t_node_first    = t_nodes_path[0]
                t_node_last     = t_nodes_path[-1]
                t_prev_neighbors = [node for node in list(G.neighbors(t_node_first)) if node[0] < t_node_first[0]]
                if len(t_prev_neighbors) == 0:
                    t_earliest_time[t_path] = t_node_first[0]
                    t_terminated_time[t_path] = 1
            # check if we can drop terminated paths. if path is terminated and time is bigger than any unterminated, drop it.
            for t_path in t_merges_options:
                t_other_times = [t for tID,t in t_earliest_time.items() if tID != t_path]
                if t_terminated_time[t_path] == 1 and t_earliest_time[t_path] > min(t_other_times):
                    t_earliest_time.pop(t_path,None)
                    t_terminated_time.pop(t_path,None)

            t_IDs_left = list(t_earliest_time.keys())
            if len(t_IDs_left) == 1: lr_multi_conn_win.append(t_IDs_left[0])
            else: assert 1 == -1, "split-merge. need to develop dominant branch further"

        # ===============================================================================================
        # ===============================================================================================
        # == find INTeRmediate interval paths, short PREv and POST history. deal with merge hist after ==
        # ===============================================================================================
        # ===============================================================================================
        lr_multi_conn_pre   = {conn:{} for conn in lr_paths_segm.keys()}
        lr_multi_conn_intr  = {conn:{} for conn in lr_paths_segm.keys()} 
        lr_multi_conn_post  = {conn:{} for conn in lr_paths_segm.keys()} 
        for t_conn, t_paths_nodes in lr_paths_segm.items():
            tID_from = t_conn[0]
            tID_to  = t_conn[1]
            # if multi merge
            t_is_part_of_merge = True if t_conn in lr_multi_conn_all else False
            t_is_part_of_merge_and_main = True if t_is_part_of_merge and t_conn in lr_multi_conn_win else False
            t_is_followed_by_merge = True if tID_to in [a[0] for a in lr_multi_conn_all] else False
            t_other_paths_nodes = []
            t_other_path_conn   = []
            if t_is_part_of_merge_and_main:
                # extract other (pseudo) merging segments. other paths only.
                t_other_path_conn = [a for a in lr_multi_conn if t_conn in a]
                t_other_path_conn = [[a for a in b if a != t_conn] for b in t_other_path_conn]
                #[t_other_path_conn[k].remove(t_conn) for k in range(len(t_other_path_conn))]
                t_other_path_conn = sum(t_other_path_conn,[])
                # preprend prev history of segment before intermediate segment
                #t_other_path_pre = {ID:[] for ID in t_other_path_conn}
                #for ID in t_other_path_conn:
                #    if type(ID[0]) != str:
                #        t_other_path_pre[ID] += segments2[ID[0]][:-1]
                # some mumbo-jumbo with string keys to differentiate lr_paths_other from lr_paths_segm
                t_other_paths_nodes = []
                for ID in t_other_path_conn:
                    if type(ID[0]) == str:
                        for t_path in lr_paths_other[ID]:
                            t_other_paths_nodes.append(t_path)  
                    else:
                        for t_path in lr_paths_segm[ID]:
                            t_other_paths_nodes.append(t_path)
                    #if type(ID[0]) == str:
                    #    for t_path in lr_paths_other[ID]:
                    #        t_other_paths_nodes.append(t_other_path_pre[ID]+t_path)  
                    #else:
                    #    for t_path in lr_paths_segm[ID]:
                    #        t_other_paths_nodes.append(t_other_path_pre[ID]+t_path)
                a = 1            
                #t_other_paths_nodes = [t_other_path_pre[ID] + lr_paths_other[ID] 
                #                       if type(ID[0]) == str 
                #                       else lr_paths_segm[ID] for ID in t_other_path_conn]
                #t_other_paths_nodes = sum(t_other_paths_nodes,[])
                #t_join_paths = t_paths_nodes + t_other_paths_nodes
                # there is no prev history for lr_paths_other, lr_paths_segm has at least 2 nodes
            elif t_is_part_of_merge:
                continue
            else:
                t_other_paths_nodes = []
            # app- and pre- pend path segments to intermediate area. 
            # case where split-merge is folled by disconnected split-merge (on of elements is not connected via overlay)
            # is problematic. 
            # >>>>>>>>>>>>>>>>!!!!!!!!!!!!!!!!!!!!!!!!!<<<<<<<<<<<<<<
            # i should split this code into two. one stablishes intermediate part
            # and other can relate intermediate parts together, if needed.

            # ====================
            # store history prior to intermediate segment. solo bubble = 1 contour = last N steps
            # pseudo merge = multiple branches => include branches and take N steps prior
            a = 1
            t_og_max_t =  segments2[tID_from][-1][0]
            if not t_is_part_of_merge:
                lr_multi_conn_pre[t_conn][tID_from] = segments2[tID_from][-4:-1]
            else:
                # find minimal time from othat branches
                t_other_min_t = min([a[0] for a in sum(t_other_paths_nodes,[])])
                # extract OG history only 3 steps prior to first branch (timewise) <<< 3 or so! idk, have to test
                lr_multi_conn_pre[t_conn][tID_from] = [a for a in segments2[tID_from] if t_og_max_t> a[0] > t_other_min_t - 4]
                t_other_IDs = [a[0] for a in t_other_path_conn]
                for tID in t_other_IDs:
                    if type(tID) != str:
                        lr_multi_conn_pre[t_conn][tID] = segments2[tID]
            # store history post intermediate segment.     solo bubble = 1 contour = next N steps
            # if followed by merge, ideally take solo N steps before merge. but this info not avalable yet. do it after
            if not t_is_followed_by_merge:
                lr_multi_conn_post[t_conn][tID_to]  = segments2[tID_to][1:4]

            t_join_paths            = t_paths_nodes + t_other_paths_nodes
            # store all paths in intermediate interval
            lr_multi_conn_intr[t_conn]   = t_paths_nodes + t_other_paths_nodes 
            # collect all viable contour IDs from intermediate time
            #t_all_nodes = sum(t_join_paths,[])
            #t_all_times = list(sorted(np.unique([a[0] for a in t_all_nodes])))
            #t_times_cIDs = {t:[] for t in t_all_times}
            #for t,cID in t_all_nodes:
            #    t_times_cIDs[t].append(cID)
            #for t in t_times_cIDs:
            #    t_times_cIDs[t] = list(sorted(np.unique(t_times_cIDs[t])))
            #lr_multi_conn_choices[t_conn] = t_times_cIDs
            # test intermediate contour combinations
            # 
        #lr_multi_conn_all and t_conn in lr_multi_conn_win

        # ===============================================================================================
        # ===============================================================================================
        # ========================== deal with POST history if its a merge ==============================
        # ===============================================================================================
        # ===============================================================================================
        # isolate easy cases where split-merge has history on both sides
        t_all_merging = [a[0] for a in lr_multi_conn_win]
        for t_conn, t_paths_choices in lr_multi_conn_post.items():
            t_end_ID = t_conn[1]
            if len(t_paths_choices) == 0 and t_end_ID in t_all_merging:
                t_next_merge_all = [a for a in lr_multi_conn_win if a[0] == t_end_ID]
                assert len(t_next_merge_all)<2, "post history-> multi merge. something i did not account for"
                t_conn_next_merge    = t_next_merge_all[0]
                t_og_min_t      = segments2[t_end_ID][0][0]
                # pick min branches time: take all branches except main, extract first node, and extract from it time. pick min
                t_other_min_t   = min([a[0][0] for ID,a in lr_multi_conn_pre[t_conn_next_merge].items() if ID != t_end_ID])

                # end history at: if there is time before merge take 3 (or so) steps before it.
                # it its earlier, squeeze interval. if merge is straight after, post history will be zero.
                t_max_time =  min(t_og_min_t + 4, t_other_min_t )
                lr_multi_conn_post[t_conn][t_end_ID] = [a for a in segments2[t_end_ID] if t_og_min_t < a[0] < t_max_time]
        #drawH(G, paths, node_positions)
        # ===============================================================================================
        # ===============================================================================================
        # ============== form combinations of elements from PRE-INTR-Post connectedness data ============
        # ===============================================================================================
        # ===============================================================================================
        lr_multi_conn_choices = {conn:{} for conn in lr_paths_segm.keys()}

        for t_conn, t_paths_choices in lr_multi_conn_intr.items():
            t_hist_pre  = lr_multi_conn_pre[t_conn]
            t_hist_post = lr_multi_conn_post[t_conn]
            t_nodes_all = sum(t_hist_pre.values(),[]) + sum(t_hist_post.values(),[]) +sum(t_paths_choices,[])
            t_nodes_all = sorted(set(t_nodes_all))
            t_times_all = [a[0] for a in t_nodes_all]
            t_times_cIDs = {t:[] for t in t_times_all}
            for t,cID in t_nodes_all:
                t_times_cIDs[t].append(cID)
            for t in t_times_cIDs:
                t_times_cIDs[t] = list(sorted(np.unique(t_times_cIDs[t])))
            lr_multi_conn_choices[t_conn] = t_times_cIDs

        for tID,test in lr_multi_conn_choices.items():
            if len(test)==0: continue
            #test    = {325: [2], 326: [1], 327: [3], 328: [2], 329: [1, 3], 330: [2], 331: [1], 332: [1], 333: [1]}
            yoo     = {t:[] for t in test}
            for t,subIDs in test.items():
                s = sum([list(itertools.combinations(subIDs, r)) for r in range(1,len(subIDs)+1)],[])
                yoo[t] += s
                a = 1
            yoo2 = {t:[] for t in test}
            for t,permList in yoo.items(): 
                for perm in permList:
                    hull = cv2.convexHull(np.vstack([g0_contours[t][c] for c in perm]))
                    yoo2[t].append(cv2.contourArea(hull))
                    #yoo2[t].append(cv2.boundingRect(np.vstack([g0_contours[t][c] for c in perm])) )

            #rectAreas = {t:[] for t in test}
            #maxArea = 0
            #minArea = 9999999999
            #for t,rectList in yoo2.items(): 
            #    for x,y,w,h in rectList:
            #        area = w*h
            #        rectAreas[t].append(area)
            #        maxArea = max(maxArea,area)
            #        minArea = min(minArea,area)
            rectAreas = yoo2
            minTime,maxTime = min(rectAreas), max(rectAreas)

            soloTimes = [a for a,b in rectAreas.items() if len(b) == 1]
            missingTimes = [a for a,b in rectAreas.items() if len(b) > 1]
            from scipy.interpolate import splev, splrep
            x = soloTimes
            y0 = np.array([b[0] for a,b in  rectAreas.items() if a in soloTimes])
            minArea,maxArea = min(y0), max(y0)
            K = maxTime - minTime
            y = (y0 - minArea) * (K / (maxArea - minArea))
            spl = splrep(soloTimes, y, k = 1, s = 0.7)

            x2 = np.linspace(min(rectAreas), max(rectAreas), 30)
            y2 = splev(x2, spl)
            y2 = y2/(K / (maxArea - minArea)) + minArea
            y = y/(K / (maxArea - minArea)) + minArea
            fig = plt.figure()
            for t in missingTimes:
                for rA in rectAreas[t]:
                    plt.scatter([t],[rA], c='b')
            plt.plot(x, y, 'o', x2, y2)
            fig.suptitle(lr_multi_conn_intr[tID][0])
            #plt.show()

        # >>>>>>>>>>>>> !!!!!!!!!!!!!!!!!!!! <<<<<<<<<<<<< CONCLUSIONS
        # long trends can be trusted: e.g long solo- split-merge - long solo
        # transition to short segments may be misleading e.g split into two parts perist - area is halfed at split
        # fast split-merge kind of reduces trust, which is related to segment length. but it should not.
        # ^ may be longterm trends should be analyzed first. idk what to do about real merges then.

            a = 1



        # usefulPoints = startTime
        ## interval gets capped at interpolatinIntevalLength if there are more points
        #interval    = min(interpolatinIntevalLength,usefulPoints)
        ## maybe max not needed, since interval is capped
        #startPoint  = max(0,startTime-interval) 
        #endPoint    = min(startTime,startPoint + interval) # not sure about interval or interpolatinIntevalLength
        ##
        #x,y = trajectory[startPoint:endPoint].T
        #t0 = np.arange(startPoint,endPoint,1)
        ## interpolate and exterpolate to t2 this window
        #spline, _ = interpolate.splprep([x, y], u=t0, s=10000,k=1)
        #t2 = np.arange(startPoint,endPoint+1,1)
        #IEpolation = np.array(interpolate.splev(t2, spline,ext=0))
        # >>>>>>>>> !!!!!!!! <<<<<<< most works. EXCEPT split into isolated node. segment merge into non-isolated nodes


        #TODO: do a reconstruction via area (maybe trajectory) of BRs using lr_multi_conn_choices


        #drawH(G, paths, node_positions)

        # analyze fake split-merge caused by partial reflections. during period of time bubble is represented by reflections on its opposite side
        # which might appear as a split. 

        # if bubble is well behavade (good segment), it will be solo contour, then split and merge in 1 or more steps.

        # find end points of rough good segments

        a = 1
    # ======================
    if 1 == -11:
        binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
        imgs = [convertGray2RGB(binarizedMaskArr[k].copy()) for k in range(binarizedMaskArr.shape[0])]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7; thickness = 4;
        for n, case in tqdm(enumerate(t_segments_new)):
            case    = sorted(case, key=lambda x: x[0])
            for k,subCase in enumerate(case):
                time,*subIDs = subCase
                for subID in subIDs:
                    cv2.drawContours(  imgs[time],   g0_contours[time], subID, cyclicColor(n), 2)
                x,y,w,h = cv2.boundingRect(np.vstack([g0_contours[time][ID] for ID in subIDs]))
                #x,y,w,h = lessRoughBRs[time][subCase]
                #x,y,w,h = g0_bigBoundingRect[time][ID]
                #cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
                [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
            for k,subCase in enumerate(case):
                time,*subIDs = subCase
                for subID in subIDs:
                    startPos2 = g0_contours[time][subID][-30][0] 
                    [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        
        #cv2.imshow('a',imgs[time])
        for k,img in enumerate(imgs):
            if k in activeTimes:
                folder = r"./post_tests/testImgs2/"
                fileName = f"{str(k).zfill(4)}.png"
                cv2.imwrite(os.path.join(folder,fileName) ,img)
                #cv2.imshow('a',img)

    a = 1
    #cv2.imshow('a', blank0)
    def centroid_area(contour):
        m = cv2.moments(contour)
        area = int(m['m00'])
        cx0, cy0 = m['m10'], m['m01']
        centroid = np.array([cx0,cy0])/area
        return  centroid, area

    from scipy import interpolate

    def exterpTest(trajectory,startTime,interpolatinIntevalLength, debug = 0 ,axes = 0, title = 'title', aspect = 'equal'):
        usefulPoints = startTime
        # interval gets capped at interpolatinIntevalLength if there are more points
        interval    = min(interpolatinIntevalLength,usefulPoints)
        # maybe max not needed, since interval is capped
        startPoint  = max(0,startTime-interval) 
        endPoint    = min(startTime,startPoint + interval) # not sure about interval or interpolatinIntevalLength
    #
        x,y = trajectory[startPoint:endPoint].T
        t0 = np.arange(startPoint,endPoint,1)
        # interpolate and exterpolate to t2 this window
        spline, _ = interpolate.splprep([x, y], u=t0, s=10000,k=1)
        t2 = np.arange(startPoint,endPoint+1,1)
        IEpolation = np.array(interpolate.splev(t2, spline,ext=0))
        if debug == 1:
            if axes == 0:
                fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
                axes.plot(trajectory[:,0],trajectory[:,1], '-o')
            axes.plot(*IEpolation, linestyle='dotted', c='orange')
            axes.plot(*IEpolation[:,-1], '--o', c='orange')
            axes.plot([IEpolation[0,-1],trajectory[endPoint,0]],[IEpolation[1,-1],trajectory[endPoint,1]], '-',c='black')
            axes.set_aspect(aspect)
            axes.set_title(title)
            if 1 == -1:
                # determin unit vector in interpolation direction
                startEnd = np.array([IEpolation[:,0],IEpolation[:,-1]])
                IEdirection = np.diff(startEnd, axis = 0)[0]
                IEdirection = IEdirection/np.linalg.norm(IEdirection)
                # measure latest displacement projection to unit vector
                displ = np.diff(trajectory[endPoint-1:endPoint+1],axis = 0)[0]
                proj = np.dot(displ,IEdirection)
                start1 = trajectory[endPoint - 1]
                end1 = start1 + proj*IEdirection
                arr = np.array([start1,end1])
                axes.plot(*arr.T, '--', c='red')
                #plt.show()

        return IEpolation, endPoint

    #fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
    # === lets check if segments are single bubbles = centroid and areas are consitent ===

    pathsDeltas = {k:{} for k in paths}   
    areaDeltas = {k:{} for k in paths}  
    interpolatinIntevalLength = 3 # from 2 to this
    pathCentroids = {k:[] for k in paths}     
    overEstimateConnections = []
    for k,pathX in paths.items():
        #pathX = paths[0]
        timesReal   = [time             for time,*subIDs in pathX]
        contourIDs  = {time:subIDs      for time,*subIDs in pathX}
        numContours = {time:len(subIDs) for time,*subIDs in pathX}
        soloBubble = True if all([1 if i == 1 else 0 for i in numContours.values()]) else False
        if soloBubble:
            pathXHulls  = [cv2.convexHull(np.vstack([g0_contours[time][subID] for subID in subIDs])) for time,*subIDs in pathX]
    
            pathXHCentroidsAreas = [centroid_area(contour) for contour in pathXHulls]
            trajectory = np.array([a[0] for a in pathXHCentroidsAreas])
            pathCentroids[k] = trajectory

            areas = np.array([a[1] for a in pathXHCentroidsAreas])
            # rescale areas to be somewhat homogeneous with time step = 1, otherwise (x,y) = (t,area) ~ (1,10000) are too squished
            # because i interpolate it as curve in time-area space, with time parameter which is redundand.
            meanArea = np.mean(areas)
            areasIntep = np.array([[i,a/meanArea] for i,a in enumerate(areas)])
        
            numPointsInTraj = len(trajectory)
            #[cv2.circle(meanImage, tuple(centroid), 3, [255,255,0], -1) for centroid in pathXHCentroids]
            #meanImage   = convertGray2RGB(np.load(meanImagePath)['arr_0'].astype(np.uint8)*0)
            # start analyzing trajectory. need 2 points to make linear extrapolation
            # get extrapolation line, take real value. compare
    
            times  = np.arange(2,numPointsInTraj - 1)
            assert len(times)>0, "traj exterp, have not intended this yet"

            debug = 0; axes,axes2 = (0,0)
        
            if debug == 1:
                fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
                axes.plot(trajectory[:,0],trajectory[:,1], '-o')

                fig2, axes2 = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
                axes2.plot(areasIntep[:,0],areasIntep[:,1], '-o')

            for t in times:
            
                IEpolation, endPoint    = exterpTest(trajectory, t, interpolatinIntevalLength, debug = debug, axes= axes, title = str(k))
                IEpolation2, endPoint2  = exterpTest(areasIntep, t, interpolatinIntevalLength, debug = debug, axes= axes2, title = str(k), aspect = 'auto')
                timeReal = timesReal[endPoint + 1]
                pathsDeltas[k][timeReal] = np.linalg.norm(IEpolation[:,-1] - trajectory[endPoint])
                areaDeltas[k][timeReal] = np.abs(int((IEpolation2[:,-1][1] - areasIntep[endPoint2][1])*meanArea))
                #plt.show()
                #axes.plot(*IEpolation, linestyle='dotted', c='orange')
                #axes.plot(*IEpolation[:,-1], '--o', c='orange')
                #axes.plot([IEpolation[0,-1],trajectory[endPoint,0]],[IEpolation[1,-1],trajectory[endPoint,1]], '-',c='black')
            
            a = 1
            plt.show() if debug == 1 else 0
        else:
            # there are times for segment when rough cluster holds two or more nearby bubbles
            # but they are close enough not to split, may be merge, since otherwise it would not be a part of segment.

            # extract time where there are more than one contour
            # common case of two bubbles at best can be described by two contours. at worst as multiple.
        
            sequences = []
            current_sequence = []
    
            for time,num in numContours.items():
                if num > 1:
                    current_sequence.append(time)
                else:
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = []

            # subset those time where there are only two contours.
            # cases of two contours are easy, since you dont have to reconstruct anything
            sequencesPair = []
            for n,seq in enumerate(sequences):
                sequencesPair.append([])
                for time in seq:
                    if numContours[time] == 2:
                        sequencesPair[n].append(time)
        
            # this test has continuous sequence, without holes. test different when assert below fails.
            assert [len(a) for a in sequences] == [len(a) for a in sequencesPair], 'examine case of dropped contour pairs'

            # make connection between pairs based on area similarity
            for times in sequencesPair:
                hulls           = {time:{} for time in times}
                centroidsAreas  = {time:{} for time in times}
                for time in times:
                    subIDs               = contourIDs[time]
                    hulls[time]          = {subID:cv2.convexHull(g0_contours[time][subID]) for subID in subIDs}
                    centroidsAreas[time] = {subID:centroid_area(contour) for subID,contour in hulls[time].items()}

                connections = []
                startNodes = []
                endNodes = []
                for n,[time,centAreasDict] in enumerate(centroidsAreas.items()):
                    temp = []
                    if time < times[-1]:
                        for ID,[centroid, area] in centAreasDict.items():
                            nextTime = times[n+1]
                            areasNext = {ID:centAreas[1] for ID, centAreas in centroidsAreas[nextTime].items()}
                            areaDiffs = {ID:np.abs(area-areaX) for ID, areaX in areasNext.items()}
                            minKey = min(areaDiffs, key=areaDiffs.get)

                            connections.append([tuple([time,ID]),tuple([nextTime,minKey])])

                            if time == times[0]:
                                startNodes.append(tuple([time,ID]))
                            elif time == times[-2]:
                                endNodes.append(tuple([nextTime,minKey]))
            


                a  =1
                #allIDs = list(sum(connections,[]))
                #G = nx.Graph()
                #G.add_edges_from(connections)
                #connected_components_all = [list(nx.node_connected_component(G, key)) for key in allIDs]
                #connected_components_all = [sorted(sub, key=lambda x: x[0]) for sub in connected_components_all]
                #connected_components_unique = []
                #[connected_components_unique.append(x) for x in connected_components_all if x not in connected_components_unique]

                # get connections to first sub-sequence from prev node and same for last node
                nodesSubSeq = [tuple([t] + contourIDs[t]) for t in times]
                nodeFirst = nodesSubSeq[0]
                neighBackward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(nodeFirst))  if time < nodeFirst[0]}
            
                nodeLast = nodesSubSeq[-1]
                neighForward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(nodeLast))  if time > nodeLast[0]}

                # remove old unrefined nodes, add new subcluster connections

                H.remove_nodes_from(nodesSubSeq)
            
                H.add_edges_from(connections)

                # add new connections that start and terminate subsequence
                newConnections = []
                for prevNode in neighBackward:
                    for node in startNodes:
                        newConnections.append([prevNode,node])

                for nextNode in neighForward:
                    for node in endNodes:
                        newConnections.append([node,nextNode])

                overEstimateConnections += newConnections

                H.add_edges_from(newConnections)


            
                #fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
                #for n, nodes in enumerate(connected_components_unique):
                #    centroids = np.array([centroidsAreas[time][ID][0] for time,ID in nodes])
                #    axes.plot(*centroids.T, '-o')
            
                1
            #test = list(H.nodes())
            #node_positions = getNodePos(test)
            #fig, ax = plt.subplots()
            #nx.set_node_attributes(H, node_positions, 'pos')

            #pos = nx.get_node_attributes(H, 'pos')
            #nx.draw(H, pos, with_labels=True, node_size=50, node_color='lightblue',font_size=6,
            #font_color='black')
            #plt.show()


            #firstTime = timesReal[0]
            # segment starts as 2 bubbles
            #if numContours[firstTime] == 2:
            #    neighBackward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(pathX[0]))  if time < pathX[0][0]}
            #    if len(neighBackward) == 2:
            #        1
    #axes.legend(prop={'size': 6})
    #axes.set_aspect('equal')
    #plt.show()

    segments2, skipped = graph_extract_paths(H,f) # 23/06/23 info in "extract paths from graphs.py"

    # Draw extracted segments with bold lines and different color.
    segments2 = [a for _,a in segments2.items() if len(a) > 0]
    segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
    paths = {i:vals for i,vals in enumerate(segments2)}
    node_positions = getNodePos(list(H.nodes()))
    #drawH(H, paths, node_positions)

    pathCentroidsAreas           = {ID:{} for ID in paths}
    pathCentroids           = {ID:{} for ID in paths}
    for ID,nodeList in paths.items():
        hulls =  [cv2.convexHull(np.vstack([g0_contours[time][subID] for subID in subIDs])) for time,*subIDs in nodeList]
        pathCentroidsAreas[ID] = [centroid_area(contour) for contour in hulls]
        pathCentroids[ID] = [a[0] for a in pathCentroidsAreas[ID]]


    pathsDeltas = {k:{} for k in paths}   
    areaDeltas = {k:{} for k in paths}

    for k,pathX in paths.items():
        timesReal   = [time             for time,*subIDs in pathX]
        pathXHulls  = [cv2.convexHull(np.vstack([g0_contours[time][subID] for subID in subIDs])) for time,*subIDs in pathX]
    
        pathXHCentroidsAreas = [centroid_area(contour) for contour in pathXHulls]
        trajectory = np.array([a[0] for a in pathXHCentroidsAreas])
        pathCentroids[k] = trajectory

        areas = np.array([a[1] for a in pathXHCentroidsAreas])
        # rescale areas to be somewhat homogeneous with time step = 1, otherwise (x,y) = (t,area) ~ (1,10000) are too squished
        # because i interpolate it as curve in time-area space, with time parameter which is redundand.
        meanArea = np.mean(areas)
        areasIntep = np.array([[i,a/meanArea] for i,a in enumerate(areas)])
        
        numPointsInTraj = len(trajectory)
        #[cv2.circle(meanImage, tuple(centroid), 3, [255,255,0], -1) for centroid in pathXHCentroids]
        #meanImage   = convertGray2RGB(np.load(meanImagePath)['arr_0'].astype(np.uint8)*0)
        # start analyzing trajectory. need 2 points to make linear extrapolation
        # get extrapolation line, take real value. compare
    
        times  = np.arange(2,numPointsInTraj - 1)
        assert len(times)>0, "traj exterp, have not intended this yet"

        debug = 0; axes,axes2 = (0,0)
        
        if debug == 1:
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes.plot(trajectory[:,0],trajectory[:,1], '-o')

            fig2, axes2 = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes2.plot(areasIntep[:,0],areasIntep[:,1], '-o')

        for t in times:
            
            IEpolation, endPoint    = exterpTest(trajectory, t, interpolatinIntevalLength, debug = debug, axes= axes, title = str(k))
            IEpolation2, endPoint2  = exterpTest(areasIntep, t, interpolatinIntevalLength, debug = debug, axes= axes2, title = str(k), aspect = 'auto')
            timeReal = timesReal[endPoint + 1]
            pathsDeltas[k][timeReal] = np.linalg.norm(IEpolation[:,-1] - trajectory[endPoint])
            areaDeltas[k][timeReal] = np.abs(int((IEpolation2[:,-1][1] - areasIntep[endPoint2][1])*meanArea))
            #plt.show()
            #axes.plot(*IEpolation, linestyle='dotted', c='orange')
            #axes.plot(*IEpolation[:,-1], '--o', c='orange')
            #axes.plot([IEpolation[0,-1],trajectory[endPoint,0]],[IEpolation[1,-1],trajectory[endPoint,1]], '-',c='black')
            
        a = 1
        plt.show() if debug == 1 else 0
    # check segment end points and which cluster they are connected to
    skippedTimes = {vals[0]:[] for vals in skipped}
    for time,*subIDs in skipped:
        skippedTimes[time].append(subIDs) 

    resolvedConnections = []
    subSegmentStartNodes = [a[0] for a in paths.values()]
    for k,pathX in paths.items():
        terminate = False
        while terminate == False:
            firstNode, lastNode = pathX[0], pathX[-1]

            # find connections-neighbors from first and last segment node
            neighForward    = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(lastNode))   if time > lastNode[0]}
            # 
            #neighForward = {ID:vals for ID,vals in neighForward.items() if ID not in subSegmentStartNodes}
            #neighBackward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(firstNode))  if time < firstNode[0]}
            if len(neighForward)>0:
                time = lastNode[0] 
                timeNext = time + 1
        
                # find neigbor centroids
                pathXHulls  = {k:cv2.convexHull(np.vstack([g0_contours[timeNext][subID] for subID in subIDs])) for k,subIDs in neighForward.items()}
                pathXHCentroids = {k:centroid_area(contour)[0] for k,contour in pathXHulls.items()}
                trajectory      = pathCentroids[k]
                testChoice      = {}
                # test each case, if it fits extrapolation- real distance via history of segment
                for key, centroid in pathXHCentroids.items():
                    trajectoryTest = np.vstack((trajectory,centroid))#np.concatenate((trajectory,[centroid]))
                    IEpolation, endPoint = exterpTest(trajectoryTest,len(trajectoryTest) - 1,interpolatinIntevalLength, 0)
                    dist = np.linalg.norm(IEpolation[:,-1] - trajectoryTest[endPoint])
                    testChoice[key] = dist
        
                # check which extrap-real dist is smallest, get history
                minKey = min(testChoice, key=testChoice.get)
                lastDeltas = [d for t,d in pathsDeltas[k].items() if time - 4 < t <= time]
                mean = np.mean(lastDeltas)
                std = np.std(lastDeltas)
                # test distance via history
                if testChoice[minKey] < mean + 2*std:
                    removeNodeConnections = [node for node in neighForward if node != minKey]
                    remEdges = []
                    remEdges += [tuple([lastNode,node]) for node in removeNodeConnections]
                    H.remove_edges_from(remEdges)
                    resolvedConn = [lastNode,minKey]

                    for conn in [resolvedConn] + remEdges:
                        if conn in overEstimateConnections: overEstimateConnections.remove(conn) # no skipped inside, may do nothing
                    resolvedConnections.append(resolvedConn)
                    paths[k].append(minKey)
                    pathX.append(minKey)
                    pathCentroids[k] = np.vstack((pathCentroids[k],pathXHCentroids[minKey]))
                    pathsDeltas[k][timeNext] = testChoice[minKey]

                    # removed edges that ar pointing to solution, other tan real one.
                    solBackNeighb = [ID for ID in list(H.neighbors(minKey)) if ID[0] < timeNext]
                    removeNodeConnections2 = [node for node in solBackNeighb if node != lastNode]
                    remEdges = []
                    remEdges += [tuple([node,minKey]) for node in removeNodeConnections2]
                    H.remove_edges_from(remEdges)

                    if minKey in subSegmentStartNodes:
                        terminate = True
                    #drawH(H, paths, node_positions)
                else:
                    terminate = True
            elif len(neighForward) == 0: # segment terminates
                terminate = True
        
                    # ==== !!!!!!!!!!!!!!!
                    # might want to iterate though skipped nodes and connect to next segment!
            a = 1
        #if timeNext in skippedTimes:
        #    testCluster = skippedTimes[timeNext]
        


    #for time,*subIDs in [pathX[X]]:
    #        [cv2.drawContours(  meanImage,   g0_contours[time], subID, 255 , 1) for subID in subIDs]
    #cv2.drawContours(  meanImage,   pathXHulls, X, 128 , 1)
    #cv2.imshow('a', meanImage)
    #drawH(H, paths, node_positions)
    a = 1


    #nx.draw(H, with_labels=True, edge_color=list(colors_edges2.values()), width = list(width2.values()))

    #end_points = [node for node in H.nodes() if H.degree[node] == 1]
    #end_points.remove(test[-1])
    #print(f'endpoints: {end_points}')
    #stor = {a:[a] for a in end_points}
    #for start in end_points:
    #    bfs_tree = nx.bfs_tree(H, start)
    #    target_node = None
    #    for node in bfs_tree.nodes():
    #        neighbors = [a for a in H.neighbors(node) if a[0]> node[0]]
    #        target_node = node
    #        stor[start].append(node)
    #        if len(neighbors) > 1:
    #            break
    #start = test[0][0]
    #end = test[-1][0]
    #times = {t:[] for t in np.arange(start, end + 1, 1)}
    #gatherByTime = {t:[] for t in np.arange(start, end + 1, 1)}

    #for t,*clusterID in test:
     
    #    #gatherByTime2[t].append((t,ID2S(g0_clusters[t][clusterID])))
        #gatherByTime[t].append((t,*clusterID))

    f = lambda x : x[0]

    segments2, skipped = graph_extract_paths(H,f) # 23/06/23 info in "extract paths from graphs.py"

    # Draw extracted segments with bold lines and different color.
    segments2 = [a for _,a in segments2.items() if len(a) > 0]
    segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
    paths = {i:vals for i,vals in enumerate(segments2)}


    #drawH(H, paths, node_positions)

    if 1 == 1:
        imgInspectPath = os.path.join(imageFolder, "inspectPaths")
        meanImage = convertGray2RGB(np.load(meanImagePath)['arr_0'].astype(np.uint8))
        for n, keysList in paths.items():
            for k,subCase in enumerate(keysList):
                img = meanImage.copy()
                time,*subIDs = subCase
                for subID in subIDs:
                    cv2.drawContours(  img,   g0_contours[time], subID, cyclicColor(n), -1)
                    if subID in g0_contours_children[time]:
                        [cv2.drawContours(  img,   g0_contours[time], ID, (0,0,0), -1) for ID in g0_contours_children[time][subID]]
                
                    startPos2 = g0_contours[time][subID][-30][0] 
                    [cv2.putText(img, str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
                fileName = f"{str(doX).zfill(3)}_{str(n).zfill(2)}_{str(time).zfill(3)}.png"
                cv2.imwrite(os.path.join(imgInspectPath,fileName) ,img)

    if 1 == -1:
        binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
        imgs = [convertGray2RGB(binarizedMaskArr[k].copy()) for k in range(binarizedMaskArr.shape[0])]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7; thickness = 4;
        for n, case in tqdm(enumerate(segments2)):
            case    = sorted(case, key=lambda x: x[0])
            for k,subCase in enumerate(case):
                time,*subIDs = subCase
                for subID in subIDs:
                    cv2.drawContours(  imgs[time],   g0_contours[time], subID, cyclicColor(n), 2)
                x,y,w,h = cv2.boundingRect(np.vstack([g0_contours[time][ID] for ID in subIDs]))
                #x,y,w,h = g0_bigBoundingRect2[time][subCase]
                #x,y,w,h = g0_bigBoundingRect[time][ID]
                cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
                [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
            for k,subCase in enumerate(case):
                time,*subIDs = subCase
                for subID in subIDs:
                    startPos2 = g0_contours[time][subID][-30][0] 
                    [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        

        for k,img in enumerate(imgs):
            folder = r"./post_tests/testImgs/"
            fileName = f"{str(k).zfill(4)}.png"
            cv2.imwrite(os.path.join(folder,fileName) ,img)


a = 1
#if useMeanWindow == 1:
#    meanImage = masksArr[whichMaskInterval(globalCounter,intervalIndecies)]

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()