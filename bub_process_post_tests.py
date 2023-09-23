from ast import For
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
        empty_positions = set(range(max(vertical_positions) + 1)) - vertical_positions

        position_mapping = {}
        sorted_IDs = sorted(t_pos,key=t_pos.get)
        t_pos2 = {ID:t_pos[ID] for ID in sorted_IDs if t_pos[ID] > min(empty_positions)}
        for i, position in t_pos2.items():
            if position > min(empty_positions) and position not in position_mapping:
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

    #def custom_spring_layout():

    #    1) get all nodes that are not fixed
    #    2) for each node create a dictionary D that holds nodes neighbors
    #    3) during iteration go though node and their neighbors:
    #        3.1) calculate force based on distance and direction from main node to each t_neighbors_time_next
    #        3.2) summ up the forces from neighbors. 
    #    4) apply displacement in the end of iteration for all free nodes
            
    def custom_spring_layout(G, pos, fixed=None, k=0.1, iterations=50, seed=None):
        """
        Custom spring layout that considers only edge attractive forces (no repulsive forces between nodes).

        Parameters:
        - G: NetworkX graph
        - fixed: Dictionary of fixed nodes and their positions (default: None)
        - k: Optimal distance between nodes
        - iterations: Number of iterations for the spring layout algorithm
        - seed: Random seed for reproducibility (default: None)

        Returns:
        - pos: Dictionary of node positions
        """
        if fixed is None:
            fixed = {}

        #pos = nx.random_layout(G, seed=seed)

        for _ in range(iterations):
            free_nodes = [node for node in G.nodes() if node not in fixed]

            # Step 1: Get all nodes that are not fixed
            for node in free_nodes:
                # Step 2: Create a dictionary D that holds node neighbors
                neighbors = {neighbor: np.array(pos[neighbor]) for neighbor in G.neighbors(node)}

                # Initialize displacement vector
                displacement = np.array([0.0, 0.0])

                # Step 3: Calculate forces
                for neighbor, neighbor_position in neighbors.items():
                    delta = neighbor_position - pos[node]
                    length = max(np.linalg.norm(delta), 0.01)  # Avoid division by zero
                    force = (k * length) * delta / length
                    displacement += force

                # Step 4: Apply displacement
                pos[node] += displacement

        return pos

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

    def combs_different_lengths(elements_list):
        return sum([list(itertools.combinations(elements_list, r)) for r in range(1,len(elements_list)+1)],[])

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

    def disperse_nodes_to_times(nodes):
        t_perms = defaultdict(list)
        for t, *t_subIDs in nodes:
            t_perms[t].extend(t_subIDs)
        return dict(t_perms) 

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

    def set_custom_node_parameters(graph, contour_data, nodes_list, calc_hull = 1):
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
        return 

    def find_key_by_value(my_dict, value_to_find):
        for key, value in my_dict.items():
            if value == value_to_find:
                return key
        # If the value is not found, you can return a default value or raise an exception.
        # In this example, I'm returning None.
        return None
    
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
                common_times_dict[pair] = list(common_times)

        return common_times_dict
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
doX = 30#60#56#20#18#2#84#30#60#30#20#15#2#1#60#84
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


set_custom_node_parameters(G, g0_contours_hulls, G.nodes(), calc_hull = 0)

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

g0_pairConnections2_OG = copy.deepcopy(g0_pairConnections2)
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
    t_merge_split_culprit_edges2 = []
    for t_from, t_forward_conns in t_neighbor_sol_all_next.items():
        if len(t_forward_conns)>1:
            for t_to in t_forward_conns.keys():
                t_merge_split_culprit_edges2.append(tuple(sorted([t_from,t_to])))

    for t_to, t_backward_conns in t_neighbor_sol_all_prev.items():
        if len(t_backward_conns) > 1:
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

    # gather successors for mixed m/s in a dict for further trajectory extension 21.09.23
    lr_conn_splits_merges_mixed_dict = defaultdict(list)
    for t_from,t_to in  lr_conn_edges_splits_merges_mixed:
        lr_conn_splits_merges_mixed_dict[t_from].append(t_to)

    for t_from, t in lr_conn_splits_merges_mixed_dict.items():
        lr_conn_splits_merges_mixed_dict[t_from] = sorted(set(t))

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

# EDIT: 20.09.23 keep zero path in t_conn_121
# >>>>>>>>>>>>>>>> WHAT I SHOULD DO <<<<<<<<<<<<<<<<<<<
# 1) E.Z consume small branch. its most likely true sol-n, but algorithm might not be ready for composite nodes
# 2) or i strip edge status from all small branches, thus its no longer zero path. so you can treat it as usual.

#t_conn_121 = [t_node for t_node in t_conn_121 if t_node not in t_conn_121_zero_path]

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

# EDIT: 20.09.23
# consume zero path
# combine contested nodes
# join segments into one.
lr_zp_redirect = {tID: tID for tID in range(len(segments2))}
for t_conn,t_dict in t_conn_121_zp_contour_combs.items():
    t_from, t_to        = t_conn
    t_times             = [t for t,t_subIDs in t_dict.items() if len(t_subIDs) > 1]
    assert len(t_times) == 1, "unexpected case. multiple times, need to insert nodes in correct order"
    t_nodes_composite   = [tuple([t] + t_dict[t]) for t in t_times]                 # join into one node (composit)
    t_nodes_solo        = [(t,t_subID) for t in t_times for t_subID in t_dict[t]]   # get solo contour nodes

    t_edges             = []
    t_nodes_prev        =   [t_node for t_node in segments2[t_from] if t_node not in t_nodes_solo] 
    t_edges             +=  [(t_nodes_prev[-1]      , t_nodes_composite[0]) ]
    segments2[t_from]   =   t_nodes_prev
    segments2[t_from]   +=  t_nodes_composite
    t_nodes_next        =   [t_node for t_node in segments2[t_to] if t_node not in t_nodes_solo]
    t_edges             +=  [(t_nodes_composite[0]  , t_nodes_next[0])      ]
    segments2[t_from]   +=  t_nodes_next
    segments2[t_to]     = []

    # modify graph: remove solo IDs, add new composite nodes. add parameters to new nodes
    G.remove_nodes_from(t_nodes_solo) 
    G.add_edges_from(t_edges)

    set_custom_node_parameters(G, g0_contours, t_nodes_composite, calc_hull = 1)
    
    # modify segment view graph.
    t_successors   = extractNeighborsNext(G2, t_to, lambda x: G2.nodes[x]["t_start"])
    
    t_edges = [(t_from,t_succ) for t_succ in t_successors]
    G2.remove_node(t_to)
    G2.add_edges_from(t_edges)
    G2.nodes()[t_from]["t_end"] = segments2[t_from][-1][0]

    # create dict to re-connect previously obtained edges.
    lr_zp_redirect[t_to] = t_from
    
 

lr_conn_edges_merges    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_merges   )
lr_conn_edges_splits    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits   )
lr_conn_edges_splits_merges_mixed = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits_merges_mixed   )
t_conn_121              = lr_reindex_masters(lr_zp_redirect, t_conn_121             ,remove_solo_ID = 1)

# lr_close_segments_simple_paths not modified.
a = 1
            
# ===============================================================================================
# ===== EDIT 10.09.23 lets find joined 121s ========================
# ===============================================================================================

if 1 == 1:
    # codename: big121
    G_seg_view_1 = nx.DiGraph()
    G_seg_view_1.add_edges_from(t_conn_121)

    for g in G_seg_view_1.nodes():
          G_seg_view_1.nodes()[g]["t_start"]    = segments2[g][0][0]
          G_seg_view_1.nodes()[g]["t_end"]      = segments2[g][-1][0]
    
    # 121s are connected to other segments, if these segments are other 121s, then we can connect 
    # them into bigger one and recover using more data.
    lr_big_121s_chains = extract_graph_connected_components(G_seg_view_1.to_undirected(), lambda x: x)

    print(f'Working on joining all continious 121s together: {lr_big_121s_chains}')
    # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        t_segment_ID_left   = [t_subIDs[0] for t_subIDs in lr_big_121s_chains]
        t_segment_ID_right  = [t_subIDs[-1] for t_subIDs in lr_big_121s_chains]
        t_temp_merge    = defaultdict(int)
        t_temp_split    = defaultdict(int)
        t_temp_in_merge = defaultdict(list)
        t_temp_in_split = defaultdict(list)
        # find if left is part of split, and with whom
        for t, t_ID in enumerate(t_segment_ID_left):
            for t_from,t_to in lr_conn_edges_splits:
                if t_to == t_ID:                    # if t_ID is branch of split from t_from
                    t_temp_split[t_ID] = t_from     # save who was mater of left segment-> t_ID:t_from
                    t_temp_in_split[t] = lr_big_121s_chains[t]     # index in big 121s: connected components. t_ID in cc.

        # find if right is part of merge, and with whom
        for t, t_ID in enumerate(t_segment_ID_right): # order in big 121s, right segment ID.
            for t_from,t_to in lr_conn_edges_merges:
                if t_from == t_ID: 
                    t_temp_merge[t_ID] = t_to
                    t_temp_in_merge[t] = lr_big_121s_chains[t]

        # extract big 121s that are in no need of real/pseudo m/s analysis.
        lr_big121_events_none = defaultdict(list) # big 121s not terminated by splits or merges. ready to resolve.
        # find if elements of big 121s are without m/s or bounded on one or both sides by m/s.
        set1 = set(t_temp_in_merge)
        set2 = set(t_temp_in_split)

        t_only_merge = set1 - set2
        t_only_split = set2 - set1

        t_merge_split = set1.intersection(set2)
        t_merge_split_no = set(range(len(lr_big_121s_chains))) - (set1.union(set2))
        #assert len(t_merge_split) == 0, "segments bounded on both sides by merge/split. not encountered before, but ez to fix"

        lr_events = {'merge':defaultdict(list), 'split':defaultdict(list)}
    # -----------------------------------------------------------------------------------------------
    # PSEUDO MERGE/SPLIT ON EDGES: DETERMINE IF M/S IS FAKE OR REAL
    # -----------------------------------------------------------------------------------------------

    # test if edge segment is part of real of fake m/s.
    # strip it from big 121s and compare area of prev segment and result of event
    # if area changed drastically then its real m/s, otherwise its same bubble with fake m/s
    # NOTE: currently i take a single end-start nodes. might take more if i test for if they are in m/s (change to segment of history avg later)
    print('Determining if end points of big 121s are pseudo events: ')
    if 1 == 1:
        t_big121s_merge_conn_fake = []
        t_big121s_split_conn_fake = []

        t_merge_split_fake = defaultdict(list) # 0 -> is part of fake split, -1 -> is part of fake merge
        t_only_merge_split_fake = []
        t_max_pre_merge_segment_length = 12 # skip if branch is too long. most likely a real branch

        t_only_merge_fake = []
        for t in t_only_merge.union(t_merge_split): #t_only_merge + t_merge_split
            t_big121_subIDs = lr_big_121s_chains[t]
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
                if t not in t_merge_split:
                    t_only_merge_fake += [t]
                    t_big121s_merge_conn_fake.append((t_from,t_from_to))
                else:       # undetermined, decide after
                    t_merge_split_fake[t] += [-1]

        t_only_split_fake = []
        for t in t_only_split.union(t_merge_split):
            t_big121_subIDs = lr_big_121s_chains[t]
            t_to            = t_big121_subIDs[0]   # possible fake merge branch (segment ID).
            t_to_to         = t_big121_subIDs[1]   # segment ID prior to that, most likely OG bubble.
            t_from_to       = t_temp_split[t_to]  # seg ID of merge.

            if len(segments2[t_to]) >= t_max_pre_merge_segment_length: continue

            t_from_to_node  = segments2[t_from_to   ][-1]                   # take end-start nodes
            t_to_to_node    = segments2[t_to_to     ][0]

            t_from_to_node_area = G.nodes[t_from_to_node]["area"]       # check their area
            t_to_to_node_area   = G.nodes[t_to_to_node  ]["area"] 
            if np.abs(t_from_to_node_area - t_to_to_node_area) / t_from_to_node_area < 0.35:
                if t not in t_merge_split:
                    t_only_split_fake += [t]
                    t_big121s_split_conn_fake.append((t_from_to,t_to))
                else:       # undetermined, decide after
                    t_merge_split_fake[t] += [0]

        for t,t_states in t_merge_split_fake.items():
            if len(t_states) == 2:                  # both ends are in fake event
                t_only_merge_split_fake += [t]
            elif 0 in t_states:                     # only split is fake
                t_to            = lr_big_121s_chains[t][0]   
                t_from_to       = t_temp_split[t_to] 
                t_only_split_fake += [t]
                t_big121s_split_conn_fake.append((t_from_to,t_to))
            else:                                  # only merge is fake
                t_from      = lr_big_121s_chains[t][-1]  
                t_from_to   = t_temp_merge[t_from]
                t_only_merge_fake += [t]
                t_big121s_merge_conn_fake.append((t_from,t_from_to))

        assert len(t_only_merge_split_fake) == 0, "case yet unexplored, written without test"        


    print(f'Fake merges: {t_big121s_merge_conn_fake}, Fake splits: {t_big121s_split_conn_fake}')
    # -----------------------------------------------------------------------------------------------
    # PSEUDO MERGE/SPLIT ON EDGES: TRIM BIG 121S AS TO DROP FAKE M/S.
    # CREATE NEW t_conn_121 -> lr_big121s_conn_121
    # -----------------------------------------------------------------------------------------------
    # hopefully fake edges of big 121s are detected, so trim them, if possible and add to future reconstruction.
    if 1 == 1:
        t_big121s_edited = [None]*len(lr_big_121s_chains) # by default everything fails. reduced len 2 cc to len 1 stays none = failed.
        for t, t_subIDs in enumerate(lr_big_121s_chains):

            if t in t_only_merge_split_fake:    # both endpoints have to be removed, not tested
                t_subIDs_cut = t_subIDs[1:-1]
            elif t in t_only_merge_fake:        # drop fake pre-merge branch
                t_subIDs_cut = t_subIDs[:-1]
            elif t in t_only_split_fake:        # drop fake post-split branch
                t_subIDs_cut = t_subIDs[1:]    
            else:                               # nothing changes
                t_subIDs_cut = t_subIDs

            if len(t_subIDs_cut) > 1:           # questionable, stays None
                t_big121s_edited[t] = t_subIDs_cut
            #else:
            #    assert 1 == 0, "check out case t_subIDs_cut of len <= 1"
            

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
    print(f'Trimmed big 121s: {lr_big121s_conn_121}/nInterpolating whole big 121s...')

    # -----------------------------------------------------------------------------------------------
    # BIG 121S INTERPOLATE COMBINED TRAJECTORIES, GIVEN FULL INFO REDISTRIBUTE FOR DATA FOR EACH CONN
    # -----------------------------------------------------------------------------------------------
    # CALCULATE interpolation of holes in data for big 121s
    if 1 == 1: # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        lr_big121s_interpolation        = defaultdict(dict)
        lr_big121s_interpolation_big    = defaultdict(dict) 

        for t,t_subIDs in enumerate(t_big121s_edited):
            if t_subIDs != None: 
                t_temp_nodes_all = []
                for t_subID in t_subIDs:
                    t_temp_nodes_all += segments2[t_subID]

                t_temp_times        = [G.nodes[t_node]["time"       ] for t_node in t_temp_nodes_all]
                t_temp_areas        = [G.nodes[t_node]["area"       ] for t_node in t_temp_nodes_all]
                t_temp_centroids    = [G.nodes[t_node]["centroid"   ] for t_node in t_temp_nodes_all]
        
                t_conns_times_dict = {}
                for t_edge in zip(t_subIDs[:-1], t_subIDs[1:]):
                    t_from,t_to = t_edge
                    t_start,t_end  = (segments2[t_from][-1][0], segments2[t_to][0][0])
                    t_times = list(np.arange(t_start + 1,t_end,1))

                    t_conns_times_dict[t_edge] = t_times
                    
                t_times_missing_all = sum(t_conns_times_dict.values(),[])
        
                a = 1
                # interpolate composite (long) parameters 
                t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, s = 15, debug = 0, aspect = 'equal', title = t_subIDs)
                t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_subIDs)
                # form dict time:centroid for convinience
                t_interpolation_centroids_1 = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_centroids_0)}
                t_interpolation_areas_1     = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_areas_0)}
                # save data with t_conns keys
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
    print('Computing contour elements for space between segments')
    if 1 == 1:

        lr_big121s_perms_pre = {}

        for t_conn in lr_big121s_conn_121:  # did not modify it after removing/combining zero path conns.
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
    print('Computing contour element permutations for each time step...')
    #lr_contour_combs_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_contour_combs.items()}
    if 1 == 1:
        lr_big121s_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_big121s_perms_pre.items()}
        for t_conn, t_times_contours in lr_big121s_perms_pre.items():
            for t_time,t_contours in t_times_contours.items():
                t_perms = combs_different_lengths(t_contours)
                #t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
                lr_big121s_perms[t_conn][t_time] = t_perms



    # ===============================================================================================
    # ====  PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ====
    # ===============================================================================================
    # REMARK: these will be reused alot in next steps, store them to avoid need of recalculation
    print('Calculating parameters for possible contour combinations...')
    if 1 == 1:    
        lr_big121s_perms_areas      = AutoCreateDict()# lr_init_perm_precomputed(lr_big121s_perms,0)
        lr_big121s_perms_centroids  = AutoCreateDict()# lr_init_perm_precomputed(lr_big121s_perms,[0,0])
        lr_big121s_perms_mom_z      = AutoCreateDict()# lr_init_perm_precomputed(lr_big121s_perms,0)

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
    print('Generating possible bubble evolution paths from prev combinations...')
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
    print('Determining evolutions thats are closest to interpolated missing data...')
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
    lr_C0_condensed_connections_relations = lr_zp_redirect.copy() #t_condensed_connections_all_nodes
    for t_subIDs in lr_C0_condensed_connections:
        for t_subID in t_subIDs:
            lr_C0_condensed_connections_relations[t_subID] = min(t_subIDs)
    print('Saving results for restored parts of big 121s')
    t_last_seg_ID_in_big_121s = [t[-1] for t in t_big121s_edited if t is not None]
    t_big121s_edited_clean = [t for t in t_big121s_edited if t is not None]
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
            set_custom_node_parameters(G, g0_contours, t_nodes_intermediate, calc_hull = 1)
    
            # wipe data if t_from is inherited
            if t_from_new != t_from:
                t_segments_new[t_from] = []
            # wipe data from t_to anyway
            t_segments_new[t_to] = []

            # modify node view graph. since these are big 121s, eventally first segment will be connected to last
            # so when t_to is last in t_big121s_edited, you can wipe all in between.
            if t_to in t_last_seg_ID_in_big_121s:
                t_index = t_last_seg_ID_in_big_121s.index(t_to) # which big 121 is it
                t_drop_IDs = t_big121s_edited_clean[t_index][1:]                  # grab all members 
                t_from_OG = t_big121s_edited_clean[t_index][0]
                t_successors   = extractNeighborsNext(G2, t_to, lambda x: G2.nodes[x]["t_start"])
                t_edges = [(t_from_OG,t_succ) for t_succ in t_successors]
                G2.remove_nodes_from(t_drop_IDs)
                G2.add_edges_from(t_edges)
                G2.nodes()[t_from_OG]["t_end"] = segments2[t_from_new][-1][0] # change master node parameter
                a = 1
            
    # at this time some segments got condenced into big 121s. right connection might be changed.
    lr_time_active_segments = {t:[] for t in lr_time_active_segments}
    for k,t_segment in enumerate(t_segments_new):
        t_times = [a[0] for a in t_segment]
        for t in t_times:
            lr_time_active_segments[t].append(k)

    #for_graph_plots(G, segs = t_segments_new) 
    # -----------------------------------------------------------------------------------------------
    # prep permutations for merges/splits fake and real
    # -----------------------------------------------------------------------------------------------
    print('Working on real and fake events (merges/splits):/nPreparing data...')
    if 1 == 1: # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        print('Generating information for:')
        # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        t_event_start_end_times = {'merge':{},'split':{},'mixed':{}}

        func_prev_neighb = lambda x: G2.nodes[x]["t_end"]
        func_next_neighb = lambda x: G2.nodes[x]["t_start"]

        # ------------------------------------- REAL MERGE/SPLIT -------------------------------------
        t_state = 'merge'
        for t_to in t_merge_real_to_ID: #t_ID is to merge ID. ID is not changed
            t_predecessors   = extractNeighborsPrevious(G2, t_to, func_prev_neighb)
            t_predecessors = [lr_C0_condensed_connections_relations[t] for t in t_predecessors] # update if inherited
            t_predecessors_times_end = {t:G2.nodes[t]["t_end"] for t in t_predecessors}
            t_t_to_start = G2.nodes[t_to]["t_start"]
            t_times         = {t_ID: np.arange(t_t_from_end, t_t_to_start + 1, 1)
                               for t_ID,t_t_from_end in t_predecessors_times_end.items()}

            t_event_start_end_times[t_state][t_to] = {'t_start':        t_predecessors_times_end,
                                                      'branches':       t_predecessors,
                                                      't_end':          t_t_to_start,
                                                      't_times':        t_times,
                                                      't_perms':        {}}

            # pre-isolate graph segment where all sub-events take place
            t_t_from_min = min(t_predecessors_times_end.values())    # take earliest branch time
            t_nodes_keep    = [node for node in G.nodes() if t_t_from_min <= node[0] <= t_t_to_start] # < excluded edges
            t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
            # working with each branch separately
            for t_from in t_times:
                #t_ref_ID = lr_C0_condensed_connections_relations[t_from]
                t_segments_keep = [t_from, t_to]      # new inherited ID
                t_start = t_predecessors_times_end[t_from]  # storage uses old ID
                
                if t_t_to_start - t_start > 1:
                    t_sol = graph_sub_isolate_connected_components(t_subgraph, t_start, t_t_to_start, lr_time_active_segments,
                                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from][-1]) 
                    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                    t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
                    #t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]
                else: t_values = None
                t_event_start_end_times[t_state][t_to]['t_perms'][t_from] = t_values

        #t_state = 'merge'
        #for t_ID in t_merge_real_to_ID: #t_ID is to merge ID. ID is not changed
        #    t_predecessors   = extractNeighborsPrevious(G2, t_ID, func_prev_neighb) 
        #    t_predecessors_times = {t:G2.nodes[t]["t_end"] for t in t_predecessors}
        #    t_end = G2.nodes[t_ID]["t_start"]
        #    t_times         = {(t,t_ID):np.arange(t_start, t_end + 1, 1) for t,t_start in t_predecessors_times.items()}
        #    t_event_start_end_times[t_state][t_ID] = {'t_start':        t_predecessors_times,
        #                                              'branches':       t_predecessors,
        #                                              't_end':          t_end,
        #                                              't_times':        t_times,
        #                                              't_perms':        {}}

        #    # pre-isolate graph segment where all sub-events take place
        #    t_time_min = min(t_predecessors_times.values())
        #    t_nodes_keep    = [node for node in G.nodes() if t_time_min <= node[0] <= t_end] # < excluded edges
        #    t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
        #    # working with each branch separately
        #    for t_from,t_to in t_times:
        #        t_ref_ID = lr_C0_condensed_connections_relations[t_from]
        #        t_segments_keep = [t_ref_ID, t_to]      # new inherited ID
        #        t_start = t_predecessors_times[t_from]  # storage uses old ID
                
        #        if t_end - t_start > 1:
        #            t_sol = graph_sub_isolate_connected_components(t_subgraph, t_start, t_end, lr_time_active_segments,
        #                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1]) 
        #            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
        #            t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]
        #        else: t_values = None
        #        t_event_start_end_times[t_state][t_ID]['t_perms'][(t_from,t_to)] = t_values

        t_state = 'split'
        for t_from in t_split_real_from_ID:
            assert 1 == 1, "never ran t_split_real_from_ID because of lack of data."
            t_from_new = lr_C0_condensed_connections_relations[t_from]                  # split segment may be inherited.
            t_from_successors   = extractNeighborsNext(G2, t_from_new, func_next_neighb)     # branches are not
            t_successors_times = {t_to:G2.nodes[t_to]["t_start"] for t_to in t_from_successors}
            t_t_start = G2.nodes[t_from_new]["t_end"]
            t_times         = {t:np.arange(t_t_start, t_t_end + 1, 1) for t,t_t_end in t_successors_times.items()}
            t_event_start_end_times[t_state][t_from_new] = {'t_start' :       t_t_start,
                                                      'branches':       t_from_successors,
                                                      't_end'   :       t_successors_times,
                                                      't_times' :       t_times,
                                                      't_perms':        {}}

            # pre-isolate graph segment where all sub-events take place
            t_t_to_max = max(t_successors_times.values())
            t_nodes_keep    = [node for node in G.nodes() if t_t_start <= node[0] <= t_t_to_max]
            t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
            # working with each branch separately
            for t_to in t_times:
                t_segments_keep = [t_from_new, t_to] 
                t_t_to_end = t_successors_times[t_to]
                if t_t_to_end - t_t_start > 1:
                    t_sol = graph_sub_isolate_connected_components(t_subgraph, t_t_start, t_t_to_end, lr_time_active_segments,
                                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from_new][-1])
                    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                    t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
                    #t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]
                else: t_values = None
                t_event_start_end_times[t_state][t_from_new]['t_perms'][t_to] = t_values
        print(f'Real merges({t_merge_real_to_ID})/splits({t_split_real_from_ID})... Done')
        #t_state = 'split'
        #for t_ID in t_split_real_from_ID:
        #    assert 1 == 1, "never ran t_split_real_from_ID because of lack of data."
        #    t_ID_new = lr_C0_condensed_connections_relations[t_ID]
        #    t_successors   = extractNeighborsNext(G2, t_ID_new, func_next_neighb) 
        #    t_successors_times = {t:G2.nodes[t]["t_start"] for t in t_successors}
        #    t_start = G2.nodes[t_ID_new]["t_end"]
        #    t_times         = {t:np.arange(t_start, t_end + 1, 1) for t,t_end in t_successors_times.items()}
        #    t_event_start_end_times[t_state][t_ID] = {'t_start' :       t_start,
        #                                              'branches':       t_successors,
        #                                              't_end'   :       t_successors_times,
        #                                              't_times' :       t_times,
        #                                              't_perms':        {}}

        #    # pre-isolate graph segment where all sub-events take place
        #    t_time_max = max(t_successors_times.values())
        #    t_nodes_keep    = [node for node in G.nodes() if t_start <= node[0] <= t_time_max]
        #    t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
        #    # working with each branch separately
        #    for t_to in t_times:
        #        #t_ref_ID = lr_C0_condensed_connections_relations[t_from]
        #        t_segments_keep = [t_ID_new, t_to] 
        #        t_end = t_successors_times[t_to]
        #        if t_end - t_start > 1:
        #            t_sol = graph_sub_isolate_connected_components(t_subgraph, t_start, t_end, lr_time_active_segments,
        #                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ID_new][-1])
        #            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            
        #            t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]
        #        else: t_values = None
        #        t_event_start_end_times[t_state][t_ID]['t_perms'][(t_from,t_to)] = t_values
        #print(f'Real merges({t_merge_real_to_ID})/splits({t_split_real_from_ID})... Done')

        # ------------------------------------- FAKE MERGE/SPLIT -------------------------------------
        t_state = 'merge'
        for t_to in t_merge_fake_to_ID:
            
            t_to_predecessors   = extractNeighborsPrevious(G2, t_to, func_prev_neighb)  # should be updated prior
            t_to_pre_predecessors = []
            for t_to_pre in t_to_predecessors:
                t_to_pre_predecessors += extractNeighborsPrevious(G2, t_to_pre, func_prev_neighb)
            # if fake branches should terminate and only 1 OG should be left. check if this is the case
            assert len(t_to_pre_predecessors) == 1, "fake branch/es consists of multiple segments, not yet encountered"
            t_to_pre_pre_ID = t_to_pre_predecessors[0]
            t_t_start         = G2.nodes[t_to_pre_pre_ID]["t_end"]
            t_t_end           = G2.nodes[t_to]["t_start"]
            t_times         = {(t_to_pre_pre_ID, t_to):np.arange(t_t_start, t_t_end + 1, 1)}
            t_event_start_end_times[t_state][t_to] = {'t_start':        t_t_start,
                                                      'pre_predecessor':t_to_pre_pre_ID,
                                                      'branches':       t_to_predecessors,
                                                      't_end':          t_t_end,
                                                      't_times' :       t_times}


            t_segments_keep = [t_to_pre_pre_ID] + t_to_predecessors + [t_to]
            
            t_ref_ID = lr_C0_condensed_connections_relations[t_to_pre_pre_ID]

            t_sol = graph_sub_isolate_connected_components(G, t_t_start, t_t_end, lr_time_active_segments,
                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1])
            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            seqs,t_nodes_pre = perms_with_branches(t_to_predecessors,t_segments_new,t_perms, return_nodes = True) 
            t_event_start_end_times[t_state][t_to]['t_combs'] = seqs
            t_event_start_end_times[t_state][t_to]['t_nodes_solo'] = t_sol  
            t_conn = (lr_C0_condensed_connections_relations[t_to_pre_pre_ID],t_to)
            for t_time, t_perms in t_nodes_pre.items():
                for t_perm in t_perms:
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))

                    t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                    lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
                    lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
                    lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z

        #t_state = 'merge'
        #for t_ID in t_merge_fake_to_ID:
        #    t_predecessors   = extractNeighborsPrevious(G2, lr_C0_condensed_connections_relations[t_ID], func_prev_neighb) # lambda takes neighbors with lower end time than t_ID. 
        #    t_pre_predecessors = []
        #    for t_ID_pre in t_predecessors:
        #        t_pre_predecessors += extractNeighborsPrevious(G2, t_ID_pre, func_prev_neighb)
        #    # if fake branches should terminate and only 1 OG should be left. check if this is the case
        #    assert len(t_pre_predecessors) == 1, "fake branch/es consists of multiple segments, not yet encountered"
        #    t_ID_pre_pred = t_pre_predecessors[0]
        #    t_start         = G2.nodes[t_ID_pre_pred]["t_end"]
        #    t_end           = G2.nodes[t_ID]["t_start"]
        #    t_times         = {(t_ID_pre_pred, t_ID):np.arange(t_start, t_end + 1, 1)}
        #    t_event_start_end_times[t_state][t_ID] = {'t_start':        t_start,
        #                                              'pre_predecessor':t_ID_pre_pred,
        #                                              'branches':       t_predecessors,
        #                                              't_end':          t_end,
        #                                              't_times' :       t_times}


        #    t_segments_keep = [t_ID_pre_pred] + t_predecessors + [t_ID]
            
        #    t_ref_ID = lr_C0_condensed_connections_relations[t_ID_pre_pred]

        #    t_sol = graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments,
        #                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1])
        #    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
        #    seqs,t_nodes_pre = perms_with_branches(t_predecessors,t_segments_new,t_perms, return_nodes = True) 
        #    t_event_start_end_times[t_state][t_ID]['t_combs'] = seqs
        #    t_event_start_end_times[t_state][t_ID]['t_nodes_solo'] = t_sol  
        #    t_conn = (lr_C0_condensed_connections_relations[t_ID_pre_pred],t_ID)
        #    for t_time, t_perms in t_nodes_pre.items():
        #        for t_perm in t_perms:
        #            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))

        #            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

        #            lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
        #            lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
        #            lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z
            a = 1


        t_state = 'split'
        for t_from in t_split_fake_from_ID:
            t_from_new = lr_C0_condensed_connections_relations[t_from]
            t_from_successors    = extractNeighborsNext(G2, t_from_new, func_next_neighb) 
            t_post_successors = []
            for t_ID_pre in t_from_successors:
                t_post_successors += extractNeighborsNext(G2, t_ID_pre, func_next_neighb)
            assert len(t_post_successors), "fake branch/es consists of multiple segments, not yet encountered"
            t_ID_post_succ  = t_post_successors[0]
            t_start         = G2.nodes[t_from_new]["t_end"]
            t_end           = G2.nodes[t_ID_post_succ]["t_start"]
            t_times         = {(t_from_new, t_ID_post_succ):np.arange(t_start, t_end + 1, 1)}
            t_event_start_end_times[t_state][t_from_new] = {'t_start':        t_start,
                                                      'post_successor': t_ID_post_succ,
                                                      'branches':       t_from_successors,
                                                      't_end':          t_end,
                                                      't_times':        t_times}

            #t_ref_ID = lr_C0_condensed_connections_relations[t_ID]
            t_segments_keep = [t_from_new] + t_from_successors + [t_ID_post_succ]
            #t_segments_keep = [lr_C0_condensed_connections_relations[t] for t in t_segments_keep]
            t_sol = graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments,
                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from_new][-1])
            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            
            #seqs = perms_with_branches(t_from_successors,t_segments_new,t_perms) 
            seqs, t_nodes_pre = perms_with_branches(t_from_successors, t_segments_new, t_perms, return_nodes = True) 

            t_event_start_end_times[t_state][t_from_new]['t_combs'] = seqs

            t_event_start_end_times[t_state][t_from_new]['t_nodes_solo'] = t_sol 

            t_conn = (t_from_new,t_ID_post_succ)
            for t_time, t_perms in t_nodes_pre.items():
                for t_perm in t_perms:
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))

                    t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                    lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
                    lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
                    lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z

        #t_state = 'split'
        #for t_ID in t_split_fake_from_ID:
        #    t_successors    = extractNeighborsNext(G2, t_ID, func_next_neighb) 
        #    t_post_successors = []
        #    for t_ID_pre in t_successors:
        #        t_post_successors += extractNeighborsNext(G2, t_ID_pre, func_next_neighb)
        #    assert len(t_post_successors), "fake branch/es consists of multiple segments, not yet encountered"
        #    t_ID_post_succ  = t_post_successors[0]
        #    t_start         = G2.nodes[t_ID]["t_end"]
        #    t_end           = G2.nodes[t_ID_post_succ]["t_start"]
        #    t_times         = {(t_ID, t_ID_post_succ):np.arange(t_start, t_end + 1, 1)}
        #    t_event_start_end_times[t_state][t_ID] = {'t_start':        t_start,
        #                                              'post_successor': t_ID_post_succ,
        #                                              'branches':       t_successors,
        #                                              't_end':          t_end,
        #                                              't_times':       t_times}

        #    t_ref_ID = lr_C0_condensed_connections_relations[t_ID]
        #    t_segments_keep = [t_ID] + t_successors + [t_ID_post_succ]
        #    t_segments_keep = [lr_C0_condensed_connections_relations[t] for t in t_segments_keep]
        #    t_sol = graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments,
        #                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1])
        #    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            
        #    #seqs = perms_with_branches(t_successors,t_segments_new,t_perms) 
        #    seqs,t_nodes_pre = perms_with_branches(t_successors,t_segments_new,t_perms, return_nodes = True) 

        #    t_event_start_end_times[t_state][t_ID]['t_combs'] = seqs

        #    t_event_start_end_times[t_state][t_ID]['t_nodes_solo'] = t_sol 

        #    t_conn = (lr_C0_condensed_connections_relations[t_ID],t_ID_post_succ)
        #    for t_time, t_perms in t_nodes_pre.items():
        #        for t_perm in t_perms:
        #            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))

        #            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

        #            lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
        #            lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
        #            lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z
            
        a = 1

        # ------------------------------------- mixed cases. for path extension -------------------------------------
        # for mixed i include target nodes into permutations. then i can check if they are a solution (not merge extension, but 121)
        t_state = 'mixed'
        for t_from,t_successors in lr_conn_splits_merges_mixed_dict.items():
            t_from_new = lr_C0_condensed_connections_relations[t_from]
            t_successors_times = {t:G2.nodes[t]["t_start"] for t in t_successors}
            t_start = G2.nodes[t_from_new]["t_end"]
            t_start2 = {t_from_new: t_start}
            t_time_max = max(t_successors_times.values())
            #t_end2 = {t_from_new: t_time_max}
            t_times         = {t:np.arange(t_start, t_end + 1, 1) for t,t_end in t_successors_times.items()}
            t_target_nodes = {t:t_segments_new[t][0] for t in t_successors}
            t_event_start_end_times[t_state][t_from_new] = {  't_start':        t_start2,
                                                              'branches':       t_successors,
                                                              't_end':          t_time_max,
                                                              't_times':        {},
                                                              't_perms':        {},
                                                              't_target_nodes': t_target_nodes}

            # pre-isolate graph segment where all sub-events take place
            
            t_nodes_keep    =  [node for node in G.nodes() if t_start <= node[0] < t_time_max] # dont take end point 
            t_subgraph      =  G.subgraph(t_nodes_keep)                                        
            # gather nodes for all targets together. its not known which is correct target it its not split or merge, but proximity event

            t_segments_keep = [t_from_new] #+  t_successors                                    # dont include body of earlier branch, or any..

            t_sol = graph_sub_isolate_connected_components(G, t_start, t_time_max, lr_time_active_segments,
                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from_new][-1])

            t_sol += [t_node for t_node in t_target_nodes.values()]                            # add firest nodes branches, so they are in perms 
            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
            #t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]    
            t_event_start_end_times[t_state][t_from_new]['t_perms'][t_from_new] = t_values
            t_event_start_end_times[t_state][t_from_new]['t_times'][t_from_new] = np.arange(t_start, t_time_max + 1, 1)

        #t_state = 'mixed'
        #for t_from,t_successors in lr_conn_splits_merges_mixed_dict.items():
        #    t_from_new = lr_C0_condensed_connections_relations[t_from]
        #    #t_successors   = extractNeighborsPrevious(G2, t_ID, func_prev_neighb) 
        #    t_successors_times = {t:G2.nodes[t]["t_start"] for t in t_successors}
        #    t_end = G2.nodes[t_from_new]["t_end"]
        #    t_times         = {(t_from_new,t):np.arange(t_start, t_end + 1, 1) for t,t_start in t_successors_times.items()}
        #    t_event_start_end_times[t_state][t_from] = {'t_start':        t_successors_times,
        #                                              'branches':       t_successors,
        #                                              't_end':          t_end,
        #                                              't_times':        t_times,
        #                                              't_combs':        []}

        #    # pre-isolate graph segment where all sub-events take place
        #    t_time_max = max(t_successors_times.values())
        #    t_nodes_keep    = [node for node in G.nodes() if t_end <= node[0] <= t_time_max] 
        #    t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
        #    # gather nodes for all targets together. its not known which is correct target it its not split or merge, but proximity event

        #    t_segments_keep = [t_from_new] +  t_successors 

        #    t_sol = graph_sub_isolate_connected_components(G, t_end, t_time_max, lr_time_active_segments,
        #                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from_new][-1])
        #    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
        #    t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]    
        #    t_event_start_end_times[t_state][t_ID]['t_combs'] = t_values

        #print(f'Fake merges({t_merge_fake_to_ID})/splits({t_split_fake_from_ID})... Done')

    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
# ===============================================================================================
# ====  Determine interpolation parameters ====
# ===============================================================================================
# we are interested in segments that are not part of psuedo branches (because they will be integrated into path)
# pre merge segments will be extended, post split branches also, but backwards. 
# for pseudo event can take an average paramets for prev and post segments.
print('Determining interpolation parameters k and s for segments...')
t_fake_branches_IDs = []
for t_ID in t_merge_fake_to_ID:
    t_fake_branches_IDs += t_event_start_end_times['merge'][t_ID]['branches']
for t_ID in t_split_fake_from_ID:
    t = lr_C0_condensed_connections_relations[t_ID]
    t_fake_branches_IDs += t_event_start_end_times['split'][t]['branches']
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

# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ===============================================================================================
# ====  deal with fake event recovery ====
# ===============================================================================================
print('Working on fake events...')
# -------------- determine segment start and end, find k_s, store info --------------------------

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
    #t_to    = lr_C0_condensed_connections_relations[t_to]
    t_from  = lr_C0_condensed_connections_relations[t_from]
    t_from_k_s  = t_segment_k_s[t_from  ]
    t_to_k_s    = t_segment_k_s[t_to    ]
    t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))   # determine which k_s to inherit (lower)
    t_fake_events_k_s_edge_master[t_to] = {'state': 'merge', 'edge': (t_from, t_to), 'k_s':t_k_s_out}
        
for t_from_old in t_split_fake_from_ID:
    t_from  = lr_C0_condensed_connections_relations[t_from_old]
    t_to = t_event_start_end_times['split'][t_from]['post_successor']
    #t_to    = lr_C0_condensed_connections_relations[t_to]
    t_from_k_s  = t_segment_k_s[t_from  ]
    t_to_k_s    = t_segment_k_s[t_to    ]
    t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))  
    t_fake_events_k_s_edge_master[t_from_old] = {'state': 'split', 'edge': (t_from, t_to), 'k_s':t_k_s_out}
# ----------------------- interpolate missing events with known history-----------------------
# interpolate
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print('Performing interpolation on data without fake branches')
a = 1
for t_ID,t_param_dict in t_fake_events_k_s_edge_master.items():
    t_state         = t_param_dict['state'  ] 
    t_from, t_to    = t_param_dict['edge'   ]
    k,s             = t_param_dict['k_s'    ]
    t_subIDs = [t_from, t_to]
    t_ID = lr_C0_condensed_connections_relations[t_ID]
    t_start = t_event_start_end_times[t_state][t_ID]['t_start'  ]
    t_end   = t_event_start_end_times[t_state][t_ID]['t_end'    ]
    if 1 == 1:
        t_temp_nodes_all = []
        for t_subID in t_subIDs:
            t_temp_nodes_all += t_segments_new[t_subID]
        t_temp_times        = [G.nodes[t_node]["time"       ] for t_node in t_temp_nodes_all]
        t_temp_areas        = [G.nodes[t_node]["area"       ] for t_node in t_temp_nodes_all]
        t_temp_centroids    = [G.nodes[t_node]["centroid"   ] for t_node in t_temp_nodes_all]
        
        t_times_missing_all = np.arange(t_start + 1, t_end, 1)
        
        a = 1
        # interpolate composite (long) parameters 
        t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, s = s, k = k, debug = 0, aspect = 'equal', title = t_subIDs)
        t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_subIDs)
        
        t_conn = (t_from, t_to)
        lr_big121s_interpolation[t_conn]['centroids'] = np.array(t_interpolation_centroids_0)
        lr_big121s_interpolation[t_conn]['areas'    ] = np.array(t_interpolation_areas_0    )   
        lr_big121s_interpolation[t_conn]['times'    ] = t_times_missing_all
# ----------------- Evaluate and find most likely evolutions of path -------------------------
print('Finding most possible path evolutions...')
if 1 == 1:
    t_temp_centroids = {}           # generate temp interpolation data storage
    t_temp_cases = {}
    t_temp_times = {}
    
    for t_ID,t_param_dict in t_fake_events_k_s_edge_master.items():
        t_ID = lr_C0_condensed_connections_relations[t_ID]
        t_state                     = t_param_dict['state'  ] 
        t_conn                      = t_param_dict['edge'   ]
        t_temp_centroids[t_conn]    = lr_big121s_interpolation[t_conn]['centroids']
        t_temp_cases[t_conn]        = t_event_start_end_times[t_state][t_ID]['t_combs']
        t_temp_times[t_conn]        = list(t_event_start_end_times[t_state][t_ID]['t_times'].values())[0]

    # combine interpolation and pre-computed options for bubble parameters. choose best fit to interp.
    t_args = [t_temp_cases,t_temp_centroids,t_temp_times,
            lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

    t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)

    t_weights   = [1,1.5,0,1]
    t_sols      = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
    lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_weights, t_sols, t_temp_cases )
a = 1 # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ----------------- Save recovered fake event paths -------------------------
print('Saving fake event data...')
if  1 == 1:
        #t_segments_new = segments2.copy()
        for t_ID_old,t_param_dict in t_fake_events_k_s_edge_master.items():
            t_ID = lr_C0_condensed_connections_relations[t_ID_old]
            t_state                     = t_param_dict['state'  ] 
            t_conn                      = t_param_dict['edge'   ]
            t_from, t_to = t_conn                                     
            t_IDs_remove = t_event_start_end_times[t_state][t_ID]['branches']
            t_sol   = lr_weighted_solutions_max[t_conn]               # pick evolution index that won (most likely)
            t_path  = t_temp_cases[t_conn][t_sol]             # t_path contains start-end points of segments !!!
            t_times = t_temp_times[t_conn]                    # get inter-segment times
            t_nodes_new = []
            for t_time,t_comb in zip(t_times,t_path):                 # create composite nodes
                t_nodes_new.append(tuple([t_time] + list(t_comb)))    # new type of nodes in solution: (time,*clusterIDs)   e.g (t1,ID1,ID2,...)

            t_nodes_all = t_event_start_end_times[t_state][t_ID]['t_nodes_solo']
            
            # basically in order to correctly reconnectd sides of inter-segment i have to also delete old edges from side nodes
            # this can be done by deleting them and adding back
            t_nodes_new_sides = [t_segments_new[t_from][-2]] + t_nodes_new + [t_segments_new[t_to][1]]
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
            t_segments_new[t_from_new] += t_segments_new[t_to]

    
            # fill centroid, area and momement of innertia zz missing for intermediate segment
            set_custom_node_parameters(G, g0_contours, t_nodes_intermediate, calc_hull = 1)
    
            # copy data from inherited
    
            # wipe data if t_from is inherited
            if t_from_new != t_from:
                t_segments_new[t_from] = []
            # wipe data from branches and t_to
            for t_ID_2 in t_IDs_remove + [t_to]:
                t_segments_new[t_ID_2] = []

            if t_state == 'merge':   
                lr_conn_merges_to_nodes.remove(t_ID_old) # its not real, remove
            else:
                lr_conn_splits_from_nodes.remove(t_ID_old)
            for t in t_IDs_remove:
                lr_C0_condensed_connections_relations[t] = t_from_new

            lr_C0_condensed_connections_relations[t_to] = t_from_new


G_seg_view_2 = nx.Graph()
G_seg_view_2.add_edges_from([(x,y) for y,x in lr_C0_condensed_connections_relations.items()])

#for g in G_seg_view_1.nodes():
#        G_seg_view_1.nodes()[g]["t_start"]    = segments2[g][0][0]
#        G_seg_view_1.nodes()[g]["t_end"]      = segments2[g][-1][0]

    
lr_C1_condensed_connections = extract_graph_connected_components(G_seg_view_2, lambda x: x)

# lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
t_condensed_connections_all_nodes = sorted(sum(lr_C1_condensed_connections,[])) # neext next
lr_C1_condensed_connections_relations = {tID: tID for tID in range(len(segments2))} 
for t_subIDs in lr_C1_condensed_connections:
    for t_subID in t_subIDs:
        lr_C1_condensed_connections_relations[t_subID] = min(t_subIDs)
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ====  EXTEND REAL MERGE/SPLIT BRANCHES ====
# ===============================================================================================
# extrapolate merge/split branches one time step at a time using prev determined k and s values
print('Analyzing real merge/split events. extending branches...')
t_merge_real_dict
t_split_real_dict

t_event_start_end_times[t_state]
t_segment_k_s_diffs
t_segment_k_s
lr_conn_splits_merges_mixed_dict
t_event_start_end_times['mixed']

t_out                   = defaultdict(dict)
t_extrapolate_sol       = defaultdict(dict)
t_extrapolate_sol_comb  = defaultdict(dict)
lr_extend_merges_IDs = list(t_merge_real_dict.keys()) + [lr_C1_condensed_connections_relations[t] for t in lr_conn_splits_merges_mixed_dict]
lr_mixed_completed      = {'full': defaultdict(dict),'partial': defaultdict(dict)} #AutoCreateDict()#
for t_ID in lr_extend_merges_IDs:
    if t_ID in t_merge_real_dict:
        t_state = 'merge'
        t_branches = t_event_start_end_times[t_state][t_ID]['branches']
        t_times_target = []
    else:
        t_state = 'mixed'
        t_branches = [t_ID] # dont have real branches, molding data to merge case
        t_nodes_target = t_event_start_end_times[t_state][t_ID]['t_target_nodes']
        t_times_target = [t[0] for t in t_nodes_target.values()]
        t_subIDs_target = {t:[] for t in t_times_target}
        for t_time, t_subIDs in t_nodes_target.values():
            t_subIDs_target[t_time] += t_subIDs

        
    
    for t_branch_ID in t_branches:
        t_conn = (t_branch_ID, t_ID)
        t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'][t_branch_ID] # last of branch
        t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end']                # first of target
        if t_t_to - t_t_from < 2:
            t_out[t_ID][t_branch_ID] = None
            continue
        t_branch_ID_new = lr_C1_condensed_connections_relations[t_branch_ID] # probly not needed anymore
        t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if t_node[0] > t_t_from - h_interp_len_max2]
        trajectory = np.array([G.nodes[t]["centroid"] for t in t_nodes])
        time       = np.array([G.nodes[t]["time"    ] for t in t_nodes])
        #trajectory          = np.array([G.nodes[t]["centroid"] for t in t_segments_new[t_branch_ID_new]])
        #time                = np.array([G.nodes[t]["time"    ] for t in t_segments_new[t_branch_ID_new]])
        
        N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
        if t_segment_k_s_diffs[t_branch_ID_new] is not None:
            t_last_deltas   = list(t_segment_k_s_diffs[t_branch_ID_new].values())[-N:]
        else:
            t_last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5

        t_norm_buffer   = CircularBuffer(N,                 t_last_deltas                    )
        t_traj_buff     = CircularBuffer(h_interp_len_max2, trajectory[  -h_interp_len_max2:])
        t_time_buff     = CircularBuffer(h_interp_len_max2, time[        -h_interp_len_max2:])
        t_time_next     = t_t_from + 1
        t_extrapolate_sol[t_conn] = {}
        t_extrapolate_sol_comb[t_conn] = {}
        t_times_accumulate_resolved = []
        if t_segment_k_s[t_branch_ID_new] is not None:
            t_k,t_s = t_segment_k_s[t_branch_ID_new]
        else:
            t_k,t_s = (1,5)

        t_combs_all = t_event_start_end_times[t_state][t_ID]['t_perms'][t_branch_ID][1:]
        t_times_all = t_event_start_end_times[t_state][t_ID]['t_times'][t_branch_ID][1:] # middle + target first
        if t_state != 'mixed':
            t_combs_all = t_combs_all[:-1]
            t_times_all = t_times_all[:-1]

        for t_time, t_permutations in zip(t_times_all,t_combs_all):
            # extrapolate traj
            t_extrap = interpolate_trajectory(t_traj_buff.get_data(), t_time_buff.get_data(), which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]

            # get possible permutations
            a = 1
            #t_permutations = sum([list(itertools.combinations(t_comb, r)) for r in range(1,len(t_comb)+1)],[])
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
                
                t_extrapolate_sol[t_conn][t_time] = t_extrap
                t_extrapolate_sol_comb[t_conn][t_time] = t_permutations[t_where_min]
                #print(f"inter!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
                t_times_accumulate_resolved.append(t_time)
            else:

                a = 1
                #print(f"stdev too much!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
                if len(t_times_accumulate_resolved) > 0:
                    print(f'recovery of t:{t_time_next} has failed, segment {t_branch_ID_new}->{t_ID} (end: {t_segments_new[t_branch_ID_new][-1]}) extended trajectory to times: {t_times_accumulate_resolved[0]}...{t_times_accumulate_resolved[-1]}')
                break

            if t_time == t_t_to - 1 and len(t_times_accumulate_resolved) > 0:
                print(f'recovering t:{t_time_next}, segment {t_branch_ID_new}->{t_ID} (end: {t_segments_new[t_branch_ID_new][-1]}) extended trajectory to times: {t_times_accumulate_resolved[0]}...{t_times_accumulate_resolved[-1]}. Recovered whole inter-segment!')
            
            if t_state == 'mixed':
                if t_time in t_times_target:
                    t_nodes_sol_subIDs = list(t_permutations[t_where_min])
                    t_nodes_solution = tuple([t_time] + t_nodes_sol_subIDs)

                    if t_nodes_solution in t_nodes_target.values():
                        t_target_ID = find_key_by_value(t_nodes_target,t_nodes_solution)
                        lr_mixed_completed['full'][t_ID]['solution'] = t_nodes_sol_subIDs
                        if 'targets' not in lr_mixed_completed['full'][t_ID]: lr_mixed_completed['full'][t_ID]['targets'] = []
                        lr_mixed_completed['full'][t_ID]['targets'].append(t_target_ID)
                    else:
                        set1 = set(t_subIDs_target[t_time])
                        set2 = set(t_nodes_sol_subIDs)
                        inter = set1.intersection(set2)

                        if len(inter)> 0:
                            lr_mixed_completed['partial'][t_ID]['solution'] = t_nodes_sol_subIDs
                            #if 'targets' not in lr_mixed_completed['partial'][t_ID]: lr_mixed_completed['partial'][t_ID]['targets'] = []
                            #lr_mixed_completed['partial'][t_ID]['targets'].append(t_target_ID)

                #if t_time == t_times_all[-1]:


            t_time_next  += 1


# search for conflict nodes for extended branches
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
lr_conn_merges_good = defaultdict(list)#{tID:[] for tID in lr_conn_merges_to_nodes}
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
    for tID in lr_extend_merges_IDs:
        if tID not in t_all_problematic_conns_to:
            t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
            lr_conn_merges_good[tID] += t_conns_relevant

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
        print('branches are resolved without conflict, or conflict resolved by redistribution of nodes')
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# weird saving system. copied from old (this) code.
if 1 == 1:
    for t_to,t_conns in lr_conn_merges_good.items():
        for t_conn in t_conns:
            t_from, t_to = t_conn
            t_from_new = lr_C1_condensed_connections_relations[t_from]
            t_combs = t_extrapolate_sol_comb[t_conn]

            if len(t_combs) == 0: continue                              # no extension, skip.

            t_nodes_all = []                                            # solo nodes will be deleted
            t_nodes_new = []                                            # new composite nodes are formed

            if t_from not in lr_mixed_completed['full']:
                for t_time,t_subIDs in t_combs.items():

                    for t_subID in t_subIDs:                                # single object nodes
                        t_nodes_all.append((t_time,t_subID))

                    t_nodes_new.append(tuple([t_time] + list(t_subIDs)))    # cluster object nodes
        
                t_nodes_all += [t_segments_new[t_from_new][-1]]                 # (*) last old may have multiple connections not
                                                                            # covered by best choices in sol-n, so they are will stay.

                # find end edges to reconnect into graph
                # if last node is composite of solo nodes, each might have its edges
                t_time_end, *t_node_subIDs =  t_nodes_new[-1]               # grab subIDs and recover solo nodes. i do this because
                t_node_last_solo_IDS = [(t_time_end,t_subID) for t_subID in t_node_subIDs] #  its easy to pick [-1], not lookup last time.

                #f = lambda x: x[0]
                t_node_last_solo_IDS_to = list(set(sum([list(G.successors(t_node)) for t_node in t_node_last_solo_IDS],[])))
                #t_node_last_solo_IDS_to = list(set(sum([extractNeighborsNext(G, t_node, f) for t_node in t_node_last_solo_IDS],[])))

                t_edges_next_new = [(t_nodes_new[-1] , t_node) for t_node in t_node_last_solo_IDS_to] # composite to old next neighbors

                G.remove_nodes_from(t_nodes_all)                            # remove all solo nodes and replace them with composite
            
                t_nodes_new_sides = t_segments_new[t_from_new][-2:] + t_nodes_new#  take nodes from -2 to end of segment, why -2 see (*)

                t_edges_sequence = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
            
                G.add_edges_from(t_edges_sequence)                          # from sequence of nodes create edges -> auto nodes
                G.add_edges_from(t_edges_next_new)                          # add future edges to solo nodes.

                t_segments_new[t_from_new] += t_nodes_new                       # append resolved to segments.

                set_custom_node_parameters(G, g0_contours, t_nodes_new, calc_hull = 1)
            


            else:
                assert len(lr_mixed_completed['full'][t_from]['targets']) == 1, 'proximity pseudo merge passes multiple nodes, not resolved/encountered'
                t_to = lr_mixed_completed['full'][t_from]['targets'][0]
                t_node_start_last = t_segments_new[t_from_new][-1]
                t_times_relevant = t_event_start_end_times['mixed'][t_from]['t_times'][t_from]
                t_combs[t_node_start_last[0]] = tuple(t_node_start_last[1:])
                t_combs = {t:t_combs[t] for t in t_times_relevant}

                for t_time,t_subIDs in t_combs.items():

                    for t_subID in t_subIDs:                                # single object nodes
                        t_nodes_all.append((t_time,t_subID))

                    t_nodes_new.append(tuple([t_time] + list(t_subIDs)))    # cluster object nodes



                # basically in order to correctly reconnectd sides of inter-segment i have to also delete old edges from side nodes
                # this can be done by deleting them and adding back
                t_nodes_new_sides = [t_segments_new[t_from_new][-2]] + t_nodes_new + [t_segments_new[t_to][1]]
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
                
                t_nodes_intermediate = list(sorted(t_nodes_new, key = lambda x: x[0]))[1:-1]
                t_segments_new[t_from_new] += t_nodes_intermediate
                t_segments_new[t_from_new] += t_segments_new[t_to]

    
                # fill centroid, area and momement of innertia zz missing for intermediate segment
                set_custom_node_parameters(G, g0_contours, t_nodes_intermediate, calc_hull = 1)
    
                
                # wipe data from t_to anyway
                t_segments_new[t_to] = []

                t_successors   = extractNeighborsNext(G2, t_to, lambda x: G2.nodes[x]["t_start"])
                t_edges = [(t_from_new,t_succ) for t_succ in t_successors]
                G2.remove_nodes_from([t_to])
                G2.add_edges_from(t_edges)
                G2.nodes()[t_from_new]["t_end"] = t_segments_new[t_from_new][-1][0] # change master node parameter

                lr_C1_condensed_connections_relations[t_to] = t_from_new
                a = 1

tt = []
for t_node in G.nodes():
    if "time" not in G.nodes[t_node]:
        set_custom_node_parameters(G, g0_contours, [t_node], calc_hull = 1)
        
        tt.append(t_node)
print(f'were missing: {tt}')
#for_graph_plots(G, segs = t_segments_new)

# ===============================================================================================
# ===============================================================================================
# ================================ Final passes. Find stray nodes ===============================
# ===============================================================================================
# WHY: some contours were dropped from cluster due to short trajectory history. 
# HOW: take original proximity relations and analyze edges of resolved nodes for each trajectory
# HOW: for convinience, you can strip certain connections/edges:
# HOW: 1) edges connecting trajectory together (known/not relevant).
# HOW: 2) edges between resolved segments ( are impossible)
# HOW: walk though resolved trajectory nodes and check remaining connections =  stray nodes.
# =============================================

G_OG = nx.Graph()                                                               # recreate original connectivity, this graph
G_OG.add_edges_from(g0_pairConnections2_OG)                                     # still contains only solo nodes, not composite

lr_time_active_segments = defaultdict(list)                                     # prepare for finding edges between segments
for k,t_segment in enumerate(t_segments_new):
    t_times = [a[0] for a in t_segment]
    for t in t_times:
        lr_time_active_segments[t].append(k)

t_nodes_resolved_per_segment = {t:defaultdict(list) for t,_ in enumerate(t_segments_new)}  
for t,t_segment in enumerate(t_segments_new):                                   # deconstruct resolved composite nodes  
    for t_time,*t_subIDs in t_segment:                                          # into time and contour ID dicts
        t_nodes_resolved_per_segment[t][t_time] += t_subIDs


t_edges_all = []                                                                # extract edges that hold segments together
for t_segment in t_segments_new:
    t_edges_big = [(x, y) for x, y in zip(t_segment[:-1], t_segment[1:])]       # edges between composite nodes ((T1,1,2),(T2,3,4)
    for t_node_from, t_node_to in t_edges_big:

        t_subIDs_from   , t_subIDs_to   = t_node_from[1:]   , t_node_to[1:]     # extract only IDs
        t_time_from     , t_time_to     = t_node_from[0]    , t_node_to[0]

        t_subIDs_edge_prod = list(itertools.product(t_subIDs_from, t_subIDs_to))# calculate solo connections: [(1,3),(1,4),(2,3),(2,4)]

        for t_from, t_to in t_subIDs_edge_prod:
            t_edges_all.append(((t_time_from, t_from),(t_time_to,t_to)))        # add times (1,3) -> ((T1,1),(T2,3))

G_OG.remove_edges_from(t_edges_all)                                             # remove from OG graph

t_segment_overlap = find_common_intervals(lr_time_active_segments)              # extract possible connections between overlapping
for (t_seg_1, t_seg_2), t_times in t_segment_overlap.items():                   # contours. only possible edges where abs(t1-t2) = 1

    t_times_staggered = [(x, y) for x, y in zip(t_times[:-1], t_times[1:])]     # list of two consequitive time steps (t1,t2)
    for t_1,t_2 in t_times_staggered:
        t_edges_inter_segment = []
        
        t_1_subIDs_1 = t_nodes_resolved_per_segment[t_seg_1][t_1]               # extract subIDs for segment 1 at t1 and 
        t_2_subIDs_1 = t_nodes_resolved_per_segment[t_seg_2][t_2]               # subIDs for segment 2 for t2
        
        t_1_subIDs_2 = t_nodes_resolved_per_segment[t_seg_2][t_1]               # extract subIDs for segment 2 at t1 and 
        t_2_subIDs_2 = t_nodes_resolved_per_segment[t_seg_1][t_2]               # subIDs for segment 1 for t2
       
        t_prod_1 = list(itertools.product(t_1_subIDs_1, t_2_subIDs_1))          # for connections between two cases. 
        t_prod_2 = list(itertools.product(t_1_subIDs_2, t_2_subIDs_2))          # order t1<t2 still holds
        
        for t_from, t_to in t_prod_1 + t_prod_2:                                # combine connections together and add times
            t_edges_inter_segment.append(((t_1, t_from),(t_2,t_to))) 

        G_OG.remove_edges_from(t_edges_inter_segment)                           # remove edges


t_segment_stray_neighbors = defaultdict(set)
for t,t_segment in enumerate(t_segments_new):
    for t_time,*t_subIDs in t_segment[1:-1]:                                    # walk composite nodes of resolved segments
        for t_subID in t_subIDs:                                                
            t_node = (t_time,t_subID)                                           # decompose nodes to fit OG graph
            t_neighbors = G_OG.neighbors(t_node)                                # check remaining connections
            t_segment_stray_neighbors[t].update(t_neighbors)                    # add stray nodes to storage



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
            
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t_subID] for t_subID in subIDs]))
            x,y,w,h = cv2.boundingRect(t_hull)
            
            cv2.drawContours(  imgs[t_time],  [t_hull], -1, cyclicColor(n), 2)
            [cv2.putText(imgs[t_time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        t_times_all = [t[0] for t in case]
        t_centroids_all = [G.nodes[t]["centroid"] for t in case]
        for k,subCase in enumerate(case):
            t_time,*subIDs = subCase
            t_k_from = max(0, k - 10)
            t_times_sub = t_times_all[t_k_from:k]
            pts = np.array(t_centroids_all[t_k_from:k]).reshape(-1, 1, 2).astype(int)
            
            cv2.polylines(imgs[t_time], [pts] ,0, (255,255,255), 3)
            cv2.polylines(imgs[t_time], [pts] ,0, cyclicColor(n), 2)
            [cv2.circle(imgs[t_time], tuple(p), 3, cyclicColor(n), -1) for [p] in pts]

            for subID in subIDs:
                startPos2 = g0_contours[t_time][subID][-30][0] 
                [cv2.putText(imgs[t_time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]

    for k,img in enumerate(imgs):
        folder = r"./post_tests/testImgs/"
        fileName = f"{str(k).zfill(4)}.png"
        cv2.imwrite(os.path.join(folder,fileName) ,img)
        #cv2.imshow('a',imgs[time])


#for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# for_graph_plots(G)
a = 1
#if useMeanWindow == 1:
#    meanImage = masksArr[whichMaskInterval(globalCounter,intervalIndecies)]

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()