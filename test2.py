import glob, csv, cv2, numpy as np, pickle, os
#imgPath = r'test\csv_img-Field OFF Series 7-sccm100-meanFix-00001-05000'
#imageLinks = glob.glob(imgPath + "**/*.png", recursive=True) 
#a = 1
#data = []
#with open(r'test\csv-Field OFF Series 7-sccm100-meanFix-00001-05000.csv', 'r', newline='') as file:
#    # Create a CSV writer
#    reader = csv.reader(file)
#    for row in reader:
#        temp = int(row[0]),int(row[1]),eval(row[2]),int(row[3])
#        data.append(temp)
#a = 1
#maxFrame = data[-1][0]
#cntr = 0
#locCntr = 0
#for frame, localID, centroid, area in data:
#    if frame > cntr:
#        locCntr = 0
#        cntr = frame
#        cv2.imwrite(".\\test\\output\\"+str(frame-1).zfill(4)+".png" ,pic)
#    if locCntr == 0:
#        pic = np.uint8(cv2.imread(imageLinks[cntr],1))
#    if frame == cntr:
#        cv2.circle(pic, centroid, 10, (0,0,255), 2)
#        locCntr +=1
#    #if frame > 3: break
#    a = 1
#lnk = r'C:\Users\mhd01\Desktop\mag\traj family\off'
#imageLinks = glob.glob(lnk + "**/*.png", recursive=True) 
#imgs = [np.uint8(cv2.imread(lnk,1)) for lnk in imageLinks]
#imgShapes =  np.array([a.shape[:-1] for a in imgs])
#maxX,maxY = [max(a) for a in imgShapes.T]
#new_image_height = maxX
#new_image_width = maxY
#channels = 3
#color = (255,255,255)
#for i in range(len(imgs)):
#    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
#    img = imgs[i]
#    old_image_height,old_image_width = imgShapes[i]
#    # compute center offset
#    x_center = (new_image_width - old_image_width) // 2
#    y_center = (new_image_height - old_image_height) // 2

#    # copy img image into center of result image
#    result[y_center:y_center+old_image_height, 
#           x_center:x_center+old_image_width] = img
#    cv2.imwrite(os.path.join(lnk,f"padded_{os.path.basename(imageLinks[i])}.png"), result)
# view result
#cv2.imshow("result", result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Import the required libraries
import os
import sys
import cv2

import numpy as np, pickle

#with open('./asd.pickle', 'rb') as handle:
#    a,b = pickle.load(handle)
#c,s,an = a
#rect1 = (tuple(map(int,c)), tuple(map(int,s)), int(an))

#c,s,an = b
#rect2 = (tuple(map(int,c)), tuple(map(int,s)), int(an))
#print(rect1 == a)
#print(rect2 == b)
#interType, points  = cv2.rotatedRectangleIntersection(rect1, rect2)
#interType, points  = cv2.rotatedRectangleIntersection(a, b)


## Create the graph (example)
#G = nx.Graph()
#G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)])

## Define the path
#path = [1, 2, 3, 4, 5]

## Create a dictionary to store edge colors
#edge_colors = []

## Iterate over the edges and check if they belong to the path
#for u, v in G.edges():
#    if u in path and v in path:
#        edge_colors.append((1,0,0))
#    else:
#        edge_colors.append('black')


## Draw the graph with colored edges
#pos = nx.spring_layout(G)
#nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color=edge_colors)
#plt.show()
#a =1

#def centroid_area_cmomzz(contour):
#    m = cv2.moments(contour)
#    area = int(m['m00'])
#    cx0, cy0 = m['m10'], m['m01']
#    centroid = np.array([cx0,cy0])/area
#    c_mom_xx, c_mom_yy = m['mu20'], m['mu02']
#    c_mom_zz = int((c_mom_xx + c_mom_yy))
#    return  centroid, area, c_mom_zz

#import cv2
#import numpy as np

## Create a black image as the background
#image0 = np.zeros((200, 200), dtype=np.uint8)

## Define the center and radius of the circle
#center = (100, 100)
#radius = 50

## Draw the filled circle on the image
#cv2.circle(image0, center, radius, (255, 255, 255), -1)

## Find contours in the image
#contours, hierarchy = cv2.findContours(image0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

## Extract the contour of the filled circle (assuming there is only one contour)
#circle_contour = contours[0]

#mask = image0.copy()
#M = cv2.moments(mask)

## Calculate the centroid coordinates
#cx = int(M['m10'] / M['m00'])
#cy = int(M['m01'] / M['m00'])

## Calculate the moment of inertia
#moment_of_inertia = 0.0
#for y in range(mask.shape[0]):
#    for x in range(mask.shape[1]):
#        if mask[y, x] == 255:  # Inside the object
#            squared_distance = ((x - cx) ** 2 + (y - cy) ** 2)
#            moment_of_inertia += squared_distance

#print("Moment of inertia:", moment_of_inertia)

#moment_of_inertia_circle = moment_of_inertia

## Create a black image as the background
#image = np.zeros((200, 200), dtype=np.uint8)

## Define the top-left and bottom-right corners of the square
#top_left = (75, 75)
#bottom_right = (125, 125)

## Draw the filled square on the image
#cv2.rectangle(image, top_left, bottom_right, 255, -1)
##cv2.imshow("Image", image)
## Find contours in the image
#contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

## Extract the contour of the filled square (assuming there is only one contour)
#square_contour = contours[0]

## Calculate the moments of the filled shape
#mask = image.copy()
#M = cv2.moments(mask)

## Calculate the centroid coordinates
#cx = int(M['m10'] / M['m00'])
#cy = int(M['m01'] / M['m00'])

## Calculate the moment of inertia
#moment_of_inertia = 0.0
#for y in range(mask.shape[0]):
#    for x in range(mask.shape[1]):
#        if mask[y, x] == 255:  # Inside the object
#            squared_distance = ((x - cx) ** 2 + (y - cy) ** 2)
#            moment_of_inertia += squared_distance

#print("Moment of inertia:", moment_of_inertia)
#moment_of_inertia_rect = moment_of_inertia
#s=  50
#hs = 50/2
#ar0 = s**2
## 4 x quater segment
#cm0 = s**4/6

#aa0 = centroid_area_cmomzz(square_contour)


## Display the image with the filled circle
##cv2.imshow("Image", image)
#ar = np.pi*radius**2
#aa = centroid_area_cmomzz(circle_contour)
#cm = 1/4*radius**4 * 2* np.pi 
#moment_of_inertia_circle
#a = 1


from collections import deque
import numpy as np

#class TrajectoryBuffer:
#    def __init__(self, max_size):
#        self.max_size = max_size
#        self.buffer = deque(maxlen=max_size)

#    def append(self, trajectory):
#        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (2,):
#            raise ValueError("Trajectory should be a 1x2 NumPy array.")
        
#        if len(self.buffer) == self.max_size:
#            self.buffer.popleft()
#        self.buffer.append(trajectory)

#    def get_data(self):
#        return np.array(self.buffer)

#    def get_last_trajectory(self):
#        if len(self.buffer) > 0:
#            return self.buffer[-1]
#        else:
#            return None

#d = deque(maxlen=5)
#d.extend(np.array([[0,0],[1,1],[2,2],[3,3],[4,4]]))
#print(d)
## deque([1, 2, 3, 4, 5], maxlen=5)
#d.append([5,5])
#print(d)
## deque([2, 3, 4, 5, 10], maxlen=5)

## Create the TrajectoryBuffer with a maximum size of 3
#trajectory_buffer = TrajectoryBuffer(max_size=3)

## Append 1x2 arrays to the buffer
#trajectory_buffer.append(np.array([1, 2]))
#trajectory_buffer.append(np.array([3, 4]))
#trajectory_buffer.append(np.array([5, 6]))

## Check the buffer content
#print(trajectory_buffer.get_data())

## Append a new 1x2 array, which will trigger the shift in the buffer
#trajectory_buffer.append(np.array([7, 8]))

## Check the buffer content after the shift
#print(trajectory_buffer.get_data())

import random
import timeit
import numpy as np
import pandas as pd
from collections import Counter

import timeit
if 1 == -1:
    def old():
        for t_conn in lr_relevant_conns:
            t_traj = lr_close_segments_simple_paths[t_conn][0]
            t_min = t_traj[0][0]
            t_max = t_traj[-1][0]
            t_nodes = {t:[] for t in np.arange(t_min, t_max + 1, 1)}
            for t_traj in lr_close_segments_simple_paths[t_conn]:
                for t_time,*t_subIDs in t_traj:
                    t_nodes[t_time] += t_subIDs
            for t_time in t_nodes:
                t_nodes[t_time] = sorted(list(set(t_nodes[t_time])))
            return t_nodes

    def new():
        for t_conn in lr_relevant_conns:
            t_traj = lr_close_segments_simple_paths[t_conn][0]
            t_min = t_traj[0][0]
            t_max = t_traj[-1][0]
            t_nodes = {t:[] for t in np.arange(t_min, t_max + 1, 1)}
            M = np.array([t[1] for sublist in lr_close_segments_simple_paths[t_conn] for t in sublist],np.uint8).reshape((-1,len(t_traj)))
            unique_elements_per_column = [np.unique(sl) for sl in M.T]
            return {key: sorted(list(value)) for key, value in zip(t_nodes, unique_elements_per_column)}
    
    time_method1 = timeit.timeit(old,       number=10000)
    time_method2 = timeit.timeit(new,   number=10000)

    print("Method 1 execution time:", time_method1)
    print("Method 2 execution time:", time_method2)
if 1 == -1:
    def set_elements_at_depth(indices, entry, *nested_lists):
        current_lists = nested_lists
        for idx in indices[:-1]:
            current_lists = [lst[idx] for lst in current_lists]
        for lst, val in zip(current_lists, entry):
            lst[indices[-1]] = val

    nested_list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    nested_list2 = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

    entry_list = [100, 200]  # Modify first elements at depth [0][0] for both nested lists

    set_elements_at_depth([2, 0], entry_list, nested_list1, nested_list2)

    print(nested_list1)  # Output: [[100, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(nested_list2)  # Output: [[200, 11, 12], [13, 14, 15], [16, 17, 18]]

    sorted([(10,), (10,), (11,), (9,), (9,)], key=lambda x: (x[0], *x[1:]))
    sorted([5, 1], key=lambda x: (x[0], *x[1:]))


import networkx as nx, copy, itertools
if 1 == -1:
    options = {
        'n1': ['c1', 'c2'],
        'n2': ['c3', 'c4']
    }
    variants = []
    for t,(key,values) in enumerate(options.items()):
        sd  = list(itertools.product([key],values))
        variants.append(sd)
    all_variants = list(itertools.product(*variants))
    a = 1
    #permutations = [
    #    ([(n1, c1)], [(n2, c2)])
    #    for (n1, c1), (n2, c2) in product(options.items(), repeat=2)
    #]

    #print(permutations)
    a = 1
if 1 == -1:
    t_extrapolate_sol_comb = {
        (0, 3): {483: (3,), 484: (4,), 485: (5,), 486: (6,), 487: (8,), 488: (4,), 489: (2, 6), 490: (7,)},
        (2, 3): {483: (1,), 484: (1,), 485: (1,), 486: (1,), 487: (1,5), 488: (0,), 489: (2,)},
        (5, 3): {483: (2,), 484: (2,), 485: (2,), 486: (2,), 487: (2,1), 488: (2,), 489: (1,)}
    }
    import copy, itertools
    from collections import defaultdict
    node_properties = defaultdict(list)
    # to find duplicates just add owners to each node. multiple owners = contested.
    for key, dic in t_extrapolate_sol_comb.items():
        t_nodes =  [(key, value) for key, values in dic.items() for value in values]
        for t_node in t_nodes:
            node_properties[t_node].extend([key])
    t_duplicates = {t_node: t_branches for t_node,t_branches in node_properties.items() if len(t_branches) > 1}
    # each contested node is an option. each choice of owner produces a different configuration branch
    variants = [] # generate these options for each contested node
    for t,(key,values) in enumerate(t_duplicates.items()):
        sd  = list(itertools.product([key],values))
        variants.append(sd)
    # generate all possible evoltions via product permutation: choice_1 -> choice_2 -> ...
    variants_all = list(itertools.product(*variants))
    # check if other paths, which will have at least one remaining node after deletion.
    variants_possible = []
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



if 1 == 1:
    t_extrapolate_sol_comb = {
        (0, 3): {483: (3,), 484: (4,), 485: (5,), 486: (6,), 487: (8,), 488: (4,), 489: (6,), 490: (7,)},
        (2, 3): {483: (1,), 484: (1,), 485: (1,), 486: (1,), 487: (1,5), 488: (0,), 489: (2,)},
        (5, 6): {483: (2,), 484: (2,), 485: (2,), 486: (2,), 487: (2,), 488: (2,), 489: (1,)}
    }

    lr_extend_merges_IDs = [(0, 'merge'), (2, 'merge'), (3, 'merge')]
    import copy, itertools
    from collections import defaultdict
    lr_conn_merges_good = defaultdict(set)

    def conflicts_stage_1(owner_dict):
        # find contested nodes by counting their owners
        node_owners = defaultdict(list)
        for owner, times_subIDs in owner_dict.items():
            # {owner:nodes,...}, where nodes = {time:*subIDs,...} = {483: (3,), 484: (4,),..}
            nodes =  [(time, subID) for time, subIDs in times_subIDs.items() for subID in subIDs]
            for node in nodes:
                node_owners[node].extend([owner])

        return {node: owners for node,owners in node_owners.items() if len(owners) > 1}

    t_duplicates = conflicts_stage_1(t_extrapolate_sol_comb)

    #assert len(t_duplicates) == 0, 'havent tested after addition of split and mixed extension code 28.09.23'
    t_all_problematic_conns = list(set(sum(list(t_duplicates.values()),[])))
    t_all_problematic_conns_to = [a[1] for a in t_all_problematic_conns]
    for tID,t_state in lr_extend_merges_IDs: # here lies problem with splits, and possibly with mixed cases!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if tID not in t_all_problematic_conns_to:
            if t_state == 'merge':
                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
            elif t_state == 'split':
                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[0] == tID]
            else:
                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
            lr_conn_merges_good[(tID,t_state)].update(t_conns_relevant)
            #lr_conn_merges_good.update((tID,t_state))

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
        return list(itertools.product(*node_owner_choices)) if len(node_owner_choices) > 0  else []

    

    variants_all = conflicts_stage_2(t_duplicates)

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

    variants_possible = conflicts_stage_3(variants_all,t_duplicates, t_extrapolate_sol_comb)


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
            lr_conn_merges_good.update(t_conns_relevant) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< not correct anymore
            a = 1
        print('branches are resolved without conflict, or conflict resolved by redistribution of nodes')

if 1 == -1:
    from collections import defaultdict

    fin_segments_conns_all = [(0, 1), (0, 5), (1, 2), (3, 4)]

    fin_segments_conns_all_dict = defaultdict(dict)
    for t_from, t_to in fin_segments_conns_all:
        fin_segments_conns_all_dict[t_from][t_to] = -1
    a = 1
    print(1)
if  1 == -1:
    def find_subset_around_unknown(T_known, T_unknown, DT):
        subset = []

        for t in T_known:
            if abs(t - T_unknown) <= DT/2:
                subset.append(t)

        return subset

    T_known = [1, 2, 4, 5]
    T_unknown = 3
    DT = 2

    result = find_subset_around_unknown(T_known, T_unknown, DT)
    print(result)  # Output: [2, 4]

if 1 == -1:
    from collections import defaultdict

    def nested_defaultdict(initialize_value):
        return defaultdict(lambda: initialize_value)

    aa = nested_defaultdict(1)
    def lr_init_perm_precomputed(possible_permutation_dict, initialize_value):

        output = defaultdict(tuple)

        for t_conn, t_times_perms in possible_permutation_dict.items():
            output[t_conn] = defaultdict(defaultdict(int))
            for t_time, t_perms in t_times_perms.items():
                for t_perm in t_perms:

                    output[t_conn][t_time] = {t_perm: initialize_value for t_perm in t_perms}
        
        return output

    def lr_init_perm_precomputed0(possible_permutation_dict, initialize_value):
        return {t_conn: {t_time: 
                                {t_perm:initialize_value for t_perm in t_perms}
                         for t_time,t_perms in t_times_perms.items()}
               for t_conn,t_times_perms in possible_permutation_dict.items()}

    #class AutoCreateDict:
    #    def __init__(self):
    #        self.data = {}

    #    def __getitem__(self, key):
    #        if key not in self.data:
    #            self.data[key] = AutoCreateDict()
    #        return self.data[key]

    #    def __setitem__(self, key, value):
    #        self.data[key] = value
    class AutoCreateDict:
        def __init__(self):
            self.data = {}

        def __getitem__(self, key):
            if key not in self.data:
                self.data[key] = AutoCreateDict()
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = value

        def keys(self):
            return self.data.keys()

        def values(self):
            return self.data.values()

        def items(self):
            return self.data.items()

        def __repr__(self):
            return repr(self.data)


    # Create an instance of AutoCreateDict
    lr_big121s_perms_areas = AutoCreateDict()

    # Use it to store values
    lr_121_stray_disperesed = {(2,3):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]},(10,11):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]}}
    for t_conn, t_dict in lr_121_stray_disperesed.items():
        for t_time,t_perms in t_dict.items():
            for t_perm in t_perms:
                lr_big121s_perms_areas[     t_conn][t_time][t_perm] = 52

    # Access the stored value
    stored_area = lr_big121s_perms_areas[t_conn][t_time][t_perm]
    print(stored_area)  # Output: 42


    #lr_121_stray_disperesed = {(2,3):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]},(10,11):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]}}
    #lr_big121s_perms = {t_conn:{t_time:t_perms for t_time,t_perms in t_dict.items()} for t_conn,t_dict in lr_121_stray_disperesed.items()}
    #lr_big121s_perms_areas0      = lr_init_perm_precomputed0(lr_big121s_perms,0)
    #lr_big121s_perms_areas      = lr_init_perm_precomputed(lr_big121s_perms,0)
    #lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
    #lr_big121s_perms_areas[(5,14)] = {}
    a = 1


if 1 == -1:
    import cv2
    import numpy as np

    # Define pts as a list of points
    pts = np.array([[[1070.94817927,  370.86694678]]]).astype(int)

    # Create an image
    imgs = np.zeros((500, 500, 3), dtype=np.uint8)

    # Draw the polyline
    cv2.polylines(imgs, [pts], isClosed=False, color=(255, 255, 255), thickness=3)

    # Show the image
    cv2.imshow("Image", imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if 1 == -1:
    from collections import defaultdict
    import timeit

    def find_common_intervals2(data):
        element_ids = set()
        for element_list in data.values():
            element_ids.update(element_list)

        active_times = defaultdict(list)
        for time, elements in data.items():
            for element in elements:
                active_times[element].append(time)
        element_pairs = list(itertools.combinations(element_ids, 2))
        common_times_dict = defaultdict(list)
        for pair in element_pairs:
            element_1, element_2 = pair
            active_times_1 = set(active_times[element_1])
            active_times_2 = set(active_times[element_2])
            common_times = active_times_1.intersection(active_times_2)
            if len(common_times) > 0:
                common_times_dict[pair] = common_times


        return common_times_dict
if 1 == -11:
    from collections import defaultdict

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

    # Sample data
    data = {
        128: [0],
        129: [0],
        130: [0, 1],
        131: [0, 1],
        132: [0, 1, 3],
        133: [0, 1, 3],
        134: [1, 3],
        135: [1, 3],
        136: [3],
        137: [3],
        138: [3],
    }


    from collections import defaultdict

    import random

    def generate_test_data(min_time, max_time, num_elements):
        data = {t:[] for t in range(min_time, max_time + 1)}
    
        for element in range(num_elements):
            start_time = random.randint(min_time, max_time)
            end_time = start_time + random.randint(1, max_time - start_time)
        
            for time in range(start_time, end_time + 1):
                data[time].append(element)
    
        return dict(data)

    test_data = generate_test_data(100, 220, 10)
    if 1 == -1:
        print(test_data)
        def test1():
            return find_common_intervals(test_data)

        def test2():
            return find_common_intervals2(test_data)

        time_method1 = timeit.timeit(test1, number=1000)
        time_method2 = timeit.timeit(test2, number=1000)

        print("Method 1 execution time:", time_method1)
        print("Method 2 execution time:", time_method2)

        # Find common intervals for each element pair
        common_intervals = find_common_intervals(data)

        print(f'common_intervals : {common_intervals}')

    if 1 == -11:
        def get_active_segments_at_time_interval(start, end, segments_active_dict):

            result = set(sum([vals for t,vals in segments_active_dict.items() if start <= t <= end],[]))
            return result

        def get_active_segments_at_time_interval2(start, end, segments_active_dict):
            store = []
            for t, vals in segments_active_dict.items():
                if start <= t <= end:
                    store += vals
            result = set(store)
            return result

        def get_active_segments_at_time_interval3(start, end, segments_active_dict):
            result = set()
            for t, vals in segments_active_dict.items():
                if start <= t <= end:
                    result.update(vals)
            return result
            #result = np.unique(sum([vals for t,vals in segments_active_dict.items() if start <= t <= end],[]))
            #result = set(val for t, vals in segments_active_dict.items() if start <= t <= end for val in vals)

            #result = set()
            #for t, vals in segments_active_dict.items():
            #    if start <= t <= end:
            #        result.update(vals)
            return result
        
        def test1():
            return get_active_segments_at_time_interval(50, 160, test_data)

        def test2():
            return get_active_segments_at_time_interval2(50, 160, test_data)
        def test3():
            return get_active_segments_at_time_interval3(50, 160, test_data)

        time_method1 = timeit.timeit(test1, number=10000)
        time_method2 = timeit.timeit(test2, number=10000)
        time_method3 = timeit.timeit(test3, number=10000)

        print("Method 1 execution time:", time_method1)
        print("Method 2 execution time:", time_method2)
        print("Method 3 execution time:", time_method3)



from collections import deque
import numpy as np
if 1== -1:
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

    aa = CircularBufferReverse(3, [3,4,5])
    aa.append(2)
    print(aa.get_data())

if 1 == -1:
    import itertools

    def split_into_bins(L, num_bins):
        n = len(L)
    
        if num_bins == 1:
            yield [sorted(L)]  # Sort the elements before yielding
            return
    
        for r in range(1, n - num_bins + 2):
            for combination in itertools.combinations(L, r):
                remaining = [x for x in L if x not in combination]
                for rest in split_into_bins(remaining, num_bins - 1):
                    yield [sorted(list(combination))] + rest  # Sort the combination sublist before yielding

    def combs_different_lengths(elements_list):
        return sum([list(itertools.combinations(elements_list, r)) for r in range(1,len(elements_list)+1)],[])
    # Example usage:
    L = [1, 2, 3, 4]
    num_bins = 2

    print(combs_different_lengths(L))
    a = list(split_into_bins(L, num_bins))
    for aa in a:
        print(aa)
if 1 == -1:
    lr_conn_splits_merges_mixed_dict = {1: [4, 5], 3: [4], 4:[7], 5:[6,7], 8:[9]}
    import itertools
    from collections import defaultdict
    # 10.10.2023 bonus for branch extension with conservative subID redistribution
    # get small connected clusters [from1, from2.] -> [to1,to2,..]. 
    t_from_IDs = lr_conn_splits_merges_mixed_dict.keys()
    t_compair_pairs = list(itertools.combinations(t_from_IDs, 2))
    t_from_connected = defaultdict(set)#{t:set([t]) for t in t_from_IDs}
    for t1,t2 in t_compair_pairs:
        t_common_elems = set(lr_conn_splits_merges_mixed_dict[t1]).intersection(set(lr_conn_splits_merges_mixed_dict[t2]))
        if len(t_common_elems) > 0:
            t_from_connected[t1].add(t2)

if 1 == -1:
    import networkx as nx

    G = nx.Graph()  # Your NetworkX graph
    # Add nodes and edges to your graph
    G.add_edges_from([(1,2),(2,3),(3,4),(5,6)])
    # Define edge colors (e.g., based on edge attribute 'color')
    for u, v in G.edges():
        #color = G[u][v].get('color', 'red')  # You can use a different edge attribute for color
        G[u][v]['color'] = 'red'

    # Export the graph to GEXF format
    nx.write_gexf(G, "your_graph_with_attributes.gexf")

if 1== -1:
    import itertools

    # Define the choices and their associated values
    choices_a = ['a1']
    choices_b = ['b1', 'b2']
    choices_c = ['c1', 'c2']

    L = [choices_a, choices_b, choices_c]

    D = {'a1': 1, 'b1': 3, 'b2': 10, 'c1': 4, 'c2': 2}
    L_vals = [[D[a] for a in b] for b in L]
    print(L)
    print(L_vals)

    # Function to build and check the choice tree
    thresh = 4

    refine_tree_condition = lambda elem_1, elem_2 : np.abs(D[elem_2] - D[elem_1]) <= thresh
    def refine_tree(L, thresh_func):
        output_edges = []
        choice_from = L[0]
        for i in list(range(1, len(L))):
            choice_to = L[i]
            conns = list(itertools.product(choice_from,choice_to))
            conns_viable = [(elem_1,elem_2) for elem_1,elem_2 in conns if thresh_func(elem_1, elem_2)]
            output_edges += conns_viable
            choice_from = [elem_2 for _,elem_2 in conns_viable]
        return output_edges, L[0], choice_from


    import networkx as nx
    edges, nodes_start, nodes_end = refine_tree(L, refine_tree_condition)
    def find_paths_from_to_multi(nodes_start, nodes_end, edges = None, G = None, construct_graph= False):

        if not construct_graph:
            G = nx.DiGraph()
            G.add_edges_from(edges)

        all_paths = []
        for start_node in nodes_start:
            for end_node in nodes_end:
                if nx.has_path(G, source=start_node, target=end_node):
                    paths = list(nx.all_simple_paths(G, source=start_node, target=end_node))
                    all_paths.extend(paths)

    paths = find_paths_from_to_multi(nodes_start, nodes_end,edges = None, G = None, construct_graph = False)
    a = 21
    def build_choice_tree(choices, current_branch, current_value):
        viable_branches = []

        if not choices:
            # End of the tree, add the branch and its value to viable_branches
            viable_branches.append((current_branch, current_value))
        else:
            current_choices = choices[0]
            remaining_choices = choices[1:]

            for choice in current_choices:
                new_branch = current_branch + [choice]
                new_value = current_value + D.get(choice, 0)
            
                if abs(D.get(choice, 0)) <= 4:  # Check the value constraint
                    viable_branches.extend(build_choice_tree(remaining_choices, new_branch, new_value))

        return viable_branches

    # Start building the choice tree
    viable_branches = build_choice_tree(L, [], 0)

    # Print the resulting viable branches
    for branch, value in viable_branches:
        print(f"Branch: {branch}, Value: {value}")

import itertools
if 1 == -1:
    def combs_different_lengths(elements_list):
        return sum([list(itertools.combinations(elements_list, r)) for r in range(1,len(elements_list)+1)],[])
    choices = [(1,2,3),(4,5)]
    choices2 = [combs_different_lengths(t) for t in choices]
    edges = list(itertools.product(*choices2))
    #edges = [((1,), (2,)), ((1, 2, 3), (4,)), ((1,), (3, 4, 5)), ((1, 2), (3, 4)),((1,1,1,1),(1,))]

    # Define a custom sorting key function
    def sorting_key(edge):
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

    # Sort the list of edges
    sorted_edges = sorted(edges, key=sorting_key, reverse=True)  

    # Print the sorted edges
    for edge in sorted_edges:
        print(edge)
if 1 == -1:
    import pickle
    pt = r'C:\Users\mhd01\source\repos\ajegorovs\bubble_process_py\post_tests\HFS 200 mT Series 4\sccm100-meanFix\00001-03000\archives'
    storeDir = os.path.join(pt, "ms-events-HFS 200 mT Series 4-sccm100-meanFix-00001-03000.pickle")
    with open(storeDir, 'rb') as handle:
        events_split_merge_mixed = pickle.load(handle)

    pt = r'C:\Users\mhd01\source\repos\ajegorovs\bubble_process_py\post_tests\HFS 200 mT Series 4\sccm100-meanFix\00001-03000\archives'
    storeDir = os.path.join(pt, "segments-HFS 200 mT Series 4-sccm100-meanFix-00001-03000.pickle")
    with open(storeDir, 'rb') as handle:
        trajectories_all_dict = pickle.load(handle)
if 1 == -1:
    import random
    def rand_int_inteval():
        interval_start = random.randint(0, 10)
        interval_length = random.randint(2, 4)
        return np.arange(interval_start, interval_start + interval_length)

    interval_gen = (rand_int_inteval() for _ in range(1000000000)) 
    first_suitable_interal = next((inteval for inteval in interval_gen if 5 in inteval))






if 1 == -1:
    import networkx as nx, numpy as np, sys

    path_modues = r'.\modules'      # os.path.join(mainOutputFolder,'modules')
    sys.path.append(path_modues)  
    from graphs_general import (for_graph_plots)

    # code allows to reduce time required to search connected segments
    # original step of finding segments that are within some time interval DT is kept. its cheap and fast
    # old approach was to isolate 'from' segment end node and 'to' segment's start node and all unused nodes inbetween
    # this way other segments, that might lie in between, were not considered as possible pathways.
    # if DT is large, there may be many possible connections between these segments.
    # old code tested them all. new approach says: 'you need stray nodes to find pathways'
    # 'instead of consructing subgraph and look for paths, see if there are time steps'
    # 'between segments in which there are no stray nodes.' paths cannot be constructed over these holes.



    #  THIS STUFF IS JUST REAL DATA IMPORT AND PREP TO LOOK LIKE IT IS A STAGE OF CODE PROCESS 
    r_folder = r'F:\UL Data\bubble_process_py_copy_02_11_2023\post_tests'
    dict_glob = {'Field OFF Series 7':{}} # HFS 125 mT Series 1 ; Field OFF Series 7; VFS 125 mT Series 5; HFS 200 mT Series 4
    pref = ["segments", "graphs", "ms-events","contorus"]
    for p_folder in dict_glob:
        folder0 = os.path.join(r_folder,p_folder)
        dirs0_all = os.listdir(folder0)
        dirs0 = [item for item in dirs0_all if os.path.isdir(os.path.join(folder0, item))]
        dirs0 = [dirs0[0]]
        for s_folder in dirs0:
            N = s_folder[4:7]#'sccm100-meanFix'
            root1 = os.path.join(folder0,s_folder)
            dirs1_all = os.listdir(root1)
            dirs1 = [item for item in dirs1_all if os.path.isdir(os.path.join(root1, item))]
            #for root1, dirs1, files in os.walk(os.path.join(root0,s_folder)):
            folder1 = dirs1[0]#'00001-03000'
            root2 = os.path.join(root1,folder1,'archives')
            for root3, dirs3, files3 in os.walk(os.path.join(root2)):
                dict_glob[p_folder][int(N)] = {prefix: os.path.join(root3, file) for file in files3 for prefix in pref if file.startswith(prefix)}


    for proj,proj_d in dict_glob.items():
        for sccm, params_d in proj_d.items():
            max_length = 0
            path_segments = params_d["segments"]
            with open(path_segments, 'rb') as handle:
                segments_d = pickle.load(handle)

            path_graphs = params_d["graphs"]
            with open(path_graphs, 'rb') as handle:
                graphs_d = pickle.load(handle)

            path_contorus = params_d["contorus"]
            with open(path_contorus, 'rb') as handle:
                contorus_d = pickle.load(handle)
        
            families = list(segments_d.keys())
            #families = [families[18]]
        
            for family in families:
                G1, G2  = graphs_d[family]

                G_time      = lambda node, graph = G1 : graph.nodes[node]['time']
                G_area      = lambda node, graph = G1 : graph.nodes[node]['area']
                G_owner     = lambda node, graph = G1 : graph.nodes[node]['owner']
                segments = segments_d[family]

                # check at which time steps there are stray (non-segment) nodes
                lr_time_stray_nodes     = defaultdict(bool)
                for node in G1.nodes():
                    if G_owner(node) == -1: lr_time_stray_nodes[G_time(node)] = True

                lr_time_stray_nodes = {t:True for t in sorted(lr_time_stray_nodes.keys())}


                #lr_time_active_segments = defaultdict(list)
                #for t_segment_index, t_segment_nodes in enumerate(segments):
                #    t_times = [G_time(node) for node in t_segment_nodes]
                #    for t_time in t_times:
                #        lr_time_active_segments[t_time].append(t_segment_index)
                ## sort keys in lr_time_active_segments
                #lr_time_active_segments = {t:lr_time_active_segments[t] for t in sorted(lr_time_active_segments.keys())}

                lr_DTPass_segm  = {}
                lr_maxDT        = 60
            
                G2_new = nx.DiGraph()
                fin_connectivity_graphs = defaultdict(list) 
                t_segments_relevant = np.array([t for t,t_seg in enumerate(segments) if len(t_seg) > 0])

                t_segment_time_start = np.array([G2.nodes[t]["t_start"] for t in t_segments_relevant])

                for t_ID_from in t_segments_relevant:        
                    G2_new.add_node(t_ID_from)
                    G2_new.nodes()[t_ID_from]["t_start"   ] = G_time(segments[t_ID_from][0] )
                    G2_new.nodes()[t_ID_from]["t_end"     ] = G_time(segments[t_ID_from][-1])
                
                    t_t_from    = G2.nodes[t_ID_from]["t_end"]
                    timeDiffs   = t_segment_time_start - t_t_from

                    t_DT_pass_index = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
                    t_IDs_DT_pass = t_segments_relevant[t_DT_pass_index]


                    node_from = segments[t_ID_from][-1]
                    time_from = G_time(node_from)
                    for t_ID_to in t_IDs_DT_pass:
                        # old approach of isolating subgraph
                        node_to         = segments[t_ID_to][0]
                        time_to         = G_time(node_to)

                        t_nodes_keep    = [node for node in G1.nodes() if time_from <= G_time(node) <= time_to and G_owner(node) is None] 
                        t_nodes_keep.extend([node_from,node_to])
                        G_sub = G1.subgraph(t_nodes_keep)
                        has_path = nx.has_path(G_sub, source = node_from, target = node_to)

                        # new approach of finding first case of missing stray nodes needed for path:

                        # walk times  between segments and find first time step without stray nodes
                        # no stray nodes means you cannot progress across this time step.
                        times = (t for t in range(time_from +1 , time_to ))
                        has_holes = next((True for t in times if t not in lr_time_stray_nodes), False)

                        # if method works properly (and data prepared prop), next if wont trigger
                        if has_path == True and has_holes == True:
                            # for_graph_plots(G1, segs = segments)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                            assert 1 == -1, "DANGER !!!"
                        # has_holes = True -> guaranteed that has_path = False
                        if has_holes and not has_path: print("yes")

                        # but has_holes = False does not mean that has_path = True
                        # so we can drop only cases which will truly fail due to impassable intervals
                        # and look for path for others
                        if has_path:
                            G2_new.add_edge(t_ID_from, t_ID_to, dist = time_to - time_from + 1) # include end points =  inter + 2
                a = 1

if 1 == -1:
    import networkx as nx

    G = nx.Graph()
    G.add_node(1,t_start = 10)
    b = lambda node , graph = G: print(graph.nodes()[node]["t_start"   ] )

    b(1) # output 10

    def gg(graph):
        graph.nodes()[1]["t_start"   ] = 20
        return graph

    gg(G)
    b(1) # ouput 20
if 1 == -1:
    import numpy as np, cv2

    # Assuming you have a numpy array of shape (N_images, height, width)
    N_images, height, width = 3, 100, 150
    binarizedMaskArr = np.zeros((N_images, height, width), dtype=np.uint8)

    # Define the border thickness
    border_thickness = 5


    binarizedMaskArr[:, :border_thickness, :] = 127
    binarizedMaskArr[:, -border_thickness:, :] = 127
    binarizedMaskArr[:, :, :border_thickness] = 127
    binarizedMaskArr[:, :, -border_thickness:] = 127

    # Display the results (showing only the first image for illustration)
    cv2.imshow('6',binarizedMaskArr[1])




    # snippet shows that if times is generated before loop, 
    # during manual debug, values are advanced without getting to code below.
    #a = []
    #times = (i for i in range(10 , 15 ))
    #for t in times:
    #for t in (i for i in range(10 , 15 )):

    #    a.append(t)
    
    #print(a)
if  1 == -1:
    def overlaps(a, b):
        """
        Return the amount of overlap, in bp
        between a and b.
        If >0, the number of bp of overlap
        If 0,  they are book-ended.
        If <0, the distance in bp between them
        """

        return min(a[1], b[1]) - max(a[0], b[0])

    i1 = [0,1]
    i2 = [8,10]
    print(overlaps(i2,i1))
    

    def ranges_overlap(range1, range2):
        return range1.stop > range2.start and range2.stop > range1.start

    # Example usage:
    range1 = range(1, 3)
    range2 = range(3, 7)

    if ranges_overlap(range1, range2):
        print("Ranges overlap.")
    else:
        print("Ranges do not overlap.")


import numpy as np, time, pickle, sys
from scipy.sparse import coo_matrix, triu
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

path_modues = r'.\modules'      
sys.path.append(path_modues) 

from misc import (addd, HH, prrr)

addd([1,2,3])
HH.add_nodes_from([4,5])
prrr()
a = 1



k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()
