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
    lr_big121s_perms_pre = {(2,3):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]},(10,11):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]}}
    for t_conn, t_dict in lr_big121s_perms_pre.items():
        for t_time,t_perms in t_dict.items():
            for t_perm in t_perms:
                lr_big121s_perms_areas[     t_conn][t_time][t_perm] = 52

    # Access the stored value
    stored_area = lr_big121s_perms_areas[t_conn][t_time][t_perm]
    print(stored_area)  # Output: 42


    #lr_big121s_perms_pre = {(2,3):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]},(10,11):{279: [3], 280: [6, 9], 281: [3, 6], 282: [6]}}
    #lr_big121s_perms = {t_conn:{t_time:t_perms for t_time,t_perms in t_dict.items()} for t_conn,t_dict in lr_big121s_perms_pre.items()}
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

    import random

    import random
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

import numpy as np

arr = np.array([2, 5, 1, 5, 7, 1, 3])

# Find the indices of the smallest elements
indices = np.where(arr == arr.min())[0]

print(indices)



k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()
