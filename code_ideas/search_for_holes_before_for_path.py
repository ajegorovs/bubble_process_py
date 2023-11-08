import networkx as nx, numpy as np, sys, pickle, os
from collections import defaultdict

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


# snippet shows that if times is generated before loop, 
# during manual debug, values are advanced without getting to code below.
#a = []
#times = (i for i in range(10 , 15 ))
#for t in times:
#for t in (i for i in range(10 , 15 )):

#    a.append(t)
    
#print(a)
