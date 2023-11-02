if 1 == -1:
        # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ===============================================================================================
        # ============== Final passes. Find k and s params for interpolating holes ======================
        # ===============================================================================================
        t_segments_IDs_relevant = [t for t,t_nodes in enumerate(t_segments_new) if len(t_nodes) > 0]

        fin_interpolate_k_s_params = defaultdict(tuple)
    
        t_report = defaultdict(list)
        for t_ID in t_segments_IDs_relevant:
            if len(t_segments_new[t_ID])< 3:
                fin_interpolate_k_s_params[t_ID] = (1,5)
                continue
            trajectory  = np.array([G.nodes[t]["centroid"   ] for t in t_segments_new[t_ID]])
            time        = np.array([G.nodes[t]["time"       ] for t in t_segments_new[t_ID]])
            params_k  = [1,2]
            params_s  = (0,1,5,10,25,50,100,1000,10000)
            t_k_s = interpolate_find_k_s(trajectory, time, params_k, params_s,  report_on = True, report = t_report[t_ID], num_sub_iterations = 20)
            fin_interpolate_k_s_params[t_ID] = min(t_k_s, key=t_k_s.get)    # (1,5) is a failsafe. will have a value of 6666666

        # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
            t_edges_inter_segment = []
            t_times_sides = [t_times[0] - 1] + t_times + [t_times[-1] +1 ]
            t_times_staggered = [(x, y) for x, y in zip(t_times[:-1], t_times[1:])]     # list of two consequitive time steps (t1,t2)
            t_end_edge   = (t_times[-1] , t_times[-1] + 1)
            t_1,t_2 = (t_times[0] - 1, t_times[0])
            if t_1 in lr_time_active_segments:
                if t_seg_1 in lr_time_active_segments[t_1] and t_seg_2 in lr_time_active_segments[t_2]:
                    t_1_subIDs_1 = t_nodes_resolved_per_segment[t_seg_1][t_1]                
                    t_2_subIDs_1 = t_nodes_resolved_per_segment[t_seg_2][t_2]               
                    t_prod_1 = list(itertools.product(t_1_subIDs_1, t_2_subIDs_1))
                    [t_edges_inter_segment.append(((t_1, t_from),(t_2,t_to))) for t_from, t_to in t_prod_1]

                if t_seg_2 in lr_time_active_segments[t_1] and t_seg_1 in lr_time_active_segments[t_2]:
                    t_1_subIDs_2 = t_nodes_resolved_per_segment[t_seg_2][t_1]
                    t_2_subIDs_2 = t_nodes_resolved_per_segment[t_seg_1][t_2]
                    t_prod_2 = list(itertools.product(t_1_subIDs_2, t_2_subIDs_2)) 
                    [t_edges_inter_segment.append(((t_1, t_from),(t_2,t_to))) for t_from, t_to in t_prod_2]

            for t_1,t_2 in t_times_staggered:
                t_1_subIDs_1 = t_nodes_resolved_per_segment[t_seg_1][t_1]               # extract subIDs for segment 1 at t1 and 
                t_2_subIDs_1 = t_nodes_resolved_per_segment[t_seg_2][t_2]               # subIDs for segment 2 for t2
        
                t_1_subIDs_2 = t_nodes_resolved_per_segment[t_seg_2][t_1]               # extract subIDs for segment 2 at t1 and 
                t_2_subIDs_2 = t_nodes_resolved_per_segment[t_seg_1][t_2]               # subIDs for segment 1 for t2
       
                t_prod_1 = list(itertools.product(t_1_subIDs_1, t_2_subIDs_1))          # for connections between two cases. 
                t_prod_2 = list(itertools.product(t_1_subIDs_2, t_2_subIDs_2))          # order t1<t2 still holds
        
                [t_edges_inter_segment.append(((t_1, t_from),(t_2,t_to))) for t_from, t_to in t_prod_1 + t_prod_2]                                # combine connections together and add times
             

            t_1, t_2 = (t_times[-1] , t_times[-1] + 1)
            if t_2 in lr_time_active_segments:
                if t_seg_1 in lr_time_active_segments[t_1] and t_seg_2 in lr_time_active_segments[t_2]:
                    t_1_subIDs_1 = t_nodes_resolved_per_segment[t_seg_1][t_1]                
                    t_2_subIDs_1 = t_nodes_resolved_per_segment[t_seg_2][t_2]               
                    t_prod_1 = list(itertools.product(t_1_subIDs_1, t_2_subIDs_1))
                    [t_edges_inter_segment.append(((t_1, t_from),(t_2,t_to))) for t_from, t_to in t_prod_1]

                if t_seg_2 in lr_time_active_segments[t_1] and t_seg_1 in lr_time_active_segments[t_2]:
                    t_1_subIDs_2 = t_nodes_resolved_per_segment[t_seg_2][t_1]
                    t_2_subIDs_2 = t_nodes_resolved_per_segment[t_seg_1][t_2]
                    t_prod_2 = list(itertools.product(t_1_subIDs_2, t_2_subIDs_2)) 
                    [t_edges_inter_segment.append(((t_1, t_from),(t_2,t_to))) for t_from, t_to in t_prod_2]

            G_OG.remove_edges_from(t_edges_inter_segment)                           # remove edges


        t_segment_stray_neighbors = defaultdict(set)
        for t,t_segment in enumerate(t_segments_new):
            for t_time,*t_subIDs in t_segment[1:-1]:                                    # walk composite nodes of resolved segments
                for t_subID in t_subIDs:                                                
                    t_node = (t_time,t_subID)                                           # decompose nodes to fit OG graph
                    t_neighbors = G_OG.neighbors(t_node)                                # check remaining connections
                    t_segment_stray_neighbors[t].update(t_neighbors)                    # add stray nodes to storage
        # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<# for_graph_plots(G_OG) 
        # ===============================================================================================
        # ===============================================================================================
        # ================================ Final passes. test and redistribute stray nodes ===============================
        # ===============================================================================================

        fin_stray_interpolation = defaultdict(dict)
        fin_stray_nodes_contested = defaultdict(dict)
        fin_stray_perms_predefined = AutoCreateDict()#{'centroids':defaultdict(dict), 'areas':defaultdict(dict), 'momz':defaultdict(dict)}
        t_stray_node_replace_dict = defaultdict(tuple)
        for t_ID, t_nodes in t_segment_stray_neighbors.items():
            if len(t_nodes)>0:
                t_nodes_contested = disperse_nodes_to_times(t_nodes)
                t_temp_nodes_all = [t_node for t_node in t_segments_new[t_ID] if t_node[0] not in t_nodes_contested]
        

                t_temp_times        = [G.nodes[t_node]["time"       ] for t_node in t_temp_nodes_all]
                t_temp_areas        = [G.nodes[t_node]["area"       ] for t_node in t_temp_nodes_all]
                t_temp_centroids    = [G.nodes[t_node]["centroid"   ] for t_node in t_temp_nodes_all]
                    
                t_times_missing_all = list(t_nodes_contested.keys())
        
                a = 1
                k, s = fin_interpolate_k_s_params[t_ID]
                # interpolate composite (long) parameters 
                t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, k = k, s = s, debug = 0, aspect = 'equal', title = t_ID)
                t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_ID)
                # form dict time:centroid for convinience
                t_interpolation_centroids_1 = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_centroids_0)}
                t_interpolation_areas_1     = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_areas_0)}

                fin_stray_interpolation[t_ID]['centroids'] = t_interpolation_centroids_1
                fin_stray_interpolation[t_ID]['areas'    ] = t_interpolation_areas_1

                # generate permutations from contested times
                t_nodes_old_at_contested_times = {t_node[0]:list(t_node[1:]) for t_node in t_segments_new[t_ID] if t_node[0] in t_times_missing_all}

                fin_stray_nodes_contested[t_ID] = {t:[] for t in t_times_missing_all} # join stray and contested for permutations
                t_min = defaultdict(tuple)
                for t_time in t_times_missing_all:
                    t_subIDs = sorted(t_nodes_contested[t_time] + t_nodes_old_at_contested_times[t_time])
                    fin_stray_nodes_contested[t_ID][t_time] = t_subIDs
                    t_perms = combs_different_lengths(t_subIDs)
                    t_sols = np.ones((2,len(t_perms)), float)
                    t_sols_min = [[]]
                    t_weight_bonuses = {t:0 for t in t_perms}
                    t_c_delta_max = 0
                    t_a_delta_max = 0

                    t_c_sols = defaultdict(dict)
                    t_a_sols = defaultdict(dict)
                    for t, t_perm in enumerate(t_perms):
                        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
                        t_centroid, t_area, t_mom_z     = centroid_area_cmomzz(t_hull)
                        fin_stray_perms_predefined['centroids'  ][t_ID][t_time][t_perm] = t_centroid
                        fin_stray_perms_predefined['areas'      ][t_ID][t_time][t_perm] = t_area
                        #fin_stray_perms_predefined['momz'       ][t_ID][t_time][t_perm] = t_mom_z
                
                        t_c_delta = np.linalg.norm(t_centroid - t_interpolation_centroids_1[t_time])
                        t_a_area = np.abs(t_area - t_interpolation_areas_1[t_time])/t_interpolation_areas_1[t_time]

                        t_c_sols[t_perm] = t_c_delta
                        t_a_sols[t_perm] = t_a_area

                    t_c_delta_max = max(t_c_sols.values())
                    t_a_delta_max = max(t_a_sols.values())
                    # we know that min = 0 is good ref, max is bad. value/max-min is how bad it is. 1- (value/max - min) should be how good it is.
                    for t_perm in t_perms:
                        t_weight_bonuses[t_perm] += (1 - t_c_sols[t_perm]/t_c_delta_max)   # not rescaled to 0-1. 
                        t_weight_bonuses[t_perm] += (1 - t_a_sols[t_perm]/t_a_delta_max)   # also cannot properly compare two measurments w/o normalization. but still, makes no diff
            
                    t_min[t_time] = max(t_weight_bonuses, key = t_weight_bonuses.get)

                    # update resolved nodes, if they changed
                for t_time, t_subIDs in t_min.items():
                    t_subIDs_old = sorted(t_nodes_old_at_contested_times[t_time])
                    t_subIDs_new = sorted(t_subIDs) 
                    if t_subIDs_new != t_subIDs_old:
                        t_node_new = tuple([t_time] + list(t_subIDs_new))
                        t_node_old = tuple([t_time] + list(t_subIDs_old))
                        t_stray_node_replace_dict[t_node_old] = t_node_new

                        t_pos = t_segments_new[t_ID].index(t_node_old)
                        t_segments_new[t_ID][t_pos] = t_node_new

        for t_node_old, t_node_new in t_stray_node_replace_dict.items():
            for t_pred in G.predecessors(t_node_old):
                if t_pred in t_stray_node_replace_dict:G.add_edge(t_stray_node_replace_dict[t_pred], t_node_new)
                else : G.add_edge(t_pred, t_node_new)
            for t_succ in G.successors(t_node_old):
                if t_succ in t_stray_node_replace_dict: G.add_edge(t_node_new, t_stray_node_replace_dict[t_succ])
                else:G.add_edge(t_node_new, t_succ)

        t_nodes_remove = []
        for t_time,*t_subIDs in t_stray_node_replace_dict.values():
            for t_subID in t_subIDs:
                t_nodes_remove.append((t_time,t_subID))

        G.remove_nodes_from(t_nodes_remove) 
        set_custom_node_parameters(G, g0_contours, t_stray_node_replace_dict.values(), None, calc_hull = 1)

        print(f'modified following stray nodes (old-new): {t_stray_node_replace_dict}')
    a = 1
