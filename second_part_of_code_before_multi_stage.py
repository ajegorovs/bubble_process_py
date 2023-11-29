t_has_holes_report = {}
    G2 = graph_check_paths(lr_maxDT, t_has_holes_report)

    print(f'\n{timeHMS()}:({doX_s}) Paths that have failed hole test: {t_has_holes_report}')

    for t_ID in fin_additional_segments_IDs:
        t_segment_k_s_diffs[t_ID] = None
        t_segment_k_s[t_ID] = None
        #lr_fake_redirect[t_ID] = t_ID

    lr_fake_redirect = {tID: tID for tID in range(len(segments2))} 
    # for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if 1 == 1:
        """
        ===============================================================================================
        ========================== FINAL PASSES. DETECT SPLIT/MERGE/MIXED EGENTS ======================
        ===============================================================================================
        NOTE 0: same as pre- merge/split/mixed extension (which was same as pre 121 recovery)
        NOTE 1: extension of branches requires more or equal number contours than branches.
        NOTE 1: thats why there is no point in recovering branches past this stage. recover only to earliest case.
        """
        print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Branches of splits/merges')

        fin_extend_info = defaultdict(dict)
            
        G2, fin_edges_main = get_event_types_from_segment_graph(G2) 
        
        state = 'merge'
        t_all_event_IDs             = [lr_fake_redirect[t_ID] for t_ID in t_event_start_end_times[state]]
        for to, predecessors in fin_edges_main[state].items():
            if to not in t_all_event_IDs: continue # event was already resolved without extrapolation. edit-comment it was not added/deleted from list
            time_to             = G2_t_start(to)#G2_dir.nodes[to]['t_start']
            times_from          = {n:G2_t_end(n) for n in predecessors} #G2_dir.nodes[n]['t_end']
            times_from_min      = min(times_from.values())
            times_from_max      = max(times_from.values())    # cant recover lower than this time
            times_subIDs        = t_event_start_end_times[state][to]['subIDs']
            times_subIDs_slice  = {t:IDs for t, IDs in times_subIDs.items() if times_from_min <= t <= time_to}
            # below not correct. for general case, should check number of branches. OR there is nothing to do except checking for == 1.
            times_solo        = [t for t, IDs in times_subIDs_slice.items() if len(IDs) == 1 and t > times_from_max] 
            if len(times_solo)>0:
                time_solo_min = min(times_solo) # min to get closer to branches
                if time_to - time_solo_min > 1:
                    times     = np.flip(np.arange(time_solo_min, time_to , 1))
                    fin_extend_info[(to,'back')] = {t: combs_different_lengths(times_subIDs_slice[t]) for t in times}


        state = 'split'
        t_all_event_IDs             = [lr_fake_redirect[t_ID] for t_ID in t_event_start_end_times[state]]
        for fr, successors in fin_edges_main[state].items():
            if fr not in t_all_event_IDs: continue # event was already resolved without extrapolation
            time_from               = G2_t_end(fr)#G2_dir.nodes[fr]['t_end']
            times_from              = {n:G2_t_start(n) for n in successors} #G2_dir.nodes[n]['t_start']
            time_to                 = max(times_from.values())
            time_to_min             = min(times_from.values())    # cant recover higher than this time
            fr_old                  = None
            for t in t_event_start_end_times[state]: # this field uses old IDs. have to recover it.
                if t_event_start_end_times[state][t]['t_start'] == time_from: fr_old = t
            
            if fr_old is not None:
                times_subIDs        = t_event_start_end_times[state][fr_old]['subIDs']
                times_subIDs_slice  = {t:IDs for t, IDs in times_subIDs.items() if time_from <= t <= time_to}
                # below not correct. for general case, should check number of branches. OR there is nothing to do except checking for == 1.
                times_solo          = [t for t, IDs in times_subIDs_slice.items() if len(IDs) == 1 and t < time_to_min] 
                if len(times_solo)>0:
                    time_solo_max = max(times_solo)
                    if time_solo_max - time_from > 1:
                        times = np.arange(time_from + 1, time_solo_max + 1 , 1)
                        fin_extend_info[(fr,'forward')] = {t: combs_different_lengths(times_subIDs_slice[t]) for t in times}
                        
        a = 1
    # for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Terminated segments')
    """
    ===============================================================================================
    =========== Final passes. examine edges of terminated segments. retrieve nodes  ===============
    ===============================================================================================
    WHAT: segments not connected to any other segments, on one or both sides, may still have 
    WHAT: stray nodes on ends. These cases were not explored previously.
    HOW: find segments with no successors or predecessors. Using Depth Search extract trailing nodes...
    NOTE: element order in t_conn now is fixed (from, to), numbering hierarchy does not represent order anymore.
    """
    start_points  = []
    end_points    = []

    for ID in G2.nodes():
        successors      = list(G2.successors(   ID))
        predecessors    = list(G2.predecessors( ID))

        if len(successors)      == 0: end_points.append(  ID)
        if len(predecessors)    == 0: start_points.append(ID)
    
    # 
    # ============= Final passes. Terminated segments. extract trailing nodes =================
    if 1 == 1:
        back_d = defaultdict(set)
        for ID in start_points:
            node_from = G2_n_from(ID)#segments2[ID][0]
            dfs_pred(G, node_from, time_lim = G2_t_start(ID) - 10, node_set = back_d[ID])
        back_d = {i:v for i,v in back_d.items() if len(v) > 1}

        forw_d = defaultdict(set)
        for ID in end_points:
            node_from = G2_n_to(ID)#segments2[ID][-1]
            dfs_succ(G, node_from, time_lim = G2_t_end(ID) + 10, node_set = forw_d[ID])
        forw_d = {i:v for i,v in forw_d.items() if len(v) > 1}

    # == Final passes. Terminated segments. Generate disperesed node dictionary for extrapolation ==
    for ID, nodes in back_d.items():
        perms           = disperse_nodes_to_times(nodes)
        perms_sorted    = sorted(perms)[:-1]
        perms_sorted.reverse()
        perms           = {i:perms[i] for i in perms_sorted}
        values          = [combs_different_lengths(IDs) for IDs in perms.values()]
        fin_extend_info[(ID,'back')] = {i:v for i,v in zip(perms, values)}

    for ID, nodes in forw_d.items():
        perms           = disperse_nodes_to_times(nodes)
        perms_sorted    = sorted(perms)[1:]
        perms           = {i:perms[i] for i in perms_sorted}
        values          = [combs_different_lengths(IDs) for IDs in perms.values()]
        fin_extend_info[(ID,'forward')] = {i:v for i,v in zip(perms, values)}

    # for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Extrapolate')
    # ============= Final passes. Terminated segments. Extrapolate segments =================
    t_report                = set()
    #t_out                   = defaultdict(dict)
    t_extrapolate_sol_comb  = defaultdict(dict)
    for (t_ID, state), t_combs in fin_extend_info.items():
        conn = (t_ID, state) # <<<<< custom. t_ID may be two states, and they have to be differentiated
        if state == 'forward':
            time_from   = G2_t_end(t_ID)#segments2[t_ID][-1][0]
            nodes       = [n for n in segments2[t_ID] if G_time(n) > time_from - h_interp_len_max2]
            node_from   = G2_n_to(t_ID)#segments2[t_ID][-1]
        else:
            time_to     = G2_t_start(t_ID)#segments2[t_ID][0][0]
            nodes       = [n for n in segments2[t_ID] if G_time(n) < time_to   + h_interp_len_max2]
            node_from   = G2_n_from(t_ID)#segments2[t_ID][0]

        trajectory = np.array([G_centroid(  n) for n in nodes])
        time       = np.array([G_time(      n) for n in nodes])
        area       = np.array([G_area(      n) for n in nodes])

        if state == 'forward':
            traj_buff     = CircularBuffer(h_interp_len_max2, trajectory)
            area_buff     = CircularBuffer(h_interp_len_max2, area)
            time_buff     = CircularBuffer(h_interp_len_max2, time)       
            time_next     = time_from   + 1                               
        if state == 'back':
            traj_buff     = CircularBufferReverse(h_interp_len_max2, trajectory) 
            area_buff     = CircularBufferReverse(h_interp_len_max2, area)
            time_buff     = CircularBufferReverse(h_interp_len_max2, time)      
            time_next     = time_to     - 1
        
        if t_segment_k_s[t_ID] is not None:
            t_k,t_s = t_segment_k_s[t_ID]
        else:
            t_k,t_s = (1,5)
        N = 5
        if t_segment_k_s_diffs[t_ID] is not None:
            last_deltas   = list(t_segment_k_s_diffs[t_ID].values())[-N:]  # not changing for splits
        else:
            last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5

        branch_IDs = [t_ID]
        branch_ID = t_ID

        norm_buffer   = CircularBuffer(N, last_deltas)
        
        all_traj_buffers = {branch_ID: traj_buff  }
        all_area_buffers = {branch_ID: area_buff  }
        all_norm_buffers = {branch_ID: norm_buffer}
        all_time_buffers = {branch_ID: time_buff  }
        
        #t_times_accumulate_resolved = []
        break_trigger = False
        for time, permutations in t_combs.items():

            if break_trigger: break # solo branch recover has failed.

            time_next = time

            traj_b    = all_traj_buffers[branch_ID].get_data()
            time_b    = all_time_buffers[branch_ID].get_data()
            area_b    = all_area_buffers[branch_ID].get_data()

            centroids_extrap  = interpolate_trajectory( traj_b, time_b, [time_next], t_s, t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]
            areas_extrap      = interpolateMiddle1D_2(  time_b, area_b, [time_next], rescale = True, s = 15, debug = 0, aspect = 'auto', title = 1)
 
            centroids = []
            areas     = []

            permutation_params = {}
            for permutation in permutations:
                hull = cv2.convexHull(np.vstack([g0_contours[time][tID] for tID in permutation]))
                centroid, area = centroid_area(hull)
                centroids.append(centroid)
                areas.append(area)
                permutation_params[permutation] = centroid_area(hull)

            diff_choices      = {}      # holds index key on which entry in perms_distribution2 has specifict differences with target values.
            diff_choices_area = {}
            perms_distribution2 = [[list(t)] for t in permutations]
            for t, redist_case in enumerate(perms_distribution2):
                centroids = np.array([permutation_params[tuple(t)][0] for t in redist_case])
                areas     = np.array([permutation_params[tuple(t)][1] for t in redist_case])
                diff_choices[t]       = np.linalg.norm(centroids_extrap - centroids, axis=1)
                diff_choices_area[t]  = np.abs(areas_extrap - areas)/areas_extrap
            a = 1
            # -------------------------------- refine simultaneous solution ---------------------------------------
            # evaluating sum of diffs is a bad approach, becaues they may behave differently

            norms_all     = {t:np.array(  all_norm_buffers[t].get_data()) for t in branch_IDs}
            max_diff_all  = {t:max(np.mean(n),5) + 5*np.std(n)              for t,n in norms_all.items()}
            relArea_all = {}
            for ID in branch_IDs:
                t_area_hist = all_area_buffers[ID].get_data()
                relArea_all[ID] = np.abs(np.diff(t_area_hist))/t_area_hist[:-1]
            t_max_relArea_all  = {t:max(np.mean(n) + 5*np.std(n), 0.35) for t, n in relArea_all.items()}

                
            choices_pass_all      = [] # holds indidicies of perms_distribution2
            choices_partial       = []
            choices_partial_sols  = {}
            choices_pass_all_both = []
            # filter solution where all branches are good. or only part is good
            for i in diff_choices:
                pass_test_all     = diff_choices[       i] < np.array(list(max_diff_all.values()        )) # compare if less then crit.
                pass_test_all2    = diff_choices_area[  i] < np.array(list(t_max_relArea_all.values()   )) # compare if less then crit.
                pass_both_sub = np.array(pass_test_all) & np.array(pass_test_all2) 
                if   all(pass_both_sub): # all branches pass   
                    choices_pass_all_both.append(i)          
                elif any(pass_both_sub):
                    choices_partial.append(i)
                    choices_partial_sols[i] = pass_both_sub
                

            if len(choices_pass_all_both) > 0:     # isolate only good choices    
                if len(choices_pass_all_both) == 1:
                    diff_norms_sum          = {choices_pass_all_both[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1                   = {i: diff_choices[     i] for i in choices_pass_all_both}
                    temp2                   = {i: diff_choices_area[i] for i in choices_pass_all_both}
                    test, diff_norms_sum    = two_crit_many_branches(temp1, temp2, len(branch_IDs))
                #test, t_diff_norms_sum = two_crit_many_branches(t_diff_choices, t_diff_choices_area, len(t_branch_IDs))
                #t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items() if t in t_choices_pass_all}

            # if only part is good, dont drop whole solution. form new crit based only on non-failed values.
            elif len(choices_partial) > 0:  
                if len(choices_partial) == 1:
                    diff_norms_sum = {choices_partial[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1 = {i: diff_choices[     i] for i in choices_partial}
                    temp2 = {i: diff_choices_area[i] for i in choices_partial}
                    test, diff_norms_sum = two_crit_many_branches(temp1, temp2, len(branch_IDs))
                    #t_temp = {}
                    #for t in t_choices_partial:      
                    #    t_where = np.where(t_choices_partial_sols[t])[0]
                    #    t_temp[t] = np.sum(t_diff_choices[t][t_where])
                    #t_diff_norms_sum = t_temp
                    #assert len(t_diff_norms_sum) > 0, 'when encountered insert a loop continue, add branches to failed' 
            # all have failed. process will continue, but it will fail checks and terminate branches.
            else:                              
                #t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items()}
                diff_norms_sum = {0:[i + 1 for i in max_diff_all.values()]} # on fail spoof case which will fail.

            where_min     = min(diff_norms_sum, key = diff_norms_sum.get)
            sol_d_norms   = diff_choices[         where_min]
            sol_subIDs    = perms_distribution2[  where_min]

            for branch_ID, subIDs, sol_d_norm in zip(branch_IDs,sol_subIDs, sol_d_norms):

                if sol_d_norm < max_diff_all[branch_ID]:

                    all_norm_buffers[branch_ID].append(sol_d_norm                           )
                    all_traj_buffers[branch_ID].append(permutation_params[tuple(subIDs)][0] )
                    all_area_buffers[branch_ID].append(permutation_params[tuple(subIDs)][1] )
                    all_time_buffers[branch_ID].append(time_next                            )
                    
                    t_extrapolate_sol_comb[conn][time_next] = tuple(subIDs)
                    t_report.add(conn)
                else:
                    break_trigger = True
                    break      # this is solo branch extrapolation, should break here. og method uses continue to recover other branches.
    
    
        a = 1
        for conn in t_report:
            (fr, state)  = conn
            dic = t_extrapolate_sol_comb[conn]

            if len(dic) == 0: continue
           
            t_min, t_max    = min(dic), max(dic)

            #if state == 'forward':
            #    node_from     = (t_min,) + tuple(dic[t_min])    #tuple([t_min] + list(dic[t_min]))
            #    node_to       = (t_max,) + tuple(dic[t_max])    #tuple([t_max] + list(dic[t_max]))
            #else:
            #    node_to       = (t_min,) + tuple(dic[t_min])    #tuple([t_min] + list(dic[t_min]))
            #    node_from     = (t_max,) + tuple(dic[t_max])    #tuple([t_max] + list(dic[t_max]))

            a, b = (t_min,) + tuple(dic[t_min]), (t_max,) + tuple(dic[t_max])

            if state == 'forward':
                node_from, node_to = a, b
            else:
                node_from, node_to = b, a

            print(f' {fr} - {state} extrapolation: {node_from}->{node_to}')
        a = 1


    # for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    ===============================================================================================
    =========== Final passes. examine edges of terminated segments. solve conflicts  ===============
    ===============================================================================================
    """
    # check if there are contested nodes in all extrapolated paths
    duplicates = conflicts_stage_1(t_extrapolate_sol_comb)
    
    if len(duplicates) > 0:   # tested on 26.10.23; had #variants_possible = 1
        # retrieve viable ways of redistribute contested nodes
        variants_all        = conflicts_stage_2(duplicates)
        variants_possible   = conflicts_stage_3(variants_all,duplicates, t_extrapolate_sol_comb)
        #if there is only one solution by default take it as answer
        if len(variants_possible) == 1:  
            choice_evol = variants_possible[0]
        elif len(variants_possible) == 0:
            problematic_conns = set()
            [problematic_conns.update(conns) for conns in duplicates.values()]
            for conn in problematic_conns:
                t_extrapolate_sol_comb.pop(conn, None)
            choice_evol = []
        else:
            # method is not yet constructed, it should be based on criterium minimization for all variants
            # current, trivial solution, is to pick solution at random. at least there is no overlap.
            assert -1 == 0, 'multiple variants of node redistribution'
            choice_evol = variants_possible[0]
         
        # redistribute nodes for best solution.
        for node, conn in choice_evol:
            tID             = conn[1]
            time, *subIDs   = node
            t_extrapolate_sol_comb[conn][time] = tuple(subIDs)
            conns_other = [c for c in duplicates[node] if c != conn] # competing branches
            for conn_other in conns_other:
                subIDs_other = t_extrapolate_sol_comb[conn_other][time]                           # old solution for time
                t_extrapolate_sol_comb[conn_other][time] = tuple(set(subIDs_other) - set(subIDs)) # remove competeing subIDs
            #t_delete_conns      = [t_c for t_c in duplicates[node] if t_c != conn]
            #for t_delete_conn in t_delete_conns:
            #    t_temp = t_extrapolate_sol_comb[t_delete_conn][time]
            #    t_temp = [t for t in t_temp if t not in subIDs]
            #    t_extrapolate_sol_comb[t_delete_conn][time] = t_temp
            #t_conns_relevant = [t_c for t_c in t_extrapolate_sol_comb if t_c[1] == tID]
            #lr_conn_merges_good.update(t_conns_relevant) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< not correct anymore

    # for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    ===============================================================================================
    =========== Final passes. examine edges of terminated segments. save extensions  ===============
    ===============================================================================================
    """
    #for (t_ID,state), t_conns in lr_conn_merges_good.items():
    for conn, combs in t_extrapolate_sol_comb.items():
        (ID, state) = conn

        if len(combs) == 0: continue                              # no extension, skip.

        if state == 'forward':
            save_connections_merges(segments2, t_extrapolate_sol_comb[conn], ID,  None, lr_fake_redirect, g0_contours)
        elif state == 'back':
            save_connections_splits(segments2, t_extrapolate_sol_comb[conn], None,  ID, lr_fake_redirect, g0_contours)


    # for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    tt = []
    for n in G.nodes():
        if "time" not in G.nodes[n]:
            set_custom_node_parameters(g0_contours, [n], None, calc_hull = 1)
            tt.append(n)
    print(f'were missing: {tt}')


    #if doX == 1: for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    print(f'\n{timeHMS()}:({doX_s}) Final. Recalculate connectivity')
    # ======= EXPORT. RECALCULATE CONNECTIONS BETWEEN SEGMENTS ========================
    
    t_has_holes_report = {}
    G2 = graph_check_paths(lr_maxDT, t_has_holes_report)                        
    # ========== EXPORT. DETERMINE TYPE OF EVENTS BETWEEN SEGMENTS ==========
    G2, export_ms_edges_main = get_event_types_from_segment_graph(G2)


    if doX == 8:

        t_conn_121_zp = export_ms_edges_main['solo_zp']

        lr_zp_redirect = {i: i for i,v in enumerate(segments2) if len(v) > 0}

        zp_process(t_conn_121_zp, segments2, g0_contours, lr_zp_redirect, time_buffer = 5)

        if len(t_conn_121_zp) > 0:
            # During zp procedure graphs where changed, can recalculate event detection, buts its faster to replace from-to relations.
            # Update inheritance to link slaves to global masters, which is not accomplished during zp procedure
            zp_inheritance_updated      = find_final_master_all(lr_zp_redirect)
            # 121 stage is only interested in remaining solo connections. update them
            export_ms_edges_main['solo']    = [(zp_inheritance_updated(fr),to) for fr, to in export_ms_edges_main['solo']]

        t_conn_121 = [tuple(sorted(edge ,key = G2_t_start)) for edge in export_ms_edges_main['solo']]
        lr_121_stray_disperesed = f121_disperse_stray_nodes(t_conn_121)
        t_big121s_edited = extract_clusters_from_edges(t_conn_121, sort_f_in = G2_t_start)
        lr_big121s_edges_relevant, lr_121_hole_interpolation = f121_interpolate_holes(t_big121s_edited, segments2)
        lr_big121s_perms = f121_calc_permutations(lr_121_stray_disperesed)
        lr_big121s_conn_121 = t_conn_121            #lr_big121s_edges_relevant
        lr_big121s_perms_areas, lr_big121s_perms_centroids, lr_big121s_perms_mom_z = f121_precompute_params(lr_big121s_perms, g0_contours)
        (lr_big121s_perms_cases,
        lr_big121s_perms_times,
        lr_drop_huge_perms) = f121_get_evolutions(lr_big121s_perms, lr_big121s_perms_areas, sort_len_diff_f, max_paths = 5000)

        lr_big121s_conn_121 = [e for e in lr_big121s_conn_121 if e not in lr_drop_huge_perms]
        [lr_big121s_perms_cases.pop(e, None) for e in lr_drop_huge_perms]
        [lr_big121s_perms_times.pop(e, None) for e in lr_drop_huge_perms]

        temp_centroids = {tc:d['centroids'] for tc, d in lr_121_hole_interpolation.items()}
        args = [lr_big121s_conn_121, lr_big121s_perms_cases,temp_centroids,lr_big121s_perms_times,
                        lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

        sols = lr_evel_perm_interp_data(*args)

        t_weights   = [1,1.5,0,1] # [sols_c, t_sols_c_i, t_sols_a, t_sols_m]
        lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(lr_big121s_conn_121,t_weights, sols, lr_big121s_perms_cases )

        #segments2 = copy.deepcopy(segments2)
        lr_121chain_redirect = {i: i for i,v in enumerate(segments2) if len(v) > 0}
        for edge in lr_big121s_conn_121:
            fr, to = edge
            fr_new, to_new = old_conn_2_new(edge,lr_121chain_redirect)
            print(f'edge :({fr},{to}) or = {G2_n_to(fr_new)}->{G2_n_from(to_new)}')  
            save_connections_two_ways(segments2, lr_121_stray_disperesed[edge], fr, to, lr_121chain_redirect, g0_contours)
