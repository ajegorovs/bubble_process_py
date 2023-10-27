import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from misc import (CircularBuffer)
from collections import defaultdict

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


def interpolate_find_k_s(trajectory, time, params_k, params_s, report_on = False, report = [], num_sub_iterations = 3, part_points_drop = 0.2):
    # WHAT: find proper k and s parameters for interpolating holes in data
    # HOW: can do this by dropping random number of points and testing interpolation mechanisms using different params
    # NOTE: all k,s combs are tested num_sub_iterations times. each has associated list of fake holes
    # NOTE: this enables code to break early when smoothing parameter stops having an impact.
    traj_length = trajectory.shape[0]
    num_points_drop = int(np.ceil( (traj_length-2) * part_points_drop)) # keep end points
    drop_indicies = [np.random.choice(np.arange(1,traj_length - 1, 1), num_points_drop, replace=False) for _ in range(num_sub_iterations)]   # take random indices from mid
    temp_diffs_all  = defaultdict(float)
    temp_diffs_all[(1,5)] = 6666666                                     # in case everything fails (1,5) will remain
    for k in params_k:                                                  # iterate though polynom orders
        last_error_k = -1.0                                             # initialize prev result
                    
        if traj_length - num_points_drop <= k: continue
        for s in params_s:
            temp_diffs = [0]*num_sub_iterations
            for t in range(num_sub_iterations):
                    
                traj_subset = np.delete(trajectory  , drop_indicies[t], axis = 0)                                      # returns a copy
                time_subset = np.delete(time        , drop_indicies[t])
                times_find  = time[         drop_indicies[t]]
                traj_find   = trajectory[   drop_indicies[t]]
                t_sol = interpolate_trajectory(traj_subset, time_subset, which_times = times_find ,s = s, k = k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')
                diffs_missing = np.linalg.norm(t_sol - traj_find, axis = 1) # calc diffs between known sols and interp
                temp_diffs[t] = np.mean(diffs_missing)

            k_s_error = np.round(np.mean(temp_diffs),3)                     # round up so its easier to compare

            if k_s_error != last_error_k:                                   # high s = too relaxed, begin to produce similar results
                last_error_k = k_s_error
                temp_diffs_all[(k,s)] = k_s_error
            else:                                                           # if result is same as previously, skip s iterations
                if report_on: report.append((k,s))                          # for debugging
                break


    return temp_diffs_all

def extrapolate_find_k_s(trajectory, time, t_k_s_combs, k_all, k_s_buffer_len_max, k_s_start_at_index = 0, debug = 0, debug_show_num_best = 3):
    trajectory_length   = trajectory.shape[0]

    do_work_next_n          = trajectory_length - k_s_buffer_len_max 

    errors_tot_all  = {}
    errors_sol_all   = {}
    errors_sol_diff_norms_all   = {}
    # track previous error value for each k and track wheter iterator for set k should skip rest s.
    t_last_error_k  = {k:None for k in k_all} # if during first iteration error is rounded to 0.0, you will have trouble.
    t_stop_k        = {k:0 for k in k_all}    # None should force first iteration for k to be finished
    # MAYBE redo so same trajectory part is ran with different k and s parameters instead of change parts for one comb of k,s
    # EDIT, it does not let you break early if no change is detected. trade off? idk
    for t_comb in t_k_s_combs:
        k  = t_comb[0]; s = t_comb[1]
        if t_stop_k[k] == 1: continue 
        t_errors        = {}
        t_sols          = {}
        
        t_traj_buff     = CircularBuffer(k_s_buffer_len_max,  trajectory[k_s_start_at_index:k_s_start_at_index + k_s_buffer_len_max])
        t_time_buff     = CircularBuffer(k_s_buffer_len_max,  time[      k_s_start_at_index:k_s_start_at_index + k_s_buffer_len_max])
        
        t_indicies = np.arange(k_s_start_at_index, k_s_start_at_index + do_work_next_n , 1)
        for t_start_index in t_indicies:# 
            h_num_points_available = min(k_s_buffer_len_max, trajectory.shape[0] - t_start_index)           # but past start, there are only total-start available
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
    assert len(errors_tot_all)>0 ,'no iterations happened'
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

# not used ================================

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