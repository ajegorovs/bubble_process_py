
import numpy as np, itertools, networkx as nx, sys, copy,  cv2, os, glob, re, pickle, time, datetime, os
from multiprocessing import shared_memory, Manager, Pool, Event
from skimage.morphology import flood_fill
#from matplotlib import pyplot as plt
#from tqdm import tqdm
#from collections import defaultdict
# import torch
# from torch.nn.functional import conv2d
# import torch.nn as nn
# import torchvision.transforms as T
# import torch.nn.functional as F

# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

calc_GPU = 1
path_modues = r'.\modules'      # os.path.join(mainOutputFolder,'modules')
sys.path.append(path_modues)
if __name__ == '__main__':
    
    if calc_GPU:
        path_modules_GPU = r'..\python-code-bits-for-image-processing-and-linear-algebra\multiprocessing-GPU'
        sys.path.append(path_modules_GPU)
        from gpu_torch_cuda_functions import (batch_axis0_mean, torch_blur, kernel_circular, morph_erode_dilate)
        from module_GPU import (dataset_create, SharedMemoryDataset)
        from torch.utils.data import DataLoader
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cuda_ref_type = torch.float16

        # def GPU_binary_pipe(i, buffer, batch, image_blurred, threshold, cuda_ref_type, cpu_ref_type, debug = 0):
        #     kernel_5 = kernel_circular(5, dtype = cuda_ref_type, normalize = False, device = device)
        #     kernel_9 = kernel_circular(9, dtype = cuda_ref_type, normalize = False, device = device)
        #     batch = batch.to(device).to(cuda_ref_type)
        #     batch = batch.unsqueeze_(1)    
        #     batch.add_(-image_blurred)

        #     batch = batch > threshold           # may be not cool since its changes type size
        #     batch = batch.to(cuda_ref_type)
        #     if debug: test0 = batch[0, 0, :, :]
        #     # remove small elements by equal size erode-dilate a.k.a morph closing/opening
        #     batch = morph_erode_dilate(batch, kernel_5, mode = 0)   # erode 
        #     if debug: test1 = batch[0, 0, :, :]

        #     batch = morph_erode_dilate(batch, kernel_5, mode = 1)   # dilate

        #     if debug:
        #         test2 = batch[0, 0, :, :]
        #         for a,b in zip(['cuda: 0 binar', 'cuda: 1 erode', 'cuda: 2 dilate'], [test0, test1, test2]):
        #             cv2.imshow(a, ((b.detach().cpu().numpy())*255.0).astype(np.uint8))

        #     # try to join closuters by larger cluster and do smaller erode. not sure if it always works.
        #     batch = morph_erode_dilate(batch, kernel_9, mode = 1)
        #     batch = morph_erode_dilate(batch, kernel_5, mode = 0)
        #     batch *= 255.0
        #     # fill cpu buffer with processed batch. they are same type now.
        #     buffer[i:i + len(batch),...] = batch.squeeze(1).to('cpu').numpy().astype(cpu_ref_type)
        #     # cv2.imshow('binar', buffer[0]); cv2.imshow('binar_old', buffer[1])
        #     return 

        def GPU_binary_pipe(dataLoader, image_blurred, threshold, arr_shape, cuda_ref_type, cpu_ref_type, debug = 0):
            sh_mem    = shared_memory.SharedMemory(name='buf_cropped')
            buffer     = np.ndarray(arr_shape, dtype = cpu_ref_type, buffer=sh_mem.buf)
            for i,batch in enumerate(dataLoader):
                kernel_5 = kernel_circular(5, dtype = cuda_ref_type, normalize = False, device = device)
                kernel_9 = kernel_circular(9, dtype = cuda_ref_type, normalize = False, device = device)
                batch = batch.to(device).to(cuda_ref_type)
                batch = batch.unsqueeze_(1)    
                batch.add_(-image_blurred)

                batch = batch > threshold           # may be not cool since its changes type size
                batch = batch.to(cuda_ref_type)
                if debug: test0 = batch[0, 0, :, :]
                # remove small elements by equal size erode-dilate a.k.a morph closing/opening
                batch = morph_erode_dilate(batch, kernel_5, mode = 0)   # erode 
                if debug: test1 = batch[0, 0, :, :]

                batch = morph_erode_dilate(batch, kernel_5, mode = 1)   # dilate

                if debug:
                    test2 = batch[0, 0, :, :]
                    for a,b in zip(['cuda: 0 binar', 'cuda: 1 erode', 'cuda: 2 dilate'], [test0, test1, test2]):
                        cv2.imshow(a, ((b.detach().cpu().numpy())*255.0).astype(np.uint8))

                # try to join closuters by larger cluster and do smaller erode. not sure if it always works.
                batch = morph_erode_dilate(batch, kernel_9, mode = 1)
                batch = morph_erode_dilate(batch, kernel_5, mode = 0)
                batch *= 255.0
                # fill cpu buffer with processed batch. they are same type now.
                buffer[i:i + len(batch),...] = batch.squeeze(1).to('cpu').numpy().astype(cpu_ref_type)
            # for g in range(0,i,5):
            #     cv2.imshow(f'{g}', buffer[g])
            # a = 1
            return 
# need functions for workers. eval out of main.            
if not calc_GPU:
    def mean_slice(from_to, crop_arr_shape, crop_arr_type, queue, report = False):
        global first_iter#, crop_arr_shape, crop_arr_type

        buf_mean    = shared_memory.SharedMemory(name='buf_cropped')
        mean_np     = np.ndarray(crop_arr_shape, dtype = crop_arr_type, buffer=buf_mean.buf)
        if report:
            if not first_iter:  print2(f'time between iterations: {track_time()}')
            else:               first_iter = False

        s_from, s_to    = from_to
        res             = np.mean(mean_np[s_from: s_to,...], axis = 0)
        queue.put((s_to - s_from , res))
        if report: print2(f'{from_to} iteration time: {track_time()}')
        return 


    def mean_slice_finish(num_elems, img_dims, queue):
        weights = np.zeros(num_elems)
        buffer  = np.zeros((num_elems, *img_dims))
        #print2(f'reserved mem {track_time()}')
        i = 0
        while not queue.empty():
            weight, image = queue.get()
            weights[i] = weight
            buffer[i,:,:] = image
            #print2(f'retrieved queue {i} w: {weight} {track_time()}')
            i += 1
        #print2(f'w: {weights}')
        return np.average(buffer, axis=0, weights=weights)

    def CPU_binary_pipe(fr_to, bin_thresh, crop_arr_shape, crop_arr_type, debug = False):

        buf_cropped = shared_memory.SharedMemory(name='buf_cropped')
        cropped_np  = np.ndarray(crop_arr_shape, dtype = crop_arr_type, buffer=buf_cropped.buf)

        kernel_5    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        kernel_9    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

        (fr,to) = fr_to
        for i in range(fr, to, 1):
            if debug:
                out = [cropped_np[i].copy()]
                out_txt = ['1) before', '2) mean_subtract', '3) threshold', '4) morph_opening', '5) dilate_big','6) erode']
                pid = os.getpid()

            # subract mean image and clip to [0,255]
            cropped_np[i] = np.clip( cropped_np[i] - mean_image, 0, 255).astype(crop_arr_type)
            if debug: out.append(cropped_np[i].copy())

            #threhold in place
            cv2.threshold(src = cropped_np[i], thresh = bin_thresh, maxval = 255, type = cv2.THRESH_BINARY, dst = cropped_np[i]) 
            if debug: out.append(cropped_np[i].copy())

            # apply morph opening ( remove small white elements)
            cv2.morphologyEx(src = cropped_np[i], op = cv2.MORPH_OPEN, kernel = kernel_5, dst = cropped_np[i], borderValue = 0)
            if debug: out.append(cropped_np[i].copy())

            # apply dilate (expand white) to bridge clusters.
            cv2.dilate(  src = cropped_np[i], kernel = kernel_9, dst = cropped_np[i], borderValue = 0)
            if debug: out.append(cropped_np[i].copy())

            cv2.erode(   src = cropped_np[i], kernel = kernel_5, dst = cropped_np[i], borderValue = 0)

            if debug: 
                out.append(cropped_np[i].copy())
                for a,b in zip(out, out_txt):
                    cv2.imshow(f'PID [{pid:>{5}}]: {b}', a)

                k = cv2.waitKey(0)
                if k == 27:  # close on ESC key
                    cv2.destroyAllWindows()
# functions for workers for general stuff                
from graphs_brects import (overlappingRotatedRectangles)

from graphs_brects import (overlappingRotatedRectangles)

from image_processing import (convertGray2RGB)

from bubble_params  import (centroid_area_cmomzz, centroid_area)

from graphs_general import (graph_extract_paths, find_paths_from_to_multi, graph_check_paths, get_connected_components,
                            comb_product_to_graph_edges, for_graph_plots,  extract_clusters_from_edges,
                            set_custom_node_parameters, G2_set_parameters, get_event_types_from_segment_graph)

from graphs_general import (G, G2, key_nodes, keys_segments, G2_t_start, G2_t_end, G2_n_from, G2_n_to, G2_edge_dist, G_time, G_area, G_centroid, G_owner, G_owner_set)

from interpolation import (interpolate_trajectory, extrapolate_find_k_s, interpolateMiddle1D_2, decide_k_s)

from misc import (cyclicColor, timeHMS, modBR, rect2contour, combs_different_lengths, sort_len_diff_f,
                disperse_nodes_to_times, disperse_composite_nodes_into_solo_nodes, find_key_by_value, CircularBuffer, CircularBufferReverse, 
                split_into_bins, lr_reindex_masters, dfs_pred, dfs_succ, old_conn_2_new, lr_evel_perm_interp_data, lr_weighted_sols, 
                save_connections_two_ways, save_connections_merges, save_connections_splits, itertools_product_length, conflicts_stage_1, 
                conflicts_stage_2, conflicts_stage_3, edge_crit_func, two_crit_many_branches, find_final_master_all,
                zp_process, f121_disperse_stray_nodes, f121_interpolate_holes, f121_calc_permutations, f121_precompute_params, f121_get_evolutions)


def track_time(reset = False):
    if reset:
        track_time.last_time = time.time()
        return '(Initializing time counter)'
    else:
        current_time = time.time()
        time_passed = current_time - track_time.last_time
        track_time.last_time = current_time
        return f'({time_passed:.2f} s)'

# if __name__ != '__main__': 
#     track_time(reset = True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'device: {device}')

def timeHMS():
    return datetime.datetime.now().strftime("%H-%M-%S")
                  
def gen_slices(data_length, slice_width):
    # input:    data_length, slice_width = 551, 100
    # output:   [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 551)] 
    return [(i,min(i+slice_width, data_length)) for i in range(0, data_length, slice_width)]

def redistribute_vals_bins(values, num_lists):
    # input: values = [0, 1, 2, 3, 4, 5, 6, 7]; num_lists  = 4
    # output : [[0, 4, 8], [1, 5], [2, 6], [3, 7]]
    max_bins = min(len(values),num_lists)
    lists = [[] for _ in range(max_bins)]
    for i, slice_range in enumerate(values):
        list_index = i % max_bins
        lists[list_index].append(slice_range)
    return lists

def three_dots_anim(inp, delay = 0.75):
    for i in range(4):
        print(f"{inp}{'.' * i}{' ' * (4 - i)}", end='\r')
        time.sleep(delay)


def print2(inp, end='\n', flush=False):
    pid = os.getpid()
    if __name__ == "__main__":
        pid_str = f'PRNT:{pid:>{5}}'
    else:
        pid_str = f'WRKR:{pid:>{5}}'
    
    return print(f'({datetime.datetime.now().strftime("%H-%M-%S")})[{pid_str}] {inp}', end = end, flush = flush)  


def check_ready_blocking(event, time_init = time.time(), wait_msg = '', done_msg = ''):
    # once you stuck waiting for kernels, can generate updating info line with time passed
    t = time.time() - time_init # time offset if counting began earlier than f-n launch
    str_len_max = max(len(wait_msg), len(done_msg)) + 10 # pad with blanks
    while not event.is_set():
        print2(f'{wait_msg} ({t:0.1f} s)'.ljust(str_len_max), end='\r', flush=True)
        time.sleep(0.1)
        t += 0.1
    print2(f'{done_msg} ({t:0.1f} s)'.ljust(str_len_max), flush=True)
    track_time()

    return

def intialize_workers(worker_ready_event, map_x, map_y, rot_param): #xywh, buf_shape, buf_type
    global mx, my, X, Y, W, H, crop_arr_shape, crop_arr_type, rotate_times
    mx, my = map_x, map_y
    rotate_times = rot_param
    #[X, Y, W, H] = xywh
    #crop_arr_shape, crop_arr_type = buf_shape, buf_type
    worker_ready_event.set()        # signal that worker has done init.
    track_time(True)    
    print2('ready')
    return

def init_mean_img(image,): #xywh, buf_shape, buf_type
    global mean_image
    mean_image = image
    return

def img_read_proc(fr_to, img_links, crop_arr_shape, crop_arr_type, xywh):

    buf_cropped   = shared_memory.SharedMemory(name='buf_cropped')
    cropped_np     = np.ndarray(crop_arr_shape, dtype = crop_arr_type, buffer=buf_cropped.buf)
    #print2(f'{cropped_np.shape}')
    (fr,to) = fr_to#, img_links = slice_link
    [X, Y, W, H] = xywh

    for i, img_link in enumerate(img_links):
        img = cv2.imread(img_link, 0)
        
        img = cv2.remap(img,  mx, my, cv2.INTER_LINEAR)
        img = img[Y:Y+H, X:X+W]
        
        #print2(f'{img.shape}, {cropped_np[fr + i,...].shape}')
        cropped_np[fr + i,...] = np.rot90(img, rotate_times)
    
def img_read_proc(fr_to, img_links, crop_arr_shape, crop_arr_type, xywh):

    buf_cropped   = shared_memory.SharedMemory(name='buf_cropped')
    cropped_np     = np.ndarray(crop_arr_shape, dtype = crop_arr_type, buffer=buf_cropped.buf)
    #print2(f'{cropped_np.shape}')
    (fr,to) = fr_to#, img_links = slice_link
    [X, Y, W, H] = xywh

    for i, img_link in enumerate(img_links):
        img = cv2.imread(img_link, 0)
        
        img = cv2.remap(img,  mx, my, cv2.INTER_LINEAR)
        img = img[Y:Y+H, X:X+W]
        
        #print2(f'{img.shape}, {cropped_np[fr + i,...].shape}')
        cropped_np[fr + i,...] = np.rot90(img, rotate_times)     


def CPU_flood_fill(fr_to, crop_arr_shape, crop_arr_type, debug = False):

    buf_cropped = shared_memory.SharedMemory(name='buf_cropped')
    cropped_np  = np.ndarray(crop_arr_shape, dtype = crop_arr_type, buffer=buf_cropped.buf)
    for i in range(*fr_to, 1):
        cv2.floodFill(cropped_np[i], None, (0,0), 0)

minArea, topFilter, leftFilter, rightFilter = 180, 80, 100, 100

filter_params = [minArea, topFilter, leftFilter, rightFilter]

def extract_contour_p1(from_to, crop_arr_shape, crop_arr_type, WH, queue, report = False):
    #global first_iter
    W, H = WH
    buffer          = shared_memory.SharedMemory(name='buf_cropped')
    binary_np       = np.ndarray(crop_arr_shape, dtype = crop_arr_type, buffer=buffer.buf)
    s_from, s_to    = from_to
    idx             = tuple(range(s_from, s_to))
    print2(f'{from_to}')
    #cv2.imshow(f'{os.getpid()}:{s_from}', binary_np[s_from])
    # for g in range(s_from, s_to, 5):
    #     cv2.imshow(f'{g}', binary_np[g])
    # k = cv2.waitKey(0)
    # if k == 27:  # close on ESC key
    #     cv2.destroyAllWindows()
    batch_contours  = {i:[] for i in idx}
    batch_centroid  = batch_contours.copy()
    batch_area      = batch_contours.copy()
    batch_hull      = batch_contours.copy()
    cluster_rect    = {i:{} for i in idx}
    cluster_nodes   = []
    # cv2.imshow('a', binary_np[0])

    # k = cv2.waitKey(0)
    # if k == 27:  # close on ESC key
    #     cv2.destroyAllWindows()
    for i in idx:
        if 1 == 1:
            #cv2.imshow(f'{os.getpid()}:{i}', binary_np[i])
            contours            = cv2.findContours(binary_np[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
            areas               = np.array([int(cv2.contourArea(contour)) for contour in contours])
            boundingRectangles  = np.array([cv2.boundingRect(contour) for contour in contours])
            whereSmall          = np.argwhere(areas < minArea)
                
            # boundingRectangles is an array of size (N,len([x,y,w,h])) where  x,y are b-box corner coordinates, w = width h = height
            topCoords       = boundingRectangles[:,1] + boundingRectangles[:,3]      # bottom    b-box coords y + h
            bottomCoords    = boundingRectangles[:,1]                                # top       b-box coords y
            leftCoords      = boundingRectangles[:,0] + boundingRectangles[:,2]      # right     b-box coords x + w
            rightCoords     = boundingRectangles[:,0]                                # left      b-box coords x

            whereTop    = np.argwhere(topCoords     < topFilter)                    # bottom of b-box is within top band
            whereBottom = np.argwhere(bottomCoords  > (H - topFilter))              # top of b-box is within bottom band
            whereLeft   = np.argwhere(leftCoords    < leftFilter)                   # -"-"-
            whereRight  = np.argwhere(rightCoords   > (W - rightFilter))            # -"-"-
                                                                                
            whereFailed = np.concatenate((whereSmall, whereTop, whereBottom, whereLeft, whereRight)).flatten()
            whereFailed = np.unique(whereFailed)

            # draw over black (cover) border elements
            [cv2.drawContours(  binary_np[i],   contours, j, 0, -1) for j in whereFailed]


        contours, hierarchy = cv2.findContours(binary_np[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
        batch_contours[i]   = contours

        if hierarchy is None:
            whereParentCs = []
        else:
            whereParentCs   = np.argwhere(hierarchy[:,:,3]==-1)[:,1] #[:,:,3] = -1 <=> (no owner)
         
        a = 1
        # get bounding rectangle of each contour for current time step
        boundingRectangles  = {ID: cv2.boundingRect(contours[ID]) for ID in whereParentCs}
        # expand bounding rectangle for small contours to 100x100 pix2 box
        brectDict = {ID: modBR(brec,100) for ID, brec in boundingRectangles.items()} 
        # find pairs of overlaping rectangles, use graph to gather clusters
        frame_clusters = extract_clusters_from_edges(edges = overlappingRotatedRectangles(brectDict,brectDict), nodes = brectDict)
        # create a bounding box for clusters
        
        for IDs in frame_clusters:
            key = (i, *tuple(IDs)) 
            cluster_nodes.append(key)
            cluster_rect[i][key] = cv2.boundingRect(np.vstack([rect2contour(brectDict[ID]) for ID in IDs]))
            

        batch_centroid[  i]   = np.zeros( (len(contours), 2)   )
        batch_area[      i]   = np.zeros( len(contours)   , int)
        batch_hull[      i]   = [None]*len(contours)

        for k, contour in enumerate(contours):
            hull                                = cv2.convexHull(contour)
            centroid, area                      = centroid_area(hull)
            batch_centroid[  i][k]    = centroid
            batch_area[      i][k]    = int(area)
            batch_hull[      i][k]    = hull

    a = 1
    queue.put((idx, batch_contours, batch_centroid, batch_area, batch_hull, cluster_rect, cluster_nodes))
    if report: print2(f'{from_to} iteration time: {track_time()}')
    return 

def gather_contours(time_steps,  queue):
    contours  = {i:[] for i in time_steps}
    centroid  = contours.copy()
    area      = contours.copy()
    hull      = contours.copy()
    cl_rect    = {i:{} for i in time_steps}
    cl_nodes   = []
    while not queue.empty():
        (idx, batch_contours, batch_centroid, batch_area, batch_hull, cluster_rect, cluster_nodes) = queue.get()
        for i in idx:
            contours[i] = batch_contours[i]
            centroid[i] = batch_centroid[i]
            area[i]     = batch_area[i]
            hull[i]     = batch_hull[i]
            cl_rect[i]  = cluster_rect[i]
            cl_nodes.extend(cluster_nodes)

    return contours, centroid, area, hull, cl_nodes, cl_rect
        
if __name__ == '__main__':
    
    manager         = Manager()
    result_queue    = manager.Queue()

    # ============================ MANAGE MODULES WITH FUNCTIONS =============================//

    # --- IF MODULES ARE NOT IN ROOT FOLDER. MODULE PATH HAS TO BE ADDED TO SYSTEM PATHS EACH RUN ---
    # path_modues = r'.\modules'      # os.path.join(mainOutputFolder,'modules')
    # sys.path.append(path_modues)    

    # NOTE: IF YOU USE MODULES, DONT FORGET EMPTY "__init__.py" FILE INSIDE MODULES FOLDER
    # NOTE: path_modules_init = os.path.join(path_modues, "__init__.py")
    # NOTE: if not os.path.exists(path_modules_init):  with open(path_modules_init, "w") as init_file: init_file.write("")

    #--------------------------- IMPORT CUSTOM FUNCITONS -------------------------------------
    from cropGUI import crop_gui_thread

    import workplace_init


    mainOutputFolder            = r'.\post_tests'                           # descritive project name e.g [gallium_bubbles, water_bubbles]
    mainOutputSubFolders =  ['Field OFF Series 7', 'sccm150-meanFix']       # sub-project folder hierarhy e.g [exp setup, parameter] 
    """                                                                       # one more layer will be created for image subset later.
    ========================================================================================================
    ============== SET NUMBER OF IMAGES ANALYZED =================
    ========================================================================================================
    """
    num_processors = 2
    inputImageFolder            = r'E:\relocated\Downloads\150 sccm imgs' #
    #inputImageFolder            = r'C:\Users\mhd01\Downloads\150 sccm imgs'
    # image data subsets are controlled by specifying image index, which is part of an image. e.g image1, image2, image20, image3
    intervalStart   = 1                            # start with this ID
    numImages       = 51                         # how many images you want to analyze.
    intervalStop    = intervalStart + numImages     # images IDs \elem [intervalStart, intervalStop); start-end will be updated depending on available data.

    useMeanWindow   = 0                             # averaging intervals will overlap half widths, read more below
    N               = 700                           # averaging window width
    rotate_times    = 2                             # rotate images 90 deg counterclockwise this many times.

    do_prerun_output= False
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7; thickness = 4;

    print(track_time(reset = True))
    """
    search for images in input folder, get correct subset of images, sort them
    create file system. for more info read docs for each function
    """
    imageLinks, *idx_interval= workplace_init.process_file_names(inputImageFolder, 'bmp', (intervalStart, intervalStop))

    out_main, out_aux = workplace_init.create_folders(mainOutputFolder, mainOutputSubFolders, idx_interval, do_prerun_output, 5)
    imageFolder, stagesFolder, dataArchiveFolder, graphsFolder, imageFolder_output, imageFolder_pre_run = out_main

    (cropMaskPath, cropMaskMissing, graphsPath, segmentsPath, contoursHulls, mergeSplitEvents, 
                meanImagePath, meanImagePathArr, archivePath, binarizedArrPath, post_binary_data) = out_aux
    
    dt = np.uint8
    mx, my      = np.load('./mapx.npy'), np.load('./mapy.npy')
    base_img = convertGray2RGB(cv2.remap(cv2.imread(imageLinks[0],0),  mx, my, cv2.INTER_LINEAR))
    og_shape    = base_img.shape[:2]

    slice_size      = 10
    slices          = gen_slices(len(imageLinks), slice_size)
    slices_links    = [((fr,to), imageLinks[fr:to]) for fr,to in slices]
    
    #if not os.path.exists(archivePath):

    # WHATS THE TARGET:
        # 3 STAGES:
            # 1) CONTOUR PROCESSING - INISIDE LAYER (MIGHT NEED DATA FROM BINARY IMGS)
            # 2) BINARY OPS         - MIDDLE (NEED CROPPED ARRAY IN MEM)
            # 3) CROPPING           - FIRST LAYER 
        # 
    # -> SAVE PROCESSED CROPPED IMAGES. KEEP IN MEMORY FOR FURTHER PROCESSING. IN SHARED
    # THUS -> RESERVE MEM FOR CROPPED ARRAY. LOAD-STREAM TO BUFFER OR FILL DURING PROCESSING
    # 
    kernels_ready = Event()
    t0_kernel = time.time() #intialize_workers(worker_ready_event, map_x, map_y, xywh, buf_shape, buf_type
    
    with Pool(processes = num_processors, initializer = intialize_workers, initargs = (kernels_ready, mx, my, rotate_times)) as pool:
        
        if cropMaskMissing: 
            # launch crop gui on a thread and do work until crop params are needed.
            print2(f"\nNo crop mask in {mainOutputFolder} folder!, creating mask : {cropMaskPath}")
            
            check_ready_thread  = crop_gui_thread(base_img)
            check_ready_thread.start()

            while check_ready_thread.is_alive():
                three_dots_anim('waiting for crop GUI to close')

            check_ready_thread.join()

            [X, Y, W, H], (p1, p2) = check_ready_thread.result

            cv2.rectangle(base_img  , p1, p2, [0,0,255], -1)
            cv2.imwrite(  cropMaskPath  , base_img)
            
        else:
            cropMask = cv2.imread(cropMaskPath,1)
            # ---------------------------- ISOLATE RED RECTANGLE BASED ON ITS HUE ------------------------------
            cropMask = cv2.cvtColor(cropMask, cv2.COLOR_BGR2HSV)

            lower_red = np.array([(0,50,50), (170,50,50)])
            upper_red = np.array([(10,255,255), (180,255,255)])

            manualMask = cv2.inRange(   cropMask, lower_red[0], upper_red[0])
            manualMask += cv2.inRange(  cropMask, lower_red[1], upper_red[1])

            # --------------------- EXTRACT MASK CONTOUR-> BOUNDING RECTANGLE (USED FOR CROP) ------------------
            contours = cv2.findContours(manualMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            [X, Y, W, H] = cv2.boundingRect(contours[0])
        # predefine buffer filled with zeros, with shape which depends on image rotation.
        if  rotate_times % 2 == 0: dims  = (len(imageLinks), H, W)
        else:                      dims  = (len(imageLinks), W, H)
        # might write a function to create a buffer. need options to clear if after and to stream data in chunks
        size_in_bytes   = int(np.prod(dims)) * np.dtype(dt).itemsize #2304000000 
        shared_data     = shared_memory.SharedMemory(create=True, size=size_in_bytes, name="buf_cropped")
        np_buff         = np.ndarray(dims, dtype=dt, buffer=shared_data.buf)
        np_buff         = np.zeros_like(np_buff)    # this part -> to stream on demand
        print2(f'dims:{dims}')
        if 1 == 1:
            """wait kernel launch"""
            if not kernels_ready.is_set():  #event_or_queue 0 or 1
                check_ready_blocking(kernels_ready, t0_kernel, 'Waiting until kernels are ready...', 'Kernels are ready!')

            """imort image, remap and crop"""  
            track_time()
           
            async_result = pool.starmap_async(img_read_proc, (s + (dims, dt, [X, Y, W, H]) for s in slices_links))
            async_result.wait()  
            if not async_result.successful():
                print2(f'shit failed: {async_result.get()}')
            
            print2(f'read remap crop rotate calc... done {track_time()}')

            bin_thresh = 10
            t0_mean_calc = time.time()
            if calc_GPU:
                #cv2.imshow('0) pre', np_buff[0])
                print2(f'create dataset... {track_time()}')
                dataset_here = SharedMemoryDataset('buf_cropped', dims, dt)
                dataloader   = DataLoader(dataset_here, batch_size=1, shuffle=False)

                print2(f'create dataset...  done {track_time()}')
                """compute stack mean"""
                mean_gpu        = batch_axis0_mean(dataloader, dims[-2:], ref_type = cuda_ref_type, device= device)
                mean_gpu        = mean_gpu.unsqueeze(0)
                print2(f'compute mean...  done {track_time()}')
                
                """blur mean image"""
                kernel_blur_size    = 5 #, dtype = ref_type, device = device)
                kernel_blur   = kernel_circular(kernel_blur_size, dtype = cuda_ref_type, normalize = True, device = device)
                mean_gpu_blur = torch_blur(mean_gpu, kernel_blur_size, kernel_blur, mode = 1, device = device)
                print2(f'compute blur...  done {track_time()}')
                """compute binary pipe"""
                #cv2.imshow('1) mean_gpu_blur', mean_gpu_blur.to('cpu').numpy().astype(dt)[0,0])
                i = 0
                # for batch in dataloader:
                #     GPU_binary_pipe(i, np_buff, batch, mean_gpu_blur, bin_thresh, cuda_ref_type, dt, debug = 1)
                #     i += len(batch)
                
                GPU_binary_pipe(dataloader, mean_gpu_blur, bin_thresh, dims, cuda_ref_type, dt, debug = 0)

                

            else:

                async_result = pool.starmap_async(mean_slice, ((s, dims, dt, result_queue) for s in slices))

                async_result.wait()                 # wait until all workers are done. can do other stuff before.
                
                if not async_result.successful():
                    print2(f'shit failed: {async_result.get()}')

                mean_image = mean_slice_finish(len(slices), dims[-2:], result_queue)
                #cv2.imshow('mean', np.uint8(mean_image))
                
                # init global mean_image to each worker
                result = pool.map_async(init_mean_img, [mean_image]*num_processors)
                result.wait

                async_result = pool.starmap_async(CPU_binary_pipe, ((s, bin_thresh, dims, dt) for s in slices))

                async_result.wait()                 # wait until all workers are done. can do other stuff before.
                
                if not async_result.successful():
                    print2(f'shit failed: {async_result.get()}')
                a = 1
            print2(f'binary time (on GPU: {calc_GPU}) = ({(time.time() - t0_mean_calc):.2f} s')
            print2(f'fill border {track_time()}')
            border_thickness = 5
            np_buff  = np.ndarray(dims, dtype=dt, buffer=shared_data.buf)

            np_buff[:,     :border_thickness   , :                 ]   = 255
            np_buff[:,     -border_thickness:  , :                 ]   = 255
            np_buff[:,     :                   , :border_thickness ]   = 255
            np_buff[:,     :                   , -border_thickness:]   = 255

            

            #cv2.imshow('uuuagua before', np_buff[1])
            print2(f'fill border end {track_time()} start flood fill')

            async_result = pool.starmap_async(CPU_flood_fill, ((s, dims, dt) for s in slices))

            async_result.wait()                 # wait until all workers are done. can do other stuff before.
            
            if not async_result.successful():
                print2(f'shit failed: {async_result.get()}')
                
            # NOTE: unfortunately you cannot use flood fill in 3D because you can flood trajectories away from border
            # remove border components on image stack. connectivity is set by footprint
            # footprint = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)).reshape(1,3,3) # prevents connectivity in time direction
            # flood_fill(image = np_buff, seed_point = (0,0,0), new_value = 0, footprint = footprint, in_place=True)
            
            print2(f'flood fill end {track_time()}')
            #cv2.imshow('uuuagua', np_buff[0])

            """Perform contour recognition on CPU = """
            track_time()
            async_result = pool.starmap_async(extract_contour_p1, ((s, dims, dt, [W, H], result_queue) for s in slices))
            async_result.wait()  
            if not async_result.successful():
                print2(f'shit failed: {async_result.get()}')
            
            print2(f'end conour calc... {track_time()}')
            sol =   gather_contours(range(dims[0]),  result_queue)
            (g0_contours, g0_contours_centroids, g0_contours_areas,
             g0_contours_hulls, pre_nodes_all , pre_rect_cluster) = sol
            
            cv2.imshow('uuuagua', np_buff[0])
            k = cv2.waitKey(0)
            if k == 27:  # close on ESC key
                cv2.destroyAllWindows()

            # cropped images are loaded into buffer.
                # next must do mean calc. either CPU or GPU.
            #
            # PARALLEL READ
            #cv2.imshow('a',np_buff[0,...])
        #     if 1 == 1:#if not os.path.exists(archivePath):
        #         print(track_time())
                
        #         dataArchive = np.zeros((len(imageLinks), *og_shape), dtype=np.uint8)
        #         for idx, imageLink in enumerate(imageLinks):
        #             #if idx % 200 == 0: print(f'({idx} 200 done: {track_time()}')
        #             dataArchive[idx, :, :] = cv2.imread(imageLink, 0)
        #         print(f'(1 core import done:) {track_time()}')
        #         #np.savez(archivePath,dataArchive)
        #         #print(f'(1 core save done:) {track_time()}')
        #     else:
        #         dataArchive = np.load(archivePath)['arr_0']
        #         print(f'(1 core load done:) {track_time()}')

        #     #dataArchive0 = None

        #     #dataArchive2 = np.zeros_like(dataArchive)
        #     for idx in range(dataArchive.shape[0]):
        #         #if (idx + 1) % 200 == 0: print(f'({idx} 200 done: {track_time()}')
        #         dataArchive[idx, :, :] = cv2.remap(dataArchive[idx],  mx, my, cv2.INTER_LINEAR)
        #     print(f'(1 core remap done:) {track_time()}')
        #     # pr = 4

        #     # with Pool(processes=pr) as pool:
        #     #     dataArchive2 = pool.starmap(cv2.remap, ((dataArchive[i], mx, my, cv2.INTER_LINEAR) for i in range(dataArchive.shape[0])))
        #     #cv2.imshow('a', dataArchive[10])

        #     dataArchive = np.rot90(dataArchive, rotate_times, (1,2))
        #     print(f'(1 core rotate done:) {track_time()}')

        #     if cropMaskMissing: # crop GUI might be still running. wait to finish
        #         while check_ready_thread.is_alive():
        #             three_dots_anim('waiting for crop GUI to close')
        #         check_ready_thread.join()
        #         [X, Y, W, H], (p1, p2) = check_ready_thread.result
        #         cv2.rectangle(base_img  , p1, p2, [0,0,255], -1)
        #         cv2.imwrite(  cropMaskPath  , base_img)

        #     dataArchive = dataArchive[:,Y:Y+H, X:X+W]
        #     print(f'(1 core crop done:) {track_time()}')
            
        #     np.savez(archivePath, dataArchive)
        #     print(f'(1 core save done:) {track_time()}')
        # else:
        #     dataArchive = np.load(archivePath)['arr_0'] 
        #     print(f'(1 core load done:) {track_time()}')

        # dataArchive = torch.from_numpy(dataArchive)

        # from torch.multiprocessing import Process, Manager, set_start_method
        # set_start_method('spawn', force=True)
        # manager = Manager()
        # result_queue = manager.Queue()

        # processes = []

        # data_size = dataArchive.shape[0]
        # # Determine the size of each slice
        # slice_size = 500

        # # Create a Manager and a shared Queue
        # manager = Manager()
        # result_queue = manager.Queue()
        
        # # Create processes
        # num_processors = 4
        # processes = []
        # slices          = gen_slices(data_size, slice_size)
        # slices_batches  = redistribute_vals_bins(slices, num_processors)   
        # print(torch.multiprocessing.get_sharing_strategy())
        # print(torch.multiprocessing.get_all_sharing_strategies())
        # dataArchive.share_memory_()
        # for i in range(num_processors):
        #     print(f'({timeHMS()})[{os.getpid()}] spawning worker: {i}')
        #     p = Process(target=compute_mean_image_batch, args=(slices_batches[i], dataArchive, result_queue))
        #     p.start()
        #     processes.append(p)
        # t0_CPU_MP = time.time()
        # print(f'({timeHMS()})[{os.getpid()}] waiting processes to end..')    
        # for p in processes:
        #     p.join()
        # print(f'({timeHMS()})[{os.getpid()}] joining results... Done!')    
        # # Collect results from the queue
        # total_elements = 0
        # running_sum = 0

        # while not result_queue.empty():
        #     num_elements, mean_image = result_queue.get()
        #     total_elements += num_elements
        #     running_sum += num_elements * mean_image

        # # Compute the final mean image
        # final_mean_image = running_sum / total_elements
        # print(f'CPU_MP time =({time.time() - t0_CPU_MP:.2f} s')
        # #chunk_size = len(dataArchive) // 2
        # #num_elements, mean_image = parallel_compute_mean_image([dataArchive[i:i+chunk_size] for i in range(0, len(dataArchive), chunk_size)])
        # cv2.imshow('mean_parallel', mean_image.to(torch.uint8).numpy())
        # t0_GPU          = time.time()
        # print(f'calculating batch time mean start: {track_time()}')

        # your_dataset = YourDataset(dataArchive)
        # dataloader = DataLoader(your_dataset, batch_size=50, shuffle=False)
        # ref_type = torch.float
        # kernel_size = 5
        # kernel_mean = kernel_circular(kernel_size, normalize = True).to(device, dtype = ref_type)

        # blurred_image_t = calculate_weighted_mean_image(dataloader, kernel_mean, kernel_size, ref_type = ref_type)
        # torch.cuda.empty_cache()

        # print(f'calculating batch time mean done: {track_time()}')

                
        # print(f'binary processing start {track_time()}')
        # processed_data_cpu = []
        # proc = binary_pipe(dataloader, blurred_image_t, 10)
        # result = torch.cat(processed_data_cpu, dim=0)
        # del result
        # print(f'binary processing end: {track_time()}')

        # print(f'GPU time =({time.time() - t0_GPU:.2f} s')

        # t0_CPU = time.time()
        # meanImage = np.mean(dataArchive, axis=0)
        # print(f'time-mean done: {track_time()}')
        # blurMean = cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT).astype(np.uint8)  
        # print(f'blur mean done: {track_time()}')

        # dataArchive -= blurMean    
        # dataArchive = dataArchive.astype(np.uint8)        
        # print(f'subract mean done {track_time()}')
        # thresh0 = 10
        # dataArchive[dataArchive < thresh0] = 0
        # dataArchive[dataArchive >= thresh0] = 255
        # #dataArchive_b = np.where(dataArchive < thresh0, 0, 255).astype(np.uint8)  
        # print(f'thresholding done: {track_time()}')
        # dataArchive = np.uint8([cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) for img in dataArchive])    # delete white objects
        # print(f'opening done: {track_time()}')
        # dataArchive = np.uint8([cv2.dilate(img, np.ones((8,8), np.uint8) ) for img in dataArchive])    
        # print(f'dilate done: {track_time()}')
        # dataArchive = np.uint8([cv2.erode(img, np.ones((5,5), np.uint8) ) for img in dataArchive]) 
        # print(f'erode done: {track_time()}')

        # print(f'CPU time =({time.time() - t0_CPU:.2f} s')
        # #cv2.imshow('b', dataArchive[10])
        a = 1

        # import concurrent.futures
        # pr = 10

        # # Split dataArchive into chunks
        # chunk_size = len(dataArchive) // pr
        # data_chunks = [dataArchive[i:i+chunk_size] for i in range(0, len(dataArchive), chunk_size)]

        # with Pool(processes=pr) as pool:
        #     # Pass each chunk to a separate process
        #     dataArchive2 = pool.starmap(process_chunk, ((chunk, mx, my) for chunk in data_chunks))

        # # Combine the results from different processes into a single list
        # dataArchive2 = [item for sublist in dataArchive2 for item in sublist]


        

        # def process_image(img):
        #     return cv2.remap(img, mx, my, cv2.INTER_LINEAR)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=pr) as executor:
        #     dataArchive2 = list(executor.map(process_image, dataArchive))
        # print(f'({pr} cores remap done:) {track_time()}')
        # print(len(dataArchive2))


        #print(dataArchive.shape)
        # values, counts = np.unique(dataArchive[:,1], return_counts=True)
        # print(track_time(), max(counts))
        a= 1
        # Convert the results to a NumPy array
        #dataArchive = np.array(results, dtype=np.uint8)

        # dataArchive = np.zeros((len(imageLinks),H,W),np.uint8)                  # predefine storage

        # mapXY       = (np.load('./mapx.npy'), np.load('./mapy.npy'))            # fish-eye correction map

        # for i,imageLink in tqdm(enumerate(imageLinks), total=len(imageLinks)):

        #     dataArchive[i] = proc_img(imageLink)
        


        # a = 1
        # cv2.imshow('a', dataArchive[10])
        # a = 1

    shared_data.close()
    shared_data.unlink()
 
    k = cv2.waitKey(0)
    if k == 27:  # close on ESC key
        cv2.destroyAllWindows()