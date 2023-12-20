
import numpy as np, itertools, networkx as nx, sys, copy,  cv2, os, glob, re, pickle, time, datetime, os
from multiprocessing import shared_memory, Manager, Pool, Event
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

calc_GPU = True

if __name__ == '__main__':
    path_modues = r'.\modules'      # os.path.join(mainOutputFolder,'modules')
    sys.path.append(path_modues)
    if calc_GPU:
        path_modules_GPU = r'..\python-code-bits-for-image-processing-and-linear-algebra\multiprocessing-GPU'
        sys.path.append(path_modules_GPU)
        from gpu_torch_cuda_functions import (batch_axis0_mean, torch_blur, kernel_circular, morph_erode_dilate)
        from module_GPU import (dataset_create, SharedMemoryDataset)
        from torch.utils.data import DataLoader
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cuda_ref_type = torch.float16

        def b_pipe(i, buffer, batch, image_blurred, threshold, cuda_ref_type, cpu_ref_type, debug = 0):
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
            # cv2.imshow('binar', buffer[0]); cv2.imshow('binar_old', buffer[1])
            return 


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
                  
# class YourDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Assuming data is a stack of images
#         sample = self.data[idx]
#         return sample


# def read_img(link):
#     return cv2.imread(link, 0)  

# def read_img2(idx, link, mode):
#     if idx % 200 == 0:
#         print(f'[{os.getpid()}]{idx} 200 done: {track_time()}')
#     return cv2.imread(link, mode)  

# def process_chunk(chunk, mx, my):
#         return [cv2.remap(img, mx, my, cv2.INTER_LINEAR) for img in chunk]

# def morph_erode_dilate(im_tensor, kernel, mode):
#     """
#     mode = 0 = erosion; mode = 1 = dilation; im_tensor is 1s and 0s
#     convolve image (stack) with a kernel. 
#     dilate  = keep partial/full overlap between image and kernel.   means masked sum > 0
#     erosion = keep only full overlap.                               means masked sum  = kernel sum.
#     erode: subtract full overlap area from result, pixels with full overlap will be 0. rest will go below 0.
#     erode:add 1 to bring full overlap pixels to value of 1. partial overlap will be below 1 and will be clamped to 0.
#     dilate: just clamp
#     """
#     padding = (kernel.shape[-1]//2,)*2
#     torch_result0   = torch.nn.functional.conv2d(im_tensor, kernel, padding = padding, groups = 1)
#     if mode == 0:
#         full_area = torch.sum(kernel)
#         torch_result0.add_(-full_area + 1)
#     return torch_result0.clamp_(0, 1)


# def binary_pipe(dataloader, blurred_image_t, threshold, device='cuda'):
#     ref_type = blurred_image_t.dtype
#     #kernel_5 = torch.ones((1,1,5,5)).to(device, dtype = ref_type)
#     #kernel_8 = torch.ones((1,1,8,8)).to(device, dtype = ref_type)
#     kernel_5 = kernel_circular(5, normalize = False).to(device, dtype = ref_type)
#     kernel_8 = kernel_circular(8, normalize = False).to(device, dtype = ref_type)
#     time_load = 0.0
#     time_save = 0.0
#     for batch in dataloader:
#         #print(f'send to device: {track_time()}')
#         t = time.time()
#         batch = batch.to(device).to(ref_type)
#         time_load += (time.time() - t)
#         batch = batch.unsqueeze_(1)    
#         batch.add_(-blurred_image_t)
#         #print(f'thld 1 start: {track_time()}')

#         batch = batch > threshold       # may be not cool since its changes type size
#         #print(f'thresholding time 1: {track_time()}')
#         batch = batch.to(ref_type)
#         #test0 = batch[0]
#         # opening = erode->dilate
#         #test0 = batch[0, 0, :, :]
#         batch = morph_erode_dilate(batch, kernel_5, mode = 0)
#         #test1 = batch[0, 0, :, :]
#         batch = morph_erode_dilate(batch, kernel_5, mode = 1)
#         #test2 = batch[0, 0, :, :]
#         #cv2.imshow('mn_diff0', ((test0.detach().cpu().numpy())*255.0).astype(np.uint8))
#         #cv2.imshow('mn_diff1', ((test1.detach().cpu().numpy())*255.0).astype(np.uint8))
#         #cv2.imshow('mn_diff2', ((test2.detach().cpu().numpy())*255.0).astype(np.uint8))

#         # dilate 8 to join and return 5
#         batch = morph_erode_dilate(batch, kernel_8, mode = 1)
#         batch = morph_erode_dilate(batch, kernel_5, mode = 0)
#         #del batch
#         #torch.cuda.empty_cache()
#         #print(f'clear mem 1: {track_time()}')
#         batch = batch.to(torch.uint8)
#         t = time.time()
#         processed_batch_cpu = batch.to('cpu')
#         time_save += (time.time() - t)
#         processed_data_cpu.append(processed_batch_cpu)

#     print(f'load time= {time_load:.2f} s, save time= {time_save:.2f} ')
#     return True

# def kernel_circular(width, normalize):
#     ker = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width))).unsqueeze_(0).unsqueeze_(0)
#     if normalize:
#         return ker/torch.sum(ker)
#     else:
#         return ker
    

# def compute_mean_image(slice_start, slice_end, data, result_queue):
#     # Process data slice from slice_start to slice_end
#     batch = data[slice_start:slice_end]
    
#     # Your computation logic here (compute mean image, etc.)
#     mean_image = batch.to(torch.float).mean(dim=0)
#     num_elements = len(batch)
#     print(f'asdas: {os.getpid()}, numImgas = {num_elements}')
#     # Put results into the queue
#     result_queue.put((num_elements, mean_image))
#     return

# def compute_mean_image_batch(slices, data, result_queue):
#     print(f'({timeHMS()})[{os.getpid()}] starting..')
#     for i, slice in enumerate(slices):
#         slice_start, slice_end = slice
#         batch = data[slice_start:slice_end]

#         mean_image = batch.to(torch.float).mean(dim=0)
#         num_elements = len(batch)
        

#         result_queue.put((num_elements, mean_image))
#         #print(f'({timeHMS()})[{os.getpid()}] finished batch = {i}: #images: {num_elements}')
#     return

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
    
        

if __name__ == '__main__':
    
    # ============================ MANAGE MODULES WITH FUNCTIONS =============================//

    # --- IF MODULES ARE NOT IN ROOT FOLDER. MODULE PATH HAS TO BE ADDED TO SYSTEM PATHS EACH RUN ---
    # path_modues = r'.\modules'      # os.path.join(mainOutputFolder,'modules')
    # sys.path.append(path_modues)    

    # NOTE: IF YOU USE MODULES, DONT FORGET EMPTY "__init__.py" FILE INSIDE MODULES FOLDER
    # NOTE: path_modules_init = os.path.join(path_modues, "__init__.py")
    # NOTE: if not os.path.exists(path_modules_init):  with open(path_modules_init, "w") as init_file: init_file.write("")

    #--------------------------- IMPORT CUSTOM FUNCITONS -------------------------------------
    from cropGUI import crop_gui_thread

    from graphs_brects import (overlappingRotatedRectangles)

    from image_processing import (convertGray2RGB, undistort)

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

            
            print2(f'end mean calc... {track_time()}')

            
            if calc_GPU:
                print2(f'create dataest... {track_time()}')
                dataset_here = SharedMemoryDataset('buf_cropped', dims, dt)
                dataloader   = DataLoader(dataset_here, batch_size=1, shuffle=False)

                print2(f'create dataest...  done{track_time()}')
                """compute stack mean"""
                mean_gpu        = batch_axis0_mean(dataloader, dims[-2:], ref_type = cuda_ref_type, device= device)
                mean_gpu        = mean_gpu.unsqueeze(0)
                """blur mean image"""
                kernel_blur_size    = 5 #, dtype = ref_type, device = device)
                kernel_blur   = kernel_circular(kernel_blur_size, dtype = cuda_ref_type, normalize = True, device = device)
                mean_gpu_blur = torch_blur(mean_gpu, kernel_blur_size, kernel_blur, device = device)

                """compute binary pipe"""
                # shared_data     = shared_memory.SharedMemory(name="buf_cropped")
                # np_buff         = np.ndarray(dims, dtype=dt, buffer=shared_data.buf)
                cv2.imshow('aa', np_buff[0])
                i = 0
                for batch in dataloader:
                    b_pipe(i, np_buff, batch, mean_gpu_blur, 10, cuda_ref_type, dt)
                    i += len(batch)
                # shared_data     = shared_memory.SharedMemory(name="buf_cropped")
                # np_buff         = np.ndarray(dims, dtype=dt, buffer=shared_data.buf)
                cv2.imshow('binar', np_buff[0])
                #cv2.imshow('mean_gpu_blur', mean_gpu_blur.to('cpu').numpy().astype(dt)[0,0])
                # for batch in dataloader:
                #     img = batch[0]
                #     cv2.imshow('a', img.numpy())


            #cv2.imshow('a', cropped_np[0,...])
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