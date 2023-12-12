
import numpy as np, itertools, networkx as nx, sys, copy,  cv2, os, glob, re, pickle, time, datetime
from matplotlib import pyplot as plt
#from tqdm import tqdm
#from collections import defaultdict
import torch
from torch.nn.functional import conv2d
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
#from scipy.ndimage import generate_binary_structure
from kornia import morphology as morph

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def track_time(reset = False):
    if reset:
        track_time.last_time = time.time()
        return '(Initializing time counter)'
    else:
        current_time = time.time()
        time_passed = current_time - track_time.last_time
        track_time.last_time = current_time
        return f'({time_passed:.2f} s)'

if __name__ != '__main__': 
    track_time(reset = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

def timeHMS():
    return datetime.datetime.now().strftime("%H-%M-%S")
                  
class YourDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming data is a stack of images
        sample = self.data[idx]
        return sample
    
def proc_img(idx,imageLink):
    #print(imageLink[-8:-4])
    #image = cv2.remap(cv2.imread(imageLink,0), mapXY[0], mapXY[1],cv2.INTER_LINEAR)[Y:Y+H, X:X+W]
    gg = np.cos(idx)
    if rotateImageBy != -1:
        return [os.getpid(),idx]#cv2.rotate(image, rotateImageBy)
    else:
        return [os.getpid(),idx]#image


def read_img(link):
    return cv2.imread(link, 0)  



def read_img2(idx, link, mode):
    if idx % 200 == 0:
        print(f'[{os.getpid()}]{idx} 200 done: {track_time()}')
    return cv2.imread(link, mode)  

def process_chunk(chunk, mx, my):
        return [cv2.remap(img, mx, my, cv2.INTER_LINEAR) for img in chunk]

def morph_erode_dilate(im_tensor, kernel, mode):
    """
    mode = 0 = erosion; mode = 1 = dilation; im_tensor is 1s and 0s
    convolve image (stack) with a kernel. 
    dilate  = keep partial/full overlap between image and kernel.   means masked sum > 0
    erosion = keep only full overlap.                               means masked sum  = kernel sum.
    erode: subtract full overlap area from result, pixels with full overlap will be 0. rest will go below 0.
    erode:add 1 to bring full overlap pixels to value of 1. partial overlap will be below 1 and will be clamped to 0.
    dilate: just clamp
    """
    padding = (kernel.shape[-1]//2,)*2
    torch_result0   = torch.nn.functional.conv2d(im_tensor, kernel, padding = padding, groups = 1)
    if mode == 0:
        full_area = torch.sum(kernel)
        torch_result0.add_(-full_area + 1)
    return torch_result0.clamp_(0, 1)

def calculate_weighted_mean_image(dataloader, kernel_mean, kernel_size, ref_type, device='cuda'):
    weighted_mean_image = None
    total_weight = 0

    for batch in dataloader:
        batch = batch.to(device, dtype = ref_type) # Move the batch to GPU
        batch = batch.unsqueeze_(1)                             # (N,h,w) -> (N,1,h,w)
        batch_mean = torch.mean(batch, dim = 0).unsqueeze(0)    # mean dim 0, bring back to shape (1,1,h,w)
        batch_weight = batch.size(0)
        
        if weighted_mean_image is None:  # Update the weighted mean
            weighted_mean_image = batch_mean.clone()
        else:
            weighted_mean_image = (weighted_mean_image * total_weight + batch_mean * batch_weight) / (total_weight + batch_weight)

        total_weight += batch_weight
        
        #del batch # Free up GPU memory
    
    weighted_mean_image = weighted_mean_image.to(device) # Move the result to GPU (if not already)

    conv = nn.Conv2d(1, 1, kernel_size = kernel_size, bias = False, padding = 'same', padding_mode ='reflect').to(device)
    conv.weight = nn.Parameter(kernel_mean) 
    blurred_image_t = conv(weighted_mean_image)
    return blurred_image_t

def binary_pipe(dataloader, blurred_image_t, threshold, device='cuda'):
    ref_type = blurred_image_t.dtype
    #kernel_5 = torch.ones((1,1,5,5)).to(device, dtype = ref_type)
    #kernel_8 = torch.ones((1,1,8,8)).to(device, dtype = ref_type)
    kernel_5 = kernel_circular(5, normalize = False).to(device, dtype = ref_type)
    kernel_8 = kernel_circular(8, normalize = False).to(device, dtype = ref_type)
    time_load = 0.0
    time_save = 0.0
    for batch in dataloader:
        #print(f'send to device: {track_time()}')
        t = time.time()
        batch = batch.to(device).to(ref_type)
        time_load += (time.time() - t)
        batch = batch.unsqueeze_(1)    
        batch.add_(-blurred_image_t)
        #print(f'thld 1 start: {track_time()}')

        batch = batch > threshold       # may be not cool since its changes type size
        #print(f'thresholding time 1: {track_time()}')
        batch = batch.to(ref_type)
        #test0 = batch[0]
        # opening = erode->dilate
        #test0 = batch[0, 0, :, :]
        batch = morph_erode_dilate(batch, kernel_5, mode = 0)
        #test1 = batch[0, 0, :, :]
        batch = morph_erode_dilate(batch, kernel_5, mode = 1)
        #test2 = batch[0, 0, :, :]
        #cv2.imshow('mn_diff0', ((test0.detach().cpu().numpy())*255.0).astype(np.uint8))
        #cv2.imshow('mn_diff1', ((test1.detach().cpu().numpy())*255.0).astype(np.uint8))
        #cv2.imshow('mn_diff2', ((test2.detach().cpu().numpy())*255.0).astype(np.uint8))

        # dilate 8 to join and return 5
        batch = morph_erode_dilate(batch, kernel_8, mode = 1)
        batch = morph_erode_dilate(batch, kernel_5, mode = 0)
        #del batch
        #torch.cuda.empty_cache()
        #print(f'clear mem 1: {track_time()}')
        batch = batch.to(torch.uint8)
        t = time.time()
        processed_batch_cpu = batch.to('cpu')
        time_save += (time.time() - t)
        processed_data_cpu.append(processed_batch_cpu)

    print(f'load time= {time_load:.2f} s, save time= {time_save:.2f} ')
    return True

def kernel_circular(width, normalize):
    ker = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width))).unsqueeze_(0).unsqueeze_(0)
    if normalize:
        return ker/torch.sum(ker)
    else:
        return ker
    
# def compute_mean_image(batch, result_queue):
    
#     num_elements = batch.size(0)
#     print(f'asdas: {os.getpid()}, numImgas = num_elements')
#     mean_image = torch.mean(batch.to(torch.float), dim=0)
#     result_queue.put((num_elements, mean_image))

def compute_mean_image(slice_start, slice_end, data, result_queue):
    # Process data slice from slice_start to slice_end
    batch = data[slice_start:slice_end]
    
    # Your computation logic here (compute mean image, etc.)
    mean_image = batch.to(torch.float).mean(dim=0)
    num_elements = len(batch)
    print(f'asdas: {os.getpid()}, numImgas = {num_elements}')
    # Put results into the queue
    result_queue.put((num_elements, mean_image))
    return

def compute_mean_image_batch(slices, data, result_queue):
    print(f'({timeHMS()})[{os.getpid()}] starting..')
    for i, slice in enumerate(slices):
        slice_start, slice_end = slice
        batch = data[slice_start:slice_end]

        mean_image = batch.to(torch.float).mean(dim=0)
        num_elements = len(batch)
        

        result_queue.put((num_elements, mean_image))
        #print(f'({timeHMS()})[{os.getpid()}] finished batch = {i}: #images: {num_elements}')
    return

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




if __name__ == '__main__':
    mainOutputFolder     = r'.\post_tests'                           # descritive project name e.g [gallium_bubbles, water_bubbles]
    mainOutputSubFolders =  ['Field OFF Series 7', 'sccm150-meanFix']   
    inputImageFolder     = r'E:\relocated\Downloads\150 sccm' #

    intervalStart   = 1                           # start with this ID
    numImages       = 1999                         # how many images you want to analyze.
    intervalStop    = intervalStart + numImages     # images IDs \elem [intervalStart, intervalStop); start-end will be updated depending on available data.

    useMeanWindow   = 0                             # averaging intervals will overlap half widths, read more below
    N               = 700                           # averaging window width
    rotateImageBy   = cv2.ROTATE_180                # -1= no rotation, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180 


    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7; thickness = 4;

    #if 1 == 1:                               # prep image links, get min/max image indexes
    # ================================================================================================
    # ============================= GET PATHS TO IMAGES AND RESORT BY ID =============================
    # 1) get image links
    # 2) extract integers in name [.\img3509, .\img351, .\img3510, ....] -> [3509, 351, 3510, ...]
    # 3) filter by index 
    # 4) sort order by index [.\img3509, .\img351, .\img3510, ...] -> [.\img351, .\img3509, .\img3510, ...]
    imageLinks = glob.glob(inputImageFolder + "**/*.bmp", recursive=True) 
    if len(imageLinks) == 0:
        input("No files inside directory, copy them and press any key to continue...")
        imageLinks = glob.glob(inputImageFolder + "**/*.bmp", recursive=True)                                          # 1)

    extractIntergerFromFileName = lambda x: int(re.findall('\d+', os.path.basename(x))[0])                             # 2)
    imageLinks = list(filter(lambda x: intervalStart <= extractIntergerFromFileName(x) < intervalStop , imageLinks))   # 3)
    imageLinks.sort(key=extractIntergerFromFileName)                                                                   # 4)

    intervalStart   = extractIntergerFromFileName(imageLinks[0])        # update start index, so its captures in subfolder name.
    intervalStop    = extractIntergerFromFileName(imageLinks[-1])       # update end index 

    """
    ========================================================================================================//
    ======================================== BUILD PROJECT FOLDER HIERARCHY ================================//
    --------------------------------------------------------------------------------------------------------//
    ------------------------------------ CREATE MAIN PROJECT, SUBPROJECT FOLDERS ---------------------------//
    """
    if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)  
    mainOutputSubFolders.append(f"{intervalStart:05}-{intervalStop:05}")       # sub-project folder hierarhy e.g [exp setup, parameter, subset of data]

    for folderName in mainOutputSubFolders:     
        mainOutputFolder = os.path.join(mainOutputFolder, folderName)               
        if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)

    # -------------------------------- CREATE VARIOUS OUTPUT FOLDERS -------------------------

    imageFolder        = os.path.join(mainOutputFolder, 'images'    )
    stagesFolder       = os.path.join(mainOutputFolder, 'stages'    )
    dataArchiveFolder  = os.path.join(mainOutputFolder, 'archives'  )
    graphsFolder       = os.path.join(mainOutputFolder, 'graphs'    )

    [os.mkdir(folder) for folder in (imageFolder, stagesFolder, dataArchiveFolder, graphsFolder) if not os.path.exists(folder)]

    imageFolder_pre_run = os.path.join(imageFolder, 'prerun')
    if not os.path.exists(imageFolder_pre_run): os.mkdir(imageFolder_pre_run)

    imageFolder_output = os.path.join(imageFolder, 'output')
    if not os.path.exists(imageFolder_output): os.mkdir(imageFolder_output)


    path_modues = r'.\modules'      # os.path.join(mainOutputFolder,'modules')
    sys.path.append(path_modues)    

    # NOTE: IF YOU USE MODULES, DONT FORGET EMPTY "__init__.py" FILE INSIDE MODULES FOLDER
    # NOTE: path_modules_init = os.path.join(path_modues, "__init__.py")
    # NOTE: if not os.path.exists(path_modules_init):  with open(path_modules_init, "w") as init_file: init_file.write("")

    #--------------------------- IMPORT CUSTOM FUNCITONS -------------------------------------
    from cropGUI import cropGUI

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


    cropMaskName = "-".join(mainOutputSubFolders[:2])+'-crop'
    cropMaskPath = os.path.join(os.path.join(*mainOutputFolder.split(os.sep)[:-1]), f"{cropMaskName}.png")
    cropMaskMissing = True if not os.path.exists(cropMaskPath) else False

    graphsPath          =   os.path.join(dataArchiveFolder  ,  "graphs.pickle"    )
    segmentsPath        =   os.path.join(dataArchiveFolder  ,  "segments.pickle"  )
    contoursHulls       =   os.path.join(dataArchiveFolder  ,  "contorus.pickle"  )
    mergeSplitEvents    =   os.path.join(dataArchiveFolder  ,  "ms-events.pickle" )

                        
    meanImagePath       =   os.path.join(dataArchiveFolder  ,  "mean.npz"         )
    meanImagePathArr    =   os.path.join(dataArchiveFolder  ,  "meanArr.npz"      )
                        
    archivePath         =   os.path.join(stagesFolder       ,  "croppedImageArr.npz"        )
    binarizedArrPath    =   os.path.join(stagesFolder       ,  "binarizedImageArr.npz"      )
    post_binary_data    =   os.path.join(stagesFolder       ,  "intermediate_data.pickle"   ) 

    print(track_time(reset = True))

    
    """
    ===================================================================================================
    ======== CROP USING A MASK (DRAW RED RECTANGLE ON EXPORTED SAMPLE IN MANUAL MASK FOLDER) ==========
    IF MASK IS MISSING YOU CAN DRAW IT USING GUI
    """
    if cropMaskMissing: # search for a mask at cropMaskPath (project -> setup -> parameter)
        print(f"\nNo crop mask in {mainOutputFolder} folder!, creating mask : {cropMaskName}.png")
        mapXY           = (np.load('./mapx.npy'), np.load('./mapy.npy'))
        cv2.imwrite(cropMaskPath, convertGray2RGB(undistort(cv2.imread(imageLinks[0],0), mapXY)))

        p1,p2           = cropGUI(cropMaskPath)
        cropMask        = cv2.imread(cropMaskPath,1)

        cv2.rectangle(cropMask, p1, p2,[0,0,255],-1)
        cv2.imwrite(  cropMaskPath,cropMask)
    else:
        cropMask = cv2.imread(cropMaskPath,1)
    # ---------------------------- ISOLATE RED RECTANGLE BASED ON ITS HUE ------------------------------
    cropMask = cv2.cvtColor(cropMask, cv2.COLOR_BGR2HSV)

    lower_red = np.array([(0,50,50), (170,50,50)])
    upper_red = np.array([(10,255,255), (180,255,255)])

    manualMask = cv2.inRange(cropMask, lower_red[0], upper_red[0])
    manualMask += cv2.inRange(cropMask, lower_red[1], upper_red[1])

    # --------------------- EXTRACT MASK CONTOUR-> BOUNDING RECTANGLE (USED FOR CROP) ------------------
    contours = cv2.findContours(manualMask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    [X, Y, W, H] = cv2.boundingRect(contours[0])
    if 1 == -1:
        print(track_time())
        from multiprocessing import Pool
        import os
        mx,my = np.load('./mapx.npy'), np.load('./mapy.npy')
        print(track_time())
        if not os.path.exists(archivePath):
            print(track_time())
            og_shape = cv2.imread(cropMaskPath,0).shape
            dataArchive = np.zeros((len(imageLinks), *og_shape), dtype=np.uint8)
            for idx, imageLink in enumerate(imageLinks):
                #if idx % 200 == 0: print(f'({idx} 200 done: {track_time()}')
                dataArchive[idx, :, :] = cv2.imread(imageLink, 0)
            print(f'(1 core import done:) {track_time()}')
            np.savez(archivePath,dataArchive)
            print(f'(1 core save done:) {track_time()}')
        else:
            dataArchive = np.load(archivePath)['arr_0']
            print(f'(1 core load done:) {track_time()}')

        #dataArchive0 = None

        #dataArchive2 = np.zeros_like(dataArchive)
        for idx in range(dataArchive.shape[0]):
            #if (idx + 1) % 200 == 0: print(f'({idx} 200 done: {track_time()}')
            dataArchive[idx, :, :] = cv2.remap(dataArchive[idx],  mx, my, cv2.INTER_LINEAR)
        print(f'(1 core remap done:) {track_time()}')
        # pr = 4

        # with Pool(processes=pr) as pool:
        #     dataArchive2 = pool.starmap(cv2.remap, ((dataArchive[i], mx, my, cv2.INTER_LINEAR) for i in range(dataArchive.shape[0])))
        #cv2.imshow('a', dataArchive[10])

        dataArchive = np.rot90(dataArchive, 2, (1,2))
        print(f'(1 core rotate done:) {track_time()}')

        dataArchive = dataArchive[:,Y:Y+H, X:X+W]
        print(f'(1 core crop done:) {track_time()}')
        
        np.savez(archivePath, dataArchive)
        print(f'(1 core save done:) {track_time()}')
    else:
         dataArchive = np.load(archivePath)['arr_0'] 
         print(f'(1 core load done:) {track_time()}')

    dataArchive = torch.from_numpy(dataArchive)

    from torch.multiprocessing import Process, Manager, set_start_method
    set_start_method('spawn', force=True)
    manager = Manager()
    result_queue = manager.Queue()

    processes = []

    data_size = dataArchive.shape[0]
    # Determine the size of each slice
    slice_size = 500

    # Create a Manager and a shared Queue
    manager = Manager()
    result_queue = manager.Queue()
    
    # Create processes
    num_processors = 4
    processes = []
    slices          = gen_slices(data_size, slice_size)
    slices_batches  = redistribute_vals_bins(slices, num_processors)   
    print(torch.multiprocessing.get_sharing_strategy())
    print(torch.multiprocessing.get_all_sharing_strategies())
    dataArchive.share_memory_()
    for i in range(num_processors):
        print(f'({timeHMS()})[{os.getpid()}] spawning worker: {i}')
        p = Process(target=compute_mean_image_batch, args=(slices_batches[i], dataArchive, result_queue))
        p.start()
        processes.append(p)
    t0_CPU_MP = time.time()
    print(f'({timeHMS()})[{os.getpid()}] waiting processes to end..')    
    for p in processes:
        p.join()
    print(f'({timeHMS()})[{os.getpid()}] joining results... Done!')    
    # Collect results from the queue
    total_elements = 0
    running_sum = 0

    while not result_queue.empty():
        num_elements, mean_image = result_queue.get()
        total_elements += num_elements
        running_sum += num_elements * mean_image

    # Compute the final mean image
    final_mean_image = running_sum / total_elements
    print(f'CPU_MP time =({time.time() - t0_CPU_MP:.2f} s')
    #chunk_size = len(dataArchive) // 2
    #num_elements, mean_image = parallel_compute_mean_image([dataArchive[i:i+chunk_size] for i in range(0, len(dataArchive), chunk_size)])
    cv2.imshow('mean_parallel', mean_image.to(torch.uint8).numpy())
    t0_GPU          = time.time()
    print(f'calculating batch time mean start: {track_time()}')

    your_dataset = YourDataset(dataArchive)
    dataloader = DataLoader(your_dataset, batch_size=50, shuffle=False)
    ref_type = torch.float
    kernel_size = 5
    kernel_mean = kernel_circular(kernel_size, normalize = True).to(device, dtype = ref_type)

    blurred_image_t = calculate_weighted_mean_image(dataloader, kernel_mean, kernel_size, ref_type = ref_type)
    torch.cuda.empty_cache()

    print(f'calculating batch time mean done: {track_time()}')

            
    print(f'binary processing start {track_time()}')
    processed_data_cpu = []
    proc = binary_pipe(dataloader, blurred_image_t, 10)
    result = torch.cat(processed_data_cpu, dim=0)
    del result
    print(f'binary processing end: {track_time()}')

    print(f'GPU time =({time.time() - t0_GPU:.2f} s')

    t0_CPU = time.time()
    meanImage = np.mean(dataArchive, axis=0)
    print(f'time-mean done: {track_time()}')
    blurMean = cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT).astype(np.uint8)  
    print(f'blur mean done: {track_time()}')

    dataArchive -= blurMean    
    dataArchive = dataArchive.astype(np.uint8)        
    print(f'subract mean done {track_time()}')
    thresh0 = 10
    dataArchive[dataArchive < thresh0] = 0
    dataArchive[dataArchive >= thresh0] = 255
    #dataArchive_b = np.where(dataArchive < thresh0, 0, 255).astype(np.uint8)  
    print(f'thresholding done: {track_time()}')
    dataArchive = np.uint8([cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) for img in dataArchive])    # delete white objects
    print(f'opening done: {track_time()}')
    dataArchive = np.uint8([cv2.dilate(img, np.ones((8,8), np.uint8) ) for img in dataArchive])    
    print(f'dilate done: {track_time()}')
    dataArchive = np.uint8([cv2.erode(img, np.ones((5,5), np.uint8) ) for img in dataArchive]) 
    print(f'erode done: {track_time()}')

    print(f'CPU time =({time.time() - t0_CPU:.2f} s')
    #cv2.imshow('b', dataArchive[10])
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

    
 
