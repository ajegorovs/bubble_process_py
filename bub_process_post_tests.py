import numpy as np, itertools, networkx as nx, sys, copy,  cv2, os, glob, re, pickle, time as time_lib
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def track_time(reset = False):
    if reset:
        track_time.last_time = time_lib.time()
        return '(Initializing time counter)'
    else:
        current_time = time_lib.time()
        time_passed = current_time - track_time.last_time
        track_time.last_time = current_time
        return f'({time_passed:.2f} s)'

#from multiprocessing import Pool, log_to_stderr, get_logger

#log_to_stderr()
#logger = get_logger()
#logger.setLevel('INFO')

#import multiprocessing, datetime, random, timeit, time as time_lib, enum
#from PIL import Image
#from tracemalloc import start

# import from custom sub-folders are defined bit lower
#from imageFunctionsP2 import (overlappingRotatedRectangles,graphUniqueComponents)

# ------------------------------------ CREATE MAIN PROJECT, SUBPROJECT FOLDERS ---------------------------//
mainOutputFolder            = r'.\post_tests'                           # descritive project name e.g [gallium_bubbles, water_bubbles]
mainOutputSubFolders =  ['Field OFF Series 7', 'sccm150-meanFix']       # sub-project folder hierarhy e.g [exp setup, parameter] 
"""                                                                       # one more layer will be created for image subset later.
========================================================================================================
============== SET NUMBER OF IMAGES ANALYZED =================
========================================================================================================
"""
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\150 sccm' #
inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\Field OFF\Series 7\150 sccm' #
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\VFS 125 mT\Series 5\150 sccm' #
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 125 mT\Series 1\350 sccm'
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\100 sccm'
# image data subsets are controlled by specifying image index, which is part of an image. e.g image1, image2, image20, image3
intervalStart   = 2000                            # start with this ID
numImages       = 2000                         # how many images you want to analyze.
intervalStop    = intervalStart + numImages     # images IDs \elem [intervalStart, intervalStop); start-end will be updated depending on available data.

useMeanWindow   = 0                             # averaging intervals will overlap half widths, read more below
N               = 700                           # averaging window width
rotateImageBy   = cv2.ROTATE_180                # -1= no rotation, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180 


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7; thickness = 4;
if 1 == 1:                               # prep image links, get min/max image indexes
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

# ============================ MANAGE MODULES WITH FUNCTIONS =============================//

# --- IF MODULES ARE NOT IN ROOT FOLDER. MODULE PATH HAS TO BE ADDED TO SYSTEM PATHS EACH RUN ---
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

from graphs_general import (graph_extract_paths, find_paths_from_to_multi, graph_check_paths,
                            comb_product_to_graph_edges, for_graph_plots,  extract_clusters_from_edges,
                            set_custom_node_parameters, G2_set_parameters, get_event_types_from_segment_graph)

from graphs_general import (G, G2, key_nodes, keys_segments, G2_t_start, G2_t_end, G2_n_from, G2_n_to, G2_edge_dist, G_time, G_area, G_centroid, G_owner, G_owner_set)

from interpolation import (interpolate_trajectory, interpolate_find_k_s, extrapolate_find_k_s, interpolateMiddle2D_2, interpolateMiddle1D_2)

from misc import (cyclicColor, timeHMS, modBR, rect2contour, combs_different_lengths, sort_len_diff_f,
                  disperse_nodes_to_times, disperse_composite_nodes_into_solo_nodes, find_key_by_value, CircularBuffer, CircularBufferReverse, 
                  split_into_bins, lr_reindex_masters, dfs_pred, dfs_succ, old_conn_2_new, lr_evel_perm_interp_data, lr_weighted_sols, 
                  save_connections_two_ways, save_connections_merges, save_connections_splits, itertools_product_length, conflicts_stage_1, 
                  conflicts_stage_2, conflicts_stage_3, edge_crit_func, two_crit_many_branches)
"""
========================================================================================================
============== Import image files and process them, store in archive or import archive =================
========================================================================================================
NOTE: mapXY IS USED TO FIX FISH-EYE DISTORTION. 
"""
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
if not os.path.exists(archivePath):
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
    """
    ===================================================================================================
    =========================== CROP/UNSISTORT AND STORE DATA INTO ARCHIVES ===========================
    ---------------------------------------------------------------------------------------------------
    """
    print(f"\n{timeHMS()}: Processing and saving archive data on drive... {track_time()}\n")
    
    if rotateImageBy % 2 == 0 and rotateImageBy != -1: W,H = H,W            # for cv2.XXX rotation commands

    dataArchive = np.zeros((len(imageLinks),H,W),np.uint8)                  # predefine storage

    mapXY       = (np.load('./mapx.npy'), np.load('./mapy.npy'))            # fish-eye correction map

    for i,imageLink in tqdm(enumerate(imageLinks), total=len(imageLinks)):
        image = undistort(cv2.imread(imageLink,0), mapXY)[Y:Y+H, X:X+W]
        if rotateImageBy != -1:
            dataArchive[i]    = cv2.rotate(image, rotateImageBy)
        else:
            dataArchive[i]    = image
    print(f"\n\n{timeHMS()}: Imporing, undistorting, cropping and rotating images ... Done! {track_time()}")
    
    np.savez(archivePath,dataArchive)
    print(f"{timeHMS()}: Saving uncompressed image stack ... Done! {track_time()}")

else:
    dataArchive = np.load(archivePath)['arr_0']
    print(f"{timeHMS()}:  Uncompressed image stack is found. Loading ... Done! {track_time()}")

if not os.path.exists(meanImagePath):
    meanImage = np.mean(dataArchive, axis=0)
    print(f"\n{timeHMS()}: Mean image is not found. Calculating mean image of a stack ... Done! {track_time()}")

    np.savez(meanImagePath,meanImage)
    print(f"\n{timeHMS()}: Mean image is not found. Saving uncompressed mean image ... Done! {track_time()}") 

else:
    meanImage = np.load(meanImagePath)['arr_0']
    print(f"\n{timeHMS()}:  Mean image found. Loading ... Done! {track_time()}")
"""
=========================================================================================================
discrete update moving average with window N, with intervcal overlap of N/2
[-interval1-]         for first segment: interval [0,N]. switch to next window at i = 3/4*N,
          |           which is middle of overlap. 
      [-interval2-]   for second segment: inteval is [i-1/4*N, i+3/4*N]
                |     third switch 1/4*N +2*[i-1/4*N, i+3/4*N] and so on. N/2 between switches
"""
if useMeanWindow:

    print(f"\n{timeHMS()}: Mean window is enabled. working on it... {track_time()}")                 # index all images
    meanIndicies = np.arange(0,dataArchive.shape[0],1)                                              # define timesteps at which averaging
    meanWindows = {}                                                                                # window is switched. eg at 0 use
    meanWindows[0] = [0,N]                                                                          # np.mean(archive[0:N])
                                                                                                    # next switch points, by geom construct
    meanSwitchPoints = np.array(1/4*N + 1/2*N*np.arange(1, int(len(meanIndicies)/(N/2)), 1), int)   # 
                                                                                                    # intervals at switch points
    for t in meanSwitchPoints:                                                                      # intervals have an overlap of N/2
        meanWindows[t] = [t-int(1/4*N),min(t+int(3/4*N),max(meanIndicies))]                         # modify last to include to the end
    meanWindows[meanSwitchPoints[-1]] = [meanWindows[meanSwitchPoints[-1]][0],max(meanIndicies)]    # index switch points {i1:0, i2:1, ...}
                                                                                                    # so i1 is zeroth interval
    if not os.path.exists(meanImagePathArr):

        masksArr = np.array([np.mean(dataArchive[start:stop], axis=0) for start,stop in meanWindows.values()])   # array of discrete averages
        print(f"\n{timeHMS()}: Mean window is enabled. No array found. Generating new... Done! {track_time()}")

        with open(meanImagePathArr, 'wb') as handle:
            pickle.dump(masksArr, handle)
        print(f"\n{timeHMS()}: Mean window is enabled. No array found. Saving new... Done! {track_time()}")

    else:
        with open(meanImagePathArr, 'rb') as handle:
                masksArr = pickle.load(handle)
        print(f"\n{timeHMS()}: Mean window is enabled. Array found. Importing data... Done! {track_time()}")

    intervalIndecies = {t:i for i,t in enumerate(meanWindows)}    

    #print(meanWindows)                                                                                       
    #print(intervalIndecies)

def whichMaskInterval(t,order):                                                                          # as frames go 0,1,..numImgs
    times = np.array(list(order))                                                                        # mean should be taken form the left
    sol = 0                                                                                              # img0:[0,N],img200:[i-a,i+b],...
    for time in times:                                                                                   # so img 199 should use img0 interval
        if time <= t:sol = time                                                                          # EZ sol just to interate and comare 
        else: break                                                                                      # and keep last one that satisfies
                                                                                                         # 
    return order[sol]      

# sort function for composite node of type (time, ID1, ID2,...): [(1,2),(2,1),(1,1,9)] -> [(1,1,9),(1,2),(2,1)]
sort_comp_n_fn = lambda x: (x[0], x[1:]) 
if not os.path.exists(binarizedArrPath):           # this will pre-calculate some stuff in case of analysis needed.
    print(f"\n{timeHMS()}: Doing initial image processing to form binary masks:")
    
    def adjustBrightness(image):
        brightness = np.sum(image) / (255 * np.prod(image.shape))
        minimum_brightness = 0.66
        ratio = brightness / minimum_brightness
        if ratio >= 1:
            return 1
        return ratio    #cv2.convertScaleAbs(image, alpha = 1 / ratio, beta = 0)
        
    if useMeanWindow: 
        binarizedMaskArr = np.zeros_like(dataArchive)
        for k in range(dataArchive.shape[0]):
            blurMean = cv2.blur(masksArr[whichMaskInterval(k,intervalIndecies)], (5,5),cv2.BORDER_REFLECT)
            binarizedMaskArr[k] = dataArchive[k] - blurMean   
        print(f"\n{timeHMS()}: Subtracting mean window image from stack ... Done! {track_time()}")

    elif not os.path.exists(binarizedArrPath):                # dont want mean window and archive does not exist
        
        blurMean = cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT)
        binarizedMaskArr = dataArchive - blurMean             # substract mean image from stack -> float
        print(f"\n{timeHMS()}: Subtracting mean image from stack ... Done! {track_time()}")

       
    #ratio = 1 #adjustBrightness(np.uint8(blurMean))
    #binarizedMaskArr = np.array([cv2.convertScaleAbs(img, alpha = ratio, beta = 0) for img in binarizedMaskArr])
    imgH,imgW = blurMean.shape
    thresh0 = 10
    binarizedMaskArr = np.where(binarizedMaskArr < thresh0, 0, 255).astype(np.uint8)                        # binarize stack
    print(f"\n{timeHMS()}: Performing binarization ... Done! {track_time()}")
    binarizedMaskArr = np.uint8([cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) for img in binarizedMaskArr])    # delete white objects
    binarizedMaskArr = np.uint8([cv2.dilate(img, np.ones((8,8), np.uint8) ) for img in binarizedMaskArr])    
    binarizedMaskArr = np.uint8([cv2.erode(img, np.ones((5,5), np.uint8) ) for img in binarizedMaskArr]) 
    print(f"\n{timeHMS()}: Performing morpological operations ... Done! {track_time()}")
    #cv2.imshow('6',np.uint8(binarizedMaskArr[20]))

    # remove edge components that touch virtual border of thickness "border_thickness"
    # draw internal border on all images
    print(f"\n{timeHMS()}: Prep required for border components removel {track_time()}")
    border_thickness = 5
    binarizedMaskArr[:,     :border_thickness   , :                 ]   = 255
    binarizedMaskArr[:,     -border_thickness:  , :                 ]   = 255
    binarizedMaskArr[:,     :                   , :border_thickness ]   = 255
    binarizedMaskArr[:,     :                   , -border_thickness:]   = 255

    print(f"\n{timeHMS()}: Starting small and border component removal... {track_time()}\n")
    #print(f"{timeHMS()}: Removing small and edge contours...")
    topFilter, bottomFilter, leftFilter, rightFilter, minArea           = 80, 40, 100, 100, 180
    for i in tqdm(range(binarizedMaskArr.shape[0]), total=binarizedMaskArr.shape[0]):
            
        # remove edge components by flooding them with black. flooding starts from virtual corner.
        cv2.floodFill(binarizedMaskArr[i], None, (0,0), 0)
        contours            = cv2.findContours(binarizedMaskArr[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
        areas               = np.array([int(cv2.contourArea(contour)) for contour in contours])
        boundingRectangles  = np.array([cv2.boundingRect(contour) for contour in contours])
        whereSmall          = np.argwhere(areas < minArea)
            
        # boundingRectangles is an array of size (N,len([x,y,w,h])) where  x,y are b-box corner coordinates, w = width h = height
        topCoords       = boundingRectangles[:,1] + boundingRectangles[:,3]      # bottom    b-box coords y + h
        bottomCoords    = boundingRectangles[:,1]                                # top       b-box coords y
        leftCoords      = boundingRectangles[:,0] + boundingRectangles[:,2]      # right     b-box coords x + w
        rightCoords     = boundingRectangles[:,0]                                # left      b-box coords x

        whereTop    = np.argwhere(topCoords     < topFilter)                    # bottom of b-box is within top band
        whereBottom = np.argwhere(bottomCoords  > (imgH - topFilter))           # top of b-box is within bottom band
        whereLeft   = np.argwhere(leftCoords    < leftFilter)                   # -"-"-
        whereRight  = np.argwhere(rightCoords   > (imgW - rightFilter))         # -"-"-
                                                                             
        whereFailed = np.concatenate((whereSmall, whereTop, whereBottom, whereLeft, whereRight)).flatten()
        whereFailed = np.unique(whereFailed)

        # draw over black (cover) border elements
        [cv2.drawContours(  binarizedMaskArr[i],   contours, j, 0, -1) for j in whereFailed]
                
    print(f"\n\n{timeHMS()}: Removing small and edge contours... Done! {track_time()}")
    # binary masks can be compressed very well. for example OG file of 1.8GB got compressed to 3.5 MB. it just takes more time, but bearable
    np.savez_compressed(binarizedArrPath,  binarizedMaskArr)
    print(f"{timeHMS()}: Saving compressed binarized image archive ... Done! {track_time()}")
else:
    binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
    print(f"\n{timeHMS()}: Loading compressed binarized image archive ... Done! {track_time()}")

if not os.path.exists(post_binary_data):
    print(f"\n{timeHMS()}: First Pass: obtaining rough clusters using bounding rectangles ... {track_time()}\n")
    num_time_steps          = binarizedMaskArr.shape[0]
    t_range                 = range(num_time_steps)
    pre_rect_cluster        = {t:{} for t in t_range}
    g0_contours             = {t:[] for t in t_range}
    #g0_contours_children = {t:{} for t in t_range}
    g0_contours_hulls       = {t:{} for t in t_range}
    g0_contours_centroids   = {t:{} for t in t_range} 
    g0_contours_areas       = {t:{} for t in t_range}
    pre_nodes_all           = []
    # stage 1: extract contours on each binary image. delete small and internal contours.
    # check for proximity between contours on frame using overlap of bounding rectangles
    # small objects are less likely to overlap, boost their bounding rectangle to 100x100 pix size
    # overlapping objects are clusters with ID (time_step,) + tuple(subIDs). contour params are stored

    for time in tqdm(range(binarizedMaskArr.shape[0]), total = binarizedMaskArr.shape[0]):
        # extract contours from current frame and hierarchy, which allows to extract only external contours (parents), not holes
        contours, hierarchy = cv2.findContours(binarizedMaskArr[time], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
        g0_contours[time]   = contours

        if hierarchy is None:
            whereParentCs = []
        else:
            whereParentCs   = np.argwhere(hierarchy[:,:,3]==-1)[:,1] #[:,:,3] = -1 <=> (no owner)
         
        # can analyze children, but i have no use of it now.    
        #whereChildrenCs = {parent:np.argwhere(hierarchy[:,:,3]==parent)[:,1].tolist() for parent in whereParentCs}
        #whereChildrenCs = {parent: children for parent,children in whereChildrenCs.items() if len(children) > 0}
        #childrenContours = list(sum(whereChildrenCs.values(),[]))
        #areasChildren   = {k:int(cv2.contourArea(contours[k])) for k in childrenContours}
        #minChildrenArea = 120
        #whereSmallChild = [k for k,area in areasChildren.items() if area <= minChildrenArea]
        #whereChildrenCs = {parent: [c for c in children if c not in whereSmallChild] for parent,children in whereChildrenCs.items()}
        #whereChildrenCs = {parent: children for parent,children in whereChildrenCs.items() if len(children) > 0}
        
        a = 1
        # get bounding rectangle of each contour for current time step
        boundingRectangles  = {ID: cv2.boundingRect(contours[ID]) for ID in whereParentCs}
        # expand bounding rectangle for small contours to 100x100 pix2 box
        brectDict = {ID: modBR(brec,100) for ID, brec in boundingRectangles.items()} 
        # find pairs of overlaping rectangles, use graph to gather clusters
        frame_clusters = extract_clusters_from_edges(edges = overlappingRotatedRectangles(brectDict,brectDict), nodes = brectDict)
        # create a bounding box for clusters
        
        for IDs in frame_clusters:
            key                         = (time,) + tuple(IDs)
            pre_rect_cluster[time][key] = cv2.boundingRect(np.vstack([rect2contour(brectDict[ID]) for ID in IDs]))
            pre_nodes_all.append(key)

        g0_contours_centroids[  time]   = np.zeros( (len(contours),2)   )
        g0_contours_areas[      time]   = np.zeros( len(contours)   , int)
        g0_contours_hulls[      time]   = [None]*len(contours)

        for i, contour in enumerate(contours):
            hull                                = cv2.convexHull(contour)
            centroid, area                      = centroid_area(hull)
            g0_contours_hulls[      time][i]    = hull
            g0_contours_centroids[  time][i]    = centroid
            g0_contours_areas[      time][i]    = int(area)
        
    print(f"\n\n{timeHMS()}: First Pass: clustering objects on single frame...Done! {track_time()}")


    # stage 2: check overlap of clusters on two consequetive frames
    # inter-frame connections are stored

    pre_edge_temporal  = []
    for time in tqdm(range(0,binarizedMaskArr.shape[0] - 1,1)):
        rect_this_time = pre_rect_cluster[time]
        rect_next_time = pre_rect_cluster[time + 1]
        pre_edge_temporal.extend(   overlappingRotatedRectangles(rect_this_time, rect_next_time)    )
    print(f"\n\n{timeHMS()}: First Pass: Obtaining overlaps between clusers time-wise... Done! {track_time()}")

    # stage 3: create a graph from inter-frame connections and extract connected components
    # these are bubble trail families which contain bubbles that came into close contact
    # which can be roughly considered to be a merged/split. but its refined later.
    sort_in                     = lambda node: node[0]                  
    sort_out                    = lambda nodes: sort_comp_n_fn(nodes[0])
    # i think sorting is not necessery, either default sorting method deals with tuples of different lenghts, or data is pepared nicely. but just in case:
    # sort nodes within family (sort_in) by time variable. sort families (sort_out) by extracting first element and sorting as composite object sort_comp_n_fn
    pre_node_families = extract_clusters_from_edges(edges = pre_edge_temporal, nodes = pre_nodes_all, sort_f_in = sort_in, sort_f_out = sort_out)
    print(f"{timeHMS()}: First Pass: Extracting families of nodes ... Done! {track_time()}")

    with open(post_binary_data, 'wb') as handle:
        pickle.dump(
        [
            pre_rect_cluster, g0_contours, g0_contours_hulls, g0_contours_centroids,
            g0_contours_areas, pre_node_families
        ], handle) 
    print(f"\n{timeHMS()}: Storing intermediate data...Done {track_time()}")
else:

    with open(post_binary_data, 'rb') as handle:
        [
            pre_rect_cluster, g0_contours, g0_contours_hulls, g0_contours_centroids,
            g0_contours_areas, pre_node_families
        ] = pickle.load(handle)
    print(f"\n{timeHMS()}: Begin loading intermediate data...Done! {track_time()}")



if 1 == -1:
    print(f"\n{timeHMS()}: Pre - run images: Generating images")
    segments2 = pre_node_families
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
            x,y,w,h = pre_rect_cluster[time][subCase]
            cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
            [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                startPos2 = g0_contours[time][subID][-30][0] 
                [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        
    print(f"\n{timeHMS()}:  Pre - run images: Saving images")
    for k,img in tqdm(enumerate(imgs)):
        webp_parameters = [int(cv2.IMWRITE_WEBP_QUALITY), 20]
        filepath = os.path.join(imageFolder_pre_run, f"{str(k).zfill(4)}.webp")
        cv2.imwrite(filepath, img, webp_parameters)
        #cv2.imshow('a',img)
    print(f"\n{timeHMS()}:  Pre - run images: stage is completed.")

if 1 == -1:
    t_all_areas = []
    for areas in g0_contours_areas.values():
        t_all_areas += areas.tolist()

    g0_area_mean    = np.mean(t_all_areas)
    g0_area_std     = np.std(t_all_areas)

    plt.hist(t_all_areas, bins=70, color='skyblue', edgecolor='black')

    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.title('Histogram of Animal Heights')      # LOL
    plt.show()

max_width = len(str(len(pre_node_families))) # for output family index padding.

trajectories_all_dict       = {}
graphs_all_dict             = {}
contours_all_dict           = {}
events_split_merge_mixed    = {}

issues_all_dict = defaultdict(dict) #[2]:#[19]:#

for doX, pre_family_nodes in enumerate(pre_node_families):
    if len(pre_family_nodes) <= 1: continue #one node, go to next doX
    doX_s = f"{doX:0{max_width}}"        # pad index with zeros so 1 is aligned with 1000: 1 -> 0001
    print('\n\n=========================================================================================================\n')
    print(f'{timeHMS()}:({doX_s}) working on family {doX} {track_time()}')
    """
    ===============================================================================================
    =============== Refine expanded BR overlap subsections with solo contours =====================
    ===============================================================================================
    WHAT: recalculate overlap between bubble bounding rectangles on to time-adjecent time frames
    WHAT: without expaning b-recs for small contours
    WHY:  expended b-rec clustering was too rough and might have falsely connected neighbor bubbles 
    HOW:  generate bounding rectangles for time step frame and determing overlap
    """
    print(f'\n{timeHMS()}:({doX_s}) generating rough contour overlaps')

    # 'pre_family_nodes' contains node which are clusters of contours
    # gather all active contour IDs for each time step
    time_subIDs = disperse_nodes_to_times(pre_family_nodes, sort = True)
    # split them into nodes containing single ID : [(time, ID1),..]
    lr_nodes_all = disperse_composite_nodes_into_solo_nodes(pre_family_nodes, sort = True, sort_fn = sort_comp_n_fn)

    lr_edge_temporal  = [] # store overlapping connections
    # check overlap between all possible pairs of bounding rectangle on two sequential frames at time t and t + 1. 
    for t in tqdm(list(time_subIDs)[:-1]):
        
        if t == 0:  oldBRs = {(t, ID):cv2.boundingRect(g0_contours[t][ID]) for ID in time_subIDs[t]}
        else:       oldBRs = newBrs # reuse previously prev time step data.

        newBrs = {(t + 1, ID):cv2.boundingRect(g0_contours[t + 1][ID]) for ID in time_subIDs[t + 1]}
        
        lr_edge_temporal.extend(overlappingRotatedRectangles(oldBRs,newBrs))

    # for_graph_plots(G)    # <<<<<<<<<<<<<<
    """
    ===============================================================================================
    =============================== DEFINE A NODE AND SEGMENT VIEW GRAPH ==========================
    ===============================================================================================
    WHAT:   create two graphs: 1) one for holding information about contours as nodes of type (time, local_ID)
    WHAT:   and edges between these nodes show contour overlap (proximity). 2) second graph contains information about
    WHAT:   how different bubble trajectories split and merge. trajectories are node chains and each has an integer
    WHAT:   identifier, which is a node on 'segment' graph. edges show that one such segment is connected to others.
    WHY:    it is best to hold all information on graphs: G = node view, holds information about each contour and their
    WHY:    connectivity. from it trajectories can be derived into G2, which is used to analyze splits/merges.
    NOTE:   many functions are defined to modify contents of graphs. to make it easy i keep working with only
    NOTE:   one instance of each graph and modify it. this way i can set it as a default target and dont have to carry around.
    NOTE:   in order to prevent cycling imports i define graphs in graphs related module, also key_nodes and keys_segments
    """
    [key_t_start, key_t_end, key_n_start, key_n_end, key_e_dist]    = key_nodes     # string keys are defined in graphs_general
    [key_time, key_area, key_centroid, key_owner, key_momz]         = keys_segments # modules. its for debugging and manual inspection.
    
    G.clear()                                                                       # clear contents of node and segment view graphs. 
    G2.clear()                                                                      # clear not redefine to keep the OG reference
    G.add_nodes_from(lr_nodes_all)                                                  # add all nodes for this family of nodes
    G.add_edges_from(lr_edge_temporal)                                              # add overlap information as edges. not all nodes are inter-connected
    
    set_custom_node_parameters(g0_contours_hulls, G.nodes(), None, calc_hull = 0)   # store node parameters, about owner = None read next stage
      
    """
    ===============================================================================================
    =========================== EXTRACT NODE CHAINS FROM NODE VIEW (G) GRAPH ======================
    ===============================================================================================
    WHY:    long G graph segmenst (which is a chain of solo contour nodes) is likely to be a 
    WHY:    single bubble raising without merges and splits. 
    HOW:    read info in graph_extract_paths(). 
    """

    seg_min_length = 2     
    segments_fb = graph_extract_paths(G, min_length = seg_min_length)

    # add node owner parameter, which shows to which chain this node belongs (index in segments_fb)
    for owner, segment in enumerate(segments_fb):
        G2_set_parameters(segment, owner)
        for node in segment:
            G_owner_set(node, owner)

    #for_graph_plots(G, segments_fb, show = True)
    """
    ===============================================================================================
    ========================= FIND OBJECTS/BUBBLES THAT ARE FROZEN IN PLACE =======================
    ===============================================================================================
    WHAT:   find which chains dont move (much)
    WHY:    these may be semi-static arftifacts or very small bubbles which stick to walls
    WHY:    they dont contribute anything meaningfull to analysis only add complexity
    HOW:    1) check displacement for all chains 
    HOW:    2) check if frozen chains are continuation of same artifacts
    HOW:    3) isolate faulty nodes from analysis (from graph)
    """
    print(f'\n{timeHMS()}:({doX_s}) Detecting frozen bubbles ... ')
    # 1)    find displacement norms if mean displ is small
    # 1)    remember average centroid -> frozen object 'lives' around this point
    # 1)    remember mean area and area stdev
    fb_radiuss = 5
    fb_mean_centroid_d = {}
    fb_area_mean_stdev_d = {}

    for t, nodes in enumerate(segments_fb):

        traj      = np.array([G_centroid(n) for n in nodes])
        displ_all = np.linalg.norm(np.diff(traj, axis = 0), axis = 1)

        if np.mean(displ_all) <= fb_radiuss:
            fb_mean_centroid_d[t]   = np.mean(  traj, axis = 0)
            areas                   = [G_area(n) for n in nodes]
            fb_area_mean_stdev_d[t] = ( np.mean(areas), np.std(areas)   )

    # 2)    generate all possible pairs of frozen segments
    # 2)    run though each pair and check if their mean centroids are close
    # 2)    pairs that are close are gathered on a graph and clusters extracted
    fb_edges_test = list(itertools.combinations(fb_mean_centroid_d, 2))
    fb_edges_close = []
    for a,b in fb_edges_test:
        dist = np.linalg.norm(fb_mean_centroid_d[a] - fb_mean_centroid_d[b])
        if dist <= fb_radiuss:
            fb_edges_close.append((a,b))
    
    fb_segment_clusters = extract_clusters_from_edges(fb_edges_close)   # [[1,2,3],[4,5],[6]]

    
    # 3a)   if there are times between frozen prior bubble segment end and next segment start 
    # 3a)   we have to find if some nodes in these time interavls (holes) are also frozen
    # 3a)   for it we need to check area stats

    fb_hole_info       = {}#{'times': {},'area_min_max': {}}

    for cluster in fb_segment_clusters:
        for edge in zip(cluster[:-1], cluster[1:]): # [1,2,3] -> [(1,2),(2,3)]. should work if ordered correctly. otherwise a weak spot.

            (fr,to) = edge

            fb_hole_info[edge] = {'times':(G2_t_end(fr) + 1 , G2_t_start(to) - 1)} #{'times':(G_time(segments_fb[f][-1]) + 1 , G_time(segments_fb[t][0]) - 1)}

            # set lower and top bounds on area. for bigger bubble + 5 * std and for smaller -5 * std
            area_mean_1, area_std_1 = fb_area_mean_stdev_d[fr]
            area_mean_2, area_std_2 = fb_area_mean_stdev_d[to]
            if area_mean_1 > area_mean_2:
                area_min_max = (area_mean_2 - 5*area_std_2, area_mean_1 + 5*area_std_1)
            else:
                area_min_max = (area_mean_1 - 5*area_std_1, area_mean_2 + 5*area_std_2)

            fb_hole_info[edge]['area_min_max'   ] = area_min_max
            
    # 3b)   extract stray nodes that have times between connected frozen segments
    # 3b)   check if stray node's have similar centroid as avergae between two segments
    # 3b)   check if stray node's area is within a threshold
    fb_remove_nodes_inter = []
    for edge in fb_hole_info:
        t_min,t_max         = fb_hole_info[edge]['times'         ]
        area_min, area_max  = fb_hole_info[edge]['area_min_max'  ]
        nodes_active        = [n for n in G.nodes if t_min <= G_time(n) <= t_max and G_owner(n) in (None, -1)]
        (fr, to)            = edge
        centroid_target     = 0.5*(fb_mean_centroid_d[fr] + fb_mean_centroid_d[to])

        for node in nodes_active:
            dist = np.linalg.norm(centroid_target - G_centroid(node))
            if dist <= fb_radiuss and (area_min <= G_area(node) <= area_max):
                fb_remove_nodes_inter.append(node)

    # 3c)   remove frozen stray nodes form a graph
    # 3c)   remove frozen segment's nodes from a graph
    remove_nodes = []
    G.remove_nodes_from(fb_remove_nodes_inter)
    for IDs in fb_segment_clusters:
        for ID in IDs:
            nodes = segments_fb[ID]
            G.remove_nodes_from(nodes)
            remove_nodes.extend(nodes)
    """
    ===============================================================================================
    =================== REDEFINE SEGMENTS AFTER FROZEN BUBBLES ARE REMOVED ========================
    ===============================================================================================
    WHY:  frozen nodes were stripped from graph, have to recalculate segments (chains)
    """
    print(f'\n{timeHMS()}:({doX_s}) Detecting frozen bubbles ... DONE')
    segments2 = graph_extract_paths(G, min_length = seg_min_length)

    if len(segments2) == 0: continue    # no segments, go to next doX
    
    # store information on segment view graph and update node ownership
    for owner,segment in enumerate(segments2):
        G2_set_parameters(segment, owner)
        for node in segment:
            G_owner_set(node, owner)
            #G.nodes[node]["owner"] = owner
  
    #if doX >= 0:
    #    for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Determining connectivity between segments... ')
    """
    ===============================================================================================
    ========================== FIND CONNECTIVITY BETWEEN SEGMENTS =================================
    ===============================================================================================
    WHAT: find if segments are connected pairwise: one segment ends -> other begins
    WHY:  its either one bubble which has optically disrupted trajectory or merge/split of multiple bubbles 
    HOW:  read 'graphs_general:graph_check_paths()' comments.
    HOW:  in short: 1) find all start-end pairs that exist with time interval of length 'lr_maxDT'
    HOW:  check if there is a path between two segments. first by custom funciton, then nx.has_path()
    """
    lr_maxDT = 60   # search connections with this many time steps.

    t_has_holes_report = {}
    G2 = graph_check_paths(lr_maxDT, t_has_holes_report)

    print(f'\n{timeHMS()}:({doX_s}) Paths that have failed hole test: {t_has_holes_report}')
     
    """
    ===============================================================================================
    =============================== RESOLVE ZERO-PATH CONNECTIONS =================================
    ===============================================================================================
    WHAT: find segment connections that dont have stray nodes in-between
    WHY:  very much likely that its the same trajectory. interruption is very brief. 
    WHY:  connection has a stray node/s and edges, which have confused chain extraction method.
    HOW:  best i can do is to 'absorb' stray nodes into solo contour nodes by forming a composite node.
    """
    G2, lr_ms_edges_main = get_event_types_from_segment_graph(G2, solo_only = True)

    t_conn_121_zp = lr_ms_edges_main['solo_zp']

    lr_zp_redirect = {i: i for i,v in enumerate(segments2) if len(v) > 0}

    # analyze zero path cases:
    for fr, to in t_conn_121_zp:
        fr = lr_zp_redirect[fr]         # in case there is a sequence of ZP events one should account for inheritance of IDs
        time_buffer = 5
        # take a certain buffer zone about ZP event. its width should not exceed sizes of segments on both sides
        # which is closer to event from left  : start of  left segment    or end      - buffer size?
        time_from   = max(G2_t_start(fr), G2_t_end(fr) - time_buffer    )   # test: say, time_buffer = 1e20 -> time_from = G2_t_start(fr)   # (1)
        # which is closer to event from right : end   of  right segment   or start    + buffer size?
        time_to     = min(G2_t_end(to)  , G2_t_start(to) + time_buffer  )
        # NOTE: at least one node should be left in order to reconnect recovered segment back. looks like current approach does it. by (1) and (2)
        nodes_keep              = []
        nodes_stray             = [node for node in G.nodes() if time_from < G_time(node) < time_to and G_owner(node) is None] 
        nodes_extra_segments    = [n for n in segments2[fr] if G_time(n) > time_from] + [n for n in segments2[to] if G_time(n) < time_to]   # (2)

        nodes_keep.extend(nodes_stray)
        nodes_keep.extend(nodes_extra_segments)   
        
        clusters    = nx.connected_components(G.subgraph(nodes_keep).to_undirected())
        sol         = next((cluster for cluster in clusters if nodes_extra_segments[0] in cluster), None)
        assert sol is not None, 'cannot find connected components'

        nodes_composite = [(t,) + tuple(IDs) for t, IDs in disperse_nodes_to_times(sol, sort = True).items()] 

        segments2[fr]   =       [n for n in segments2[fr] if G_time(n) <= time_from]                                                           
        segments2[fr].extend(   nodes_composite)                                                     
        segments2[fr].extend(   [n for n in segments2[to] if G_time(n) >= time_to]    )
        segments2[to]   = []
         
        G.remove_nodes_from(nodes_stray)           # stray nodes are absorbed into composite nodes. they have to go from G
        
        G.remove_nodes_from(nodes_extra_segments)  # old parts of segments have to go. they will be replaced by composite nodes.

        G.add_nodes_from(nodes_composite)
        set_custom_node_parameters( g0_contours, segments2[fr], fr, calc_hull = 1)

        # have to reconnect reconstructed interval on G. get nodes on interval edges
        ref_left    = next((n for n in segments2[fr] if G_time(n) == time_from))
        ref_right   = next((n for n in segments2[fr] if G_time(n) == time_to))
        # generate edges by staggered zip method.
        nodes_temp = [ref_left] + nodes_composite + [ref_right]
        edges = [(a, b) for (a, b) in zip(nodes_temp[:-1], nodes_temp[1:])]

        G.add_edges_from(edges)
        
        edges_next = [(fr, i, key_e_dist) for i in successors]
       
        G2_set_parameters(segments2[fr], fr, edges = edges_next)
            
        G2.remove_node(to)
            
        lr_zp_redirect[to] = fr
        print(f'zero path: joined segments: {conn}')

        a = 1

    print(f'\n{timeHMS()}:({doX_s}) Working on one-to-one (121) segment connections ... ')
    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    ===============================================================================================
    === PROCESS SEGMENT-SEGMENT CONNECTIONS THAT ARE CONNECTED ONLY TOGETHER (ONE-TO-ONE; 121) ====
    ===============================================================================================
    WHY:    Sometimes, due to optical lighting artifacts or other artifacts, next to bubble pops out
    WHY:    an unwanted object, or its a same bubble that visually splits and recombines. 
    WHY:    That means that solo contour trajectory is interrupted by a 'hole'  in graph, which is actually
    WHY:    an 'explositon' of connections into 'stray' nodes and 'collapse' back to a solo trajectory.
    WHY:    These holes can be easily patched, but first they have to be identified
    WHY:    if this happens often, then instead of analyzing such 'holes' individually, we can analyze 
    WHY:    whole trajectory with all holes and patch them more effectively using longer bubble history.

    FLOWCHART OF PROCESSING 121 (one-to-one) SEGMENTS:
    
    (121.A) isolate 121 connections
    (121.B) extract inter-segment nodes as time slices : {t1:[*subIDs],...}
    DEL-(121.C) detect which cases are zero-path ZP. its 121 that splits into segment and a node. byproduct of a method.
    DEL-(121.D) resolve case with ZP
    (121.E) find which 121 form a long chain- its an interrupted solo bubble's trejctory.
              missing data in 'holes' is easier to interpolate from larger chains 
    DEL-(121.F) chain edges may be terminated by artifacts - 'fake' events, find them and refine chain list elements.
    (121.G) hole interpolation
    (121.H) prep data (area, centroids) for all subID combinations
    (121.I) generate permutations from subIDs
    (121.K) generate all possible evolutions of subIDs perms though time. 
    (121.J) find best fit evolution by minimizing area and isplacement criterium
    (121.L) save data into graps and storage.
    """
    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    if len(t_conn_121_zp) > 0:

        t_has_holes_report = {}
        G2 = graph_check_paths(lr_maxDT, t_has_holes_report)

        print(f'\n{timeHMS()}:({doX_s}) Paths that have failed hole test: {t_has_holes_report}')

        G2, lr_ms_edges_main = get_event_types_from_segment_graph(G2, solo_only = True)

    t_conn_121 = lr_ms_edges_main['solo']# for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    ===================================================================================================
    ==================== SLICE INTER-SEGMENT (121) NOTE-SUBID SPACE W.R.T TIME ========================
    ===================================================================================================
    WHAT: 121 segments are connected via stay (non-segment) nodes. extract them and reformat into  {TIME1:[*SUBIDS_AT_TIME1],...}
    WHY:  at each time step bubble may be any combination of SUBIDS_AT_TIMEX. most likely whole SUBIDS_AT_TIMEX. but possibly not
    """
    lr_big121s_perms_pre = {}
    for fr, to in t_conn_121:
            
        node_from, node_to = G2_n_to(fr)        , G2_n_from(to)
        time_from, time_to = G_time(node_from)  , G_time(node_to)

        # isolate stray nodes on graph at time interval between two connected segments
        
        nodes_keep    = [node for node in G.nodes() if time_from < G_time(node) < time_to and G_owner(node) is None] 
        nodes_keep.extend([node_from,node_to])
            
        # extract connected nodes using depth search. not sure why i switched from connected components. maybe i used it wrong and it was slow
        #sol     = set()
        #g_limited           = G.subgraph(nodes_keep)
        #dfs_succ(g_limited, node_from, time_lim = time_to + 1, node_set = sol)

        clusters    = nx.connected_components(G.subgraph(nodes_keep).to_undirected())
        sol         = next((cluster for cluster in clusters if node_from in cluster), None)
        assert sol is not None, 'cannot find connected components'
        
        lr_big121s_perms_pre[(fr,to)] = disperse_nodes_to_times(sol, sort = True)
        
        a = 1

    
    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
    """
    ===============================================================================================
    ============================== FIND CHAINS OF 121 CONNECTED SEGMENTS ==========================
    ===============================================================================================
    WHAT: find which segments are continiously connected via 121 connection to other segments
    WHY:  they are most likely single bubble which has an optical split.
    HOW:  use 121 edges and condence to a graph, extract connected compontents
    NOTE: same principle as in 'fb_segment_clusters'
    """
    t_big121s_edited = extract_clusters_from_edges(t_conn_121)
    print(f'\n{timeHMS()}:({doX_s}) Working on joining all continious 121s together: {t_big121s_edited}')
     
    if 1 == 1:
        """
        ===============================================================================================
        ==================== 121 CHAINS: INTERPOLATE DATA IN HOLES OF LONG CHAINS =====================
        ===============================================================================================
        WHAT: we have chains of segments that represent solo bubble. its at least 2 segments (1 hole). interpolate data
        WHY:  if there are more than 2 segments, we will have more history and interpolation will be of better quality
        HOW:  scipy interpolate 
        """
        lr_big121s_edges_relevant = []
        lr_big121s_interpolation        = defaultdict(dict)

        for t, subIDs in enumerate(t_big121s_edited):
            if subIDs != None: 
                # prepare data: resolved  time steps, centroids, areas G2
                temp_nodes_all    = sum([segments2[i]   for i   in subIDs           ],[])
                temp_times        =     [G_time(n)      for n   in temp_nodes_all   ]
                temp_areas        =     [G_area(n)      for n   in temp_nodes_all   ]
                temp_centroids    =     [G_centroid(n)  for n   in temp_nodes_all   ]
                
                # generate time steps in holes for interpolation
                conns_times_dict = {}
                for (fr, to) in zip(subIDs[:-1], subIDs[1:]):
                    conns_times_dict[(fr, to)]  = range(G2_t_end(fr) + 1, G2_t_start(to), 1)

                lr_big121s_edges_relevant.extend(conns_times_dict.keys())
                times_missing_all = []; [times_missing_all.extend(times) for times in conns_times_dict.values()]
        
                # interpolate composite (long) parameters 
                t_interpolation_centroids_0 = interpolateMiddle2D_2(temp_times,np.array(temp_centroids), times_missing_all, s = 15, debug = 0, aspect = 'equal', title = subIDs)
                t_interpolation_areas_0     = interpolateMiddle1D_2(temp_times,np.array(temp_areas),times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = subIDs)
                # form dict = {time:centroid} for convinience
                t_interpolation_centroids_1 = {t: c for t, c in zip(times_missing_all,t_interpolation_centroids_0)}
                t_interpolation_areas_1     = {t: c for t, c in zip(times_missing_all,t_interpolation_areas_0)    }
                # save data with t_conns keys
                for conn, times in conns_times_dict.items():
                    conn_new    = old_conn_2_new( conn,lr_zp_redirect )
                    centroids   = [c    for t, c    in t_interpolation_centroids_1.items()  if t in times]
                    areas       = [a    for t, a    in t_interpolation_areas_1.items()      if t in times]

                    lr_big121s_interpolation[conn_new]['centroids'] = np.array(centroids)
                    lr_big121s_interpolation[conn_new]['areas'    ] = areas
                    lr_big121s_interpolation[conn_new]['times'    ] = times
        """            
        ===================================================================================================
        ================ 121 CHAINS: CONSTRUCT PERMUTATIONS FROM CLUSTER ELEMENT CHOICES ==================
        ===================================================================================================
        WHAT: generate different permutation of subIDs for each time step.
        WHY:  Bubble may be any combination of contour subIDs at a given time. should consider all combs as solution
        HOW:  itertools combinations of varying lenghts
        """
        print(f'\n{timeHMS()}:({doX_s}) Computing contour element permutations for each time step...')
        
        # if "lr_zp_redirect" is not trivial, drop resolved zp edges
        lr_big121s_perms            = {}
        lr_big121s_perms_pre_old    = copy.deepcopy(lr_big121s_perms_pre)
        lr_big121s_perms_pre        = {}
        # lr_big121s_perms_pre = {(t_from,t_to):{time_1:*subIDs1,...}}
        for conn, times_subIDs in lr_big121s_perms_pre_old.items():

            fr_new, to_new  = old_conn_2_new(conn,lr_zp_redirect)
            edge            = (fr_new,to_new)

            if fr_new != to_new and edge in lr_big121s_edges_relevant:

                lr_big121s_perms[       edge]   = {}
                lr_big121s_perms_pre[   edge]   = times_subIDs

                for time, subIDs in times_subIDs.items():
                    lr_big121s_perms[edge][time] = combs_different_lengths(subIDs)

        lr_big121s_conn_121 = lr_big121s_edges_relevant
        """
        ===============================================================================================
        =========== 121 CHAINS: PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ===========
        ===============================================================================================
        WHY: these will be reused alot in next steps, store beforehand
        """
        print(f'\n{timeHMS()}:({doX_s}) Calculating parameters for possible contour combinations...')
        
        lr_big121s_perms_areas      = {}
        lr_big121s_perms_centroids  = {}
        lr_big121s_perms_mom_z      = {}
            
        for conn, times_perms in lr_big121s_perms.items():
            lr_big121s_perms_areas[     conn]  = {t:{} for t in times_perms}
            lr_big121s_perms_centroids[ conn]  = {t:{} for t in times_perms}
            lr_big121s_perms_mom_z[     conn]  = {t:{} for t in times_perms}
            for time,perms in times_perms.items():
                for perm in perms:
                    hull = cv2.convexHull(np.vstack([g0_contours[time][subID] for subID in perm]))
                    centroid, area, mom_z                           = centroid_area_cmomzz(hull)
                    lr_big121s_perms_areas[     conn][time][perm] = area
                    lr_big121s_perms_centroids[ conn][time][perm] = centroid
                    lr_big121s_perms_mom_z[     conn][time][perm] = mom_z

        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        """
        ===============================================================================================
        ========== 121 CHAINS: CONSTUCT UNIQUE EVOLUTIONS THROUGH CONTOUR PERMUTATION SPACE ===========
        ===============================================================================================
        WHAT: using subID permutations at missing time construct choice tree. 
        WHY:  each branch represents a bubbles contour ID evolution through unresolved intervals.
        HOW:  either though itertools product or, if number of branches is big, dont consider
        HOW:  branches where area changes more than a set threshold. 
        -----------------------------------------------------------------------------------------------
        NOTE: number of branches (evolutions) from start to finish may be very large, depending on 
        NOTE: number of time steps and contour permutation count at each step. 
        NOTE: it can be calculated as 't_branches_count' = len(choices_t1)*len(choices_t2)*...
        case 0)   if count is small, then use combinatorics product function.
        case 1a)  if its large, dont consider evoltions where bubble changes area rapidly from one step to next.
        case 1b)  limit number of branches retrieved by 't_max_paths'. we can leverage known assumption to
                  get better branches in this batch. assumption- bubbles most likely include all 
                  subcontours at each time step. Transitions from large cluster to large cluster has a priority.
                  if we create a graph with large to large cluster edges first, pathfinding method
                  will use them to generate first evolutions. so you just have to build graph in specific order.
        case 2)   optical splits have 121 chains in inter-segment space. these chains are not atomized into
                  solo nodes, but can be included into evolutions as a whole. i.e if 2 internal chains are
                  present, you can create paths which include 1st, 2nd on both branches simultaneously.
        """
        print(f'\n{timeHMS()}:({doX_s}) Generating possible bubble evolution paths from prev combinations...')

        lr_big121s_perms_cases = {}
        lr_big121s_perms_times = {}
        lr_drop_huge_perms = []
        for t_conn, t_times_perms in lr_big121s_perms.items():
            func = lambda edge : edge_crit_func(t_conn,edge, lr_big121s_perms_areas, 2)

            conn_new          = old_conn_2_new(t_conn,lr_zp_redirect)
            values            = list(t_times_perms.values())
            times               = list(t_times_perms.keys())
            branches_count    = itertools_product_length(values) # large number of paths is expected
            max_paths         = 5000
            if branches_count >= max_paths:
                # 1a) keep only tranitions that dont change area very much
                choices                       = [[(t,) + p for p in perms] for t, perms in zip(times,values)]
                edges, nodes_start, nodes_end   = comb_product_to_graph_edges(choices, func)

                if len(nodes_end) == 0: # finding paths will fail, since no target node
                    nodes_end.add(choices[-1][0])   # add last, might also want edges to that node, since they have failed
                    edges.extend(list(itertools.product(*choices[-2:])))
                
                # 1b) resort edges for pathfining graph: sort by large to large first subIDs first: ((1,2),(3,4)) -> ((1,2),(3,))
                # 1b) for same size sort by cluster size uniformity e.g ((1,2),(3,4)) -> ((1,2,3),(4,)) 
                sorted_edges    = sorted(edges, key=sort_len_diff_f, reverse=True) 
                sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, graph = None, edges = sorted_edges, only_subIDs = True, max_paths = max_paths - 1)
                # 'fail' can be either because 't_max_paths' is reached or there is no path from source to target
                if fail == 'to_many_paths':    # add trivial solution where evolution is transition between max element per time step number clusters                                           
                    seq_0 = list(itertools.product(*[[t[-1]] for t in values] ))
                    if seq_0[0] not in sequences: sequences = seq_0 + sequences # but first check. in case of max paths it should be on top anyway.
                elif fail == 'no_path': # increase rel area change threshold
                    func = lambda edge : edge_crit_func(t_conn,edge, lr_big121s_perms_areas, 5)
                    edges, nodes_start, nodes_end   = comb_product_to_graph_edges(choices, func)
                    sorted_edges                    = sorted(edges, key=sort_len_diff_f, reverse=True) 
                    sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, graph = None, edges = sorted_edges, only_subIDs = True, max_paths = max_paths - 1)

            # 0) -> t_branches_count < t_max_paths. use product
            else:
                sequences = list(itertools.product(*values))
                
            if len(sequences) == 0: # len = 0 because second pass of rel_area_thresh has failed.
                sequences   = []    # since there is no path, dont solve this conn
                times       = []
                lr_drop_huge_perms.append(t_conn)

            lr_big121s_perms_cases[conn_new] = sequences
            lr_big121s_perms_times[conn_new] = times

        lr_big121s_conn_121 = [tc for tc in lr_big121s_conn_121 if tc not in lr_drop_huge_perms]
        [lr_big121s_perms_cases.pop(tc, None) for tc in lr_drop_huge_perms]
        [lr_big121s_perms_times.pop(tc, None) for tc in lr_drop_huge_perms]
        print(f'\n{timeHMS()}:({doX_s}) Dropping huge permutations: {lr_drop_huge_perms}') 
        """
        ===============================================================================================
        ========= 121 CHAINS: FIND BEST FIT EVOLUTIONS OF CONTOURS THOUGH HOLES (TIME-WISE) ===========
        ===============================================================================================
        WHAT: calculate how area and centroids behave for each evolution
        WHY:  evolutions with least path length and area changes should be right ones pregenerated data
        HOW:  for each conn explore permutations fo each evolution. generate trajectory and areas from.
        HOW:  evaluate 4 criterions and get evolution indicies where there crit are the least:
        HOW:  1) displacement error form predicted trajetory (interpolated data)
        HOW:  2) sum of displacements for each evolution = total path. should be minimal
        HOW:  3) relative area chainges its mean and stdev values (least mean rel area change, least mean stdev)
        HOW:  4) same with moment z, which is very close to 3) except scaled. (although min result is different) 
        HOW:  currently you get 4 bins with index of evolution with minimal crit. ie least 1) is for evol #1
        HOW:  least #2 is for evol 64, least #3 is for #1 again,.. etc
        """ 
        print(f'\n{timeHMS()}:({doX_s}) Determining evolutions thats are closest to interpolated missing data...')
        # NOTE: >>> this needs refactoring, not internals, but argument data management <<<<
        temp_centroids = {tc:d['centroids'] for tc, d in lr_big121s_interpolation.items()}
        args = [lr_big121s_conn_121, lr_big121s_perms_cases,temp_centroids,lr_big121s_perms_times,
                lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

        sols = lr_evel_perm_interp_data(*args)

        # ======================= FROM MULTIPLE THRESHOLDS GET WINNING ONE ==================================
        # NOTE: dont like this approach. you should evaluate actual values/errors, rather then indicies.
        # NOTE: its easy if multiple indicies are in winning bins.
        t_weights   = [1,1.5,0,1]
        #t_sols      = [sols_c, t_sols_c_i, t_sols_a, t_sols_m]
        lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(lr_big121s_conn_121,t_weights, sols, lr_big121s_perms_cases )

        lr_121chain_redirect = lr_zp_redirect.copy()
        t_zp_redirect_inheritance = {a:b for a,b in lr_zp_redirect.items() if a != b}

        print(f'\n{timeHMS()}:({doX_s}) Saving results for restored parts of big 121s')

        t_big121s_edited_clean      = [IDs for IDs in t_big121s_edited if IDs is not None]

        t_segments_new = copy.deepcopy(segments2)
        for conn in lr_big121s_conn_121:
            fr, to = conn
            fr_new, to_new = old_conn_2_new(conn,lr_121chain_redirect)
            print(f'edge :({fr},{to}) or = {G2_n_to(fr_new)}->{G2_n_from(to_new)}')  
            save_connections_two_ways(t_segments_new, lr_big121s_perms_pre[conn], fr, to, lr_121chain_redirect, g0_contours)

        # zp relations (edges) were not used in saving 121s. so they were not relinked.
        for slave, master in t_zp_redirect_inheritance.items():
            lr_121chain_redirect[slave] = lr_121chain_redirect[master]
        # at this time some segments got condenced into big 121s. right connection might be changed.

        
        #lr_time_active_segments = defaultdict(list)
        #for t_segment_index, t_segment_nodes in enumerate(t_segments_new):
        #    for t_time in [G_time(node) for node in t_segment_nodes]:
        #        lr_time_active_segments[t_time].append(t_segment_index)
        ## sort keys in lr_time_active_segments
        #lr_time_active_segments = {t:lr_time_active_segments[t] for t in sorted(lr_time_active_segments.keys())}

        #for_graph_plots(G, segs = t_segments_new) 
        print(f'\n{timeHMS()}:({doX_s}) Working on real and fake events (merges/splits):\nPreparing data...')
        """
        ===============================================================================================
        ============ RECALCULATE MERGE/SPLIT/MIXED EVENTS (FOR ACTUAL EVENT PROCESSING) ===============
        ===============================================================================================
        NOTE: same as process before 121 recovery. except now mixed type is calculated.
        NOTE: describtion is in "get_event_types_from_segment_graph" body/
        """
        # for_graph_plots(G, segs = t_segments_new, suptitle = f'{doX}_before', show = True)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if 1 == 1: 

            G2, lr_ms_edges_main = get_event_types_from_segment_graph(G2)

            # -----------------------------------------------------------------------------------------------
            # ------ PREPARE DATA INTO USABLE FORM ----
            # -----------------------------------------------------------------------------------------------

            t_merge_real_to_ID_relevant = []
            t_split_real_from_ID_relevant = []
            """
            ----------------------------------------------------------------------------------------------
            ------ GENERATE INFORMATON ABOUT REAL AND FAKE MERGE/SPLIT EVENTS----
            ----------------------------------------------------------------------------------------------
            IDEA: for real events gather possible choices for contours for each time step during merge event (t_perms)
            IDEA: for fake events gather evolutions of contours which integrate fake branches and stray nodes (t_combs)
            WORKFLOW: take master event ID- either node into which merge happens or node from which spit happens
            WORKFLOW: real: find branches, gather nodes between branches and master, find all options for time steps
            WORKFLOW: fake: determine branches and their parents, gather options that go though branch parent- master path
            NOTE: everything is held in one dictionary, but keys and contents are different
            """
            print('Generating information for:')
            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            t_event_start_end_times = {'merge':{}, 'split':{}, 'mixed':{}}

            # ------------------------------------- REAL MERGE/SPLIT -------------------------------------
            state = 'merge'
            for to in lr_ms_edges_main[state]: 
                predecessors            = list(G2.predecessors(to))# G2_t_start G2_t_end
                predecessors            = [ lr_121chain_redirect[t] for t in G2.predecessors(to)] # update if inherited
                times_end               = { t: G2_t_end(t) for t in predecessors}
                time_end                = G2_t_start(to)
                times                   = { i: np.arange(t_from_end, time_end + 1, 1) for i, t_from_end in times_end.items()}
                time_start_min          =   min(times_end.values())    # take earliest branch time

                if time_end - time_start_min <= 1: continue            # no time steps between from->to =  nothing to resolve

                # pre-isolate graph segment where all sub-events take place
                node_to = G2_n_from(to)  #t_segments_new[to][0]
                nodes_keep    = [n for n in G.nodes() if time_start_min < G_time(n) < time_end and G_owner(n) in (None, -1)] 
                nodes_keep.append(node_to)

                dfs_sol = set()
                dfs_pred(G.subgraph(nodes_keep), node_to, time_lim = time_start_min , node_set = dfs_sol)
                node_subIDs_all = disperse_nodes_to_times(dfs_sol) # reformat sol into time:subIDs
                node_subIDs_all = {t:sorted(node_subIDs_all[t]) for t in sorted(node_subIDs_all)}
                
                active_IDs = {t:[] for t in np.arange(time_start_min + 1, time_end)}

                for fr, times_all in times.items():
                    for time in times_all[1:-1]: # remove start-end points
                        active_IDs[time].append(fr)

                t_event_start_end_times[state][to] = {
                                                            't_start'       :   times_end,
                                                            'branches'      :   predecessors,
                                                            't_end'         :   time_end,
                                                            'times'         :   times,
                                                            't_perms'       :   {},
                                                            'subIDs'        :   node_subIDs_all,
                                                            'active_IDs'    :   active_IDs
                                                            }

                t_merge_real_to_ID_relevant.append(to)
                
            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            state = 'split'
            for fr in lr_ms_edges_main[state]:
                fr_new              = lr_121chain_redirect[fr]
                successors          = list(G2.successors(fr_new))
                times_end           = {to: G2_t_start(to) for to in successors} 
                time_start          = G2_t_end(fr_new)
                times               = {t: np.arange(time_start, t_end + 1, 1) for t, t_end in times_end.items()}
                
                node_from           = G2_n_to(fr_new)      #t_segments_new[fr_new][-1]
                # pre-isolate graph segment where all sub-events take place
                time_end_max        = max(times_end.values())

                if time_end_max - time_start <= 1: continue

                nodes_keep          = [n for n in G.nodes() if time_start < G_time(n) < time_end_max and G_owner(n) in (None, -1)]
                nodes_keep.append(node_from)
                dfs_sol             = set()
                dfs_succ(G.subgraph(nodes_keep), node_from, time_lim = time_end_max , node_set = dfs_sol)
                node_subIDs_all     = disperse_nodes_to_times(dfs_sol) # reformat sol into time:subIDs
                node_subIDs_all     = {t:sorted(node_subIDs_all[t]) for t in sorted(node_subIDs_all)}

                active_IDs          = {t:[] for t in np.arange(time_end_max -1, time_start, -1)} #>>> reverse for reverse re

                for fr, times_all in times.items():
                    for time in times_all[1:-1]: # remove start-end points
                        active_IDs[time].append(fr)

                t_event_start_end_times[state][fr_new] = {
                                                                't_start'       :   time_start,
                                                                'branches'      :   successors,
                                                                't_end'         :   times_end,
                                                                'times'         :   times,
                                                                't_perms'       :   {},
                                                                'subIDs'        :   node_subIDs_all,
                                                                'active_IDs'    :   active_IDs
                                                                }

                t_split_real_from_ID_relevant.append(fr_new)

            print(f'\n{timeHMS()}:({doX_s}) Real merges({lr_ms_edges_main["merge"]})/splits({lr_ms_edges_main["split"]})... Done')

            
            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            state = 'mixed'
            for fr_all, to_all in lr_ms_edges_main[state].items():

                fr_all_new  = tuple([lr_121chain_redirect[i] for i in fr_all ])
                          
                times_end   = {i: G2_t_end(   i)  for i in fr_all_new   }
                times_start = {i: G2_t_start( i)  for i in to_all       }
                                                        
                nodes_from  = {i: G2_n_to(   i)  for i in fr_all_new   }
                nodes_to    = {i: G2_n_from( i)  for i in to_all       } #t_segments_new[t][0]

                # pre-isolate graph segment where all sub-events take place
                time_start_min    = min(times_end.values())    # take earliest branch time
                time_end_max      = max(times_start.values()) 
                times_all     = np.arange(time_start_min, time_end_max + 1, 1)
                active_IDs    = {t:[] for t in times_all[1:]}

                # get non-segment nodes within event time inteval
                nodes_keep    = [n for n in G.nodes() if time_start_min < G_time(n) < time_end_max and G_owner(n) in (None, -1)] 
                
                # i had problems with nodes due to internal events. node_subIDs_all was missing times.
                # ill add all relevant segments to subgraph along stray nodes. gather connected components
                # and then subract all segment nodes, except first target nodes. 
                # EDIT 12.12.2023 idk what was the issue then. i add all branches, then delete from segments, then add target nodes..
                # EDIT 12.12.2023 it works for zero inter-event node cases.
                nodes_segments_all    = []
                branches_all          = fr_all_new + to_all
                for i in branches_all:
                    nodes_segments_all.extend(t_segments_new[i])
                nodes_keep.extend(nodes_segments_all)

                # had cases where target nodes are decoupled and are split into different CC clusters.
                # for safety create fake edges by connecting all branches in sequence. but only those that not exist already
                x = {**nodes_from, **nodes_to} # inc segments last node, target seg first node
                fake_edges = [(x[a],x[b]) for a,b in zip(branches_all[:-1], branches_all[1:]) if not G.has_edge(x[a],x[b])]
                

                G.add_edges_from(fake_edges)
                subgraph = G.subgraph(nodes_keep)   

                ref_node = G2_n_from(to_all[0])      #t_segments_new[to_all[0]][0]

                clusters    = nx.connected_components(subgraph.to_undirected())
                
                sol     = next((cluster for cluster in clusters if ref_node in cluster), None)
                assert sol is not None, 'cannot find connected components'
                sol     = [t for t in sol if t not in nodes_segments_all]

                sol.extend(nodes_to.values())

                G.remove_edges_from(fake_edges)
                node_subIDs_all = disperse_nodes_to_times(sol) # reformat sol into time:subIDs
                node_subIDs_all = {t:node_subIDs_all[t] for t in sorted(node_subIDs_all)}

                for fr in fr_all_new: # from end of branch time to event max time
                    for time in np.arange(times_end[fr] + 1, time_end_max + 1, 1):
                        active_IDs[time].append(fr)

                t_event_start_end_times[state][fr_all] = {
                                                                't_start'       :   times_end,
                                                                'branches'      :   fr_all_new,
                                                                't_end'         :   times_start,
                                                                'times'         :   {},
                                                                't_perms'       :   {},
                                                                'target_nodes'  :   nodes_to,
                                                                'subIDs'        :   node_subIDs_all,
                                                                'active_IDs'    :   active_IDs}

        # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    ===============================================================================================
    ====  Determine interpolation parameters ====
    ===============================================================================================
    we are interested in segments that are not part of psuedo branches (because they will be integrated into path)
    pre merge segments will be extended, post split branches also, but backwards. 
    for pseudo event can take an average paramets for prev and post segments.
    """
    print(f'\n{timeHMS()}:({doX_s}) Determining interpolation parameters k and s for segments...')
    
    # get segments that have possibly inherited other segments and that are not branches
    t_segments_IDs_relevant = [i for i,traj in enumerate(t_segments_new) if len(traj) > 0]

    k_s_buffer_len_max = 8          # i want this long history at max
    k_s_buffer_len_min = 3      # 
    k_s_start_at_index = 0   # default start from index 0
    k_s_anal_points_max = 20   # at best analyze this many points, it requires total traj of len k_s_anal_points_max + "num pts for interp"
    k_s_anal_points_min = 5    # 
    k_all = (1,2)
    s_all = (0,1,5,10,25,50,100,1000,10000)
    t_k_s_combs = list(itertools.product(k_all, s_all))

    t_segment_k_s       = defaultdict(tuple)
    t_segment_k_s_diffs = defaultdict(dict)
    for t_ID in t_segments_IDs_relevant:
        trajectory          = np.array([G_centroid(n)   for n in t_segments_new[t_ID]])     #G.nodes[n]["centroid"   ]
        time                = np.array([G_time(n)       for n in t_segments_new[t_ID]])     #G.nodes[n]["time"       ]
        t_do_k_s_anal = False
    
        if  trajectory.shape[0] > k_s_anal_points_max + k_s_buffer_len_max:   # history is large

            h_start_point_index2 = trajectory.shape[0] - k_s_anal_points_max - k_s_buffer_len_max

            h_interp_len_max2 = k_s_buffer_len_max
        
            t_do_k_s_anal = True

        elif trajectory.shape[0] > k_s_anal_points_min + k_s_buffer_len_min:  # history is smaller, give prio to interp length rather inspected number count
                                                                        # traj [0,1,2,3,4,5], min_p = 2 -> 4 & 5, inter_min_len = 3
            h_start_point_index2 = 0                                    # interp segments: [2,3,4 & 5], [1,2,3 & 4]
                                                                        # but have 1 elem of hist to spare: [1,2,3,4 & 5], [0,1,2,3 & 4]
            h_interp_len_max2 = trajectory.shape[0]  - k_s_anal_points_min  # so inter_min_len = len(traj) - min_p = 6 - 2 = 4
        
            t_do_k_s_anal = True

        else:
            h_interp_len_max2 = trajectory.shape[0]                     # traj is very small

        if t_do_k_s_anal:                                               # test differet k and s combinations and gather inerpolation history.
            t_comb_sol, errors_sol_diff_norms_all = extrapolate_find_k_s(trajectory, time, t_k_s_combs, k_all, h_interp_len_max2, h_start_point_index2, debug = 0, debug_show_num_best = 2)
            t_segment_k_s_diffs[t_ID] = errors_sol_diff_norms_all[t_comb_sol]
            t_segment_k_s[      t_ID] = t_comb_sol
        else:                                                           
            t_segment_k_s[t_ID]         = None
            t_segment_k_s_diffs[t_ID]   = None

    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    ===============================================================================================
    ================================ DEAL WITH FAKE EVENT RECOVERY ================================
    ===============================================================================================
    """
    print(f'\n{timeHMS()}:({doX_s}) Working on fake events...')
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

    # ----------------- Evaluate and find most likely evolutions of path -------------------------
    print(f'\n{timeHMS()}:({doX_s}) Finding most possible path evolutions...')
                
    # >>>  this approach of reconnecting lr_121chain_redirect is old. it was used in situation similar to ZP case.
    # >>>  where inheritance did not automatically resolved during saving. i should fix fake splits & merges and fix this after it.
    G_seg_view_2 = nx.Graph()
    G_seg_view_2.add_edges_from([(x,y) for y,x in lr_121chain_redirect.items()])

    lr_C1_condensed_connections = [sorted(c, key = lambda x: x) for c in nx.connected_components(G_seg_view_2)]
    lr_C1_condensed_connections = sorted(lr_C1_condensed_connections, key = lambda x: x[0])
    # lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
    t_condensed_connections_all_nodes = sorted(sum(lr_C1_condensed_connections,[])) # neext next
    lr_fake_redirect = {tID: tID for tID in range(len(segments2))} 
    for subIDs in lr_C1_condensed_connections:
        for t_subID in subIDs:
            lr_fake_redirect[t_subID] = min(subIDs)


    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    ===============================================================================================
    ============================  EXTEND REAL MERGE/SPLIT BRANCHES ================================
    ===============================================================================================
    iteratively extend branches though split/merge event. to avoid conflicts with shared nodes
    extension is simultaneous for all branches at each time step and contours are redistributed conservatively
    -> at time T, two branches [B1, B2] have to be recovered from node cluster C = [1,2,3]
    -> try node redistributions- ([partition_B1,partition_B2], ..) = ([[1],[2,3]], [[2],[1,3]], [[3]lr_fake_redirect,[1,2]])    
    """
    t_out                   = defaultdict(dict)
    t_extrapolate_sol       = defaultdict(dict)
    t_extrapolate_sol_comb  = defaultdict(dict)
    ms_branch_extend_IDs =  [(t,'merge') for t in t_merge_real_to_ID_relevant] 
    ms_branch_extend_IDs += [(lr_fake_redirect[t],'split') for t in t_split_real_from_ID_relevant]
    #ms_branch_extend_IDs += [(t,'mixed') for t in lr_conn_mixed_from_to]
    ms_branch_extend_IDs += [(t,'mixed') for t in lr_ms_edges_main['mixed']]
    

    ms_mixed_completed      = {'full': defaultdict(dict),'partial': defaultdict(dict)} 
    lr_post_branch_rec_info = {}
    ms_early_termination = {}
    print(f'\n{timeHMS()}:({doX_s}) Analyzing real merge/split events. extending branches: {ms_branch_extend_IDs} ... ')
    for t_ID, state in ms_branch_extend_IDs:
        """
        ===========================================================================================
        =================== DETERMINE BRANCHES, START AND END TIMES OF EVENT ======================
        ===========================================================================================
        """
        if state in ('merge','split'):
            branches = t_event_start_end_times[state][t_ID]['branches']
            times_target = []
        elif state == 'mixed':
            branches      = t_event_start_end_times[state][t_ID]['branches']
            nodes_to  = t_event_start_end_times[state][t_ID]['target_nodes']
            times_target  = [G_time(t) for t in nodes_to.values()]
            subIDs_target = {t:[] for t in times_target}
            for t_time, *subIDs in nodes_to.values():
                subIDs_target[t_time] += subIDs

        if state == 'split':
            t_start = t_event_start_end_times[state][t_ID]['t_start']
            t_end   = min(t_event_start_end_times[state][t_ID]['t_end'].values())
        elif state == 'mixed':
            t_start = min(t_event_start_end_times[state][t_ID]['t_start'].values())
            t_end   = max(t_event_start_end_times[state][t_ID]['t_end'].values())
        else:
            t_start = min(t_event_start_end_times[state][t_ID]['t_start'].values())
            t_end   = t_event_start_end_times[state][t_ID]['t_end']
        """
        ===========================================================================================
        =============================== PREPARE DATA FOR EACH BRANCH ==============================
        ===========================================================================================
        """
        all_norm_buffers, all_traj_buffers, all_area_buffers, all_time_buffers, t_all_k_s  = {}, {}, {}, {}, {}
         
        for branch_ID in branches: 
            
            branch_ID_new = lr_fake_redirect[branch_ID]
            if state in ('merge','mixed'):
                
                t_from    = t_event_start_end_times[state][t_ID]['t_start'][branch_ID] # last of branch
                node_from = G2_n_to(branch_ID_new)     

                if state == 'merge':
                    t_to      = t_event_start_end_times[state][t_ID]['t_end']                # first of target
                    node_to   = G2_n_from(t_ID)      
                    conn = (branch_ID, t_ID)
                else: 
                    t_to = max(t_event_start_end_times[state][t_ID]['t_end'].values())
                    node_to = (-1)
                    conn = (branch_ID,)
            else:
                conn = (t_ID, branch_ID)
                t_from    = t_event_start_end_times[state][t_ID]['t_start'] # last of branch
                t_to      = t_event_start_end_times[state][t_ID]['t_end'][branch_ID]                # first of target
                node_from = G2_n_to(t_ID)                #t_segments_new[t_ID][-1]
                node_to   = G2_n_from(branch_ID)       #t_segments_new[branch_ID][0]
            if state in ('split', 'merge') and np.abs(t_to - t_from) < 2:    # note: there are mixed cases with zero event nodes
                t_out[t_ID][branch_ID] = None
                continue
            """
            =======================================================================================
            ============== GET SHORT PRIOR HISTORY FOR CENTROIDS, AREAS OF A BRANCH  ==============
            =======================================================================================
            """
            if state in ('merge','mixed'):
                nodes = [n for n in t_segments_new[branch_ID_new] if G_time(n) > t_from - h_interp_len_max2]
            else:
                nodes = [n for n in t_segments_new[branch_ID_new] if G_time(n) < t_to + h_interp_len_max2]

            trajectory  = np.array([G_centroid( n)   for n in nodes])
            time        = np.array([G_time(     n)   for n in nodes])
            area        = np.array([G_area(     n)   for n in nodes])
            
            N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
            if t_segment_k_s_diffs[branch_ID_new] is not None:
                last_deltas   = list(t_segment_k_s_diffs[branch_ID_new].values())[-N:]  # not changing for splits
            else:
                last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5

            all_norm_buffers[branch_ID] = CircularBuffer(N, last_deltas)
            if state in ('merge','mixed'):
                all_traj_buffers[branch_ID] = CircularBuffer(h_interp_len_max2, trajectory)
                all_area_buffers[branch_ID] = CircularBuffer(h_interp_len_max2, area)
                all_time_buffers[branch_ID] = CircularBuffer(h_interp_len_max2, time)
            else: 
                all_traj_buffers[branch_ID] = CircularBufferReverse(h_interp_len_max2, trajectory)
                all_area_buffers[branch_ID] = CircularBufferReverse(h_interp_len_max2, area)
                all_time_buffers[branch_ID] = CircularBufferReverse(h_interp_len_max2, time)

            t_extrapolate_sol[      conn] = {}
            t_extrapolate_sol_comb[ conn] = {}
            lr_post_branch_rec_info[conn] = state
            t_times_accumulate_resolved     = []
            
            if t_segment_k_s[branch_ID_new] is not None:
                t_k,t_s = t_segment_k_s[branch_ID_new]
                t_all_k_s[branch_ID] = t_segment_k_s[branch_ID_new]
            else:
                t_k,t_s = (1,5)
                t_all_k_s[branch_ID] = (1,5)
        

        # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        """
        ===========================================================================================
        ================================ START BRANCH EXTRAPOLATION ===============================
        ===========================================================================================
        walk times from start of an event to end. each time step there are branches "t_branch_IDs" that have to be recovered 
        by default branch is being recovered from time it ended to end of event => "t_branch_IDs_OG" = "t_branch_IDs",
        but once branch recovery is terminated it should not be considered anymore, so t_branch_IDs_OG is wrong. refine
        each step branches use contours from pool of contours "subIDs". some branches may start later.
        """
        branch_failed = []
        t_report = set()
        for time_next, t_branch_IDs_OG in t_event_start_end_times[state][t_ID]['active_IDs'].items():
            if state == 'split':  t_time = time_next + 1                            # t_time_next is in the past
            else:                   t_time = time_next - 1                            # t_time_next is in the future

            branch_IDs = [t for t in t_branch_IDs_OG if t not in branch_failed]     # t_branch_failed are resolved or terminated
            if len(branch_IDs) == 0: continue                                         # no branches to resolve, try next step.

            subIDs = t_event_start_end_times[state][t_ID]['subIDs'][time_next]  # contour pool

            if len(subIDs) < len(branch_IDs):                                       # more branches than contorus available.
                ms_early_termination[t_ID, state] = [time_next,branch_IDs]        # cant resolve this.
                branch_failed.extend(branch_IDs)                                    # consider both branches failed.
                continue                                                                # continue to end event time maybe there are any other 
                                                                                        # branches to fix. see "extend_branchs_solo_node_continue.png"
            centroids_extrap = np.zeros((len(branch_IDs),2))
            areas_extrap = np.zeros(len(branch_IDs))

            for t, branch_ID in enumerate(branch_IDs):                              # extrapolate traj of active branches
                traj_b    = all_traj_buffers[branch_ID].get_data()
                time_b    = all_time_buffers[branch_ID].get_data()
                area_b    = all_area_buffers[branch_ID].get_data()
                t_k, t_s    = t_all_k_s[branch_ID]
                centroids_extrap[t] = interpolate_trajectory(traj_b, time_b, which_times = [time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]
                areas_extrap[t]     = interpolateMiddle1D_2(time_b, area_b, [time_next], rescale = True, s = 15, debug = 0, aspect = 'auto', title = 1)
                    
            
            if len(branch_IDs) == 1:
                perms_distribution2 = [[list(t)] for t in combs_different_lengths(subIDs)]  # if only one choice, gen diff perms of contours
            else:
                perms_distribution2 = list(split_into_bins(subIDs,len(branch_IDs)))       # multiple choices, get perm distrib options
            perms_distribution2 = [[tuple(sorted(b)) for b in a] for a in perms_distribution2]

            # calculate parameters of different permutations of contours.
            permutation_params = {}
            permutations = combs_different_lengths(subIDs)
            permutations = [tuple(sorted(a)) for a in permutations]
            for permutation in permutations:
                hull = cv2.convexHull(np.vstack([g0_contours[time_next][tID] for tID in permutation]))
                permutation_params[permutation] = centroid_area(hull)


            # evaluate differences between extrapolated data and possible solutions in subID permutations/combinations
            diff_choices      = {}      # holds index key on which entry in perms_distribution2 has specifict differences with target values.
            diff_choices_area = {}
            for k, redist_case in enumerate(perms_distribution2):
                centroids             = np.array([permutation_params[t][0] for t in redist_case])
                areas                 = np.array([permutation_params[t][1] for t in redist_case])
                diff_choices[k]       = np.linalg.norm(centroids_extrap - centroids, axis=1)
                diff_choices_area[k]  = np.abs(areas_extrap - areas)/areas_extrap
            a = 1
            # -------------------------------- refine simultaneous solution ---------------------------------------
            # evaluating sum of diffs is a bad approach, because they may behave differently

            norms_all     = {t:np.array(  all_norm_buffers[t].get_data()) for t in branch_IDs}
            max_diff_all  = {t:max(np.mean(n),5) + 5*np.std(n)              for t,n in norms_all.items()}
            relArea_all = {}
            for k in branch_IDs:
                t_area_hist         = all_area_buffers[k].get_data()
                relArea_all[k]    = np.abs(np.diff(t_area_hist))/t_area_hist[:-1]
            t_max_relArea_all  = {t:max(np.mean(n) + 5*np.std(n), 0.35) for t,n in relArea_all.items()}

                
            choices_pass_all      = [] # holds indidicies of perms_distribution2
            choices_partial       = []
            choices_partial_sols  = {}
            choices_pass_all_both = []
            # filter solution where all branches are good. or only part is good
            for t in diff_choices:
                pass_test_all     = diff_choices[       t] < np.array(list(max_diff_all.values())) # compare if less then crit.
                pass_test_all2    = diff_choices_area[  t] < np.array(list(t_max_relArea_all.values())) # compare if less then crit.
                pass_both_sub = np.array(pass_test_all) & np.array(pass_test_all2) 
                if   all(pass_both_sub):   choices_pass_all_both.append(t)          # all branches pass
                elif any(pass_both_sub):
                    choices_partial.append(t)
                    choices_partial_sols[t] = pass_both_sub
                

            if len(choices_pass_all_both) > 0:     # isolate only good choices    
                if len(choices_pass_all_both) == 1:
                    diff_norms_sum = {choices_pass_all_both[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1 = {t: diff_choices[     t] for t in choices_pass_all_both}
                    temp2 = {t: diff_choices_area[t] for t in choices_pass_all_both}
                    test, diff_norms_sum = two_crit_many_branches(temp1, temp2, len(branch_IDs))
                #test, t_diff_norms_sum = two_crit_many_branches(t_diff_choices, t_diff_choices_area, len(t_branch_IDs))
                #t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items() if t in t_choices_pass_all}

            # if only part is good, dont drop whole solution. form new crit based only on non-failed values.
            elif len(choices_partial) > 0:  
                if len(choices_partial) == 1:
                    diff_norms_sum = {choices_partial[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1 = {t: diff_choices[     t] for t in choices_partial}
                    temp2 = {t: diff_choices_area[t] for t in choices_partial}
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
                diff_norms_sum = {0:[t + 1 for t in max_diff_all.values()]} # on fail spoof case which will fail.

            where_min = min(diff_norms_sum, key = diff_norms_sum.get)
            sol_d_norms = diff_choices[where_min]
            sol_subIDs = perms_distribution2[where_min]
            branch_pass = []
            for branch_ID, subIDs, sol_d_norm in zip(branch_IDs,sol_subIDs, sol_d_norms):

                if sol_d_norm < max_diff_all[branch_ID]:
                    all_norm_buffers[branch_ID].append(sol_d_norm)
                    all_traj_buffers[branch_ID].append(permutation_params[tuple(subIDs)][0])
                    all_area_buffers[branch_ID].append(permutation_params[tuple(subIDs)][1])
                    all_time_buffers[branch_ID].append(time_next)
                    
                    if state == 'merge':      t_conn = (branch_ID, t_ID)
                    elif state == 'mixed':    t_conn = (branch_ID,)
                    else:                       t_conn = (t_ID, branch_ID)

                    t_extrapolate_sol_comb[t_conn][time_next] = tuple(subIDs)
                    branch_pass.append(branch_ID)
                    t_report.add(t_conn)
                else:
                    branch_failed.append(branch_ID)
                    continue


            if state in  ('mixed'):                        # for mixed cases i should search of extended incoming branch has reached any of target branches
                if time_next in times_target:            # if recovered time step is at same time as some of first target nodes
                    for branch_ID in branch_pass:        # test successfully recovered branches for contour ID overlap

                        nodes_sol_subIDs = t_extrapolate_sol_comb[(branch_ID,)][time_next]     # subIDs in solution
                        nodes_solution = tuple([time_next] + list(nodes_sol_subIDs))           # reconstruct a node of a solution

                        if nodes_solution in nodes_to.values():
                            target_ID = find_key_by_value(nodes_to,nodes_solution)         # if this node is in list of target nodes

                            ms_mixed_completed['full'][(branch_ID,)]['solution'] = nodes_sol_subIDs  # consider path recovered
                            if 'targets' not in ms_mixed_completed['full'][(branch_ID,)]: ms_mixed_completed['full'][(branch_ID,)]['targets'] = []
                            ms_mixed_completed['full'][(branch_ID,)]['targets'].append(target_ID)    
                            branch_failed.append(branch_ID)                                          # add branch to failed list. to stop from extending it further
                        else:                                                                            # if not in list of tar nodes
                            set1 = set(subIDs_target[time_next])
                            set2 = set(nodes_sol_subIDs)
                            inter = set1.intersection(set2)                                              # check subID intersection

                            if len(inter)> 0:                                                            # if there is an intersection
                                t_intersecting_branches_IDs = {t for t,subIDs in nodes_to.items() if set(subIDs[1:]).intersection(set(nodes_sol_subIDs)) != {}}
                                ms_mixed_completed['partial'][(branch_ID,)]['solution'] = nodes_sol_subIDs  # add here, but idk what to do  yet
                                print(t_intersecting_branches_IDs)
                                branch_failed.append(branch_ID)

                                if 'mixed' not in issues_all_dict[doX]: issues_all_dict[doX]['mixed'] = []
                                issues_all_dict[doX]['mixed'].append(f'branch {branch_ID} extention resulted in multiple branches : {nodes_solution} ')
                                t_extrapolate_sol_comb[(branch_ID,)].pop(time_next,None)
                                #assert 1 == -1, 'extension of mixed type branch resulted in partial success. target cntr is in subIDs. check this case more closely'
                                #if 'targets' not in ms_mixed_completed['partial'][t_ID]: ms_mixed_completed['partial'][t_ID]['targets'] = []
                                #ms_mixed_completed['partial'][t_ID]['targets'].append(t_target_ID)
        a = 1
        for conn in t_report:
            dic = t_extrapolate_sol_comb[conn]
            if len(dic) == 0: continue
            if len(conn) == 1:
                fr, to = conn[0], -1
            else:
                (fr, to) = conn
            t_min, t_max    = min(dic), max(dic)
            node_from       = tuple([t_min] + list(dic[t_min]))
            node_to         = tuple([t_max] + list(dic[t_max]))
            print(f' {state}:connection :{(fr, to)} = {node_from}->{node_to}')
        a = 1

    lr_conn_merges_good = set()#dict()#defaultdict(set)
    
    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    lr_conn_merges_good = [(t_ID, state) for t_ID, state in lr_post_branch_rec_info.items() if len(t_extrapolate_sol_comb[t_ID]) > 0]
    
    for t_conn, state in lr_conn_merges_good:
        # principle of save_connections_X is decribed in "misc.save_connections_two_ways"
        if len(t_conn) == 2:    t_from, t_to = t_conn
        else:                   t_from, t_to = t_conn[0], -1
        t_from_new = lr_fake_redirect[t_from]
        t_combs = t_extrapolate_sol_comb[t_conn]

        if len(t_combs) == 0: continue                              # no extension, skip.

        if state in ('merge', 'mixed') and t_conn not in ms_mixed_completed['full']:
            save_connections_merges(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from_new,  None, lr_fake_redirect, g0_contours)
        elif state == 'split':
            save_connections_splits(t_segments_new, t_extrapolate_sol_comb[t_conn], None,  t_to, None, lr_fake_redirect, g0_contours)
        else:
            t_to = ms_mixed_completed['full'][t_conn]['targets'][0]
            # remove other edges from mixed connection segment graph
            t_from_other_predecessors = [lr_fake_redirect[t] for t in G2.predecessors(t_to) if t != t_conn[0]]
            edges = [(t, t_to) for t in t_from_other_predecessors]
            G2.remove_edges_from(edges)
            save_connections_two_ways(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from_new,  t_to, lr_fake_redirect, g0_contours)


    # for_graph_plots(G, segs = t_segments_new, suptitle = f'{doX}_after', show = True)        


    aaa = defaultdict(set)
    for t, segment in enumerate(t_segments_new):
        for node in segment:
            aaa[t].add(G_owner(node)) #G.nodes[t_node]["owner"]
    tt = []
    for n in G.nodes():
        if "time" not in G.nodes[n]:
            set_custom_node_parameters(g0_contours, [n], None, calc_hull = 1)
            tt.append(n)
    print(f'were missing: {tt}')
    

    print(f'\n{timeHMS()}:({doX_s}) Final. Recompute new straight segments')
    """
    ===============================================================================================
    ============== Final passes. New straight segments. ===================
    ===============================================================================================
    WHY: after merge/split/merge extensions there may be explicit branches left over in event area
    HOW: analyze recent graph for straight segments and determine which segments are different from old
    """
    t_segments_fin = graph_extract_paths(G, min_length = 2)

    unresolved_new = list(range(len(t_segments_fin)))
    unresolved_old = [t for t, nodes in enumerate(t_segments_new) if len(nodes) > 0]
    resolved_new = []
    resolved_old = []
    for i_new, traj in enumerate(t_segments_fin):                           # first test if first elements are the same
        for i_old     in unresolved_old:
            if traj[0] == t_segments_new[i_old][0]:
                resolved_new.append(i_new)
                resolved_old.append(i_old)
                break
    unresolved_new = [t for t in unresolved_new if t not in resolved_new]
    unresolved_old = [t for t in unresolved_old if t not in resolved_old]
    temp_u_new = copy.deepcopy(unresolved_new)                                # (cant iterate though changing massives. copy)
    temp_u_old = copy.deepcopy(unresolved_old)
    for i_new in temp_u_new:         # can be improved. minor                 # then test unresolved using set intersection
        for i_old in temp_u_old:                                              # might be needed if segments got wider
            if i_old not in resolved_old:
                intersection_length = len(set(t_segments_fin[i_new]).intersection(set(t_segments_new[i_old])))
                if intersection_length > 0:
                    unresolved_new.remove(i_new)
                    unresolved_old.remove(i_old)
                    resolved_new.append(i_new)
                    resolved_old.append(i_old)
                    break
    t_start_ID = len(t_segments_new)
    fin_additional_segments_IDs = list(range(t_start_ID, t_start_ID + len(unresolved_new), 1))

    for ID in unresolved_new:      # add new segments to end of old storage

        ID_new = len(t_segments_new)

        t_segments_new.append(t_segments_fin[ID])

        node_start, node_end = t_segments_new[ID_new][0], t_segments_new[ID_new][-1] # t_segments_new[ID_new][[0, -1]]

        set_custom_node_parameters(g0_contours, t_segments_fin[ID], ID_new, calc_hull = 1)        # 
        G2_set_parameters(t_segments_fin[ID], ID_new)#, G_time)


    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for_graph_plots(G, segs = t_segments_fin)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Update connectivity')
    # ============== Final passes. New straight segments. Update connectivity ===================
    #G2.add_edges_from(G2.edges())
    fin_connectivity_graphs = defaultdict(list) #fin_additional_segments_IDs
    t_has_holes_report = {}
    G2 = graph_check_paths(lr_maxDT, t_has_holes_report)

    print(f'\n{timeHMS()}:({doX_s}) Paths that have failed hole test: {t_has_holes_report}')

    #fin_connectivity_graphs = {t_conn:t_vals for t_conn, t_vals in fin_connectivity_graphs.items() if len(t_vals) > 2}
    for t_ID in fin_additional_segments_IDs:
        t_segment_k_s_diffs[t_ID] = None
        t_segment_k_s[t_ID] = None
        lr_fake_redirect[t_ID] = t_ID
    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if 1 == 1:
        print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Branches of splits/merges')
        """
        ===============================================================================================
        ========================= Final passes. recompute split/merge/mixed info ======================
        ===============================================================================================
        NOTE 0: same as pre- merge/split/mixed extension (which was same as pre 121 recovery)
        NOTE 1: extension of branches requires more or equal number contours than branches.
        NOTE 1: thats why there is no point in recovering branches past this stage. recover only to earliest case.
        """
        fin_extend_info = defaultdict(dict)
            
        #G2_dir, fin_edges_main = get_event_types_from_segment_graph(G2)   
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
    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
            node_from = G2_n_from(ID)#t_segments_new[ID][0]
            dfs_pred(G, node_from, time_lim = G2_t_start(ID) - 10, node_set = back_d[ID])
        back_d = {i:v for i,v in back_d.items() if len(v) > 1}

        forw_d = defaultdict(set)
        for ID in end_points:
            node_from = G2_n_to(ID)#t_segments_new[ID][-1]
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

    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Extrapolate')
    # ============= Final passes. Terminated segments. Extrapolate segments =================
    t_report                = set()
    t_out                   = defaultdict(dict)
    t_extrapolate_sol       = defaultdict(dict)
    t_extrapolate_sol_comb  = defaultdict(dict)
    for (t_ID, state), t_combs in fin_extend_info.items():
        conn = (t_ID, state) # <<<<< custom. t_ID may be two states, and they have to be differentiated
        if state == 'forward':
            time_from   = G2_t_end(t_ID)#t_segments_new[t_ID][-1][0]
            nodes       = [n for n in t_segments_new[t_ID] if G_time(n) > time_from - h_interp_len_max2]
            node_from   = G2_n_to(t_ID)#t_segments_new[t_ID][-1]
        else:
            time_to     = G2_t_start(t_ID)#t_segments_new[t_ID][0][0]
            nodes       = [n for n in t_segments_new[t_ID] if G_time(n) < time_to   + h_interp_len_max2]
            node_from   = G2_n_from(t_ID)#t_segments_new[t_ID][0]

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
        
        t_times_accumulate_resolved = []
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


    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
            save_connections_merges(t_segments_new, t_extrapolate_sol_comb[conn], ID,  None, lr_fake_redirect, g0_contours)
        elif state == 'back':
            save_connections_splits(t_segments_new, t_extrapolate_sol_comb[conn], None,  ID, lr_fake_redirect, g0_contours)


    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    tt = []
    for n in G.nodes():
        if "time" not in G.nodes[n]:
            set_custom_node_parameters(g0_contours, [n], None, calc_hull = 1)
            tt.append(n)
    print(f'were missing: {tt}')


    #if doX == 1: for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    print(f'\n{timeHMS()}:({doX_s}) Final. Recalculate connectivity')
    # ======= EXPORT. RECALCULATE CONNECTIONS BETWEEN SEGMENTS ========================
    
    t_has_holes_report = {}
    G2 = graph_check_paths(lr_maxDT, t_has_holes_report)                        
    # ========== EXPORT. DETERMINE TYPE OF EVENTS BETWEEN SEGMENTS ==========
    G2, export_ms_edges_main = get_event_types_from_segment_graph(G2)


    trajectories_all_dict[doX]  = t_segments_new
    graphs_all_dict[doX]        = [copy.deepcopy(G),copy.deepcopy(G2)]
    events_split_merge_mixed[doX] = export_ms_edges_main

contours_all_dict      = g0_contours

if 1 == 1:
    #[key_time, key_area, key_centroid, key_owner]
    #[key_t_start, key_t_end, key_n_start, key_n_end, key_e_dist] 
    t_merges_splits_rects = {'merge':defaultdict(list), 'split': defaultdict(list), 'mixed':defaultdict(list)}
    minTime = 100000000; maxTime = 0
    for doX, [Gb,G2b] in graphs_all_dict.items():
        # G and G2 are defined from last iteration but are referenced to intial object defined in modules/graphs_general
        # in order for G_ and G2_ functions to work they have to use old reference. so clear their contents and copy new data

        G.clear()
        G2.clear()
        
        G.add_nodes_from(Gb.nodes(data=True))
        G.add_edges_from(Gb.edges(data=True))

        G2.add_nodes_from(G2b.nodes(data=True))
        G2.add_edges_from(G2b.edges(data=True))

        #contours  = contours_all_dict[doX]['contours']
        for node in G.nodes():
            [cx,cy] = G_centroid(node)#G.nodes[node]['centroid']
            del G.nodes[node][key_centroid]
            G.nodes[node]['cx'] = cx
            G.nodes[node]['cy'] = cy
            if G_owner(node) == None: G_owner_set(node, -1)
            t_time, *subIDs = node

        if len(trajectories_all_dict[doX]) > 0:

            temp = for_graph_plots(G, segs = trajectories_all_dict[doX],optimize_pos = False, 
                                   show = False, node_size = 30, edge_width_path = 3, edge_width = 1, font_size = 7)

            all_nodes_pos, edge_width, edge_color, node_size = temp

            for node, (x,y) in all_nodes_pos.items():

                G.nodes[node]['viz'] = {'position': {'x': x, 'y': y*12, 'z': 0.0}, 
                                        'size': 0.23, 
                                        'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}}

            for k, edge in enumerate(G.edges()):

                #rgba = [int(a*255) for a in edge_color[k]] + [0.5]
                rgba = [int(a * 255) for a in edge_color[k]] + [0.5]
                G.edges[edge]['viz'] = {'shape':'dashed', 'color': {c:v for c,v in zip(['r', 'g', 'b', 'a'],rgba)}} 
                G.edges[edge]['viz'] = {'color': {'r': rgba[0], 'g': rgba[1], 'b': rgba[2], 'a': rgba[3]}, 'attributes': []}
            file_path = os.path.join(graphsFolder, f'G_{doX}.gexf')
            nx.write_gexf(G, file_path)

        for t_seg in G2.nodes():

            minTime = min(minTime,  G2_t_start( t_seg)) # G2.nodes[t_seg]['t_start'   ]
            maxTime = max(maxTime,  G2_t_end(   t_seg)) # G2.nodes[t_seg]['t_end'     ]

            state_from = 'merge'
            if t_seg in events_split_merge_mixed[doX][state_from]:
                predecessors  = events_split_merge_mixed[doX][state_from][t_seg]
                t_nodes_min_max = [G2_n_from(t_seg)] + [G2_n_to(t_pred) for t_pred in predecessors] 
                t_combine_contours = [] # 
                for t_time,*subIDs in t_nodes_min_max:
                    for t_subID in subIDs:
                        t_combine_contours.append(contours_all_dict[t_time][t_subID])
                t_rb_params = cv2.boundingRect(np.vstack(t_combine_contours))

                t_time_max  = G2_t_start(t_seg) #G2.nodes[t_seg]["t_start"]
                t_time_min  = min([G2_t_end(n) for n in predecessors]) #G2.nodes[t]["t_end"] 
                
                for t in np.arange(t_time_min, t_time_max + 1):
                    t_merges_splits_rects[state_from][t].append(t_rb_params)

            state_to = 'split'
            if t_seg in events_split_merge_mixed[doX][state_to]:
                t_successors    = events_split_merge_mixed[doX][state_to][t_seg]
                t_nodes_min_max = [G2_n_to(t_seg)] + [G2_n_from(t_succ) for t_succ in t_successors]
                t_combine_contours = []
                for t_time,*subIDs in t_nodes_min_max:
                    for t_subID in subIDs:
                        t_combine_contours.append(contours_all_dict[t_time][t_subID])
                t_rb_params = cv2.boundingRect(np.vstack(t_combine_contours))

                t_time_max  = max([G2_t_start(n) for n in t_successors])    #G2.nodes[t]["t_start"]
                t_time_min  = G2_t_end(t_seg)                               #G2.nodes[t_seg]["t_end"]
                
                for t in np.arange(t_time_min, t_time_max + 1):
                    t_merges_splits_rects[state_to][t].append(t_rb_params)
        state = 'mixed'
        for t_froms, t_tos in  events_split_merge_mixed[doX][state].items():
            t_nodes_last_from   = [G2_n_to(    t_from  )   for t_from  in t_froms  ] 
            t_nodes_first_to    = [G2_n_from(  t_to    )   for t_to    in t_tos    ]
            t_nodes_all         = t_nodes_last_from + t_nodes_first_to
            t_combine_contours = []
            for t_time,*subIDs in t_nodes_all:
                for t_subID in subIDs:
                    t_combine_contours.append(contours_all_dict[t_time][t_subID])
            t_rb_params = cv2.boundingRect(np.vstack(t_combine_contours))

            t_time_max  = max([G2_t_start(  n) for n in t_froms + t_tos])
            t_time_min  = min([G2_t_end(    n) for n in t_froms + t_tos])
                
            for t in np.arange(t_time_min, t_time_max + 1):
                t_merges_splits_rects[state][t].append(t_rb_params)

        # save edied graphs into storage
        graphs_all_dict[doX]    = [copy.deepcopy(G),copy.deepcopy(G2)]


    print(f"\n{timeHMS()}: Saving Archives...")
    with open(graphsPath,   'wb')   as handle:
        pickle.dump(graphs_all_dict,        handle) 

    with open(segmentsPath, 'wb') as handle:
        pickle.dump(trajectories_all_dict,  handle) 

    with open(contoursHulls,'wb') as handle:
        pickle.dump(contours_all_dict,      handle) 

    with open(mergeSplitEvents, 'wb') as handle:
        pickle.dump(events_split_merge_mixed,        handle) 

    print(f"\n{timeHMS()}: Saving Archives... Done")  
    a = 1
    # image export needs edited graphs
    if 1 == 1:
        print(f"\n{timeHMS()}: Processing output images...")
        binarizedMaskArr = np.load(archivePath)['arr_0']#np.load(binarizedArrPath)['arr_0']
        imgs = [None]*binarizedMaskArr.shape[0]
        relevant_times = np.arange(minTime,maxTime + 1,1)
        for k in relevant_times:
            imgs[k] = convertGray2RGB(binarizedMaskArr[k].copy())
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7; thickness = 4;
        fontScale2 = 0.5; thickness2 = 4;
        for doX, t_segments_new in tqdm(trajectories_all_dict.items()):
            for n, case in enumerate(t_segments_new):
                case    = sorted(case, key=lambda x: x[0])
                for k,subCase in enumerate(case):
                    t_time,*subIDs = subCase

                    if t_time in t_merges_splits_rects['merge']:
                        for x,y,w,h in t_merges_splits_rects['merge'][t_time]:
                            cv2.rectangle(imgs[t_time], (x, y), (x + w, y + h), (0,0,255), 2)
                            cv2.putText(imgs[t_time], 'merge', (20,20), font, fontScale2, (0,0,255), 2, cv2.LINE_AA)
                    if t_time in t_merges_splits_rects['split']:
                        for x,y,w,h in t_merges_splits_rects['split'][t_time]:
                            cv2.rectangle(imgs[t_time], (x, y), (x + w, y + h), (0,255,0), 2)
                            cv2.putText(imgs[t_time], 'split', (70,20), font, fontScale2, (0,255,0), 2, cv2.LINE_AA)
                    if t_time in t_merges_splits_rects['mixed']:
                        for x,y,w,h in t_merges_splits_rects['mixed'][t_time]:
                            cv2.rectangle(imgs[t_time], (x, y), (x + w, y + h), (0,255,255), 2)
                            cv2.putText(imgs[t_time], 'mixed', (120,20), font, fontScale2, (0,255,255), 2, cv2.LINE_AA)

                    for subID in subIDs:
                        cv2.drawContours(  imgs[t_time],   contours_all_dict[t_time], subID, cyclicColor(n), 2)
            
                    hull = cv2.convexHull(np.vstack([contours_all_dict[t_time][t_subID] for t_subID in subIDs]))
                    x,y,w,h = cv2.boundingRect(hull)
                    str_ID = f'{n}({doX})'
                    cv2.drawContours(  imgs[t_time],  [hull], -1, cyclicColor(n), 2)
                    [cv2.putText(imgs[t_time], str_ID, (x,y), font, fontScale2, clr,s, cv2.LINE_AA) for s, clr in zip([thickness2,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
                t_times_all = [t[0] for t in case]
                x = [graphs_all_dict[doX][0].nodes[t]["cx"] for t in case]
                y = [graphs_all_dict[doX][0].nodes[t]["cy"] for t in case]
                t_centroids_all = np.vstack((x,y)).T
            
                for k,subCase in enumerate(case):
                    t_time,*subIDs = subCase
                    t_k_from = max(0, k - 10)
                    t_times_sub = t_times_all[t_k_from:k]
                    pts = np.array(t_centroids_all[t_k_from:k]).reshape(-1, 1, 2).astype(int)
            
                    cv2.polylines(imgs[t_time], [pts] ,0, (255,255,255), 3)
                    cv2.polylines(imgs[t_time], [pts] ,0, cyclicColor(n), 2)
                    [cv2.circle(imgs[t_time], tuple(p), 3, cyclicColor(n), -1) for [p] in pts]

                    for subID in subIDs:
                        startPos2 = contours_all_dict[t_time][subID][-30][0] 
                        [cv2.putText(imgs[t_time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        print(f"\n{timeHMS()}: Processing output images...Done")
        print(f"{timeHMS()}: Saving {len(relevant_times)} Images...")

        #def parallel_task(x):
        #    process_id = os.getpid() 
        #    logger.info(f"Process {process_id}: {x}")
        #    return x * 2

        #if __name__ == '__main__':
        #    N = 100
        #    data = list(range(N))

        #    with Pool(processes=4) as pool:
        #        results = list(tqdm(pool.imap(parallel_task, data), total=N))
        webp_parameters = [int(cv2.IMWRITE_WEBP_QUALITY), 20]

        for k in tqdm(relevant_times):
            
            filepath = os.path.join(imageFolder_output, f"{str(k).zfill(4)}.webp")
            cv2.imwrite(filepath, imgs[k], webp_parameters)

            #cv2.imshow('a',imgs[time])
        print(f"{timeHMS()}: Saving {len(relevant_times)} Images... Done")
    
    
# for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       
a = 1

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()