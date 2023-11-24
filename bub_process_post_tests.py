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
                                                                        # one more layer will be created for image subset later.
# ========================================================================================================
# ============== SET NUMBER OF IMAGES ANALYZED =================
# ========================================================================================================
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\150 sccm' #
inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\Field OFF\Series 7\150 sccm' #
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\VFS 125 mT\Series 5\150 sccm' #
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 125 mT\Series 1\350 sccm'
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\100 sccm'
# image data subsets are controlled by specifying image index, which is part of an image. e.g image1, image2, image20, image3
intervalStart   = 1                            # start with this ID
numImages       = 1999                          # how many images you want to analyze.
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

# ========================================================================================================//
# ======================================== BUILD PROJECT FOLDER HIERARCHY ================================//
# --------------------------------------------------------------------------------------------------------//
# ------------------------------------ CREATE MAIN PROJECT, SUBPROJECT FOLDERS ---------------------------//

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

from graphs_brects import (overlappingRotatedRectangles, graphUniqueComponents)

from image_processing import (convertGray2RGB, undistort)

from bubble_params  import (centroid_area_cmomzz, centroid_area)

from graphs_general import (extractNeighborsNext, extractNeighborsPrevious, graph_extract_paths, find_paths_from_to_multi, graph_check_paths,
                            comb_product_to_graph_edges, for_graph_plots, extract_graph_connected_components, extract_clusters_from_edges,
                            find_segment_connectivity_isolated,  graph_sub_isolate_connected_components, set_custom_node_parameters, G2_set_parameters, get_event_types_from_segment_graph)

from graphs_general import (seg_t_start, seg_t_end, seg_n_from, seg_n_to, seg_edge_d, node_time, node_area, node_centroid, node_owner)

from interpolation import (interpolate_trajectory, interpolate_find_k_s, extrapolate_find_k_s, interpolateMiddle2D_2, interpolateMiddle1D_2)

from misc import (cyclicColor, closes_point_contours, timeHMS, modBR, rect2contour, combs_different_lengths, unique_sort_list, sort_len_diff_f,
                  disperse_nodes_to_times, disperse_composite_nodes_into_solo_nodes,
                  find_key_by_value, CircularBuffer, CircularBufferReverse, AutoCreateDict, find_common_intervals,
                  unique_active_segments, split_into_bins, lr_reindex_masters, dfs_pred, dfs_succ, getNodePos,
                  getNodePos2, segment_conn_end_start_points, old_conn_2_new, lr_evel_perm_interp_data, lr_weighted_sols,
                  perms_with_branches, save_connections_two_ways, save_connections_merges, save_connections_splits,
                  itertools_product_length, conflicts_stage_1, conflicts_stage_2, conflicts_stage_3, edge_crit_func,
                  two_crit_many_branches)

# ========================================================================================================
# ============== Import image files and process them, store in archive or import archive =================
# ========================================================================================================
# NOTE: mapXY IS USED TO FIX FISH-EYE DISTORTION. 

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
    # ===================================================================================================
    # ======== CROP USING A MASK (DRAW RED RECTANGLE ON EXPORTED SAMPLE IN MANUAL MASK FOLDER) ==========
    # IF MASK IS MISSING YOU CAN DRAW IT USING GUI
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

    # ===================================================================================================
    # =========================== CROP/UNSISTORT AND STORE DATA INTO ARCHIVES ===========================
    # ---------------------------------------------------------------------------------------------------

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

# =========================================================================================================
# discrete update moving average with window N, with intervcal overlap of N/2
# [-interval1-]         for first segment: interval [0,N]. switch to next window at i = 3/4*N,
#           |           which is middle of overlap. 
#       [-interval2-]   for second segment: inteval is [i-1/4*N, i+3/4*N]
#                 |     third switch 1/4*N +2*[i-1/4*N, i+3/4*N] and so on. N/2 between switches

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
        g0_contours[time]          = contours

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
        
        for k, subElems in enumerate(frame_clusters):
            key                         = tuple([time]+subElems)
            pre_rect_cluster[time][key] = cv2.boundingRect(np.vstack([rect2contour(brectDict[c]) for c in subElems]))
            pre_nodes_all.append(key)

        g0_contours_centroids[  time]   = np.zeros((len(contours),2))
        g0_contours_areas[      time]   = np.zeros(len(contours), int)
        g0_contours_hulls[      time]   = [None]*len(contours)

        for k,t_contour in enumerate(contours):
            t_hull                        = cv2.convexHull(t_contour)
            t_centroid,t_area             = centroid_area(t_hull)
            g0_contours_hulls[      time][k] = t_hull
            g0_contours_centroids[  time][k] = t_centroid
            g0_contours_areas[      time][k] = int(t_area)
        
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
    for t_areas in g0_contours_areas.values():
        t_all_areas += t_areas.tolist()

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

    # ===============================================================================================
    # =============== Refine expanded BR overlap subsections with solo contours =====================
    # ===============================================================================================
    # WHAT: recalculate overlap between bubble bounding rectangles on to time-adjecent time frames
    # WHAT: without expaning b-recs for small contours
    # WHY:  expended b-rec clustering was too rough and might have falsely connected neighbor bubbles 
    # HOW:  generate bounding rectangles for time step frame and determing overlap

    print(f'\n{timeHMS()}:({doX_s}) generating rough contour overlaps')

    # 'pre_family_nodes' contains node which are clusters of contours
    # gather all active contour IDs for each time step
    t_time_subIDs = disperse_nodes_to_times(pre_family_nodes, sort = True)
    # split them into nodes containing single ID : [(time, ID1),..]
    lr_nodes_all = disperse_composite_nodes_into_solo_nodes(pre_family_nodes, sort = True, sort_fn = sort_comp_n_fn)

    lr_edge_temporal  = [] # store overlapping connections
    # check overlap between all possible pairs of bounding rectangle on two sequential frames at time t and t + 1. 
    for t in tqdm(list(t_time_subIDs)[:-1]):
        
        if t == 0:  oldBRs = {(t, ID):cv2.boundingRect(g0_contours[t][ID]) for ID in t_time_subIDs[t]}
        else:       oldBRs = newBrs # reuse previously prev time step data.

        newBrs = {(t + 1, ID):cv2.boundingRect(g0_contours[t + 1][ID]) for ID in t_time_subIDs[t + 1]}
        
        lr_edge_temporal.extend(overlappingRotatedRectangles(oldBRs,newBrs))

    # for_graph_plots(G)    # <<<<<<<<<<<<<<
    # ===============================================================================================
    # ======================== DEFINE A NODE VIEW GRAPH, SET NODE PARAMETERS ========================
    # ===============================================================================================
    # WHAT: create a graph with nodes which represent separate contours on each frame, attach contour parameters.
    # WHY:  hold most information on graph so all important information is held in one place and is easy to retrieve.
    # NOTE: lambda functions are defined to reference G by id(), so dont redefine G because ref will not update to new obj.

    G = nx.DiGraph()
    G.add_nodes_from(lr_nodes_all)                          # add all nodes
    G.add_edges_from(lr_edge_temporal)                      # add overlap information as edges. not all nodes are inter-connected
    
    set_custom_node_parameters(G, g0_contours_hulls, G.nodes(), None, calc_hull = 0) # pre define params and init owner to None 
    
    G_time      = lambda node: node_time(       node, G)    # def functions to retrieve bubble params.
    G_area      = lambda node: node_area(       node, G)    # watch out default graph G is called by reference.  
    G_centroid  = lambda node: node_centroid(   node, G)    # if you redefine G it will use old graph values 
    G_owner     = lambda node: node_owner(      node, G)    # code uses stray node owner None, but gephi needs an int. so its -1 for end code  

    # ==============================================================================================
    # ========== DEFINE A SEGMENT (TRAJETORY ABSTRACTION) VIEW GRAPH, SET NODE PARAMETERS ==========
    # ==============================================================================================
    # WHAT: create a graph that does not hold all nodes, but represents interaction between trajectories.
    # WHY:  this graph is used to analyze connectivity between trajectories and stores most relevant info.
    # NOTE: same as G, dotn redefine G2, only update its values.
    G2 = nx.DiGraph()
    
    G2_t_start      = lambda node : seg_t_start(node, G2)   # trajectory start time
    G2_t_end        = lambda node : seg_t_end(  node, G2)  
    G2_n_start      = lambda node : seg_n_from( node, G2)   # trajectory starts with a node
    G2_n_end        = lambda node : seg_n_to(   node, G2)      
    G2_edge_dist    = lambda edge : seg_edge_d( edge, G2)   # time inteval between two trajectories

    print(f'\n{timeHMS()}:({doX_s}) Detecting frozen bubbles ... ')
   
    seg_min_length = 2     
    segments_fb = graph_extract_paths(G, min_length = seg_min_length)

    #for_graph_plots(G, segments_fb, show = True)

    # ===============================================================================================
    # ========================= FIND OBJECTS/BUBBLES THAT ARE FROZEN IN PLACE =======================
    # ===============================================================================================
    # WHAT: find which chains dont move
    # WHY:  these may be semi-static arftifacts or very small bubbles which stick to walls
    # HOW:  1) check displacement for all chains 
    # HOW:  2) check if frozen chains are continuation of same artifacts
    # HOW:  3) isolate faulty nodes from analysis (from graph)
    
    # 1)    find displacement norms if mean displ is small
    # 1)    remember average centroid -> frozen object 'lives' around this point
    # 1)    remember mean area and area stdev
    fb_radiuss = 5
    fb_mean_centroid_d = {}
    fb_area_mean_stdev_d = {}

    for t,t_nodes in enumerate(segments_fb):

        t_traj      = np.array([G_centroid(t_node) for t_node in t_nodes])
        t_displ_all = np.linalg.norm(np.diff(t_traj, axis = 0), axis = 1)

        if np.mean(t_displ_all) <= fb_radiuss:
            fb_mean_centroid_d[t] = np.mean(t_traj, axis = 0)
            t_areas = np.array([G_area(t_node) for t_node in t_nodes])
            fb_area_mean_stdev_d[t] = (np.mean(t_areas),np.std(t_areas))

    # 2)    generate all possible pairs of frozen segments
    # 2)    run though each pair and check if their mean centroids are close
    # 2)    pairs that are close are gathered on a graph and clusters extracted
    fb_edges_test = list(itertools.combinations(fb_mean_centroid_d, 2))
    fb_edges_close = []
    for a,b in fb_edges_test:
        t_dist = np.linalg.norm(fb_mean_centroid_d[a] - fb_mean_centroid_d[b])
        if t_dist <= fb_radiuss:
            fb_edges_close.append((a,b))
    
    fb_segment_clusters = extract_clusters_from_edges(fb_edges_close)   # [[1,2,3],[4,5],[6]]

    
    # 3a)   if there are times between frozen prior bubble segment end and next segment start 
    # 3a)   we have to find if some nodes in these time interavls (holes) are also frozen
    # 3a)   for it we need to check area stats

    fb_hole_info       = {}#{'times': {},'area_min_max': {}}

    for t_cluster in fb_segment_clusters:
        for t_edge in zip(t_cluster[:-1], t_cluster[1:]): # [1,2,3] -> [(1,2),(2,3)]. should work if ordered correctly. otherwise a weak spot.

            (t_from,t_to) = t_edge

            fb_hole_info[t_edge] = {'times':(G_time(segments_fb[t_from][-1]) + 1 , G_time(segments_fb[t_to][0]) - 1)}

            # set lower and top bounds on area. for bigger bubble + 5 * std and for smaller -5 * std
            t_area_mean_1, t_area_std_1 = fb_area_mean_stdev_d[t_from]
            t_area_mean_2, t_area_std_2 = fb_area_mean_stdev_d[t_to]
            if t_area_mean_1 > t_area_mean_2:
                t_area_min_max = (t_area_mean_2 - 5*t_area_std_2, t_area_mean_1 + 5*t_area_std_1)
            else:
                t_area_min_max = (t_area_mean_1 - 5*t_area_std_1, t_area_mean_2 + 5*t_area_std_2)

            fb_hole_info[t_edge]['area_min_max'   ] = t_area_min_max
            
    # 3b)   extract stray nodes that have times between connected frozen segments
    # 3b)   check if stray node's have similar centroid as avergae between two segments
    # 3b)   check if stray node's area is within a threshold
    fb_remove_nodes_inter = []
    for t_edge in fb_hole_info:
        t_min,t_max             = fb_hole_info[t_edge]['times'         ]
        t_area_min,t_area_max   = fb_hole_info[t_edge]['area_min_max'  ]
        t_nodes_active = [t_node for t_node in G.nodes if t_min <= G_time(t_node) <= t_max and G_owner(t_node) is None]
        (t_from,t_to) = t_edge
        t_centroid_target = 0.5*(fb_mean_centroid_d[t_from] + fb_mean_centroid_d[t_to])
        for t_node in t_nodes_active:
            t_dist = np.linalg.norm(t_centroid_target - G_centroid(t_node))
            if t_dist <= fb_radiuss and (t_area_min <= G_area(t_node) <= t_area_max):
                fb_remove_nodes_inter.append(t_node)

    # 3c)   remove frozen stray nodes form a graph
    # 3c)   remove frozen segment's nodes from a graph
    t_remove_nodes = []
    G.remove_nodes_from(fb_remove_nodes_inter)
    for t_segment_IDs in fb_segment_clusters:
        for t_segment_ID in t_segment_IDs:
            t_nodes = segments_fb[t_segment_ID]
            G.remove_nodes_from(t_nodes)
            t_remove_nodes.extend(t_nodes)
    # ===============================================================================================
    # =================== REDEFINE SEGMENTS AFTER FROZEN BUBBLES ARE REMOVED ========================
    # ===============================================================================================
    # WHY:  frozen nodes were stripped from graph, have to recalc connectivity and segments
    print(f'\n{timeHMS()}:({doX_s}) Detecting frozen bubbles ... DONE')
    segments2 = graph_extract_paths(G, min_length = seg_min_length)

    if len(segments2) == 0: continue    # no segments, go to next doX
    
    # store information on segment view graph and update node ownership
    for t_owner,t_segment in enumerate(segments2):
        G2_set_parameters(G, G2, t_segment, t_owner)
        for t_node in t_segment:
            G.nodes[t_node]["owner"] = t_owner
  
    #if doX >= 0:
    #    for_graph_plots(G, segs = segments2)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Determining connectivity between segments... ')
    # ===============================================================================================
    # ========================== FIND CONNECTIVITY BETWEEN SEGMENTS =================================
    # ===============================================================================================
    # WHAT: find if segments are conneted pairwise: one segment ends -> other begins
    # WHY:  its either one bubble which has optically disrupted trajectory or merge/split of multiple bubbles 
    # HOW:  read 'graphs_general:graph_check_paths()' comments.
    # HOW:  in short: 1) find all start-end pairs that exist with time interval of length 'lr_maxDT'
    # HOW:  check if there is a path between two segments. first by custom funciton, then nx.has_path()

    lr_maxDT = 60   # search connections with this many time steps.

    t_has_holes_report = {}
    G2 = graph_check_paths(G, G2, G2, lr_maxDT, t_has_holes_report)

    print(f'\n{timeHMS()}:({doX_s}) Paths that have failed hole test: {t_has_holes_report}')
     
    print(f'\n{timeHMS()}:({doX_s}) Working on one-to-one (121) segment connections ... ')
    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # === PROCESS SEGMENT-SEGMENT CONNECTIONS THAT ARE CONNECTED ONLY TOGETHER (ONE-TO-ONE; 121) ====
    # ===============================================================================================
    # WHY:  long graph segment (which is a chain of one contour nodes) is likely a single bubble
    # WHY:  raising without merges and splits. sometimes, due to optical lighting artifacts or other 
    # WHY:  artifacts, next to bubble pops out an unwanted object, or its a same bubble that visually
    # WHY:  splits and recombines. that means that solo contour trajectory is interrupted by a 'hole'
    # WHY:  in graph, which is actually an 'explositon' of connections into 'stray' nodes and 'collapse' 
    # WHY:  back to a solo trajectory.
    # WHY:  these holes can be easily patched, but first they have to be identified
    # WHY:  if this happens often, then instead of analyzing such 'holes' locally, we can analyze 
    # WHY:  whole trajectory with all holes and patch them more effectively using longer bubble history.

    # FLOWCHART OF PROCESSING 121 (one-to-one) SEGMENTS:
    #
    # (121.A) isolate 121 connections
    # (121.B) extract inter-segment nodes as time slices : {t1:[*subIDs],...}
    # (121.C) detect which cases are zero-path ZP. its 121 that splits into segment and a node. byproduct of a method.
    # (121.D) resolve case with ZP
    # (121.E) find which 121 form a long chain- its an interrupted solo bubble's trejctory.
    #           missing data in 'holes' is easier to interpolate from larger chains 
    # DEL-(121.F) chain edges may be terminated by artifacts - 'fake' events, find them and refine chain list elements.
    # (121.G) hole interpolation
    # (121.H) prep data (area, centroids) for all subID combinations
    # (121.I) generate permutations from subIDs
    # (121.K) generate all possible evolutions of subIDs perms though time. 
    # (121.J) find best fit evolution by minimizing area and isplacement criterium
    # (121.L) save data into graps and storage.

    # ===============================================================================================
    # === extract segment-segment connections that are connected only together (one-to-one; 121) ====
    # ===============================================================================================
    #
    # ===============================================================================================
    # === DETERMINE EDGE TYPES OF EVENTS (PREP FOR FAKE EVENT DETECTION ON EDGES OF 121 CHAINS) =====
    # ===============================================================================================
    # WHAT: extract all 121 type connections. splits merges are not important now.
   
    G2_dir, lr_ms_edges_main = get_event_types_from_segment_graph(G2)

    t_conn_121 = lr_ms_edges_main['solo']
    # ===================================================================================================
    # ==================== SLICE INTER-SEGMENT (121) NOTE-SUBID SPACE W.R.T TIME ========================
    # ===================================================================================================
    # WHAT: 121 segments are connected via stay (non-segment) nodes. extract them and reformat into  {TIME1:[*SUBIDS_AT_TIME1],...}
    # WHY:  at each time step bubble may be any combination of SUBIDS_AT_TIMEX. most likely whole SUBIDS_AT_TIMEX. but possibly not

    lr_big121s_perms_pre = {}
    for t_from, t_to in t_conn_121:
            
        t_node_from, t_node_to = G2_n_end(t_from)   , G2_n_start(t_to)
        t_time_from, t_time_to = G_time(t_node_from), G_time(t_node_to)

        # isolate stray nodes on graph at time interval between two connected segments
        t_dist      = G2.edges[(t_from,t_to)]['dist']
        if t_dist   == 2:# zero-path connections have to include edge times.
            t_nodes_keep = [node for node, time in G.nodes(data='time') if t_time_from <= time <= t_time_to]
        else:
            #t_nodes_keep    = [node for node in G.nodes() if t_time_from < G_time(node) < t_time_to and G_owner(node) is None] 
            t_nodes_keep = [node for node, time in G.nodes(data='time') if t_time_from < time < t_time_to and G_owner(node) is None]
            t_nodes_keep.extend([t_node_from,t_node_to])
            
        # extract connected nodes using depth search. not sure why i switched from connected components. maybe i used it wrong and it was slow
        t_connected_nodes   = set()
        g_limited           = G.subgraph(t_nodes_keep)
        dfs_succ(g_limited, t_node_from, time_lim = t_time_to + 1, node_set = t_connected_nodes)

        # probly something needed for zero path. like nodes parallel to t_node_from.
        t_nodes_after_from_end = [node for node in t_connected_nodes if G_time(node) == t_time_from + 1]

        t_predecessors = set()
        for t_node in t_nodes_after_from_end:
            t_preds = list(g_limited.predecessors(t_node))
            t_predecessors.update(t_preds)
        t_connected_nodes.update(t_predecessors)
        
        lr_big121s_perms_pre[(t_from,t_to)] = disperse_nodes_to_times(t_connected_nodes, sort = True)
        a = 1

    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===================================================================================================
    # ============================ SEPARATE 121 PATHS WITH ZERO INTER LENGTH ============================
    # ===================================================================================================
    # WHAT:     121 connection with no internal nodes = nothing to restore. segment+ node -> split case
    # HOW:      absorb split/merge node into segment and stick together
    # NOTE:     dont particularly like how its done
    lr_zp_redirect = {tID: tID for tID in range(len(segments2))}
    if 1 == 1:
        t_conn_121_zero_path = []
        for t_node in t_conn_121:
            if len(lr_big121s_perms_pre[t_node]) == 2:
                t_t_min = min(lr_big121s_perms_pre[t_node])
                t_t_max = max(lr_big121s_perms_pre[t_node])
                if len(lr_big121s_perms_pre[t_node][t_t_min]) > 1 or len(lr_big121s_perms_pre[t_node][t_t_max]) > 1:
                    t_conn_121_zero_path.append(t_node)

        
        for t_conn in t_conn_121_zero_path: 
            t_dict = lr_big121s_perms_pre[t_conn]
            t_from, t_to        = t_conn                                                   
            t_from_new          = lr_zp_redirect[t_from]
            t_times             = [t for t,t_subIDs in t_dict.items() if len(t_subIDs) > 1]
            if len(t_times) > 1:
                if 'zero_path' not in issues_all_dict[doX]: issues_all_dict[doX]['zero_path'] = []
                issues_all_dict[doX]['zero_path'].append(f'multiple times: {t_dict}')
                lr_big121s_perms_pre.pop(t_conn,None)
                t_conn_121.remove(t_conn)
                continue
            
            t_nodes_composite       = [tuple([t] + t_dict[t]) for t in t_times]                 # join into one node (composite)
            t_nodes_solo            = [(t,t_subID) for t in t_times for t_subID in t_dict[t]]   # get solo contour nodes

            # composite node contains solo node on either the end of a 'from' segment on at the start of a 'to' segment
            # that node is contained in t_nodes_solo.
            t_nodes_prev            =   [t_node for t_node in segments2[t_from_new] if t_node not in t_nodes_solo]  # gather up until composite node
            t_nodes_next            =   [t_node for t_node in segments2[t_to]       if t_node not in t_nodes_solo]  # add post composite 

            t_edges                 =  [    (t_nodes_prev[-1]      , t_nodes_composite[0]   ),                      # weird note: if segments2[t_from_new] changes,
                                            (t_nodes_composite[0]  , t_nodes_next[0]        )   ]                   # t_nodes_prev also changes. so its not a copy but ref

            segments2[t_from_new]   =   t_nodes_prev                                                               
            segments2[t_from_new].extend(t_nodes_composite)                                                     
            segments2[t_from_new].extend(t_nodes_next)
    
            
            
            # modify graph: remove solo IDs, add new composite nodes. add parameters to new nodes
            G.remove_nodes_from(t_nodes_solo) 
            G.add_edges_from(t_edges)
            # composite not is new and does not exist on G. add it and its params
            set_custom_node_parameters(G, g0_contours, t_nodes_composite, t_from_new, calc_hull = 1)

            for t_node in t_nodes_next: 
                    G.nodes[t_node]["owner"] =  t_from_new

            # inherit G2 params. time end, node end and successors:
            G2.nodes()[t_from_new]['node_end']  = G2_n_end(t_to)
            G2.nodes()[t_from_new]["t_end"]     = G2_t_end(t_to) 

            t_successors   = G2.successors(t_to)
            t_edges_next = [(t_from_new,t_succ) for t_succ in t_successors]
            G2.add_edges_from(t_edges_next)
            
            # clear storage of 't_to'
            G2.remove_node(t_to)
            segments2[t_to] = []
            # store reference (inheritance)
            lr_zp_redirect[t_to] = t_from_new
            print(f'zero path: joined segments: {t_conn}')
    
    t_conn_121              = lr_reindex_masters(lr_zp_redirect, t_conn_121, remove_solo_ID = 1)
    #lr_conn_edges_merges    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_merges   )
    #lr_conn_edges_splits    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits   )
    #lr_conn_edges_splits_merges_mixed = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits_merges_mixed   )
    #temp = {}
    #for t_state, t_dict in lr_ms_edges_brnch.items():
    #    temp[t_state] = {}
    #    for a,b in t_dict.items():
    #        c, d = lr_zp_redirect[a], lr_zp_redirect[b]
    #        if c != d: temp[t_state][c] = d
    #lr_ms_edges_brnch = temp
    a = 1

    #'fk_event_branchs' inherits 'lr_ms_edges_main', but 'solo' and 'mixed' are not used
    #temp = {}
    #for t_state in ['merge','split']:
    #    temp[t_state] = {}
    #    for t_ID, t_subIDs in lr_ms_edges_main[t_state].items():
    #        t_ID_new = lr_zp_redirect[t_ID]
    #        t_subIDs_new = [lr_zp_redirect[t] for t in t_subIDs]
    #        temp[t_state][t_ID_new] = t_subIDs_new
    #lr_ms_edges_main = temp

    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
    # ===============================================================================================
    # ============================== FIND CHAINS OF 121 CONNECTED SEGMENTS ==========================
    # ===============================================================================================
    # WHAT: find which segments are continiously connected via 121 connection to other segments
    # WHY:  they are most likely single bubble which has an optical split.
    # HOW:  use 121 edges and condence to a graph, extract connected compontents
    # NOTE: same principle as in 'fb_segment_clusters'
    
    lr_big_121s_chains = extract_clusters_from_edges(t_conn_121)

    print(f'\n{timeHMS()}:({doX_s}) Working on joining all continious 121s together: {lr_big_121s_chains}')
     
    if 1 == 1:
        
        t_big121s_edited = lr_big_121s_chains.copy()
        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ===============================================================================================
        # ==================== 121 CHAINS: INTERPOLATE DATA IN HOLES OF LONG CHAINS =====================
        # ===============================================================================================
        # WHAT: we have chains of segments that represent solo bubble. its at least 2 segments (1 hole). interpolate data
        # WHY:  if there are more than 2 segments, we will have more history and interpolation will be of better quality
        # HOW:  scipy interpolate 
        
        lr_big121s_edges_relevant = []
        lr_big121s_interpolation        = defaultdict(dict)

        for t,t_subIDs in enumerate(t_big121s_edited):
            if t_subIDs != None: 
                # prepare data: resolved  time steps, centroids, areas G2
                t_temp_nodes_all    = sum(  [segments2[t_subID]   for t_subID in t_subIDs         ],[])
                t_temp_times        =       [G_time(t_node)       for t_node  in t_temp_nodes_all ]
                t_temp_areas        =       [G_area(t_node)       for t_node  in t_temp_nodes_all ]
                t_temp_centroids    =       [G_centroid(t_node)   for t_node  in t_temp_nodes_all ]
                
                # generate time steps in holes for interpolation
                t_conns_times_dict = {}
                for (t_from, t_to) in zip(t_subIDs[:-1], t_subIDs[1:]):
                    t_conns_times_dict[(t_from, t_to)]  = range(G2_t_end(t_from) + 1, G2_t_start(t_to), 1)

                lr_big121s_edges_relevant.extend(t_conns_times_dict.keys())
                t_times_missing_all = []; [t_times_missing_all.extend(times) for times in t_conns_times_dict.values()]
        
                # interpolate composite (long) parameters 
                t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, s = 15, debug = 0, aspect = 'equal', title = t_subIDs)
                t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_subIDs)
                # form dict = {time:centroid} for convinience
                t_interpolation_centroids_1 = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_centroids_0)}
                t_interpolation_areas_1     = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_areas_0)}
                # save data with t_conns keys
                for t_conn,t_times_relevant in t_conns_times_dict.items():
                    t_conn_new = old_conn_2_new(t_conn,lr_zp_redirect)
                    t_centroids = [t_centroid   for t_time, t_centroid   in t_interpolation_centroids_1.items()  if t_time in t_times_relevant]
                    t_areas     = [t_area       for t_time, t_area       in t_interpolation_areas_1.items()      if t_time in t_times_relevant]

                    lr_big121s_interpolation[t_conn_new]['centroids'] = np.array(t_centroids)
                    lr_big121s_interpolation[t_conn_new]['areas'    ] = t_areas
                    lr_big121s_interpolation[t_conn_new]['times'    ] = t_times_relevant
                    
        # ===================================================================================================
        # ================ 121 CHAINS: CONSTRUCT PERMUTATIONS FROM CLUSTER ELEMENT CHOICES ==================
        # ===================================================================================================
        # WHAT: generate different permutation of subIDs for each time step.
        # WHY:  Bubble may be any combination of contour subIDs at a given time. should consider all combs as solution
        # HOW:  itertools combinations of varying lenghts

        print(f'\n{timeHMS()}:({doX_s}) Computing contour element permutations for each time step...')
        
        # if "lr_zp_redirect" is not trivial, drop resolved zp edges
        lr_big121s_perms = {}
        lr_big121s_perms_pre_old = copy.deepcopy(lr_big121s_perms_pre)
        lr_big121s_perms_pre = {}
        # lr_big121s_perms_pre = {(t_from,t_to):{time_1:*subIDs1,...}}
        for t_conn, t_times_subIDs in lr_big121s_perms_pre_old.items():
            t_from_new, t_to_new = old_conn_2_new(t_conn,lr_zp_redirect)
            if t_from_new != t_to_new and (t_from_new,t_to_new) in lr_big121s_edges_relevant:
                lr_big121s_perms[(t_from_new,t_to_new)] = {}
                lr_big121s_perms_pre[(t_from_new,t_to_new)] = t_times_subIDs
                for t_time, t_subIDs in t_times_subIDs.items():
                    lr_big121s_perms[(t_from_new,t_to_new)][t_time] = combs_different_lengths(t_subIDs)

        lr_big121s_conn_121 = lr_big121s_edges_relevant
        # ===============================================================================================
        # =========== 121 CHAINS: PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ===========
        # ===============================================================================================
        # WHY: these will be reused alot in next steps, store beforehand
        print(f'\n{timeHMS()}:({doX_s}) Calculating parameters for possible contour combinations...')
        
        lr_big121s_perms_areas      = {}
        lr_big121s_perms_centroids  = {}
        lr_big121s_perms_mom_z      = {}
            
        for t_conn, t_times_perms in lr_big121s_perms.items():
            lr_big121s_perms_areas[     t_conn]  = {t:{} for t in t_times_perms}
            lr_big121s_perms_centroids[ t_conn]  = {t:{} for t in t_times_perms}
            lr_big121s_perms_mom_z[     t_conn]  = {t:{} for t in t_times_perms}
            for t_time,t_perms in t_times_perms.items():
                for t_perm in t_perms:
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
                    t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
                    lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
                    lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
                    lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z

        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ===============================================================================================
        # ========== 121 CHAINS: CONSTUCT UNIQUE EVOLUTIONS THROUGH CONTOUR PERMUTATION SPACE ===========
        # ===============================================================================================
        # WHAT: using subID permutations at missing time construct choice tree. 
        # WHY:  each branch represents a bubbles contour ID evolution through unresolved intervals.
        # HOW:  either though itertools product or, if number of branches is big, dont consider
        # HOW:  branches where area changes more than a set threshold. 
        # -----------------------------------------------------------------------------------------------
        # NOTE: number of branches (evolutions) from start to finish may be very large, depending on 
        # NOTE: number of time steps and contour permutation count at each step. 
        # NOTE: it can be calculated as 't_branches_count' = len(choices_t1)*len(choices_t2)*...
        # case 0)   if count is small, then use combinatorics product function.
        # case 1a)  if its large, dont consider evoltions where bubble changes area rapidly from one step to next.
        # case 1b)  limit number of branches retrieved by 't_max_paths'. we can leverage known assumption to
        #           get better branches in this batch. assumption- bubbles most likely include all 
        #           subcontours at each time step. Transitions from large cluster to large cluster has a priority.
        #           if we create a graph with large to large cluster edges first, pathfinding method
        #           will use them to generate first evolutions. so you just have to build graph in specific order.
        # case 2)   optical splits have 121 chains in inter-segment space. these chains are not atomized into
        #           solo nodes, but can be included into evolutions as a whole. i.e if 2 internal chains are
        #           present, you can create paths which include 1st, 2nd on both branches simultaneously.
        print(f'\n{timeHMS()}:({doX_s}) Generating possible bubble evolution paths from prev combinations...')

        lr_big121s_perms_cases = {}
        lr_big121s_perms_times = {}
        lr_drop_huge_perms = []
        for t_conn, t_times_perms in lr_big121s_perms.items():
            func = lambda edge : edge_crit_func(t_conn,edge, lr_big121s_perms_areas, 2)

            t_conn_new = old_conn_2_new(t_conn,lr_zp_redirect)
            t_values = list(t_times_perms.values())
            t_times = list(t_times_perms.keys())
            t_branches_count = itertools_product_length(t_values) # large number of paths is expected
            t_max_paths = 5000
            if t_branches_count >= t_max_paths:
                # 1a) keep only tranitions that dont change area very much
                t_choices = [[(t_time,) + t_perm for t_perm in t_perms] for t_time,t_perms in zip(t_times,t_values)]
                edges, nodes_start, nodes_end = comb_product_to_graph_edges(t_choices, func)

                if len(nodes_end) == 0: # finding paths will fail, since no target node
                    nodes_end.add(t_choices[-1][0])   # add last, might also want edges to that node, since they have failed
                    edges.extend(list(itertools.product(*t_choices[-2:])))
                
                # 1b) resort edges for pathfining graph: sort by large to large first subIDs first: ((1,2),(3,4)) -> ((1,2),(3,))
                # 1b) for same size sort by cluster size uniformity e.g ((1,2),(3,4)) -> ((1,2,3),(4,)) 
                sorted_edges = sorted(edges, key=sort_len_diff_f, reverse=True) 
                sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, G = None, edges = sorted_edges, only_subIDs = True, max_paths = t_max_paths - 1)
                # 'fail' can be either because 't_max_paths' is reached or there is no path from source to target
                if fail == 'to_many_paths':    # add trivial solution where evolution is transition between max element per time step number clusters                                           
                    seq_0 = list(itertools.product(*[[t[-1]] for t in t_values] ))
                    if seq_0[0] not in sequences: sequences = seq_0 + sequences # but first check. in case of max paths it should be on top anyway.
                elif fail == 'no_path': # increase rel area change threshold
                    func = lambda edge : edge_crit_func(t_conn,edge, lr_big121s_perms_areas, 5)
                    edges, nodes_start, nodes_end = comb_product_to_graph_edges(t_choices, func)
                    sorted_edges = sorted(edges, key=sort_len_diff_f, reverse=True) 
                    sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, G = None, edges = sorted_edges, only_subIDs = True, max_paths = t_max_paths - 1)

            # 0) -> t_branches_count < t_max_paths. use product
            else:
                sequences = list(itertools.product(*t_values))
                
            if len(sequences) == 0: # len = 0 because second pass of rel_area_thresh has failed.
                sequences = []      # since there is no path, dont solve this conn
                t_times = []
                lr_drop_huge_perms.append(t_conn)

            lr_big121s_perms_cases[t_conn_new] = sequences
            lr_big121s_perms_times[t_conn_new] = t_times

        lr_big121s_conn_121 = [t_conn for t_conn in lr_big121s_conn_121 if t_conn not in lr_drop_huge_perms]
        [lr_big121s_perms_cases.pop(t_conn,None) for t_conn in lr_drop_huge_perms]
        [lr_big121s_perms_times.pop(t_conn,None) for t_conn in lr_drop_huge_perms]
        print(f'\n{timeHMS()}:({doX_s}) Dropping huge permutations: {lr_drop_huge_perms}') 
        # ===============================================================================================
        # ========= 121 CHAINS: FIND BEST FIT EVOLUTIONS OF CONTOURS THOUGH HOLES (TIME-WISE) ===========
        # ===============================================================================================
        # WHAT: calculate how area and centroids behave for each evolution
        # WHY:  evolutions with least path length and area changes should be right ones pregenerated data
        # HOW:  for each conn explore permutations fo each evolution. generate trajectory and areas from.
        # HOW:  evaluate 4 criterions and get evolution indicies where there crit are the least:
        # HOW:  1) displacement error form predicted trajetory (interpolated data)
        # HOW:  2) sum of displacements for each evolution = total path. should be minimal
        # HOW:  3) relative area chainges its mean and stdev values (least mean rel area change, least mean stdev)
        # HOW:  4) same with moment z, which is very close to 3) except scaled. (although min result is different) 
        # HOW:  currently you get 4 bins with index of evolution with minimal crit. ie least 1) is for evol #1
        # HOW:  least #2 is for evol 64, least #3 is for #1 again,.. etc
          
        print(f'\n{timeHMS()}:({doX_s}) Determining evolutions thats are closest to interpolated missing data...')
        # NOTE: >>> this needs refactoring, not internals, but argument data management <<<<
        t_temp_centroids = {t_conn:t_dict['centroids'] for t_conn,t_dict in lr_big121s_interpolation.items()}
        t_args = [lr_big121s_conn_121, lr_big121s_perms_cases,t_temp_centroids,lr_big121s_perms_times,
                lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

        t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)

        # ======================= FROM MULTIPLE THRESHOLDS GET WINNING ONE ==================================
        # NOTE: dont like this approach. you should evaluate actual values/errors, rather then indicies.
        # NOTE: its easy if multiple indicies are in winning bins.
        t_weights   = [1,1.5,0,1]
        t_sols      = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
        lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(lr_big121s_conn_121,t_weights, t_sols, lr_big121s_perms_cases )

        lr_121chain_redirect = lr_zp_redirect.copy()
        t_zp_redirect_inheritance = {a:b for a,b in lr_zp_redirect.items() if a != b}
        print(f'\n{timeHMS()}:({doX_s}) Saving results for restored parts of big 121s')
        t_big121s_edited_clean      = [t for t in t_big121s_edited if t is not None]

        t_segments_new = copy.deepcopy(segments2)
        for t_conn in lr_big121s_conn_121:
            t_from, t_to = t_conn
            t_from_new, t_to_new = old_conn_2_new(t_conn,lr_121chain_redirect)
            print(f'edge :({t_from},{t_to}) or = {G2_n_end(t_from_new)}->{G2_n_start(t_to_new)}')  
            save_connections_two_ways(t_segments_new, lr_big121s_perms_pre[t_conn], t_from,  t_to, G, G2, lr_121chain_redirect, g0_contours)

        # zp relations (edges) were not used in saving 121s. so they were not relinked.
        for t_slave, t_master in t_zp_redirect_inheritance.items():
            lr_121chain_redirect[t_slave] = lr_121chain_redirect[t_master]
        # at this time some segments got condenced into big 121s. right connection might be changed.

        
        #lr_time_active_segments = defaultdict(list)
        #for t_segment_index, t_segment_nodes in enumerate(t_segments_new):
        #    for t_time in [G_time(node) for node in t_segment_nodes]:
        #        lr_time_active_segments[t_time].append(t_segment_index)
        ## sort keys in lr_time_active_segments
        #lr_time_active_segments = {t:lr_time_active_segments[t] for t in sorted(lr_time_active_segments.keys())}

        #for_graph_plots(G, segs = t_segments_new) 
        print(f'\n{timeHMS()}:({doX_s}) Working on real and fake events (merges/splits):\nPreparing data...')
        # ===============================================================================================
        # ============ RECALCULATE MERGE/SPLIT/MIXED EVENTS (FOR ACTUAL EVENT PROCESSING) ===============
        # ===============================================================================================
        # NOTE: same as process before 121 recovery. except now mixed type is calculated.
        # NOTE: describtion is in "get_event_types_from_segment_graph" body/

        # for_graph_plots(G, segs = t_segments_new, suptitle = f'{doX}_before', show = True)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if 1 == 1: 

            G2_dir, lr_ms_edges_main = get_event_types_from_segment_graph(G2)

            # -----------------------------------------------------------------------------------------------
            # ------ PREPARE DATA INTO USABLE FORM ----
            # -----------------------------------------------------------------------------------------------

            t_merge_real_to_ID_relevant = []
            t_split_real_from_ID_relevant = []

            # ----------------------------------------------------------------------------------------------
            # ------ GENERATE INFORMATON ABOUT REAL AND FAKE MERGE/SPLIT EVENTS----
            # ----------------------------------------------------------------------------------------------
            # IDEA: for real events gather possible choices for contours for each time step during merge event (t_perms)
            # IDEA: for fake events gather evolutions of contours which integrate fake branches and stray nodes (t_combs)
            # WORKFLOW: take master event ID- either node into which merge happens or node from which spit happens
            # WORKFLOW: real: find branches, gather nodes between branches and master, find all options for time steps
            # WORKFLOW: fake: determine branches and their parents, gather options that go though branch parent- master path
            # NOTE: everything is held in one dictionary, but keys and contents are different
            print('Generating information for:')
            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            t_event_start_end_times = {'merge':{}, 'split':{}, 'mixed':{}}

            # ------------------------------------- REAL MERGE/SPLIT -------------------------------------
            t_state = 'merge'
            for t_to in lr_ms_edges_main[t_state]: 
                t_predecessors              = list(G2.predecessors(t_to))# G2_t_start G2_t_end
                t_predecessors              = [lr_121chain_redirect[t] for t in G2.predecessors(t_to)] # update if inherited
                t_predecessors_times_end    = {t:G2_t_end(t) for t in t_predecessors}
                t_t_to_start                = G2_t_start(t_to)
                t_times                     = {t_ID: np.arange(t_t_from_end, t_t_to_start + 1, 1)
                                                for t_ID,t_t_from_end in t_predecessors_times_end.items()}
                t_t_from_min = min(t_predecessors_times_end.values())    # take earliest branch time

                if t_t_to_start - t_t_from_min <= 1: continue            # no time steps between from->to =  nothing to resolve

                # pre-isolate graph segment where all sub-events take place
                t_node_to_first = G2_n_start(t_to)  #t_segments_new[t_to][0]
                t_nodes_keep    = [node for node in G.nodes() if t_t_from_min < G_time(node) < t_t_to_start and G_owner(node) is None] 
                t_nodes_keep.append(t_node_to_first)

                t_subgraph = G.subgraph(t_nodes_keep)   

                t_dfs_sol = set()
                dfs_pred(t_subgraph, t_node_to_first, time_lim = t_t_from_min , node_set = t_dfs_sol)
                t_node_subIDs_all = disperse_nodes_to_times(t_dfs_sol) # reformat sol into time:subIDs
                t_node_subIDs_all = {t:sorted(t_node_subIDs_all[t]) for t in sorted(t_node_subIDs_all)}
                
                t_active_IDs = {t:[] for t in np.arange(t_t_from_min + 1, t_t_to_start)}
                for t_from, t_times_all in t_times.items():
                    for t_time in t_times_all[1:-1]: # remove start-end points
                        t_active_IDs[t_time].append(t_from)

                t_event_start_end_times[t_state][t_to] = {
                                                            't_start'       :   t_predecessors_times_end,
                                                            'branches'      :   t_predecessors,
                                                            't_end'         :   t_t_to_start,
                                                            't_times'       :   t_times,
                                                            't_perms'       :   {},
                                                            't_subIDs'      :   t_node_subIDs_all,
                                                            't_active_IDs'  :   t_active_IDs
                                                            }

                t_merge_real_to_ID_relevant.append(t_to)
                
            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            t_state = 'split'
            for t_from in lr_ms_edges_main[t_state]:
                t_from_new          = lr_121chain_redirect[t_from]
                t_from_successors   = list(G2.successors(t_from_new))
                t_successors_times  = {t_to:G2_t_start(t_to) for t_to in t_from_successors} 
                t_t_start           = G2_t_end(t_from_new)
                t_times             = {t:np.arange(t_t_start, t_t_end + 1, 1) for t,t_t_end in t_successors_times.items()}
                
                t_node_from = G2_n_end(t_from_new)      #t_segments_new[t_from_new][-1]
                # pre-isolate graph segment where all sub-events take place
                t_t_to_max = max(t_successors_times.values())

                if t_t_to_max - t_t_start <= 1: continue

                t_nodes_keep    = [node for node in G.nodes() if t_t_start < G_time(node) < t_t_to_max and G_owner(node) is None]
                t_nodes_keep.append(t_node_from)
                t_subgraph      = G.subgraph(t_nodes_keep)   # big subgraph

                t_dfs_sol           = set()
                dfs_succ(t_subgraph, t_node_from, time_lim = t_t_to_max , node_set = t_dfs_sol)
                t_node_subIDs_all   = disperse_nodes_to_times(t_dfs_sol) # reformat sol into time:subIDs
                t_node_subIDs_all   = {t:sorted(t_node_subIDs_all[t]) for t in sorted(t_node_subIDs_all)}

                t_active_IDs = {t:[] for t in np.arange(t_t_to_max -1, t_t_start, -1)} #>>> reverse for reverse re

                for t_from, t_times_all in t_times.items():
                    for t_time in t_times_all[1:-1]: # remove start-end points
                        t_active_IDs[t_time].append(t_from)

                t_event_start_end_times[t_state][t_from_new] = {
                                                                't_start'       :   t_t_start,
                                                                'branches'      :   t_from_successors,
                                                                't_end'         :   t_successors_times,
                                                                't_times'       :   t_times,
                                                                't_perms'       :   {},
                                                                't_subIDs'      :   t_node_subIDs_all,
                                                                't_active_IDs'  :   t_active_IDs
                                                                }

                t_split_real_from_ID_relevant.append(t_from_new)

            print(f'\n{timeHMS()}:({doX_s}) Real merges({lr_ms_edges_main["merge"]})/splits({lr_ms_edges_main["split"]})... Done')

            
            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            t_state = 'mixed'
            for t_from_all,t_to_all in lr_ms_edges_main[t_state].items():
                t_from_all_new           = tuple([lr_121chain_redirect[t]   for t       in t_from_all       ])
                t_predecessors_times_end = {t:      G2_t_end(t      )       for t       in t_from_all_new   }
                t_successors_times_start = {t_to:   G2_t_start(t_to )       for t_to    in t_to_all         }
                t_target_nodes           = {t:      G2_n_start(t    )       for t       in t_to_all         } #t_segments_new[t][0]
                t_from_nodes             = {t:      G2_n_end(t      )       for t       in t_from_all_new   }
                # pre-isolate graph segment where all sub-events take place
                t_t_from_min    = min(t_predecessors_times_end.values())    # take earliest branch time
                t_t_to_max      = max(t_successors_times_start.values()) 
                t_times_all     = np.arange(t_t_from_min, t_t_to_max + 1, 1)
                t_active_IDs    = {t:[] for t in t_times_all[1:]}

                # get non-segment nodes within event time inteval
                t_nodes_keep    = [node for node in G.nodes() if t_t_from_min < G_time(node) < t_t_to_max and G_owner(node) is None] 
                
                # i had problems with nodes due to internal events. t_node_subIDs_all was missing times.
                # ill add all relevant segments to subgraph along stray nodes. gather connected components
                # and then subract all segment nodes, except first target nodes. 
                # EDIT 12.12.2023 idk what was the issue then. i add all branches, then delete from segments, then add target nodes..
                # EDIT 12.12.2023 it works for zero inter-event node cases.
                t_nodes_segments_all    = []
                t_branches_all          = t_from_all_new + t_to_all
                for t in t_branches_all:
                    t_nodes_segments_all.extend(t_segments_new[t])
                t_nodes_keep.extend(t_nodes_segments_all)

                # had cases where target nodes are decoupled and are split into different CC clusters.
                # for safety create fake edges by connecting all branches in sequence. but only those that not exist already
                t_d = {**t_from_nodes, **t_target_nodes} # inc segments last node, target seg first node
                t_fake_edges = [(t_d[a],t_d[b]) for a,b in zip(t_branches_all[:-1], t_branches_all[1:]) if not G.has_edge(t_d[a],t_d[b])]
                

                G.add_edges_from(t_fake_edges)
                t_subgraph = G.subgraph(t_nodes_keep)   

                ref_node = G2_n_start(t_to_all[0])      #t_segments_new[t_to_all[0]][0]

                t_sols = nx.connected_components(t_subgraph.to_undirected())
                
                t_sol = next((t for t in t_sols if ref_node in t), None)
                assert t_sol is not None, 'cannot find connected components'
                t_sol = [t for t in t_sol if t not in t_nodes_segments_all]

                t_sol.extend(t_target_nodes.values())

                G.remove_edges_from(t_fake_edges)
                t_node_subIDs_all = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                t_node_subIDs_all = {t:t_node_subIDs_all[t] for t in sorted(t_node_subIDs_all)}

                for t_from in t_from_all_new: # from end of branch time to event max time
                    for t_time in np.arange(t_predecessors_times_end[t_from] + 1, t_t_to_max + 1, 1):
                        t_active_IDs[t_time].append(t_from)

                t_event_start_end_times[t_state][t_from_all] = {
                                                                't_start'       :   t_predecessors_times_end,
                                                                'branches'      :   t_from_all_new,
                                                                't_end'         :   t_successors_times_start,
                                                                't_times'       :   {},
                                                                't_perms'       :   {},
                                                                't_target_nodes':   t_target_nodes,
                                                                't_subIDs'      :   t_node_subIDs_all,
                                                                't_active_IDs'  :   t_active_IDs}

        # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # ===============================================================================================
    # ====  Determine interpolation parameters ====
    # ===============================================================================================
    # we are interested in segments that are not part of psuedo branches (because they will be integrated into path)
    # pre merge segments will be extended, post split branches also, but backwards. 
    # for pseudo event can take an average paramets for prev and post segments.

    print(f'\n{timeHMS()}:({doX_s}) Determining interpolation parameters k and s for segments...')
    
    # get segments that have possibly inherited other segments and that are not branches
    t_segments_IDs_relevant = [t_ID for t_ID,t_traj in enumerate(t_segments_new) if len(t_traj)>0]

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
        trajectory          = np.array([G.nodes[t]["centroid"   ] for t in t_segments_new[t_ID]])
        time                = np.array([G.nodes[t]["time"       ] for t in t_segments_new[t_ID]])
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

    # ===============================================================================================
    # ====  deal with fake event recovery ====
    # ===============================================================================================
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
    for t_subIDs in lr_C1_condensed_connections:
        for t_subID in t_subIDs:
            lr_fake_redirect[t_subID] = min(t_subIDs)


    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # ============================  EXTEND REAL MERGE/SPLIT BRANCHES ================================
    # ===============================================================================================
    # iteratively extend branches though split/merge event. to avoid conflicts with shared nodes
    # extension is simultaneous for all branches at each time step and contours are redistributed conservatively
    # -> at time T, two branches [B1, B2] have to be recovered from node cluster C = [1,2,3]
    # -> try node redistributions- ([partition_B1,partition_B2], ..) = ([[1],[2,3]], [[2],[1,3]], [[3]lr_fake_redirect,[1,2]])    

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
    for t_ID, t_state in ms_branch_extend_IDs:

        # ===========================================================================================
        # =================== DETERMINE BRANCHES, START AND END TIMES OF EVENT ======================
        # ===========================================================================================
        if t_state in ('merge','split'):
            t_branches = t_event_start_end_times[t_state][t_ID]['branches']
            t_times_target = []
        elif t_state == 'mixed':
            t_branches      = t_event_start_end_times[t_state][t_ID]['branches']
            t_nodes_target  = t_event_start_end_times[t_state][t_ID]['t_target_nodes']
            t_times_target  = [G_time(t) for t in t_nodes_target.values()]
            t_subIDs_target = {t:[] for t in t_times_target}
            for t_time, *t_subIDs in t_nodes_target.values():
                t_subIDs_target[t_time] += t_subIDs

        if t_state == 'split':
            t_start = t_event_start_end_times[t_state][t_ID]['t_start']
            t_end   = min(t_event_start_end_times[t_state][t_ID]['t_end'].values())
        elif t_state == 'mixed':
            t_start = min(t_event_start_end_times[t_state][t_ID]['t_start'].values())
            t_end   = max(t_event_start_end_times[t_state][t_ID]['t_end'].values())
        else:
            t_start = min(t_event_start_end_times[t_state][t_ID]['t_start'].values())
            t_end   = t_event_start_end_times[t_state][t_ID]['t_end']

        # ===========================================================================================
        # =============================== PREPARE DATA FOR EACH BRANCH ==============================
        # ===========================================================================================
        t_all_norm_buffers, t_all_traj_buffers, t_all_area_buffers, t_all_time_buffers, t_all_k_s  = {}, {}, {}, {}, {}
         
        for t_branch_ID in t_branches: 
            
            t_branch_ID_new = lr_fake_redirect[t_branch_ID]
            if t_state in ('merge','mixed'):
                
                t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'][t_branch_ID] # last of branch
                t_node_from = G2_n_end(t_branch_ID_new)     

                if t_state == 'merge':
                    t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end']                # first of target
                    t_node_to   = G2_n_start(t_ID)      
                    t_conn = (t_branch_ID, t_ID)
                else: 
                    t_t_to = max(t_event_start_end_times[t_state][t_ID]['t_end'].values())
                    t_node_to = (-1)
                    t_conn = (t_branch_ID,)
            else:
                t_conn = (t_ID, t_branch_ID)
                t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'] # last of branch
                t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end'][t_branch_ID]                # first of target
                t_node_from = G2_n_end(t_ID)                #t_segments_new[t_ID][-1]
                t_node_to   = G2_n_start(t_branch_ID)       #t_segments_new[t_branch_ID][0]
            if t_state in ('split', 'merge') and np.abs(t_t_to - t_t_from) < 2:    # note: there are mixed cases with zero event nodes
                t_out[t_ID][t_branch_ID] = None
                continue
            # =======================================================================================
            # ============== GET SHORT PRIOR HISTORY FOR CENTROIDS, AREAS OF A BRANCH  ==============
            # =======================================================================================
            if t_state in ('merge','mixed'):
                t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if G_time(t_node) > t_t_from - h_interp_len_max2]
            else:
                t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if G_time(t_node) < t_t_to + h_interp_len_max2]

            trajectory  = np.array([G_centroid(t)   for t in t_nodes])
            time        = np.array([G_time(t)       for t in t_nodes])
            area        = np.array([G_area(t)       for t in t_nodes])
            
            N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
            if t_segment_k_s_diffs[t_branch_ID_new] is not None:
                t_last_deltas   = list(t_segment_k_s_diffs[t_branch_ID_new].values())[-N:]  # not changing for splits
            else:
                t_last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5

            t_all_norm_buffers[t_branch_ID] = CircularBuffer(N, t_last_deltas)
            if t_state in ('merge','mixed'):
                t_all_traj_buffers[t_branch_ID] = CircularBuffer(h_interp_len_max2, trajectory)
                t_all_area_buffers[t_branch_ID] = CircularBuffer(h_interp_len_max2, area)
                t_all_time_buffers[t_branch_ID] = CircularBuffer(h_interp_len_max2, time)
            else: 
                t_all_traj_buffers[t_branch_ID] = CircularBufferReverse(h_interp_len_max2, trajectory)
                t_all_area_buffers[t_branch_ID] = CircularBufferReverse(h_interp_len_max2, area)
                t_all_time_buffers[t_branch_ID] = CircularBufferReverse(h_interp_len_max2, time)

            t_extrapolate_sol[      t_conn] = {}
            t_extrapolate_sol_comb[ t_conn] = {}
            lr_post_branch_rec_info[t_conn] = t_state
            t_times_accumulate_resolved     = []
            
            if t_segment_k_s[t_branch_ID_new] is not None:
                t_k,t_s = t_segment_k_s[t_branch_ID_new]
                t_all_k_s[t_branch_ID] = t_segment_k_s[t_branch_ID_new]
            else:
                t_k,t_s = (1,5)
                t_all_k_s[t_branch_ID] = (1,5)
        

        # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ===========================================================================================
        # ================================ START BRANCH EXTRAPOLATION ===============================
        # ===========================================================================================
        # walk times from start of an event to end. each time step there are branches "t_branch_IDs" that have to be recovered 
        # by default branch is being recovered from time it ended to end of event => "t_branch_IDs_OG" = "t_branch_IDs",
        # but once branch recovery is terminated it should not be considered anymore, so t_branch_IDs_OG is wrong. refine
        # each step branches use contours from pool of contours "t_subIDs". some branches may start later.

        t_branch_failed = []
        t_report = set()
        for t_time_next, t_branch_IDs_OG in t_event_start_end_times[t_state][t_ID]['t_active_IDs'].items():
            if t_state == 'split':  t_time = t_time_next + 1                            # t_time_next is in the past
            else:                   t_time = t_time_next - 1                            # t_time_next is in the future

            t_branch_IDs = [t for t in t_branch_IDs_OG if t not in t_branch_failed]     # t_branch_failed are resolved or terminated
            if len(t_branch_IDs) == 0: continue                                         # no branches to resolve, try next step.

            t_subIDs = t_event_start_end_times[t_state][t_ID]['t_subIDs'][t_time_next]  # contour pool

            if len(t_subIDs) < len(t_branch_IDs):                                       # more branches than contorus available.
                ms_early_termination[t_ID, t_state] = [t_time_next,t_branch_IDs]        # cant resolve this.
                t_branch_failed.extend(t_branch_IDs)                                    # consider both branches failed.
                continue                                                                # continue to end event time maybe there are any other 
                                                                                        # branches to fix. see "extend_branchs_solo_node_continue.png"
            t_centroids_extrap = np.zeros((len(t_branch_IDs),2))
            t_areas_extrap = np.zeros(len(t_branch_IDs))

            for t, t_branch_ID in enumerate(t_branch_IDs):                              # extrapolate traj of active branches
                t_traj_b    = t_all_traj_buffers[t_branch_ID].get_data()
                t_time_b    = t_all_time_buffers[t_branch_ID].get_data()
                t_area_b    = t_all_area_buffers[t_branch_ID].get_data()
                t_k, t_s    = t_all_k_s[t_branch_ID]
                t_centroids_extrap[t] = interpolate_trajectory(t_traj_b, t_time_b, which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]
                t_areas_extrap[t]     = interpolateMiddle1D_2(t_time_b, t_area_b, [t_time_next], rescale = True, s = 15, debug = 0, aspect = 'auto', title = 1)
                    
            
            if len(t_branch_IDs) == 1:
                t_perms_distribution2 = [[list(t)] for t in combs_different_lengths(t_subIDs)]  # if only one choice, gen diff perms of contours
            else:
                t_perms_distribution2 = list(split_into_bins(t_subIDs,len(t_branch_IDs)))       # multiple choices, get perm distrib options
            t_perms_distribution2 = [[tuple(sorted(b)) for b in a] for a in t_perms_distribution2]

            # calculate parameters of different permutations of contours.
            t_permutation_params = {}
            t_permutations = combs_different_lengths(t_subIDs)
            t_permutations = [tuple(sorted(a)) for a in t_permutations]
            for t_permutation in t_permutations:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time_next][tID] for tID in t_permutation]))
                t_permutation_params[t_permutation] = centroid_area(t_hull)


            # evaluate differences between extrapolated data and possible solutions in subID permutations/combinations
            t_diff_choices      = {}      # holds index key on which entry in t_perms_distribution2 has specifict differences with target values.
            t_diff_choices_area = {}
            for k, t_redist_case in enumerate(t_perms_distribution2):
                t_centroids             = np.array([t_permutation_params[t][0] for t in t_redist_case])
                t_areas                 = np.array([t_permutation_params[t][1] for t in t_redist_case])
                t_diff_choices[k]       = np.linalg.norm(t_centroids_extrap - t_centroids, axis=1)
                t_diff_choices_area[k]  = np.abs(t_areas_extrap - t_areas)/t_areas_extrap
            a = 1
            # -------------------------------- refine simultaneous solution ---------------------------------------
            # evaluating sum of diffs is a bad approach, because they may behave differently

            t_norms_all     = {t:np.array(  t_all_norm_buffers[t].get_data()) for t in t_branch_IDs}
            t_max_diff_all  = {t:max(np.mean(n),5) + 5*np.std(n)              for t,n in t_norms_all.items()}
            t_relArea_all = {}
            for k in t_branch_IDs:
                t_area_hist         = t_all_area_buffers[k].get_data()
                t_relArea_all[k]    = np.abs(np.diff(t_area_hist))/t_area_hist[:-1]
            t_max_relArea_all  = {t:max(np.mean(n) + 5*np.std(n), 0.35) for t,n in t_relArea_all.items()}

                
            t_choices_pass_all      = [] # holds indidicies of t_perms_distribution2
            t_choices_partial       = []
            t_choices_partial_sols  = {}
            t_choices_pass_all_both = []
            # filter solution where all branches are good. or only part is good
            for t in t_diff_choices:
                t_pass_test_all     = t_diff_choices[       t] < np.array(list(t_max_diff_all.values())) # compare if less then crit.
                t_pass_test_all2    = t_diff_choices_area[  t] < np.array(list(t_max_relArea_all.values())) # compare if less then crit.
                t_pass_both_sub = np.array(t_pass_test_all) & np.array(t_pass_test_all2) 
                if   all(t_pass_both_sub):   t_choices_pass_all_both.append(t)          # all branches pass
                elif any(t_pass_both_sub):
                    t_choices_partial.append(t)
                    t_choices_partial_sols[t] = t_pass_both_sub
                

            if len(t_choices_pass_all_both) > 0:     # isolate only good choices    
                if len(t_choices_pass_all_both) == 1:
                    t_diff_norms_sum = {t_choices_pass_all_both[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1 = {t: t_diff_choices[     t] for t in t_choices_pass_all_both}
                    temp2 = {t: t_diff_choices_area[t] for t in t_choices_pass_all_both}
                    test, t_diff_norms_sum = two_crit_many_branches(temp1, temp2, len(t_branch_IDs))
                #test, t_diff_norms_sum = two_crit_many_branches(t_diff_choices, t_diff_choices_area, len(t_branch_IDs))
                #t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items() if t in t_choices_pass_all}

            # if only part is good, dont drop whole solution. form new crit based only on non-failed values.
            elif len(t_choices_partial) > 0:  
                if len(t_choices_partial) == 1:
                    t_diff_norms_sum = {t_choices_partial[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1 = {t: t_diff_choices[     t] for t in t_choices_partial}
                    temp2 = {t: t_diff_choices_area[t] for t in t_choices_partial}
                    test, t_diff_norms_sum = two_crit_many_branches(temp1, temp2, len(t_branch_IDs))
                    #t_temp = {}
                    #for t in t_choices_partial:      
                    #    t_where = np.where(t_choices_partial_sols[t])[0]
                    #    t_temp[t] = np.sum(t_diff_choices[t][t_where])
                    #t_diff_norms_sum = t_temp
                    #assert len(t_diff_norms_sum) > 0, 'when encountered insert a loop continue, add branches to failed' 
            # all have failed. process will continue, but it will fail checks and terminate branches.
            else:                              
                #t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items()}
                t_diff_norms_sum = {0:[t + 1 for t in t_max_diff_all.values()]} # on fail spoof case which will fail.

            t_where_min = min(t_diff_norms_sum, key = t_diff_norms_sum.get)
            t_sol_d_norms = t_diff_choices[t_where_min]
            t_sol_subIDs = t_perms_distribution2[t_where_min]
            t_branch_pass = []
            for t_branch_ID, t_subIDs, t_sol_d_norm in zip(t_branch_IDs,t_sol_subIDs, t_sol_d_norms):

                if t_sol_d_norm < t_max_diff_all[t_branch_ID]:
                    t_all_norm_buffers[t_branch_ID].append(t_sol_d_norm)
                    t_all_traj_buffers[t_branch_ID].append(t_permutation_params[tuple(t_subIDs)][0])
                    t_all_area_buffers[t_branch_ID].append(t_permutation_params[tuple(t_subIDs)][1])
                    t_all_time_buffers[t_branch_ID].append(t_time_next)
                    
                    if t_state == 'merge':      t_conn = (t_branch_ID, t_ID)
                    elif t_state == 'mixed':    t_conn = (t_branch_ID,)
                    else:                       t_conn = (t_ID, t_branch_ID)

                    t_extrapolate_sol_comb[t_conn][t_time_next] = tuple(t_subIDs)
                    t_branch_pass.append(t_branch_ID)
                    t_report.add(t_conn)
                else:
                    t_branch_failed.append(t_branch_ID)
                    continue


            if t_state in  ('mixed'):                        # for mixed cases i should search of extended incoming branch has reached any of target branches
                if t_time_next in t_times_target:            # if recovered time step is at same time as some of first target nodes
                    for t_branch_ID in t_branch_pass:        # test successfully recovered branches for contour ID overlap

                        t_nodes_sol_subIDs = t_extrapolate_sol_comb[(t_branch_ID,)][t_time_next]     # subIDs in solution
                        t_nodes_solution = tuple([t_time_next] + list(t_nodes_sol_subIDs))           # reconstruct a node of a solution

                        if t_nodes_solution in t_nodes_target.values():
                            t_target_ID = find_key_by_value(t_nodes_target,t_nodes_solution)         # if this node is in list of target nodes

                            ms_mixed_completed['full'][(t_branch_ID,)]['solution'] = t_nodes_sol_subIDs  # consider path recovered
                            if 'targets' not in ms_mixed_completed['full'][(t_branch_ID,)]: ms_mixed_completed['full'][(t_branch_ID,)]['targets'] = []
                            ms_mixed_completed['full'][(t_branch_ID,)]['targets'].append(t_target_ID)    
                            t_branch_failed.append(t_branch_ID)                                          # add branch to failed list. to stop from extending it further
                        else:                                                                            # if not in list of tar nodes
                            set1 = set(t_subIDs_target[t_time_next])
                            set2 = set(t_nodes_sol_subIDs)
                            inter = set1.intersection(set2)                                              # check subID intersection

                            if len(inter)> 0:                                                            # if there is an intersection
                                t_intersecting_branches_IDs = {t for t,t_subIDs in t_nodes_target.items() if set(t_subIDs[1:]).intersection(set(t_nodes_sol_subIDs)) != {}}
                                ms_mixed_completed['partial'][(t_branch_ID,)]['solution'] = t_nodes_sol_subIDs  # add here, but idk what to do  yet
                                print(t_intersecting_branches_IDs)
                                t_branch_failed.append(t_branch_ID)

                                if 'mixed' not in issues_all_dict[doX]: issues_all_dict[doX]['mixed'] = []
                                issues_all_dict[doX]['mixed'].append(f'branch {t_branch_ID} extention resulted in multiple branches : {t_nodes_solution} ')
                                t_extrapolate_sol_comb[(t_branch_ID,)].pop(t_time_next,None)
                                #assert 1 == -1, 'extension of mixed type branch resulted in partial success. target cntr is in subIDs. check this case more closely'
                                #if 'targets' not in ms_mixed_completed['partial'][t_ID]: ms_mixed_completed['partial'][t_ID]['targets'] = []
                                #ms_mixed_completed['partial'][t_ID]['targets'].append(t_target_ID)
        a = 1
        for t_conn in t_report:
            t_dict = t_extrapolate_sol_comb[t_conn]
            if len(t_dict) == 0: continue
            if len(t_conn) == 1:
                t_from, t_to = t_conn[0], -1
            else:
                (t_from, t_to) = t_conn
            t_min,t_max = min(t_dict),max(t_dict)
            t_node_from = tuple([t_min] + list(t_dict[t_min]))
            t_node_to   = tuple([t_max] + list(t_dict[t_max]))
            print(f' {t_state}:connection :{(t_from, t_to)} = {t_node_from}->{t_node_to}')
        a = 1

    lr_conn_merges_good = set()#dict()#defaultdict(set)
    
    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    lr_conn_merges_good = [(t_ID, t_state) for t_ID, t_state in lr_post_branch_rec_info.items() if len(t_extrapolate_sol_comb[t_ID]) > 0]
    
    for t_conn, t_state in lr_conn_merges_good:
        # principle of save_connections_X is decribed in "misc.save_connections_two_ways"
        if len(t_conn) == 2:    t_from, t_to = t_conn
        else:                   t_from, t_to = t_conn[0], -1
        t_from_new = lr_fake_redirect[t_from]
        t_combs = t_extrapolate_sol_comb[t_conn]

        if len(t_combs) == 0: continue                              # no extension, skip.

        if t_state in ('merge', 'mixed') and t_conn not in ms_mixed_completed['full']:
            save_connections_merges(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from_new,  None, G, G2, lr_fake_redirect, g0_contours)
        elif t_state == 'split':
            save_connections_splits(t_segments_new, t_extrapolate_sol_comb[t_conn], None,  t_to, G, G2, lr_fake_redirect, g0_contours)
        else:
            t_to = ms_mixed_completed['full'][t_conn]['targets'][0]
            # remove other edges from mixed connection segment graph
            t_from_other_predecessors = [lr_fake_redirect[t] for t in G2.predecessors(t_to) if t != t_conn[0]]
            t_edges = [(t, t_to) for t in t_from_other_predecessors]
            G2.remove_edges_from(t_edges)
            save_connections_two_ways(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from_new,  t_to, G, G2, lr_fake_redirect, g0_contours)


    # for_graph_plots(G, segs = t_segments_new, suptitle = f'{doX}_after', show = True)        


    aaa = defaultdict(set)
    for t, t_segment in enumerate(t_segments_new):
        for t_node in t_segment:
            aaa[t].add(G.nodes[t_node]["owner"])
    tt = []
    for t_node in G.nodes():
        if "time" not in G.nodes[t_node]:
            set_custom_node_parameters(G, g0_contours, [t_node], None, calc_hull = 1)
            tt.append(t_node)
    print(f'were missing: {tt}')
    

    print(f'\n{timeHMS()}:({doX_s}) Final. Recompute new straight segments')
    # ===============================================================================================
    # ============== Final passes. New straight segments. ===================
    # ===============================================================================================
    # WHY: after merge/split/merge extensions there may be explicit branches left over in event area
    # HOW: analyze recent graph for straight segments and determine which segments are different from old
    t_segments_fin = graph_extract_paths(G, min_length = 2)
    #t_segments_fin_dic, skipped = graph_extract_paths(G,lambda x : x[0])
    #t_segments_fin = [t for t in t_segments_fin_dic.values() if len(t) > 2]

    t_unresolved_new = list(range(len(t_segments_fin)))
    t_unresolved_old = [t for t,t_nodes in enumerate(t_segments_new) if len(t_nodes) > 0]
    t_resolved_new = []
    t_resolved_old = []
    for t_i_new, t_traj in enumerate(t_segments_fin):                           # first test if first elements are the same
        for t_i_old     in t_unresolved_old:
            if t_traj[0] == t_segments_new[t_i_old][0]:
                t_resolved_new.append(t_i_new)
                t_resolved_old.append(t_i_old)
                break
    t_unresolved_new = [t for t in t_unresolved_new if t not in t_resolved_new]
    t_unresolved_old = [t for t in t_unresolved_old if t not in t_resolved_old]
    temp_u_new = copy.deepcopy(t_unresolved_new)                                # (cant iterate though changing massives. copy)
    temp_u_old = copy.deepcopy(t_unresolved_old)
    for t_i_new in temp_u_new:         # can be improved. minor                 # then test unresolved using set intersection
        for t_i_old in temp_u_old:                                              # might be needed if segments got wider
            if t_i_old not in t_resolved_old:
                intersection_length = len(set(t_segments_fin[t_i_new]).intersection(set(t_segments_new[t_i_old])))
                if intersection_length > 0:
                    t_unresolved_new.remove(t_i_new)
                    t_unresolved_old.remove(t_i_old)
                    t_resolved_new.append(t_i_new)
                    t_resolved_old.append(t_i_old)
                    break
    t_start_ID = len(t_segments_new)
    fin_additional_segments_IDs = list(range(t_start_ID, t_start_ID + len(t_unresolved_new), 1))

    for t_ID in t_unresolved_new:      # add new segments to end of old storage

        t_new_ID = len(t_segments_new)

        t_segments_new.append(t_segments_fin[t_ID])

        node_start, node_end = t_segments_new[t_new_ID][0], t_segments_new[t_new_ID][-1]

        #G2.add_node(t_new_ID) 
        #G2.nodes()[t_new_ID]["t_start"      ]   =   G_time(node_start   )
        #G2.nodes()[t_new_ID]["t_end"        ]   =   G_time(node_end     )
        #G2.nodes()[t_new_ID]['node_start'   ]   =   node_start
        #G2.nodes()[t_new_ID]['node_end'     ]   =   node_end
        #set_custom_node_parameters(G, g0_contours_hulls, t_segments_fin[t_ID], t_new_ID, calc_hull = 0) # straight hulls since solo nodes 
        set_custom_node_parameters(G, g0_contours, t_segments_fin[t_ID], t_new_ID, calc_hull = 1)        # 
        G2_set_parameters(G, G2, t_segments_fin[t_ID], t_new_ID)#, G_time)


    #lr_time_active_segments = defaultdict(list)
    #for t_segment_index, t_segment_nodes in enumerate(t_segments_new):
    #    t_times = [G_time(node) for node in t_segment_nodes]
    #    for t_time in t_times:
    #        lr_time_active_segments[t_time].append(t_segment_index)
    ## sort keys in lr_time_active_segments
    #lr_time_active_segments = {t:lr_time_active_segments[t] for t in sorted(lr_time_active_segments.keys())}

    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for_graph_plots(G, segs = t_segments_fin)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Update connectivity')
    # ============== Final passes. New straight segments. Update connectivity ===================
    #G2.add_edges_from(G2.edges())
    fin_connectivity_graphs = defaultdict(list) #fin_additional_segments_IDs
    t_has_holes_report = {}
    G2_new = graph_check_paths(G, G2, nx.DiGraph(), lr_maxDT, t_has_holes_report)

    print(f'\n{timeHMS()}:({doX_s}) Paths that have failed hole test: {t_has_holes_report}')

    #fin_connectivity_graphs = {t_conn:t_vals for t_conn, t_vals in fin_connectivity_graphs.items() if len(t_vals) > 2}
    for t_ID in fin_additional_segments_IDs:
        t_segment_k_s_diffs[t_ID] = None
        t_segment_k_s[t_ID] = None
        lr_fake_redirect[t_ID] = t_ID
    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if 1 == 1:
        print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Branches of splits/merges')
        # ===============================================================================================
        # ========================= Final passes. recompute split/merge/mixed info ======================
        # ===============================================================================================
        # NOTE 0: same as pre- merge/split/mixed extension (which was same as pre 121 recovery)
        # NOTE 1: extension of branches requires more or equal number contours than branches.
        # NOTE 1: thats why there is no point in recovering branches past this stage. recover only to earliest case.
        fin_extend_info = defaultdict(dict)
            
        #G2_dir, fin_edges_main = get_event_types_from_segment_graph(G2)   
        G2_dir, fin_edges_main = get_event_types_from_segment_graph(G2_new) 
        
        t_state = 'merge'
        t_all_event_IDs             = [lr_fake_redirect[t_ID] for t_ID in t_event_start_end_times[t_state]]
        for t_to, t_predecessors in fin_edges_main[t_state].items():
            if t_to not in t_all_event_IDs: continue # event was already resolved without extrapolation. edit-comment it was not added/deleted from list
            t_time_to               = G2_dir.nodes[t_to]['t_start']
            t_times_from            = {t:G2_dir.nodes[t]['t_end'] for t in t_predecessors}
            t_time_from             = min(t_times_from.values())
            t_times_from_max        = max(t_times_from.values())    # cant recover lower than this time
            t_times_subIDs          = t_event_start_end_times[t_state][t_to]['t_subIDs']
            t_times_subIDs_slice    = {t_time:t_subIDs for t_time, t_subIDs in t_times_subIDs.items() if t_time_from <= t_time <= t_time_to}
            # below not correct. for general case, should check number of branches. OR there is nothing to do except checking for == 1.
            t_times_solo = [t_time for t_time,t_subIDs in t_times_subIDs_slice.items() if len(t_subIDs) == 1 and t_time > t_times_from_max] 
            if len(t_times_solo)>0:
                t_time_solo_min = min(t_times_solo) # min to get closer to branches
                if t_time_to - t_time_solo_min > 1:
                    t_times     = np.flip(np.arange(t_time_solo_min, t_time_to , 1))
                    fin_extend_info[(t_to,'back')] = {t_time: combs_different_lengths(t_times_subIDs_slice[t_time]) for t_time in t_times}


        t_state = 'split'
        t_all_event_IDs             = [lr_fake_redirect[t_ID] for t_ID in t_event_start_end_times[t_state]]
        for t_from, t_successors in fin_edges_main[t_state].items():
            if t_from not in t_all_event_IDs: continue # event was already resolved without extrapolation
            t_time_from             = G2_dir.nodes[t_from]['t_end']
            t_times_from            = {t:G2_dir.nodes[t]['t_start'] for t in t_successors}
            t_time_to               = max(t_times_from.values())
            t_time_to_min           = min(t_times_from.values())    # cant recover higher than this time
            t_from_old              = None
            for t in t_event_start_end_times[t_state]: # this field uses old IDs. have to recover it.
                if t_event_start_end_times[t_state][t]['t_start'] == t_time_from: t_from_old = t
            
            if t_from_old is not None:
                t_times_subIDs      = t_event_start_end_times[t_state][t_from_old]['t_subIDs']
                t_times_subIDs_slice = {t_time:t_subIDs for t_time, t_subIDs in t_times_subIDs.items() if t_time_from <= t_time <= t_time_to}
                # below not correct. for general case, should check number of branches. OR there is nothing to do except checking for == 1.
                t_times_solo = [t_time for t_time,t_subIDs in t_times_subIDs_slice.items() if len(t_subIDs) == 1 and t_time < t_time_to_min] 
                if len(t_times_solo)>0:
                    t_time_solo_max = max(t_times_solo)
                    if t_time_solo_max - t_time_from > 1:
                        t_times = np.arange(t_time_from + 1, t_time_solo_max + 1 , 1)
                        fin_extend_info[(t_from,'forward')] = {t_time: combs_different_lengths(t_times_subIDs_slice[t_time]) for t_time in t_times}
                        
        a = 1
    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Terminated segments')
    # ===============================================================================================
    # =========== Final passes. examine edges of terminated segments. retrieve nodes  ===============
    # ===============================================================================================
    # WHAT: segments not connected to any other segments, on one or both sides, may still have 
    # WHAT: stray nodes on ends. These cases were not explored previously.
    # HOW: find segments with no successors or predecessors. Using Depth Search extract trailing nodes...
    # NOTE: element order in t_conn now is fixed (from, to), numbering hierarchy does not represent order anymore.
    t_start_points  = []
    t_end_points    = []

    for t_ID in G2_dir.nodes():
        t_successors    = list(G2_dir.successors(  t_ID))
        t_predecessors  = list(G2_dir.predecessors(t_ID))
        if len(t_successors)    == 0: t_end_points.append(t_ID)
        if len(t_predecessors)  == 0: t_start_points.append(t_ID)
    
    # 
    # ============= Final passes. Terminated segments. extract trailing nodes =================
    if 1 == 1:
        t_back = defaultdict(set)
        for t_ID in t_start_points:
            t_node_from = t_segments_new[t_ID][0]
            dfs_pred(G, t_node_from, time_lim = t_node_from[0] - 10, node_set = t_back[t_ID])
        t_back = {t:v for t,v in t_back.items() if len(v) > 1}
        t_forw = defaultdict(set)
        for t_ID in t_end_points:
            t_node_from = t_segments_new[t_ID][-1]
            dfs_succ(G, t_node_from, time_lim = t_node_from[0] + 10, node_set = t_forw[t_ID])
        t_forw = {t:v for t,v in t_forw.items() if len(v) > 1}

    # == Final passes. Terminated segments. Generate disperesed node dictionary for extrapolation ==
    for t_ID , t_nodes in t_back.items():
        t_perms = disperse_nodes_to_times(t_nodes)
        t_sorted = sorted(t_perms)[:-1]
        t_sorted.reverse()
        t_perms = {t:t_perms[t] for t in t_sorted}
        t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
        fin_extend_info[(t_ID,'back')] = {t:v for t,v in zip(t_perms, t_values)}

    for t_ID , t_nodes in t_forw.items():
        t_perms = disperse_nodes_to_times(t_nodes)
        t_sorted = sorted(t_perms)[1:]
        t_perms = {t:t_perms[t] for t in t_sorted}
        t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
        fin_extend_info[(t_ID,'forward')] = {t:v for t,v in zip(t_perms, t_values)}

    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Extrapolate edges. Extrapolate')
    # ============= Final passes. Terminated segments. Extrapolate segments =================
    t_report                = set()
    t_out                   = defaultdict(dict)
    t_extrapolate_sol       = defaultdict(dict)
    t_extrapolate_sol_comb  = defaultdict(dict)
    for (t_ID, t_state), t_combs in fin_extend_info.items():
        t_conn = (t_ID, t_state) # <<<<< custom. t_ID may be two states, and they have to be differentiated
        if t_state == 'forward':
            t_t_from = t_segments_new[t_ID][-1][0]
            t_nodes = [t_node for t_node in t_segments_new[t_ID] if t_node[0] > t_t_from - h_interp_len_max2]
            t_node_from = t_segments_new[t_ID][-1]
        else:
            t_t_to = t_segments_new[t_ID][0][0]
            t_nodes = [t_node for t_node in t_segments_new[t_ID] if t_node[0] < t_t_to + h_interp_len_max2]
            t_node_from = t_segments_new[t_ID][0]

        trajectory = np.array([G.nodes[t]["centroid"] for t in t_nodes])
        time       = np.array([G.nodes[t]["time"    ] for t in t_nodes])
        area       = np.array([G.nodes[t]["area"    ] for t in t_nodes])

        if t_state == 'forward':
            t_traj_buff     = CircularBuffer(h_interp_len_max2, trajectory)
            t_area_buff     = CircularBuffer(h_interp_len_max2, area)
            t_time_buff     = CircularBuffer(h_interp_len_max2, time)       
            t_time_next     = t_t_from    + 1                               
        if t_state == 'back':
            t_traj_buff     = CircularBufferReverse(h_interp_len_max2, trajectory) 
            t_area_buff     = CircularBufferReverse(h_interp_len_max2, area)
            t_time_buff     = CircularBufferReverse(h_interp_len_max2, time)      
            t_time_next     = t_t_to      - 1
        
        if t_segment_k_s[t_ID] is not None:
            t_k,t_s = t_segment_k_s[t_ID]
        else:
            t_k,t_s = (1,5)
        N = 5
        if t_segment_k_s_diffs[t_ID] is not None:
            t_last_deltas   = list(t_segment_k_s_diffs[t_ID].values())[-N:]  # not changing for splits
        else:
            t_last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5
        t_branch_IDs = [t_ID]
        t_branch_ID = t_ID
        t_norm_buffer   = CircularBuffer(N, t_last_deltas)
        
        t_all_traj_buffers = {t_branch_ID: t_traj_buff  }
        t_all_area_buffers = {t_branch_ID: t_area_buff  }
        t_all_norm_buffers = {t_branch_ID: t_norm_buffer}
        t_all_time_buffers = {t_branch_ID: t_time_buff  }
        
        t_times_accumulate_resolved = []
        break_trigger = False
        for t_time, t_permutations in t_combs.items():

            if break_trigger: break # solo branch recover has failed.

            t_time_next = t_time

            t_traj_b    = t_all_traj_buffers[t_branch_ID].get_data()
            t_time_b    = t_all_time_buffers[t_branch_ID].get_data()
            t_area_b    = t_all_area_buffers[t_branch_ID].get_data()

            t_centroids_extrap  = interpolate_trajectory(t_traj_b, t_time_b, which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]
            t_areas_extrap      = interpolateMiddle1D_2(t_time_b, t_area_b, [t_time_next], rescale = True, s = 15, debug = 0, aspect = 'auto', title = 1)
            #t_centroids_extrap  = t_extrap
            #t_areas_extrap      = t_area_ext
            t_centroids = []
            t_areas     = []

            t_permutation_params = {}
            for t_permutation in t_permutations:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][tID] for tID in t_permutation]))
                t_centroid, t_area = centroid_area(t_hull)
                t_centroids.append(t_centroid)
                t_areas.append(t_area)
                t_permutation_params[t_permutation] = centroid_area(t_hull)

            t_diff_choices      = {}      # holds index key on which entry in t_perms_distribution2 has specifict differences with target values.
            t_diff_choices_area = {}
            t_perms_distribution2 = [[list(t)] for t in t_permutations]
            for t, t_redist_case in enumerate(t_perms_distribution2):
                t_centroids = np.array([t_permutation_params[tuple(t)][0] for t in t_redist_case])
                t_areas     = np.array([t_permutation_params[tuple(t)][1] for t in t_redist_case])
                t_diff_choices[t]       = np.linalg.norm(t_centroids_extrap - t_centroids, axis=1)
                t_diff_choices_area[t]  = np.abs(t_areas_extrap - t_areas)/t_areas_extrap
            a = 1
            # -------------------------------- refine simultaneous solution ---------------------------------------
            # evaluating sum of diffs is a bad approach, becaues they may behave differently

            t_norms_all     = {t:np.array(  t_all_norm_buffers[t].get_data()) for t in t_branch_IDs}
            t_max_diff_all  = {t:max(np.mean(n),5) + 5*np.std(n)              for t,n in t_norms_all.items()}
            t_relArea_all = {}
            for t in t_branch_IDs:
                t_area_hist = t_all_area_buffers[t].get_data()
                t_relArea_all[t] = np.abs(np.diff(t_area_hist))/t_area_hist[:-1]
            t_max_relArea_all  = {t:max(np.mean(n) + 5*np.std(n), 0.35) for t,n in t_relArea_all.items()}

                
            t_choices_pass_all      = [] # holds indidicies of t_perms_distribution2
            t_choices_partial       = []
            t_choices_partial_sols  = {}
            t_choices_pass_all_both = []
            # filter solution where all branches are good. or only part is good
            for t in t_diff_choices:
                t_pass_test_all     = t_diff_choices[       t] < np.array(list(t_max_diff_all.values())) # compare if less then crit.
                t_pass_test_all2    = t_diff_choices_area[  t] < np.array(list(t_max_relArea_all.values())) # compare if less then crit.
                t_pass_both_sub = np.array(t_pass_test_all) & np.array(t_pass_test_all2) 
                if   all(t_pass_both_sub):   t_choices_pass_all_both.append(t)          # all branches pass
                elif any(t_pass_both_sub):
                    t_choices_partial.append(t)
                    t_choices_partial_sols[t] = t_pass_both_sub
                

            if len(t_choices_pass_all_both) > 0:     # isolate only good choices    
                if len(t_choices_pass_all_both) == 1:
                    t_diff_norms_sum = {t_choices_pass_all_both[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1 = {t: t_diff_choices[     t] for t in t_choices_pass_all_both}
                    temp2 = {t: t_diff_choices_area[t] for t in t_choices_pass_all_both}
                    test, t_diff_norms_sum = two_crit_many_branches(temp1, temp2, len(t_branch_IDs))
                #test, t_diff_norms_sum = two_crit_many_branches(t_diff_choices, t_diff_choices_area, len(t_branch_IDs))
                #t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items() if t in t_choices_pass_all}

            # if only part is good, dont drop whole solution. form new crit based only on non-failed values.
            elif len(t_choices_partial) > 0:  
                if len(t_choices_partial) == 1:
                    t_diff_norms_sum = {t_choices_partial[0]:0} # one choice, take it but spoof results, since no need to calc.
                else:
                    temp1 = {t: t_diff_choices[     t] for t in t_choices_partial}
                    temp2 = {t: t_diff_choices_area[t] for t in t_choices_partial}
                    test, t_diff_norms_sum = two_crit_many_branches(temp1, temp2, len(t_branch_IDs))
                    #t_temp = {}
                    #for t in t_choices_partial:      
                    #    t_where = np.where(t_choices_partial_sols[t])[0]
                    #    t_temp[t] = np.sum(t_diff_choices[t][t_where])
                    #t_diff_norms_sum = t_temp
                    #assert len(t_diff_norms_sum) > 0, 'when encountered insert a loop continue, add branches to failed' 
            # all have failed. process will continue, but it will fail checks and terminate branches.
            else:                              
                #t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items()}
                t_diff_norms_sum = {0:[t + 1 for t in t_max_diff_all.values()]} # on fail spoof case which will fail.

            t_where_min = min(t_diff_norms_sum, key = t_diff_norms_sum.get)
            t_sol_d_norms = t_diff_choices[t_where_min]
            t_sol_subIDs = t_perms_distribution2[t_where_min]

            for t_branch_ID, t_subIDs, t_sol_d_norm in zip(t_branch_IDs,t_sol_subIDs, t_sol_d_norms):

                if t_sol_d_norm < t_max_diff_all[t_branch_ID]:
                    t_all_norm_buffers[t_branch_ID].append(t_sol_d_norm)
                    t_all_traj_buffers[t_branch_ID].append(t_permutation_params[tuple(t_subIDs)][0])
                    t_all_area_buffers[t_branch_ID].append(t_permutation_params[tuple(t_subIDs)][1])
                    t_all_time_buffers[t_branch_ID].append(t_time_next)
                    
                    t_extrapolate_sol_comb[t_conn][t_time_next] = tuple(t_subIDs)
                    t_report.add(t_conn)
                else:
                    break_trigger = True
                    break      # this is solo branch extrapolation, should break here. og method uses continue to recover other branches.
    
    
        a = 1
        for t_conn in t_report:
            (t_from, t_state)  = t_conn
            t_dict = t_extrapolate_sol_comb[t_conn]
            if len(t_dict) == 0: continue
           
            t_min, t_max    = min(t_dict), max(t_dict)
            if t_state == 'forward':
                t_node_from     = tuple([t_min] + list(t_dict[t_min]))
                t_node_to       = tuple([t_max] + list(t_dict[t_max]))
            else:
                t_node_to       = tuple([t_min] + list(t_dict[t_min]))
                t_node_from     = tuple([t_max] + list(t_dict[t_max]))

            print(f' {t_from} - {t_state} extrapolation: {t_node_from}->{t_node_to}')
        a = 1


    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # =========== Final passes. examine edges of terminated segments. solve conflicts  ===============
    # ===============================================================================================

    # check if there are contested nodes in all extrapolated paths
    t_duplicates = conflicts_stage_1(t_extrapolate_sol_comb)
    
    if len(t_duplicates) > 0:   # tested on 26.10.23; had #variants_possible = 1
        # retrieve viable ways of redistribute contested nodes
        variants_all        = conflicts_stage_2(t_duplicates)
        variants_possible   = conflicts_stage_3(variants_all,t_duplicates, t_extrapolate_sol_comb)
        #if there is only one solution by default take it as answer
        if len(variants_possible) == 1:  
            t_choice_evol = variants_possible[0]
        elif len(variants_possible) == 0:
            t_problematic_conns = set()
            [t_problematic_conns.update(t_conns) for t_conns in t_duplicates.values()]
            for t_conn in t_problematic_conns:
                t_extrapolate_sol_comb.pop(t_conn, None)
            t_choice_evol = []
        else:
            # method is not yet constructed, it should be based on criterium minimization for all variants
            # current, trivial solution, is to pick solution at random. at least there is no overlap.
            assert -1 == 0, 'multiple variants of node redistribution'
            t_choice_evol = variants_possible[0]
         
        # redistribute nodes for best solution.
        for t_node,t_conn in t_choice_evol:
            tID                 = t_conn[1]
            t_time, *t_subIDs   = t_node
            t_extrapolate_sol_comb[t_conn][t_time] = tuple(t_subIDs)
            t_conns_other = [con for con in t_duplicates[t_node] if con != t_conn] # competing branches
            for t_conn_other in t_conns_other:
                t_subIDs_other = t_extrapolate_sol_comb[t_conn_other][t_time]                           # old solution for t_time
                t_extrapolate_sol_comb[t_conn_other][t_time] = tuple(set(t_subIDs_other) - set(t_subIDs)) # remove competeing subIDs
            #t_delete_conns      = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
            #for t_delete_conn in t_delete_conns:
            #    t_temp = t_extrapolate_sol_comb[t_delete_conn][t_time]
            #    t_temp = [t for t in t_temp if t not in t_subIDs]
            #    t_extrapolate_sol_comb[t_delete_conn][t_time] = t_temp
            #t_conns_relevant = [t_c for t_c in t_extrapolate_sol_comb if t_c[1] == tID]
            #lr_conn_merges_good.update(t_conns_relevant) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< not correct anymore

    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # =========== Final passes. examine edges of terminated segments. save extensions  ===============
    # ===============================================================================================

    #for (t_ID,t_state), t_conns in lr_conn_merges_good.items():
    for t_conn, t_combs in t_extrapolate_sol_comb.items():
        (t_ID, t_state) = t_conn

        if len(t_combs) == 0: continue                              # no extension, skip.

        if t_state == 'forward':
            save_connections_merges(t_segments_new, t_extrapolate_sol_comb[t_conn], t_ID,  None, G, G2_dir, lr_fake_redirect, g0_contours)
        elif t_state == 'back':
            save_connections_splits(t_segments_new, t_extrapolate_sol_comb[t_conn], None,  t_ID, G, G2_dir, lr_fake_redirect, g0_contours)


    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    tt = []
    for t_node in G.nodes():
        if "time" not in G.nodes[t_node]:
            set_custom_node_parameters(G, g0_contours, [t_node], None, calc_hull = 1)
            tt.append(t_node)
    print(f'were missing: {tt}')



    export_time_active_segments = defaultdict(list)                                     # prepare for finding edges between segments
    for k,t_segment in enumerate(t_segments_new):
        t_times = [a[0] for a in t_segment]
        for t in t_times:
            export_time_active_segments[t].append(k)

    #if doX == 1: for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    print(f'\n{timeHMS()}:({doX_s}) Final. Recalculate connectivity')
    # ======= EXPORT. RECALCULATE CONNECTIONS BETWEEN SEGMENTS ========================
    
    t_has_holes_report = {}
    G2_new = graph_check_paths(G, G2, nx.DiGraph(), lr_maxDT, t_has_holes_report)                        
    # ========== EXPORT. DETERMINE TYPE OF EVENTS BETWEEN SEGMENTS ==========
    G2_dir, export_ms_edges_main = get_event_types_from_segment_graph(G2_new)


    trajectories_all_dict[doX]  = t_segments_new
    graphs_all_dict[doX]        = [G,G2_dir]
    events_split_merge_mixed[doX] = export_ms_edges_main

contours_all_dict      = g0_contours

if 1 == 1:
    t_merges_splits_rects = {'merge':defaultdict(list), 'split': defaultdict(list), 'mixed':defaultdict(list)}
    minTime = 100000000; maxTime = 0
    for doX, [G,G2] in graphs_all_dict.items():
        #contours  = contours_all_dict[doX]['contours']
        for node in G.nodes():
            [cx,cy] = G.nodes[node]['centroid']
            del G.nodes[node]['centroid']
            G.nodes[node]['cx'] = cx
            G.nodes[node]['cy'] = cy
            if G.nodes[node]['owner'] == None: G.nodes[node]['owner'] = -1
            t_time, *t_subIDs = node

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

            minTime = min(minTime,  G2.nodes[t_seg]['t_start'   ])
            maxTime = max(maxTime,  G2.nodes[t_seg]['t_end'     ])

            state_from = 'merge'
            if t_seg in events_split_merge_mixed[doX][state_from]:
                t_predecessors  = events_split_merge_mixed[doX][state_from][t_seg]
                t_nodes_min_max = [trajectories_all_dict[doX][t_seg][0]] + [trajectories_all_dict[doX][t_pred][-1] for t_pred in t_predecessors] 
                t_combine_contours = []
                for t_time,*t_subIDs in t_nodes_min_max:
                    for t_subID in t_subIDs:
                        t_combine_contours.append(contours_all_dict[t_time][t_subID])
                t_rb_params = cv2.boundingRect(np.vstack(t_combine_contours))

                t_time_max  = G2.nodes[t_seg]["t_start"]
                t_time_min  = min([G2.nodes[t]["t_end"] for t in t_predecessors])
                
                for t in np.arange(t_time_min, t_time_max + 1):
                    t_merges_splits_rects[state_from][t].append(t_rb_params)

            state_to = 'split'
            if t_seg in events_split_merge_mixed[doX][state_to]:
                t_successors    = events_split_merge_mixed[doX][state_to][t_seg]
                t_nodes_min_max = [trajectories_all_dict[doX][t_seg][-1]] + [trajectories_all_dict[doX][t_succ][0] for t_succ in t_successors]
                t_combine_contours = []
                for t_time,*t_subIDs in t_nodes_min_max:
                    for t_subID in t_subIDs:
                        t_combine_contours.append(contours_all_dict[t_time][t_subID])
                t_rb_params = cv2.boundingRect(np.vstack(t_combine_contours))

                t_time_max  = max([G2.nodes[t]["t_start"] for t in t_successors])
                t_time_min  = G2.nodes[t_seg]["t_end"]
                
                for t in np.arange(t_time_min, t_time_max + 1):
                    t_merges_splits_rects[state_to][t].append(t_rb_params)
        state = 'mixed'
        for t_froms, t_tos in  events_split_merge_mixed[doX][state].items():
            t_nodes_last_from   = [trajectories_all_dict[doX][t_from][-1]   for t_from  in t_froms] 
            t_nodes_first_to    = [trajectories_all_dict[doX][t_to  ][0]    for t_to    in t_tos]
            t_nodes_all         = t_nodes_last_from + t_nodes_first_to
            t_combine_contours = []
            for t_time,*t_subIDs in t_nodes_all:
                for t_subID in t_subIDs:
                    t_combine_contours.append(contours_all_dict[t_time][t_subID])
            t_rb_params = cv2.boundingRect(np.vstack(t_combine_contours))

            t_time_max  = max([G2.nodes[t]["t_start"] for t in t_froms + t_tos])
            t_time_min  = min([G2.nodes[t]["t_end"] for t in t_froms + t_tos])
                
            for t in np.arange(t_time_min, t_time_max + 1):
                t_merges_splits_rects[state][t].append(t_rb_params)



    
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
            
                    t_hull = cv2.convexHull(np.vstack([contours_all_dict[t_time][t_subID] for t_subID in subIDs]))
                    x,y,w,h = cv2.boundingRect(t_hull)
                    str_ID = f'{n}({doX})'
                    cv2.drawContours(  imgs[t_time],  [t_hull], -1, cyclicColor(n), 2)
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