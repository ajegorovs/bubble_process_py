import numpy as np, itertools, networkx as nx, sys, copy,  cv2, os, glob, re, pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict
#from multiprocessing import Pool, log_to_stderr, get_logger

#log_to_stderr()
#logger = get_logger()
#logger.setLevel('INFO')

#import multiprocessing, datetime, random, timeit, time as time_lib, enum
#from PIL import Image
#from tracemalloc import start

# import from custom sub-folders are defined bit lower
#from imageFunctionsP2 import (overlappingRotatedRectangles,graphUniqueComponents)

# ========================================================================================================
# ============== SET NUMBER OF IMAGES ANALYZED =================
# ========================================================================================================
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\350 sccm' #
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\Field OFF\Series 7\350 sccm' #
inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\VFS 125 mT\Series 5\350 sccm' #
#inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 125 mT\Series 1\350 sccm'
# image data subsets are controlled by specifying image index, which is part of an image. e.g image1, image2, image20, image3
intervalStart   = 1                            # start with this ID
numImages       = 3000                          # how many images you want to analyze.
intervalStop    = intervalStart + numImages     # images IDs \elem [intervalStart, intervalStop); start-end will be updated depending on available data.

exportArchive       = 0                         # implies there are no results that can be reused, of you want to force data initialization stage.
useIntermediateData = 1 

useMeanWindow   = 0                             # averaging intervals will overlap half widths, read more below
N               = 700                           # averaging window width
rotateImageBy   = cv2.ROTATE_180                # -1= no rotation, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180 

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

mainOutputFolder            = r'.\post_tests'                           # descritive project name e.g [gallium_bubbles, water_bubbles]
if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)  
mainOutputSubFolders =  ['VFS 125 mT Series 5', 'sccm350-meanFix', 
                         f"{intervalStart:05}-{intervalStop:05}"]       # sub-project folder hierarhy e.g [exp setup, parameter, subset of data]

for folderName in mainOutputSubFolders:     
    mainOutputFolder = os.path.join(mainOutputFolder, folderName)               
    if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)

# -------------------------------- CREATE VARIOUS OUTPUT FOLDERS -------------------------
a = ['images', 'stages',  'archives', 'graphs']
b = ['']*len(a)                                                                

for i,folderName in enumerate(a):   
    tempFolder = os.path.join(mainOutputFolder, folderName)
    if not os.path.exists(tempFolder): os.mkdir(tempFolder)
    b[i] = tempFolder
imageFolder, stagesFolder, dataArchiveFolder, graphsFolder = b
imageFolder_pre_run = 'prerun'
imageFolder_pre_run = os.path.join(imageFolder, imageFolder_pre_run)
if not os.path.exists(imageFolder_pre_run): os.mkdir(imageFolder_pre_run)
imageFolder_output = 'output'
imageFolder_output = os.path.join(imageFolder, imageFolder_output)
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

from graphs_general import (extractNeighborsNext, extractNeighborsPrevious, graph_extract_paths, find_paths_from_to_multi,
                            comb_product_to_graph_edges, for_graph_plots, extract_graph_connected_components, extract_graph_connected_components_autograph,
                            find_segment_connectivity_isolated,  graph_sub_isolate_connected_components, set_custom_node_parameters, get_event_types_from_segment_graph)

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

meanImagePath       = os.path.join(dataArchiveFolder,   "-".join(["mean"                ]+mainOutputSubFolders)+".npz")
meanImagePathArr    = os.path.join(dataArchiveFolder,   "-".join(["meanArr"             ]+mainOutputSubFolders)+".npz")

archivePath         = os.path.join(stagesFolder,        "-".join(["croppedImageArr"     ]+mainOutputSubFolders)+".npz")
binarizedArrPath    = os.path.join(stagesFolder,        "-".join(["binarizedImageArr"   ]+mainOutputSubFolders)+".npz")

graphsPath          = os.path.join(dataArchiveFolder,   "-".join(["graphs"              ]+mainOutputSubFolders)+".pickle")
segmentsPath        = os.path.join(dataArchiveFolder,   "-".join(["segments"            ]+mainOutputSubFolders)+".pickle")
contoursHulls       = os.path.join(dataArchiveFolder,   "-".join(["contorus"            ]+mainOutputSubFolders)+".pickle")
mergeSplitEvents    = os.path.join(dataArchiveFolder,   "-".join(["ms-events"           ]+mainOutputSubFolders)+".pickle")

exportArchive = 1 if not os.path.exists(archivePath) else 0 

if exportArchive == 1:
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

    print(f"\n{timeHMS()}: Processing and saving archive data on drive...")
    
    if rotateImageBy % 2 == 0 and rotateImageBy != -1: W,H = H,W            # for cv2.XXX rotation commands

    dataArchive = np.zeros((len(imageLinks),H,W),np.uint8)                        # predefine storage

    mapXY       = (np.load('./mapx.npy'), np.load('./mapy.npy'))            # fish-eye correction map

    for i,imageLink in tqdm(enumerate(imageLinks), total=len(imageLinks)):
        image = undistort(cv2.imread(imageLink,0), mapXY)[Y:Y+H, X:X+W]
        if rotateImageBy != -1:
            dataArchive[i]    = cv2.rotate(image, rotateImageBy)
        else:
            dataArchive[i]    = image

    print(f"\n{timeHMS()}: Processing and saving archive data on drive...saving compressed")
    np.savez_compressed(archivePath,dataArchive)
    
    print(f"\n{timeHMS()}: Exporting mean image...calculating mean")

    meanImage = np.mean(dataArchive, axis=0)
    print(f"\n{timeHMS()}: Exporting mean image...saving compressed")
    np.savez_compressed(meanImagePath,meanImage)
      
    print(f"\n{timeHMS()}: Processing and saving archive data on drive... Done!")

    useIntermediateData = 0           # no need to re-read data you have just created

elif not os.path.exists(archivePath): # did not find ropped image array.
    print(f"\n{timeHMS()}: No archive detected! Please generate it from project images. set exportArchive = 1")

elif useIntermediateData:
    print(f"\n{timeHMS()}: Existing archive found! Importing data...")
    dataArchive = np.load(archivePath)['arr_0']

    print(f"\n{timeHMS()}: Existing archive found! Importing data... Done!")

    if not os.path.exists(meanImagePath):
        print(f"\n{timeHMS()}: No mean image found... Calculating mean")

        meanImage = np.mean(dataArchive, axis=0)
        print(f"\n{timeHMS()}: No mean image found... Saving compressed")
        np.savez_compressed(meanImagePath,meanImage)

        print(f"\n{timeHMS()}: No mean image found... Done")
    else:
        meanImage = np.load(meanImagePath)['arr_0']
else:
    print(f"\n{timeHMS()}: Did not generate new data, nor imported existing... set exportArchive = 1 or useIntermediateData = 1")

#cv2.imshow(f'mean',meanImage.astype(np.uint8))

# =========================================================================================================
# discrete update moving average with window N, with intervcal overlap of N/2
# [-interval1-]         for first segment: interval [0,N]. switch to next window at i = 3/4*N,
#           |           which is middle of overlap. 
#       [-interval2-]   for second segment: inteval is [i-1/4*N, i+3/4*N]
#                 |     third switch 1/4*N +2*[i-1/4*N, i+3/4*N] and so on. N/2 between switches

if useMeanWindow == 1 and not useIntermediateData:
    meanIndicies = np.arange(0,dataArchive.shape[0],1)                                           # index all images
    meanWindows = {}                                                                             # define timesteps at which averaging
    meanWindows[0] = [0,N]                                                                       # window is switched. eg at 0 use
                                                                                                 # np.mean(archive[0:N])
    meanSwitchPoints = np.array(1/4*N + 1/2*N*np.arange(1, int(len(meanIndicies)/(N/2)), 1), int)# next switch points, by geom construct
                                                                                                 # 
    for t in meanSwitchPoints:                                                                   # intervals at switch points
        meanWindows[t] = [t-int(1/4*N),min(t+int(3/4*N),max(meanIndicies))]                      # intervals have an overlap of N/2
    meanWindows[meanSwitchPoints[-1]] = [meanWindows[meanSwitchPoints[-1]][0],max(meanIndicies)] # modify last to include to the end
    intervalIndecies = {t:i for i,t in enumerate(meanWindows)}                                   # index switch points {i1:0, i2:1, ...}
                                                                                                 # so i1 is zeroth interval
    print(meanWindows)                                                                                       
    print(intervalIndecies)

    if not os.path.exists(meanImagePathArr):
        print(f"\n{timeHMS()}: Mean window is enabled. No mean image array found. Generating and saving new...")
        masksArr = np.array([np.mean(dataArchive[start:stop], axis=0) for start,stop in meanWindows.values()])   # array of discrete averages

        with open(meanImagePathArr, 'wb') as handle:
            pickle.dump(masksArr, handle)
        print(f"\n{timeHMS()}: Mean window is enabled. No mean image array found. Generating and saving new... Done")
                                                     

    else:
        print(f"\n{timeHMS()}: Mean window is enabled. Mean image array found. Importing data...")
        with open(meanImagePathArr, 'rb') as handle:
                masksArr = pickle.load(handle)
        print(f"\n{timeHMS()}: Mean window is enabled. Mean image array found. Importing data... Done!")

def whichMaskInterval(t,order):                                                                          # as frames go 0,1,..numImgs
    times = np.array(list(order))                                                                        # mean should be taken form the left
    sol = 0                                                                                              # img0:[0,N],img200:[i-a,i+b],...
    for time in times:                                                                                   # so img 199 should use img0 interval
        if time <= t:sol = time                                                                          # EZ sol just to interate and comare 
        else: break                                                                                      # and keep last one that satisfies
                                                                                                         # 
    return order[sol]      

if not useIntermediateData:           # this will pre-calculate some stuff in case of analysis needed.

    def adjustBrightness(image):
        brightness = np.sum(image) / (255 * np.prod(image.shape))
        minimum_brightness = 0.66
        ratio = brightness / minimum_brightness
        if ratio >= 1:
            return 1
        return ratio    #cv2.convertScaleAbs(image, alpha = 1 / ratio, beta = 0)
        
    do_stage = False
    if useMeanWindow and not os.path.exists(binarizedArrPath): # want mean window and archive does not exist.
        binarizedMaskArr = np.zeros(dataArchive.shape)
        for k in range(dataArchive.shape[0]):
            blurMean = cv2.blur(masksArr[whichMaskInterval(k,intervalIndecies)], (5,5),cv2.BORDER_REFLECT)
            binarizedMaskArr[k] = dataArchive[k] - blurMean   
        #meanImage = masksArr[whichMaskInterval(globalCounter,intervalIndecies)] # copied from diff code
        do_stage = True

    elif not os.path.exists(binarizedArrPath):                # dont want mean window and archive does not exist
        blurMean = cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT)
        #cv2.imshow('1',np.uint8(blurMean))
        binarizedMaskArr = dataArchive - blurMean             # substract mean image from stack -> float
        #cv2.imshow('1',np.uint8(binarizedMaskArr[20]))
        do_stage = True

    if do_stage:                  # archive does not exist
        ratio = 1 #adjustBrightness(np.uint8(blurMean))
        #binarizedMaskArr = np.array([cv2.convertScaleAbs(img, alpha = ratio, beta = 0) for img in binarizedMaskArr])
        #cv2.imshow('2',np.uint8(binarizedMaskArr[20]))
        imgH,imgW = blurMean.shape
        thresh0 = 10
        binarizedMaskArr = np.where(binarizedMaskArr < thresh0, 0, 255).astype(np.uint8)                        # binarize stack
        #cv2.imshow('3',np.uint8(binarizedMaskArr[20]))
        binarizedMaskArr = np.uint8([cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) for img in binarizedMaskArr])    # delete white objects
        #cv2.imshow('4',np.uint8(binarizedMaskArr[20]))
        binarizedMaskArr = np.uint8([cv2.dilate(img, np.ones((8,8), np.uint8) ) for img in binarizedMaskArr])    
        #cv2.imshow('5',np.uint8(binarizedMaskArr[20]))
        binarizedMaskArr = np.uint8([cv2.erode(img, np.ones((5,5), np.uint8) ) for img in binarizedMaskArr]) 
        #cv2.imshow('6',np.uint8(binarizedMaskArr[20]))
        print(f"{timeHMS()}: Removing small and edge contours...")
        topFilter, bottomFilter, leftFilter, rightFilter, minArea    = 80, 40, 100, 100, 180
        for i in tqdm(range(binarizedMaskArr.shape[0]), total=binarizedMaskArr.shape[0]):
            contours            = cv2.findContours(binarizedMaskArr[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
            areas               = np.array([int(cv2.contourArea(contour)) for contour in contours])
            boundingRectangles  = np.array([cv2.boundingRect(contour) for contour in contours])
            whereSmall      = np.argwhere(areas < minArea)
            
            # img coords: x increases from left to right; y increases from top to bottom (not usual)
            topCoords       = np.array([y+h for x,y,w,h in boundingRectangles])     # bottom    b-box coords
            bottomCoords    = np.array([y   for x,y,w,h in boundingRectangles])     # top       b-box coords
            leftCoords      = np.array([x+w for x,y,w,h in boundingRectangles])     # right     b-box coords
            rightCoords     = np.array([x   for x,y,w,h in boundingRectangles])     # left      b-box coords

            whereTop    = np.argwhere(topCoords     < topFilter)                    # bottom of b-box is within top band
            whereBottom = np.argwhere(bottomCoords  > (imgH - topFilter))           # top of b-box is within bottom band
            whereLeft   = np.argwhere(leftCoords    < leftFilter)                   # -"-"-
            whereRight  = np.argwhere(rightCoords   > (imgW - rightFilter))         # -"-"-
                                                                             
            whereFailed = np.concatenate((whereSmall, whereTop, whereBottom, whereLeft, whereRight)).flatten()
            whereFailed = np.unique(whereFailed)

            # draw over black (cover) border elements
            [cv2.drawContours(  binarizedMaskArr[i],   contours, j, 0, -1) for j in whereFailed]
                

        print(f"\n{timeHMS()}: Removing small and edge contours... Done")
        print(f"\n{timeHMS()}: Binarized Array archive not found... Saving")
        np.savez_compressed(binarizedArrPath,binarizedMaskArr)
        print(f"\n{timeHMS()}: Binarized Array archive not found... Done")
    else:
        print(f"\n{timeHMS()}: Binarized Array archive located... Loading")
        binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
        print(f"\n{timeHMS()}: Binarized Array archive located... Done")

    #err             = cv2.morphologyEx(err.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    print(f"\n{timeHMS()}: First Pass: obtaining rough clusters using bounding rectangles...")
    t_range              = range(binarizedMaskArr.shape[0])
    g0_bigBoundingRect   = {t:[] for t in t_range}
    g0_bigBoundingRect2  = {t:{} for t in t_range}
    g0_clusters          = {t:[] for t in t_range}
    g0_clusters2         = {t:[] for t in t_range}
    g0_contours          = {t:[] for t in t_range}
    g0_contours_children = {t:{} for t in t_range}
    g0_contours_hulls    = {t:{} for t in t_range}
    g0_contours_centroids= {t:{} for t in t_range} 
    g0_contours_areas    = {t:{} for t in t_range}

    # stage 1: extract contours on each binary image. delete small and internal contours.
    # check for proximity between contours on frame using overlap of bounding rectangles
    # small objects are less likely to overlap, boost their bounding rectangle to 100x100 pix size
    # overlapping objects are clusters with ID (time_step,) + tuple(subIDs). contour params are stored

    for i in tqdm(range(binarizedMaskArr.shape[0]), total = binarizedMaskArr.shape[0]):
        # find all local contours
        #contours            = cv2.findContours(binarizedMaskArr[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours, hierarchy = cv2.findContours(binarizedMaskArr[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
        if hierarchy is None:
            whereParentCs = []
        else:
            whereParentCs   = np.argwhere(hierarchy[:,:,3]==-1)[:,1] #[:,:,3] = -1 <=> (no owner)
                
        whereChildrenCs = {parent:np.argwhere(hierarchy[:,:,3]==parent)[:,1].tolist() for parent in whereParentCs}
        whereChildrenCs = {parent: children for parent,children in whereChildrenCs.items() if len(children) > 0}
        childrenContours = list(sum(whereChildrenCs.values(),[]))
        areasChildren   = {k:int(cv2.contourArea(contours[k])) for k in childrenContours}
        minChildrenArea = 120
        whereSmallChild = [k for k,area in areasChildren.items() if area <= minChildrenArea]
        whereChildrenCs = {parent: [c for c in children if c not in whereSmallChild] for parent,children in whereChildrenCs.items()}
        whereChildrenCs = {parent: children for parent,children in whereChildrenCs.items() if len(children) > 0}
        
        a = 1
        # get bounding rectangle of each contour
        boundingRectangles  = {pID: cv2.boundingRect(contours[pID]) for pID in whereParentCs}#np.array([cv2.boundingRect(contour) for contour in contours])
        # expand bounding rectangle for small contours to 100x100 box
        brectDict = {pID: modBR(brec,100) for pID,brec in boundingRectangles.items()} #np.array([modBR(brec,100) for brec in boundingRectangles])
        # for a dictionary of bounding rect and local IDs
        #brectDict = {k:brec for k,brec in enumerate(boundingRectangles2)}
        # check overlap of bounding rectangles, hint on clusters
        combosSelf = np.array(overlappingRotatedRectangles(brectDict,brectDict))
        # get cluster of overlapping bounding boxes
        cc_unique  = graphUniqueComponents(list(brectDict.keys()), combosSelf) 
        # create a bounding box of of clusters
        bigBoundingRect     = [cv2.boundingRect(np.vstack([rect2contour(brectDict[k]) for k in comb])) for comb in cc_unique]

        g0_contours[i]          = contours
        g0_contours_children[i] = whereChildrenCs
        g0_clusters[i]          = cc_unique
        g0_bigBoundingRect[i]   = bigBoundingRect
        for k,subElems in enumerate(cc_unique):
            key = tuple([i]+subElems)
            g0_bigBoundingRect2[i][key] = cv2.boundingRect(np.vstack([rect2contour(brectDict[c]) for c in subElems]))
        
        g0_contours_centroids[i]= np.zeros((len(contours),2))
        g0_contours_areas[i]    = np.zeros(len(contours), int)
        g0_contours_hulls[i]    = [None]*len(contours)
        for k,t_contour in enumerate(contours):
            t_hull                        = cv2.convexHull(t_contour)
            t_centroid,t_area             = centroid_area(t_hull)
            g0_contours_hulls[      i][k] = t_hull
            g0_contours_centroids[  i][k] = t_centroid
            g0_contours_areas[      i][k] = int(t_area)
        #img = binarizedMaskArr[i].copy()
        #[cv2.rectangle(img, (x,y), (x+w,y+h), 128, 1) for x,y,w,h in bigBoundingRect]
        #cv2.imshow('a',img)
        #a = 1
    print(f"\n{timeHMS()}: First Pass: obtaining rough clusters using bounding rectangles...Done!")


    # stage 2: check overlap of clusters on two consequetive frames
    # inter-frame connections are stored
    print(f"\n{timeHMS()}: First Pass: forming inter-frame relations for rough clusters...")

    debug = 0
    g0_clusterConnections  = {ID:[] for ID in range(binarizedMaskArr.shape[0]-1)}
    g0_clusterConnectionsRepr = g0_clusterConnections.copy()
    g0_pairConnections  = []
    times = np.array(list(g0_bigBoundingRect2.keys()))
    for t in tqdm(times[:-2]):
        # grab this and next frame cluster bounding boxes
        oldBRs = g0_bigBoundingRect2[t]
        newBrs = g0_bigBoundingRect2[t+1]
        # grab all cluster IDs on these frames
        allKeys = list(oldBRs.keys()) + list(newBrs.keys())
        # find overlaps between frames
        combosSelf = overlappingRotatedRectangles(oldBRs,newBrs)                                       
        for conn in combosSelf:
            assert len(conn) == 2, "overlap connects more than 2 elems, not designed for"
            pairCons = list(itertools.combinations(conn, 2))
            pairCons2 = sorted(pairCons, key=lambda x: [a[0] for a in x])
            [g0_pairConnections.append(x) for x in pairCons2]
        cc_unique  = graphUniqueComponents(allKeys, combosSelf)                                       
        g0_clusterConnections[t] = cc_unique
        if t == -1:                                                                                 #  can test 2 frames
            #imgs = [convertGray2RGB(binarizedMaskArr[t].copy()), convertGray2RGB(binarizedMaskArr[t+1].copy())]
            img = convertGray2RGB(np.uint8(np.mean(binarizedMaskArr[t:t+2],axis = 0)))
            rectParamsAll = {**oldBRs,**newBrs}
            for k,comp in enumerate(cc_unique):
                for c in comp:
                    frame = c[0]-t
                    x,y,w,h = rectParamsAll[c]                                                         # connected clusters = same color
                    #cv2.rectangle(imgs[frame], (x,y), (x+w,y+h), cyclicColor(k), 1)
                    cv2.rectangle(img, (x,y), (x+w,y+h), cyclicColor(k), 1)
            #cv2.imshow(f'iter:{t} old',imgs[0])
            #cv2.imshow(f'iter:{t} new',imgs[1])
            cv2.imshow(f'iter:{t}',img)

        a = 1
    # stage 3: create a graph from inter-frame connections and extract connected components
    # these are bubble trail families which contains bubbles that came into close contact
    # which can be roughly considered to have merged/split. but its refined later.
    allIDs = sum([list(g0_bigBoundingRect2[t].keys()) for t in g0_bigBoundingRect2],[])

    # form a graph from all IDs and pairwise connections
    H = nx.Graph()
    #H = nx.DiGraph()
    H.add_nodes_from(allIDs)
    H.add_edges_from(g0_pairConnections)
    connected_components_unique = [sorted(c, key = lambda x: (x[0], x[1])) for c in nx.connected_components(H)]
    connected_components_unique = sorted(connected_components_unique, key = lambda x: x[0][0])
    #connected_components_unique = extract_graph_connected_components(H, sort_function = lambda x: (x[0], x[1]))
    print(f"\n{timeHMS()}: Storing intermediate data...")
    if 1 == 1:
        storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
        with open(storeDir, 'wb') as handle:
            pickle.dump(
            [
                g0_bigBoundingRect2, g0_clusters2, g0_contours,g0_contours_hulls, g0_contours_centroids,
                g0_contours_areas, g0_pairConnections, H, connected_components_unique, g0_contours_children
            ], handle) 
        print(f"\n{timeHMS()}: Storing intermediate data...Done")
else:
    print(f"\n{timeHMS()}: Begin loading intermediate data...")
    storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
    with open(storeDir, 'rb') as handle:
        [
            g0_bigBoundingRect2, g0_clusters2, g0_contours,g0_contours_hulls, g0_contours_centroids,
            g0_contours_areas, g0_pairConnections, H, connected_components_unique, g0_contours_children
        ] = pickle.load(handle)
    print(f"\n{timeHMS()}: Begin loading intermediate data...Done!, test sub-case")

connected_components_unique_0 = copy.deepcopy(connected_components_unique)
H_0 = copy.deepcopy(H)
# ======================
if 1 == -1:
    print(f"\n{timeHMS()}: Pre - run images: Generating images")
    segments2 = connected_components_unique_0
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
            x,y,w,h = g0_bigBoundingRect2[time][subCase]
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
t_all_areas = []
for t_areas in g0_contours_areas.values():
    t_all_areas += t_areas.tolist()


g0_area_mean    = np.mean(t_all_areas)
g0_area_std     = np.std(t_all_areas)

if 1 == -1:
    plt.hist(t_all_areas, bins=70, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.title('Histogram of Animal Heights')      # LOL

    # Show the plot
   #plt.show()


a = 1

# analyze single strand
doX = 30#60#30#60#56#20#18#2#84#30#60#30#20#15#2#1#60#84
#lessRough_all = copy.deepcopy(connected_components_unique_0)

trajectories_all_dict = {}
graphs_all_dict = {}
contours_all_dict = {}
events_split_merge_mixed = {}

issues_all_dict = defaultdict(dict) #[2]:#[19]:#
temp = [59]
#temp = [k for k, seg in enumerate(connected_components_unique_0) if len(seg) > 70]
#temp = [k for k, seg in enumerate(connected_components_unique_0) if len(seg) < 70]
temp = range(len(connected_components_unique_0))
max_width = len(str(temp[-1]))

for doX in temp:
    doX_s = f"{doX:0{max_width}}"
    print(f'\n\n\n{timeHMS()}:({doX_s}) working on family {doX}')
    test = connected_components_unique_0[doX]                       # specific family of nodes

    # isolate a family of nodes into new graph. i avoid using graph.subgraph here because it will still store old data.
    H = nx.DiGraph()
    for t_node in test:
        H.add_node(t_node, **H_0.nodes[t_node])

    segments2, skipped = graph_extract_paths(H, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

    segments2 = [a for _,a in segments2.items() if len(a) > 0]
    segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
    paths = {i:vals for i,vals in enumerate(segments2)}
    # for_graph_plots(H)    # <<<<<<<<<<<<<<

    # ===============================================================================================
    # ===============================================================================================
    # =============== Refine expanded BR overlap subsections with solo contours =====================
    # ===============================================================================================
    # ===============================================================================================
    # less rough relations pass
    # drop idea of expanded bounding rectangle.
    # 1 contour cluster is the best case.



    activeNodes = list(H.nodes())
    activeTimes = np.unique([a[0] for a in activeNodes])

    # extract all contours active at each time step.
    lessRoughIDs = {t:[] for t in activeTimes}
    for time, *subIDs in activeNodes:
        lessRoughIDs[time] += subIDs

    lessRoughIDs = {time: sorted(list(set(subIDs))) for time, subIDs in lessRoughIDs.items()}

    lessRoughBRs = {time: {tuple([time, ID]):cv2.boundingRect(g0_contours[time][ID]) for ID in subIDs} for time, subIDs in lessRoughIDs.items()}

     # grab this and next frame cluster bounding boxes
    print(f'\n{timeHMS()}:({doX_s}) generating rough contour overlaps')
    g0_pairConnections2  = []
    for t in tqdm(activeTimes[:-2]):
        oldBRs = lessRoughBRs[t]
        newBrs = lessRoughBRs[t+1]
        # grab all cluster IDs on these frames
        allKeys = list(oldBRs.keys()) + list(newBrs.keys())
        # find overlaps between frames
        combosSelf = overlappingRotatedRectangles(oldBRs,newBrs)                                       
        for conn in combosSelf:
            assert len(conn) == 2, "overlap connects more than 2 elems, not designed for"
            pairCons = list(itertools.combinations(conn, 2))
            pairCons2 = sorted(pairCons, key=lambda x: [a[0] for a in x])
            [g0_pairConnections2.append(x) for x in pairCons2]

    if 1 == -1:
        print(f"\n{timeHMS()}:({doX_s}) Refining small BR overlap with c-c distance...")
        # ===============================================================================================
        # ===============================================================================================
        # ================ Refine two large close passing bubble pre-merge connection ===================
        # ===============================================================================================
        # REMARK: to reduce stress future node permutation computation we can detect cases where two
        # REMARK: bubbles pass in close proximity. bounding rectangle method connects them due to higher
        # REMARK: bounding rectange overlap chance. Characteristics of these cases are: bubble (node) is
        # REMARK: connected to future itself and future neighbor. Difference is that distance to itself
        # REMARK: is much smaller. Proper distance is closest contour-contour distance (which is slow)

        t_temp_dists = {t_edge:0 for t_edge in g0_pairConnections2}
        for t_edge in tqdm(g0_pairConnections2):                                                        # examine all edges
            t_node_from , t_node_to   = t_edge                                                           

            t_time_from , t_ID_from   = t_node_from                                                      
            t_time_to   , t_ID_to     = t_node_to                                                        
                                                                                                 
            t_contour_from_area = g0_contours_areas[t_time_from ][t_ID_from ]                           # grab contour are of node_from 
            t_contour_to_area   = g0_contours_areas[t_time_to   ][t_ID_to   ]                           
            if t_contour_from_area >= 0.5*g0_area_mean and t_contour_to_area >= 0.5*g0_area_mean:       # if both are relatively large
                t_contour_from  = g0_contours[t_time_from ][t_ID_from ]                                 # grab contours
                t_contour_to    = g0_contours[t_time_to   ][t_ID_to   ]                                  
                t_temp_dists[t_edge] = closes_point_contours(t_contour_from,t_contour_to, step = 5)[1]  # find closest contour-contour distance
                                                                                                 
        t_temp_dists_large  = [t_edge   for t_edge, t_dist      in t_temp_dists.items() if t_dist >= 7] # extract edges with large dist
        t_edges_from        = [t_node_1 for t_node_1, _         in t_temp_dists_large]                  # extract 'from' node, it may fork into multiple
        t_duplicates_pre    = [t_edge   for t_edge in t_temp_dists if t_edge[0] in t_edges_from]        # find edges that start from possible forking nodes
        t_duplicates_edges_pre_from = [t_node_1 for t_node_1, _ in t_duplicates_pre]                    # extract 'from' node, to count fork branches

        from collections import Counter                                                                  
        t_edges_from_count  = Counter(t_duplicates_edges_pre_from)                                      # count forking edges 
        t_edges_from_count_two = [t_node for t_node,t_num in t_edges_from_count.items() if t_num == 2]  # extract cases with 2 connections- future-
                                                                                                        # itself and future neighbor
        t_edges_combs = {t_node:[] for t_node in t_edges_from_count_two}                                # prep t_from_node:neighbors storage

        for t_node_1, t_node_2 in t_temp_dists:                                                         # walk though all edges
            if t_node_1 in t_edges_from_count_two:                                                      # visit relevant
                t_edges_combs[t_node_1] += [t_node_2]                                                   # store neighbors
                                                                                                        # filter out cases where big nodes
        t_edges_combs_dists =   {t_node:                                                                # grab distances for these edges
                                    {t_neighbor:t_temp_dists[(t_node,t_neighbor)] for t_neighbor in t_neighbors} 
                                        for t_node, t_neighbors in t_edges_combs.items()}

        g0_edges_merge_strong   = [(0,0)]*len(t_edges_combs_dists)                                      # predefine storage (bit faster)
        g0_edges_merge_weak     = [(0,0)]*len(t_edges_combs_dists)
        for t_k,(t_node_from, t_neighbor_dists) in enumerate(t_edges_combs_dists.items()):

            t_node_dist_small   = min(t_neighbor_dists, key = t_neighbor_dists.get)                     # small distance = most likely connection
            t_node_dist_big     = max(t_neighbor_dists, key = t_neighbor_dists.get)                     # opposite

            g0_edges_merge_strong[  t_k] = (  t_node_from, t_node_dist_small  )                         # these will be asigned as extra property
            g0_edges_merge_weak[    t_k] = (  t_node_from, t_node_dist_big    )                         # for future analysis on graph formation

        G_e_m = nx.Graph()
        G_e_m.add_edges_from(g0_edges_merge_strong)
        g0_edges_merge_strong_clusters = extract_graph_connected_components(G_e_m, lambda x: (x[0],x[1]))
        print(f"\n{timeHMS()}:({doX_s}) Refining small BR overlap with c-c distance... done")                     

    a = 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7; thickness = 4;

    # for_graph_plots(G)    # <<<<<<<<<<<<<<
    # ===============================================================================================
    # ===============================================================================================
    # =========== Extract solo-to-solo bubble trajectories from less rough graphs ===================
    # ===============================================================================================
    # REMARK: it is very likely that solo-to-solo (later called 121) is pseudo split-merge, optical effect

    allIDs = sum([list(a.keys()) for a in lessRoughBRs.values()],[])
    allIDs = sorted(allIDs, key=lambda x: (x[0], x[1]))

    G = nx.DiGraph()
    G.add_nodes_from(allIDs)
    G.add_edges_from(g0_pairConnections2)


    set_custom_node_parameters(G, g0_contours_hulls, G.nodes(), None, calc_hull = 0) # pre define params and init owner to None 

    G_time      = lambda node, graph = G : graph.nodes[node]['time']
    G_area      = lambda node, graph = G : graph.nodes[node]['area']
    G_centroid  = lambda node, graph = G : graph.nodes[node]['centroid']
    G_owner     = lambda node, graph = G : graph.nodes[node]['owner']

    if 1 == -1:
        #for t_edge in g0_edges_merge_strong:                                          # assign previously acquired 2 bubble merge 
        #    G[t_edge[0]][t_edge[1]]['edge_merge_strong'] = True                       # edge of high of high likelihood

        #for t_edge in g0_edges_merge_weak:
        #    G[t_edge[0]][t_edge[1]]['edge_merge_strong'] = False

        for t_edge in G.edges():
            (t_time_from,t_ID_from),(t_time_to,t_ID_to) = t_edge
            t_area_before   = g0_contours_areas[t_time_from ][t_ID_from ]
            t_area_after    = g0_contours_areas[t_time_to   ][t_ID_to   ]
            t_relative_area_change = np.abs((t_area_after-t_area_before)/t_area_before)

            t_centroids_before  = g0_contours_centroids[t_time_from ][t_ID_from ]
            t_centroids_after   = g0_contours_centroids[t_time_to   ][t_ID_to   ]
            t_edge_distance     = np.linalg.norm(t_centroids_after-t_centroids_before)

            G[t_edge[0]][t_edge[1]]['edge_rel_area_change'  ] = t_relative_area_change
            G[t_edge[0]][t_edge[1]]['edge_distance'         ] = t_edge_distance


        distances       = nx.get_edge_attributes(G, 'edge_distance'         )
        area_changes    = nx.get_edge_attributes(G, 'edge_rel_area_change'  )
        tt = []
        tt2 = []
        for path in g0_edges_merge_strong_clusters:
            path_distances      = [distances[edge] for edge in zip(path, path[1:])]
            path_area_changes   = [area_changes[edge] for edge in zip(path, path[1:])]
            tt.append(path_distances)
            tt2.append(path_area_changes)
        t_edges_strong_2 = []
        for t_nodes_path, t_dists, t_areas_rel in zip(g0_edges_merge_strong_clusters,tt,tt2):
            t_dist_mean = np.mean(t_dists)
            t_dist_std  = np.std( t_dists)
            t_dist_low_max          = True if max(t_dists)              <  20    else False
            t_dist_low_dispersion   = True if 2*t_dist_std/t_dist_mean  <= 0.2   else False
            t_area_rel_low_max      = True if max(t_areas_rel)          <  0.2   else False
            if t_dist_low_max and t_dist_low_dispersion and t_area_rel_low_max:
                t_edges = [t_edge for t_edge in zip(t_nodes_path, t_nodes_path[1:])]
                t_edges_strong_2 += t_edges

        t_edges_strong_node_start = [t_edge[0] for t_edge in t_edges_strong_2]
        t_edges_weak_discard = []
        for t_edge in g0_edges_merge_weak:
            if t_edge[0] in t_edges_strong_node_start:
                t_edges_weak_discard.append(t_edge)

        G.remove_edges_from(t_edges_weak_discard)
        g0_pairConnections2 = [t_edge for t_edge in g0_pairConnections2 if t_edge not in t_edges_weak_discard]

        g0_pairConnections2_OG = copy.deepcopy(g0_pairConnections2)
    a = 1
    print(f'\n{timeHMS()}:({doX_s}) Detecting frozen bubbles ... ')
    f = lambda x : x[0]

    segments_fb, skipped = graph_extract_paths(G, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

    segments_fb = [a for _,a in segments_fb.items() if len(a) > 2]  # >>>>>>>> LIMIT MIN LENGTH
    segments_fb = list(sorted(segments_fb, key=lambda x: x[0][0]))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # i want to solve frozen bubbles here. how to do it:
    # 1) check dispacements of segments- mean displacement (not total) will be very low.
    # 2) detemine which frozen segments are the same bubble
    # 3) get time steps between frozen segments and isolate nodes

    # 1) calc mean displacement, isolate segments with low displ, remember mean position
    fb_radiuss = 5
    segments_small_displ_mean_centroids = {}
    segments_small_displ_areas_mean_std = {}
    for t,t_nodes in enumerate(segments_fb):
        if len(t_nodes) >= 2 :
            #t_traj = np.array([G.nodes[t_node]['centroid'] for t_node in t_nodes])
            t_traj = np.array([G_centroid(t_node) for t_node in t_nodes])
            t_displ_all = np.linalg.norm(np.diff(t_traj, axis = 0), axis = 1)
            if np.mean(t_displ_all) <= fb_radiuss:
                segments_small_displ_mean_centroids[t] = np.mean(t_traj, axis = 0)
                #t_areas = np.array([G.nodes[t_node]['area'] for t_node in t_nodes])
                t_areas = np.array([G_area(t_node) for t_node in t_nodes])
                segments_small_displ_areas_mean_std[t] = (np.mean(t_areas),np.std(t_areas))

    a = 1
    # 2)
    # check which segments have similar mean position -  most likely same frozen bubble
    
    fb_edges_test = list(itertools.combinations(segments_small_displ_mean_centroids, 2))
    fb_edges_dist_pass = []
    for a,b in fb_edges_test:
        t_dist = np.linalg.norm(segments_small_displ_mean_centroids[a] - segments_small_displ_mean_centroids[b])
        if t_dist <= fb_radiuss:
            fb_edges_dist_pass.append((a,b))
    
    fb_segment_clusters = extract_graph_connected_components_autograph(fb_edges_dist_pass)

    fb_segment_clusters = [sorted(a, key = lambda x: segments_fb[x][0][0]) for a in fb_segment_clusters]

    # extract times between fb segments.
    fb_hole_stats_IDs   = []
    fb_hole_stats       = {'times': {},'area_min_max': {}}
    for t_cluster in fb_segment_clusters:
        for t_edge in zip(t_cluster[:-1], t_cluster[1:]): # neighbor pairs
            (t_from,t_to) = t_edge
            # estimate range of areas for unresolved fb bubbles in inter-segment node space.
            t_area_mean_1, t_area_std_1 = segments_small_displ_areas_mean_std[t_from]
            t_area_mean_2, t_area_std_2 = segments_small_displ_areas_mean_std[t_to]
            if t_area_mean_1 > t_area_mean_2:
                t_area_min_max = (t_area_mean_2 - 5*t_area_std_2, t_area_mean_1 + 5*t_area_std_1)
            else:
                t_area_min_max = (t_area_mean_1 - 5*t_area_std_1, t_area_mean_2 + 5*t_area_std_2)

            fb_hole_stats_IDs.append(t_edge)
            fb_hole_stats['times'          ][t_edge] = np.arange(segments_fb[t_from][-1][0] + 1, segments_fb[t_to][0][0])
            fb_hole_stats['area_min_max'   ][t_edge] = t_area_min_max

    a = 1
    fb_remove_nodes_inter = []
    for t_edge in fb_hole_stats_IDs:
        t_times                 = fb_hole_stats['times'         ][t_edge]
        t_area_min,t_area_max   = fb_hole_stats['area_min_max'  ][t_edge]
        t_nodes_active = [t_node for t_node in G.nodes if G_time(t_node) in t_times]
        (t_from,t_to) = t_edge
        t_centroid_target = 0.5*(segments_small_displ_mean_centroids[t_from] + segments_small_displ_mean_centroids[t_to])
        for t_node in t_nodes_active:
            t_dist = np.linalg.norm(t_centroid_target - G_centroid(t_node))
            if t_dist <= fb_radiuss and (t_area_min <= G_area(t_node) <= t_area_max):
                fb_remove_nodes_inter.append(t_node)
    a = 1
    t_remove_nodes = []
    G.remove_nodes_from(fb_remove_nodes_inter)
    for t_segment_IDs in fb_segment_clusters:
        for t_segment_ID in t_segment_IDs:
            t_nodes = segments_fb[t_segment_ID]
            G.remove_nodes_from(t_nodes)
            t_remove_nodes.extend(t_nodes)

    allIDs = [t_node for t_node in allIDs if t_node not in t_remove_nodes + fb_remove_nodes_inter]

    print(f'\n{timeHMS()}:({doX_s}) Detecting frozen bubbles ... DONE')
    segments2, skipped = graph_extract_paths(G, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

    segments2 = [a for _,a in segments2.items() if len(a) > 2]  # >>>>>>>> LIMIT MIN LENGTH
    segments2 = list(sorted(segments2, key=lambda x: x[0][0]))

    for t,t_segment in enumerate(segments2): # assign owners params
        for t_node in t_segment:
            G.nodes[t_node]["owner"] = t

    lr_nodes_non_segment = [node for node in G.nodes if G_owner(node) is None] # !! may be same as &skipped !!

    # add segment abstraction level graph (no individual G nodes)
    G2 = nx.Graph()
    for t_seg_index in range(len(segments2)):

        G2.add_node(t_seg_index)

        G2.nodes()[t_seg_index]["t_start"   ] = G_time(segments2[t_seg_index][0])
        G2.nodes()[t_seg_index]["t_end"     ] = G_time(segments2[t_seg_index][-1])

    G2_t_start  = lambda node, graph = G2 : graph.nodes[node]['t_start' ]
    G2_t_end    = lambda node, graph = G2 : graph.nodes[node]['t_end'   ]
    G2_edge_dist= lambda edge, graph = G2 : graph.edges[edge]['dist'   ]

    lr_time_active_segments = defaultdict(list)
    for t_segment_index, t_segment_nodes in enumerate(segments2):
        t_times = [G_time(node) for node in t_segment_nodes]
        for t_time in t_times:
            lr_time_active_segments[t_time].append(t_segment_index)
    # sort keys in lr_time_active_segments
    lr_time_active_segments = {t:lr_time_active_segments[t] for t in sorted(lr_time_active_segments.keys())}

    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Determining connectivity between segments... ')
    # ===============================================================================================
    # ===============================================================================================
    # === find POSSIBLE interval start-end connectedness: start-end exist within set time interval ==
    # ===============================================================================================
    # ===============================================================================================
    # REMARK: it is expected that INTER segment space is limited, so no point searching paths from
    # REMARK: one segment to each other in graph. instead inspect certain intervals of set length 
    # REMARK: lr_maxDT. Caveat is that space between can include other segments.. its dealt with after

    # get start and end of good segments
    lr_seg_nodes_start_end  = [[t[0],t[-1]] for t in segments2]
    lr_all_start            = np.array([a[0][0] for a in lr_seg_nodes_start_end])
    lr_seg_end_all          = np.array([a[1][0] for a in lr_seg_nodes_start_end])

    # get nodes that begin not as a part of a segment = node with no prior neigbors
    lr_nodes_other = []
    lr_nodes_solo = []
    for t_node in lr_nodes_non_segment:
        oldNeigbors = list(G.predecessors(  t_node))
        newNeigbors = list(G.successors(    t_node))
        if      len(oldNeigbors) == 0 and len(newNeigbors) == 0:lr_nodes_solo.append(   t_node)
        elif    len(oldNeigbors) == 0:                          lr_nodes_other.append(  t_node)

    # find which segements are separated by small number of time steps
    #  might not be connected, but can check later
    lr_maxDT = 60

    # test segments. search for endpoint-to-startpoint DT

    lr_DTPass_segm = {}
    for k,endTime in enumerate(lr_seg_end_all):
        timeDiffs = lr_all_start - endTime
        goodDTs = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
        if len(goodDTs) > 0:
            lr_DTPass_segm[k] = goodDTs
    
    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # ===============================================================================================
    # === find ACTUAL interval start-end connectedness: get all connected paths if there are any ==
    # ===============================================================================================
    # REMARK: refine previously acquired potential segment connectedness by searching paths between
    print(f'\n{timeHMS()}:({doX_s}) Checking paths between segments ...')
    for t_ID_from, t_IDs_to in tqdm(lr_DTPass_segm.items()):
        # analyze graph for paths between segment connection that satisfy DT criterium.
        # to avoid paths though different segments, nodes with 2 criterium are isolated on a subgraph:
        # 1) nodes that exist on (time_min , time_min + DT) interval (not really DT, but to start of a  next segment)
        # 2) nodes that are not part of segments (owner is None)
        node_from = segments2[t_ID_from][-1]
        time_from = G_time(node_from)
        for t_ID_to in t_IDs_to:
            node_to         = segments2[t_ID_to][0]
            time_to         = G_time(node_to)
            t_nodes_keep    = [node for node in G.nodes() if time_from <= G_time(node) <= time_to and G_owner(node) is None] 
            t_nodes_keep.extend([node_from,node_to])
            G_sub = G.subgraph(t_nodes_keep)
            hasPath = nx.has_path(G_sub, source = node_from, target = node_to)
            if hasPath:
                #shortest_path = list(nx.all_shortest_paths(G_sub, node_from, node_to))
                G2.add_edge(t_ID_from, t_ID_to, dist = time_to - time_from + 1) # include end points =  inter + 2

    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # ===============================================================================================
    # === extract closest neighbors of segments ===
    # ===============================================================================================
    # ===============================================================================================
    # REMARK: inspecting closest neighbors allows to elliminate cases with segments inbetween 
    # REMARK: connected segments via lr_maxDT. problem is that affects branches of splits/merges
    # EDIT 27/07/24 maybe should redo in directed way. should be shorter and faster
    if 1 == 1:
        t_nodes = sorted(list(G2.nodes()))
        t_neighbor_sol_all_prev = {tID:{} for tID in t_nodes}
        t_neighbor_sol_all_next = {tID:{} for tID in t_nodes}
        for node in t_nodes:
            # Get the neighboring nodes
            t_neighbors     = list(G2.neighbors(node))
            t_time_start    = G2_t_start(   node)
            t_time_end      = G2_t_end(     node)
            t_neighbors_prev = []
            t_neighbors_next = []
            for t_neighbor in t_neighbors:
                t_time_neighbor_start    = G2_t_start(  t_neighbor)
                t_time_neighbor_end      = G2_t_end(    t_neighbor)

                if t_time_end < t_time_neighbor_start:
                    t_neighbors_next.append(t_neighbor)
                elif t_time_neighbor_end < t_time_start:
                    t_neighbors_prev.append(t_neighbor)
            # check if neighbors are not lost, or path generation is incorrect, like looping back in time
            assert len(t_neighbors) == len(t_neighbors_prev) + len(t_neighbors_next), "missing neighbors, time position assumption is wrong"
    
            t_neighbors_weights_prev = {}
            t_neighbors_weights_next = {}
    
            for t_neighbor in t_neighbors_prev: # back weights are negative
                t_neighbors_weights_prev[t_neighbor] = -1*G2_edge_dist((node,t_neighbor))
            for t_neighbor in t_neighbors_next:
                t_neighbors_weights_next[t_neighbor] = G2_edge_dist((node,t_neighbor))
            t_neighbors_time_prev = {tID:segments2[tID][-1][0]  for tID in t_neighbors_weights_prev}
            t_neighbors_time_next = {tID:segments2[tID][0][0]   for tID in t_neighbors_weights_next}
    
            if len(t_neighbors_weights_prev)>0:
                # neg weights, so max, get time of nearset branch t0. get all connections within [t0 - 2, t0] in case of split
                t_key_min_main = max(t_neighbors_weights_prev, key = t_neighbors_weights_prev.get)
                t_key_main_ref_time = t_neighbors_time_prev[t_key_min_main]
                t_sol = [key for key,t in t_neighbors_time_prev.items() if t_key_main_ref_time - 1 <= t <=  t_key_main_ref_time] #not sure why-+2, changed to -1 
                for t_node in t_sol:t_neighbor_sol_all_prev[node][t_node] = t_neighbors_weights_prev[t_node]
        
            if len(t_neighbors_weights_next)>0:
                t_key_min_main = min(t_neighbors_weights_next, key = t_neighbors_weights_next.get)
                t_key_main_ref_time = t_neighbors_time_next[t_key_min_main]
                t_sol = [key for key,t in t_neighbors_time_next.items() if t_key_main_ref_time <= t <=  t_key_main_ref_time + 1] # not sure why +2, changed to +1 
                for t_node in t_sol: t_neighbor_sol_all_next[node][t_node] = t_neighbors_weights_next[t_node]
            a = 1

    
    if 1 == -1:
        # ===============================================================================================
        # ====== rudamentary split/merge analysis based on nearest neighbor connectedness symmetry ======
        # ===============================================================================================
        # REMARK: nearest neighbors elliminates unidirectional connectedness between split/merge branches

        # by analyzing nearest neighbors you can see anti-symmetry in case of splits and merges.
        # if one branch of merge is closer to merge product, it will have strong connection both ways
        # further branches will only have unidirectional connection
        if 1==1:
            lr_connections_unidirectional   = []
            lr_connections_forward          = []
            lr_connections_backward         = []
            for t_from, t_forward_conns in t_neighbor_sol_all_next.items():
                for t_to in t_forward_conns.keys():
                    lr_connections_forward.append(tuple(sorted([t_from,t_to])))
            for t_to, t_backward_conns in t_neighbor_sol_all_prev.items():
                for t_from in t_backward_conns.keys():
                    lr_connections_backward.append(tuple(sorted([t_from,t_to])))

            lr_connections_unidirectional   = sorted(list(set(lr_connections_forward) & set(lr_connections_backward)), key = lambda x: x[0])
            lr_connections_directed         = [t_conn for t_conn in lr_connections_forward + lr_connections_backward if t_conn not in lr_connections_unidirectional]
            lr_connections_directed         = sorted(list(set(lr_connections_directed)), key = lambda x: x[0])

            # lr_connections_directed contain merges/splits, but also lr_connections_unidirectional contain part of splits/merges.
            # unidirectional means last/next segment is connected via unidirectional ege
            t_merge_split_culprit_edges = []
            for t_conn in lr_connections_directed:
                if t_conn in lr_connections_forward:    t_from, t_to = t_conn  
                else:                                   t_to, t_from = t_conn
                # by knowing direction of direactional connection, i can tell that opposite direction connection is absent.
                # there are other connection/s in that opposite (time-wise) directions which are other directional connectsions or unidir
                t_time_to   = segments2[t_to    ][0][0]
                t_time_from = segments2[t_from  ][0][0]
                # if directed connection i time-forward, then unidir connections are from t_to node back in time
                t_forward = True if t_time_to - t_time_from > 0 else False
                if t_forward:
                    t_unidir_neighbors = list(t_neighbor_sol_all_prev[t_to].keys())
                else:
                    t_unidir_neighbors = list(t_neighbor_sol_all_next[t_to].keys())
                t_unidir_conns = [tuple(sorted([t_to,t])) for t in t_unidir_neighbors]
                t_merge_split_culprit_edges += [t_conn]
                t_merge_split_culprit_edges += t_unidir_conns
                a = 1
                #if t_to in t_neighbor_sol_all_prev[t_from]:
            t_merge_split_culprit_edges = sorted(t_merge_split_culprit_edges, key = lambda x: x[0])


            # simply extract nodes and their neigbors if there are multiple neighbors
            t_merge_split_culprit_edges2 = []
            for t_from, t_forward_conns in t_neighbor_sol_all_next.items():
                if len(t_forward_conns)>1:
                    for t_to in t_forward_conns.keys():
                        t_merge_split_culprit_edges2.append(tuple(sorted([t_from,t_to])))

            for t_to, t_backward_conns in t_neighbor_sol_all_prev.items():
                if len(t_backward_conns) > 1:
                    for t_from in t_backward_conns.keys():
                        t_merge_split_culprit_edges2.append(tuple(sorted([t_from,t_to])))

            t_merge_split_culprit_edges2 = sorted(t_merge_split_culprit_edges2, key = lambda x: x[0])
    
            t_merge_split_culprit_edges_all = sorted(list(set(t_merge_split_culprit_edges + t_merge_split_culprit_edges2)), key = lambda x: x[0])
            t_merge_split_culprit_node_combos = segment_conn_end_start_points(t_merge_split_culprit_edges_all, segment_list = segments2, nodes = 1)

            # find clusters of connected nodes of split/merge events. this way instead of sigment IDs, because one
            # segment may be sandwitched between any of these events and two cluster will be clumped together

            T = nx.Graph()
            T.add_edges_from(t_merge_split_culprit_node_combos)
            connected_components_unique = extract_graph_connected_components(T, sort_function = lambda x: x)
    
            # relate connected node families ^ to segments
            lr_merge_split_node_families = []
            for t_node_cluster in connected_components_unique:
                t_times_active = [t_node[0] for t_node in t_node_cluster]
                t_active_segments = set(sum([lr_time_active_segments[t_time] for t_time in t_times_active],[]))
                t_sol = []
                for t_node in t_node_cluster:
                    for t_segment_ID in t_active_segments:
                        if t_node in segments2[t_segment_ID]:
                            t_sol.append(t_segment_ID)
                            break
                lr_merge_split_node_families.append(sorted(t_sol))

        # ===============================================================================================
        # merge/split classification will be here
        # ===============================================================================================
        # REMARK: if a node has MULTIPLE neighbors from one of side (time-wise)
        # REMARK: then split or merge happened, depending which side it is
        # REMARK: if cluster has both split and merge of nodes, its classified separately
        # REMARK: edges of these 3 events are stored for further analysis
        lr_conn_edges_splits                = []
        lr_conn_edges_merges                = []
        lr_conn_edges_splits_merges_mixed   = []    # not used yet

        lr_conn_merges_to_nodes             = []
        lr_conn_splits_from_nodes           = []    # not used yet

        if 1 == 1:
    
            for t_cluster in lr_merge_split_node_families:

                t_neighbors_prev = {tID:[] for tID in t_cluster}
                t_neighbors_next = {tID:[] for tID in t_cluster}

                for tID in t_cluster:
                    t_neighbors_all = [t for t in list(G2.neighbors(tID)) if t in t_cluster]
                    t_node_start    = G2.nodes[tID]["t_start"]
                    t_node_end      = G2.nodes[tID]["t_end"]
                    for t_neighbor in t_neighbors_all:
                        t_neighbor_start    = G2.nodes[t_neighbor]["t_start"]
                        t_neighbor_end      = G2.nodes[t_neighbor]["t_end"]
                        if t_neighbor_start > t_node_end:
                            t_neighbors_next[tID].append(t_neighbor)
                        elif t_neighbor_end < t_node_start:
                            t_neighbors_prev[tID].append(t_neighbor)
    
                t_neighbors_prev_large = {tID:t_neighbors for tID,t_neighbors in t_neighbors_prev.items() if len(t_neighbors) > 1}
                t_neighbors_next_large = {tID:t_neighbors for tID,t_neighbors in t_neighbors_next.items() if len(t_neighbors) > 1}
    
                t_edges_merge =  sum([[tuple(sorted([id1,id2])) for id2 in subIDs] for id1,subIDs in t_neighbors_prev_large.items()],[])
                t_edges_split =  sum([[tuple(sorted([id1,id2])) for id2 in subIDs] for id1,subIDs in t_neighbors_next_large.items()],[])
    
                if len(t_neighbors_next_large) == 0 and len(t_neighbors_prev_large) > 0:
                    lr_conn_edges_merges += t_edges_merge
                elif len(t_neighbors_prev_large) == 0 and len(t_neighbors_next_large) > 0:
                    lr_conn_edges_splits += t_edges_split
                else:
                    lr_conn_edges_splits_merges_mixed += (t_edges_merge + t_edges_split)
                a = 1

            lr_conn_merges_to_nodes     = sorted(list(set([t_node[1] for t_node in lr_conn_edges_merges])))
            lr_conn_splits_from_nodes   = sorted(list(set([t_node[0] for t_node in lr_conn_edges_splits])))

            # gather successors for mixed m/s in a dict for further trajectory extension 21.09.23
            lr_conn_splits_merges_mixed_dict = defaultdict(list)
            for t_from,t_to in  lr_conn_edges_splits_merges_mixed:
                lr_conn_splits_merges_mixed_dict[t_from].append(t_to)

            for t_from, t in lr_conn_splits_merges_mixed_dict.items():
                lr_conn_splits_merges_mixed_dict[t_from] = sorted(set(t))
    
            # 10.10.2023 bonus for branch extension with conservative subID redistribution
            # get small connected clusters [from1, from2,..] -> [to1,to2,..]. 
            t_compair_pairs = list(itertools.combinations(lr_conn_splits_merges_mixed_dict, 2))
            t_from_connected = defaultdict(set)#{t:set([t]) for t in t_from_IDs}
            for t1,t2 in t_compair_pairs:
                t_common_elems = set(lr_conn_splits_merges_mixed_dict[t1]).intersection(set(lr_conn_splits_merges_mixed_dict[t2]))
                if len(t_common_elems) > 0:
                    t_from_connected[t1].add(t2)
            lr_conn_mixed_from_to = {}
            for t_ID_main, t_subIDs in t_from_connected.items():
                t_from_subIDs = sorted([t_ID_main] + list(t_subIDs))
                t_to_subIDs = sorted(set([t_to for t in t_from_subIDs for t_to in lr_conn_splits_merges_mixed_dict[t]]))
                lr_conn_mixed_from_to[tuple(t_from_subIDs)] = tuple(t_to_subIDs)
            a = 1
    print(f'\n{timeHMS()}:({doX_s}) Working on one-to-one (121) segment connections ... ')
    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # === PROCESS SEGMENT-SEGMENT CONNECTIONS THAT ARE CONNECTED ONLY TOGETHER (ONE-TO-ONE; 121) ====
    # ===============================================================================================
    # WHY: long graph segment (which is a chain of nodes that each consists only of one contour) is likely
    # WHY: a single bubble raising without merges and splits. sometimes, due to optical lighting artifacts
    # WHY: or other artifacts, next to bubble pops out an unwanted object, or its a same bubble that visually
    # WHY: splits and recombines. that means that solo contour trajctory is interrupted with a 'hole'
    # WHY: which is actually an 'explositon' of connections into 'stray' nodes and 'collapse' back to
    # WHY: a solo trajectory.
    # WHY: these holes can be easily patched, but first they have to be identified
    # WHY: if this happens often, then instead of analyzing such 'holes' locally, we can bring 
    # WHY: whole trajectory with all holes and patch them more effectively using more bubble history.
    # NOTE: there is also one caveat where we cannot trust end-points of combined bubble trajectories
    # NITE: because of the effect of 'fake' split or merge -  when object splits, but only optically.

    # FLOWCHART OF PROCESSING 121 (one-to-one) SEGMENTS:
    #
    # (121.A) isolate 121 connections
    # (121.B) extract inter-segment nodes as time slices : {t1:[*subIDs],...}
    # (121.C) detect which cases are zero-path ZP. its 121 that splits into segment and a node. byproduct of a method.
    # (121.D) resolve case with ZP
    # (121.E) find which 121 form a long chain- its an interrupted solo bubble's trejctory.
    #           missing data in 'holes' is easier to interpolate from larger chains 
    # (121.F) chain edges may be terminated by artifacts - 'fake' events, find them and refine chain list elements.
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
    # WHAT: extract all 121 type connections, additinally, info on merges/splits is needed for 'fake' stuff
   
    G2_dir, lr_ms_edges_main = get_event_types_from_segment_graph(G2)
    lr_conn_edges_merges = [(t_from,t_to) for t_to,t_froms in lr_ms_edges_main['merge'].items() for t_from in t_froms]
    lr_conn_edges_splits = [(t_from,t_to) for t_from,t_tos in lr_ms_edges_main['split'].items() for t_to in t_tos]
    lr_conn_edges_splits_merges_mixed = []
    for t_froms, t_tos in lr_ms_edges_main['mixed'].items():
        t_edges = [t_edge for t_edge in itertools.product(t_froms,t_tos) if t_edge in G2_dir.edges]
        lr_conn_edges_splits_merges_mixed.extend(t_edges)

    lr_ms_edges_brnch = {'merge':{}, 'split':{}}
    for ID_main, IDs_branch in  lr_ms_edges_main['merge'].items():
        for ID_branch in IDs_branch:
            lr_ms_edges_brnch['merge'][ID_branch] = ID_main
    for ID_main, IDs_branch in  lr_ms_edges_main['split'].items():
        for ID_branch in IDs_branch:
            lr_ms_edges_brnch['split'][ID_branch] = ID_main

    t_conn_121 = lr_ms_edges_main['solo']
    # ===================================================================================================
    # ==================== SLICE INTER-SEGMENT (121) NOTE-SUBID SPACE W.R.T TIME ========================
    # ===================================================================================================
    # CREATE A DICTIONARY OF TYPE {TIME1:[*SUBIDS_AT_TIME1],...}
    # WHAT: unresolved inter-segment space, for 121 connections, may contain multiple contours associated
    # WHAT: with a single bubble at any single time. Real bubble may be any combination of these bubbles

    if 1 == 1:
        lr_big121s_perms_pre = {}
        tIDs = [t_conn[0] for t_conn in t_conn_121]
        for t_from in tIDs:
            for t_to,t_dist in t_neighbor_sol_all_next[t_from].items(): # is a dict but walk the keys.
                t_from_node_last    = segments2[t_from][-1]
                t_to_node_first     = segments2[t_to][0]

                # for directed graph i can disable cutoff, since paths can progress only forwards
                t_from_node_last_time   = t_from_node_last[0]
                t_to_node_first_time    = t_to_node_first[0]
                t_from_to_max_time_steps= t_to_node_first_time - t_from_node_last_time + 1
                # soo.. all_simple_paths method goes into infinite (?) loop when graph is not limited, shortest_simple_paths does not.
                t_start, t_end = t_from_node_last[0], t_to_node_first[0]
                # zero-path connection needs connected nodes at same time as segment end/start
                if t_dist == 2:
                    t_nodes_keep = [t_node for t_node, t_time in G.nodes(data='time') if t_start <= t_time <= t_end]
                else:
                    t_nodes_keep    = [node for node in G.nodes() if t_from_node_last_time < G_time(node) < t_to_node_first_time and G_owner(node) is None] 
                    t_nodes_keep.extend([t_from_node_last,t_to_node_first])
                g_limited = G.subgraph(t_nodes_keep)

                # ====== CREATE ACTIVE CONTOUR DICT IN INTER-SEGMENT SPACE: {T1:[C1,C2],T2:[C2,C4],...} ====
                # DO IT BY EXPLORING DIRECTED CONNECTIONS FROM END OF LEFT SEGMENT USING DEPTH SEARCH
                # include end point times =  ideally only segment node will be there, but not always (its due to overlap artifacts)
                t_connected_nodes = set()
                dfs_succ(g_limited, t_from_node_last, time_lim = t_to_node_first_time + 1, node_set = t_connected_nodes)
                # to_start time nodes are included. now get from_end time nodes
                t_nodes_after_from_end = [t_node for t_node in t_connected_nodes if t_node[0] == t_from_node_last_time + 1]
                t_predecessors = set()
                for t_node in t_nodes_after_from_end:
                    t_preds = list(g_limited.predecessors(t_node))
                    t_predecessors.update(t_preds)
                t_connected_nodes.update(t_predecessors)

                t_temp = {t:[] for t in np.arange(t_from_node_last_time, t_to_node_first_time + 1,1)}
                for t_time,*t_subIDs in t_connected_nodes:
                    t_temp[t_time] += t_subIDs
                lr_big121s_perms_pre[(t_from,t_to)] = t_temp
                a = 1

    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # separate 121 paths with zero inter length
    # ===============================================================================================
    # REMARK: these cases occur when segment terminates into split with or merge with a node, not a segment
    # REMARK: and 121 separation only concerns itself with segment-segment connecivity
    # REMARK: this case might be inside merge/split merge.
    lr_zp_redirect = {tID: tID for tID in range(len(segments2))}
    if 1 == 1:
        t_conn_121_zero_path = []
        for t_node in t_conn_121:
            if len(lr_big121s_perms_pre[t_node]) == 2:
                t_t_min = min(lr_big121s_perms_pre[t_node])
                t_t_max = max(lr_big121s_perms_pre[t_node])
                if len(lr_big121s_perms_pre[t_node][t_t_min]) > 1 or len(lr_big121s_perms_pre[t_node][t_t_max]) > 1:
                    t_conn_121_zero_path.append(t_node)

        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        for t_conn in t_conn_121_zero_path: 
            t_dict = lr_big121s_perms_pre[t_conn]
            t_from, t_to        = t_conn                                                    # no inheritance is yet needed
            t_from_new          = lr_zp_redirect[t_from]
            t_times             = [t for t,t_subIDs in t_dict.items() if len(t_subIDs) > 1]
            if len(t_times) > 1:
                if 'zero_path' not in issues_all_dict[doX]: issues_all_dict[doX]['zero_path'] = []
                issues_all_dict[doX]['zero_path'].append(f'multiple times: {t_dict}')
                lr_big121s_perms_pre.pop(t_conn,None)
                t_conn_121.remove(t_conn)
                continue
            #assert len(t_times) == 1, "unexpected case. multiple times, need to insert nodes in correct order"
            t_nodes_composite   = [tuple([t] + t_dict[t]) for t in t_times]                 # join into one node (composit)
            t_nodes_solo        = [(t,t_subID) for t in t_times for t_subID in t_dict[t]]   # get solo contour nodes

            t_edges             = []
            t_nodes_prev        =   [t_node for t_node in segments2[t_from_new] if t_node not in t_nodes_solo] 
            t_edges             +=  [(t_nodes_prev[-1]      , t_nodes_composite[0]) ]
            segments2[t_from_new]   =   t_nodes_prev
            segments2[t_from_new]   +=  t_nodes_composite
            t_nodes_next        =   [t_node for t_node in segments2[t_to] if t_node not in t_nodes_solo]
            t_edges             +=  [(t_nodes_composite[0]  , t_nodes_next[0])      ]
            segments2[t_from_new]   +=  t_nodes_next
    
            # modify graph: remove solo IDs, add new composite nodes. add parameters to new nodes
            G.remove_nodes_from(t_nodes_solo) 
            G.add_edges_from(t_edges)

            set_custom_node_parameters(G, g0_contours, t_nodes_composite, t_from_new, calc_hull = 1)

            for t_node in t_nodes_next: # t_nodes_prev + 
                    G.nodes[t_node]["owner"] =  t_from_new

            segments2[t_to]     = []

            # modify segment view graph.
            t_successors   = extractNeighborsNext(G2, t_to, lambda x: G2.nodes[x]["t_start"])
    
            t_edges = [(t_from_new,t_succ) for t_succ in t_successors]
            G2.remove_node(t_to)
            G2.add_edges_from(t_edges)
            G2.nodes()[t_from_new]["t_end"] = segments2[t_from_new][-1][0]

            # create dict to re-connect previously obtained edges.
            lr_zp_redirect[t_to] = t_from_new
            print(f'zero path: joined segments: {t_conn}')
    
    t_conn_121              = lr_reindex_masters(lr_zp_redirect, t_conn_121, remove_solo_ID = 1)
    lr_conn_edges_merges    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_merges   )
    lr_conn_edges_splits    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits   )
    lr_conn_edges_splits_merges_mixed = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits_merges_mixed   )
    temp = {}
    for t_state, t_dict in lr_ms_edges_brnch.items():
        temp[t_state] = {}
        for a,b in t_dict.items():
            c, d = lr_zp_redirect[a], lr_zp_redirect[b]
            if c != d: temp[t_state][c] = d
    lr_ms_edges_brnch = temp
    a = 1

    #'fk_event_branchs' inherits 'lr_ms_edges_main', but 'solo' and 'mixed' are not used
    temp = {}
    for t_state in ['merge','split']:
        temp[t_state] = {}
        for t_ID, t_subIDs in lr_ms_edges_main[t_state].items():
            t_ID_new = lr_zp_redirect[t_ID]
            t_subIDs_new = [lr_zp_redirect[t] for t in t_subIDs]
            temp[t_state][t_ID_new] = t_subIDs_new
    lr_ms_edges_main = temp


    fk_time_active_segments = defaultdict(list)                                     
    for k,t_segment in enumerate(segments2):
        t_times = [G.nodes[a]["time"] for a in t_segment]
        for t in t_times:
            fk_time_active_segments[t].append(k)

    # ===============================================================================================
    # ===== EDIT 10.09.23 lets find joined 121s ========================
    # ===============================================================================================

    if 1 == 1:
        # codename: big121
        G_seg_view_1 = nx.DiGraph()
        G_seg_view_1.add_edges_from(t_conn_121)

        for g in G_seg_view_1.nodes():
              G_seg_view_1.nodes()[g]["t_start"]    = segments2[g][0][0]
              G_seg_view_1.nodes()[g]["t_end"]      = segments2[g][-1][0]
    
        # 121s are connected to other segments, if these segments are other 121s, then we can connect 
        # them into bigger one and recover using more data.
        #lr_big_121s_chains = extract_graph_connected_components(G_seg_view_1.to_undirected(), lambda x: x)
        lr_big_121s_chains = [sorted(c, key = lambda x: x) for c in nx.connected_components(G_seg_view_1.to_undirected())]
        lr_big_121s_chains = sorted(lr_big_121s_chains, key = lambda x: x[0])
        print(f'\n{timeHMS()}:({doX_s}) Working on joining all continious 121s together: {lr_big_121s_chains}')
        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<        
    # big 121s terminate at zero neighbors, real or pseudo merges/splits 
        # both cannot easily tell the difference between real and pseudo. have search for recombination 
        # pseudo events means start/end of a big 121 contain only part of bubble and cannot 
        # be used to determine history. For the real m/s start/end segments can be used.  
        # how to tell real and pseudo apart? 
        # since pseudo m/s is one bubble: 
        # 1) split branches have limited and rather short length before pseudo merge. 
        # 2) ! area is recovered after pseudo branch 

        # -----------------------------------------------------------------------------------------------
        # PSEUDO MERGE/SPLIT ON EDGES: DETECT BIG 121S WITH M/S
        # -----------------------------------------------------------------------------------------------
        # check which segment chains have leftmost and rightmost segmentsa s parts of splits and merges respectively

        #fk_event_branchs = {'merge':defaultdict(list),'split':defaultdict(list)}
        #[fk_event_branchs['merge'][t_to     ].append(t_from ) for t_from,t_to in lr_conn_edges_merges]    
        #[fk_event_branchs['split'][t_from   ].append(t_to   ) for t_from,t_to in lr_conn_edges_splits]
        fk_event_branchs = lr_ms_edges_main.copy()  #lr_ms_edges_main used last time
        if 1 == 1:
            t_segment_ID_left   = [t_subIDs[0] for t_subIDs in lr_big_121s_chains]
            t_segment_ID_right  = [t_subIDs[-1] for t_subIDs in lr_big_121s_chains]
            fk_merge_to     = {}                                        # chain segment ID : merge to which segment
            fk_split_from   = {}
            t_index_has_merge = []
            t_index_has_split = []
            # find if left is part of split, and with whom
            for t, t_ID in enumerate(t_segment_ID_left):
                if t_ID in lr_ms_edges_brnch['split']:
                    fk_split_from[t_ID] = lr_ms_edges_brnch['split'][t_ID]                    # save who was mater of left segment-> t_ID:t_from
                    t_index_has_split.append(t) 

            # find if right is part of merge, and with whom
            for t, t_ID in enumerate(t_segment_ID_right):               # order in big 121s, right segment ID.
                if t_ID in lr_ms_edges_brnch['merge']:
                    fk_merge_to[t_ID] = lr_ms_edges_brnch['merge'][t_ID]    # lr_ms_edges_brnch used last time
                    t_index_has_merge.append(t) 

            # extract big 121s that are in no need of real/pseudo m/s analysis.
            #lr_big121_events_none = defaultdict(list) # big 121s not terminated by splits or merges. ready to resolve.
            # find if elements of big 121s are without m/s or bounded on one or both sides by m/s.
            set1 = set(t_index_has_merge)
            set2 = set(t_index_has_split)

            t_only_merge = set1 - set2
            t_only_split = set2 - set1

            t_merge_split = set1.intersection(set2)

        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # -----------------------------------------------------------------------------------------------
        # PSEUDO MERGE/SPLIT ON EDGES: DETERMINE IF M/S IS FAKE OR REAL
        # -----------------------------------------------------------------------------------------------
        # 19.10.23 comments:
        # looking at end points of chains is a very niche case. seem to be important in our case.
        # split or merge does not have to be violent, bubbles may simply touch for few frames.
        # you cannot tell apart fake and real events, by area, on frames right before and after event
        # if bubble softly touches or splits, short term dynamics of total area can be subtle.
        # - branch may be a bubble that poped out near, so total area will grow rapidly
        # - or your main segment may be a branch that popped up (had this case)
        # 19.10.23 what can you do:
        # 1)    check if branch is long. might indicate that its not a fluke/optial artifact
        # 2)    look at cases with only 2 branches. most likely due to sideways illum producing only side reflecitons
        # 3)    check if that other branch is isolated- does not have a history. 
        # 4)    other branch spans larger times. kind of meh crit. cannot know in case of random bubble popping out.
        # 5)    check area change right near an event. this will rule out random neighbor bubble.
        #       this will also rule out if your segment is a random bubble.
        print(f'\n{timeHMS()}:({doX_s}) Determining if end points of big 121s are pseudo events: ')
        if 1 == 1:
            t_big121s_merge_conn_fake = []
            t_big121s_split_conn_fake = []

            t_big121s_merge_conn_fake2 = {}
            t_big121s_split_conn_fake2 = {}

            t_merge_split_fake = defaultdict(list) # 0 -> is part of fake split, -1 -> is part of fake merge
            t_only_merge_split_fake = []
            t_max_pre_merge_segment_length = 12 # skip if branch is too long. most likely a real branch

            t_only_merge_fake = []
            for t in t_only_merge.union(t_merge_split): # check these big chains IDs with merge event
                t_big121_subIDs = lr_big_121s_chains[t]
                #if t not in t_merge_split:
                t_from      = t_big121_subIDs[-1]   # possible fake merge branch (segment ID).
                t_from_from = t_big121_subIDs[-2]   # segment ID prior to that, most likely OG bubble.
                t_from_to   = fk_merge_to[t_from]   # seg ID of merge.
                t_from_branches = fk_event_branchs['merge'][t_from_to]
                if len(segments2[t_from]) >= t_max_pre_merge_segment_length: 
                    continue
                if len(t_from_branches) > 2: # dont consider difficult cases
                    continue     

                t_from_other            = [t for t in t_from_branches if t != t_from][0]
                t_from_other_new = lr_zp_redirect[t_from_other]
                t_time_from_other_start = G.nodes[  segments2[t_from_other  ][0 ]   ]["time"]
                t_time_from_from_end    = G.nodes[  segments2[t_from_from   ][-1]   ]["time"]
                if t_time_from_from_end >= t_time_from_other_start:  # branch overlaps from_from
                    continue    

                t_from_other_predecessors = extractNeighborsPrevious(G2, t_from_other, lambda x: G2.nodes[x]["t_end"])
                if len(t_from_other_predecessors) > 0:                          continue     # branch lives on its life. probly real

                t_nodes_to_first = segments2[t_from_to   ][:3]
                # take a time period slice of both branches. should contain first times at which both branches are present.
                t_time = G.nodes[segments2[t_from_to][0]]['time']
                t_cntr = 0
                t_times_common = []
                while t_time >= t_time_from_from_end:
                    t_inter = set(t_from_branches).intersection(set(fk_time_active_segments[t_time]))
                    if len(t_inter) == len(t_from_branches):
                        t_cntr += 1
                        t_times_common.append(t_time)
                    t_time -= 1
                    if t_cntr >= 3:
                        break
                        
                if len(t_times_common) == 0: continue 

                t_nodes_at_times_common = []
                for k in t_from_branches:
                    t_nodes_at_times_common.extend([t_node for t_node in segments2[k] if G.nodes[t_node]['time'] in t_times_common])

                t_dispered_nodes = disperse_nodes_to_times(t_nodes_at_times_common)

                t_from_areas_last = []
                for t_time, t_subIDs in t_dispered_nodes.items():
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t] for t in t_subIDs]))
                    t_from_areas_last.append(centroid_area(t_hull)[1])

                t_area_mean_to_first            = np.mean([G.nodes[n]["area"] for n in t_nodes_to_first])
                t_area_mean_from_last_combined  = np.mean(t_from_areas_last)

                t_rel_area_change_from_to = np.abs(t_area_mean_to_first - t_area_mean_from_last_combined) / t_area_mean_to_first
                if  t_rel_area_change_from_to> 0.35: continue

                t_from_to_node      = segments2[t_from_to   ][0 ]                   # take end-start nodes
                t_from_from_node    = segments2[t_from_from ][-1]

                t_from_to_node_area     = G.nodes[t_from_to_node    ]["area"]       # check their area
                t_from_from_node_area   = G.nodes[t_from_from_node  ]["area"] 
                if np.abs(t_from_to_node_area - t_from_from_node_area) / t_from_to_node_area < 0.35:
                    if t not in t_merge_split:
                        t_only_merge_fake += [t]
                        #t_big121s_merge_conn_fake.append((t_from,t_from_to))
                        t_big121s_merge_conn_fake2[(t_from_from,t_from_to)] = t_from_branches
                    else:       # undetermined, decide after
                        t_merge_split_fake[t] += [-1]
                        assert 1 == -1, 'havent been here'

            t_only_split_fake = [] ## for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            for t in t_only_split.union(t_merge_split):
                t_big121_subIDs = lr_big_121s_chains[t]
                t_to            = t_big121_subIDs[0]   # possible fake merge branch (segment ID).
                t_to_to         = t_big121_subIDs[1]   # segment ID prior to that, most likely OG bubble.
                t_from_to       = fk_split_from[t_to]  # seg ID of merge.
                t_to_branches = fk_event_branchs['split'][t_from_to]
                if len(segments2[t_to]) >= t_max_pre_merge_segment_length:  continue     # long branch - probly real
                if len(t_to_branches) > 2:                                  continue     # dont consider difficult cases

                t_to_other          = [t for t in t_to_branches if t != t_to][0]
                t_time_to_other_end = G.nodes[  segments2[t_to_other  ][-1]     ]["time"]
                t_time_to_to_start  = G.nodes[  segments2[t_to_to     ][0 ]     ]["time"]
                if t_time_to_other_end >= t_time_to_to_start:               continue     # branch longer than expected end

                t_to_other_successors = extractNeighborsNext(G2, t_to_other, lambda x: G2.nodes[x]["t_start"])
                if len(t_to_other_successors) > 0:                          continue     # branch lives one its life. probly real

                t_nodes_from_last = segments2[t_from_to   ][-3:]
                # take a time period slice of both branches. should contain first times at which both branches are present.
                t_time = G.nodes[segments2[t_from_to][-1]]['time']
                t_cntr = 0
                t_times_common = []
                while t_time <= t_time_to_to_start:
                    t_inter = set(t_to_branches).intersection(set(fk_time_active_segments[t_time]))
                    if len(t_inter) == len(t_to_branches):
                        t_cntr += 1
                        t_times_common.append(t_time)
                    t_time += 1
                    if t_cntr >= 3:
                        break
                        
                if len(t_times_common) == 0: continue     # branches not intersect in time

                t_nodes_at_times_common = []
                for k in t_to_branches:
                    t_nodes_at_times_common.extend([t_node for t_node in segments2[k] if G.nodes[t_node]['time'] in t_times_common])

                t_dispered_nodes = disperse_nodes_to_times(t_nodes_at_times_common)

                t_to_areas_first = []
                for t_time, t_subIDs in t_dispered_nodes.items():
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t] for t in t_subIDs]))
                    t_to_areas_first.append(centroid_area(t_hull)[1])

                t_area_mean_from_last           = np.mean([G.nodes[n]["area"] for n in t_nodes_from_last])
                t_area_mean_to_first_combined   = np.mean(t_to_areas_first)

                t_rel_area_change_from_to = np.abs(t_area_mean_from_last - t_area_mean_to_first_combined) / t_area_mean_from_last
                if  t_rel_area_change_from_to> 0.35: continue

                t_from_to_node  = segments2[t_from_to   ][-1]                   # take end-start nodes
                t_to_to_node    = segments2[t_to_to     ][0]

                t_from_to_node_area = G.nodes[t_from_to_node]["area"]       # check their area
                t_to_to_node_area   = G.nodes[t_to_to_node  ]["area"] 
                if np.abs(t_from_to_node_area - t_to_to_node_area) / t_from_to_node_area < 0.35:
                    if t not in t_merge_split:
                        t_only_split_fake += [t]
                        #t_big121s_split_conn_fake.append((t_from_to,t_to))
                        t_big121s_split_conn_fake2[(t_from_to,t_to_to)] = t_to_branches
                    else:       # undetermined, decide after
                        t_merge_split_fake[t] += [0]
                        assert 1 == -1, 'havent been here'

            for t,t_states in t_merge_split_fake.items():

                assert 1 == -1, 'havent been here t_merge_split_fake'
                if len(t_states) == 2:                  # both ends are in fake event
                    t_only_merge_split_fake += [t]
                elif 0 in t_states:                     # only split is fake
                    t_to            = lr_big_121s_chains[t][0]   
                    t_from_to       = fk_split_from[t_to] 
                    t_only_split_fake += [t]
                    t_big121s_split_conn_fake.append((t_from_to,t_to))
                else:                                  # only merge is fake
                    t_from      = lr_big_121s_chains[t][-1]  
                    t_from_to   = fk_merge_to[t_from]
                    t_only_merge_fake += [t]
                    t_big121s_merge_conn_fake.append((t_from,t_from_to))

            assert len(t_only_merge_split_fake) == 0, "case yet unexplored, written without test"        

        # experimental fake 121 recovery. time:subID perms dont exist for these events, it should include branches inside.
        t_big121s_fake_split_merge = {**t_big121s_split_conn_fake2,**t_big121s_merge_conn_fake2}

        for (t_from,t_to), t_branches_IDs in t_big121s_fake_split_merge.items():# t_big121s_split_conn_fake2
            t_from_node_last    = segments2[t_from][-1]
            t_to_node_first     = segments2[t_to][0]

            t_t_from_min = G_time(t_from_node_last)
            t_t_to_start = G_time(t_to_node_first)

            t_owners_keep = [None] + t_branches_IDs
            t_nodes_keep    = [node for node in G.nodes() if t_t_from_min < G_time(node) < t_t_to_start and G_owner(node) in t_owners_keep] 
            t_nodes_keep.extend([t_from_node_last,t_to_node_first])

            t_subgraph = G.subgraph(t_nodes_keep)   

            t_segments_keep = [t_from, t_to] + t_branches_IDs
            t_sol = graph_sub_isolate_connected_components(t_subgraph, t_t_from_min, t_t_to_start, fk_time_active_segments,
                                                                    segments2, t_segments_keep, ref_node = t_from_node_last) 
            t_node_subIDs_all = disperse_nodes_to_times(t_sol) 
                
            lr_big121s_perms_pre[(t_from,t_to)] = t_node_subIDs_all

        print(f'\n{timeHMS()}:({doX_s}) Fake merges: {t_big121s_merge_conn_fake}, Fake splits: {t_big121s_split_conn_fake}')
        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # -----------------------------------------------------------------------------------------------
        # PSEUDO MERGE/SPLIT ON EDGES: TRIM BIG 121S AS TO DROP FAKE M/S.
        # CREATE NEW t_conn_121 -> lr_big121s_conn_121
        # -----------------------------------------------------------------------------------------------
        # hopefully fake edges of big 121s are detected, so trim them, if possible and add to future reconstruction.
        if 1 == 1:
            t_big121s_edited = [None]*len(lr_big_121s_chains) # by default everything fails. reduced len 2 cc to len 1 stays none = failed.
            for t, t_subIDs in enumerate(lr_big_121s_chains):

                if t in t_only_merge_split_fake:    # both endpoints have to be removed, not tested
                    t_subIDs_cut = t_subIDs[1:-1]
                elif t in t_only_merge_fake:        # drop fake pre-merge branch
                    t_subIDs_cut = t_subIDs[:-1]
                elif t in t_only_split_fake:        # drop fake post-split branch
                    t_subIDs_cut = t_subIDs[1:]    
                else:                               # nothing changes
                    t_subIDs_cut = t_subIDs

                if len(t_subIDs_cut) > 1:           # questionable, stays None
                    t_big121s_edited[t] = t_subIDs_cut
                #else:
                #    assert 1 == 0, "check out case t_subIDs_cut of len <= 1"

            # side segments may have been dropped. drop according graph edges.
            lr_big121s_conn_121 = []
            for t_conn in t_conn_121:
                t_from,t_to = t_conn
                for t_subIDs in t_big121s_edited:            
                    # check if t_conn fully in connected components
                    if t_subIDs != None and t_from in t_subIDs:
                        if t_to in t_subIDs:
                            lr_big121s_conn_121.append(t_conn)
                            break
        print(f'\n{timeHMS()}:({doX_s}) Trimmed big 121s: {lr_big121s_conn_121}\nInterpolating whole big 121s...')

        # for_graph_plots(G, segs = segments2, show = False)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # -----------------------------------------------------------------------------------------------
        # BIG 121S INTERPOLATE COMBINED TRAJECTORIES, GIVEN FULL INFO REDISTRIBUTE FOR DATA FOR EACH CONN
        # -----------------------------------------------------------------------------------------------
        # CALCULATE interpolation of holes in data for big 121s
        if 1 == 1: 
            lr_big121s_edges_relevant = []
            lr_big121s_interpolation        = defaultdict(dict)
            lr_big121s_interpolation_big    = defaultdict(dict) 
            t_fake_from_to = [list(t_conn) for t_conn in t_big121s_fake_split_merge]
            for t,t_subIDs in enumerate(t_big121s_edited + t_fake_from_to):
                if t_subIDs != None: 
                    t_temp_nodes_all = []
                    for t_subID in t_subIDs:
                        t_temp_nodes_all += segments2[t_subID]

                    t_temp_times        = [G.nodes[t_node]["time"       ] for t_node in t_temp_nodes_all]
                    t_temp_areas        = [G.nodes[t_node]["area"       ] for t_node in t_temp_nodes_all]
                    t_temp_centroids    = [G.nodes[t_node]["centroid"   ] for t_node in t_temp_nodes_all]
        
                    t_conns_times_dict = {}
                    for t_edge in zip(t_subIDs[:-1], t_subIDs[1:]):
                        t_from,t_to = t_edge
                        t_start,t_end  = (G_time(segments2[t_from][-1]), G_time(segments2[t_to][0]))
                        t_times = list(np.arange(t_start + 1,t_end,1))

                        t_conns_times_dict[t_edge] = t_times
                        lr_big121s_edges_relevant.append(t_edge)
                    t_times_missing_all = sum(t_conns_times_dict.values(),[])
        
                    a = 1
                    # interpolate composite (long) parameters 
                    t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, s = 15, debug = 0, aspect = 'equal', title = t_subIDs)
                    t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_subIDs)
                    # form dict time:centroid for convinience
                    t_interpolation_centroids_1 = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_centroids_0)}
                    t_interpolation_areas_1     = {t_time:t_centroid for t_time,t_centroid in zip(t_times_missing_all,t_interpolation_areas_0)}
                    # save data with t_conns keys
                    for t_conn,t_times_relevant in t_conns_times_dict.items():
                        t_conn_new = old_conn_2_new(t_conn,lr_zp_redirect)
                        t_centroids = [t_centroid for t_time,t_centroid in t_interpolation_centroids_1.items() if t_time in t_times_relevant]
                        t_areas     = [t_area for t_time,t_area in t_interpolation_areas_1.items() if t_time in t_times_relevant]

                        lr_big121s_interpolation[t_conn_new]['centroids'] = np.array(t_centroids)
                        lr_big121s_interpolation[t_conn_new]['areas'    ] = t_areas
                        lr_big121s_interpolation[t_conn_new]['times'    ] = t_times_relevant
                    # save whole big 121
                    lr_big121s_interpolation_big[tuple(t_subIDs)]['centroids'] = t_interpolation_centroids_1
                    lr_big121s_interpolation_big[tuple(t_subIDs)]['areas'    ] = t_interpolation_areas_1

    
    
        # ===================================================================================================
        # ===================== CONSTRUCT PERMUTATIONS FROM CLUSTER ELEMENT CHOICES =========================
        # ===================================================================================================
        # WHAT: generate different permutation of subIDs for each time step.
        # WHY:  Bubble may be any combination of contour subIDs at a given time. 
        # HOW:  itertools combinations of varying lenghts

        print(f'\n{timeHMS()}:({doX_s}) Computing contour element permutations for each time step...')
        if 1 == 1:
            # if "lr_zp_redirect" is not trivial, drop resolved zp edges
            lr_big121s_edges_relevant
            lr_big121s_perms_pre        = {old_conn_2_new(t_conn,lr_zp_redirect):v for t_conn,v in lr_big121s_perms_pre.items()}
            lr_big121s_perms_pre_keys   = [t_conn for t_conn in lr_big121s_perms_pre if  t_conn[0] != t_conn[1] and t_conn in lr_big121s_edges_relevant]
            #lr_big121s_perms            = {old_conn_2_new(t_conn,lr_zp_redirect):{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_big121s_perms_pre.items()}
            lr_big121s_perms            = {t_conn:{t_time:[] for t_time in lr_big121s_perms_pre[t_conn]} for t_conn in lr_big121s_perms_pre_keys}
            for t_conn in lr_big121s_perms_pre_keys:
                t_times_contours = lr_big121s_perms_pre[t_conn]
                t_conn_new = old_conn_2_new(t_conn,lr_zp_redirect)
                if t_conn_new[0] != t_conn_new[1]:
                    for t_time,t_contours in t_times_contours.items():
                        t_perms = combs_different_lengths(t_contours)
                        lr_big121s_perms[t_conn_new][t_time] = t_perms


        lr_big121s_conn_121 = lr_big121s_edges_relevant
        # ===============================================================================================
        # =================  PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ================
        # ===============================================================================================
        # WHY: these will be reused alot in next steps, store them to avoid need of recalculation
        print(f'\n{timeHMS()}:({doX_s}) Calculating parameters for possible contour combinations...')
        if 1 == 1:    
            lr_big121s_perms_areas      = AutoCreateDict()
            lr_big121s_perms_centroids  = AutoCreateDict()
            lr_big121s_perms_mom_z      = AutoCreateDict()

            for t_conn, t_times_perms in lr_big121s_perms.items():
                t_conn_new = old_conn_2_new(t_conn,lr_zp_redirect)
                for t_time,t_perms in t_times_perms.items():
                    for t_perm in t_perms:
                        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
                        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                        lr_big121s_perms_areas[     t_conn_new][t_time][t_perm] = t_area
                        lr_big121s_perms_centroids[ t_conn_new][t_time][t_perm] = t_centroid
                        lr_big121s_perms_mom_z[     t_conn_new][t_time][t_perm] = t_mom_z

        # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ===============================================================================================
        # =================  CONSTUCT UNIQUE EVOLUTIONS THOUGH CONTOUR PERMUTATION SPACE ================
        # ===============================================================================================
        # WHAT: using subID permutations at missing time construct choice tree. 
        # WHY:  each branch represents a bubbles contour ID evolution through unresolved intervals.
        # HOW: either though itertools product, or if number of branches is big, dont consider
        # HOW: branches where area changes more than set threshold. 
        print(f'\n{timeHMS()}:({doX_s}) Generating possible bubble evolution paths from prev combinations...')

        

        lr_big121s_perms_cases = {}
        lr_big121s_perms_times = {}
        lr_drop_huge_perms = []
        for t_conn, t_times_perms in lr_big121s_perms.items():
            rel_area_thresh = 2
            func = lambda edge : edge_crit_func(t_conn,edge, lr_big121s_perms_areas)

            t_conn_new = old_conn_2_new(t_conn,lr_zp_redirect)
            t_values = list(t_times_perms.values())
            t_times = list(t_times_perms.keys())
            t_branches_count = itertools_product_length(t_values) # large number of paths is expected
            t_max_paths = 5000
            if t_branches_count >= t_max_paths:
                # refine possible transition from one time to next via limit on max area change.
                # generate choices nodes, so area can be retrieved
                t_choices = [[(t_time,) + t_perm for t_perm in t_perms] for t_time,t_perms in zip(t_times,t_values)]
                # get all acceptable edges.
                edges, nodes_start, nodes_end = comb_product_to_graph_edges(t_choices, func)
                if len(nodes_end) == 0: # finding paths will fail, since no target node
                    nodes_end.add(t_choices[-1][0])   # add last, might also want edges to that node, since they have failed
                    edges.extend(list(itertools.product(*t_choices[-2:])))
                # generally i expect correct solution to have max number of elements in each cluster at each time
                # i want fidning path algorithm to retrieve evolutions with large nodes.
                # ive noticed that 'all_simple_paths' generator traverses nodes in order you have supplied them to graph
                # so supply first edges with highest total element count, but also sort them with least difference of
                # subIDs first. e.g ((1,2,3),(4,)) is worse than ((1,2),(3,4))
                sorted_edges = sorted(edges, key=sort_len_diff_f, reverse=True) 
                # retrieve 
                sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, G = None, edges = sorted_edges, only_subIDs = True, max_paths = t_max_paths - 1)
                # 'fail' can be either because 't_max_paths' is reached or there is no path from source to target
                if fail == 'to_many_paths':    # add trivial solution where evolution is transition between max element per time step number clusters                                           
                    seq_0 = list(itertools.product(*[[t[-1]] for t in t_values] ))
                    if seq_0[0] not in sequences: sequences = seq_0 + sequences # but first check. in case of max paths it should be on top anyway.
                elif fail == 'no_path': # increase rel area change threshold
                    rel_area_thresh = 5
                    func = lambda edge : edge_crit_func(t_conn,edge, lr_big121s_perms_areas)
                    edges, nodes_start, nodes_end = comb_product_to_graph_edges(t_choices, func)
                    sorted_edges = sorted(edges, key=sort_len_diff_f, reverse=True) 
                    sequences, fail = find_paths_from_to_multi(nodes_start, nodes_end, construct_graph = True, G = None, edges = sorted_edges, only_subIDs = True, max_paths = t_max_paths - 1)

                # for fake event (artificial split-merge), we have constraints that can lower number of paths
                # these events include combination of internal branches. means no explicit edges between branche nodes. consider to include branch whole.
                if t_conn in t_big121s_fake_split_merge and len(sequences) >= t_max_paths:
                    t_branches_fake = t_big121s_fake_split_merge[t_conn]
                    t_perms = lr_big121s_perms_pre[t_conn]
                    seqs, t_nodes_pre = perms_with_branches(t_branches_fake, segments2, t_perms, return_nodes = True)
                    if len(seqs) < len(sequences): sequences = seqs

            else:# -> t_branches_count < t_max_paths
                sequences = list(itertools.product(*t_values))
                
            if len(sequences) == 0: # len = 0 because seconda pass of rel_area_thresh failed.
                sequences = []                                # since there is no path, dont solve this conn
                t_times = []
                lr_drop_huge_perms.append(t_conn)

            lr_big121s_perms_cases[t_conn_new] = sequences
            lr_big121s_perms_times[t_conn_new] = t_times

        lr_big121s_conn_121 = [t_conn for t_conn in lr_big121s_conn_121 if t_conn not in lr_drop_huge_perms]
        [lr_big121s_perms_cases.pop(t_conn,None) for t_conn in lr_drop_huge_perms]
        [lr_big121s_perms_times.pop(t_conn,None) for t_conn in lr_drop_huge_perms]
            #) 
        # ===============================================================================================
        # ==== EVALUATE HULL CENTROIDS AND AREAS FOR EACH EVOLUTION, FIND CASES WITH LEAST DEVIATIONS====
        # ===============================================================================================
        # WHAT: calculate how area and centroids behave for each evolution
        # WHY:  evolutions with least path length and area changes should be right ones
        print(f'\n{timeHMS()}:({doX_s}) Determining evolutions thats are closest to interpolated missing data...')
        
        t_temp_centroids = {t_conn:t_dict['centroids'] for t_conn,t_dict in lr_big121s_interpolation.items()}
        t_args = [lr_big121s_conn_121, lr_big121s_perms_cases,t_temp_centroids,lr_big121s_perms_times,
                lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

        t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)

        t_weights   = [1,1.5,0,1]
        t_sols      = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
        lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(lr_big121s_conn_121,t_weights, t_sols, lr_big121s_perms_cases )




        lr_C0_condensed_connections_relations = lr_zp_redirect.copy()
        t_zp_redirect_inheritance = {a:b for a,b in lr_zp_redirect.items() if a != b}
        print(f'\n{timeHMS()}:({doX_s}) Saving results for restored parts of big 121s')
        t_last_seg_ID_in_big_121s   = [t[-1]    for t in t_big121s_edited if t is not None]
        t_big121s_edited_clean      = [t        for t in t_big121s_edited if t is not None]

        # experimental fake recovery at 121
        for (t_from,t_to), t_branches_IDs in t_big121s_fake_split_merge.items():
            for t in t_branches_IDs:
                lr_C0_condensed_connections_relations[t] = t_from
                segments2[t] = []
            G2.remove_nodes_from(t_branches_IDs)
            #G2.add_edge(t_from, t_to)

        t_segments_new = copy.deepcopy(segments2)
        for t_conn in lr_big121s_conn_121:
            t_from, t_to = t_conn
            t_from_new = lr_C0_condensed_connections_relations[t_from]
            t_to_new = lr_C0_condensed_connections_relations[t_to]
            print(f'edge :({t_from},{t_to}) or = {t_segments_new[t_from_new][-1]}->{t_segments_new[t_to_new][0]}')   
            save_connections_two_ways(t_segments_new, lr_big121s_perms_pre[t_conn], t_from,  t_to, G, G2, lr_C0_condensed_connections_relations, g0_contours)

        # zp relations (edges) were not used in saving 121s. so they were not relinked.
        for t_slave, t_master in t_zp_redirect_inheritance.items():
            lr_C0_condensed_connections_relations[t_slave] = lr_C0_condensed_connections_relations[t_master]
        # at this time some segments got condenced into big 121s. right connection might be changed.

        # experimental fake recovery at 121. branches changed out of proper loop. re-assign manually
        for (t_master, _), t_branches_IDs in t_big121s_fake_split_merge.items():
            for t_slave in t_branches_IDs:
                lr_C0_condensed_connections_relations[t_slave] = lr_C0_condensed_connections_relations[t_master]

        lr_time_active_segments = defaultdict(list)
        for t_segment_index, t_segment_nodes in enumerate(t_segments_new):
            t_times = [G_time(node) for node in t_segment_nodes]
            for t_time in t_times:
                lr_time_active_segments[t_time].append(t_segment_index)
        # sort keys in lr_time_active_segments
        lr_time_active_segments = {t:lr_time_active_segments[t] for t in sorted(lr_time_active_segments.keys())}

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
            t_merge_fake_to_ID  = unique_sort_list([t_conn[1] for t_conn in t_big121s_merge_conn_fake])                                          # grab master node e.g [10]
            t_split_fake_from_ID = unique_sort_list([t_conn[0] for t_conn in t_big121s_split_conn_fake])                                        # e.g [15]

            t_merge_real_to_ID  = [t_ID for t_ID in set([t_conn[1] for t_conn in lr_conn_edges_merges]) if t_ID not in t_merge_fake_to_ID]      # e.g [13,23]
            t_split_real_from_ID  = [t_ID for t_ID in set([t_conn[0] for t_conn in lr_conn_edges_splits]) if t_ID not in t_split_fake_from_ID]    # e.g []

            t_merge_fake_conns  = [t_conn for t_conn in lr_conn_edges_merges if t_conn[1] in t_merge_fake_to_ID]    # grab all edges with that node. e.g [(8, 10), (9, 10)]
            t_split_fake_conns  = [t_conn for t_conn in lr_conn_edges_splits if t_conn[0] in t_split_fake_from_ID]

            t_merge_fake_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_merges if t_conn[1] == t_ID] for t_ID in t_merge_fake_to_ID}       # e.g {10:[(8, 10), (9, 10)]}
            t_split_fake_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_splits if t_conn[0] == t_ID] for t_ID in t_split_fake_from_ID}     # e.g {15:[(15, 16), (15, 17)]} 

            t_merge_real_to_ID_resolved = []
            t_split_real_from_ID_resolved = []

                    
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
            t_event_start_end_times = {'merge':{},'split':{},'mixed':{}}

            func_prev_neighb = lambda x: G2.nodes[x]["t_end"]
            func_next_neighb = lambda x: G2.nodes[x]["t_start"]

            # ------------------------------------- REAL MERGE/SPLIT -------------------------------------
            t_state = 'merge'
            #for t_to in t_merge_real_to_ID: 
            for t_to in lr_ms_edges_main[t_state]: 
                t_predecessors   = extractNeighborsPrevious(G2, t_to, func_prev_neighb)
                t_predecessors = [lr_C0_condensed_connections_relations[t] for t in t_predecessors] # update if inherited
                t_predecessors_times_end = {t:G2.nodes[t]["t_end"] for t in t_predecessors}
                t_t_to_start = G2.nodes[t_to]["t_start"]
                t_times         = {t_ID: np.arange(t_t_from_end, t_t_to_start + 1, 1)
                                   for t_ID,t_t_from_end in t_predecessors_times_end.items()}
                t_t_from_min = min(t_predecessors_times_end.values())    # take earliest branch time

                if t_t_to_start - t_t_from_min <= 1: continue


                # pre-isolate graph segment where all sub-events take place
                t_node_to_first = t_segments_new[t_to][0]
                t_nodes_keep    = [node for node in G.nodes() if t_t_from_min < G_time(node) < t_t_to_start and G_owner(node) is None] 
                #for t in t_predecessors:
                #    t_nodes_keep.append(t_segments_new[t][-1])
                t_nodes_keep.append(t_node_to_first)

                t_subgraph = G.subgraph(t_nodes_keep)   

                #t_segments_keep = t_predecessors + [t_to]  
                #t_sol = graph_sub_isolate_connected_components(t_subgraph, t_t_from_min, t_t_to_start, lr_time_active_segments,
                #                                                       t_segments_new, t_segments_keep, ref_node = t_node_to_first) 
                #t_node_subIDs_all = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs

                t_sol2 = set()
                dfs_pred(t_subgraph, t_node_to_first, time_lim = t_t_from_min , node_set = t_sol2)
                t_node_subIDs_all = disperse_nodes_to_times(t_sol2) # reformat sol into time:subIDs
                t_node_subIDs_all = {t:sorted(t_node_subIDs_all[t]) for t in sorted(t_node_subIDs_all)}
                
                t_active_IDs = {t:[] for t in np.arange(t_t_from_min + 1, t_t_to_start)}
                for t_from, t_times_all in t_times.items():
                    for t_time in t_times_all[1:-1]: # remove start-end points
                        t_active_IDs[t_time].append(t_from)
                t_event_start_end_times[t_state][t_to] = {'t_start':        t_predecessors_times_end,
                                                          'branches':       t_predecessors,
                                                          't_end':          t_t_to_start,
                                                          't_times':        t_times,
                                                          't_perms':        {},
                                                          't_subIDs':       t_node_subIDs_all,
                                                          't_active_IDs':   t_active_IDs}

                t_merge_real_to_ID_resolved.append(t_to)
                
            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            t_state = 'split'
            #for t_from in t_split_real_from_ID:
            for t_from in lr_ms_edges_main[t_state]:
                t_from_new = lr_C0_condensed_connections_relations[t_from]                  # split segment may be inherited.
                t_from_successors   = extractNeighborsNext(G2, t_from_new, func_next_neighb)     # branches are not
                t_successors_times = {t_to:G2.nodes[t_to]["t_start"] for t_to in t_from_successors}
                t_t_start = G2.nodes[t_from_new]["t_end"]
                t_times         = {t:np.arange(t_t_start, t_t_end + 1, 1) for t,t_t_end in t_successors_times.items()}
                
                t_node_from = t_segments_new[t_from_new][-1]
                # pre-isolate graph segment where all sub-events take place
                t_t_to_max = max(t_successors_times.values())

                if t_t_to_max - t_t_start <= 1: continue

                t_nodes_keep    = [node for node in G.nodes() if t_t_start < G_time(node) < t_t_to_max and G_owner(node) is None]

                #for t in t_from_successors:
                #    t_nodes_keep.append(t_segments_new[t][0])
                t_nodes_keep.append(t_segments_new[t_from_new][-1])

                t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph

                #t_segments_keep = t_from_successors + [t_from_new]  
                #t_sol = graph_sub_isolate_connected_components(t_subgraph, t_t_start, t_t_to_max, lr_time_active_segments,
                #                                                       t_segments_new, t_segments_keep, ref_node = t_node_from) 
                #t_node_subIDs_all = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs

                t_sol2 = set()
                dfs_succ(t_subgraph, t_node_from, time_lim = t_t_to_max , node_set = t_sol2)
                t_node_subIDs_all = disperse_nodes_to_times(t_sol2) # reformat sol into time:subIDs
                t_node_subIDs_all = {t:sorted(t_node_subIDs_all[t]) for t in sorted(t_node_subIDs_all)}

                t_active_IDs = {t:[] for t in np.arange(t_t_to_max -1, t_t_start, -1)} #>>> reverse for reverse re

                for t_from, t_times_all in t_times.items():
                    for t_time in t_times_all[1:-1]: # remove start-end points
                        t_active_IDs[t_time].append(t_from)
                t_event_start_end_times[t_state][t_from_new] = {'t_start' :       t_t_start,
                                                              'branches':       t_from_successors,
                                                              't_end'   :       t_successors_times,
                                                              't_times' :       t_times,
                                                              't_perms':        {},
                                                              't_subIDs':       t_node_subIDs_all,
                                                              't_active_IDs':   t_active_IDs}

                t_split_real_from_ID_resolved.append(t_from_new)

            print(f'\n{timeHMS()}:({doX_s}) Real merges({lr_ms_edges_main["merge"]})/splits({lr_ms_edges_main["split"]})... Done')

            # ------------------------------------- FAKE MERGE/SPLIT -------------------------------------
            t_state = 'merge'
            for t_to in copy.deepcopy(t_merge_fake_to_ID):
            
                t_to_predecessors   = extractNeighborsPrevious(G2, t_to, func_prev_neighb)  # should be updated prior
                t_to_pre_predecessors = []
                for t_to_pre in t_to_predecessors:
                    t_to_pre_predecessors += extractNeighborsPrevious(G2, t_to_pre, func_prev_neighb)
                if len(t_to_pre_predecessors) > 1:
                    t_merge_fake_to_ID.remove(t_to)
                    continue
                # if fake branches should terminate and only 1 OG should be left. check if this is the case
                #assert len(t_to_pre_predecessors) == 1, "fake branch/es consists of multiple segments, not yet encountered"
                t_to_pre_pre_ID = t_to_pre_predecessors[0]
                t_t_start         = G2.nodes[t_to_pre_pre_ID]["t_end"]
                t_t_end           = G2.nodes[t_to]["t_start"]
                t_times         = {(t_to_pre_pre_ID, t_to):np.arange(t_t_start, t_t_end + 1, 1)}
                t_event_start_end_times[t_state][t_to] = {'t_start':        t_t_start,
                                                          'pre_predecessor':t_to_pre_pre_ID,
                                                          'branches':       t_to_predecessors,
                                                          't_end':          t_t_end,
                                                          't_times' :       t_times}

                #[t_segments_new[t][-1] for t in t_to_predecessors]
                t_segments_keep = [t_to_pre_pre_ID] + t_to_predecessors + [t_to]
            
                t_ref_ID = lr_C0_condensed_connections_relations[t_to_pre_pre_ID]

                t_sol = graph_sub_isolate_connected_components(G, t_t_start, t_t_end, lr_time_active_segments,
                                                               t_segments_new, t_segments_keep, ref_node = t_segments_new[t_ref_ID][-1])
                t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                seqs,t_nodes_pre = perms_with_branches(t_to_predecessors,t_segments_new,t_perms, return_nodes = True) 
                t_event_start_end_times[t_state][t_to]['t_combs'] = seqs
                t_event_start_end_times[t_state][t_to]['t_nodes_solo'] = t_sol  
                t_conn = (lr_C0_condensed_connections_relations[t_to_pre_pre_ID],t_to)
                for t_time, t_perms in t_nodes_pre.items():
                    for t_perm in t_perms:
                        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))

                        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                        lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
                        lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
                        lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z

            t_state = 'split'
            for t_from in t_split_fake_from_ID:
                t_from_new = lr_C0_condensed_connections_relations[t_from]
                t_from_successors    = extractNeighborsNext(G2, t_from_new, func_next_neighb) 
                t_post_successors = []
                for t_ID_pre in t_from_successors:
                    t_post_successors += extractNeighborsNext(G2, t_ID_pre, func_next_neighb)
                assert len(t_post_successors), "fake branch/es consists of multiple segments, not yet encountered"
                t_ID_post_succ  = t_post_successors[0]
                t_start         = G2.nodes[t_from_new]["t_end"]
                t_end           = G2.nodes[t_ID_post_succ]["t_start"]
                t_times         = {(t_from_new, t_ID_post_succ):np.arange(t_start, t_end + 1, 1)}
                t_event_start_end_times[t_state][t_from_new] = {'t_start':        t_start,
                                                          'post_successor': t_ID_post_succ,
                                                          'branches':       t_from_successors,
                                                          't_end':          t_end,
                                                          't_times':        t_times}
                t_from_last_node = t_segments_new[t_from_new][-1]
                # if main branch is long and recovered in 121s =  has composite nodes, then default algorithm will try to 
                # construct permutation from already resolved nodes, which is not needed. only consider stage with fake branch
                # assuming fake is short
                #t_end = min([G_time(t_segments_new[t][-1]) for t in  t_from_successors])
                #t_segments_keep = [None, t_from_new, t_ID_post_succ] + t_from_successors
                #t_from_last_node = t_segments_new[t_from_new][-1]
                #t_nodes_keep    = [node for node in G.nodes() if t_start < G_time(node) <= t_end and G_owner(node) in t_segments_keep] # < excluded edges
                #t_nodes_keep.append(t_from_last_node)

                #t_subgraph = G.subgraph(t_nodes_keep)

                #t_segments_keep = [t_from_new] + t_from_successors + [t_ID_post_succ]
                t_sol = graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments,
                                                               t_segments_new, t_segments_keep, ref_node = t_from_last_node)
                t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            
                seqs, t_nodes_pre = perms_with_branches(t_from_successors, t_segments_new, t_perms, return_nodes = True) 

                t_event_start_end_times[t_state][t_from_new]['t_combs'] = seqs

                t_event_start_end_times[t_state][t_from_new]['t_nodes_solo'] = t_sol 

                t_conn = (t_from_new,t_ID_post_succ)
                for t_time, t_perms in t_nodes_pre.items():
                    for t_perm in t_perms:
                        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))

                        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                        lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
                        lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
                        lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z

            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            t_state = 'mixed'
            for t_from_all,t_to_all in lr_ms_edges_main[t_state].items():
                t_from_all_new = [lr_C0_condensed_connections_relations[t] for t in t_from_all]
                t_predecessors_times_end = {t:G2.nodes[t]["t_end"] for t in t_from_all_new}
                
                t_successors_times_start = {t_to:G2.nodes[t_to]["t_start"] for t_to in t_to_all}
                t_target_nodes = {t:t_segments_new[t][0] for t in t_to_all}

                # pre-isolate graph segment where all sub-events take place
                t_t_from_min    = min(t_predecessors_times_end.values())    # take earliest branch time
                t_t_to_max      = max(t_successors_times_start.values()) 
                t_times_all = np.arange(t_t_from_min, t_t_to_max + 1, 1)
                t_active_IDs = {t:[] for t in t_times_all[1:]}

                # get non-segment nodes within event time inteval
                t_nodes_keep    = [node for node in G.nodes() if t_t_from_min < G_time(node) < t_t_to_max and G_owner(node) is None] # < excluded edges
                # add start of t_to segment, which are branch extension targets. 
                #for t in t_to_all:
                #    t_nodes_keep.append(t_segments_new[t][0])

                # i had problems with nodes due to  internal events. t_node_subIDs_all was missing times.
                # ill add all relevant segments to subgraph along stray nodes. gather connected components
                # and then subract all segment nodes, except first target nodes.
                t_nodes_segments_all = []
                for t in t_from_all + t_to_all:
                    t_nodes_segments_all.extend(t_segments_new[t])
                t_nodes_keep.extend(t_nodes_segments_all)

                # had cases where target nodes are decoupled and are out of connected components. connect them temporary
                t_fake_edges = list(itertools.combinations(t_target_nodes.values(), 2))
                G.add_edges_from(t_fake_edges)
                t_subgraph = G.subgraph(t_nodes_keep)   

                ref_node = t_segments_new[t_to_all[0]][0]

                t_sols = nx.connected_components(t_subgraph.to_undirected())
                
                t_sol = next((t for t in t_sols if ref_node in t), None)
                assert t_sol is not None, 'cannot find connected components'
                t_sol = [t for t in t_sol if t not in t_nodes_segments_all]

                for t in t_to_all:
                    t_sol.append(t_segments_new[t][0])
                #t_segments_keep = t_from_all_new + list(t_to_all)
                #t_sol = graph_sub_isolate_connected_components(t_subgraph, t_t_from_min + 1, t_t_to_max, lr_time_active_segments,
                #                                                t_segments_new, t_segments_keep, ref_node = t_segments_new[t_to_all[0]][0]) 
                G.remove_edges_from(t_fake_edges)
                t_node_subIDs_all = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                t_node_subIDs_all = {t:t_node_subIDs_all[t] for t in sorted(t_node_subIDs_all)}

                for t_from in t_from_all_new: # from end of branch time to event max time
                    for t_time in np.arange(t_predecessors_times_end[t_from] + 1, t_t_to_max + 1, 1):
                        t_active_IDs[t_time].append(t_from)

                t_event_start_end_times[t_state][t_from_all] = {'t_start':        t_predecessors_times_end,
                                                                'branches':       t_from_all_new,
                                                                't_end':          t_successors_times_start,
                                                                't_times':        {},
                                                                't_perms':        {},
                                                                't_target_nodes': t_target_nodes,
                                                                't_subIDs':       t_node_subIDs_all,
                                                                't_active_IDs':   t_active_IDs}

        # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # ===============================================================================================
    # ====  Determine interpolation parameters ====
    # ===============================================================================================
    # we are interested in segments that are not part of psuedo branches (because they will be integrated into path)
    # pre merge segments will be extended, post split branches also, but backwards. 
    # for pseudo event can take an average paramets for prev and post segments.
    print(f'\n{timeHMS()}:({doX_s}) Determining interpolation parameters k and s for segments...')
    t_fake_branches_IDs = []
    for t_ID in t_merge_fake_to_ID:
        t_fake_branches_IDs += t_event_start_end_times['merge'][t_ID]['branches']
    for t_ID in t_split_fake_from_ID:
        t = lr_C0_condensed_connections_relations[t_ID]
        t_fake_branches_IDs += t_event_start_end_times['split'][t]['branches']
    # get segments that have possibly inherited other segments and that are not branches
    t_segments_IDs_relevant = [t_ID for t_ID,t_traj in enumerate(t_segments_new) if len(t_traj)>0 and t_ID not in t_fake_branches_IDs]

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

    t_fake_events_k_s_edge_master = {}
    for t_to in t_merge_fake_to_ID:
        t_from =  t_event_start_end_times['merge'][t_to]['pre_predecessor']
        t_from  = lr_C0_condensed_connections_relations[t_from]
        t_from_k_s  = t_segment_k_s[t_from  ]
        t_to_k_s    = t_segment_k_s[t_to    ]
        t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))   # determine which k_s to inherit (lower)
        t_fake_events_k_s_edge_master[t_to] = {'state': 'merge', 'edge': (t_from, t_to), 'k_s':t_k_s_out}
        
    for t_from_old in t_split_fake_from_ID:
        t_from  = lr_C0_condensed_connections_relations[t_from_old]
        t_to = t_event_start_end_times['split'][t_from]['post_successor']
        t_from_k_s  = t_segment_k_s[t_from  ]
        t_to_k_s    = t_segment_k_s[t_to    ]
        t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))  
        t_fake_events_k_s_edge_master[t_from_old] = {'state': 'split', 'edge': (t_from, t_to), 'k_s':t_k_s_out}
    # ----------------------- interpolate missing events with known history-----------------------
    # interpolate
    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Performing interpolation on data without fake branches')
    a = 1
    for t_ID,t_param_dict in t_fake_events_k_s_edge_master.items():
        t_state         = t_param_dict['state'  ] 
        t_from, t_to    = t_param_dict['edge'   ]
        k,s             = t_param_dict['k_s'    ]
        t_subIDs = [t_from, t_to]
        t_ID = lr_C0_condensed_connections_relations[t_ID]
        t_start = t_event_start_end_times[t_state][t_ID]['t_start'  ]
        t_end   = t_event_start_end_times[t_state][t_ID]['t_end'    ]
        if 1 == 1:
            t_temp_nodes_all = []
            for t_subID in t_subIDs:
                t_temp_nodes_all += t_segments_new[t_subID]
            t_temp_times        = [G.nodes[t_node]["time"       ] for t_node in t_temp_nodes_all]
            t_temp_areas        = [G.nodes[t_node]["area"       ] for t_node in t_temp_nodes_all]
            t_temp_centroids    = [G.nodes[t_node]["centroid"   ] for t_node in t_temp_nodes_all]
        
            t_times_missing_all = np.arange(t_start + 1, t_end, 1)
        
            a = 1
            # interpolate composite (long) parameters 
            t_interpolation_centroids_0 = interpolateMiddle2D_2(t_temp_times,np.array(t_temp_centroids), t_times_missing_all, s = s, k = k, debug = 0, aspect = 'equal', title = t_subIDs)
            t_interpolation_areas_0     = interpolateMiddle1D_2(t_temp_times,np.array(t_temp_areas),t_times_missing_all, rescale = True, s = 15, debug = 0, aspect = 'auto', title = t_subIDs)
        
            t_conn = (t_from, t_to)
            lr_big121s_interpolation[t_conn]['centroids'] = np.array(t_interpolation_centroids_0)
            lr_big121s_interpolation[t_conn]['areas'    ] = np.array(t_interpolation_areas_0    )   
            lr_big121s_interpolation[t_conn]['times'    ] = t_times_missing_all
    # ----------------- Evaluate and find most likely evolutions of path -------------------------
    print(f'\n{timeHMS()}:({doX_s}) Finding most possible path evolutions...')
    if 1 == 1:
        t_temp_centroids = {}           # generate temp interpolation data storage
        t_temp_cases = {}
        t_temp_times = {}
    
        for t_ID,t_param_dict in t_fake_events_k_s_edge_master.items():
            t_ID = lr_C0_condensed_connections_relations[t_ID]
            t_state                     = t_param_dict['state'  ] 
            t_conn                      = t_param_dict['edge'   ]
            t_temp_centroids[t_conn]    = lr_big121s_interpolation[t_conn]['centroids']
            t_temp_cases[t_conn]        = t_event_start_end_times[t_state][t_ID]['t_combs']
            t_temp_times[t_conn]        = list(t_event_start_end_times[t_state][t_ID]['t_times'].values())[0]

        # combine interpolation and pre-computed options for bubble parameters. choose best fit to interp.
        t_args = [t_temp_cases, t_temp_cases,t_temp_centroids,t_temp_times,
                lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

        t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)

        t_weights   = [1,1.5,0,1]
        t_sols      = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
        lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_temp_cases, t_weights, t_sols, t_temp_cases )
    a = 1 # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ----------------- Save recovered fake event paths -------------------------
    print(f'\n{timeHMS()}:({doX_s}) Saving fake event data...')
    if  1 == 1:
            for t_ID_old,t_param_dict in t_fake_events_k_s_edge_master.items():
                t_ID = lr_C0_condensed_connections_relations[t_ID_old]
                t_state         = t_param_dict['state'  ] 
                t_conn          = t_param_dict['edge'   ]
                t_branches      = t_event_start_end_times[t_state][t_ID]['branches']
                t_sol           = lr_weighted_solutions_max[t_conn]
                t_from, t_to    = t_conn                                     
                # prep combs similar to 121 connection saving, except branches are scooped out and included in combs, and start-end IDs changed.
                t_combs = {t_time:list(t_subIDs) for t_time,t_subIDs in zip(t_temp_times[t_conn], t_temp_cases[t_conn][t_sol])}
                t_node_from = t_segments_new[lr_C0_condensed_connections_relations[t_from   ]][-1   ]
                t_node_to   = t_segments_new[lr_C0_condensed_connections_relations[t_to     ]][0    ]
                print(f'fake {t_state} event ({t_conn}) saved: {t_node_from} -> {t_node_to}') 
                save_connections_two_ways(t_segments_new, t_combs, t_from, t_to, G, G2, lr_C0_condensed_connections_relations, g0_contours)

                for t_segment_remove in t_branches:        # remove and change refs of fake branches
                    t_segments_new[t_segment_remove] = []
                    G2.remove_node(t_segment_remove)
                    t_from_new = lr_C0_condensed_connections_relations[t_from]
                    lr_C0_condensed_connections_relations[t_segment_remove] = t_from_new

                if t_state == 'merge':   
                    lr_conn_merges_to_nodes.remove(t_ID_old) # its not real, remove
                else:
                    lr_conn_splits_from_nodes.remove(t_ID_old)

                #for_graph_plots(G, segs = t_segments_new)
                a = 1
                
    G_seg_view_2 = nx.Graph()
    G_seg_view_2.add_edges_from([(x,y) for y,x in lr_C0_condensed_connections_relations.items()])

    #lr_C1_condensed_connections = extract_graph_connected_components(G_seg_view_2, lambda x: x)
    lr_C1_condensed_connections = [sorted(c, key = lambda x: x) for c in nx.connected_components(G_seg_view_2)]
    lr_C1_condensed_connections = sorted(lr_C1_condensed_connections, key = lambda x: x[0])
    # lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
    t_condensed_connections_all_nodes = sorted(sum(lr_C1_condensed_connections,[])) # neext next
    lr_C1_condensed_connections_relations = {tID: tID for tID in range(len(segments2))} 
    for t_subIDs in lr_C1_condensed_connections:
        for t_subID in t_subIDs:
            lr_C1_condensed_connections_relations[t_subID] = min(t_subIDs)


    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ===============================================================================================
    # ============================  EXTEND REAL MERGE/SPLIT BRANCHES ================================
    # ===============================================================================================
    # iteratively extend branches though split/merge event. to avoid conflicts with shared nodes
    # extension is simultaneous for all branches at each time step and contours are redistributed conservatively
    # -> at time T, two branches [B1, B2] have to be recovered from node cluster C = [1,2,3]
    # -> try node redistributions- ([partition_B1,partition_B2], ..) = ([[1],[2,3]], [[2],[1,3]], [[3],[1,2]])    

    t_out                   = defaultdict(dict)
    t_extrapolate_sol       = defaultdict(dict)
    t_extrapolate_sol_comb  = defaultdict(dict)
    ms_branch_extend_IDs =  [(t,'merge') for t in t_merge_real_to_ID_resolved] 
    ms_branch_extend_IDs += [(lr_C1_condensed_connections_relations[t],'split') for t in t_split_real_from_ID_resolved]
    #ms_branch_extend_IDs += [(t,'mixed') for t in lr_conn_mixed_from_to]
    ms_branch_extend_IDs += [(t,'mixed') for t in lr_ms_edges_main['mixed']]
    

    ms_mixed_completed      = {'full': defaultdict(dict),'partial': defaultdict(dict)} 
    lr_post_branch_rec_info = {}
    ms_early_termination = {}
    print(f'\n{timeHMS()}:({doX_s}) Analyzing real merge/split events. extending branches: {ms_branch_extend_IDs} ... ')
    for t_ID, t_state in ms_branch_extend_IDs:
        if t_state in ('merge','split'):
            t_branches = t_event_start_end_times[t_state][t_ID]['branches']
            t_times_target = []
        elif t_state == 'mixed':
            t_branches = t_event_start_end_times[t_state][t_ID]['branches']
            t_nodes_target = t_event_start_end_times[t_state][t_ID]['t_target_nodes']
            t_times_target = [t[0] for t in t_nodes_target.values()]
            t_subIDs_target = {t:[] for t in t_times_target}
            for t_time, *t_subIDs in t_nodes_target.values():
                t_subIDs_target[t_time] += t_subIDs

        if 1 == 1:

            if t_state == 'split':
                t_start = t_event_start_end_times[t_state][t_ID]['t_start']
                t_end   = min(t_event_start_end_times[t_state][t_ID]['t_end'].values())
            elif t_state == 'mixed':
                t_start = min(t_event_start_end_times[t_state][t_ID]['t_start'].values())
                t_end   = max(t_event_start_end_times[t_state][t_ID]['t_end'].values())
            else:
                t_start = min(t_event_start_end_times[t_state][t_ID]['t_start'].values())
                t_end   = t_event_start_end_times[t_state][t_ID]['t_end']

            a = 1
            t_all_norm_buffers  = {}
            t_all_traj_buffers  = {}
            t_all_area_buffers  = {}
            t_all_time_buffers  = {}
            t_all_k_s           = {}
            # ==== prepare data for extrapolation START ====
            for t_branch_ID in t_branches:  
                t_branch_ID_new = lr_C1_condensed_connections_relations[t_branch_ID]
                if t_state in ('merge','mixed'):
                
                    t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'][t_branch_ID] # last of branch
                    t_node_from = t_segments_new[t_branch_ID_new][-1]
                    if t_state == 'merge':
                        t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end']                # first of target
                        t_node_to   = t_segments_new[t_ID][0]
                        t_conn = (t_branch_ID, t_ID)
                    else: 
                        t_t_to = max(t_event_start_end_times[t_state][t_ID]['t_end'].values())
                        t_node_to = (-1)
                        t_conn = (t_branch_ID,)
                else:
                    t_conn = (t_ID, t_branch_ID)
                    t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'] # last of branch
                    t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end'][t_branch_ID]                # first of target
                    t_node_from = t_segments_new[t_ID][-1]
                    t_node_to   = t_segments_new[t_branch_ID][0]
                if t_state in ('split', 'merge') and np.abs(t_t_to - t_t_from) < 2:    # there are mixed cases with zero event nodes
                    t_out[t_ID][t_branch_ID] = None
                    continue

                if t_state in ('merge','mixed'):
                    t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if t_node[0] > t_t_from - h_interp_len_max2]
                else:
                    t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if t_node[0] < t_t_to + h_interp_len_max2]

                trajectory  = np.array([G.nodes[t]["centroid"] for t in t_nodes])
                time        = np.array([G.nodes[t]["time"    ] for t in t_nodes])
                area        = np.array([G.nodes[t]["area"    ] for t in t_nodes])
            
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
            # ==== prep data for extrapolation END ====

            # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # ==== extrapolation iterations START ====
            t_branch_failed = []
            t_report = set()
            # walk times from start of an event. each time has a branch IDs that has to be recovered "t_branch_IDs_OG" = "t_branch_IDs",
            # and pool of available contours "t_subIDs". some branches may start later and start using contours form pool.
            # once branch has terminated, it stops being investigated (deleted from "t_branch_IDs_OG" -> new "t_branch_ID")
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
                    t_areas_extrap[t]     = interpolateMiddle1D_2(t_time_b,t_area_b,[t_time_next], rescale = True, s = 15, debug = 0, aspect = 'auto', title = 1)
                    
                    #t_centroids_extrap[t] = t_centr_ext
                    #t_areas_extrap[t]     = t_area_ext
                #t_centroids_extrap = np.array(t_centroids_extrap).reshape(-1,2)
                if len(t_branch_IDs) == 1:
                    t_perms_distribution2 = [[list(t)] for t in combs_different_lengths(t_subIDs)]  # if only one choice, gen diff perms of contours
                else:
                    t_perms_distribution2 = list(split_into_bins(t_subIDs,len(t_branch_IDs)))       # multiple choices, get perm distrib options
                t_perms_distribution2 = [[tuple(sorted(b)) for b in a] for a in t_perms_distribution2]
                t_permutation_params = {}
                t_permutations = combs_different_lengths(t_subIDs)
                t_permutations = [tuple(sorted(a)) for a in t_permutations]
                for t_permutation in t_permutations:
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time_next][tID] for tID in t_permutation]))
                    t_permutation_params[t_permutation] = centroid_area(t_hull)

                t_diff_choices      = {}      # holds index key on which entry in t_perms_distribution2 has specifict differences with target values.
                t_diff_choices_area = {}
                for k, t_redist_case in enumerate(t_perms_distribution2):
                    t_centroids             = np.array([t_permutation_params[t][0] for t in t_redist_case])
                    t_areas                 = np.array([t_permutation_params[t][1] for t in t_redist_case])
                    t_diff_choices[k]       = np.linalg.norm(t_centroids_extrap - t_centroids, axis=1)
                    t_diff_choices_area[k]  = np.abs(t_areas_extrap - t_areas)/t_areas_extrap
                a = 1
                # -------------------------------- refine simultaneous solution ---------------------------------------
                # evaluating sum of diffs is a bad approach, becaues they may behave differently

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
        t_from_new = lr_C1_condensed_connections_relations[t_from]
        t_combs = t_extrapolate_sol_comb[t_conn]

        if len(t_combs) == 0: continue                              # no extension, skip.

        if t_state in ('merge', 'mixed') and t_conn not in ms_mixed_completed['full']:
            save_connections_merges(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from_new,  None, G, G2, lr_C1_condensed_connections_relations, g0_contours)
        elif t_state == 'split':
            save_connections_splits(t_segments_new, t_extrapolate_sol_comb[t_conn], None,  t_to, G, G2, lr_C1_condensed_connections_relations, g0_contours)
        else:
            t_to = ms_mixed_completed['full'][t_conn]['targets'][0]
            # remove other edges from mixed connection segment graph
            t_from_other_predecessors = [lr_C1_condensed_connections_relations[t] for t in extractNeighborsPrevious(G2, t_to, func_prev_neighb) if t != t_conn[0]]
            t_edges = [(t, t_to) for t in t_from_other_predecessors]
            G2.remove_edges_from(t_edges)
            save_connections_two_ways(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from_new,  t_to, G, G2, lr_C1_condensed_connections_relations, g0_contours)


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
    t_segments_fin_dic, skipped = graph_extract_paths(G,lambda x : x[0])
    t_segments_fin = [t for t in t_segments_fin_dic.values() if len(t) > 2]

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
        G2.add_node(t_new_ID)
        G2.nodes()[t_new_ID]["t_start"] =  G_time(t_segments_new[t_new_ID][0] )
        G2.nodes()[t_new_ID]["t_end"]   =  G_time(t_segments_new[t_new_ID][-1])
        set_custom_node_parameters(G, g0_contours_hulls, t_segments_fin[t_ID], t_new_ID, calc_hull = 0) # straight hulls since solo nodes 
        #set_custom_node_parameters(G, g0_contours, t_segments_fin[t_ID], t_new_ID, calc_hull = 1)        # 


    lr_time_active_segments = defaultdict(list)
    for t_segment_index, t_segment_nodes in enumerate(t_segments_new):
        t_times = [G_time(node) for node in t_segment_nodes]
        for t_time in t_times:
            lr_time_active_segments[t_time].append(t_segment_index)
    # sort keys in lr_time_active_segments
    lr_time_active_segments = {t:lr_time_active_segments[t] for t in sorted(lr_time_active_segments.keys())}

    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for_graph_plots(G, segs = t_segments_fin)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(f'\n{timeHMS()}:({doX_s}) Final. Update connectivity')
    # ============== Final passes. New straight segments. Update connectivity ===================
    #G2.add_edges_from(G2.edges())
    G2_new = nx.DiGraph()
    fin_connectivity_graphs = defaultdict(list) #fin_additional_segments_IDs
    t_segments_relevant = np.array([t for t,t_seg in enumerate(t_segments_new) if len(t_seg) > 0])

    t_segment_time_start = np.array([G2.nodes[t]["t_start"] for t in t_segments_relevant])

    for t_ID_from in t_segments_relevant:        
        G2_new.add_node(t_ID_from)
        G2_new.nodes()[t_ID_from]["t_start"   ] = G_time(t_segments_new[t_ID_from][0] )
        G2_new.nodes()[t_ID_from]["t_end"     ] = G_time(t_segments_new[t_ID_from][-1])
        #t_DT = lr_maxDT                     
        t_t_from = G2.nodes[t_ID_from]["t_end"]
        timeDiffs = t_segment_time_start - t_t_from

        t_DT_pass_index = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
        t_IDs_DT_pass = t_segments_relevant[t_DT_pass_index]
        node_from = t_segments_new[t_ID_from][-1]
        time_from = G_time(node_from)
        for t_ID_to in t_IDs_DT_pass:
            node_to         = t_segments_new[t_ID_to][0]
            time_to         = G_time(node_to)
            t_nodes_keep    = [node for node in G.nodes() if time_from <= G_time(node) <= time_to and G_owner(node) is None] 
            t_nodes_keep.extend([node_from,node_to])
            G_sub = G.subgraph(t_nodes_keep)
            hasPath = nx.has_path(G_sub, source = node_from, target = node_to)
            if hasPath:
                G2_new.add_edge(t_ID_from, t_ID_to, dist = time_to - time_from + 1) # include end points =  inter + 2
                
        a = 1

    #fin_connectivity_graphs = {t_conn:t_vals for t_conn, t_vals in fin_connectivity_graphs.items() if len(t_vals) > 2}
    for t_ID in fin_additional_segments_IDs:
        t_segment_k_s_diffs[t_ID] = None
        t_segment_k_s[t_ID] = None
        lr_C1_condensed_connections_relations[t_ID] = t_ID
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
        t_all_event_IDs             = [lr_C1_condensed_connections_relations[t_ID] for t_ID in t_event_start_end_times[t_state]]
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
        t_all_event_IDs             = [lr_C1_condensed_connections_relations[t_ID] for t_ID in t_event_start_end_times[t_state]]
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
        for t_time, t_permutations in t_combs.items():
            
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
                    continue
    
    
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
            save_connections_merges(t_segments_new, t_extrapolate_sol_comb[t_conn], t_ID,  None, G, G2_dir, lr_C1_condensed_connections_relations, g0_contours)
        elif t_state == 'back':
            save_connections_splits(t_segments_new, t_extrapolate_sol_comb[t_conn], None,  t_ID, G, G2_dir, lr_C1_condensed_connections_relations, g0_contours)


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
    # for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    print(f'\n{timeHMS()}:({doX_s}) Final. Recalculate connectivity')
    # ======= EXPORT. RECALCULATE CONNECTIONS BETWEEN SEGMENTS ========================
    G2_new = nx.DiGraph()

    t_segments_relevant     = np.array([t for t,t_seg in enumerate(t_segments_new) if len(t_seg) > 0])
    t_segment_time_start    = np.array([G2.nodes[t]["t_start"] for t in t_segments_relevant])
    

    for t_ID_from in tqdm(t_segments_relevant):        
        G2_new.add_node(t_ID_from)
        G2_new.nodes()[t_ID_from]["t_start"   ] = G_time(t_segments_new[t_ID_from][0] )
        G2_new.nodes()[t_ID_from]["t_end"     ] = G_time(t_segments_new[t_ID_from][-1])

        t_t_from = G2.nodes[t_ID_from]["t_end"]
        timeDiffs = t_segment_time_start - t_t_from

        t_DT_pass_index = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
        t_IDs_DT_pass = t_segments_relevant[t_DT_pass_index]
        node_from = t_segments_new[t_ID_from][-1]
        time_from = G_time(node_from)
        for t_ID_to in t_IDs_DT_pass:
            node_to         = t_segments_new[t_ID_to][0]
            time_to         = G_time(node_to)
            t_nodes_keep    = [node for node in G.nodes() if time_from <= G_time(node) <= time_to and G_owner(node) is None] 
            t_nodes_keep.extend([node_from,node_to])
            G_sub = G.subgraph(t_nodes_keep)
            hasPath = nx.has_path(G_sub, source = node_from, target = node_to)
            if hasPath:
                G2_new.add_edge(t_ID_from, t_ID_to, dist = time_to - time_from + 1) # include end points =  inter + 2
                
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
            all_nodes_pos, edge_width, edge_color, node_size = for_graph_plots(G, segs = trajectories_all_dict[doX], show = False, node_size = 30, edge_width_path = 3, edge_width = 1, font_size = 7)
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