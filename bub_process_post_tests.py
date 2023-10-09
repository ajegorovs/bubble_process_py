from ast import For
import enum, copy, random, timeit
from collections import defaultdict
from tracemalloc import start
import numpy as np, itertools, networkx as nx, sys, time as time_lib
import cv2, os, glob, datetime, re, pickle#, multiprocessing
# import glob
from matplotlib import pyplot as plt
from tqdm import tqdm

# import from custom sub-folders are defined bit lower
#from imageFunctionsP2 import (overlappingRotatedRectangles,graphUniqueComponents)

# =========== BUILD OUTPUT FOLDERS =============//
inputOutsideRoot            = 1                                                  # bmp images inside root, then input folder hierarchy will
mainInputImageFolder        = r'.\inputFolder'                                   # be created with final inputImageFolder, else custom. NOT USED?!?
inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\100 sccm' #

mainOutputFolder            = r'.\post_tests'                        # these are main themed folders, sub-projects go inside.
if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)
sys.path.append( os.path.join(mainOutputFolder,'modules'))
#from post_tests.modules.cropGUI import cropGUI
from cropGUI import cropGUI

from graphs_brects import (overlappingRotatedRectangles, graphUniqueComponents)

#from graph_visualisation_01 import (draw_graph_with_height)

from image_processing import (convertGray2RGB, undistort)

from bubble_params import (centroid_area_cmomzz, centroid_area)

from graphs_general import (extractNeighborsNext, extractNeighborsPrevious, graph_extract_paths, for_graph_plots, extract_graph_connected_components,
                            find_segment_connectivity_isolated,  graph_sub_isolate_connected_components, set_custom_node_parameters)

from interpolation import (interpolate_trajectory, interpolate_find_k_s, extrapolate_find_k_s, interpolateMiddle2D_2, interpolateMiddle1D_2)

from misc import (cyclicColor, closes_point_contours, timeHMS, modBR, rect2contour, combs_different_lengths, unique_sort_list, disperse_nodes_to_times, disperse_composite_nodes_into_solo_nodes,
                  find_key_by_value, CircularBuffer, CircularBufferReverse, AutoCreateDict, find_common_intervals, unique_active_segments, lr_reindex_masters, getNodePos,
                  getNodePos2, segment_conn_end_start_points, lr_evel_perm_interp_data, lr_weighted_sols, perms_with_branches, save_connections_two_ways, save_connections_merges, save_connections_splits)

mainOutputSubFolders =  ['HFS 200 mT Series 4','sccm100-meanFix', "00001-05000"]
for folderName in mainOutputSubFolders:     
    mainOutputFolder = os.path.join(mainOutputFolder, folderName)
    if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)

a = outputImagesFolderName, outputStagesFolderName, outputStorageFolderName = ['images', 'stages',  'archives']
b = ['']*len(a)

for i,folderName in enumerate(a):   
    tempFolder = os.path.join(mainOutputFolder, folderName)
    if not os.path.exists(tempFolder): os.mkdir(tempFolder)
    b[i] = tempFolder
imageFolder, stagesFolder, dataArchiveFolder = b



# check crop mask 

cropMaskName = "-".join(mainOutputSubFolders[:2])+'-crop'
cropMaskPath = os.path.join(os.path.join(*mainOutputFolder.split(os.sep)[:-1]), f"{cropMaskName}.png")
cropMaskMissing = True if not os.path.exists(cropMaskPath) else False

meanImagePath   = os.path.join(dataArchiveFolder, "-".join(["mean"]+mainOutputSubFolders)+".npz")
meanImagePathArr= os.path.join(dataArchiveFolder, "-".join(["meanArr"]+mainOutputSubFolders)+".npz")


archivePath             = os.path.join(stagesFolder, "-".join(["croppedImageArr"]+mainOutputSubFolders)+".npz")
binarizedArrPath        = os.path.join(stagesFolder, "-".join(["binarizedImageArr"]+mainOutputSubFolders)+".npz")

dataStart           = 0 #736 #300 #  53+3+5
dataNum             = 500#5005 #130 # 7+5   

assistManually      = 1
assistFramesG       = []    #845,810,1234 2070,2187,1396
assistFrames        = [a - dataStart for a in assistFramesG]
doIntermediateData              = 1                                         # dump backups ?
intermediateDataStepInterval    = 500                                       # dumps latest data field even N steps
readIntermediateData            = 1                                         # load backups ?
startIntermediateDataAtG        = 60700                             # frame stack index ( in X_Data). START FROM NEXT FRAME
startIntermediateDataAt         = startIntermediateDataAtG - dataStart      # to global, which is pseudo global ofc. global to dataStart
# ------------------- this manual if mode  is not 0, 1  or 2



# ========================================================================================================
# ============== Import image files and process them, store in archive or import archive =================
# ========================================================================================================
exportArchive   = 0
useIntermediateData = 1
rotateImageBy   = cv2.ROTATE_180 # -1= no rotation, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180 
startFrom       = 1   #0 or 1 for sub exp                               # offset from ordered list of images- global offset?! yes archive adds images from list as range(startFrom, numImages)
numImages       = 5005 # DONT! intervalStart is what you are after!!!!! # take this many, but it will be updated: min(dataNum,len(imageLinks)-startFrom), if there are less available
postfix         = "-00001-05000"

intervalStart   = 0                         # in ordered list of images start from number intervalStart
intervalStop    = intervalStart + numImages  # and end at number intervalStop
useMeanWindow   = 0                          # averaging intervals will overlap half widths, read more below
N               = 500                        # averaging window width

#--NOTES: startFrom might work incorrectly, use it starting from 0.   
if exportArchive == 1:
    # ===================================================================================================
    # =========================== Get list of paths to image files in working folder ====================
    # ===================================================================================================
    imageLinks = glob.glob(inputImageFolder + "**/*.bmp", recursive=True) 
    if len(imageLinks) == 0:
        input("No files inside directory, copy them and press any key to continue...")
        imageLinks = glob.glob(inputImageFolder + "**/*.bmp", recursive=True) 
    # ===================================================================================================
    # ========================== Splice and sort file names based on criteria ===========================
    # ===================================================================================================
    # here is an example of badly padded data: [.\imt3509,.\img351,.\img3510,.\img3511,...]
    # ------------------------ can filter out integer values out of range -------------------------------
    # ---------------------------------------------------------------------------------------------------
             
    
    extractIntergerFromFileName = lambda x: int(re.findall('\d+', os.path.basename(x))[0])
    imageLinks = list(filter(lambda x: intervalStop > extractIntergerFromFileName(x) > intervalStart , imageLinks))
    # ----------------------- can sort numerically based on integer part---------------------------------
    imageLinks.sort(key=extractIntergerFromFileName)         # and sort alphabetically
    # ===================================================================================================
    # ======== Crop using a mask (draw red rectangle on exportad sample in manual mask folder) ==========
    # ===================================================================================================
    
    if cropMaskMissing:
        # === draw crop rectangle using gui. module only works with img link. so i save it on disk
        # open via GUI, get crop rectangle corners and draw red mask on top and save it. ===
        print(f"No crop mask in {mainOutputFolder} folder!, creating mask : {cropMaskName}.png")
        mapXY = (np.load('./mapx.npy'), np.load('./mapy.npy'))
        cv2.imwrite(cropMaskPath, convertGray2RGB(undistort(cv2.imread(imageLinks[0],0), mapXY)))
        p1,p2 = cropGUI(cropMaskPath)
        cropMask = cv2.imread(cropMaskPath,1)
        cv2.rectangle(cropMask, p1, p2,[0,0,255],-1)
        cv2.imwrite(cropMaskPath,cropMask)

        
    else:
        cropMask = cv2.imread(cropMaskPath,1)
    # ---------------------------- Isolate red rectangle based on its hue ------------------------------
    cropMask = cv2.cvtColor(cropMask, cv2.COLOR_BGR2HSV)

    lower_red = np.array([(0,50,50), (170,50,50)])
    upper_red = np.array([(10,255,255), (180,255,255)])

    manualMask = cv2.inRange(cropMask, lower_red[0], upper_red[0])
    manualMask += cv2.inRange(cropMask, lower_red[1], upper_red[1])

    # --------------------- Extract mask contour-> bounding rectangle (used for crop) ------------------
    contours = cv2.findContours(manualMask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    [X, Y, W, H] = cv2.boundingRect(contours[0])
    # ---------------------------------------------------------------------------------------------------
    # ---------------- Rotate and fix optical distortion (should be prepared previously)-----------------
    # ---------------------------------------------------------------------------------------------------
    print(f"{timeHMS()}: Processing and saving archive data on drive...")

    numImages   = min(dataNum,len(imageLinks)-startFrom)
    if rotateImageBy % 2 == 0 and rotateImageBy != -1: W,H = H,W                       # for cv2.XXX rotation commands
    dataArchive = np.zeros((numImages,H,W),np.uint8)
    mapXY       = (np.load('./mapx.npy'), np.load('./mapy.npy'))
    for i,j in tqdm(enumerate(range(startFrom-1, numImages))):
        if rotateImageBy != -1:
            dataArchive[i]    = cv2.rotate(undistort(cv2.imread (imageLinks[j],0), mapXY)[Y:Y+H, X:X+W],rotateImageBy)
        else:
            dataArchive[i]    = undistort(cv2.imread (imageLinks[j],0), mapXY)[Y:Y+H, X:X+W]
    print(f"{timeHMS()}: Processing and saving archive data on drive...saving compressed")
    np.savez_compressed(archivePath,dataArchive)
    #with open(archivePath, 'wb') as handle: 
    #    pickle.dump(dataArchive, handle) 
    
    print(f"{timeHMS()}: Exporting mean image...calculating mean")

    meanImage = np.mean(dataArchive, axis=0)
    print(f"{timeHMS()}: Exporting mean image...saving compressed")
    np.savez_compressed(meanImagePath,meanImage)
    #with open(meanImagePath, 'wb') as handle:
    #    pickle.dump(meanImage, handle)
    
    print(f"{timeHMS()}: Processing and saving archive data on drive... Done!")

elif not os.path.exists(archivePath):
    print(f"{timeHMS()}: No archive detected! Please generate it from project images.")

elif not useIntermediateData:
    print(f"{timeHMS()}: Existing archive found! Importing data...")
    dataArchive = np.load(archivePath)['arr_0']
    #with open(archivePath, 'rb') as handle:
    #    dataArchive = pickle.load(handle)
    print(f"{timeHMS()}: Existing archive found! Importing data... Done!")

    if not os.path.exists(meanImagePath):
        print(f"{timeHMS()}: No mean image found... Calculating mean")

        meanImage = np.mean(dataArchive, axis=0)
        print(f"{timeHMS()}: No mean image found... Saving compressed")
        np.savez_compressed(meanImagePath,meanImage)

        #with open(meanImagePath, 'wb') as handle:
        #    pickle.dump(meanImage, handle)
        print(f"{timeHMS()}: No mean image found... Done")
    else:
        meanImage = np.load(meanImagePath)['arr_0']
        #with open(meanImagePath, 'rb') as handle:
        #    meanImage = pickle.load(handle)
#cv2.imshow(f'mean',meanImage.astype(np.uint8))



# =========================================================================================================
# discrete update moving average with window N, with intervcal overlap of N/2
# [-interval1-]         for first segment: interval [0,N]. switch to next window at i = 3/4*N,
#           |           which is middle of overlap. 
#       [-interval2-]   for second segment: inteval is [i-1/4*N, i+3/4*N]
#                 |     third switch 1/4*N +2*[i-1/4*N, i+3/4*N] and so on. N/2 between switches
if useMeanWindow == 1 and not useIntermediateData:
    meanIndicies = np.arange(0,dataArchive.shape[0],1)                                                       # index all images
    meanWindows = {}                                                                                         # define timesteps at which averaging
    meanWindows[0] = [0,N]                                                                                   # window is switched. eg at 0 use
                                                                                                             # np.mean(archive[0:N])
    meanSwitchPoints = np.array(1/4*N + 1/2*N*np.arange(1, int(len(meanIndicies)/(N/2)), 1), int)            # next switch points, by geom construct
                                                                                                             # 
    for t in meanSwitchPoints:                                                                               # intervals at switch points
        meanWindows[t] = [t-int(1/4*N),min(t+int(3/4*N),max(meanIndicies))]                                  # intervals have an overlap of N/2
    meanWindows[meanSwitchPoints[-1]] = [meanWindows[meanSwitchPoints[-1]][0],max(meanIndicies)]             # modify last to include to the end
    intervalIndecies = {t:i for i,t in enumerate(meanWindows)}                                               # index switch points {i1:0, i2:1, ...}
                                                                                                             # so i1 is zeroth interval
    print(meanWindows)                                                                                       
    print(intervalIndecies)

    if not os.path.exists(meanImagePathArr):
        print(f"{timeHMS()}: Mean window is enabled. No mean image array found. Generating and saving new...")
        masksArr = np.array([np.mean(dataArchive[start:stop], axis=0) for start,stop in meanWindows.values()])   # array of discrete averages

        with open(meanImagePathArr, 'wb') as handle:
            pickle.dump(masksArr, handle)
        print(f"{timeHMS()}: Mean window is enabled. No mean image array found. Generating and saving new... Done")
                                                     

    else:
        print(f"{timeHMS()}: Mean window is enabled. Mean image array found. Importing data...")
        with open(meanImagePathArr, 'rb') as handle:
                masksArr = pickle.load(handle)
        print(f"{timeHMS()}: Mean window is enabled. Mean image array found. Importing data... Done!")

def whichMaskInterval(t,order):                                                                          # as frames go 0,1,..numImgs
    times = np.array(list(order))                                                                        # mean should be taken form the left
    sol = 0                                                                                              # img0:[0,N],img200:[i-a,i+b],...
    for time in times:                                                                                   # so img 199 should use img0 interval
        if time <= t:sol = time                                                                          # EZ sol just to interate and comare 
        else: break                                                                                      # and keep last one that satisfies
                                                                                                         # 
    return order[sol]      

if not useIntermediateData:

    blurMean = cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT)
    if not os.path.exists(binarizedArrPath):
        binarizedMaskArr = dataArchive - blurMean                                                               # substract mean image from stack -> float
        imgH,imgW = blurMean.shape
        thresh0 = 10
        binarizedMaskArr = np.where(binarizedMaskArr < thresh0, 0, 255).astype(np.uint8)                        # binarize stack
        binarizedMaskArr = np.uint8([cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) for img in binarizedMaskArr])    # delete white objects

        print(f"{timeHMS()}: Removing small and edge contours...")
        topFilter, bottomFilter, leftFilter, rightFilter, minArea    = 80, 40, 100, 100, 180
        for i in tqdm(range(binarizedMaskArr.shape[0])):
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
        print(f"{timeHMS()}: Binarized Array archive not found... Saving")
        np.savez_compressed(binarizedArrPath,binarizedMaskArr)
        print(f"{timeHMS()}: Binarized Array archive not found... Done")
    else:
        print(f"{timeHMS()}: Binarized Array archive located... Loading")
        binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
        print(f"{timeHMS()}: Binarized Array archive located... Done")

    #err             = cv2.morphologyEx(err.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))



    print(f"{timeHMS()}: First Pass: obtaining rough clusters using bounding rectangles...")
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
    for i in tqdm(range(binarizedMaskArr.shape[0])):
        # find all local contours
        #contours            = cv2.findContours(binarizedMaskArr[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours, hierarchy = cv2.findContours(binarizedMaskArr[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #cv2.RETR_EXTERNAL; cv2.RETR_LIST; cv2.RETR_TREE
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

    print(f"{timeHMS()}: First Pass: forming inter-frame relations for rough clusters...")
    def ID2S(arr, delimiter = '-'):
        return delimiter.join(list(map(str,sorted(arr))))
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
    cutoff  = 500
    cutoff = min(cutoff,binarizedMaskArr.shape[0])
    g0_clusterConnections_sub = {ID:vals for ID,vals in g0_clusterConnections.items() if ID <= cutoff}
    #g0_splitsMerges = {ID:[a for a in vals if len(a)>2] for ID,vals in g0_clusterConnections_sub.items()}
    #[g0_splitsMerges.pop(ID,None) for ID,vals in g0_splitsMerges.copy().items() if len(vals) == 0]
    #g0_splitsMerges2 = {ID:[[[a,g0_clusters[a][i]] for a,i in subvals] for subvals in vals] for ID, vals in g0_splitsMerges.items()}





    allIDs = sum([list(g0_bigBoundingRect2[t].keys()) for t in g0_bigBoundingRect2],[])

    # form a graph from all IDs and pairwise connections
    H = nx.Graph()
    #H = nx.DiGraph()
    H.add_nodes_from(allIDs)
    H.add_edges_from(g0_pairConnections)
    connected_components_unique = extract_graph_connected_components(H, sort_function = lambda x: (x[0], x[1]))

    if 1 == 1:
        storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
        with open(storeDir, 'wb') as handle:
            pickle.dump(
            [
                g0_bigBoundingRect2, g0_clusters2, g0_contours,g0_contours_hulls, g0_contours_centroids,
                g0_contours_areas, g0_pairConnections, H, connected_components_unique, g0_contours_children
            ], handle) 






print(f"\n{timeHMS()}: Begin loading intermediate data...")
storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
with open(storeDir, 'rb') as handle:
    [
        g0_bigBoundingRect2, g0_clusters2, g0_contours,g0_contours_hulls, g0_contours_centroids,
        g0_contours_areas, g0_pairConnections, H, connected_components_unique, g0_contours_children
    ] = pickle.load(handle)
print(f"\n{timeHMS()}: Begin loading intermediate data...Done!, test sub-case")


# ======================
if 1 == -1:
    segments2 = connected_components_unique
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
            #x,y,w,h = cv2.boundingRect(np.vstack([g0_contours[time][ID] for ID in subIDs]))
            x,y,w,h = g0_bigBoundingRect2[time][subCase]
            #x,y,w,h = g0_bigBoundingRect[time][ID]
            cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
            [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                startPos2 = g0_contours[time][subID][-30][0] 
                [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        

    for k,img in enumerate(imgs):
        folder = r"./post_tests/testImgs2/"
        fileName = f"{str(k).zfill(4)}.png"
        cv2.imwrite(os.path.join(folder,fileName) ,img)
        #cv2.imshow('a',img)

t_all_areas = []
for t_areas in g0_contours_areas.values():
    t_all_areas += t_areas.tolist()


g0_area_mean    = np.mean(t_all_areas)
g0_area_std     = np.std(t_all_areas)

if 1 == -1:
    # Plot the histogram
    plt.hist(t_all_areas, bins=70, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.title('Histogram of Animal Heights')      # LOL

    # Show the plot
   #plt.show()


a = 1

# analyze single strand
doX = 60#30#60#56#20#18#2#84#30#60#30#20#15#2#1#60#84
lessRough_all = connected_components_unique.copy()
test = connected_components_unique[doX]
# find times where multiple elements are connected (split/merge)




node_positions = getNodePos(test)
allNodes = list(H.nodes())
removeNodes = list(range(len(connected_components_unique)))
[allNodes.remove(x) for x in test]

for x in allNodes:
    H.remove_node(x)

H = H.to_directed()
t_H_edges_old = list(H.edges)
H.clear_edges()
t_H_edges_new = [sorted(list(t_edge), key = lambda x: x[0]) for t_edge in t_H_edges_old]# [((262, 4), (263, 4, 7))]
H.add_edges_from(t_H_edges_new)

#nx.draw(H, with_labels=True)
#plt.show()

segments2, skipped = graph_extract_paths(H, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

# Draw extracted segments with bold lines and different color.
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
    #cc_unique  = graphUniqueComponents(allKeys, combosSelf)                                       
    #g0_clusterConnections[t] = cc_unique

print(f"\n{timeHMS()}: Refining small BR overlap with c-c distance...")
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
print(f"\n{timeHMS()}: Refining small BR overlap with c-c distance... done")                     

a = 1
#bigBoundingRect     = [cv2.boundingRect(np.vstack([rect2contour(brectDict[k]) for k in comb])) for comb in cc_unique]

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
#G = nx.Graph()
#G.add_nodes_from(allIDs)
#G.add_edges_from(g0_pairConnections2)

G = nx.DiGraph()
G.add_nodes_from(allIDs)
G.add_edges_from(g0_pairConnections2)


set_custom_node_parameters(G, g0_contours_hulls, G.nodes(), None, calc_hull = 0) # pre define params and init owner to None 

for t_edge in g0_edges_merge_strong:                                          # assign previously acquired 2 bubble merge 
    G[t_edge[0]][t_edge[1]]['edge_merge_strong'] = True                       # edge of high of high likelihood

for t_edge in g0_edges_merge_weak:
    G[t_edge[0]][t_edge[1]]['edge_merge_strong'] = False

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
node_positions = getNodePos(allIDs)

f = lambda x : x[0]

segments2, skipped = graph_extract_paths(G, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

a = 1

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
#paths = {i:vals for i,vals in enumerate(segments2)}

for t,t_segment in enumerate(segments2): # assign owners params
    for t_node in t_segment:
        G.nodes[t_node]["owner"] = t

lr_allNodesSegm = sum(segments2,[])
lr_missingNodes = [node for node in allIDs if node not in lr_allNodesSegm] # !! may be same as &skipped !!
assert set(skipped)==set(lr_missingNodes), "set(skipped) is not same as set(lr_missingNodes)"



# ===============================================================================================
# ===============================================================================================
# === find POSSIBLE interval start-end connectedness: start-end exist within set time interval ==
# ===============================================================================================
# ===============================================================================================
# REMARK: it is expected that INTER segment space is limited, so no point searching paths from
# REMARK: one segment to each other in graph. instead inspect certain intervals of set length 
# REMARK: lr_maxDT. Caveat is that inter-space can include smaller segments.. its dealth with after

# get start and end of good segments
lr_start_end    = [[segm[0],segm[-1]] for segm in segments2]
lr_all_start    = np.array([a[0][0] for a in lr_start_end])
lr_all_end      = np.array([a[1][0] for a in lr_start_end])

# get nodes that begin not as a part of a segment = node with no prior neigbors
lr_nodes_other = []
lr_nodes_solo = []
for node in lr_missingNodes:
    #neighbors = list(G.neighbors(node))
    #oldNeigbors = [n for n in neighbors if n[0] < node[0]]
    #newNeigbors = [n for n in neighbors if n[0] > node[0]]
    oldNeigbors = list(G.predecessors(  node))
    newNeigbors = list(G.successors(    node))
    if len(oldNeigbors) == 0 and len(newNeigbors) == 0: lr_nodes_solo.append(node)
    elif len(oldNeigbors) == 0:                         lr_nodes_other.append(node)

# find which segements are separated by small number of time steps
#  might not be connected, but can check later
lr_maxDT = 20

# test non-segments
lr_DTPass_other = {}
for k,node in enumerate(lr_nodes_other):
    timeDiffs = lr_all_start - node[0]
    goodDTs = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
    if len(goodDTs) > 0:
        lr_DTPass_other[k] = goodDTs


# test segments. search for endpoint-to-startpoint DT

lr_DTPass_segm = {}
for k,endTime in enumerate(lr_all_end):
    timeDiffs = lr_all_start - endTime
    goodDTs = np.where((1 <= timeDiffs) & (timeDiffs <= lr_maxDT))[0]
    if len(goodDTs) > 0:
        lr_DTPass_segm[k] = goodDTs

# ===============================================================================================
# ===============================================================================================
# === find ACTUAL interval start-end connectedness: get all connected paths if there are any ==
# ===============================================================================================
# REMARK: refine previously acquired potential segment connectedness by searching paths between

# check connection between "time-localized" segments
# isolate all nodes on graph that are active in unresolved time between segment existance
# find all paths from one segment end to other start
lr_paths_segm   = {}
for startID,endIDs in lr_DTPass_segm.items():
    startNode = lr_start_end[startID][1]
    startTime = startNode[0]
    for endID in endIDs:
        endNode = lr_start_end[endID][0]
        endTime = lr_all_end[endID]
        #activeNodes = [node for node in allIDs if startTime <= node[0] <= endTime]
        activeNodes = [node for node, time_attr in G.nodes(data='time') if startTime <= time_attr <= endTime]
        subgraph = G.subgraph(activeNodes)

        try:
            shortest_path = list(nx.all_shortest_paths(subgraph, startNode, endNode))
            
        except nx.NetworkXNoPath:
            shortest_path = []
        
        if len(shortest_path)>0: lr_paths_segm[tuple([startID,endID])] = shortest_path

lr_paths_other  = {}
for startID,endIDs in lr_DTPass_other.items():
    startNode = lr_nodes_other[startID]
    startTime = startNode[0]
    for endID in endIDs:
        endNode = lr_start_end[endID][0]
        endTime = lr_all_end[endID]
        #activeNodes = [node for node in allIDs if startTime <= node[0] <= endTime]
        activeNodes = [node for node, time_attr in G.nodes(data='time') if startTime <= time_attr <= endTime]
        subgraph = G.subgraph(activeNodes)
        try:
            shortest_path = list(nx.all_shortest_paths(subgraph, startNode, endNode))
        except nx.NetworkXNoPath:
            shortest_path = []
        
        if len(shortest_path)>0: lr_paths_other[tuple([str(startID),str(endID)])] = shortest_path

lr_paths_segm2 = {a:[b[0][0],b[0][-1]] for a,b in lr_paths_segm.items()}
a = 1

# remember which segment indicies are active at each time step
lr_time_active_segments = {t:[] for t in lessRoughBRs}
for k,t_segment in enumerate(segments2):
    t_times = [a[0] for a in t_segment]
    for t in t_times:
        lr_time_active_segments[t].append(k)

# extract trusted segments.
# high length might indicate high trust

lr_segments_lengths = {}
for k,t_segment in enumerate(segments2):
    lr_segments_lengths[k]  = len(t_segment)


lr_trusted_segment_length = 7
lr_trusted_segments_by_length = []

for k,t_segment in enumerate(segments2):

    lr_segments_lengths[k]  = len(t_segment)

    if len(t_segment) >= lr_trusted_segment_length:
        lr_trusted_segments_by_length.append(k)

# small paths between segments might mean that longer, thus thrusted, segment is present

# store interval lentghs
lr_intervals_lengths = {}

for t_conn, intervals in lr_paths_segm.items():

    lr_intervals_lengths[t_conn] = len(intervals[0])

# extract possible good intervals

lr_trusted_max_interval_length_prio = 5
lr_trusted_max_interval_length      = 3

lr_trusted_interval_test_prio   = []
lr_trusted_interval_test        = []
for t_conn, t_interval_length in lr_intervals_lengths.items():

    if t_interval_length > lr_trusted_max_interval_length_prio: continue

    t_from,t_to = t_conn

    t_good_from = t_from in lr_trusted_segments_by_length
    t_good_to   = t_to in lr_trusted_segments_by_length

    if (t_good_from and t_good_to):
        lr_trusted_interval_test_prio.append(t_conn)
    elif t_interval_length <= lr_trusted_max_interval_length:
        lr_trusted_interval_test.append(t_conn)
  

# check interconnectedness of segments
G2 = nx.Graph()
G2.add_nodes_from(range(len(segments2))) # segment might not interact via edges, so need to add to graph explicitly
# Iterate over the dictionary and add edges with weights
for (node1, node2), weight in lr_intervals_lengths.items():
    G2.add_edge(node1, node2, weight=weight)

for g in G2.nodes():
      G2.nodes()[g]["t_start"] = segments2[g][0][0]
      G2.nodes()[g]["t_end"] = segments2[g][-1][0]

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
        t_time_start    = G2.nodes()[node]["t_start"]#segments2[node][0][0]
        t_time_end      = G2.nodes()[node]["t_end"]#segments2[node][-1][0]
        t_neighbors_prev = []
        t_neighbors_next = []
        for t_neighbor in t_neighbors:
            t_time_neighbor_start    = G2.nodes()[t_neighbor]["t_start" ]#segments2[t_neighbor][0][0]
            t_time_neighbor_end      = G2.nodes()[t_neighbor]["t_end"   ]#segments2[t_neighbor][-1][0]
            #if t_time_start < t_time_neighbor_start and t_time_end < t_time_neighbor_start:  # EDIT 24.08.23
            #    t_neighbors_next.append(t_neighbor)                                          # EDIT 24.08.23
            #elif t_time_start > t_time_neighbor_end and t_time_end > t_time_neighbor_end:    # EDIT 24.08.23
            #    t_neighbors_prev.append(t_neighbor)                                          # EDIT 24.08.23
            if t_time_end < t_time_neighbor_start:
                t_neighbors_next.append(t_neighbor)
            elif t_time_neighbor_end < t_time_start:
                t_neighbors_prev.append(t_neighbor)
        # check if neighbors are not lost, or path generation is incorrect, like looping back in time
        assert len(t_neighbors) == len(t_neighbors_prev) + len(t_neighbors_next), "missing neighbors, time position assumption is wrong"
    
        t_neighbors_weights_prev = {}
        t_neighbors_weights_next = {}
    
        for t_neighbor in t_neighbors_prev: # back weights are negative
            t_neighbors_weights_prev[t_neighbor] = -1*G2[node][t_neighbor]['weight']
        for t_neighbor in t_neighbors_next:
            t_neighbors_weights_next[t_neighbor] = G2[node][t_neighbor]['weight']
        t_neighbors_time_prev = {tID:segments2[tID][-1][0]  for tID in t_neighbors_weights_prev}
        t_neighbors_time_next = {tID:segments2[tID][0][0]   for tID in t_neighbors_weights_next}
    
        if len(t_neighbors_weights_prev)>0:
            # neg weights, so max, get time of nearset branch t0. get all connections within [t0 - 2, t0] in case of split
            t_val_min = max(t_neighbors_weights_prev.values())
            t_key_min_main = max(t_neighbors_weights_prev, key = t_neighbors_weights_prev.get)
            t_key_main_ref_time = t_neighbors_time_prev[t_key_min_main]
            t_sol = [key for key,t in t_neighbors_time_prev.items() if t_key_main_ref_time - 1 <= t <=  t_key_main_ref_time] #not sure why-+2, changed to -1 
            #t_sol = [key for key, value in t_neighbors_weights_prev.items() if t_val_min - 1 <= value <=  t_val_min]
            for t_node in t_sol:t_neighbor_sol_all_prev[node][t_node] = t_neighbors_weights_prev[t_node]
        
        if len(t_neighbors_weights_next)>0:
            t_val_min = min(t_neighbors_weights_next.values())
            t_key_min_main = min(t_neighbors_weights_next, key = t_neighbors_weights_next.get)
            t_key_main_ref_time = t_neighbors_time_next[t_key_min_main]
            t_sol = [key for key,t in t_neighbors_time_next.items() if t_key_main_ref_time <= t <=  t_key_main_ref_time + 1] # not sure why +2, changed to +1 
            #t_sol = [key for key, value in t_neighbors_weights_next.items() if t_val_min +1 >= value >= t_val_min]          # cause i was overextending minimal segment lenght of 2
            for t_node in t_sol: t_neighbor_sol_all_next[node][t_node] = t_neighbors_weights_next[t_node]
        a = 1

# ===============================================================================================
# ===============================================================================================
# === wipe inter-segement edges from G2 graph and replot only nearest connections ===
# ===============================================================================================
# ===============================================================================================
# prepare plot for segment interconnectedness
G2.remove_edges_from(list(G2.edges()))
for node, t_conn in t_neighbor_sol_all_prev.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2)
for node, t_conn in t_neighbor_sol_all_next.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2)
#for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ===============================================================================================
# === calculate all paths between nearest segments ===
# ===============================================================================================
# ===============================================================================================
# >>>>>>> MAYBE FOR BIG INTER-SEGMENT SAVE ONLY SOLO PATH, THIS WAY YOU CAN KNOW IT CANT BE ANYTHING ELSE <<<<<
#segment_conn_end_start_points((11,20), nodes = 1)
aa = G.is_directed();print(f'G s directed:{aa}')

if 1 == 1:
    lr_close_segments_simple_paths = {}
    lr_close_segments_simple_paths_inter = {}
    tIDs = [tID for tID,t_dic in t_neighbor_sol_all_next.items() if len(t_dic) >0]
    for t_from in tIDs:
        for t_to in t_neighbor_sol_all_next[t_from]:
            t_from_node_last   = segments2[t_from][-1]
            t_to_node_first      = segments2[t_to][0]

            # for directed graph i can disable cutoff, since paths can progress only forwards
            t_from_node_last_time   = t_from_node_last[0]
            t_to_node_first_time    = t_to_node_first[0]
            t_from_to_max_time_steps= t_to_node_first_time - t_from_node_last_time + 1
            hasPath = nx.has_path(G, source=t_from_node_last, target=t_to_node_first)
            # soo.. all_simple_paths method goes into infinite (?) loop when graph is not limited, shortest_simple_paths does not.
            t_start, t_end = t_from_node_last[0], t_to_node_first[0]
            activeNodes = [t_node for t_node, t_time in G.nodes(data='time') if t_start <= t_time <= t_end]
            g_limited = G.subgraph(activeNodes)
            #t_from_to_paths_simple  = list(nx.shortest_simple_paths(G, t_from_node_last, t_to_node_first)) # , cutoff = t_from_to_max_time_steps

            #if t_to == -1:
            #    t_short = nx.shortest_path(g_limited, source=t_from_node_last, target=t_to_node_first, weight=None, method='dijkstra')
            #    t_short_edges = []
            #    for t_node in t_short[1:-1]:
            #        t_short_edges += g_limited.predecessors(t_node)
            #        t_short_edges += g_limited.successors(t_node)
            #    t_short_edges2 = [t_node for t_node in t_short_edges if t_node not in t_short]
            #    a = 1
            t_from_to_paths_simple  = list(nx.all_simple_paths(g_limited, t_from_node_last, t_to_node_first, cutoff = t_from_to_max_time_steps)) 
            t_from_to_paths_nodes_all       = sorted(set(sum(t_from_to_paths_simple,[])),key=lambda x: x[0])
            t_from_to_paths_nodes_all_inter = [t_node for t_node in t_from_to_paths_nodes_all if t_node not in [t_from_node_last,t_to_node_first]]

            lr_close_segments_simple_paths[         tuple([t_from,t_to])]  = t_from_to_paths_simple
            lr_close_segments_simple_paths_inter[   tuple([t_from,t_to])]  = t_from_to_paths_nodes_all_inter

    node_positions2 = {}
    t_segments_times = {}
    labels = {}
    for t,segment in enumerate(segments2):
        t_times = [a[0] for a in segment]
        t_segments_times[t] = int(np.mean(t_times))
        labels[t] = f'{t}_{segment[0]}'
    node_positions2 = getNodePos2(t_segments_times, S = 1)

    
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

# ===============================================================================================
# ===============================================================================================
# === extract segment-segment connections that are connected only together (one-to-one; 121) ====
# ===============================================================================================
# REMARK: as discussed before directed connections are part of merges, but in unidirectional
# REMARK: connectios exists an associated (main) connection that makes other a directional
# REMARK: it was a result of performing nearest neighbor refinement.
# REMARK: so non split/merge connection is refinement of unidir connection list with info of merge/split

t_conn_121 = [t_conn for t_conn in lr_connections_unidirectional if t_conn not in t_merge_split_culprit_edges_all]

# ===============================================================================================
# separate 121 paths with zero inter length
# ===============================================================================================
# REMARK: these cases occur when segment terminates into split with or merge with a node, not a segment
# REMARK: and 121 separation only concerns itself with segment-segment connecivity
# REMARK: this case might be inside merge/split merge.
t_conn_121_zero_path = []
for t_node in t_conn_121:
    if len(lr_close_segments_simple_paths_inter[t_node]) == 0:
        t_conn_121_zero_path.append(t_node)

# EDIT: 20.09.23 keep zero path in t_conn_121
# >>>>>>>>>>>>>>>> WHAT I SHOULD DO <<<<<<<<<<<<<<<<<<<
# 1) E.Z consume small branch. its most likely true sol-n, but algorithm might not be ready for composite nodes
# 2) or i strip edge status from all small branches, thus its no longer zero path. so you can treat it as usual.

#t_conn_121 = [t_node for t_node in t_conn_121 if t_node not in t_conn_121_zero_path]

# ===============================================================================================
# reconstruct 121 zero path neighbors
# ===============================================================================================
# REMARK: zero path implies that solo nodes are connected to prev segment end or next segment start
t_conn_121_zero_path_nodes = {t_conn:[] for t_conn in t_conn_121_zero_path}
t_conn_121_zp_contour_combs= {t_conn:{} for t_conn in t_conn_121_zero_path}
for t_conn in t_conn_121_zero_path:
    t_from, t_to    = t_conn
    t_from_node_end = segments2[t_from  ][-1]
    t_to_node_start = segments2[t_to    ][0]
    #t_from_neigbors_next    = extractNeighborsNext(     G, t_from_node_end,  lambda x: x[0])
    #t_to_neigbors_prev      = extractNeighborsPrevious( G, t_to_node_start,  lambda x: x[0])
    t_from_neigbors_next    = list(G.successors(t_from_node_end))
    t_to_neigbors_prev      = list(G.predecessors(t_to_node_start))
    t_conn_121_zero_path_nodes[t_conn] = sorted(set(t_from_neigbors_next+t_to_neigbors_prev),key=lambda x: (x[0], x[1]))
    t_times = sorted(set([t_node[0] for t_node in t_conn_121_zero_path_nodes[t_conn]]))
    t_conn_121_zp_contour_combs[t_conn] = {t_time:[] for t_time in t_times}
    
    for t_time,*t_subIDs in t_conn_121_zero_path_nodes[t_conn]:
        t_conn_121_zp_contour_combs[t_conn][t_time] += t_subIDs

    for t_time in t_conn_121_zp_contour_combs[t_conn]:
         t_conn_121_zp_contour_combs[t_conn][t_time] = sorted(t_conn_121_zp_contour_combs[t_conn][t_time])
a = 1

# EDIT: 20.09.23
# consume zero path
# combine contested nodes
# join segments into one.
lr_zp_redirect = {tID: tID for tID in range(len(segments2))}
for t_conn,t_dict in t_conn_121_zp_contour_combs.items():
    t_from, t_to        = t_conn                                                    # no inheritance is yet needed
    t_times             = [t for t,t_subIDs in t_dict.items() if len(t_subIDs) > 1]
    assert len(t_times) == 1, "unexpected case. multiple times, need to insert nodes in correct order"
    t_nodes_composite   = [tuple([t] + t_dict[t]) for t in t_times]                 # join into one node (composit)
    t_nodes_solo        = [(t,t_subID) for t in t_times for t_subID in t_dict[t]]   # get solo contour nodes

    t_edges             = []
    t_nodes_prev        =   [t_node for t_node in segments2[t_from] if t_node not in t_nodes_solo] 
    t_edges             +=  [(t_nodes_prev[-1]      , t_nodes_composite[0]) ]
    segments2[t_from]   =   t_nodes_prev
    segments2[t_from]   +=  t_nodes_composite
    t_nodes_next        =   [t_node for t_node in segments2[t_to] if t_node not in t_nodes_solo]
    t_edges             +=  [(t_nodes_composite[0]  , t_nodes_next[0])      ]
    segments2[t_from]   +=  t_nodes_next
    
    # modify graph: remove solo IDs, add new composite nodes. add parameters to new nodes
    G.remove_nodes_from(t_nodes_solo) 
    G.add_edges_from(t_edges)

    set_custom_node_parameters(G, g0_contours, t_nodes_composite, t_from, calc_hull = 1)

    for t_node in t_nodes_next: # t_nodes_prev + 
            G.nodes[t_node]["owner"] =  t_from

    segments2[t_to]     = []

    # modify segment view graph.
    t_successors   = extractNeighborsNext(G2, t_to, lambda x: G2.nodes[x]["t_start"])
    
    t_edges = [(t_from,t_succ) for t_succ in t_successors]
    G2.remove_node(t_to)
    G2.add_edges_from(t_edges)
    G2.nodes()[t_from]["t_end"] = segments2[t_from][-1][0]

    # create dict to re-connect previously obtained edges.
    lr_zp_redirect[t_to] = t_from
    
lr_conn_edges_merges    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_merges   )
lr_conn_edges_splits    = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits   )
lr_conn_edges_splits_merges_mixed = lr_reindex_masters(lr_zp_redirect, lr_conn_edges_splits_merges_mixed   )
t_conn_121              = lr_reindex_masters(lr_zp_redirect, t_conn_121             ,remove_solo_ID = 1)

# lr_close_segments_simple_paths not modified.
a = 1
            
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
    lr_big_121s_chains = extract_graph_connected_components(G_seg_view_1.to_undirected(), lambda x: x)

    print(f'Working on joining all continious 121s together: {lr_big_121s_chains}')
    # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # big 121s terminate at zero neighbors, real or pseudo merges/splits 
    # both cannot easily tell te difference between real and pseudo. have search for recombination
    # pseudo events means start/end of a big 121 contain only part of bubble and cannot
    # be used to determine history. For the real m/s start/end segments can be used. 
    # how to tell real and pseudo apart? 
    # since pseudo m/s is one bubble:
    # 1) split branches have limited and rather short length before pseudo merge.
    # 2) ! area is recovered after pseudo branch

    # -----------------------------------------------------------------------------------------------
    # PSEUDO MERGE/SPLIT ON EDGES: DETECT BIG 121S WITH M/S
    # -----------------------------------------------------------------------------------------------
    # check left side = part of split
    
    if 1 == 1:
        t_segment_ID_left   = [t_subIDs[0] for t_subIDs in lr_big_121s_chains]
        t_segment_ID_right  = [t_subIDs[-1] for t_subIDs in lr_big_121s_chains]
        t_temp_merge    = defaultdict(int)
        t_temp_split    = defaultdict(int)
        t_temp_in_merge = defaultdict(list)
        t_temp_in_split = defaultdict(list)
        # find if left is part of split, and with whom
        for t, t_ID in enumerate(t_segment_ID_left):
            for t_from,t_to in lr_conn_edges_splits:
                if t_to == t_ID:                    # if t_ID is branch of split from t_from
                    t_temp_split[t_ID] = t_from     # save who was mater of left segment-> t_ID:t_from
                    t_temp_in_split[t] = lr_big_121s_chains[t]     # index in big 121s: connected components. t_ID in cc.

        # find if right is part of merge, and with whom
        for t, t_ID in enumerate(t_segment_ID_right): # order in big 121s, right segment ID.
            for t_from,t_to in lr_conn_edges_merges:
                if t_from == t_ID: 
                    t_temp_merge[t_ID] = t_to
                    t_temp_in_merge[t] = lr_big_121s_chains[t]

        # extract big 121s that are in no need of real/pseudo m/s analysis.
        lr_big121_events_none = defaultdict(list) # big 121s not terminated by splits or merges. ready to resolve.
        # find if elements of big 121s are without m/s or bounded on one or both sides by m/s.
        set1 = set(t_temp_in_merge)
        set2 = set(t_temp_in_split)

        t_only_merge = set1 - set2
        t_only_split = set2 - set1

        t_merge_split = set1.intersection(set2)
        t_merge_split_no = set(range(len(lr_big_121s_chains))) - (set1.union(set2))
        #assert len(t_merge_split) == 0, "segments bounded on both sides by merge/split. not encountered before, but ez to fix"

        lr_events = {'merge':defaultdict(list), 'split':defaultdict(list)}
    # -----------------------------------------------------------------------------------------------
    # PSEUDO MERGE/SPLIT ON EDGES: DETERMINE IF M/S IS FAKE OR REAL
    # -----------------------------------------------------------------------------------------------

    # test if edge segment is part of real of fake m/s.
    # strip it from big 121s and compare area of prev segment and result of event
    # if area changed drastically then its real m/s, otherwise its same bubble with fake m/s
    # NOTE: currently i take a single end-start nodes. might take more if i test for if they are in m/s (change to segment of history avg later)
    print('Determining if end points of big 121s are pseudo events: ')
    if 1 == 1:
        t_big121s_merge_conn_fake = []
        t_big121s_split_conn_fake = []

        t_merge_split_fake = defaultdict(list) # 0 -> is part of fake split, -1 -> is part of fake merge
        t_only_merge_split_fake = []
        t_max_pre_merge_segment_length = 12 # skip if branch is too long. most likely a real branch

        t_only_merge_fake = []
        for t in t_only_merge.union(t_merge_split): #t_only_merge + t_merge_split
            t_big121_subIDs = lr_big_121s_chains[t]
            #if t not in t_merge_split:
            t_from      = t_big121_subIDs[-1]   # possible fake merge branch (segment ID).
            t_from_from = t_big121_subIDs[-2]   # segment ID prior to that, most likely OG bubble.
            t_from_to   = t_temp_merge[t_from]  # seg ID of merge.

            if len(segments2[t_from]) >= t_max_pre_merge_segment_length: continue

            t_from_to_node      = segments2[t_from_to   ][0 ]                   # take end-start nodes
            t_from_from_node    = segments2[t_from_from ][-1]

            t_from_to_node_area     = G.nodes[t_from_to_node    ]["area"]       # check their area
            t_from_from_node_area   = G.nodes[t_from_from_node  ]["area"] 
            if np.abs(t_from_to_node_area - t_from_from_node_area) / t_from_to_node_area < 0.35:
                if t not in t_merge_split:
                    t_only_merge_fake += [t]
                    t_big121s_merge_conn_fake.append((t_from,t_from_to))
                else:       # undetermined, decide after
                    t_merge_split_fake[t] += [-1]

        t_only_split_fake = []
        for t in t_only_split.union(t_merge_split):
            t_big121_subIDs = lr_big_121s_chains[t]
            t_to            = t_big121_subIDs[0]   # possible fake merge branch (segment ID).
            t_to_to         = t_big121_subIDs[1]   # segment ID prior to that, most likely OG bubble.
            t_from_to       = t_temp_split[t_to]  # seg ID of merge.

            if len(segments2[t_to]) >= t_max_pre_merge_segment_length: continue

            t_from_to_node  = segments2[t_from_to   ][-1]                   # take end-start nodes
            t_to_to_node    = segments2[t_to_to     ][0]

            t_from_to_node_area = G.nodes[t_from_to_node]["area"]       # check their area
            t_to_to_node_area   = G.nodes[t_to_to_node  ]["area"] 
            if np.abs(t_from_to_node_area - t_to_to_node_area) / t_from_to_node_area < 0.35:
                if t not in t_merge_split:
                    t_only_split_fake += [t]
                    t_big121s_split_conn_fake.append((t_from_to,t_to))
                else:       # undetermined, decide after
                    t_merge_split_fake[t] += [0]

        for t,t_states in t_merge_split_fake.items():
            if len(t_states) == 2:                  # both ends are in fake event
                t_only_merge_split_fake += [t]
            elif 0 in t_states:                     # only split is fake
                t_to            = lr_big_121s_chains[t][0]   
                t_from_to       = t_temp_split[t_to] 
                t_only_split_fake += [t]
                t_big121s_split_conn_fake.append((t_from_to,t_to))
            else:                                  # only merge is fake
                t_from      = lr_big_121s_chains[t][-1]  
                t_from_to   = t_temp_merge[t_from]
                t_only_merge_fake += [t]
                t_big121s_merge_conn_fake.append((t_from,t_from_to))

        assert len(t_only_merge_split_fake) == 0, "case yet unexplored, written without test"        


    print(f'Fake merges: {t_big121s_merge_conn_fake}, Fake splits: {t_big121s_split_conn_fake}')
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
    print(f'Trimmed big 121s: {lr_big121s_conn_121}/nInterpolating whole big 121s...')

    # -----------------------------------------------------------------------------------------------
    # BIG 121S INTERPOLATE COMBINED TRAJECTORIES, GIVEN FULL INFO REDISTRIBUTE FOR DATA FOR EACH CONN
    # -----------------------------------------------------------------------------------------------
    # CALCULATE interpolation of holes in data for big 121s
    if 1 == 1: # for_graph_plots(G)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        lr_big121s_interpolation        = defaultdict(dict)
        lr_big121s_interpolation_big    = defaultdict(dict) 

        for t,t_subIDs in enumerate(t_big121s_edited):
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
                    t_start,t_end  = (segments2[t_from][-1][0], segments2[t_to][0][0])
                    t_times = list(np.arange(t_start + 1,t_end,1))

                    t_conns_times_dict[t_edge] = t_times
                    
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

                    t_centroids = [t_centroid for t_time,t_centroid in t_interpolation_centroids_1.items() if t_time in t_times_relevant]
                    t_areas     = [t_area for t_time,t_area in t_interpolation_areas_1.items() if t_time in t_times_relevant]

                    lr_big121s_interpolation[t_conn]['centroids'] = np.array(t_centroids)
                    lr_big121s_interpolation[t_conn]['areas'    ] = t_areas
                    lr_big121s_interpolation[t_conn]['times'    ] = t_times_relevant
                # save whole big 121
                lr_big121s_interpolation_big[tuple(t_subIDs)]['centroids'] = t_interpolation_centroids_1
                lr_big121s_interpolation_big[tuple(t_subIDs)]['areas'    ] = t_interpolation_areas_1

    


    # ===============================================================================================
    # ====  EXTRACT POSSIBLE CONTOUR ELEMENTS FROM PATHS ====
    # ===============================================================================================
    # REMARK: gather all inter-segment nodes and grab slices w.r.t time
    # REMARK: so result is {time_01:[ID1,ID2],time_02:[ID4,ID5],...}
    # REMARK: i expect most of the time solution to be all elements in cluster. except real merges
    print('Computing contour elements for space between segments')
    if 1 == 1:

        lr_big121s_perms_pre = {}

        for t_conn in lr_big121s_conn_121:  # did not modify it after removing/combining zero path conns.
            t_traj = lr_close_segments_simple_paths[t_conn][0]
            t_nodes = {t_time:[] for t_time in [t_node[0] for t_node in t_traj]}
            for t_traj in lr_close_segments_simple_paths[t_conn]:
                for t_time,*t_subIDs in t_traj:
                    t_nodes[t_time] += t_subIDs
            for t_time in t_nodes:
                t_nodes[t_time] = sorted(list(set(t_nodes[t_time])))
            lr_big121s_perms_pre[t_conn] = t_nodes
    
    # ===============================================================================================
    # ====  CONSTRUCT PERMUTATIONS FROM CLUSTER ELEMENT CHOICES ====
    # ===============================================================================================
    # REMARK: grab different combinations of elements in clusters
    print('Computing contour element permutations for each time step...')
    if 1 == 1:
        lr_big121s_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_big121s_perms_pre.items()}
        for t_conn, t_times_contours in lr_big121s_perms_pre.items():
            for t_time,t_contours in t_times_contours.items():
                t_perms = combs_different_lengths(t_contours)
                lr_big121s_perms[t_conn][t_time] = t_perms



    # ===============================================================================================
    # ====  PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ====
    # ===============================================================================================
    # REMARK: these will be reused alot in next steps, store them to avoid need of recalculation
    print('Calculating parameters for possible contour combinations...')
    if 1 == 1:    
        lr_big121s_perms_areas      = AutoCreateDict()# lr_init_perm_precomputed(lr_big121s_perms,0)
        lr_big121s_perms_centroids  = AutoCreateDict()# lr_init_perm_precomputed(lr_big121s_perms,[0,0])
        lr_big121s_perms_mom_z      = AutoCreateDict()# lr_init_perm_precomputed(lr_big121s_perms,0)

        for t_conn, t_times_perms in lr_big121s_perms.items():
            for t_time,t_perms in t_times_perms.items():
                for t_perm in t_perms:
                    t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
                    t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

                    lr_big121s_perms_areas[     t_conn][t_time][t_perm] = t_area
                    lr_big121s_perms_centroids[ t_conn][t_time][t_perm] = t_centroid
                    lr_big121s_perms_mom_z[     t_conn][t_time][t_perm] = t_mom_z

    

        #drawH(G, paths, node_positions)
    # ===============================================================================================
    # ====  CONSTUCT DIFFERENT PATHS FROM UNIQUE CHOICES ====
    # ===============================================================================================
    # REMARK: try different evolutions of inter-segment paths
    print('Generating possible bubble evolution paths from prev combinations...')
    if 1 == 1:
        lr_big121s_perms_cases = {t_conn:[] for t_conn in lr_big121s_perms_pre}
        lr_big121s_perms_times = {t_conn:[] for t_conn in lr_big121s_perms_pre}
        for t_conn, t_times_perms in lr_big121s_perms.items():
    
            t_values = list(t_times_perms.values())
            t_times = list(t_times_perms.keys())

            sequences = list(itertools.product(*t_values))

            lr_big121s_perms_cases[t_conn] = sequences
            lr_big121s_perms_times[t_conn] = t_times

    # ===============================================================================================
    # ==== EVALUATE HULL CENTROIDS AND AREAS FOR EACH EVOLUTION, FIND CASES WITH LEAST DEVIATIONS====
    # ===============================================================================================
    # REMARK: evolutions with least path length and are changes should be right ones
    print('Determining evolutions thats are closest to interpolated missing data...')
    if 1 == 1:
        t_temp_centroids = {t_conn:t_dict['centroids'] for t_conn,t_dict in lr_big121s_interpolation.items()}
        t_args = [lr_big121s_perms_cases,t_temp_centroids,lr_big121s_perms_times,
                lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

        t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)

        t_weights   = [1,1.5,0,1]
        t_sols      = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
        lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_weights, t_sols, lr_big121s_perms_cases )



    # for_graph_plots(G, segs = segments2)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #C0 = nx.Graph()
    #C0.add_edges_from(list(lr_big121s_perms_cases.keys()))
    
    #lr_C0_condensed_connections = extract_graph_connected_components(C0, lambda x: x)

    ## lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
    #t_condensed_connections_all_nodes = sorted(sum(lr_C0_condensed_connections,[])) # neext next
    #lr_C0_condensed_connections_relations = lr_zp_redirect.copy() #t_condensed_connections_all_nodes
    #for t_subIDs in lr_C0_condensed_connections:
    #    for t_subID in t_subIDs:
    #        lr_C0_condensed_connections_relations[t_subID] = min(t_subIDs)

    lr_C0_condensed_connections_relations = lr_zp_redirect.copy()
    print('Saving results for restored parts of big 121s')
    t_last_seg_ID_in_big_121s   = [t[-1]    for t in t_big121s_edited if t is not None]
    t_big121s_edited_clean      = [t        for t in t_big121s_edited if t is not None]

    t_segments_new = copy.deepcopy(segments2)
    for t_conn in lr_big121s_perms_cases:
        t_from, t_to = t_conn
        t_from_new = lr_C0_condensed_connections_relations[t_from]
        t_to_new = lr_C0_condensed_connections_relations[t_to]
        print(f'edge :({t_from},{t_to}) or = {t_segments_new[t_from_new][-1]}->{t_segments_new[t_to_new][0]}')   
        save_connections_two_ways(t_segments_new, lr_big121s_perms_pre[t_conn], t_from,  t_to, G, G2, lr_C0_condensed_connections_relations, g0_contours)
            
                       

            
    # at this time some segments got condenced into big 121s. right connection might be changed.
    lr_time_active_segments = {t:[] for t in lr_time_active_segments}
    for k,t_segment in enumerate(t_segments_new):
        t_times = [a[0] for a in t_segment]
        for t in t_times:
            lr_time_active_segments[t].append(k)

    #for_graph_plots(G, segs = t_segments_new) 
    # -----------------------------------------------------------------------------------------------
    # prep permutations for merges/splits fake and real
    # -----------------------------------------------------------------------------------------------
    print('Working on real and fake events (merges/splits):/nPreparing data...')
    if 1 == 1: # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # -----------------------------------------------------------------------------------------------
        # ------ PREPARE DATA INTO USABLE FORM ----
        # -----------------------------------------------------------------------------------------------
        t_merge_fake_to_ID = unique_sort_list([t_conn[1] for t_conn in t_big121s_merge_conn_fake])                                          # grab master node e.g [10]
        t_split_fake_from_ID = unique_sort_list([t_conn[0] for t_conn in t_big121s_split_conn_fake])                                        # e.g [15]

        t_merge_real_to_ID  = [t_ID for t_ID in set([t_conn[1] for t_conn in lr_conn_edges_merges]) if t_ID not in t_merge_fake_to_ID]      # e.g [13,23]
        t_split_real_from_ID  = [t_ID for t_ID in set([t_conn[0] for t_conn in lr_conn_edges_splits]) if t_ID not in t_split_fake_from_ID]    # e.g []

        t_merge_fake_conns  = [t_conn for t_conn in lr_conn_edges_merges if t_conn[1] in t_merge_fake_to_ID]    # grab all edges with that node. e.g [(8, 10), (9, 10)]
        t_split_fake_conns  = [t_conn for t_conn in lr_conn_edges_splits if t_conn[0] in t_split_fake_from_ID]

        t_merge_real_conns  = [t_conn for t_conn in lr_conn_edges_merges if t_conn not in t_merge_fake_conns]   # extract real from rest of conns. e.g [(15, 16), (15, 17)]
        t_split_real_conns  = [t_conn for t_conn in lr_conn_edges_splits if t_conn not in t_split_fake_conns]
    
        t_merge_fake_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_merges if t_conn[1] == t_ID] for t_ID in t_merge_fake_to_ID}       # e.g {10:[(8, 10), (9, 10)]}
        t_split_fake_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_splits if t_conn[0] == t_ID] for t_ID in t_split_fake_from_ID}     # e.g {15:[(15, 16), (15, 17)]} 

        t_merge_real_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_merges if t_conn[1] == t_ID] for t_ID in t_merge_real_to_ID}
        t_split_real_dict   = {t_ID:[t_conn for t_conn in  lr_conn_edges_splits if t_conn[0] == t_ID] for t_ID in t_split_real_from_ID}

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
        t_event_start_end_times = {'merge':{},'split':{},'mixed':{},'mixed2':{}}

        func_prev_neighb = lambda x: G2.nodes[x]["t_end"]
        func_next_neighb = lambda x: G2.nodes[x]["t_start"]

        # ------------------------------------- REAL MERGE/SPLIT -------------------------------------
        t_state = 'merge'
        for t_to in t_merge_real_to_ID: #t_ID is to merge ID. ID is not changed
            t_predecessors   = extractNeighborsPrevious(G2, t_to, func_prev_neighb)
            t_predecessors = [lr_C0_condensed_connections_relations[t] for t in t_predecessors] # update if inherited
            t_predecessors_times_end = {t:G2.nodes[t]["t_end"] for t in t_predecessors}
            t_t_to_start = G2.nodes[t_to]["t_start"]
            t_times         = {t_ID: np.arange(t_t_from_end, t_t_to_start + 1, 1)
                               for t_ID,t_t_from_end in t_predecessors_times_end.items()}

            t_event_start_end_times[t_state][t_to] = {'t_start':        t_predecessors_times_end,
                                                      'branches':       t_predecessors,
                                                      't_end':          t_t_to_start,
                                                      't_times':        t_times,
                                                      't_perms':        {},
                                                      't_subIDs':       {},
                                                      't_active_IDs':   defaultdict(list)}

            # pre-isolate graph segment where all sub-events take place
            t_t_from_min = min(t_predecessors_times_end.values())    # take earliest branch time
            t_nodes_keep    = [node for node in G.nodes() if t_t_from_min <= node[0] <= t_t_to_start] # < excluded edges
            t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
            # working with each branch separately
            for t_from, t_times_all in t_times.items():
                t_segments_keep = [t_from, t_to]      # new inherited ID
                t_start = t_predecessors_times_end[t_from]  # storage uses old ID
                
                if t_t_to_start - t_start > 1:
                    t_sol = graph_sub_isolate_connected_components(t_subgraph, t_start, t_t_to_start, lr_time_active_segments,
                                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from][-1]) 
                    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                    t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
                else:
                    t_perms = {}
                    t_values = None
                t_event_start_end_times[t_state][t_to]['t_perms'        ][t_from] = t_values
                t_event_start_end_times[t_state][t_to]['t_subIDs'       ][t_from] = t_perms
                for t_time in t_times_all[1:-1]: # remove start-end points
                    t_event_start_end_times[t_state][t_to]['t_active_IDs'   ][t_time].append(t_from)

        t_state = 'split'
        for t_from in t_split_real_from_ID:
            assert 1 == 1, "never ran t_split_real_from_ID because of lack of data."
            t_from_new = lr_C0_condensed_connections_relations[t_from]                  # split segment may be inherited.
            t_from_successors   = extractNeighborsNext(G2, t_from_new, func_next_neighb)     # branches are not
            t_successors_times = {t_to:G2.nodes[t_to]["t_start"] for t_to in t_from_successors}
            t_t_start = G2.nodes[t_from_new]["t_end"]
            t_times         = {t:np.arange(t_t_start, t_t_end + 1, 1) for t,t_t_end in t_successors_times.items()}
            t_event_start_end_times[t_state][t_from_new] = {'t_start' :       t_t_start,
                                                      'branches':       t_from_successors,
                                                      't_end'   :       t_successors_times,
                                                      't_times' :       t_times,
                                                      't_perms':        {},
                                                      't_subIDs':       {},
                                                      't_active_IDs':   defaultdict(list)}

            # pre-isolate graph segment where all sub-events take place
            t_t_to_max = max(t_successors_times.values())
            t_nodes_keep    = [node for node in G.nodes() if t_t_start <= node[0] <= t_t_to_max]
            t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
            # working with each branch separately
            for t_to, t_times_all in t_times.items():
                t_segments_keep = [t_from_new, t_to] 
                t_t_to_end = t_successors_times[t_to]
                if t_t_to_end - t_t_start > 1:
                    t_sol = graph_sub_isolate_connected_components(t_subgraph, t_t_start, t_t_to_end, lr_time_active_segments,
                                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from_new][-1])
                    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                    t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
                    #t_values = [sum([list(itertools.combinations(t_subIDs, r)) for r in range(1,len(t_subIDs)+1)],[]) for t_subIDs in t_perms.values()]
                else: 
                    t_perms = {}
                    t_values = None
                t_event_start_end_times[t_state][t_from_new]['t_perms'      ][t_to] = t_values
                t_event_start_end_times[t_state][t_from_new]['t_subIDs'     ][t_to] = t_perms
                for t_time in t_times_all[1:-1]:
                    t_event_start_end_times[t_state][t_from_new]['t_active_IDs' ][t_time].append(t_from_new)
        print(f'Real merges({t_merge_real_to_ID})/splits({t_split_real_from_ID})... Done')

        # ------------------------------------- FAKE MERGE/SPLIT -------------------------------------
        t_state = 'merge'
        for t_to in t_merge_fake_to_ID:
            
            t_to_predecessors   = extractNeighborsPrevious(G2, t_to, func_prev_neighb)  # should be updated prior
            t_to_pre_predecessors = []
            for t_to_pre in t_to_predecessors:
                t_to_pre_predecessors += extractNeighborsPrevious(G2, t_to_pre, func_prev_neighb)
            # if fake branches should terminate and only 1 OG should be left. check if this is the case
            assert len(t_to_pre_predecessors) == 1, "fake branch/es consists of multiple segments, not yet encountered"
            t_to_pre_pre_ID = t_to_pre_predecessors[0]
            t_t_start         = G2.nodes[t_to_pre_pre_ID]["t_end"]
            t_t_end           = G2.nodes[t_to]["t_start"]
            t_times         = {(t_to_pre_pre_ID, t_to):np.arange(t_t_start, t_t_end + 1, 1)}
            t_event_start_end_times[t_state][t_to] = {'t_start':        t_t_start,
                                                      'pre_predecessor':t_to_pre_pre_ID,
                                                      'branches':       t_to_predecessors,
                                                      't_end':          t_t_end,
                                                      't_times' :       t_times}


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

            t_segments_keep = [t_from_new] + t_from_successors + [t_ID_post_succ]
            t_sol = graph_sub_isolate_connected_components(G, t_start, t_end, lr_time_active_segments,
                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from_new][-1])
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

        # ------------------------------------- mixed cases. for path extension -------------------------------------
        # for mixed i include target nodes into permutations. then i can check if they are a solution (not merge extension, but 121)
        t_state = 'mixed'
        for t_from,t_successors in lr_conn_splits_merges_mixed_dict.items():
            t_from_new = lr_C0_condensed_connections_relations[t_from]
            t_successors_times = {t:G2.nodes[t]["t_start"] for t in t_successors}
            t_start = G2.nodes[t_from_new]["t_end"]
            t_start2 = {t_from_new: t_start}
            t_time_max = max(t_successors_times.values())
            t_times         = {t:np.arange(t_start, t_end + 1, 1) for t,t_end in t_successors_times.items()}
            t_target_nodes = {t:t_segments_new[t][0] for t in t_successors}
            t_event_start_end_times[t_state][t_from_new] = {  't_start':        t_start2,
                                                              'branches':       t_successors,
                                                              't_end':          t_time_max,
                                                              't_times':        {},
                                                              't_perms':        {},
                                                              't_target_nodes': t_target_nodes,
                                                              't_subIDs':       {},
                                                              't_active_IDs':   defaultdict(list)}

            # pre-isolate graph segment where all sub-events take place
            
            t_nodes_keep    =  [node for node in G.nodes() if t_start <= node[0] < t_time_max] # dont take end point 
            t_subgraph      =  G.subgraph(t_nodes_keep)                                        
            # gather nodes for all targets together. its not known which is correct target it its not split or merge, but proximity event

            t_segments_keep = [t_from_new] #+  t_successors                                    # dont include body of earlier branch, or any..

            t_sol = graph_sub_isolate_connected_components(G, t_start, t_time_max, lr_time_active_segments,
                                                           t_segments_new, t_segments_keep, ref_node = t_segments_new[t_from_new][-1])

            t_sol += [t_node for t_node in t_target_nodes.values()]                            # add firest nodes branches, so they are in perms 
            t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
            t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
            t_event_start_end_times[t_state][t_from_new]['t_perms'][t_from_new] = t_values
            t_event_start_end_times[t_state][t_from_new]['t_times'][t_from_new] = np.arange(t_start, t_time_max + 1, 1)
            t_event_start_end_times[t_state][t_from_new]['t_subIDs'][t_from_new] = t_perms
            #t_event_start_end_times[t_state][t_from_new]['t_active_IDs'][t_from_new] = np.arange(t_start, t_time_max + 1, 1)

        
        t_state = 'mixed2'
        for t_from_all,t_to_all in lr_conn_mixed_from_to.items():
            #t_predecessors   = extractNeighborsPrevious(G2, t_to, func_prev_neighb)
            #t_predecessors = [lr_C0_condensed_connections_relations[t] for t in t_predecessors] # update if inherited
            t_from_all_new = [lr_C0_condensed_connections_relations[t] for t in t_from_all]
            t_predecessors_times_end = {t:G2.nodes[t]["t_end"] for t in t_from_all_new}
            #t_t_to_start = G2.nodes[t_to]["t_start"]
            #t_times         = {t_ID: np.arange(t_t_from_end, t_t_to_start + 1, 1)
            #                   for t_ID,t_t_from_end in t_predecessors_times_end.items()}

            #t_from_new = lr_C0_condensed_connections_relations[t_from]                  # split segment may be inherited.
            #t_from_successors   = extractNeighborsNext(G2, t_from_new, func_next_neighb)     # branches are not

            t_successors_times_start = {t_to:G2.nodes[t_to]["t_start"] for t_to in t_to_all}
            t_target_nodes = {t:t_segments_new[t][0] for t in t_to_all}
            #t_t_start = G2.nodes[t_from_new]["t_end"]
            #t_times         = {t:np.arange(t_t_start, t_t_end + 1, 1) for t,t_t_end in t_successors_times.items()}

            t_event_start_end_times[t_state][t_from_all] = {'t_start':        t_predecessors_times_end,
                                                            'branches':       t_from_all_new,
                                                            't_end':          t_successors_times_start,
                                                            't_times':        {},
                                                            't_perms':        {},
                                                            't_target_nodes': t_target_nodes,
                                                            't_subIDs':       {},
                                                            't_active_IDs':   {}}

            # pre-isolate graph segment where all sub-events take place
            t_t_from_min    = min(t_predecessors_times_end.values())    # take earliest branch time
            t_t_to_max      = max(t_successors_times_start.values()) 
            t_times_all = np.arange(t_t_from_min, t_t_to_max + 1, 1)
            t_active_IDs = {t:[] for t in t_times_all[1:]}
            t_nodes_keep    = [node for node in G.nodes() if t_t_from_min < node[0] <= t_t_to_max] # < excluded edges
            t_subgraph = G.subgraph(t_nodes_keep)   # big subgraph
            # >> have to create t_subIDs and t_active_IDs. t_subIDs
            #for t_from, t_times_all in t_times.items():
            for t_from_new in t_from_all_new:
                #t_nodes_keep2 = t_nodes_keep.copy()
                #t_nodes_keep2 += [t_segments_new[t_from][-1]]
                #t_subgraph = G.subgraph(t_nodes_keep)
                t_segments_keep = [t_from] + list(t_to_all)  
                t_start = t_predecessors_times_end[t_from_new]  # storage uses old ID
                t_t_to_start = t_t_to_max
                if t_t_to_start - t_start > 1:
                    t_sol = graph_sub_isolate_connected_components(t_subgraph, t_start + 1, t_t_to_start, lr_time_active_segments,
                                                                   t_segments_new, t_segments_keep, ref_node = t_segments_new[t_to_all[0]][0]) 
                    t_perms = disperse_nodes_to_times(t_sol) # reformat sol into time:subIDs
                    t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
                else:
                    t_perms = {}
                    t_values = None
                t_event_start_end_times[t_state][t_from_all]['t_perms'        ][t_from_new] = t_values
                t_event_start_end_times[t_state][t_from_all]['t_subIDs'       ][t_from_new] = t_perms

                for t_time in set([t[0] for t in t_sol]):
                    t_active_IDs[t_time].append(t_from_new)
            t_event_start_end_times[t_state][t_from_all]['t_active_IDs' ] = t_active_IDs
                #for t_time in t_times_all[1:-1]: # remove start-end points
                #    t_event_start_end_times[t_state][t_from_all]['t_active_IDs'   ][t_time].append(t_from)
    # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
# ===============================================================================================
# ====  Determine interpolation parameters ====
# ===============================================================================================
# we are interested in segments that are not part of psuedo branches (because they will be integrated into path)
# pre merge segments will be extended, post split branches also, but backwards. 
# for pseudo event can take an average paramets for prev and post segments.
print('Determining interpolation parameters k and s for segments...')
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
print('Working on fake events...')
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
    #t_to    = lr_C0_condensed_connections_relations[t_to]
    t_from  = lr_C0_condensed_connections_relations[t_from]
    t_from_k_s  = t_segment_k_s[t_from  ]
    t_to_k_s    = t_segment_k_s[t_to    ]
    t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))   # determine which k_s to inherit (lower)
    t_fake_events_k_s_edge_master[t_to] = {'state': 'merge', 'edge': (t_from, t_to), 'k_s':t_k_s_out}
        
for t_from_old in t_split_fake_from_ID:
    t_from  = lr_C0_condensed_connections_relations[t_from_old]
    t_to = t_event_start_end_times['split'][t_from]['post_successor']
    #t_to    = lr_C0_condensed_connections_relations[t_to]
    t_from_k_s  = t_segment_k_s[t_from  ]
    t_to_k_s    = t_segment_k_s[t_to    ]
    t_k_s_out = get_k_s(t_from_k_s, t_to_k_s, backup = (1,5))  
    t_fake_events_k_s_edge_master[t_from_old] = {'state': 'split', 'edge': (t_from, t_to), 'k_s':t_k_s_out}
# ----------------------- interpolate missing events with known history-----------------------
# interpolate
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print('Performing interpolation on data without fake branches')
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
print('Finding most possible path evolutions...')
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
    t_args = [t_temp_cases,t_temp_centroids,t_temp_times,
            lr_big121s_perms_centroids,lr_big121s_perms_areas,lr_big121s_perms_mom_z]

    t_sols_c, t_sols_c_i, t_sols_a, t_sols_m = lr_evel_perm_interp_data(*t_args)

    t_weights   = [1,1.5,0,1]
    t_sols      = [t_sols_c, t_sols_c_i, t_sols_a, t_sols_m]
    lr_weighted_solutions_max, lr_weighted_solutions_accumulate_problems =  lr_weighted_sols(t_weights, t_sols, t_temp_cases )
a = 1 # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ----------------- Save recovered fake event paths -------------------------
print('Saving fake event data...')
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
            #for t_time,t_comb in zip(t_times,t_path):                 # create composite nodes
            #    t_nodes_new.append(tuple([t_time] + list(t_comb)))    # new type of nodes in solution: (time,*clusterIDs)   e.g (t1,ID1,ID2,...)

            #t_nodes_all = t_event_start_end_times[t_state][t_ID]['t_nodes_solo']
            
            ## basically in order to correctly reconnectd sides of inter-segment i have to also delete old edges from side nodes
            ## this can be done by deleting them and adding back
            #t_nodes_new_sides = [t_segments_new[t_from][-2]] + t_nodes_new + [t_segments_new[t_to][1]]
            ## find element which will be deleted, but will be readded later
            #t_common_nodes = set(t_nodes_all).intersection(set(t_nodes_new_sides))
            ## store their parameters
            #t_node_params = {t:dict(G.nodes[t]) for t in t_common_nodes}
            
            #G.remove_nodes_from(t_nodes_all)                          # edges will be also dropped
            

            #t_pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
    
            #G.add_edges_from(t_pairs)
            ## restore deleted parameters
            #for t,t_params in t_node_params.items():
            #    G.add_node(t, **t_params)
            ## determine t_from masters index, send that segment intermediate nodes and second segment
            ## if t_from is its own master, it still works, check it with simple ((0,1),(1,2)) and {0:0,1:0,2:0}
            #t_from_new = lr_C0_condensed_connections_relations[t_from]
            #t_nodes_intermediate = list(sorted(t_nodes_new, key = lambda x: x[0]))[1:-1]
            #t_segments_new[t_from_new] += t_nodes_intermediate
            #t_segments_new[t_from_new] += t_segments_new[t_to]
    
            ## fill centroid, area and momement of innertia zz missing for intermediate segment
            #set_custom_node_parameters(G, g0_contours, t_nodes_intermediate, t_from_new, calc_hull = 1)
    
            #for t_node in t_segments_new[t_to]:
            #    G.nodes[t_node]["owner"] = t_from_new
            ## copy data from inherited
    
            ## wipe data if t_from is inherited
            #if t_from_new != t_from:
            #    t_segments_new[t_from] = []
            ## wipe data from branches and t_to
            #for t_ID_2 in t_IDs_remove + [t_to]:
            #    t_segments_new[t_ID_2] = []

            #if t_state == 'merge':   
            #    lr_conn_merges_to_nodes.remove(t_ID_old) # its not real, remove
            #else:
            #    lr_conn_splits_from_nodes.remove(t_ID_old)
            #for t in t_IDs_remove:
            #    lr_C0_condensed_connections_relations[t] = t_from_new

            #lr_C0_condensed_connections_relations[t_to] = t_from_new
            ## modify node view graph
            #t_successors   = extractNeighborsNext(G2, t_to, lambda x: G2.nodes[x]["t_start"])
            #t_edges = [(t_from_new,t_succ) for t_succ in t_successors]
            #G2.remove_nodes_from(t_IDs_remove + [t_to])
            #G2.add_edges_from(t_edges)
            #G2.nodes()[t_from_new]["t_end"] = t_segments_new[t_from_new][-1][0]

G_seg_view_2 = nx.Graph()
G_seg_view_2.add_edges_from([(x,y) for y,x in lr_C0_condensed_connections_relations.items()])

lr_C1_condensed_connections = extract_graph_connected_components(G_seg_view_2, lambda x: x)

# lets condense all sub-segments into one with smallest index. EDIT: give each segment index its master. since number of segments will shrink anyway
t_condensed_connections_all_nodes = sorted(sum(lr_C1_condensed_connections,[])) # neext next
lr_C1_condensed_connections_relations = {tID: tID for tID in range(len(segments2))} 
for t_subIDs in lr_C1_condensed_connections:
    for t_subID in t_subIDs:
        lr_C1_condensed_connections_relations[t_subID] = min(t_subIDs)
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ====  EXTEND REAL MERGE/SPLIT BRANCHES ====
# ===============================================================================================
# extrapolate merge/split branches one time step at a time using prev determined k and s values
print('Analyzing real merge/split events. extending branches...')

t_out                   = defaultdict(dict)
t_extrapolate_sol       = defaultdict(dict)
t_extrapolate_sol_comb  = defaultdict(dict)
lr_extend_merges_IDs =  [(t,'merge') for t in t_merge_real_dict.keys()] 
lr_extend_merges_IDs += [(t,'mixed2') for t in lr_conn_mixed_from_to]
lr_extend_merges_IDs += [(lr_C1_condensed_connections_relations[t],'split') for t in t_split_real_dict.keys()]
lr_extend_merges_IDs += [(lr_C1_condensed_connections_relations[t],'mixed') for t in lr_conn_splits_merges_mixed_dict]
#lr_extend_merges_IDs = list() + list(t_split_real_dict.keys()) + [lr_C1_condensed_connections_relations[t] for t in lr_conn_splits_merges_mixed_dict]
lr_mixed_completed      = {'full': defaultdict(dict),'partial': defaultdict(dict)} #AutoCreateDict()#
for t_ID, t_state in lr_extend_merges_IDs:
    if t_state in ('merge','split'):
        t_branches = t_event_start_end_times[t_state][t_ID]['branches']
        t_times_target = []
    elif t_state == 'mixed2':
        #t_state == 'mixed2'
        t_branches = t_event_start_end_times[t_state][t_ID]['branches']
        t_nodes_target = t_event_start_end_times[t_state][t_ID]['t_target_nodes']
        t_times_target = [t[0] for t in t_nodes_target.values()]
        t_subIDs_target = {t:[] for t in t_times_target}
        for t_time, *t_subIDs in t_nodes_target.values():
            t_subIDs_target[t_time] += t_subIDs
    else:
        t_branches = [t_ID] # dont have real branches, molding data to merge case structure
        t_nodes_target = t_event_start_end_times[t_state][t_ID]['t_target_nodes']
        t_times_target = [t[0] for t in t_nodes_target.values()]
        t_subIDs_target = {t:[] for t in t_times_target}
        for t_time, *t_subIDs in t_nodes_target.values():
            t_subIDs_target[t_time] += t_subIDs
# =========== EXPERIMENTAL PART START =============
# generate simultaneous, self excluding choices for all branches for all time steps of merge.
# -> at time T, two branches have to be recovered from cluster C = [1,2,3]
# -> for branches [B1, B2] try redistributions ([[1],[2,3]], [[2],[1,3]], [[3],[1,2]])
    if t_state in ('mixed2','mixed'):#('merge','split'):

        def split_into_bins(L, num_bins):
            n = len(L)
    
            if num_bins == 1:
                yield [L]#([list(t)] for t in combs_different_lengths(L))  # Sort the elements before yielding
                return
    
            for r in range(1, n - num_bins + 2):
                for combination in itertools.combinations(L, r):
                    remaining = [x for x in L if x not in combination]
                    for rest in split_into_bins(remaining, num_bins - 1):
                        yield [sorted(list(combination))] + rest  # Sort the combination sublist before yielding

        if t_state == 'split':
            t_start = min(t_event_start_end_times[t_state][t_ID]['t_start'].values())
            t_end   = t_event_start_end_times[t_state][t_ID]['t_end']
        elif t_state == 'mixed2':
            t_start = min(t_event_start_end_times[t_state][t_ID]['t_start'].values())
            t_end   = max(t_event_start_end_times[t_state][t_ID]['t_end'].values())
        else:
            t_start = min(t_event_start_end_times[t_state][t_ID]['t_start'].values())
            t_end   = t_event_start_end_times[t_state][t_ID]['t_end']

        t_subIDs_dict = {t_time:set() for t_time in np.arange(t_start,t_end + 1 ,1)} # pregen time where branches have to be resolved

        for t_branch_ID in t_branches:                                               # combine subIDs of branch choices for set time t
            for t_time, t_subIDs in t_event_start_end_times[t_state][t_ID]['t_subIDs'][t_branch_ID].items():
                t_subIDs_dict[t_time].update(t_subIDs)
        a = 1
        #t_perms_distribution = defaultdict(dict)                                     # generate choices from pool
        #t_perms_parameters = defaultdict(dict) 
        #for t_time, t_subIDs_branch in t_event_start_end_times[t_state][t_ID]['t_active_IDs'].items():
        #    t_subIDs = t_subIDs_dict[t_time]
        #    if len(t_subIDs_branch) == 1:
        #        t_perms_distribution[t_time][tuple(t_subIDs_branch)] = [[list(t)] for t in combs_different_lengths(t_subIDs)]
        #    else:
        #        t_perms_distribution[t_time][tuple(t_subIDs_branch)] = list(split_into_bins(t_subIDs,len(t_subIDs_branch)))
        #    t_permutations = combs_different_lengths(t_subIDs)
        #    for t_permutation in t_permutations:
        #        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][tID] for tID in t_permutation]))
        #        t_perms_parameters[t_time][t_permutation] = centroid_area(t_hull)
                
        a = 1
        t_all_norm_buffers  = {}
        t_all_traj_buffers  = {}
        t_all_time_buffers  = {}
        t_all_k_s           = {}
        for t_branch_ID in t_branches:
            if t_state in ('merge','mixed','mixed2'):
                t_conn = (t_branch_ID, t_ID)
                t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'][t_branch_ID] # last of branch
                t_branch_ID_new = lr_C1_condensed_connections_relations[t_branch_ID] 
                t_node_from = t_segments_new[t_branch_ID_new][-1]
                if t_state == 'merge':
                    t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end']                # first of target
                    t_node_to   = t_segments_new[t_ID][0]
                else: 
                    t_t_to = max(t_event_start_end_times[t_state][t_ID]['t_end'].values())
                    t_node_to = (-1)
            else:
                t_conn = (t_ID, t_branch_ID)
                t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'] # last of branch
                t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end'][t_branch_ID]                # first of target
                t_node_from = t_segments_new[t_ID][-1]
                t_node_to   = t_segments_new[t_branch_ID][0]
            if np.abs(t_t_to - t_t_from) < 2:
                t_out[t_ID][t_branch_ID] = None
                continue
            t_branch_ID_new = lr_C1_condensed_connections_relations[t_branch_ID] # probly not needed anymore

            if t_state in ('merge','mixed','mixed2'):
                t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if t_node[0] > t_t_from - h_interp_len_max2]
            else:
                t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if t_node[0] < t_t_to + h_interp_len_max2]

            trajectory = np.array([G.nodes[t]["centroid"] for t in t_nodes])
            time       = np.array([G.nodes[t]["time"    ] for t in t_nodes])
            
            N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
            if t_segment_k_s_diffs[t_branch_ID_new] is not None:
                t_last_deltas   = list(t_segment_k_s_diffs[t_branch_ID_new].values())[-N:]  # not changing for splits
            else:
                t_last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5

            #t_norm_buffer   = CircularBuffer(N, t_last_deltas)
            t_all_norm_buffers[t_branch_ID] = CircularBuffer(N, t_last_deltas)
            if t_state in ('merge','mixed','mixed2'):
                t_all_traj_buffers[t_branch_ID] = CircularBuffer(h_interp_len_max2, trajectory)
                t_all_time_buffers[t_branch_ID] = CircularBuffer(h_interp_len_max2, time)
            else: 
                t_all_traj_buffers[t_branch_ID] = CircularBufferReverse(h_interp_len_max2, trajectory)
                t_all_time_buffers[t_branch_ID] = CircularBufferReverse(h_interp_len_max2, time)

            t_extrapolate_sol[      t_conn] = {}
            t_extrapolate_sol_comb[ t_conn] = {}
            t_times_accumulate_resolved     = []

            if t_segment_k_s[t_branch_ID_new] is not None:
                t_k,t_s = t_segment_k_s[t_branch_ID_new]
                t_all_k_s[t_branch_ID] = t_segment_k_s[t_branch_ID_new]
            else:
                t_k,t_s = (1,5)
                t_all_k_s[t_branch_ID] = (1,5)
        # for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        t_failed_branch = []
        t_times_accumulate_resolved_start = t_event_start_end_times[t_state][t_ID]['t_start']
        t_times_accumulate_resolved_start_end = {t:-1 for t in t_times_accumulate_resolved_start}
        for t_time_next, t_subIDs_branch in t_event_start_end_times[t_state][t_ID]['t_active_IDs'].items():
            t_time = t_time_next - 1
            t_extrapolate_results = {}
            t_branch_IDs = [t for t in t_subIDs_branch if t not in t_failed_branch]
            if len(t_branch_IDs) == 0: continue
            for t_branch_ID in t_branch_IDs:
                t_traj_b    = t_all_traj_buffers[t_branch_ID].get_data()
                t_time_b    = t_all_time_buffers[t_branch_ID].get_data()
                t_k, t_s    = t_all_k_s[t_branch_ID]
                t_extrap = interpolate_trajectory(t_traj_b, t_time_b, which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]
                t_extrapolate_results[t_branch_ID] = t_extrap

            a = 1
            t_centroids_extrap = np.array([t_extrapolate_results[t] for t in t_branch_IDs])

            t_subIDs = t_subIDs_dict[t_time_next] # which contours avalable
            if len(t_branch_IDs) == 1:
                t_perms_distribution2 = [[list(t)] for t in combs_different_lengths(t_subIDs)]
            else:
                t_perms_distribution2 = list(split_into_bins(t_subIDs,len(t_branch_IDs)))
            t_perms_parameters2 = {}
            t_permutations = combs_different_lengths(t_subIDs)
            for t_permutation in t_permutations:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time_next][tID] for tID in t_permutation]))
                t_perms_parameters2[t_permutation] = centroid_area(t_hull)[0]

            t_diff_choices = {}
            for t, t_redist_case in enumerate(t_perms_distribution2):
                t_centroids = np.array([t_perms_parameters2[tuple(t)] for t in t_redist_case])
                
                t_diffs = np.array(t_centroids_extrap) - t_centroids
                t_diff_norms = np.linalg.norm(t_diffs, axis=1)
                t_diff_choices[t] = t_diff_norms
            t_diff_norms_sum = {t:np.sum(v) for t,v in t_diff_choices.items()}
            a = 1
            t_where_min = min(t_diff_norms_sum, key = t_diff_norms_sum.get) #np.argmin(t_diff_choices)
            t_sol_d_norms = t_diff_choices[t_where_min]
            t_sol_subIDs = t_perms_distribution2[t_where_min]
            t_branch_pass = {}
            for t_branch_ID, t_subIDs, t_sol_d_norm in zip(t_branch_IDs,t_sol_subIDs, t_sol_d_norms):
                t_known_norms = np.array(t_all_norm_buffers[t_branch_ID].get_data())
                t_mean = np.mean(t_known_norms)
                t_std = np.std(t_known_norms)
                if t_sol_d_norm < max(t_mean, 5) + 5* t_std:
                    t_all_norm_buffers[t_branch_ID].append(t_sol_d_norm)
                    t_all_traj_buffers[t_branch_ID].append(t_extrapolate_results[t_branch_ID])
                    t_all_time_buffers[t_branch_ID].append(t_time_next)
                
                    t_extrapolate_sol_comb[t_conn][t_time_next] = tuple(t_subIDs)
                    t_times_accumulate_resolved_start_end[t_branch_ID] = t_time_next
                else:
                    t_failed_branch.append(t_branch_ID)
                    continue


            if t_state in  ('mixed','mixed2'):
                if t_time in t_times_target:
                    t_nodes_sol_subIDs = list(t_permutations[t_where_min])
                    t_nodes_solution = tuple([t_time] + t_nodes_sol_subIDs)

                    if t_nodes_solution in t_nodes_target.values():
                        t_target_ID = find_key_by_value(t_nodes_target,t_nodes_solution)
                        lr_mixed_completed['full'][t_ID]['solution'] = t_nodes_sol_subIDs
                        if 'targets' not in lr_mixed_completed['full'][t_ID]: lr_mixed_completed['full'][t_ID]['targets'] = []
                        lr_mixed_completed['full'][t_ID]['targets'].append(t_target_ID)
                    else:
                        set1 = set(t_subIDs_target[t_time])
                        set2 = set(t_nodes_sol_subIDs)
                        inter = set1.intersection(set2)

                        if len(inter)> 0:
                            lr_mixed_completed['partial'][t_ID]['solution'] = t_nodes_sol_subIDs
                            #if 'targets' not in lr_mixed_completed['partial'][t_ID]: lr_mixed_completed['partial'][t_ID]['targets'] = []
                            #lr_mixed_completed['partial'][t_ID]['targets'].append(t_target_ID)
        for t_branch_ID,t_start in t_times_accumulate_resolved_start.items():
            t_branch_ID_new = lr_C1_condensed_connections_relations[t_branch_ID]
            t_end = t_times_accumulate_resolved_start_end[t_branch_ID]
            t_node_from = t_segments_new[t_branch_ID_new][0]
            t_node_to   = t_segments_new[t_branch_ID_new][-1]
            print(f' {t_state}:branch :{t_branch_ID_new} = {t_node_from}->{t_node_to} recovered times:{t_start}->{t_end}')
    

# =========== EXPERIMENTAL PART END =============
    #for t_branch_ID in t_branches:
    #    t_combs_all = t_event_start_end_times[t_state][t_ID]['t_perms'][t_branch_ID]
    #    t_times_all = t_event_start_end_times[t_state][t_ID]['t_times'][t_branch_ID]

    for t_branch_ID in t_branches:
        if t_state in ('merge','mixed'):
            t_conn = (t_branch_ID, t_ID)
            t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'][t_branch_ID] # last of branch
            t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end']                # first of target
            t_branch_ID_new = lr_C1_condensed_connections_relations[t_branch_ID] 
            t_node_from = t_segments_new[t_branch_ID_new][-1]
            t_node_to   = t_segments_new[t_ID][0]
        else:
            t_conn = (t_ID, t_branch_ID)
            t_t_from    = t_event_start_end_times[t_state][t_ID]['t_start'] # last of branch
            t_t_to      = t_event_start_end_times[t_state][t_ID]['t_end'][t_branch_ID]                # first of target
            t_node_from = t_segments_new[t_ID][-1]
            t_node_to   = t_segments_new[t_branch_ID][0]
        if np.abs(t_t_to - t_t_from) < 2:
            t_out[t_ID][t_branch_ID] = None
            continue
        t_branch_ID_new = lr_C1_condensed_connections_relations[t_branch_ID] # probly not needed anymore

        if t_state in ('merge','mixed'):
            t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if t_node[0] > t_t_from - h_interp_len_max2]
        else:
            t_nodes = [t_node for t_node in t_segments_new[t_branch_ID_new] if t_node[0] < t_t_to + h_interp_len_max2]

        trajectory = np.array([G.nodes[t]["centroid"] for t in t_nodes])
        time       = np.array([G.nodes[t]["time"    ] for t in t_nodes])
            
        N = 5       # errors_sol_diff_norms_all might be smaller than N, no fake numbers are initialized inside
        if t_segment_k_s_diffs[t_branch_ID_new] is not None:
            t_last_deltas   = list(t_segment_k_s_diffs[t_branch_ID_new].values())[-N:]  # not changing for splits
        else:
            t_last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5

        t_norm_buffer   = CircularBuffer(N, t_last_deltas)

        if t_state in ('merge','mixed'):
            t_traj_buff     = CircularBuffer(h_interp_len_max2, trajectory) # added element to buffer is deployed 
            t_time_buff     = CircularBuffer(h_interp_len_max2, time)       # in the end, first removed
            t_time_next     = t_t_from    + 1                               # [        -h_interp_len_max2:]
        else: 
            t_traj_buff     = CircularBufferReverse(h_interp_len_max2, trajectory) # added element to buffer is deployed 
            t_time_buff     = CircularBufferReverse(h_interp_len_max2, time)       # in the start, last removed
            t_time_next     = t_t_to      - 1

        t_extrapolate_sol[      t_conn] = {}
        t_extrapolate_sol_comb[ t_conn] = {}
        t_times_accumulate_resolved     = []

        if t_segment_k_s[t_branch_ID_new] is not None:
            t_k,t_s = t_segment_k_s[t_branch_ID_new]
        else:
            t_k,t_s = (1,5)
        
        t_combs_all = t_event_start_end_times[t_state][t_ID]['t_perms'][t_branch_ID]
        t_times_all = t_event_start_end_times[t_state][t_ID]['t_times'][t_branch_ID] 

        if t_state == 'merge':
            t_combs_all = t_combs_all[1:-1]
            t_times_all = t_times_all[1:-1]
        elif t_state == 'split':
            t_combs_all = t_combs_all[-2::-1]
            t_times_all = t_times_all[-2::-1]
        else:
            t_combs_all = t_combs_all[1:]               # last element is being dropped during saving. should examine if needed. EDIT: not sure if relevant
            t_times_all = t_times_all[1:]

        for t_time, t_permutations in zip(t_times_all,t_combs_all):
            t_extrap = interpolate_trajectory(t_traj_buff.get_data(), t_time_buff.get_data(), which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]

            t_centroids = []
            t_areas     = []
            for t_permutation in t_permutations:
                t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][tID] for tID in t_permutation]))
                t_centroid, t_area = centroid_area(t_hull)
                t_centroids.append(t_centroid)
                t_areas.append(t_area)
        
            t_diffs = np.array(t_centroids).reshape(-1,2) - t_extrap
            t_diff_norms = np.linalg.norm(t_diffs, axis=1)
            t_where_min = np.argmin(t_diff_norms)
            t_sol_d_norm = t_diff_norms[t_where_min]
            t_known_norms = np.array(t_norm_buffer.get_data())
            t_mean = np.mean(t_known_norms)
            t_std = np.std(t_known_norms)
            if t_diff_norms[t_where_min] < max(t_mean, 5) + 5* t_std:
                t_norm_buffer.append(t_sol_d_norm)
                t_traj_buff.append(t_extrap)
                t_time_buff.append(t_time_next)
                
                t_extrapolate_sol[t_conn][t_time] = t_extrap
                t_extrapolate_sol_comb[t_conn][t_time] = t_permutations[t_where_min]
                #print(f"inter!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
                t_times_accumulate_resolved.append(t_time)
            else:
                break

            if t_state == 'mixed':
                if t_time in t_times_target:
                    t_nodes_sol_subIDs = list(t_permutations[t_where_min])
                    t_nodes_solution = tuple([t_time] + t_nodes_sol_subIDs)

                    if t_nodes_solution in t_nodes_target.values():
                        t_target_ID = find_key_by_value(t_nodes_target,t_nodes_solution)
                        lr_mixed_completed['full'][t_ID]['solution'] = t_nodes_sol_subIDs
                        if 'targets' not in lr_mixed_completed['full'][t_ID]: lr_mixed_completed['full'][t_ID]['targets'] = []
                        lr_mixed_completed['full'][t_ID]['targets'].append(t_target_ID)
                    else:
                        set1 = set(t_subIDs_target[t_time])
                        set2 = set(t_nodes_sol_subIDs)
                        inter = set1.intersection(set2)

                        if len(inter)> 0:
                            lr_mixed_completed['partial'][t_ID]['solution'] = t_nodes_sol_subIDs
                            #if 'targets' not in lr_mixed_completed['partial'][t_ID]: lr_mixed_completed['partial'][t_ID]['targets'] = []
                            #lr_mixed_completed['partial'][t_ID]['targets'].append(t_target_ID)
            t_time_next  += 1

        if len(t_times_accumulate_resolved) > 0:
            if 1 <= len(t_times_accumulate_resolved)<= 3:
                t =f'{t_times_accumulate_resolved}'
            else:
                t = f'{t_times_accumulate_resolved[0]}->{t_times_accumulate_resolved[-1]}'

            print(f' {t_state}:edge :({t_branch_ID_new},{t_ID})= {t_node_from}->{t_node_to}  time: {t}')
        else:
            print(f'!{t_state}:edge :({t_branch_ID_new},{t_ID})= {t_node_from}->{t_node_to} failed, no times')


# search for conflict nodes for extended branches
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# ====  FOR BRANCH EXTENSION, SEE IF THERE ARE CONTESTED NODES AND REDISTRIBUTE THEM ====
# ===============================================================================================
# WHY: extensions are performed effectively in parallel, without coupling between processes.
# WHY: thats why some recovered paths may overlap. and overlap has to be removed.
# HOW: nodes that are in multiple extensions are called contested. Since each contested node can have
# HOW: only one owner, we construct all possible choices for this kind of node redistribution.
# HOW: additionally, we cannot strip node from path that will be left with no nodes at any time step.
# NOTE: only case where no contested nodes are found and only one possible redistribution is scripted. <<<
def conflicts_stage_1(owner_dict):
    # find contested nodes by counting their owners
    node_owners = defaultdict(list)
    for owner, times_subIDs in owner_dict.items():
        # {owner:nodes,...}, where nodes = {time:*subIDs,...} = {483: (3,), 484: (4,),..}
        nodes =  [(time, subID) for time, subIDs in times_subIDs.items() for subID in subIDs]
        for node in nodes:
            node_owners[node].extend([owner])

    return {node: owners for node,owners in node_owners.items() if len(owners) > 1}

def conflicts_stage_2(contested_dict):
    # prep data for redistribution schemes. get options of node owners: node1:owners1 -> 
    # -> (owner) choices for node 1 = ([node1, owner11],[node1, owner12],..)
    # node_owner_choices -> [choices for node 1, choices for node 2,...]
    node_owner_choices = [] 
    for key,values in contested_dict.items():
        node_owner_choices.append(list(itertools.product([key],values)))
    # for fair node distribution, only one owner choice from each "choices for node X" is possible
    # so we construct combinations of single choices for each "choices for node X"
    # which is well described by permutation product (a branching choice tree).
    return list(itertools.product(*node_owner_choices))

def conflicts_stage_3(node_destribution_options,contested_node_owners_dict, owner_dict):
    # if this choice of node redistribution is correct, i have to delete contested nodes from
    # alternative owners. if some of alternative owners are left with no nodes, its incorrect choice
    variants_possible = []
    for nodes_owners in node_destribution_options: # examine one redistribution variant
        t_skip = False
        for node, owner in nodes_owners:           # examine one particular node
            time, *subIDs = node                   # determine alternative owners
            owners_others = [t for t in contested_node_owners_dict[node] if t != owner]
            for owner_other in owners_others:
                leftover_subIDs = set(owner_dict[owner_other][time]) - set(subIDs)
                if  len(leftover_subIDs) == 0:     # check subIDs after removal of contested subIDs
                    t_skip = True
                    break                          # stop examining alternative owners. case failed
            if t_skip: break                       # stop examining nodes of case
        if not t_skip:
            variants_possible.append(nodes_owners)
    return variants_possible

lr_conn_merges_good = set()#dict()#defaultdict(set)
if 1 == 1:
   
    # check if there are contested nodes in all extrapolated paths
    t_duplicates = conflicts_stage_1(t_extrapolate_sol_comb)
    assert len(t_duplicates) == 0, 'havent tested after addition of split and mixed extension code 28.09.23'
    if len(t_duplicates) > 0:
        # retrieve viable ways of redistribute contested nodes
        variants_all        = conflicts_stage_2(t_duplicates)
        variants_possible   = conflicts_stage_3(variants_all,t_duplicates, t_extrapolate_sol_comb)
        #if there is only one solution by default take it as answer
        if len(variants_possible) == 1:  
            t_choice_evol = variants_possible[0]
        else:
            # method is not yet constructed, it should be based on criterium minimization for all variants
            # current, trivial solution, is to pick solution at random. at least there is no overlap.
            assert -1 == 0, 'multiple variants of node redistribution'
            t_choice_evol = variants_possible[0]
         
        # redistribute nodes for best solution.
        for t_node,t_conn in t_choice_evol:
            tID                 = t_conn[1]     # not correct w.r.t different states <<<
            t_time, *t_subIDs   = t_node
            t_delete_conns      = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
            for t_delete_conn in t_delete_conns:
                t_temp = t_extrapolate_sol_comb[t_delete_conn][t_time]
                t_temp = [t for t in t_temp if t not in t_subIDs]
                t_extrapolate_sol_comb[t_delete_conn][t_time] = t_temp
            t_conns_relevant = [t_c for t_c in t_extrapolate_sol_comb if t_c[1] == tID]
            lr_conn_merges_good.update(t_conns_relevant) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< not correct anymore

    t_all_problematic_conns = list(set(sum(list(t_duplicates.values()),[])))
    t_all_problematic_conns_to = [a[1] for a in t_all_problematic_conns]
    for tID,t_state in lr_extend_merges_IDs: # here lies problem with splits, and possibly with mixed cases!!!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if tID not in t_all_problematic_conns_to:
            if t_state      == 'merge':
                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
            elif t_state    == 'split':
                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[0] == tID]
            else:
                t_conns_relevant = [t_conn for t_conn in t_extrapolate_sol_comb if t_conn[1] == tID]
            for t_conn in t_conns_relevant:
                lr_conn_merges_good.add((t_conn,t_state))
            #lr_conn_merges_good[(tID,t_state)].update(t_conns_relevant)
            #lr_conn_merges_good.update((tID,t_state))
    
        print('branches are resolved without conflict, or conflict resolved by redistribution of nodes')
# for_graph_plots(G, segs = t_segments_new)         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#for (t_ID,t_state), t_conns in lr_conn_merges_good.items():
for t_conn, t_state in lr_conn_merges_good:
    # modifying graphs and storage based on recovered path though solo node space (web) between segments (see (6))
    # path consists of composite nodes. they are disperesed back to solo nodes and deleted from graph. new comp nodes are added
    # if path is extended partially, special treatment of edges is needed (see (3))
    #for t_conn in t_conns:
    t_from, t_to = t_conn
    t_combs = t_extrapolate_sol_comb[t_conn]

    if len(t_combs) == 0: continue                              # no extension, skip.

    if t_state in ('merge', 'mixed') and t_from not in lr_mixed_completed['full']:
        save_connections_merges(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from,  t_to, G, G2, lr_C1_condensed_connections_relations, g0_contours)
    elif t_state == 'split':
        save_connections_splits(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from,  t_to, G, G2, lr_C1_condensed_connections_relations, g0_contours)
    else:
        t_to = lr_mixed_completed['full'][t_from]['targets'][0]
        save_connections_two_ways(t_segments_new, t_extrapolate_sol_comb[t_conn], t_from,  t_to, G, G2, lr_C1_condensed_connections_relations, g0_contours)
        
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
#for_graph_plots(G, segs = t_segments_new)

# ===============================================================================================
# ============== Final passes. finding new straight segments after refinement ====================
# ===============================================================================================
# WHY: after merge extension and mixed extension there may be branches left over in interaction area
# HOW: analyze recent graph for straight segments and determine which segments are different from old
t_segments_fin_dic, skipped = graph_extract_paths(G,lambda x : x[0])
t_segments_fin = [t for t in t_segments_fin_dic.values() if len(t) > 0]

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
    G2.nodes()[t_new_ID]["t_start"] = t_segments_new[t_new_ID][0][0]
    G2.nodes()[t_new_ID]["t_end"]   = t_segments_new[t_new_ID][-1][0]
    set_custom_node_parameters(G, g0_contours_hulls, t_segments_fin[t_ID], t_new_ID, calc_hull = 0) # straight hulls since solo nodes 
    #for t_time, *t_subIDs in  t_segments_fin[t_ID]:
    #    lr_time_active_segments[t_time].append(t_new_ID)


lr_time_active_segments = {t:[] for t in lr_time_active_segments}
for k,t_segment in enumerate(t_segments_new):
    t_times = [a[0] for a in t_segment]
    for t in t_times:
        lr_time_active_segments[t].append(k)

# for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# for_graph_plots(G, segs = t_segments_fin)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ===============================================================================================
# ============== Final passes. new straight. update connectivity ====================
# ===============================================================================================
G2.add_edges_from(G2.edges())
fin_connectivity_graphs = defaultdict(list) #fin_additional_segments_IDs
t_segments_relevant = [t for t,t_seg in enumerate(t_segments_new) if len(t_seg) > 0]
for t_ID in t_segments_relevant: # check left and right connected neighbors. path only though free nodes.
    t_DT = 20                            # first subgraph contains nodes up to, but not including source (*)
                                         # as to drop segments that end in same time, so connections are not made in reverse. 
    t_t_min = G2.nodes[t_ID]["t_start" ]
    t_start = t_t_min - t_DT                
    t_end   = t_t_min - 1
    t_add_custom_nodes = t_segments_new[t_ID][:1]    # one source node is added to compensate (*)
    t_node_from = t_add_custom_nodes[0]
    t_segments_active_all = unique_active_segments(t_start, t_end, lr_time_active_segments) # put outside function
    
    t_conns_left = find_segment_connectivity_isolated(G, t_start, t_end, t_add_custom_nodes, t_segments_active_all, t_ID, source_node = t_node_from, target_nodes = None)
    
    t_t_max = G2.nodes[t_ID]["t_end"] 
    t_start = t_t_max + 1                
    t_end   = t_t_max + t_DT
    t_add_custom_nodes = t_segments_new[t_ID][-1:]
    t_node_from = t_add_custom_nodes[0]
    t_segments_active_all = unique_active_segments(t_start, t_end, lr_time_active_segments)  # put outside function
    
    t_conns_right, t_conn_graphs = find_segment_connectivity_isolated(G, t_start, t_end, t_add_custom_nodes, t_segments_active_all, t_ID, source_node = t_node_from, target_nodes = None, return_cc = True)
    for t_conn, t_vals in t_conn_graphs.items():
        fin_connectivity_graphs[t_conn] = t_vals
    G2.add_edges_from(t_conns_left + t_conns_right)
    a = 1

fin_connectivity_graphs = {t_conn:t_vals for t_conn, t_vals in fin_connectivity_graphs.items() if len(t_vals) > 2}
for t_ID in fin_additional_segments_IDs: t_segment_k_s_diffs[t_ID] = None   
# for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# =========== Final passes. examine edges of terminated segments. retrieve nodes  ===============
# ===============================================================================================
# NOTE: element order in t_conn now is fixed (from, to), numbering hierarchy does not represent order anymore.
t_start_points  = []
t_end_points    = []

for t_ID in G2.nodes():
    t_successors    = extractNeighborsNext(     G2, t_ID, func_next_neighb)
    t_predecessors  = extractNeighborsPrevious( G2, t_ID, func_prev_neighb)
    if len(t_successors)    == 0: t_end_points.append(t_ID)
    if len(t_predecessors)  == 0: t_start_points.append(t_ID)

if 1 == 1:
    
    def dfs_pred(graph, node, time_lim, node_set):
        node_set.add(node)
        predecessors = list(graph.predecessors(node))
        for predecessor in predecessors:
            if predecessor not in node_set and graph.nodes[predecessor]['time'] > time_lim:
                dfs_pred(graph, predecessor, time_lim, node_set)

    def dfs_succ(graph, node, time_lim, node_set):
        node_set.add(node)
        successors = list(graph.successors(node))
        for successor in successors:
            if successor not in node_set and graph.nodes[successor]['time'] < time_lim:
                dfs_succ(graph, successor, time_lim, node_set)

    t_back = defaultdict(set)
    for t_ID in t_start_points:
        t_node_from = t_segments_new[t_ID][0]
        dfs_pred(G, t_node_from, time_lim = t_node_from[0] - 10, node_set = t_back[t_ID])
    t_back = {t:v for t,v in t_back.items() if len(v) > 1}
    t_forw = defaultdict(set)
    for t_ID in t_end_points:
        t_node_from = t_segments_new[t_ID][-1]
        dfs_pred(G, t_node_from, time_lim = t_node_from[0] + 10, node_set = t_forw[t_ID])
    t_forw = {t:v for t,v in t_forw.items() if len(v) > 1}


fafafs = defaultdict(dict)
for t_ID , t_nodes in t_back.items():
    t_perms = disperse_nodes_to_times(t_nodes)
    t_sorted = sorted(t_perms)[:-1]
    t_sorted.reverse()
    t_perms = {t:t_perms[t] for t in t_sorted}
    t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
    fafafs[(t_ID,'back')] = {t:v for t,v in zip(t_perms, t_values)}

for t_ID , t_nodes in t_forw.items():
    t_perms = disperse_nodes_to_times(t_nodes)
    t_sorted = sorted(t_perms)[1:]
    t_perms = {t:t_perms[t] for t in t_sorted}
    t_values = [combs_different_lengths(t_subIDs) for t_subIDs in t_perms.values()]
    fafafs[(t_ID,'forward')] = {t:v for t,v in zip(t_perms, t_values)}


# ===============================================================================================
# =========== Final passes. examine edges of terminated segments. extrapolate ===============
# ===============================================================================================
t_out                   = defaultdict(dict)
t_extrapolate_sol       = defaultdict(dict)
#t_extrapolate_sol_back_forw = {'back':defaultdict(dict), 'forward': defaultdict(dict)}
t_extrapolate_sol_comb  = defaultdict(dict)
#t_extrapolate_sol_comb_back_forw = {'back':defaultdict(dict), 'forward': defaultdict(dict)}
for (t_ID, t_state), t_combs in fafafs.items():
    t_conn = (t_ID, t_state) # <<<<< custom. t_ID may be two states, and they have to be differentiated
    #t_nodes = t_segments_new[t_ID]
    if t_state == 'forward':
        assert 1== -1, 'not tested. examine!'
        t_t_from = t_segments_new[t_ID][-1][0]
        t_nodes = [t_node for t_node in t_segments_new[t_ID] if t_node[0] > t_t_from - h_interp_len_max2]
        t_node_from = t_segments_new[t_ID][-1]
    else:
        t_t_to = t_segments_new[t_ID][0][0]
        t_nodes = [t_node for t_node in t_segments_new[t_ID] if t_node[0] < t_t_to + h_interp_len_max2]
        t_node_from = t_segments_new[t_ID][0]

    trajectory = np.array([G.nodes[t]["centroid"] for t in t_nodes])
    time       = np.array([G.nodes[t]["time"    ] for t in t_nodes])

    if t_state == 'forward':
        t_traj_buff     = CircularBuffer(h_interp_len_max2, trajectory) 
        t_time_buff     = CircularBuffer(h_interp_len_max2, time)       
        t_time_next     = t_t_from    + 1                               
    if t_state == 'back':
        t_traj_buff     = CircularBufferReverse(h_interp_len_max2, trajectory) 
        t_time_buff     = CircularBufferReverse(h_interp_len_max2, time)      
        t_time_next     = t_t_to      - 1
        #t_times = [t for t int_combs]
        #t_combs = {t:p for t,p in t_combs.items() }
    N = 5
    if t_segment_k_s_diffs[t_ID] is not None:
        t_last_deltas   = list(t_segment_k_s_diffs[t_ID].values())[-N:]  # not changing for splits
    else:
        t_last_deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)[-N:]*0.5

    t_norm_buffer   = CircularBuffer(N, t_last_deltas)
    t_times_accumulate_resolved = []
    for t_time, t_permutations in t_combs.items():
        
        t_time_next = t_time
        t_extrap = interpolate_trajectory(t_traj_buff.get_data(), t_time_buff.get_data(), which_times = [t_time_next] ,s = t_s, k = t_k, debug = 0 ,axes = 0, title = 'title', aspect = 'equal')[0]

        t_centroids = []
        t_areas     = []
        for t_permutation in t_permutations:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][tID] for tID in t_permutation]))
            t_centroid, t_area = centroid_area(t_hull)
            t_centroids.append(t_centroid)
            t_areas.append(t_area)
        
        t_diffs = np.array(t_centroids).reshape(-1,2) - t_extrap
        t_diff_norms = np.linalg.norm(t_diffs, axis=1)
        t_where_min = np.argmin(t_diff_norms)
        t_sol_d_norm = t_diff_norms[t_where_min]
        t_known_norms = np.array(t_norm_buffer.get_data())
        t_mean = np.mean(t_known_norms)
        t_std = np.std(t_known_norms)
        if t_diff_norms[t_where_min] < max(t_mean, 5) + 5 * t_std:
            t_norm_buffer.append(t_sol_d_norm)
            t_traj_buff.append(t_extrap)
            t_time_buff.append(t_time_next)
                
            t_extrapolate_sol[t_conn][t_time] = t_extrap
            t_extrapolate_sol_comb[t_conn][t_time] = t_permutations[t_where_min]
            #t_extrapolate_sol_back_forw[t_state][t_ID] = t_extrap
            #t_extrapolate_sol_comb_back_forw[t_state][t_ID] = t_permutations[t_where_min]
            #print(f"inter!:c{t_permutations[t_where_min]}, m{t_mean}, s{t_std}, df{t_diff_norms[t_where_min]}")
            t_times_accumulate_resolved.append(t_time)
        else:
            break
    
    
    if len(t_times_accumulate_resolved) > 0:

        t_node_to = tuple([t_time] + list(t_extrapolate_sol_comb[t_conn][t_time]))

        if 1 <= len(t_times_accumulate_resolved)<= 3:
            t =f'{t_times_accumulate_resolved}'
        else:
            t = f'{t_times_accumulate_resolved[0]}->{t_times_accumulate_resolved[-1]}'

        print(f' {t_state}: {t_ID} = {t_node_from}->{t_node_to}  time: {t}')
    else:
        t_node_to = '-'
        print(f'!{t_state}: {t_ID} = {t_node_from}->{t_node_to} failed, no times')
a = 1

# for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# =========== Final passes. examine edges of terminated segments. solve conflicts  ===============
# ===============================================================================================

# check if there are contested nodes in all extrapolated paths
t_duplicates = conflicts_stage_1(t_extrapolate_sol_comb)
assert len(t_duplicates) == 0, 'havent tested after addition of split and mixed extension code 28.09.23'
if len(t_duplicates) > 0:
    # retrieve viable ways of redistribute contested nodes
    variants_all        = conflicts_stage_2(t_duplicates)
    variants_possible   = conflicts_stage_3(variants_all,t_duplicates, t_extrapolate_sol_comb)
    #if there is only one solution by default take it as answer
    if len(variants_possible) == 1:  
        t_choice_evol = variants_possible[0]
    else:
        # method is not yet constructed, it should be based on criterium minimization for all variants
        # current, trivial solution, is to pick solution at random. at least there is no overlap.
        assert -1 == 0, 'multiple variants of node redistribution'
        t_choice_evol = variants_possible[0]
         
    # redistribute nodes for best solution.
    for t_node,t_conn in t_choice_evol:
        tID                 = t_conn[1]
        t_time, *t_subIDs   = t_node
        t_delete_conns      = [t_c for t_c in t_duplicates[t_node] if t_c != t_conn]
        for t_delete_conn in t_delete_conns:
            t_temp = t_extrapolate_sol_comb[t_delete_conn][t_time]
            t_temp = [t for t in t_temp if t not in t_subIDs]
            t_extrapolate_sol_comb[t_delete_conn][t_time] = t_temp
        t_conns_relevant = [t_c for t_c in t_extrapolate_sol_comb if t_c[1] == tID]
        lr_conn_merges_good.update(t_conns_relevant) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< not correct anymore

# for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===============================================================================================
# =========== Final passes. examine edges of terminated segments. save extensions  ===============
# ===============================================================================================

#for (t_ID,t_state), t_conns in lr_conn_merges_good.items():
for t_conn, t_combs in t_extrapolate_sol_comb.items():
    (t_ID, t_state) = t_conn
    # ref for comments to versions encountered previously.
    #for t_conn in t_conns:
    #t_from, t_to = t_conn
    #t_combs = t_extrapolate_sol_comb[t_conn]

    if len(t_combs) == 0: continue                              # no extension, skip.

    if t_state == 'forward':
        save_connections_merges(t_segments_new, t_extrapolate_sol_comb[t_conn], t_ID,  None, G, G2, lr_C1_condensed_connections_relations, g0_contours)
    elif t_state == 'back':
        save_connections_splits(t_segments_new, t_extrapolate_sol_comb[t_conn], None,  t_ID, G, G2, lr_C1_condensed_connections_relations, g0_contours)
    
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
    #t_start_edge = (t_times[0] - 1, t_times[0])
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

# for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ===============================================================================================
# ===============================================================================================
# ================================ Final passes. recompute split/merge/mixed info ===============================
# ===============================================================================================

# create new directed graph so its easier to analyze events. add dist parameter, which will be useful 
# to decide whether to extend segment

G2_dir = nx.DiGraph()

fin_seg_conns_forward = defaultdict(dict)
fin_seg_conns_backward = defaultdict(dict)
for t_node in G2.nodes():
    G2_dir.add_node(t_node, **dict(G2.nodes[t_node]))
    for t_targ in G2.neighbors(t_node):
        if G2.nodes[t_node]["t_start"] > G2.nodes[t_targ]["t_end"]:
            t_attr = {'dist': G2.nodes[t_node]["t_start"] - G2.nodes[t_targ]["t_end"]}
            G2_dir.add_edge(t_targ, t_node,  **t_attr)
        else:
            t_attr = {'dist': G2.nodes[t_targ]["t_start"] - G2.nodes[t_node]["t_end"]}
            G2_dir.add_edge(t_node, t_targ,  **t_attr)
    a = 1

# analyze state of each segments start & end. info is one-sided, does not represent true events
for t_node in G2_dir.nodes():
    t_succ = list(G2_dir.successors(t_node))
    if len(t_succ) == 0 :
        G2_dir.nodes[t_node]["to_state"] = 'end'
    elif len(t_succ) == 1: 
        G2_dir.nodes[t_node]["to_state"] = 'solo'
    else:
        G2_dir.nodes[t_node]["to_state"] = 'split'

    t_pred = list(G2_dir.predecessors(t_node))
    if len(t_pred) == 0 :
        G2_dir.nodes[t_node]["from_state"] = 'start'
    elif len(t_pred) == 1: 
        G2_dir.nodes[t_node]["from_state"] = 'solo'
    else:
        G2_dir.nodes[t_node]["from_state"] = 'merge'


# now we consider edges as parts of proper events. compare two ends and their view of events. still not competely true
for t_conn in G2_dir.edges():
    t_from, t_to = t_conn
    if G2_dir.nodes[t_from]["to_state"] == 'solo':
        if G2_dir.nodes[t_to]["from_state"] == 'solo':
            G2_dir.edges[t_conn]['event'] = 'solo'
        elif G2_dir.nodes[t_to]["from_state"] == 'merge':
            G2_dir.edges[t_conn]['event'] = 'merge'
        else: assert 1 == -1, 'not encountered'
    elif G2_dir.nodes[t_from]["to_state"] == 'split':
        if G2_dir.nodes[t_to]["from_state"] == 'solo':
            G2_dir.edges[t_conn]['event'] = 'split'
        elif G2_dir.nodes[t_to]["from_state"] == 'merge':
            G2_dir.edges[t_conn]['event'] = 'mixed'
        else: assert 1 == -1, 'not encountered'

# if one edge in event is mixed split-merge, all edges in event should be marked as mixed
t_events_mixed = [t_conn for t_conn in G2_dir.edges() if G2_dir.edges[t_conn]['event'] == 'mixed']
assert len(t_events_mixed) == 0, 'havent yet dealt with mixed events at this stage'
for t_conn in t_events_mixed:
    t_from, t_to = t_conn
    t_successors = list(G2_dir.successors(t_from))
    t_predecessors = list(G2_dir.predecessors(t_to))
    for t_succ in t_successors:
        G2_dir.edges[(t_from, t_succ)]['event'] = 'mixed'
    for t_pred in t_predecessors:
        G2_dir.edges[(t_pred, t_to  )]['event'] = 'mixed'

fin_events_dict = defaultdict(dict)


a = 1


# for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<







if 1 == 1:
    binarizedMaskArr = np.load(binarizedArrPath)['arr_0']
    imgs = [convertGray2RGB(binarizedMaskArr[k].copy()) for k in range(binarizedMaskArr.shape[0])]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7; thickness = 4;
    for n, case in tqdm(enumerate(t_segments_new)):
        case    = sorted(case, key=lambda x: x[0])
        for k,subCase in enumerate(case):
            t_time,*subIDs = subCase
            for subID in subIDs:
                cv2.drawContours(  imgs[t_time],   g0_contours[t_time], subID, cyclicColor(n), 2)
            
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t_subID] for t_subID in subIDs]))
            x,y,w,h = cv2.boundingRect(t_hull)
            
            cv2.drawContours(  imgs[t_time],  [t_hull], -1, cyclicColor(n), 2)
            [cv2.putText(imgs[t_time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        t_times_all = [t[0] for t in case]
        t_centroids_all = [G.nodes[t]["centroid"] for t in case]
        for k,subCase in enumerate(case):
            t_time,*subIDs = subCase
            t_k_from = max(0, k - 10)
            t_times_sub = t_times_all[t_k_from:k]
            pts = np.array(t_centroids_all[t_k_from:k]).reshape(-1, 1, 2).astype(int)
            
            cv2.polylines(imgs[t_time], [pts] ,0, (255,255,255), 3)
            cv2.polylines(imgs[t_time], [pts] ,0, cyclicColor(n), 2)
            [cv2.circle(imgs[t_time], tuple(p), 3, cyclicColor(n), -1) for [p] in pts]

            for subID in subIDs:
                startPos2 = g0_contours[t_time][subID][-30][0] 
                [cv2.putText(imgs[t_time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]

    for k,img in enumerate(imgs):
        folder = r"./post_tests/testImgs/"
        fileName = f"{str(k).zfill(4)}.png"
        cv2.imwrite(os.path.join(folder,fileName) ,img)
        #cv2.imshow('a',imgs[time])


#for_graph_plots(G, segs = t_segments_new)         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# for_graph_plots(G)
a = 1
#if useMeanWindow == 1:
#    meanImage = masksArr[whichMaskInterval(globalCounter,intervalIndecies)]

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()