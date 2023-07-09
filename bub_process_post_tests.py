import enum
import numpy as np, itertools, networkx as nx, sys
import cv2, os, glob, datetime, re, pickle#, multiprocessing
# import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
# import from custom sub-folders are defined bit lower
#from imageFunctionsP2 import (overlappingRotatedRectangles,graphUniqueComponents)
# functions below
if 1 == 1:
    def convertGray2RGB(image):
        if len(image.shape) == 2:
            return cv2.cvtColor(image.copy(),cv2.COLOR_GRAY2RGB)
        else:
            return image
    mapXY = (np.load('./mapx.npy'), np.load('./mapy.npy'))
    def undistort(image):
        return cv2.remap(image,mapXY[0],mapXY[1],cv2.INTER_LINEAR)


    def timeHMS():
        return datetime.datetime.now().strftime("%H-%M-%S")

    colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
    np.random.seed(2);np.random.shuffle(colorList);np.random.seed()

    def cyclicColor(index):
        return colorList[index % len(colorList)].tolist()


    def modBR(BR,side):
        x,y,w,h  = BR
        return [x - int(max((side-w)/2,0)), y - int(max((side-h)/2,0)), max(side,w), max(side,h)]

    def rotRect(rect):
        x,y,w,h = rect
        return (tuple((int(x+w/2),int(y+h/2))), tuple((int(w),int(h))), 0)

    def rect2contour(rect):
        x,y,w,h = rect
        return np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],int).reshape(-1,1,2)

    def compositeRectArea(rects, rectsParamsArr):
        if len(rects) > 1:
            # find region where all rectangles live. working smaller region should be faster.
            minX, minY = 100000, 100000                 # Initialize big low and work down to small
            maxX, maxY = 0, 0                           # Initialize small high  and work up to big

            for ID in rects:
                x,y,w,h = rectsParamsArr[ID]
                minX = min(minX,x)
                minY = min(minY,y)
                maxX = max(maxX,x+w)
                maxY = max(maxY,y+h)

            width = maxX - minX                         # Get composite width
            height = maxY - minY                        # Get composite height
            blank = np.zeros((height,width),np.uint8)   # create a smaller canvas
            for ID in rects:                            # draw masks with offset to composite edge
                x,y,w,h = rectsParamsArr[ID]
                cntr = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],int).reshape(-1,1,2)
                cv2.drawContours( blank, [cntr], -1, 255, -1, offset = (-minX,-minY))
         
            return int(np.count_nonzero(blank))         # count pixels = area

        else:
            x,y,w,h = rectsParamsArr[rects[0]]
            return int(w*h)
    
    def graph_extract_paths(H,f):
        nodeCopy = list(H.nodes()).copy()
        segments2 = {a:[] for a in nodeCopy}
        resolved = []
        skipped = []
        for node in nodeCopy:
            goForward = True if node not in resolved else False
            nextNode = node
            prevNode = None
            while goForward == True:
                neighbors = list(H.neighbors(nextNode))
                nextNodes = [a for a in neighbors if f(a) > f(nextNode)]
                prevNodes = [a for a in neighbors if f(a) < f(nextNode)]
                # find if next node exists and its single
                soloNext    = True if len(nextNodes) == 1 else False
                soloPrev    = True if len(prevNodes) == 1 else False # or prevNode is None
                soloPrev2   = True if soloPrev and (prevNode is None or prevNodes[0] == prevNode) else False

                # if looking one step ahead, starting node can have one back and/or forward connection to split/merge
                # this would still mean that its a chain and next/prev node will be included.
                # to fix this, check if next/prev are merges/splits
                # find if next is not merge:
                nextNotMerge = False
                if soloNext:
                    nextNeighbors = list(H.neighbors(nextNodes[0]))
                    nextPrevNodes = [a for a in nextNeighbors if f(a) < f(nextNodes[0])]
                    if len(nextPrevNodes) == 1: 
                        nextNotMerge = True

                nextNotSplit = False
                if soloNext:
                    nextNeighbors = list(H.neighbors(nextNodes[0]))
                    nextNextNodes = [a for a in nextNeighbors if f(a) > f(nextNodes[0])]
                    if len(nextNextNodes) <= 1:   # if it ends, it does not split. (len = 0)
                        nextNotSplit = True

                prevNotSplit = False
                if soloPrev2:
                    prevNeighbors = list(H.neighbors(prevNodes[0]))
                    prevNextNodes = [a for a in prevNeighbors if f(a) > f(prevNodes[0])]
                    if len(prevNextNodes) == 1:
                        prevNotSplit = True


                saveNode = False
                # test if it is a chain start point:
                # if prev is a split, implies only one prevNode 
                if prevNode is None:                # starting node
                    if len(prevNodes) == 0:         # if no previos node =  possible chain start
                        if nextNotMerge:            # if it does not change into merge, it is good
                            saveNode = True
                        else:
                            skipped.append(node)
                            goForward = False
                    elif not prevNotSplit:
                        if nextNotMerge:
                            saveNode = True
                        else: 
                            skipped.append(node)
                            goForward = False
                    else:
                        skipped.append(node)
                        goForward = False
                else:
                # check if its an endpoint
                    # dead end = zero forward neigbors
                    if len(nextNodes) == 0:
                        saveNode = True
                        goForward = False
                    # end of chain =  merge of forward neigbor
                    elif not nextNotMerge:
                        saveNode = True
                        goForward = False
                    elif not nextNotSplit:
                        saveNode = True
                
                    # check if it is part of a chain
                    elif nextNotMerge:
                        saveNode = True


                if saveNode:
                    segments2[node].append(nextNode)
                    resolved.append(nextNode)
                    prevNode = nextNode
                    if goForward :
                        nextNode = nextNodes[0]

    
    
        return segments2, skipped
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
from graph_visualisation_01 import (draw_graph_with_height)


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
        cv2.imwrite(cropMaskPath, convertGray2RGB(undistort(cv2.imread(imageLinks[0],0))))
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

    numImages = min(dataNum,len(imageLinks)-startFrom)
    if rotateImageBy % 2 == 0 and rotateImageBy != -1: W,H = H,W                       # for cv2.XXX rotation commands
    dataArchive = np.zeros((numImages,H,W),np.uint8)
    for i,j in tqdm(enumerate(range(startFrom-1, numImages))):
        if rotateImageBy != -1:
            dataArchive[i]    = cv2.rotate(undistort(cv2.imread (imageLinks[j],0))[Y:Y+H, X:X+W],rotateImageBy)
        else:
            dataArchive[i]    = undistort(cv2.imread (imageLinks[j],0))[Y:Y+H, X:X+W]
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
    g0_bigBoundingRect   = {t:[] for t in range(binarizedMaskArr.shape[0])}
    g0_bigBoundingRect2  = {t:{} for t in range(binarizedMaskArr.shape[0])}
    g0_clusters          = {t:[] for t in range(binarizedMaskArr.shape[0])}
    g0_clusters2         = {t:[] for t in range(binarizedMaskArr.shape[0])}
    g0_contours          = {t:[] for t in range(binarizedMaskArr.shape[0])}
    g0_contours_children = {t:{} for t in range(binarizedMaskArr.shape[0])}
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
    H.add_nodes_from(allIDs)
    H.add_edges_from(g0_pairConnections)
    connected_components_all = [list(nx.node_connected_component(H, key)) for key in allIDs]
    connected_components_all = [sorted(sub, key=lambda x: (x[0], x[1])) for sub in connected_components_all]
    # extract connected families
    connected_components_unique = []
    [connected_components_unique.append(x) for x in connected_components_all if x not in connected_components_unique]
    
    if 1 == 1:
        storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
        with open(storeDir, 'wb') as handle:
            pickle.dump(
            [
                g0_bigBoundingRect2, g0_clusters2, g0_contours, g0_pairConnections, H, connected_components_unique,
                g0_contours_children
            ], handle) 






print(f"\n{timeHMS()}: Begin loading intermediate data...")
storeDir = os.path.join(stagesFolder, "intermediateData.pickle")
with open(storeDir, 'rb') as handle:
    [
        g0_bigBoundingRect2, g0_clusters2, g0_contours, g0_pairConnections, H, connected_components_unique,
        g0_contours_children
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
        folder = r"./post_tests/testImgs/"
        fileName = f"{str(k).zfill(4)}.png"
        cv2.imwrite(os.path.join(folder,fileName) ,img)
        #cv2.imshow('a',img)



# analyze single strand
doX = 60#84
test = connected_components_unique[doX]
# find times where multiple elements are connected (split/merge)
def getNodePos2(dic0, S = 20):
    dups, cnts = np.unique([a for a in dic0.values()], return_counts = True)
    # relate times to 'local IDs', which are also y-positions or y-indexes
    dic = {a:np.arange(b) for a,b in zip(dups,cnts)} # each time -> arange(numDups)
    # give duplicates different y-offset 0,1,2,..
    dic2 = {t:{s:k for s,k in zip(c,[tID for tID, t_time in dic0.items() if t_time == t])} for t,c in dic.items()}
    node_positions = {}
    # offset scaled by S
    #S = 20
    for t,c in dic.items():
        # scale and later offset y-pos by mid-value
        d = c*S
        meanD = np.mean(d)
        for c2,key in dic2[t].items():
            if len(dic2[t]) == 1:
                dy = np.random.randint(low=-3, high=3)
            else: dy = 0
            # form dict in form key: position. time is x-pos. y is modified by order.
            node_positions[key] = (t,c2*S-meanD + dy)

    return node_positions
def getNodePos(test):
    dups, cnts = np.unique([a[0] for a in test], return_counts = True)
    # relate times to 'local IDs', which are also y-positions or y-indexes
    dic = {a:np.arange(b) for a,b in zip(dups,cnts)}
    # give duplicates different y-offset 0,1,2,..
    dic2 = {t:{s:k for s,k in zip(c,[a for a in test if a[0] == t])} for t,c in dic.items()}
    node_positions = {}
    # offset scaled by S
    S = 20
    for t,c in dic.items():
        # scale and later offset y-pos by mid-value
        d = c*S
        meanD = np.mean(d)
        for c2,key in dic2[t].items():
            # form dict in form key: position. time is x-pos. y is modified by order.
            node_positions[key] = (t,c2*S-meanD)
    return node_positions
node_positions = getNodePos(test)
allNodes = list(H.nodes())
removeNodes = list(range(len(connected_components_unique)))
[allNodes.remove(x) for x in test]

for x in allNodes:
    H.remove_node(x)
#print(list(H.nodes()))




#ax.set_aspect('equal')

#plt.show()
f = lambda x : x[0]

segments2, skipped = graph_extract_paths(H,f) # 23/06/23 info in "extract paths from graphs.py"

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
paths = {i:vals for i,vals in enumerate(segments2)}

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

#lessRoughBRs = {}
#for time, subIDs in lessRoughIDs.items():
#    for ID in subIDs:
#        lessRoughBRs[tuple([time, ID])] = cv2.boundingRect(g0_contours[time][ID])

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






a = 1
#bigBoundingRect     = [cv2.boundingRect(np.vstack([rect2contour(brectDict[k]) for k in comb])) for comb in cc_unique]

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7; thickness = 4;


def drawH(H, paths, node_positions):
    colors = {i:np.array(cyclicColor(i))/255 for i in paths}
    colors = {i:np.array([R,G,B]) for i,[B,G,R] in colors.items()}
    colors_edges2 = {}
    width2 = {}
    # set colors to different chains. iteratively sets default color until finds match. slow but whatever
    for u, v in H.edges():
        for i, path in paths.items():
            if u in path and v in path:

                colors_edges2[(u,v)] = colors[i]
                width2[(u,v)] = 5
                break
            else:

                colors_edges2[(u,v)] = np.array((0,0,0))
                width2[(u,v)] = 2

    nx.set_node_attributes(H, node_positions, 'pos')

    pos = nx.get_node_attributes(H, 'pos')

    fig, ax = plt.subplots(figsize=( 10,5))
    nx.draw(H, pos, with_labels=True, node_size=50, node_color='lightblue',font_size=6,
            font_color='black', edge_color=list(colors_edges2.values()), width = list(width2.values()))
    plt.show()

#drawH(H, paths, node_positions)

def extractNeighborsNext(graph, node, time_from_node_function):
    neighbors = list(graph.neighbors(node))
    return [n for n in neighbors if time_from_node_function(n) > time_from_node_function(node)]
def extractNeighborsPrevious(graph, node, time_from_node_function):
    neighbors = list(graph.neighbors(node))
    return [n for n in neighbors if time_from_node_function(n) < time_from_node_function(node)]
# ===============================================================================================
# ===============================================================================================
# =========== Extract solo-to-solo bubble trajectories from less rough graphs ===================
# ===============================================================================================
# ===============================================================================================

allIDs = sum([list(a.keys()) for a in lessRoughBRs.values()],[])
allIDs = sorted(allIDs, key=lambda x: (x[0], x[1]))
G = nx.Graph()
G.add_nodes_from(allIDs)
G.add_edges_from(g0_pairConnections2)
node_positions = getNodePos(allIDs)

f = lambda x : x[0]

segments2, skipped = graph_extract_paths(G,f) # 23/06/23 info in "extract paths from graphs.py"

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
paths = {i:vals for i,vals in enumerate(segments2)}

lr_allNodesSegm = sum(segments2,[])
lr_missingNodes = [node for node in allIDs if node not in lr_allNodesSegm] # !! may be same as &skipped !!
assert set(skipped)==set(lr_missingNodes), "set(skipped) is not same as set(lr_missingNodes)"

# ===============================================================================================
# ===============================================================================================
# === find POSSIBLE interval start-end connectedness: start-end exist within set time interval ==
# ===============================================================================================
# ===============================================================================================

# get start and end of good segments
lr_start_end    = [[segm[0],segm[-1]] for segm in segments2]
lr_all_start    = np.array([a[0][0] for a in lr_start_end])
lr_all_end      = np.array([a[1][0] for a in lr_start_end])

# get nodes that begin not as a part of a segment = node with no prior neigbors
lr_nodes_other = []
lr_nodes_solo = []
for node in lr_missingNodes:
    neighbors = list(G.neighbors(node))
    oldNeigbors = [n for n in neighbors if n[0] < node[0]]
    newNeigbors = [n for n in neighbors if n[0] > node[0]]
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
# ===============================================================================================
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
        activeNodes = [node for node in allIDs if startTime <= node[0] <= endTime]
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
        activeNodes = [node for node in allIDs if startTime <= node[0] <= endTime]
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
lr_trusted_segment_length = 7
lr_trusted_segments_by_length = []
lr_segments_lengths = {}

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

# Iterate over the dictionary and add edges with weights
for (node1, node2), weight in lr_intervals_lengths.items():
    G2.add_edge(node1, node2, weight=weight)

#pos = nx.spring_layout(G2)
#edge_widths = [data['weight'] for _, _, data in G2.edges(data=True)]
#nx.draw(G2, pos, with_labels=True, width=edge_widths)

#plt.show()
t_nodes = sorted(list(G2.nodes()))
t_neighbor_sol_all_prev = {tID:{} for tID in t_nodes}
t_neighbor_sol_all_next = {tID:{} for tID in t_nodes}
for node in t_nodes:
    # Get the neighboring nodes
    # segments2[node]; segments2[t_neighbor]
    t_neighbors = list(G2.neighbors(node))
    t_time_start    = segments2[node][0][0]
    t_time_end      = segments2[node][-1][0]
    t_neighbors_prev = []
    t_neighbors_next = []
    for t_neighbor in t_neighbors:
        t_time_neighbor_start    = segments2[t_neighbor][0][0]
        t_time_neighbor_end      = segments2[t_neighbor][-1][0]
        if t_time_start < t_time_neighbor_start and t_time_end < t_time_neighbor_start:
            t_neighbors_next.append(t_neighbor)
        elif t_time_start > t_time_neighbor_end and t_time_end > t_time_neighbor_end:
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
    #t_neighbor_sol = {}
    
    if len(t_neighbors_weights_prev)>0:
        # neg weights, so max, get time of nearset branch t0. get all connections within [t0 - 2, t0] in case of split
        t_val_min = max(t_neighbors_weights_prev.values())
        t_key_min_main = max(t_neighbors_weights_prev, key = t_neighbors_weights_prev.get)
        t_key_main_ref_time = t_neighbors_time_prev[t_key_min_main]
        t_sol = [key for key,t in t_neighbors_time_prev.items() if t_key_main_ref_time - 2 <= t <=  t_key_main_ref_time]
        #t_sol = [key for key, value in t_neighbors_weights_prev.items() if t_val_min - 1 <= value <=  t_val_min]
        for t_node in t_sol:t_neighbor_sol_all_prev[node][t_node] = t_neighbors_weights_prev[t_node]
        
    if len(t_neighbors_weights_next)>0:
        t_val_min = min(t_neighbors_weights_next.values())
        t_key_min_main = min(t_neighbors_weights_next, key = t_neighbors_weights_next.get)
        t_key_main_ref_time = t_neighbors_time_next[t_key_min_main]
        t_sol = [key for key,t in t_neighbors_time_next.items() if t_key_main_ref_time <= t <=  t_key_main_ref_time + 2]
        #t_sol = [key for key, value in t_neighbors_weights_next.items() if t_val_min +1 >= value >= t_val_min]
        for t_node in t_sol: t_neighbor_sol_all_next[node][t_node] = t_neighbors_weights_next[t_node]
    #t_neighbors_prev = extractNeighborsPrevious(G2, node,    lambda x: x[0])
    #t_neighbors_next = extractNeighborsNext(    G2, node,    lambda x: x[0])
    # Iterate through the neighboring nodes
    a = 1

# prepare plot for segment interconnectedness
G2.remove_edges_from(list(G2.edges()))
for node, t_conn in t_neighbor_sol_all_prev.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2, weight=1/np.abs(weight))
for node, t_conn in t_neighbor_sol_all_next.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2, weight=1/weight)
    
node_positions2 = {}
t_segments_times = {}
labels = {}
for t,segment in enumerate(segments2):
    t_times = [a[0] for a in segment]
    t_segments_times[t] = int(np.mean(t_times))
    labels[t] = f'{t}_{segment[0]}'
node_positions2 = getNodePos2(t_segments_times, S = 1)


for g in G2.nodes():
  G2.nodes()[g]["height"] = node_positions2[g][0]
#draw_graph_with_height(G2,figsize=(5,5), labels=labels)


#pos = nx.spring_layout(G2, pos = node_positions2, k = 1, iterations = 10)

#nx.draw(G2, pos=node_positions2, labels=labels)
#plt.show()

# find segments that connect only to one neighbor segment both directions
# from node "t_node" to node "t_neighbor"
t_conn_one_to_one = []
for t_node in G2.nodes():
    t_node_neighbors = list(G2.neighbors(t_node))
    t_node_time_end      = segments2[t_node][-1][0]
    t_node_neighbors_next = []
    for t_neighbor in t_node_neighbors:
        t_neighbor_time_start    = segments2[t_neighbor][0][0]
        if t_node_time_end < t_neighbor_time_start:
            t_node_neighbors_next.append(t_neighbor)
    if len(t_node_neighbors_next) == 1:
        t_neighbor              = t_node_neighbors_next[0]
        t_neighbor_time_start   = segments2[t_neighbor][0][0]
        t_neighbor_neighbors   = list(G2.neighbors(t_neighbor))
        
        t_neighbor_neighbors_back = []
        for t_neighbor_neighbor in t_neighbor_neighbors:
            t_neighbor_neighbor_time_end      = segments2[t_neighbor_neighbor][-1][0]
            if t_neighbor_time_start > t_neighbor_neighbor_time_end:
                t_neighbor_neighbors_back.append(t_neighbor)
        if len(t_neighbor_neighbors_back) == 1:
            t_conn_one_to_one.append(tuple([t_node,t_neighbor]))
    
        
t_conn_one_to_one_test = [(segments2[start][-1][0],segments2[end][0][0]) for start,end in t_conn_one_to_one]
a = 1
# Draw the graph with constrained x-positions and automatic y-positions
#nx.draw(G, pos, with_labels=True)

##drawH(G, paths, node_positions)

# check if extracted intervals are "clean", dont contain merge
# clean means there are nodes attached to all available paths
# => check neighbors of all nodes in path and see if they are their own neigbors
# this test is too strict, but allows to isolate pseudo split-merges 
t_sols = []
for t_conn in lr_trusted_interval_test_prio:
    a = lr_paths_segm2[t_conn]
    t_from,t_to = t_conn
    t_node_from_last   = segments2[t_from][-1]
    t_node_to_first    = segments2[t_to][0]
    t_neighbors_from_next       = extractNeighborsNext(     G, t_node_from_last,    lambda x: x[0])
    t_neighbors_to_previous     = extractNeighborsPrevious( G, t_node_to_first,     lambda x: x[0])

    t_all_path_nodes = sorted(set(sum(lr_paths_segm[t_conn],[])),key=lambda x: x[0])
    
    t_all_path_nodes = t_all_path_nodes
    t_all_path_neigbors = []
    for t_node in t_all_path_nodes[1:-1]:
        t_all_path_neigbors.append(list(G.neighbors(t_node)))

    t_all_path_neigbors.append(t_neighbors_from_next)
    t_all_path_neigbors.append(t_neighbors_to_previous)

    t_all_path_neigbors = sorted(set(sum(t_all_path_neigbors,[])),key=lambda x: x[0])
    #t_all_path_neigbors = [t_node for t_node in t_all_path_neigbors if t_node not in [t_node_from_last,t_node_to_first]]
    if t_all_path_neigbors != t_all_path_nodes: continue
    t_sols.append(t_conn)

    #t_nodes_all = sum(t_hist_pre.values(),[]) + sum(t_hist_post.values(),[]) +sum(t_paths_choices,[])
    #t_nodes_all = sorted(set(t_nodes_all))
    #t_times_all = [a[0] for a in t_nodes_all]
    #t_times_cIDs = {t:[] for t in t_times_all}
    #for t,cID in t_nodes_all:
    #    t_times_cIDs[t].append(cID)
    #for t in t_times_cIDs:
    #    t_times_cIDs[t] = list(sorted(np.unique(t_times_cIDs[t])))
a = 1
#drawH(G, paths, node_positions)
# ===============================================================================================
# ===============================================================================================
# === find split-merge (possibly fake) events; detect split-merges that look like real merges ===
# ===============================================================================================
# ===============================================================================================

# since nodes and edges represent spacial overlap of cluster elements at each time paths between two nodes
# should (?) contain all elements for permutations for bubble reconstruction from partial reflections

# !!!!!!!!!!!!!!====================<<<<<<<<<<<<<<<
# best case scenario of a path is overlap between single contour bubbles.
# in that case solution is shortest trajectory :)
# might want to check for area befor and after, in case of merge. overall, i have to do permutations.
# elements that begin from nothing are problematic. in case split-merge, i should include them into paths, but in reverse.
# same with split-merge that happend 2+ steps. means bubble splits into two segments and is reconstructed.
# but also might split into one segment, and other starts from nothing. so merge them in reverse path.

# 1) check if multiple segments are merged into 1. might be merge, might be split-merge fake or not. area will help.
# take endpoints of all paths
lr_paths_endpoints = {**{key:a[0][-1] for key,a in lr_paths_segm.items()},
                      **{key:a[0][-1] for key,a in lr_paths_other.items()}}

# find common endpoints. inverse dictionary = endpoint:connections
rev_multidict = {}
for key, value in lr_paths_endpoints.items():
    rev_multidict.setdefault(value, set()).add(key)
# if endpoint is associated with multiple connections, extract these connections
lr_multi_conn = [list(a) for a in rev_multidict.values() if len(a)>1]
lr_multi_conn_all = sum(lr_multi_conn,[])
# decide which merge branch is main. expore every merge path
# path length is not correct metric, since it may be continuation via merge
lr_multi_conn_win = []
for t_merges_options in lr_multi_conn:
    # grab time at which segments merge
    t_merge_time = segments2[int(t_merges_options[0][1])][0][0]
    # initialize times for paths which will be compared for 'path length/age' and if set time is legit or intermediate
    t_earliest_time     = {t_ID:0 for t_ID in t_merges_options}
    t_terminated_time   = {t_ID:0 for t_ID in t_merges_options}
    for t_path in t_merges_options:
        # t_path is ID in segments2, take its starting node.
        # lr_paths_other has no history (no prior neighbors)
        if type(t_path[0]) == str:
            t_earliest_time[t_path] = t_merge_time - 1
            t_terminated_time[t_path] = 1
            continue
        t_nodes_path    = segments2[t_path[0]]                       

        t_node_first    = t_nodes_path[0]
        t_node_last     = t_nodes_path[-1]
        t_prev_neighbors = [node for node in list(G.neighbors(t_node_first)) if node[0] < t_node_first[0]]
        if len(t_prev_neighbors) == 0:
            t_earliest_time[t_path] = t_node_first[0]
            t_terminated_time[t_path] = 1
    # check if we can drop terminated paths. if path is terminated and time is bigger than any unterminated, drop it.
    for t_path in t_merges_options:
        t_other_times = [t for tID,t in t_earliest_time.items() if tID != t_path]
        if t_terminated_time[t_path] == 1 and t_earliest_time[t_path] > min(t_other_times):
            t_earliest_time.pop(t_path,None)
            t_terminated_time.pop(t_path,None)

    t_IDs_left = list(t_earliest_time.keys())
    if len(t_IDs_left) == 1: lr_multi_conn_win.append(t_IDs_left[0])
    else: assert 1 == -1, "split-merge. need to develop dominant branch further"

# ===============================================================================================
# ===============================================================================================
# == find INTeRmediate interval paths, short PREv and POST history. deal with merge hist after ==
# ===============================================================================================
# ===============================================================================================
lr_multi_conn_pre   = {conn:{} for conn in lr_paths_segm.keys()}
lr_multi_conn_intr  = {conn:{} for conn in lr_paths_segm.keys()} 
lr_multi_conn_post  = {conn:{} for conn in lr_paths_segm.keys()} 
for t_conn, t_paths_nodes in lr_paths_segm.items():
    tID_from = t_conn[0]
    tID_to  = t_conn[1]
    # if multi merge
    t_is_part_of_merge = True if t_conn in lr_multi_conn_all else False
    t_is_part_of_merge_and_main = True if t_is_part_of_merge and t_conn in lr_multi_conn_win else False
    t_is_followed_by_merge = True if tID_to in [a[0] for a in lr_multi_conn_all] else False
    t_other_paths_nodes = []
    t_other_path_conn   = []
    if t_is_part_of_merge_and_main:
        # extract other (pseudo) merging segments. other paths only.
        t_other_path_conn = [a for a in lr_multi_conn if t_conn in a]
        t_other_path_conn = [[a for a in b if a != t_conn] for b in t_other_path_conn]
        #[t_other_path_conn[k].remove(t_conn) for k in range(len(t_other_path_conn))]
        t_other_path_conn = sum(t_other_path_conn,[])
        # preprend prev history of segment before intermediate segment
        #t_other_path_pre = {ID:[] for ID in t_other_path_conn}
        #for ID in t_other_path_conn:
        #    if type(ID[0]) != str:
        #        t_other_path_pre[ID] += segments2[ID[0]][:-1]
        # some mumbo-jumbo with string keys to differentiate lr_paths_other from lr_paths_segm
        t_other_paths_nodes = []
        for ID in t_other_path_conn:
            if type(ID[0]) == str:
                for t_path in lr_paths_other[ID]:
                    t_other_paths_nodes.append(t_path)  
            else:
                for t_path in lr_paths_segm[ID]:
                    t_other_paths_nodes.append(t_path)
            #if type(ID[0]) == str:
            #    for t_path in lr_paths_other[ID]:
            #        t_other_paths_nodes.append(t_other_path_pre[ID]+t_path)  
            #else:
            #    for t_path in lr_paths_segm[ID]:
            #        t_other_paths_nodes.append(t_other_path_pre[ID]+t_path)
        a = 1            
        #t_other_paths_nodes = [t_other_path_pre[ID] + lr_paths_other[ID] 
        #                       if type(ID[0]) == str 
        #                       else lr_paths_segm[ID] for ID in t_other_path_conn]
        #t_other_paths_nodes = sum(t_other_paths_nodes,[])
        #t_join_paths = t_paths_nodes + t_other_paths_nodes
        # there is no prev history for lr_paths_other, lr_paths_segm has at least 2 nodes
    elif t_is_part_of_merge:
        continue
    else:
        t_other_paths_nodes = []
    # app- and pre- pend path segments to intermediate area. 
    # case where split-merge is folled by disconnected split-merge (on of elements is not connected via overlay)
    # is problematic. 
    # >>>>>>>>>>>>>>>>!!!!!!!!!!!!!!!!!!!!!!!!!<<<<<<<<<<<<<<
    # i should split this code into two. one stablishes intermediate part
    # and other can relate intermediate parts together, if needed.

    # ====================
    # store history prior to intermediate segment. solo bubble = 1 contour = last N steps
    # pseudo merge = multiple branches => include branches and take N steps prior
    a = 1
    t_og_max_t =  segments2[tID_from][-1][0]
    if not t_is_part_of_merge:
        lr_multi_conn_pre[t_conn][tID_from] = segments2[tID_from][-4:-1]
    else:
        # find minimal time from othat branches
        t_other_min_t = min([a[0] for a in sum(t_other_paths_nodes,[])])
        # extract OG history only 3 steps prior to first branch (timewise) <<< 3 or so! idk, have to test
        lr_multi_conn_pre[t_conn][tID_from] = [a for a in segments2[tID_from] if t_og_max_t> a[0] > t_other_min_t - 4]
        t_other_IDs = [a[0] for a in t_other_path_conn]
        for tID in t_other_IDs:
            if type(tID) != str:
                lr_multi_conn_pre[t_conn][tID] = segments2[tID]
    # store history post intermediate segment.     solo bubble = 1 contour = next N steps
    # if followed by merge, ideally take solo N steps before merge. but this info not avalable yet. do it after
    if not t_is_followed_by_merge:
        lr_multi_conn_post[t_conn][tID_to]  = segments2[tID_to][1:4]

    t_join_paths            = t_paths_nodes + t_other_paths_nodes
    # store all paths in intermediate interval
    lr_multi_conn_intr[t_conn]   = t_paths_nodes + t_other_paths_nodes 
    # collect all viable contour IDs from intermediate time
    #t_all_nodes = sum(t_join_paths,[])
    #t_all_times = list(sorted(np.unique([a[0] for a in t_all_nodes])))
    #t_times_cIDs = {t:[] for t in t_all_times}
    #for t,cID in t_all_nodes:
    #    t_times_cIDs[t].append(cID)
    #for t in t_times_cIDs:
    #    t_times_cIDs[t] = list(sorted(np.unique(t_times_cIDs[t])))
    #lr_multi_conn_choices[t_conn] = t_times_cIDs
    # test intermediate contour combinations
    # 
#lr_multi_conn_all and t_conn in lr_multi_conn_win

# ===============================================================================================
# ===============================================================================================
# ========================== deal with POST history if its a merge ==============================
# ===============================================================================================
# ===============================================================================================
# isolate easy cases where split-merge has history on both sides
t_all_merging = [a[0] for a in lr_multi_conn_win]
for t_conn, t_paths_choices in lr_multi_conn_post.items():
    t_end_ID = t_conn[1]
    if len(t_paths_choices) == 0 and t_end_ID in t_all_merging:
        t_next_merge_all = [a for a in lr_multi_conn_win if a[0] == t_end_ID]
        assert len(t_next_merge_all)<2, "post history-> multi merge. something i did not account for"
        t_conn_next_merge    = t_next_merge_all[0]
        t_og_min_t      = segments2[t_end_ID][0][0]
        # pick min branches time: take all branches except main, extract first node, and extract from it time. pick min
        t_other_min_t   = min([a[0][0] for ID,a in lr_multi_conn_pre[t_conn_next_merge].items() if ID != t_end_ID])

        # end history at: if there is time before merge take 3 (or so) steps before it.
        # it its earlier, squeeze interval. if merge is straight after, post history will be zero.
        t_max_time =  min(t_og_min_t + 4, t_other_min_t )
        lr_multi_conn_post[t_conn][t_end_ID] = [a for a in segments2[t_end_ID] if t_og_min_t < a[0] < t_max_time]
#drawH(G, paths, node_positions)
# ===============================================================================================
# ===============================================================================================
# ============== form combinations of elements from PRE-INTR-Post connectedness data ============
# ===============================================================================================
# ===============================================================================================
lr_multi_conn_choices = {conn:{} for conn in lr_paths_segm.keys()}

for t_conn, t_paths_choices in lr_multi_conn_intr.items():
    t_hist_pre  = lr_multi_conn_pre[t_conn]
    t_hist_post = lr_multi_conn_post[t_conn]
    t_nodes_all = sum(t_hist_pre.values(),[]) + sum(t_hist_post.values(),[]) +sum(t_paths_choices,[])
    t_nodes_all = sorted(set(t_nodes_all))
    t_times_all = [a[0] for a in t_nodes_all]
    t_times_cIDs = {t:[] for t in t_times_all}
    for t,cID in t_nodes_all:
        t_times_cIDs[t].append(cID)
    for t in t_times_cIDs:
        t_times_cIDs[t] = list(sorted(np.unique(t_times_cIDs[t])))
    lr_multi_conn_choices[t_conn] = t_times_cIDs

for tID,test in lr_multi_conn_choices.items():
    if len(test)==0: continue
    #test    = {325: [2], 326: [1], 327: [3], 328: [2], 329: [1, 3], 330: [2], 331: [1], 332: [1], 333: [1]}
    yoo     = {t:[] for t in test}
    for t,subIDs in test.items():
        s = sum([list(itertools.combinations(subIDs, r)) for r in range(1,len(subIDs)+1)],[])
        yoo[t] += s
        a = 1
    yoo2 = {t:[] for t in test}
    for t,permList in yoo.items(): 
        for perm in permList:
            hull = cv2.convexHull(np.vstack([g0_contours[t][c] for c in perm]))
            yoo2[t].append(cv2.contourArea(hull))
            #yoo2[t].append(cv2.boundingRect(np.vstack([g0_contours[t][c] for c in perm])) )

    #rectAreas = {t:[] for t in test}
    #maxArea = 0
    #minArea = 9999999999
    #for t,rectList in yoo2.items(): 
    #    for x,y,w,h in rectList:
    #        area = w*h
    #        rectAreas[t].append(area)
    #        maxArea = max(maxArea,area)
    #        minArea = min(minArea,area)
    rectAreas = yoo2
    minTime,maxTime = min(rectAreas), max(rectAreas)

    soloTimes = [a for a,b in rectAreas.items() if len(b) == 1]
    missingTimes = [a for a,b in rectAreas.items() if len(b) > 1]
    from scipy.interpolate import splev, splrep
    x = soloTimes
    y0 = np.array([b[0] for a,b in  rectAreas.items() if a in soloTimes])
    minArea,maxArea = min(y0), max(y0)
    K = maxTime - minTime
    y = (y0 - minArea) * (K / (maxArea - minArea))
    spl = splrep(soloTimes, y, k = 1, s = 0.7)

    x2 = np.linspace(min(rectAreas), max(rectAreas), 30)
    y2 = splev(x2, spl)
    y2 = y2/(K / (maxArea - minArea)) + minArea
    y = y/(K / (maxArea - minArea)) + minArea
    fig = plt.figure()
    for t in missingTimes:
        for rA in rectAreas[t]:
            plt.scatter([t],[rA], c='b')
    plt.plot(x, y, 'o', x2, y2)
    fig.suptitle(lr_multi_conn_intr[tID][0])
    #plt.show()

# >>>>>>>>>>>>> !!!!!!!!!!!!!!!!!!!! <<<<<<<<<<<<< CONCLUSIONS
# long trends can be trusted: e.g long solo- split-merge - long solo
# transition to short segments may be misleading e.g split into two parts perist - area is halfed at split
# fast split-merge kind of reduces trust, which is related to segment length. but it should not.
# ^ may be longterm trends should be analyzed first. idk what to do about real merges then.

    a = 1



# usefulPoints = startTime
## interval gets capped at interpolatinIntevalLength if there are more points
#interval    = min(interpolatinIntevalLength,usefulPoints)
## maybe max not needed, since interval is capped
#startPoint  = max(0,startTime-interval) 
#endPoint    = min(startTime,startPoint + interval) # not sure about interval or interpolatinIntevalLength
##
#x,y = trajectory[startPoint:endPoint].T
#t0 = np.arange(startPoint,endPoint,1)
## interpolate and exterpolate to t2 this window
#spline, _ = interpolate.splprep([x, y], u=t0, s=10000,k=1)
#t2 = np.arange(startPoint,endPoint+1,1)
#IEpolation = np.array(interpolate.splev(t2, spline,ext=0))
# >>>>>>>>> !!!!!!!! <<<<<<< most works. EXCEPT split into isolated node. segment merge into non-isolated nodes


#TODO: do a reconstruction via area (maybe trajectory) of BRs using lr_multi_conn_choices


#drawH(G, paths, node_positions)

# analyze fake split-merge caused by partial reflections. during period of time bubble is represented by reflections on its opposite side
# which might appear as a split. 

# if bubble is well behavade (good segment), it will be solo contour, then split and merge in 1 or more steps.

# find end points of rough good segments


# ======================
if 1 == -1:
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
            x,y,w,h = lessRoughBRs[time][subCase]
            #x,y,w,h = g0_bigBoundingRect[time][ID]
            cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
            [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                startPos2 = g0_contours[time][subID][-30][0] 
                [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        
    #cv2.imshow('a',imgs[time])
    for k,img in enumerate(imgs):
        if k in activeTimes:
            folder = r"./post_tests/testImgs2/"
            fileName = f"{str(k).zfill(4)}.png"
            cv2.imwrite(os.path.join(folder,fileName) ,img)
            #cv2.imshow('a',img)

a = 1
#cv2.imshow('a', blank0)
def centroid_area(contour):
    m = cv2.moments(contour)
    area = int(m['m00'])
    cx0, cy0 = m['m10'], m['m01']
    centroid = np.array([cx0,cy0])/area
    return  centroid, area

from scipy import interpolate

def exterpTest(trajectory,startTime,interpolatinIntevalLength, debug = 0 ,axes = 0, title = 'title', aspect = 'equal'):
    usefulPoints = startTime
    # interval gets capped at interpolatinIntevalLength if there are more points
    interval    = min(interpolatinIntevalLength,usefulPoints)
    # maybe max not needed, since interval is capped
    startPoint  = max(0,startTime-interval) 
    endPoint    = min(startTime,startPoint + interval) # not sure about interval or interpolatinIntevalLength
#
    x,y = trajectory[startPoint:endPoint].T
    t0 = np.arange(startPoint,endPoint,1)
    # interpolate and exterpolate to t2 this window
    spline, _ = interpolate.splprep([x, y], u=t0, s=10000,k=1)
    t2 = np.arange(startPoint,endPoint+1,1)
    IEpolation = np.array(interpolate.splev(t2, spline,ext=0))
    if debug == 1:
        if axes == 0:
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes.plot(trajectory[:,0],trajectory[:,1], '-o')
        axes.plot(*IEpolation, linestyle='dotted', c='orange')
        axes.plot(*IEpolation[:,-1], '--o', c='orange')
        axes.plot([IEpolation[0,-1],trajectory[endPoint,0]],[IEpolation[1,-1],trajectory[endPoint,1]], '-',c='black')
        axes.set_aspect(aspect)
        axes.set_title(title)
        if 1 == -1:
            # determin unit vector in interpolation direction
            startEnd = np.array([IEpolation[:,0],IEpolation[:,-1]])
            IEdirection = np.diff(startEnd, axis = 0)[0]
            IEdirection = IEdirection/np.linalg.norm(IEdirection)
            # measure latest displacement projection to unit vector
            displ = np.diff(trajectory[endPoint-1:endPoint+1],axis = 0)[0]
            proj = np.dot(displ,IEdirection)
            start1 = trajectory[endPoint - 1]
            end1 = start1 + proj*IEdirection
            arr = np.array([start1,end1])
            axes.plot(*arr.T, '--', c='red')
            #plt.show()

    return IEpolation, endPoint

#fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
# === lets check if segments are single bubbles = centroid and areas are consitent ===

pathsDeltas = {k:{} for k in paths}   
areaDeltas = {k:{} for k in paths}  
interpolatinIntevalLength = 3 # from 2 to this
pathCentroids = {k:[] for k in paths}     
overEstimateConnections = []
for k,pathX in paths.items():
    #pathX = paths[0]
    timesReal   = [time             for time,*subIDs in pathX]
    contourIDs  = {time:subIDs      for time,*subIDs in pathX}
    numContours = {time:len(subIDs) for time,*subIDs in pathX}
    soloBubble = True if all([1 if i == 1 else 0 for i in numContours.values()]) else False
    if soloBubble:
        pathXHulls  = [cv2.convexHull(np.vstack([g0_contours[time][subID] for subID in subIDs])) for time,*subIDs in pathX]
    
        pathXHCentroidsAreas = [centroid_area(contour) for contour in pathXHulls]
        trajectory = np.array([a[0] for a in pathXHCentroidsAreas])
        pathCentroids[k] = trajectory

        areas = np.array([a[1] for a in pathXHCentroidsAreas])
        # rescale areas to be somewhat homogeneous with time step = 1, otherwise (x,y) = (t,area) ~ (1,10000) are too squished
        # because i interpolate it as curve in time-area space, with time parameter which is redundand.
        meanArea = np.mean(areas)
        areasIntep = np.array([[i,a/meanArea] for i,a in enumerate(areas)])
        
        numPointsInTraj = len(trajectory)
        #[cv2.circle(meanImage, tuple(centroid), 3, [255,255,0], -1) for centroid in pathXHCentroids]
        #meanImage   = convertGray2RGB(np.load(meanImagePath)['arr_0'].astype(np.uint8)*0)
        # start analyzing trajectory. need 2 points to make linear extrapolation
        # get extrapolation line, take real value. compare
    
        times  = np.arange(2,numPointsInTraj - 1)
        assert len(times)>0, "traj exterp, have not intended this yet"

        debug = 0; axes,axes2 = (0,0)
        
        if debug == 1:
            fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes.plot(trajectory[:,0],trajectory[:,1], '-o')

            fig2, axes2 = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            axes2.plot(areasIntep[:,0],areasIntep[:,1], '-o')

        for t in times:
            
            IEpolation, endPoint    = exterpTest(trajectory, t, interpolatinIntevalLength, debug = debug, axes= axes, title = str(k))
            IEpolation2, endPoint2  = exterpTest(areasIntep, t, interpolatinIntevalLength, debug = debug, axes= axes2, title = str(k), aspect = 'auto')
            timeReal = timesReal[endPoint + 1]
            pathsDeltas[k][timeReal] = np.linalg.norm(IEpolation[:,-1] - trajectory[endPoint])
            areaDeltas[k][timeReal] = np.abs(int((IEpolation2[:,-1][1] - areasIntep[endPoint2][1])*meanArea))
            #plt.show()
            #axes.plot(*IEpolation, linestyle='dotted', c='orange')
            #axes.plot(*IEpolation[:,-1], '--o', c='orange')
            #axes.plot([IEpolation[0,-1],trajectory[endPoint,0]],[IEpolation[1,-1],trajectory[endPoint,1]], '-',c='black')
            
        a = 1
        plt.show() if debug == 1 else 0
    else:
        # there are times for segment when rough cluster holds two or more nearby bubbles
        # but they are close enough not to split, may be merge, since otherwise it would not be a part of segment.

        # extract time where there are more than one contour
        # common case of two bubbles at best can be described by two contours. at worst as multiple.
        
        sequences = []
        current_sequence = []
    
        for time,num in numContours.items():
            if num > 1:
                current_sequence.append(time)
            else:
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = []

        # subset those time where there are only two contours.
        # cases of two contours are easy, since you dont have to reconstruct anything
        sequencesPair = []
        for n,seq in enumerate(sequences):
            sequencesPair.append([])
            for time in seq:
                if numContours[time] == 2:
                    sequencesPair[n].append(time)
        
        # this test has continuous sequence, without holes. test different when assert below fails.
        assert [len(a) for a in sequences] == [len(a) for a in sequencesPair], 'examine case of dropped contour pairs'

        # make connection between pairs based on area similarity
        for times in sequencesPair:
            hulls           = {time:{} for time in times}
            centroidsAreas  = {time:{} for time in times}
            for time in times:
                subIDs               = contourIDs[time]
                hulls[time]          = {subID:cv2.convexHull(g0_contours[time][subID]) for subID in subIDs}
                centroidsAreas[time] = {subID:centroid_area(contour) for subID,contour in hulls[time].items()}

            connections = []
            startNodes = []
            endNodes = []
            for n,[time,centAreasDict] in enumerate(centroidsAreas.items()):
                temp = []
                if time < times[-1]:
                    for ID,[centroid, area] in centAreasDict.items():
                        nextTime = times[n+1]
                        areasNext = {ID:centAreas[1] for ID, centAreas in centroidsAreas[nextTime].items()}
                        areaDiffs = {ID:np.abs(area-areaX) for ID, areaX in areasNext.items()}
                        minKey = min(areaDiffs, key=areaDiffs.get)

                        connections.append([tuple([time,ID]),tuple([nextTime,minKey])])

                        if time == times[0]:
                            startNodes.append(tuple([time,ID]))
                        elif time == times[-2]:
                            endNodes.append(tuple([nextTime,minKey]))
            


            a  =1
            #allIDs = list(sum(connections,[]))
            #G = nx.Graph()
            #G.add_edges_from(connections)
            #connected_components_all = [list(nx.node_connected_component(G, key)) for key in allIDs]
            #connected_components_all = [sorted(sub, key=lambda x: x[0]) for sub in connected_components_all]
            #connected_components_unique = []
            #[connected_components_unique.append(x) for x in connected_components_all if x not in connected_components_unique]

            # get connections to first sub-sequence from prev node and same for last node
            nodesSubSeq = [tuple([t] + contourIDs[t]) for t in times]
            nodeFirst = nodesSubSeq[0]
            neighBackward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(nodeFirst))  if time < nodeFirst[0]}
            
            nodeLast = nodesSubSeq[-1]
            neighForward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(nodeLast))  if time > nodeLast[0]}

            # remove old unrefined nodes, add new subcluster connections

            H.remove_nodes_from(nodesSubSeq)
            
            H.add_edges_from(connections)

            # add new connections that start and terminate subsequence
            newConnections = []
            for prevNode in neighBackward:
                for node in startNodes:
                    newConnections.append([prevNode,node])

            for nextNode in neighForward:
                for node in endNodes:
                    newConnections.append([node,nextNode])

            overEstimateConnections += newConnections

            H.add_edges_from(newConnections)


            
            #fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
            #for n, nodes in enumerate(connected_components_unique):
            #    centroids = np.array([centroidsAreas[time][ID][0] for time,ID in nodes])
            #    axes.plot(*centroids.T, '-o')
            
            1
        #test = list(H.nodes())
        #node_positions = getNodePos(test)
        #fig, ax = plt.subplots()
        #nx.set_node_attributes(H, node_positions, 'pos')

        #pos = nx.get_node_attributes(H, 'pos')
        #nx.draw(H, pos, with_labels=True, node_size=50, node_color='lightblue',font_size=6,
        #font_color='black')
        #plt.show()


        #firstTime = timesReal[0]
        # segment starts as 2 bubbles
        #if numContours[firstTime] == 2:
        #    neighBackward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(pathX[0]))  if time < pathX[0][0]}
        #    if len(neighBackward) == 2:
        #        1
#axes.legend(prop={'size': 6})
#axes.set_aspect('equal')
#plt.show()

segments2, skipped = graph_extract_paths(H,f) # 23/06/23 info in "extract paths from graphs.py"

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
paths = {i:vals for i,vals in enumerate(segments2)}
node_positions = getNodePos(list(H.nodes()))
#drawH(H, paths, node_positions)

pathCentroidsAreas           = {ID:{} for ID in paths}
pathCentroids           = {ID:{} for ID in paths}
for ID,nodeList in paths.items():
    hulls =  [cv2.convexHull(np.vstack([g0_contours[time][subID] for subID in subIDs])) for time,*subIDs in nodeList]
    pathCentroidsAreas[ID] = [centroid_area(contour) for contour in hulls]
    pathCentroids[ID] = [a[0] for a in pathCentroidsAreas[ID]]


pathsDeltas = {k:{} for k in paths}   
areaDeltas = {k:{} for k in paths}

for k,pathX in paths.items():
    timesReal   = [time             for time,*subIDs in pathX]
    pathXHulls  = [cv2.convexHull(np.vstack([g0_contours[time][subID] for subID in subIDs])) for time,*subIDs in pathX]
    
    pathXHCentroidsAreas = [centroid_area(contour) for contour in pathXHulls]
    trajectory = np.array([a[0] for a in pathXHCentroidsAreas])
    pathCentroids[k] = trajectory

    areas = np.array([a[1] for a in pathXHCentroidsAreas])
    # rescale areas to be somewhat homogeneous with time step = 1, otherwise (x,y) = (t,area) ~ (1,10000) are too squished
    # because i interpolate it as curve in time-area space, with time parameter which is redundand.
    meanArea = np.mean(areas)
    areasIntep = np.array([[i,a/meanArea] for i,a in enumerate(areas)])
        
    numPointsInTraj = len(trajectory)
    #[cv2.circle(meanImage, tuple(centroid), 3, [255,255,0], -1) for centroid in pathXHCentroids]
    #meanImage   = convertGray2RGB(np.load(meanImagePath)['arr_0'].astype(np.uint8)*0)
    # start analyzing trajectory. need 2 points to make linear extrapolation
    # get extrapolation line, take real value. compare
    
    times  = np.arange(2,numPointsInTraj - 1)
    assert len(times)>0, "traj exterp, have not intended this yet"

    debug = 0; axes,axes2 = (0,0)
        
    if debug == 1:
        fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
        axes.plot(trajectory[:,0],trajectory[:,1], '-o')

        fig2, axes2 = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
        axes2.plot(areasIntep[:,0],areasIntep[:,1], '-o')

    for t in times:
            
        IEpolation, endPoint    = exterpTest(trajectory, t, interpolatinIntevalLength, debug = debug, axes= axes, title = str(k))
        IEpolation2, endPoint2  = exterpTest(areasIntep, t, interpolatinIntevalLength, debug = debug, axes= axes2, title = str(k), aspect = 'auto')
        timeReal = timesReal[endPoint + 1]
        pathsDeltas[k][timeReal] = np.linalg.norm(IEpolation[:,-1] - trajectory[endPoint])
        areaDeltas[k][timeReal] = np.abs(int((IEpolation2[:,-1][1] - areasIntep[endPoint2][1])*meanArea))
        #plt.show()
        #axes.plot(*IEpolation, linestyle='dotted', c='orange')
        #axes.plot(*IEpolation[:,-1], '--o', c='orange')
        #axes.plot([IEpolation[0,-1],trajectory[endPoint,0]],[IEpolation[1,-1],trajectory[endPoint,1]], '-',c='black')
            
    a = 1
    plt.show() if debug == 1 else 0
# check segment end points and which cluster they are connected to
skippedTimes = {vals[0]:[] for vals in skipped}
for time,*subIDs in skipped:
    skippedTimes[time].append(subIDs) 

resolvedConnections = []
subSegmentStartNodes = [a[0] for a in paths.values()]
for k,pathX in paths.items():
    terminate = False
    while terminate == False:
        firstNode, lastNode = pathX[0], pathX[-1]

        # find connections-neighbors from first and last segment node
        neighForward    = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(lastNode))   if time > lastNode[0]}
        # 
        #neighForward = {ID:vals for ID,vals in neighForward.items() if ID not in subSegmentStartNodes}
        #neighBackward   = {tuple([time] + subIDs): subIDs for time,*subIDs in list(H.neighbors(firstNode))  if time < firstNode[0]}
        if len(neighForward)>0:
            time = lastNode[0] 
            timeNext = time + 1
        
            # find neigbor centroids
            pathXHulls  = {k:cv2.convexHull(np.vstack([g0_contours[timeNext][subID] for subID in subIDs])) for k,subIDs in neighForward.items()}
            pathXHCentroids = {k:centroid_area(contour)[0] for k,contour in pathXHulls.items()}
            trajectory      = pathCentroids[k]
            testChoice      = {}
            # test each case, if it fits extrapolation- real distance via history of segment
            for key, centroid in pathXHCentroids.items():
                trajectoryTest = np.vstack((trajectory,centroid))#np.concatenate((trajectory,[centroid]))
                IEpolation, endPoint = exterpTest(trajectoryTest,len(trajectoryTest) - 1,interpolatinIntevalLength, 0)
                dist = np.linalg.norm(IEpolation[:,-1] - trajectoryTest[endPoint])
                testChoice[key] = dist
        
            # check which extrap-real dist is smallest, get history
            minKey = min(testChoice, key=testChoice.get)
            lastDeltas = [d for t,d in pathsDeltas[k].items() if time - 4 < t <= time]
            mean = np.mean(lastDeltas)
            std = np.std(lastDeltas)
            # test distance via history
            if testChoice[minKey] < mean + 2*std:
                removeNodeConnections = [node for node in neighForward if node != minKey]
                remEdges = []
                remEdges += [tuple([lastNode,node]) for node in removeNodeConnections]
                H.remove_edges_from(remEdges)
                resolvedConn = [lastNode,minKey]

                for conn in [resolvedConn] + remEdges:
                    if conn in overEstimateConnections: overEstimateConnections.remove(conn) # no skipped inside, may do nothing
                resolvedConnections.append(resolvedConn)
                paths[k].append(minKey)
                pathX.append(minKey)
                pathCentroids[k] = np.vstack((pathCentroids[k],pathXHCentroids[minKey]))
                pathsDeltas[k][timeNext] = testChoice[minKey]

                # removed edges that ar pointing to solution, other tan real one.
                solBackNeighb = [ID for ID in list(H.neighbors(minKey)) if ID[0] < timeNext]
                removeNodeConnections2 = [node for node in solBackNeighb if node != lastNode]
                remEdges = []
                remEdges += [tuple([node,minKey]) for node in removeNodeConnections2]
                H.remove_edges_from(remEdges)

                if minKey in subSegmentStartNodes:
                    terminate = True
                #drawH(H, paths, node_positions)
            else:
                terminate = True
        elif len(neighForward) == 0: # segment terminates
            terminate = True
        
                # ==== !!!!!!!!!!!!!!!
                # might want to iterate though skipped nodes and connect to next segment!
        a = 1
    #if timeNext in skippedTimes:
    #    testCluster = skippedTimes[timeNext]
        


#for time,*subIDs in [pathX[X]]:
#        [cv2.drawContours(  meanImage,   g0_contours[time], subID, 255 , 1) for subID in subIDs]
#cv2.drawContours(  meanImage,   pathXHulls, X, 128 , 1)
#cv2.imshow('a', meanImage)
#drawH(H, paths, node_positions)
a = 1


#nx.draw(H, with_labels=True, edge_color=list(colors_edges2.values()), width = list(width2.values()))

#end_points = [node for node in H.nodes() if H.degree[node] == 1]
#end_points.remove(test[-1])
#print(f'endpoints: {end_points}')
#stor = {a:[a] for a in end_points}
#for start in end_points:
#    bfs_tree = nx.bfs_tree(H, start)
#    target_node = None
#    for node in bfs_tree.nodes():
#        neighbors = [a for a in H.neighbors(node) if a[0]> node[0]]
#        target_node = node
#        stor[start].append(node)
#        if len(neighbors) > 1:
#            break
#start = test[0][0]
#end = test[-1][0]
#times = {t:[] for t in np.arange(start, end + 1, 1)}
#gatherByTime = {t:[] for t in np.arange(start, end + 1, 1)}

#for t,*clusterID in test:
     
#    #gatherByTime2[t].append((t,ID2S(g0_clusters[t][clusterID])))
    #gatherByTime[t].append((t,*clusterID))

f = lambda x : x[0]

segments2, skipped = graph_extract_paths(H,f) # 23/06/23 info in "extract paths from graphs.py"

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
segments2 = list(sorted(segments2, key=lambda x: x[0][0]))
paths = {i:vals for i,vals in enumerate(segments2)}


#drawH(H, paths, node_positions)

if 1 == 1:
    imgInspectPath = os.path.join(imageFolder, "inspectPaths")
    meanImage = convertGray2RGB(np.load(meanImagePath)['arr_0'].astype(np.uint8))
    for n, keysList in paths.items():
        for k,subCase in enumerate(keysList):
            img = meanImage.copy()
            time,*subIDs = subCase
            for subID in subIDs:
                cv2.drawContours(  img,   g0_contours[time], subID, cyclicColor(n), -1)
                if subID in g0_contours_children[time]:
                    [cv2.drawContours(  img,   g0_contours[time], ID, (0,0,0), -1) for ID in g0_contours_children[time][subID]]
                
                startPos2 = g0_contours[time][subID][-30][0] 
                [cv2.putText(img, str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
            fileName = f"{str(doX).zfill(3)}_{str(n).zfill(2)}_{str(time).zfill(3)}.png"
            cv2.imwrite(os.path.join(imgInspectPath,fileName) ,img)

if 1 == -1:
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
            x,y,w,h = cv2.boundingRect(np.vstack([g0_contours[time][ID] for ID in subIDs]))
            #x,y,w,h = g0_bigBoundingRect2[time][subCase]
            #x,y,w,h = g0_bigBoundingRect[time][ID]
            cv2.rectangle(imgs[time], (x,y), (x+w,y+h), cyclicColor(n), 1)
            [cv2.putText(imgs[time], str(n), (x,y), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]# connected clusters = same color
        for k,subCase in enumerate(case):
            time,*subIDs = subCase
            for subID in subIDs:
                startPos2 = g0_contours[time][subID][-30][0] 
                [cv2.putText(imgs[time], str(subID), startPos2, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
        

    for k,img in enumerate(imgs):
        folder = r"./post_tests/testImgs/"
        fileName = f"{str(k).zfill(4)}.png"
        cv2.imwrite(os.path.join(folder,fileName) ,img)


a = 1
#if useMeanWindow == 1:
#    meanImage = masksArr[whichMaskInterval(globalCounter,intervalIndecies)]

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()