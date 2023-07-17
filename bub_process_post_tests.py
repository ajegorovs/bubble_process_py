import enum
import numpy as np, itertools, networkx as nx, sys
import cv2, os, glob, datetime, re, pickle#, multiprocessing
# import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import interpolate
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

    def centroid_area(contour):
        m = cv2.moments(contour)
        area = int(m['m00'])
        cx0, cy0 = m['m10'], m['m01']
        centroid = np.array([cx0,cy0])/area
        return  centroid, area
    
    # ive tested c_mom_zz in centroid_area_cmomzz. it looks correct from definition m_zz = sum_x,y[r^2] = sum_x,y[(x-x0)^2 + (y-y0)^2]
    # which is the same as sum_x,y[(x-x0)^2] + sum_x,y[(y-y0)^2] = mu(2,0) + mu(0,2), and from tests on images.
    def centroid_area_cmomzz(contour):
        m = cv2.moments(contour)
        area = int(m['m00'])
        cx0, cy0 = m['m10'], m['m01']
        centroid = np.array([cx0,cy0])/area
        c_mom_xx, c_mom_yy = m['mu20'], m['mu02']
        c_mom_zz = int((c_mom_xx + c_mom_yy))
        return  centroid, area, c_mom_zz
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
# REMARK: it is very likely that solo-to-solo (later called 121) is pseudo split-merge, optical effect

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


def segment_conn_end_start_points(connections, nodes = 0):
    if connections is not None:
        if type(connections) == tuple:
            start,end = connections
            if nodes == 1:
                return (segments2[start][-1],segments2[end][0])
            else:
                return (segments2[start][-1][0],segments2[end][0][0])
        if type(connections) == list:
            if nodes == 1:
                return [(segments2[start][-1],segments2[end][0]) for start,end in connections]
            else:
                return [(segments2[start][-1][0],segments2[end][0][0]) for start,end in connections]
        else:
            return None
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

# Iterate over the dictionary and add edges with weights
for (node1, node2), weight in lr_intervals_lengths.items():
    G2.add_edge(node1, node2, weight=weight)

for g in G2.nodes():
      G2.nodes()[g]["t_start"] = segments2[g][0][0]
      G2.nodes()[g]["t_end"] = segments2[g][-1][0]
#pos = nx.spring_layout(G2)
#edge_widths = [data['weight'] for _, _, data in G2.edges(data=True)]
#nx.draw(G2, pos, with_labels=True, width=edge_widths)

#plt.show()

# ===============================================================================================
# ===============================================================================================
# === extract closest neighbors of segments ===
# ===============================================================================================
# ===============================================================================================
# REMARK: inspecting closest neighbors allows to elliminate cases with segments inbetween 
# REMARK: connected segments via lr_maxDT. problem is that affects branches of splits/merges
if 1 == 1:
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

# ===============================================================================================
# ===============================================================================================
# === wipe inter-segement edges from G2 graph and replot only nearest connections ===
# ===============================================================================================
# ===============================================================================================
# prepare plot for segment interconnectedness
G2.remove_edges_from(list(G2.edges()))
for node, t_conn in t_neighbor_sol_all_prev.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2, weight=1/np.abs(weight))
for node, t_conn in t_neighbor_sol_all_next.items():
    for node2, weight in t_conn.items():
        G2.add_edge(node, node2, weight=1/weight)

# ===============================================================================================
# ===============================================================================================
# === calculate all paths between  nearest segments ===
# ===============================================================================================
# ===============================================================================================
if 1 == 1:
    lr_close_segments_simple_paths = {}
    lr_close_segments_simple_paths_inter = {}
    tIDs = [tID for tID,t_dic in t_neighbor_sol_all_next.items() if len(t_dic) >0]
    for t_from in tIDs:
        for t_to in t_neighbor_sol_all_next[t_from]:
            t_from_node_last   = segments2[t_from][-1]
            t_to_node_first      = segments2[t_to][0]

            t_from_node_last_time   = t_from_node_last[0]
            t_to_node_first_time    = t_to_node_first[0]
            t_from_to_max_time_steps= t_to_node_first_time - t_from_node_last_time + 1
            t_from_to_paths_simple  = list(nx.all_simple_paths(G, t_from_node_last, t_to_node_first, cutoff = t_from_to_max_time_steps))

        
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


    for g in G2.nodes():
      G2.nodes()[g]["height"] = node_positions2[g][0]
    #draw_graph_with_height(G2,figsize=(5,5), labels=labels)


    #pos = nx.spring_layout(G2, pos = node_positions2, k = 1, iterations = 10)

    #nx.draw(G2, pos=node_positions2, labels=labels)
    #plt.show()
    1

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
        segment_conn_end_start_points(t_unidir_conns)
        segment_conn_end_start_points(t_conn)
        a = 1
        #if t_to in t_neighbor_sol_all_prev[t_from]:
    t_merge_split_culprit_edges = sorted(t_merge_split_culprit_edges, key = lambda x: x[0])


    # simply extract nodes and their neigbors if there are multiple neighbors
    t_merge_split_culprit_edges2= []
    for t_from, t_forward_conns in t_neighbor_sol_all_next.items():
        if len(t_forward_conns)>1:
            for t_to in t_forward_conns.keys():
                t_merge_split_culprit_edges2.append(tuple(sorted([t_from,t_to])))

    for t_to, t_backward_conns in t_neighbor_sol_all_prev.items():
        if len(t_backward_conns)>1:
            for t_from in t_backward_conns.keys():
                t_merge_split_culprit_edges2.append(tuple(sorted([t_from,t_to])))

    t_merge_split_culprit_edges2 = sorted(t_merge_split_culprit_edges2, key = lambda x: x[0])
    segment_conn_end_start_points(t_merge_split_culprit_edges2)

    t_merge_split_culprit_edges_all = sorted(list(set(t_merge_split_culprit_edges + t_merge_split_culprit_edges2)), key = lambda x: x[0])
    t_merge_split_culprit_node_combos = segment_conn_end_start_points(t_merge_split_culprit_edges_all, nodes = 1)

    # find clusters of connected nodes of split/merge events. this way instead of sigment IDs, because one
    # segment may be sandwitched between any of these events and two cluster will be clumped together

    T = nx.Graph()
    T.add_edges_from(t_merge_split_culprit_node_combos)
    connected_components_all = [list(nx.node_connected_component(T, key)) for key in T.nodes()]
    connected_components_all = [sorted(sub) for sub in connected_components_all]
    # extract connected families
    connected_components_unique = []
    [connected_components_unique.append(x) for x in connected_components_all if x not in connected_components_unique]

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

if 1 == 1:
    lr_conn_edges_splits = []
    lr_conn_edges_merges = []
    lr_conn_edges_splits_merges_mixed = []

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
    
        #bb = segment_conn_end_start_points(t_edges_merge, nodes = 1)
        #bb2 = segment_conn_end_start_points(t_edges_split, nodes = 1)

        if len(t_neighbors_next_large) == 0 and len(t_neighbors_prev_large) > 0:
            lr_conn_edges_merges += t_edges_merge
        elif len(t_neighbors_prev_large) == 0 and len(t_neighbors_next_large) > 0:
            lr_conn_edges_splits += t_edges_split
        else:
            lr_conn_edges_splits_merges_mixed += (t_edges_merge + t_edges_split)
        a = 1

    #segment_conn_end_start_points(lr_conn_edges_splits, nodes = 1)
    #segment_conn_end_start_points(lr_conn_edges_merges, nodes = 1)
    a = 1
#drawH(G, paths, node_positions)
#segment_conn_end_start_points(lr_connections_directed)
#segment_conn_end_start_points(lr_connections_unidirectional)

# ===============================================================================================
# ===============================================================================================
# === extract segment-segment connections that are connected only together (one-to-one; 121) ====
# ===============================================================================================
# REMARK: as discussed before directed connections are part of merges, but in unidirectional
# REMARK: connectios exists an associated (main) connection that makes other a directional
# REMARK: it was a result of performing nearest neighbor refinement.
# REMARK: so non split/merge connection is refinement of unidir connection list with info of merge/split

t_conn_121 = [t_conn for t_conn in lr_connections_unidirectional if t_conn not in t_merge_split_culprit_edges_all]
aa0  = sorted(   segment_conn_end_start_points(t_conn_121, nodes = 1),   key = lambda x: x[0])
if 1 == 1:
    
    # find segments that connect only to one neighbor segment both directions
    # from node "t_node" to node "t_neighbor"
    #t_conn_121 = []
    #for t_node in G2.nodes():
    #    t_node_neighbors = list(G2.neighbors(t_node))
    #    t_node_time_end      = segments2[t_node][-1][0]
    #    t_node_neighbors_next = []
    #    for t_neighbor in t_node_neighbors:
    #        t_neighbor_time_start    = segments2[t_neighbor][0][0]
    #        if t_node_time_end < t_neighbor_time_start:
    #            t_node_neighbors_next.append(t_neighbor)
    #    if len(t_node_neighbors_next) == 1:
    #        t_neighbor              = t_node_neighbors_next[0]
    #        t_neighbor_time_start   = segments2[t_neighbor][0][0]
    #        t_neighbor_neighbors   = list(G2.neighbors(t_neighbor))
        
    #        t_neighbor_neighbors_back = []
    #        for t_neighbor_neighbor in t_neighbor_neighbors:
    #            t_neighbor_neighbor_time_end      = segments2[t_neighbor_neighbor][-1][0]
    #            if t_neighbor_time_start > t_neighbor_neighbor_time_end:
    #                t_neighbor_neighbors_back.append(t_neighbor)
    #        if len(t_neighbor_neighbors_back) == 1:
    #            t_conn_121.append(tuple([t_node,t_neighbor]))
    
    #t_conn_one_to_one2 = []
    #for t_node,t_forward_connections in t_neighbor_sol_all_next.items():
    #    if len(t_forward_connections) ==  1:
    #        t_forward_neighbor = list(t_forward_connections.keys())[0]
    #        t_forward_neighbor_back_neighbors = t_neighbor_sol_all_prev[t_forward_neighbor]
    #        if len(t_forward_neighbor_back_neighbors) == 1 and list(t_forward_neighbor_back_neighbors.keys())[0] == t_node:
    #            t_conn_one_to_one2.append(tuple([t_node,t_forward_neighbor]))
    #aa  = sorted(   segment_conn_end_start_points(t_conn_121, nodes = 1),   key = lambda x: x[0])
    #aa2 = sorted(   segment_conn_end_start_points(t_conn_one_to_one2, nodes = 1),  key = lambda x: x[0])
    #t_conn_one_to_one_test = [(segments2[start][-1][0],segments2[end][0][0]) for start,end in t_conn_121]
    1

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

t_conn_121 = [t_node for t_node in t_conn_121 if t_node not in t_conn_121_zero_path]

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
    t_from_neigbors_next    = extractNeighborsNext(     G, t_from_node_end,  lambda x: x[0])
    t_to_neigbors_prev      = extractNeighborsPrevious( G, t_to_node_start,  lambda x: x[0])
    t_conn_121_zero_path_nodes[t_conn] = sorted(set(t_from_neigbors_next+t_to_neigbors_prev),key=lambda x: (x[0], x[1]))
    t_times = sorted(set([t_node[0] for t_node in t_conn_121_zero_path_nodes[t_conn]]))
    t_conn_121_zp_contour_combs[t_conn] = {t_time:[] for t_time in t_times}
    
    for t_time,*t_subIDs in t_conn_121_zero_path_nodes[t_conn]:
        t_conn_121_zp_contour_combs[t_conn][t_time] += t_subIDs

    for t_time in t_conn_121_zp_contour_combs[t_conn]:
         t_conn_121_zp_contour_combs[t_conn][t_time] = sorted(t_conn_121_zp_contour_combs[t_conn][t_time])
a = 1

# ===============================================================================================
# gather segment hulls and centroids
# ===============================================================================================
# REMARK: not doing anything with zero path because it has two or more nodes at one time.
t_all_121_segment_IDs = sorted(list(set(sum([list(a) for a in t_conn_121],[]))))
t_segments_121_centroids    = {tID:{} for tID in t_all_121_segment_IDs}
t_segments_121_areas        = {tID:{} for tID in t_all_121_segment_IDs}
t_segments_121_mom_z        = {tID:{} for tID in t_all_121_segment_IDs}
for tID in t_all_121_segment_IDs:
    t_segment = segments2[tID]
    for t_time,*t_subIDs in t_segment:
        t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_subIDs]))
        t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)
        t_node = tuple([t_time] + t_subIDs)
        t_segments_121_centroids[   tID][t_node]   = t_centroid
        t_segments_121_areas[       tID][t_node]   = t_area
        t_segments_121_mom_z[       tID][t_node]   = t_mom_z

            

# ===============================================================================================
# = 2X LARGE Segments: extract one-to-one segment connections of large segments <=> high trust ==
# ===============================================================================================
# lets extract segment-segment (one2one) connections that are both of large length
# most likely a pseudo split-merge (p_split-merge). large length => most likely solo bubble
lr_segment_len_large = 7
lr_conn_one_to_one_large = []
for t_from,t_to in t_conn_121:
    t_segment_length_from   = lr_segments_lengths[t_from]
    t_segment_length_to     = lr_segments_lengths[t_to]
    if t_segment_length_from >= lr_segment_len_large and t_segment_length_to >= lr_segment_len_large:
        lr_conn_one_to_one_large.append(tuple([t_from,t_to]))
lr_conn_one_to_one_large_test = [(segments2[start][-1][0],segments2[end][0][0]) for start,end in lr_conn_one_to_one_large]


# ===============================================================================================
# ======== SHORT SEGMENTS BUT ISOLATED: check if one2one (121) connection is isolated: ==========
# ===============================================================================================
# REMARK: isolated connection is only connected to its own inter-segment path nodes
# REMARK: and not branching out 1 step further. pretty strict condition

# take inter-segment simple paths, extract nodes; check neighbors of nodes;
# isolate nonpath nodes; see what isleft. if path had a closed boundary, leftover will be zero.

if 1 == 1:
    lr_conn_one_to_one_other = [t_conn for t_conn in t_conn_121 if t_conn not in lr_conn_one_to_one_large]
    lr_conn_one_to_one_other_test = [(segments2[start][-1][0],segments2[end][0][0]) for start,end in lr_conn_one_to_one_other]

    t_conn_121_other_isolated_not = []
    t_conn_121_other_isolated = []
    for t_from,t_to in lr_conn_one_to_one_other:

        t_from_node_last   = segments2[t_from][-1]
        t_to_node_first      = segments2[t_to][0]

        t_from_node_last_time   = t_from_node_last[0]
        t_to_node_first_time    = t_to_node_first[0]
        t_from_to_max_time_steps= t_to_node_first_time - t_from_node_last_time + 1
        #t_from_to_paths_simple = list(nx.all_simple_paths(G, t_from_node_last, t_to_node_first, cutoff = t_from_to_max_time_steps))
        #t_from_to_paths_simple = lr_close_segments_simple_paths[tuple([t_from,t_to])]
        #t_from_to_paths_nodes_all = sorted(set(sum(t_from_to_paths_simple,[])),key=lambda x: x[0])
        #t_from_to_paths_nodes_all_inter = [t_node for t_node in t_from_to_paths_nodes_all if t_node not in [t_from_node_last,t_to_node_first]]
        t_from_to_paths_nodes_all_inter = lr_close_segments_simple_paths_inter[   tuple([t_from,t_to])]
        t_all_path_neighbors = []
        for t_node in t_from_to_paths_nodes_all_inter:
            t_all_path_neighbors.append(list(G.neighbors(t_node)))
        t_all_path_neighbors_node_all = sorted(set(sum(t_all_path_neighbors,[])),key=lambda x: x[0])
        t_nides_not_in_main_path = [t_node for t_node in t_all_path_neighbors_node_all if t_node not in t_from_to_paths_nodes_all_inter + [t_from_node_last,t_to_node_first]]
        if len(t_nides_not_in_main_path):   t_conn_121_other_isolated_not.append(tuple([t_from,t_to]))
        else:                               t_conn_121_other_isolated.append(tuple([t_from,t_to]))

    t_conn_121_other_isolated_not_test  = sorted(   segment_conn_end_start_points(t_conn_121_other_isolated_not, nodes = 1),   key = lambda x: x[0])#[(segments2[start][-1][0],segments2[end][0][0]) for start,end in t_conn_121_other_isolated_not]
    t_conn_121_other_isolated_test      = sorted(   segment_conn_end_start_points(t_conn_121_other_isolated, nodes = 1),   key = lambda x: x[0])#[(segments2[start][-1][0],segments2[end][0][0]) for start,end in t_conn_121_other_isolated]

#drawH(G, paths, node_positions)
# ===============================================================================================
# =============== CHECK IF SEGMENTS ARE NOT RELATED TO SPLIT MERGE PREV/POST ====================
# ===============================================================================================
# REMARK: example as shows schematically below outlines a problem where pseudo split-merge is not recovered
# REMARK:      /3b\   /5b--6b\
# REMARK: 1--2      4          7--8 
# REMARK:      \3a/    5a--6a/
# REMARK: [1,2]->X->[4,5b,6b] is correct 121 event. although its clear that (4,5a) edge is missing

if 1==1:
    sorted(   segment_conn_end_start_points(t_conn_121, nodes = 1),   key = lambda x: x[0])

    t_conn_121_other_terminated = []
    t_conn_121_other_terminated_inspect = []
    t_conn_121_other_terminated_failed = []

    for t_conn in t_conn_121_other_isolated:

        t_from, t_to = t_conn

        t_from_start    = G2.nodes[t_from   ]["t_start"]
        t_to_end        = G2.nodes[t_to     ]["t_end"]

        t_from_neighbors_prev   = [tID for tID in list(G2.neighbors(t_from))    if G2.nodes[tID]["t_end"    ] < t_from_start]
        t_to_neighbors_next     = [tID for tID in list(G2.neighbors(t_to))      if G2.nodes[tID]["t_start"  ] > t_to_end    ]

        t_conn_back = [tuple(sorted([tID,t_from]))  for tID in t_from_neighbors_prev]
        t_conn_forw = [tuple(sorted([t_to,tID]))    for tID in t_to_neighbors_next  ]
    
        t_prev_was_split    = True if len([t for t in t_conn_back if t in lr_conn_edges_splits]) > 0 else False
        t_next_is_merge     = True if len([t for t in t_conn_forw if t in lr_conn_edges_merges]) > 0 else False

        if not t_prev_was_split and not t_next_is_merge:
            t_conn_121_other_terminated.append(t_conn)
        elif ((t_prev_was_split and not t_prev_was_split) or  (not t_prev_was_split and t_next_is_merge)):
            t_conn_121_other_terminated_inspect.append(t_conn)
        else: 
            t_conn_121_other_terminated_failed.append(t_conn)

    a = 1
    if  1 == 1:
        #t_conn_121_other_terminated = []
        #t_conn_121_other_terminated_inspect = []
        #t_conn_121_other_terminated_failed = []
        #for t_from,t_to in t_conn_121_other_isolated:
        #    t_from_large    = True if lr_segments_lengths[t_from]   >= lr_trusted_segment_length else False
        #    t_to_large      = True if lr_segments_lengths[t_to]     >= lr_trusted_segment_length else False
        #    t_from_pass     = False
        #    t_to_pass       = False
        #    t_from_inspect  = False
        #    t_to_inspect    = False
        #    # find neighbors of segment-segment
        #    if t_to_large == False:
    
        #        t_to_node_last          = segments2[t_to][-1]
        #        # check is node has connections forward or is terminated
        #        t_to_neighbors_next     = extractNeighborsNext(     G, t_to_node_last,    lambda x: x[0])
        #        # 
        #        t_to_conn_to = [t_conn for t_conn in lr_close_segments_simple_paths if t_conn[0] == t_to]
        
        #        if len(t_to_neighbors_next) == 0:   # terminated
        #            t_to_pass = True
        #        elif len(t_to_conn_to) == 1:        # not terminated but has connections to one segment
        #            t_conn = t_to_conn_to[0]
        #            tID = t_conn[1]                 # check if solo target segment is connected from multiple prev segments
        #            t_to_conn_to_from = [t for t in lr_close_segments_simple_paths if t[1] == tID]
        #            if len(t_to_conn_to_from) == 1: # solo connections, check for parasitic nodes one step prev to target
        #                t_to_neighbors_next_node = segments2[tID][0]
        #                t_to_neighbors_next_prev = extractNeighborsPrevious( G, t_to_neighbors_next_node,     lambda x: x[0])
        #                # comment: checking only one prev step is not enough, since merge may be shared formultiple steps. but cant do anything now
        #                t_to_not_in_main = [ t for t in t_to_neighbors_next_prev if t not in lr_close_segments_simple_paths_inter[t_conn] + [t_to_node_last, t_to_neighbors_next_node]]
        #                if len(t_to_not_in_main) == 0:
        #                    t_to_pass = True
        #                else:
        #                    t_to_inspect = True
        #            else:
        #                t_to_inspect = True
        #        else:                               # it splits. should be ok as long as its the only parent
        #            t_to_conn_to_from_all = []
        #            for t_conn in t_to_conn_to:     # connections from t_to to next -> (t_to, next1), (t_to, next2), ...
        #                tID = t_conn[1]             # next; find all connections to next
        #                t_to_conn_to_from_all.append([t for t in lr_close_segments_simple_paths if t[1] == tID])
        #            t_to_conn_to_from_all = sum(t_to_conn_to_from_all,[])
        #            t_to_not_in_main = [t for t in t_to_conn_to_from_all if t not in t_to_conn_to]
        #            if len(t_to_not_in_main) == 0:
        #                t_to_pass = True
        #            else:
        #                t_to_inspect = True

        #    else:
        #        t_to_pass = True

        #    if t_from_large == False: # not tested well due to lack of cases. just symmetric to top variant. i hope
        #        t_from_node_first       = segments2[t_from][0]
        #        t_from_neighbors_prev   = extractNeighborsPrevious( G, t_from_node_first,     lambda x: x[0])
        #        t_from_conn_from = [t_conn for t_conn in lr_close_segments_simple_paths if t_conn[1] == t_from]
        #        if len(t_from_neighbors_prev) == 0:
        #            t_from_pass = True
        #        elif len(t_from_conn_from) == 1:    # one connection from-> prev, does not mean prev-> has one conn, but may split
        #            t_conn = t_from_conn_from[0]
        #            tID = t_conn[0]                 # check forward connections prev-> forward. so step back and foward = from->to
        #            t_from_conn_from_to = [t_conn for t_conn in lr_close_segments_simple_paths if t_conn[0] == tID]
        #            if len(t_from_conn_from_to) == 1:
        #                #assert 1 == -1, "untested territory"
        #                t_from_neighbors_prev_node = segments2[tID][-1]
        #                t_from_neighbors_prev_next = extractNeighborsNext( G, t_from_neighbors_prev_node,     lambda x: x[0])
        #                # comment: checking only one prev step is not enough, since merge may be shared formultiple steps. but cant do anything now
        #                t_from_not_in_main = [ t for t in t_from_neighbors_prev_next if t not in lr_close_segments_simple_paths_inter[t_conn] + [t_from_neighbors_prev_node, t_from_node_first]]
        #                if len(t_from_not_in_main) == 0:
        #                    t_from_pass = True
        #                else:
        #                    t_from_inspect = True
        #            else:
        #                t_from_inspect = True
        #        else:
        #            assert 1 == -1, "unknown territory"
        #    else:
        #        t_from_pass = True

        #    if t_from_pass and t_to_pass:
        #        t_conn_121_other_terminated.append(tuple([t_from,t_to]))
        #    elif ((t_from_pass and t_to_inspect) or  (t_from_inspect and t_to_pass)):
        #        t_conn_121_other_terminated_inspect.append(tuple([t_from,t_to]))
        #    else: 
        #        t_conn_121_other_terminated_failed.append(tuple([t_from,t_to]))
        a = 1


    a = 1
# maybe deal with t_conn_121_other_terminated_inspect here, or during analysis
#drawH(G, paths, node_positions)
# ===============================================================================================
# ===============================================================================================
# ============ TEST INTERMERDIATE SEGMENTS FOR LARGE AND CONFIRMED SHORT 121s ===================
# ===============================================================================================
# ===============================================================================================

lr_relevant_conns = lr_conn_one_to_one_large + t_conn_121_other_terminated

# ===============================================================================================
# ============ interpolate trajectories between segments for further tests ===================
# ===============================================================================================
if 1 == 1:
    def interpolateMiddle2D(t_conn, t_centroid_dict, segments2, t_inter_times, histLen = 5, s = 15, debug = 0, aspect = 'equal'):
        t_from,t_to     = t_conn
        t_hist_prev     = segments2[t_from][-histLen:]
        t_hist_next     = segments2[t_to][:histLen]
        t_traj_prev     = np.array([t_centroid_dict[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next     = np.array([t_centroid_dict[  t_to][t_node] for t_node in t_hist_next])
        t_times_prev    = [t_node[0] for t_node in t_hist_prev]
        t_times_next    = [t_node[0] for t_node in t_hist_next]
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
            axes.set_title(t_conn)
            axes.set_aspect(aspect)
            axes.legend(prop={'size': 6})
            plt.show()
        return IEpolation.T

    def interpolateMiddle1D(t_conn, t_property_dict, segments2, t_inter_times, rescale = True, histLen = 5, s = 15, debug = 0, aspect = 'equal'):
        t_from,t_to     = t_conn
        t_hist_prev     = segments2[t_from][-histLen:]
        t_hist_next     = segments2[t_to][:histLen]
        t_traj_prev     = np.array([t_property_dict[t_from][t_node] for t_node in t_hist_prev])
        t_traj_next     = np.array([t_property_dict[  t_to][t_node] for t_node in t_hist_next])
        t_times_prev    = [t_node[0] for t_node in t_hist_prev]
        t_times_next    = [t_node[0] for t_node in t_hist_next]
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
            axes.set_title(t_conn)
            axes.set_aspect(aspect)
            axes.legend(prop={'size': 6})
            plt.show()
        return IEpolation.T

lr_121_interpolation_times = {}

for t_conn in lr_relevant_conns:
    t_from,t_to     = t_conn
    t_time_prev     = segments2[t_from][-1][0]
    t_time_next     = segments2[t_to][0][0]
    lr_121_interpolation_times[t_conn] = np.arange(t_time_prev+1,t_time_next, 1)

lr_121_interpolation_centroids  = {t_conn:[] for t_conn in lr_relevant_conns}
lr_121_interpolation_areas      = {t_conn:[] for t_conn in lr_relevant_conns}
lr_121_interpolation_moment_z   = {t_conn:[] for t_conn in lr_relevant_conns}


for t_conn in lr_relevant_conns:
    lr_121_interpolation_centroids[t_conn]  = interpolateMiddle2D(t_conn, t_segments_121_centroids,
                                                                 segments2, lr_121_interpolation_times[t_conn],
                                                                 histLen = 5, s = 15, debug = 0)

    lr_121_interpolation_areas[t_conn]      = interpolateMiddle1D(t_conn, t_segments_121_areas,
                                                                 segments2, lr_121_interpolation_times[t_conn],  rescale = True,
                                                                 histLen = 5, s = 15, debug = 0, aspect = 'auto')

    lr_121_interpolation_moment_z[t_conn]   = interpolateMiddle1D(t_conn, t_segments_121_mom_z,
                                                                 segments2, lr_121_interpolation_times[t_conn],  rescale = True,
                                                                 histLen = 5, s = 15, debug = 0, aspect = 'auto')
    a = 1
    
# ===============================================================================================
# ====  EXTRACT POSSIBLE CONTOUR ELEMENTS FROM PATHS ====
# ===============================================================================================
# REMARK: i expect most of the time solution to be all elements in cluster. except real merges

lr_contour_combs                = {tID:{} for tID in lr_relevant_conns}
for t_conn in lr_relevant_conns:
    t_traj = lr_close_segments_simple_paths[t_conn][0]
    t_min = t_traj[0][0]
    t_max = t_traj[-1][0]
    #t_nodes = {t:[] for t in np.arange(t_min + 1, t_max, 1)}
    t_nodes = {t:[] for t in np.arange(t_min, t_max + 1, 1)}
    for t_traj in lr_close_segments_simple_paths[t_conn]:
        #t_traj = t_traj[1:-1]
        for t_time,*t_subIDs in t_traj:
            t_nodes[t_time] += t_subIDs
    for t_time in t_nodes:
        t_nodes[t_time] = sorted(list(set(t_nodes[t_time])))
    lr_contour_combs[t_conn] = t_nodes
a = 1
# ===============================================================================================
# ====  CONSTRUCT PERMUTATIONS FROM CLUSTER ELEMENT CHOICES ====
# ===============================================================================================
# REMARK: grab different combinations of elements in clusters
lr_contour_combs_perms = {t_conn:{t_time:[] for t_time in t_dict} for t_conn,t_dict in lr_contour_combs.items()}
for t_conn, t_times_contours in lr_contour_combs.items():
    for t_time,t_contours in t_times_contours.items():
        t_perms = sum([list(itertools.combinations(t_contours, r)) for r in range(1,len(t_contours)+1)],[])
        lr_contour_combs_perms[t_conn][t_time] = t_perms

# ===============================================================================================
# ====  PRE-CALCULATE HULL CENTROIDS AND AREAS FOR EACH PERMUTATION ====
# ===============================================================================================
# REMARK: these will be reused alot in next steps, store them to avoid need of recalculation

lr_permutation_areas_precomputed    = {t_conn:
                                            {t_time:
                                                    {t_perm:0 for t_perm in t_perms}
                                             for t_time,t_perms in t_times_perms.items()}
                                        for t_conn,t_times_perms in lr_contour_combs_perms.items()}

lr_permutation_centroids_precomputed= {t_conn:
                                            {t_time:
                                                    {t_perm:[0,0] for t_perm in t_perms}
                                             for t_time,t_perms in t_times_perms.items()}
                                        for t_conn,t_times_perms in lr_contour_combs_perms.items()}

lr_permutation_mom_z_precomputed    = {t_conn:
                                            {t_time:
                                                    {t_perm:0 for t_perm in t_perms}
                                             for t_time,t_perms in t_times_perms.items()}
                                        for t_conn,t_times_perms in lr_contour_combs_perms.items()}
for t_conn, t_times_perms in lr_contour_combs_perms.items():
    for t_time,t_perms in t_times_perms.items():
        for t_perm in t_perms:
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][subID] for subID in t_perm]))
            t_centroid, t_area, t_mom_z = centroid_area_cmomzz(t_hull)

            lr_permutation_areas_precomputed[t_conn][t_time][t_perm] = t_area
            lr_permutation_centroids_precomputed[t_conn][t_time][t_perm] = t_centroid
            lr_permutation_mom_z_precomputed[t_conn][t_time][t_perm] = t_mom_z

#drawH(G, paths, node_positions)
# ===============================================================================================
# ====  CONSTUCT DIFFERENT PATHS FROM UNIQUE CHOICES ====
# ===============================================================================================
# REMARK: try different evolutions of inter-segment paths
a = 1
lr_permutation_cases = {t_conn:[] for t_conn in lr_contour_combs}
lr_permutation_times = {t_conn:[] for t_conn in lr_contour_combs}
for t_conn, t_times_perms in lr_contour_combs_perms.items():
    
    t_values = list(t_times_perms.values())
    t_times = list(t_times_perms.keys())

    sequences = list(itertools.product(*t_values))

    lr_permutation_cases[t_conn] = sequences
    lr_permutation_times[t_conn] = t_times

# ===============================================================================================
# ==== EVALUATE HULL CENTROIDS AND AREAS FOR EACH EVOLUTION, FIND CASES WITH LEAST DEVIATIONS====
# ===============================================================================================
# REMARK: evolutions with least path length and are changes should be right ones

if 1 == 1:
    t_all_traj      = {t_conn:[] for t_conn in lr_permutation_cases}
    t_all_areas     = {t_conn:[] for t_conn in lr_permutation_cases}
    t_all_moms      = {t_conn:[] for t_conn in lr_permutation_cases}
    t_sols_c        = {t_conn:[] for t_conn in lr_permutation_cases}
    t_sols_c_i      = {t_conn:[] for t_conn in lr_permutation_cases}
    t_sols_a        = {t_conn:[] for t_conn in lr_permutation_cases}
    t_sols_m        = {t_conn:[] for t_conn in lr_permutation_cases}

    for t_conn in lr_permutation_cases:
        t_c_interp = lr_121_interpolation_centroids[t_conn]
        for t,t_perms in enumerate(lr_permutation_cases[t_conn]):
            # dump sequences of areas and centroids for each possible trajectory
            t_temp_c = []
            t_temp_a = []     
            t_temp_m = [] 
            for t_time,t_perm in zip(lr_permutation_times[t_conn],t_perms):
                t_temp_c.append(lr_permutation_centroids_precomputed[t_conn][t_time][t_perm])
                t_temp_a.append(lr_permutation_areas_precomputed[t_conn][t_time][t_perm])
                t_temp_m.append(lr_permutation_mom_z_precomputed[t_conn][t_time][t_perm])
            t_all_traj[t_conn].append(np.array(t_temp_c).reshape(-1,2))
            t_all_areas[t_conn].append(np.array(t_temp_a))
            t_all_moms[t_conn].append(np.array(t_temp_m))

        #
        t_c_inter_traj =  [t[1:-1]       for t in t_all_traj[t_conn]]
        t_c_inter_traj_diff =  [t - t_c_interp      for t in t_c_inter_traj]
        t_c_inter_traj_diff_norms = [np.linalg.norm(t, axis=1)  for t in t_c_inter_traj_diff]
        t_c_i_traj_d_norms_means  = [np.mean(t) for t in t_c_inter_traj_diff_norms]
        t_c_i_mean_min            = np.argmin(t_c_i_traj_d_norms_means)

        # check displacement norms, norm means and stdevs
        t_c_diffs = [np.diff(t,axis = 0)        for t in t_all_traj[t_conn]]
        t_c_norms = [np.linalg.norm(t, axis=1)  for t in t_c_diffs]
        t_c_means = np.mean(t_c_norms, axis=1)
        t_c_stdevs= np.std( t_c_norms, axis=1)

        t_c_mean_min    = np.argmin(t_c_means)
        t_c_stdevs_min  = np.argmin(t_c_stdevs)


        # same with areas
        t_areas = np.array(t_all_areas[t_conn])
        t_a_diffs = np.diff(t_areas, axis=1)
        t_a_d_abs = np.array([np.abs(t) for t in t_a_diffs])
        t_a_d_a_sum = np.sum(t_a_d_abs, axis = 1)
        t_a_means = np.mean(t_a_d_abs, axis=1)
        t_a_stdevs= np.std( t_a_d_abs, axis=1)

        t_a_mean_min    = np.argmin(t_a_means)
        t_a_stdevs_min  = np.argmin(t_a_stdevs)

        # same with moments
        t_moments = np.array(t_all_moms[t_conn])
        t_m_diffs = np.diff(t_moments, axis=1)
        t_m_d_abs = np.array([np.abs(t) for t in t_m_diffs])
        t_m_d_a_sum = np.sum(t_m_d_abs, axis = 1)
        t_m_means = np.mean(t_m_d_abs, axis=1)
        t_m_stdevs= np.std( t_m_d_abs, axis=1)

        t_m_mean_min    = np.argmin(t_m_means)
        t_m_stdevs_min  = np.argmin(t_m_stdevs)

        # save cases with least mean and stdev
        t_sols_c[t_conn] += [t_c_mean_min,t_c_stdevs_min]
        t_sols_c_i[t_conn] += [t_c_i_mean_min]
        t_sols_a[t_conn] += [t_a_mean_min,t_a_stdevs_min]
        t_sols_m[t_conn] += [t_m_mean_min,t_m_stdevs_min]
    a = 1

# ===============================================================================================
# ========== CALCULATED WEIGHTED SOLUTONS =========
# ===============================================================================================
# REMARK: give methods (centroid, area,..) different weights and calculate total weights
# REMARK: of each evolution occurence ID. Then re-eight solution to 0-1.

if 1 == 1:
    lr_weighted_solutions = {t_conn:{} for t_conn in lr_permutation_cases}

    # set normalized weights for methods
    lr_weight_c    = 1
    lr_weight_c_i  = 1
    lr_weight_a    = 1
    lr_weight_m    = 1
    lr_weights = np.array([lr_weight_c, lr_weight_c_i,lr_weight_a, lr_weight_m])
    lr_weight_total = np.sum(lr_weights)
    lr_weight_c, lr_weight_c_i,lr_weight_a, lr_weight_m = lr_weights / lr_weight_total

    for t_conn in lr_permutation_cases:
        t_all_sols = []
        t_all_sols += t_sols_c[t_conn]
        t_all_sols += t_sols_c_i[t_conn]
        t_all_sols += t_sols_a[t_conn]
        t_all_sols += t_sols_m[t_conn]

        t_all_sols_unique = sorted(list(set(t_sols_c[t_conn] + t_sols_c_i[t_conn] + t_sols_a[t_conn] + t_sols_m[t_conn])))
     
        lr_weighted_solutions[t_conn] = {tID:0 for tID in t_all_sols_unique}
        for tID in t_all_sols_unique:
            if tID in t_sols_c[   t_conn]:
                lr_weighted_solutions[t_conn][tID] += lr_weight_c
            if tID in t_sols_c_i[   t_conn]:
                lr_weighted_solutions[t_conn][tID] += lr_weight_c_i
            if tID in t_sols_a[   t_conn]:
                lr_weighted_solutions[t_conn][tID] += lr_weight_a
            if tID in t_sols_m[   t_conn]:
                lr_weighted_solutions[t_conn][tID] += lr_weight_m
        # normalize weights for IDs
        r_total = np.sum([t_weight for t_weight in lr_weighted_solutions[t_conn].values()])
        lr_weighted_solutions[t_conn] = {tID:round(t_val/r_total,3) for tID,t_val in lr_weighted_solutions[t_conn].items()}

 
    
    lr_weighted_solutions_max = {t_conn:0 for t_conn in lr_permutation_cases}
    lr_weighted_solutions_accumulate_problems = {}
    for t_conn in lr_permutation_cases:
        t_weight_max = max(lr_weighted_solutions[t_conn].values())
        t_keys_max = [tID for tID, t_weight in lr_weighted_solutions[t_conn].items() if t_weight == t_weight_max]
        if len(t_keys_max) == 1:
            lr_weighted_solutions_max[t_conn] = t_keys_max[0]
        else:
            # >>>>>>>>>>VERY CUSTOM: take sol with max elements in total <<<<<<<<<<<<<<
            t_combs = [lr_permutation_cases[t_conn][tID] for tID in t_keys_max]
            t_combs_lens = [np.sum([len(t_perm) for t_perm in t_path]) for t_path in t_combs]
            t_sol = np.argmax(t_combs_lens) # picks first if there are same lengths
            lr_weighted_solutions_max[t_conn] = t_keys_max[t_sol]
            t_count = t_combs_lens.count(max(t_combs_lens))
            if t_count > 1: lr_weighted_solutions_accumulate_problems[t_conn] = t_combs_lens # holds poistion of t_keys_max, != all keys
            a = 1


# ===============================================================================================
# ========== INTEGRATE RESOLVED PATHS INTO GRAPH; REMOVE SECONDARY SOLUTIONS =========
# ===============================================================================================
# REMARK: refactor nodes from solo objects to clusters. remove all previous nodes and replace with new.

lr_weighted_solutions_max
lr_permutation_times
lr_permutation_cases
lr_contour_combs
for t_conn in lr_permutation_cases:
    t_sol   = lr_weighted_solutions_max[t_conn]
    t_path  = lr_permutation_cases[t_conn][t_sol]             # t_path contains start-end points of segments !!!
    t_times = lr_permutation_times[t_conn]
    #t_nodes_old = []
    t_nodes_new = []
    for t_time,t_comb in zip(t_times,t_path):
        #for tID in t_comb:
        #    t_nodes_old.append(tuple([t_time, tID]))          # old type of nodes in solution: (time,contourID)     e.g (t1,ID1)
        t_nodes_new.append(tuple([t_time] + list(t_comb)))    # new type of nodes in solution: (time,*clusterIDs)   e.g (t1,ID1,ID2,...)

    t_nodes_all = []
    for t_time,t_comb in lr_contour_combs[t_conn].items():
        for tID in t_comb:
            t_nodes_all.append(tuple([t_time, tID]))
    
    # its easy to remove start-end point nodes, but they will lose connection to segments
    G.remove_nodes_from(t_nodes_all)
    # so add extra nodes to make edges with segments, which will create start-end points again.
    t_from, t_to = t_conn                                     
    t_nodes_new_sides = [segments2[t_from][-2]] + t_nodes_new + [segments2[t_to][1]]

    pairs = [(x, y) for x, y in zip(t_nodes_new_sides[:-1], t_nodes_new_sides[1:])]
    
    G.add_edges_from(pairs)
    #t_nodes_remaining = [t_node for t_node in t_nodes_all if t_node not in t_nodes_old]
    


# ===============================================================================================
# ========== DEAL WITH ZERO PATH 121s >> SKIP IT FOR NOW, IDK WHAT TO DO << =========
# ===============================================================================================
# REMARK: zero paths might be part of branches, which means segment relations should be refactored

t_conn_121_zp_contour_combs

t_conn_121_other_terminated_failed

G_copy = G.copy()
node_positions_c = getNodePos(G_copy.nodes())
segments2_c, skipped_c = graph_extract_paths(G_copy, lambda x : x[0]) # 23/06/23 info in "extract paths from graphs.py"

# Draw extracted segments with bold lines and different color.
segments2_c = [a for _,a in segments2_c.items() if len(a) > 0]
segments2_c = list(sorted(segments2_c, key=lambda x: x[0][0]))
paths_c = {i:vals for i,vals in enumerate(segments2_c)}

#drawH(G, paths, node_positions)

#drawH(G_copy, paths_c, node_positions_c)

# ===============================================================================================
# ========= deal with 121s that are partly connected with merging/splitting segments << =========
# ===============================================================================================
# REMARK: 121s in 't_conn_121_other_terminated_inspect' are derived from split branch
# REMARK: or result in a merge branch. split->121 might be actually split->merge, same with 
# REMARK: 121->merge. these are ussualy short-lived, so i might combine them and test

t_conn_121_other_terminated_inspect
t_conn_121_other_terminated_failed
lr_conn_edges_splits
lr_conn_edges_merges
segment_conn_end_start_points(t_conn_121_other_terminated_inspect, nodes = 1)
lr_inspect_contour_combs = {t_conn:{} for t_conn in t_conn_121_other_terminated_inspect}
for t_conn in t_conn_121_other_terminated_inspect:
    t_from,t_to = t_conn
    spl = [(t_from_from, t_from_to) for t_from_from, t_from_to in lr_conn_edges_splits if t_from == t_from_to] # 121 left   is some other conn right
    mrg = [(t_to_from, t_to_to) for t_to_from, t_to_to in lr_conn_edges_merges if t_to == t_to_from]           # 121 right  is some other conn left

    # test on merge, no spl data yet
    if len(mrg)>0:
        t_to_merging_IDs = list(set([t[1] for t in mrg]))
        assert len(t_to_merging_IDs) == 1, "t_conn_121_other_terminated_inspect merge with multiple, this should not trigger"
        t_to_all_merging_segments = [t_to_from for t_to_from, t_to_to in lr_conn_edges_merges if t_to_to == t_to_merging_IDs[0] and t_to_from != t_to]

        t_from_time_start = segments2[t_from][-4:][0][0] # [-4:][0] = take at furthest from back of at least size 4. list([a,b])[-4:][0] = a
        t_to_from_time_start = segments2[t_to_merging_IDs[0]][:4][-1][0]

        activeNodes = [node for node in G.nodes() if t_from_time_start <= node[0] <= t_to_from_time_start]
        subgraph = G.subgraph(activeNodes)

        connected_components_all = [list(nx.node_connected_component(subgraph, key)) for key in activeNodes]

        connected_components_all = [sorted(sub, key=lambda x: (x[0],x[1])) for sub in connected_components_all] # key=lambda x: (x[0],x[1]))
        connected_components_unique = []
        [connected_components_unique.append(x) for x in connected_components_all if x not in connected_components_unique]




        sol = [t_cc for t_cc in connected_components_unique if segments2[t_from][-1] in t_cc]
        assert len(sol) == 1, "t_conn_121_other_terminated_inspect inspect path relates to multiple clusters, dont expect it ever to occur"

        t_times = np.arange(t_from_time_start,t_to_from_time_start + 1, 1)
        t_inspect_contour_combs = {t_time:[] for t_time in t_times}
    
        for t_time,*t_subIDs in sol[0]:
            t_inspect_contour_combs[t_time] += t_subIDs

        for t_time in t_inspect_contour_combs:
            t_inspect_contour_combs[t_time] = sorted(t_inspect_contour_combs[t_time])
        
        lr_inspect_contour_combs[t_conn] = t_inspect_contour_combs

    else:
        assert 1 == -1, "t_conn_121_other_terminated_inspect case split unexplored" 
    

a = 1
# ===============================================================================================
# ========== INTERPOLATE PATHS BETWEEN SEGMENTS AND COMPARE WITH INTER-SEGMENT SOLUTION =========
# ===============================================================================================
# REMARK: least path and area devs might not agree, test remaining choices with interpolation




def interpolateMiddle(t_conn,t_sols_c,t_sols_a,t_segments_121_centroids,t_all_traj, lr_permutation_times, segments2, histLen = 5, s = 15, debug = 0):
    t_from,t_to = t_conn
    t_possible_sols = sorted(list(set(t_sols_c[t_conn] + t_sols_a[t_conn])))
    t_trajectories = {tID:t_all_traj[t_conn][tID] for tID in t_possible_sols}
    t_hist_prev     = segments2[t_from][-histLen:]
    t_hist_next     = segments2[t_to][:histLen]
    t_traj_prev     = np.array([t_segments_121_centroids[t_from][t_node] for t_node in t_hist_prev])
    t_traj_next     = np.array([t_segments_121_centroids[  t_to][t_node] for t_node in t_hist_next])
    t_times_prev    = [t_node[0] for t_node in t_hist_prev]
    t_times_next    = [t_node[0] for t_node in t_hist_next]
    t_traj_concat   = np.concatenate((t_traj_prev,t_traj_next))
    t_times         = t_times_prev + t_times_next
    x, y            = t_traj_concat.T
    spline, _       = interpolate.splprep([x, y], u=t_times, s=s,k=1)

    t2              = np.arange(t_times[0],t_times[-1],0.1)
    IEpolation      = np.array(interpolate.splev(t2, spline,ext=0))
    t_missing       = lr_permutation_times[t_conn][1:-1]
    t_interp_missing= np.array(interpolate.splev(t_missing, spline,ext=0)).T   # notice transpose sempai :3

    t_sol_diffs     = {}
    for t_sol, t_traj in t_trajectories.items():
        t_diffs = t_interp_missing- t_traj[1:-1]
        t_sol_diffs[t_sol] = [np.linalg.norm(t_diff) for t_diff in t_diffs]
        
    if debug:
        fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
        axes.plot(*t_traj_prev.T, '-o', c= 'black')
        axes.plot(*t_traj_next.T, '-o', c= 'black')
        
        axes.plot(*IEpolation, linestyle='dotted', c='orange')
    

        for t_sol, t_traj in t_trajectories.items():
            axes.plot(*t_traj.T, '-o', label = t_sol)
        axes.set_title(t_conn)
        axes.set_aspect('equal')
        axes.legend(prop={'size': 6})
        plt.show()
    return t_sol_diffs

# tested this second, first below cssssssssssssss
sol = []
for t_conn in t_sols_c:#[(18,19)]
    aa = interpolateMiddle(t_conn,t_sols_c,t_sols_a,t_segments_121_centroids,t_all_traj,lr_permutation_times, segments2, debug = 0)
    sol.append(aa)
a = 1
    

for t_conn in t_sols_c:#[(18,19)]
    t_from,t_to = t_conn
    t_possible_sols = sorted(list(set(t_sols_c[t_conn] + t_sols_a[t_conn])))
    t_trajectories = {tID:t_all_traj[t_conn][tID] for tID in t_possible_sols}
    t_hist_prev     = segments2[t_from][-4:-1]
    t_hist_next     = segments2[t_to][1:4]
    t_traj_prev     = np.array([t_segments_121_centroids[t_from][t_node] for t_node in t_hist_prev])
    t_traj_next     = np.array([t_segments_121_centroids[  t_to][t_node] for t_node in t_hist_next])
    t_times_prev    = [t_node[0] for t_node in t_hist_prev]
    t_times_next    = [t_node[0] for t_node in t_hist_next]
    #t_mean_displ = {tID:np.diff(t_traj,axis = 0)       for tID, t_traj in t_trajectories.items()}
    t_mean_displ = {tID:np.mean(np.diff(t_traj,axis = 0) ,axis = 0) for tID, t_traj in t_trajectories.items()}
    for t_sol, t_traj in t_trajectories.items():
        #t_diag = t_mean_displ[t_sol]
        #t_diag_inv = 1/t_diag
        #t_scale_inv = np.diag(t_diag)
        #t_scale = np.diag(t_diag_inv)
        #t_traj_scaled = np.dot(t_traj,t_scale)#t_scale @ t_traj.T #np.matmul(t_scale,)
        #t = np.diff(t_traj_scaled,axis = 0)
        #t_traj_2 = np.dot(t_traj_scaled,t_scale_inv)

        #x,y = t_traj_scaled.T
        x0,y0 = t_traj.T
        t_concat = np.concatenate((t_traj_prev,t_traj,t_traj_next))
        x,y = t_concat.T
        t0 = lr_permutation_times[t_conn]
        t_times = t_times_prev + list(t0) + t_times_next
        spline, _ = interpolate.splprep([x, y], u=t_times, s=10,k=1)

        #t2 = np.arange(t0[0],t0[-1],0.1)
        t2 = np.arange(t_times[0],t_times[-1],0.1)
        IEpolation = np.array(interpolate.splev(t2, spline,ext=0))
    
        fig, axes = plt.subplots(1, 1, figsize=( 1*5,5), sharex=True, sharey=True)
        
        for t,t_perms in  enumerate(lr_permutation_cases[t_conn][t_sol]):
            t_time  = t0[t]
            t_hull = cv2.convexHull(np.vstack([g0_contours[t_time][t_subID] for t_subID in t_perms]))
            axes.plot(*t_hull.reshape(-1,2).T, '-',  c='black', linewidth = 0.3)
            for tID in t_perms:
                axes.plot(*g0_contours[t_time][tID].reshape(-1,2).T, '-',  c=np.array(cyclicColor(t))/255, linewidth = 0.5)
        axes.plot(*t_traj_prev.T, '-o',  c='black')
        axes.plot(*t_traj_next.T, '-o',  c='black')
        axes.plot(x0,y0, '-o')
        
        axes.plot(*IEpolation, linestyle='dotted', c='orange')
        axes.set_title(t_conn)
        axes.set_aspect('equal')
        fig.show()
    a = 1


for t_conn, test in lr_contour_combs.items():
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