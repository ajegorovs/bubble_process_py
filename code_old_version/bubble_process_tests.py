# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:30:19 2022

@author: User
"""
#print("\014") #clear spyder console, if you use it
from ast import For
import cv2, csv
# from cv2.ximgproc import anisotropicDiffusion
# from cv2.ximgproc import getDisparityVis
# from skimage.segmentation import (morphological_chan_vese,
#                                   morphological_geodesic_active_contour,
#                                   inverse_gaussian_gradient,
#                                   checkerboard_level_set)
import numpy as np, itertools, networkx as nx
import os, glob, datetime, re#, multiprocessing
# import glob
from matplotlib import pyplot as plt
# from skimage.morphology import disk
from skimage.morphology import medial_axis, skeletonize
from itertools import combinations, permutations
# from skimage.metrics import structural_similarity
import alphashape
from descartes import PolygonPatch
generateMasks = 0
cropUsingMask = 1
imageMainFolder = r'D:/Alex/Darbs.exe/Python_general/bubble_process/imageMainFolder/'
# imageMainFolder = r'F:/bubble archives/100 sccm/'
# imageMainFolder = r'D:/Alex/Darbs.exe/Python_general/bubble_process/mihails/'
imageMainFolderName = list(filter(None, os.path.split(imageMainFolder))) #del ""
mapXY = (np.load('./mapx.npy'), np.load('./mapy.npy'))
font = cv2.FONT_HERSHEY_SIMPLEX
# if cropUsingMask == 1 and not generateMasks == 1:
#     k = 0
#     for i in countImages:
#         parallelMaskLinks.extend([masks[k]]*i)
#         k += 1
#     parallelSauce = list(zip(imageLinks,parallelMaskLinks))

adjustBrightness        = 0
global runTh
runTh = False
from imageFunctions import (resizeImage,convertGray2RGB,convertRGB2Gray,
                            maskDilateErode,maskSegmentAndHull,
                            getPaddedSegmentCoords,maskedBlend,
                            undistort,cropImage,adjustBrightness)
from imageFunctionsP2 import (initImport, init, bubbleTypeCheck,
                              drawContoursGeneral, boundingRect2Contour, cntParentChildHierarchy,
                              maskByValue, getCentroidPos,getCentroidPosContours, compareMoments,
                              checkMoments,closes_point_contours, distStatPrediction,detectStuckBubs
                              ,getMasksParams,getCentroidPosCentroidsAndAreas,centroidSumPermutationsMOD,
                              getContourHullArea, centroidAreaSumPermutations, centroidAreaSumPermutations2, listFormat,#centroidAreaSumPermutations2 04/03/23
                              distStatPredictionVect,distStatPredictionVect2,updateStat,overlappingRotatedRectangles,
                              multiContourBoundingRect,stuckBubHelp,doubleCritMinimum,dropDoubleCritCopies,
                              radialStatsImageEllipse,radialStatsContoursEllipse,compareRadial,                         # radialStatsImage,radialStatsContours,compareRadial 03/03/23
                              tempStore,tempStore2,alphashapeHullModded,mergeCrit,graphUniqueComponents,                # tempStore 17/03/23   alphashapeHullModded (replaced) 22(24)/03/23 ,mergeCrit 23/03/23, graphUniqueComponents 25/03/23 
                              interpolateHull,getOverlap,radialAnal,mergeSplitDetect,closestDistancesContours)         # interpolateHull 26/03/23 radialAnal 02/04/23   mergeSplitDetect 05/04/23, closestDistancesContours 12/04/23
def resizeImage(img,frac):
    width = int(img.shape[1] * frac)
    height = int(img.shape[0] * frac)
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def resizeToMaxHW(image,width=1200,height=600):
    w0= image.shape[1] 
    h0 = image.shape[0]
    alpha,beta = width/w0,height/h0
    if w0*beta > width:
        return resizeImage(image,alpha)
    else:
        return resizeImage(image,beta)
def print01(string,toggle):
    if toggle == 1:
        print(string)
# --------------------------------------------------------------------------      
        # What i want to get: 
        # detect single full bubbles (bubbleType = 1)- avg = 1.
        # detect single partial bubbles (bubbleType = 0)-  avg = 0
        # mix full + full, close by avg ~< 1
        # strat - count bubbles, then watershed them in two
        # mix full + partial close by 0<avg<1. full stays last
        # strat- erode partial, dilate back full, extract partial.
        # mix partial +partial close by avg = 0, no solution yet.

def debugOnly(section):
    global globalCounter, debugSections, debugSteps
    if debugSections[0] == -1 or debugSteps[0] == 0:
        return False
    if (section in debugSections or len(debugSections) == 0) and (globalCounter in debugSteps or len(debugSteps) ==0 ):
        return True
    else:
        return False

#  21- distance lines, 22- colored neighbors, 23 - match template & overlap contours, 31 - merge stuff
debugSectionsGFX = [12, 21, 22, 23, 31]
debugStepsGFX = [0]
def debugOnlyGFX(section):
    global globalCounter, debugSections, debugSteps
    if debugSectionsGFX[0] == -1 or debugStepsGFX[0] == 0:
        return False
    if (section in debugSectionsGFX or len(debugSectionsGFX) == 0) and (globalCounter in debugStepsGFX or len(debugStepsGFX) ==0 ):
        return True
    else:
        return False
predictVectorPathFolder = r'./debugImages/predictVect'
if not os.path.exists(predictVectorPathFolder): os.makedirs(predictVectorPathFolder)


def ID2S(arr, delimiter = '-'):
    return delimiter.join(list(map(str,sorted(arr))))
imgNum = 46

#imageMainFolder = r'D:/Alex/Darbs.exe/Python_general/bubble_process/imageMainFolder/'
#init(imageMainFolder,imgNum)




def exportFirstFrame(markFirstExport,dataStart):
    #global manualMasksFolder
    if markFirstExport == 1:

        orig0 = dataArchive[dataStart]
        orig = orig0 -cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT)
    
        orig[orig < 0] = 0                  # if orig0 > mean
        orig = np.array(orig, dtype = np.uint8)
        origTH = np.array(cv2.threshold(orig,thresh0,255,cv2.THRESH_BINARY)[1], dtype = np.uint8)
        err = cv2.morphologyEx(origTH.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        #exp = np.maximum.reduce([orig,origTH]) # element wise max() for two matrices. kind of useless
        #cv2.imshow('orig',orig)
        #cv2.imshow('origTH',origTH)
        #cv2.imshow('exp',exp)
        cv2.imwrite(os.path.join(manualMasksFolder, "frame"+str(dataStart).zfill(4)+".png") ,err)
        return 1
    else: return 0

#def extractManualMask(index = dataStart, debug  = 0): # either draw red masks over or using Paint, set bg color to red and freehand select and delete areas.
#    manualMask = cv2.imread(os.path.join(manualMasksFolder, "frame"+str(index).zfill(4)+" - Copy.png",),1)#"./manualMask/frame"+str(index).zfill(4)+" - Copy.png"
    
#    if debug == 1:
#        cv2.imshow(f'extractManualMask {index}: import', manualMask)
#        cv2.imshow(f'extractManualMask {index}: blue',  np.uint8(manualMask[:,:,0]))
#        cv2.imshow(f'extractManualMask {index}: green', np.uint8(manualMask[:,:,1]))
#        cv2.imshow(f'extractManualMask {index}: red',   np.uint8(manualMask[:,:,2]))
#    manualMask = np.uint8(manualMask[:,:,2])
#    manualMask = cv2.threshold(manualMask,230,255,cv2.THRESH_BINARY)[1]
#    if debug == 1: cv2.imshow(f'extractManualMask {index}: red', manualMask)
    
#    #cv2.imshow('manualMask',manualMask)
#    contours = cv2.findContours(manualMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
#    params = [cv2.boundingRect(c) for c in contours]
#    #for x,y,w,h in params:
#    #   cv2.rectangle(manualMask,(x,y),(x+w,y+h),128,3)
#    #cv2.drawContours(manualMask, contours, -1, 128, 5)
#    #cv2.imshow('manualMask',manualMask)
#    IDs = np.arange(0,len(params))
#    return contours,{ID:param for ID,param in enumerate(params)}


#typeFull,typeRing, typeRecoveredRing, typeElse, typeFrozen,typeRecoveredElse,typePreMerge,typeRecoveredFrozen = np.int8(0),np.int8(1),np.int8(2),np.int8(3), np.int8(4), np.int8(5), np.int8(6), np.int8(7)
def adjustBrightness(image,adjustBrightness):
    if adjustBrightness == 1:
        brightness = np.sum(image) / (255 * np.prod(image.shape))
        minimum_brightness = 0.66
        ratio = brightness / minimum_brightness
        if ratio >= 1:
            print("Image already bright enough")
            return image
        # Otherwise, adjust brightness to get the target brightness
        return cv2.convertScaleAbs(image, alpha = 1 / ratio, beta = 0)
    else:
        return image

def undistort(image):
    mapXY = (np.load('./mapx.npy'), np.load('./mapy.npy'))
    return cv2.remap(image,mapXY[0],mapXY[1],cv2.INTER_LINEAR)

def cropImage(image , importMaskLink,cropUsingMask):
    if cropUsingMask == 1:
        mask          = cv2.imread(importMaskLink,1)
        mask   = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)[:,:,0]
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        [X, Y, W, H] = cv2.boundingRect(contours[0])
        # cv2.imshow("mask" , image[Y:Y+H, X:X+W])
        return image[Y:Y+H, X:X+W]
    else:
        return image

def matchTemplateBub(image,template,rectParams,graphics=0,prefix = ""):
    # https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    (x0,y0,w0,h0) = rectParams
    offset = int(max(0.4*np.sqrt(w0**2+h0**2), 50)) # for small objects
    (x,y,w,h) = getPaddedSegmentCoords(image, x0, y0, w0, h0, offset)
    img = image.copy()[y:y+h, x:x+w]; tmplt = template.copy()
    # print(rectParams,tmplt.shape)
    res = cv2.matchTemplate(img,tmplt,cv2.TM_SQDIFF)#cv2.TM_CCOEFF_NORMED
    min_loc = np.array(cv2.minMaxLoc(res)[2])#;print(min_val)
    # min_loc = np.array(min_loc)
    min_loc += [x,y];[xf,yf] = min_loc
    if graphics == 1:
        gfx = convertGray2RGB(image.copy())
        blank = np.full(image.shape,0,dtype=np.uint8)
        blank[yf:yf + h0,xf:xf + w0] = tmplt
        blank2 = cv2.dilate(blank.copy(),np.ones((7,7),np.uint8),iterations = 1)
        msk = cv2.bitwise_xor(blank2, blank)
        gfx = maskedBlend(gfx,(0,0,255),msk,1)
        cv2.rectangle(gfx, (x,y), (x+w,y+h),(255,0,0),2)
        cv2.imshow(prefix+'matchTemplateBub',resizeToMaxHW(gfx))
    return [xf,yf,w0,h0]

def overlapingContours(contours, contourIDs, mask, maskParams, gfx = 0,prefix = ''):
    (x,y,w,h) = maskParams#;print(maskParams);print(mask.shape)
    overlapingContourIDList = list()
    for CID,cntr in zip(contourIDs,contours[contourIDs]): # <<<< maybe problem because missing children. very unlikely.
        # print(cntr)
        for [[x1,y1]] in cntr:
            if (x1>x and x1<x+w and y1>y and y1<y+h):
                if mask[y1-y,x1-x] == 255:
                    overlapingContourIDList.append(CID)
                    break
    if len(overlapingContourIDList)>0:
        tempCntrJoin = np.vstack([contours[k] for k in overlapingContourIDList])
        xt,yt,wt,ht = cv2.boundingRect(tempCntrJoin)
        tempMask = np.zeros((ht,wt), dtype=np.uint8)
        [cv2.drawContours( tempMask,   contours, k, 255, -1, offset = (-xt,-yt)) for k in overlapingContourIDList]
        
        if gfx == 1:
            # since i dont draw in global ref., some translation is required to align results. purely visual stuff.
            x_min,y_min = min(x,xt), min(y,yt)
            x_max,y_max = max(x+w,xt+wt), max(y+h,yt+ht) # bounding box coords
            base = cv2.copyMakeBorder(mask.copy(), top = y-y_min, bottom = y_max-(y+h), left = x-x_min,
                                            right = x_max-(x + w), borderType = cv2.BORDER_CONSTANT,
                                            dst = None, value = 0) # template padding
            [cv2.drawContours( base,   contours, k, 128, 2, offset = (-xt + (xt-x_min),-yt+ (yt-y_min))) for k in overlapingContourIDList]

            cv2.imshow(prefix+'overlapingContours',base)

        return tempMask, [xt,yt,wt,ht], overlapingContourIDList
    else:
        return None, [] , []
    
# cv2.imshow('asdas',immaa)
# cv2.imshow('asdas v2',resizeToMaxHW(immaa,800,640))


colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
np.random.seed(1);np.random.shuffle(colorList);np.random.seed()

def timeHMS():
    return datetime.datetime.now().strftime("%H-%M-%S")
def cyclicColor(index):
    return colorList[index % len(colorList)].tolist()


def getMergeCritParams(ellipseParamsDict,IDs,scaled,angleThreshold):
    ellipseParams       = [ellipseParamsDict[key] for key in IDs] 
    ellipseCentroids    = [np.array(c,int) for c,_,_ in ellipseParams]
    ccVector            = np.diff(ellipseCentroids,axis=0)[0]                                   # 2d vector with direction from one ellipse centroid to another
    ccPerpendicularDir  = np.matmul(np.array([[0, -1],[1, 0]]), ccVector)                       # rotate to get tanget direction
    ccPerpendicularDir  = ccPerpendicularDir/np.linalg.norm(ccPerpendicularDir)                 # normalize. ~ tangent direction to bubble interfaces
    avgRadii            =[int((a+b)/4) for _,(a,b),_ in ellipseParams]                          # average radii of ellipses
    refRadius           = min(avgRadii)                                                         # take smallest as reference
    midPoint            = np.average(ellipseCentroids, weights = np.flip(avgRadii), axis = 0).astype(int) # average radii weighted centroids ~ point of bubble merge
    return [int(scaled*refRadius),ccPerpendicularDir,angleThreshold,midPoint,None]                   # notice weights are swapped, merge point will be closer to smaller centroid


import pickle

def dumpPickle(exportFileName, data, folders=[]):
    # export data file with name exportFileName.pickle 
    # into directory root/folders[0]/folders[1]/...
    path = exportFileName+".pickle"                         # default path to root/filename
    if len(folders)>0:                                      # if specifict subfolders required
        pathFolder = '.'                                    # build them from root
        for folderName in folders:
            pathFolder = os.path.join(pathFolder,folderName)
            if not os.path.exists(pathFolder):              # if folder does not exist
                os.mkdir(pathFolder)                        # hierarchically build folder
        path = os.path.join(pathFolder,path)                # add final dir + file name
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)                           # export

toList = lambda x: [x] if type(x) != list else x

#grays = np.array(np.where((l_images_old[key] < 200) & (l_images_old[key] > 45),255,0),np.uint8)
#            graysA = int(np.sum(grays)/255)
#            graysM = int(np.sum(l_masks_old[key])/255)
#            if graysA/graysM > 0.7:

# =========== BUILD OUTPUT FOLDERS =============//
inputOutsideRoot            = 1                                                  # bmp images inside root, then input folder hierarchy will
mainInputImageFolder        = r'.\inputFolder'                                   # be created with final inputImageFolder, else custom. NOT USED?!?
inputImageFolder            = r'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\100 sccm' #
# 'F:\UL Data\Bubbles - Optical Imaging\Actual\Field OFF\Series 7\100 sccm'
#F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 265 mT\Series 3\350 sccm
# F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 265 mT\Series 4\300 sccm
#'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 265 mT\Series 3\350 sccm'
#'F:\UL Data\Bubbles - Optical Imaging\Actual\HFS 200 mT\Series 4\350 sccm'
mainOutputFolder            = r'.\imageMainFolder_output'                        # these are main themed folders, sub-projects go inside.
mainIntermediateDataFolder  = r'.\intermediateData'                              # these are main themed folders, sub-projects go inside.
mainManualMasksFolder       = r'.\manualMask'                                    # these are main themed folders, sub-projects go inside.
mainDataArchiveFolder       = r'.\archives'
for directory in [mainOutputFolder, mainIntermediateDataFolder, mainManualMasksFolder, mainInputImageFolder, mainDataArchiveFolder]:
        if not os.path.exists(directory):
             os.mkdir(directory) 

#mainOutputSubFolders = ['ringDetect']                                             # specific sub-project folder heirarchy
#mainOutputSubFolders = ['VFS 125 mT Series 5','sccm250-meanFix', "10000-15000"]                        # "00001-02500" "02500-05000"  "05000-07500" "07500-10000" "00001-05000" "05000-10000" "10000-15000"
mainOutputSubFolders =  ['HFS 200 mT Series 4','sccm100-meanFix', "00001-05000"]
# crop mask name is    mainOutputSubFolders[:2]   as project=> subproject and not subcase                                                                          

imageFolder             = mainOutputFolder                                       # defaults output to main folder
intermediateDataFolder  = mainIntermediateDataFolder                             # defaults output to main folder
manualMasksFolder       = mainManualMasksFolder                                  # defaults output to main folder
dataArchiveFolder       = mainDataArchiveFolder
for folderName in mainOutputSubFolders:                                          # or creates heararchy of folders "subfolder/subsub/.."
    
        
    imageFolder             = os.path.join(imageFolder,             folderName)
    intermediateDataFolder  = os.path.join(intermediateDataFolder,  folderName)
    manualMasksFolder       = os.path.join(manualMasksFolder,       folderName)
    dataArchiveFolder       = os.path.join(dataArchiveFolder,       folderName)
    if inputOutsideRoot == 0:
        inputImageFolder        = os.path.join(inputImageFolder,        folderName)
        for directory in [imageFolder, intermediateDataFolder, manualMasksFolder,dataArchiveFolder,inputImageFolder]:
            if not os.path.exists(directory):
                 os.mkdir(directory)
    else:
        for directory in [imageFolder, intermediateDataFolder, manualMasksFolder,dataArchiveFolder]:
            if not os.path.exists(directory):
                 os.mkdir(directory)

# //=========== BUILD OUTPUT FOLDERS =============


thresh0             = 3
thresh1             = 15


plotAreaRep         = 0
inspectBubbleDecay  = 211
testThreshold       = 222
testEdgePres        = 3

mode = 1 # mode: 0 - read new images into new array, 1- get one img from existing array, 2- recall pickled image
big = 1
# dataStart = 71+52 ###520
# dataNum = 7
dataStart           = 0 #736 #300 #  53+3+5
dataNum             = 5005#5005 #130 # 7+5   

assistManually      = 1
assistFramesG       = []    #845,810,1234 2070,2187,1396
assistFrames        = [a - dataStart for a in assistFramesG]
doIntermediateData              = 1                                         # dump backups ?
intermediateDataStepInterval    = 500                                       # dumps latest data field even N steps
readIntermediateData            = 1                                         # load backups ?
startIntermediateDataAtG        = 60700                             # frame stack index ( in X_Data). START FROM NEXT FRAME
startIntermediateDataAt         = startIntermediateDataAtG - dataStart      # to global, which is pseudo global ofc. global to dataStart
# ------------------- this manual if mode  is not 0, 1  or 2
workBigArray        = 0
recalcMean          = 0  
readSingleFromArray = 1

# this loads images from image list
pickleNewDataLoad   = 0
# this saves persepective-transformed and cropped imgs to pickle array
pickleNewDataSave   = 0

# this saves imgNum = 11 case into img0011.pickle case
pickleSingleCaseSave= 1
# bigRun = range(1815,1855,1)

# ------------------- 
drawFileName        = 1
# workSingleCase      = 0

markFirstMaskManually = 1
markFirstExport     = 0 # see exportFirstFrame() lower after dataArchive import

exportImages        = 1
exportImagesMHTX    = 0
drawAfter           = 0


globalCounter = 0

# debug section numbers: 11- RB IDs, 12- RB recovery, 21- Else IDs, 22- else recovery, 31- merge
# debug on specific steps- empty list, do all steps, or time steps in list
debugSections = [11,21,12,22,31]
debugSteps = [5]
debugVecPredict = 0



# ========================================================================================================
# ============== Import image files and process them, store in archive or import archive =================
# ========================================================================================================
exportArchive   = 0
if len(assistFramesG) > 0: exportArchive   = 0
rotateImageBy   = cv2.ROTATE_180 # -1= no rotation, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180 
startFrom       = 1   #0 or 1 for sub exp                               # offset from ordered list of images- global offset?! yes archive adds images from list as range(startFrom, numImages)
numImages       = 5005 # DONT! intervalStart is what you are after!!!!! # take this many, but it will be updated: min(dataNum,len(imageLinks)-startFrom), if there are less available
postfix         = "-00001-05000"

intervalStart   = 0                         # in ordered list of images start from number intervalStart
intervalStop    = intervalStart + numImages  # and end at number intervalStop
useMeanWindow   = 1                          # averaging intervals will overlap half widths, read more below
N               = 500                        # averaging window width
    
archivePath     = os.path.join(dataArchiveFolder, "-".join(mainOutputSubFolders)+postfix+".pickle")
meanImagePath   = os.path.join(dataArchiveFolder, "-".join(["mean"]+mainOutputSubFolders)+".pickle")
meanImagePathArr= os.path.join(dataArchiveFolder, "-".join(["meanArr"]+mainOutputSubFolders)+".pickle")
csv_path        = os.path.join(dataArchiveFolder, "-".join(["csv"]+mainOutputSubFolders)+".csv")          # for MHT-X, [frame,localID,centroid,area]
    
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
    cropMaskName = "-".join(mainOutputSubFolders[:2])+'-crop'
    cropMaskPath = os.path.join(mainManualMasksFolder, f"{cropMaskName}.png")
    
    if not os.path.exists(cropMaskPath):
        print(f"No crop mask in {mainManualMasksFolder} folder!, creating mask : {cropMaskName}.png")
        cv2.imwrite(cropMaskPath, convertGray2RGB(undistort(cv2.imread(imageLinks[0],0))))
        input( "modify it and press a key to continue..")
        cropMask = cv2.imread(cropMaskPath,1)
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
    for i,j in enumerate(range(startFrom, numImages)):
        if rotateImageBy != -1:
            dataArchive[i]    = cv2.rotate(undistort(cv2.imread (imageLinks[j],0))[Y:Y+H, X:X+W],rotateImageBy)
        else:
            dataArchive[i]    = undistort(cv2.imread (imageLinks[j],0))[Y:Y+H, X:X+W]
        
    with open(archivePath, 'wb') as handle: 
        pickle.dump(dataArchive, handle) 
    
    print(f"{timeHMS()}: Exporting mean image...")

    meanImage = np.mean(dataArchive, axis=0)

    with open(meanImagePath, 'wb') as handle:
        pickle.dump(meanImage, handle)
    
    print(f"{timeHMS()}: Processing and saving archive data on drive... Done!")

elif not os.path.exists(archivePath):
    print(f"{timeHMS()}: No archive detected! Please generate it from project images.")

else:
    print(f"{timeHMS()}: Existing archive found! Importing data...")
    with open(archivePath, 'rb') as handle:
        dataArchive = pickle.load(handle)
    print(f"{timeHMS()}: Existing archive found! Importing data... Done!")

    if not os.path.exists(meanImagePath):
        print(f"{timeHMS()}: No mean image found. Generating and saving new...")

        meanImage = np.mean(dataArchive, axis=0)

        with open(meanImagePath, 'wb') as handle:
            pickle.dump(meanImage, handle)
        print(f"{timeHMS()}: No mean image found. Generating and saving new... Done")
    else:
        with open(meanImagePath, 'rb') as handle:
            meanImage = pickle.load(handle)
#cv2.imshow(f'mean',meanImage.astype(np.uint8))



# =========================================================================================================
# discrete update moving average with window N, with intervcal overlap of N/2
# [-interval1-]         for first segment: interval [0,N]. switch to next window at i = 3/4*N,
#           |           which is middle of overlap. 
#       [-interval2-]   for second segment: inteval is [i-1/4*N, i+3/4*N]
#                 |     third switch 1/4*N +2*[i-1/4*N, i+3/4*N] and so on. N/2 between switches
if useMeanWindow == 1 :
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

#cv2.imshow('1000',np.uint8(masksArr[whichMaskInterval(1000,intervalIndecies)]))
#cv2.imshow('2000',np.uint8(masksArr[whichMaskInterval(2000,intervalIndecies)]))


# ====================================================================================================
# ======= blank full size (Cropped) images for MHT-X algorithm =======================================
# ====================================================================================================
#  get frames which are not already exported, export them on drawing start
csvImgDirectory = os.path.join(dataArchiveFolder,"-".join(["csv_img"]+mainOutputSubFolders))
if not os.path.exists(csvImgDirectory): os.mkdir(csvImgDirectory)
csvImageLinks               = glob.glob(csvImgDirectory + "**/*.png", recursive=True)
extractIntergerFromFileName = lambda x: int(re.findall('\d+', os.path.basename(x))[0])
csvImageLinksNumbers        = [extractIntergerFromFileName(x) for x in csvImageLinks]
allRelevantFrames           = np.arange(numImages)
exportCSVimgsIDs            = [x for x in allRelevantFrames if x not in csvImageLinksNumbers]
# ====================================================================================================
a = 1

def extractManualMask2(index = dataStart, debug  = 0): # either draw red masks over or using Paint, set bg color to red and freehand select and delete areas.
    #manualMask = cv2.imread("./manualMask/frame"+str(index).zfill(4)+" - Copy.png",1)
    manualMask = cv2.imread(os.path.join(manualMasksFolder, "frame"+str(index).zfill(4)+" - Copy.png",),1)
    if debug == 1:cv2.imshow(f'extractManualMask {index}: import', manualMask)
    manualMask = np.array(np.where((manualMask[:,:,1] > 10) & (manualMask[:,:,1] < 245),255,0),np.uint8)
    #manualMask = cv2.threshold(manualMask,230,255,cv2.THRESH_BINARY)[1]
    if debug == 1: cv2.imshow(f'extractManualMask {index}: red', manualMask)
    
    #cv2.imshow('manualMask',manualMask)
    contours = cv2.findContours(manualMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    params = [cv2.boundingRect(c) for c in contours]
    #for x,y,w,h in params:
    #   cv2.rectangle(manualMask,(x,y),(x+w,y+h),128,3)
    #cv2.drawContours(manualMask, contours, -1, 128, 5)
    #cv2.imshow('manualMask',manualMask)
    IDs = np.arange(0,len(params))
    return {ID:contour for ID,contour in enumerate(contours)},{ID:param for ID,param in enumerate(params)}

a = 1



#dataArchive, meanImage, imageLinks = initImport(mode,workBigArray,recalcMean,readSingleFromArray,pickleNewDataLoad,pickleNewDataSave,pickleSingleCaseSave)


[typeFull,typeRing, typeRecoveredRing, typeElse, typeFrozen,typeRecoveredElse,typePreMerge,typeRecoveredFrozen,typeMerge] = np.array([0,1,2,3,4,5,6,7,8])
typeStrFromTypeID = {tID:tStr for tID,tStr in zip(np.array([0,1,2,3,4,5,6,7,8]),['OB','RB', 'rRB', 'DB', 'FB', 'rDB', 'pm', 'rF', 'MB'])}
# g_(...) are global storage variable that contain info from all processed times.
# l_(...) are local storage variable that contain info from current time.
# l_(...)_old are global variables that contain info from previous time. its technically global to acces across two time steps, but are local by idea.
# (...)_r stands for recovered
g_Centroids, g_Rect_parms, g_Ellipse_parms, g_Areas, g_Masks, g_Images, g_old_new_IDs, g_bubble_type, g_child_contours = {}, {}, {}, {}, {}, {}, {}, {}, {}
g_areas_hull = {}#g_drop_keep_IDs = {}
l_RBub_masks_old, l_RBub_images_old, l_RBub_rect_parms_old, l_RBub_centroids_old, l_RBub_areas_hull_old, l_RBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
l_DBub_masks_old, l_DBub_images_old, l_DBub_rect_parms_old, l_DBub_centroids_old, l_DBub_areas_hull_old, l_DBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old, l_FBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
l_MBub_masks_old, l_MBub_images_old, l_MBub_rect_parms_old, l_MBub_centroids_old, l_MBub_areas_hull_old, l_MBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
frozenGlobal = {}; l_MBub_info_old, g_MBub_info, g_merges, g_areas_IDs, g_splits, l_splits_old = {}, {}, {}, {}, {}, {};frozenBuffer_old, frozenBufferSize,frozenBufferMaxDT = {}, 5, 50
allFrozenIDs, activeFrozen = {}, {}
g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull, g_FBub_glob_IDs = {},{},{},{}
g_contours, g_contours_hull,  frozenBubs, frozenBubsTimes,  l_bubble_type_old = {}, {}, {}, {}, {}
l_centroids_old_all, l_Areas_old,l_Areas_hull_old, l_rect_parms_all, l_rect_parms_all_old = {}, {}, {}, {}, {}
g_predict_displacement, g_predict_area_hull  = {}, {}; frozenIDs = []
l_masks_old, l_images_old, l_rect_parms_old, l_ellipse_parms_old, l_centroids_old, l_old_new_IDs_old, l_areas_hull_old, l_contours_hull_old = {}, {}, {}, {}, {}, {}, {}, {}
fakeBox, steepAngle, g_bublle_type_by_gc_by_type, g_dropIDs = {}, {} ,{}, {}   # inlet area rectangle
test666 = {};STBubs = {}
def mainer(index):
    global globalCounter, g_contours, g_contours_hull, frozenBubs, frozenBubsTimes, l_centroids_old_all, l_Areas_old, l_Areas_hull_old, l_rect_parms_all, l_rect_parms_all_old, g_areas_hull
    global l_RBub_masks_old, l_RBub_images_old, l_RBub_rect_parms_old, l_RBub_centroids_old, l_RBub_areas_hull_old, l_RBub_old_new_IDs_old
    global l_DBub_masks_old, l_DBub_images_old, l_DBub_rect_parms_old, l_DBub_centroids_old, l_DBub_areas_hull_old, l_DBub_old_new_IDs_old
    global l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old, l_FBub_old_new_IDs_old
    global l_MBub_masks_old, l_MBub_images_old, l_MBub_rect_parms_old, l_MBub_centroids_old, l_MBub_areas_hull_old, l_MBub_old_new_IDs_old
    global g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull, g_FBub_glob_IDs
    global g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Areas,g_Masks,g_Images,g_old_new_IDs,g_bubble_type,l_bubble_type_old,g_child_contours
    global g_predict_displacement, g_predict_area_hull, frozenIDs,  l_MBub_info_old, g_MBub_info, g_merges
    global l_masks_old, l_images_old, l_rect_parms_old, l_ellipse_parms_old, l_centroids_old, l_old_new_IDs_old, l_areas_hull_old, l_contours_hull_old
    global fakeBox, drawAfter, steepAngle, assistManually, assistFrames, dataStart, frozenBuffer_old, frozenBufferSize, allFrozenIDs, activeFrozen
    global debugVecPredict, contoursFilter_RectParams_dropIDs_old, contoursFilter_RectParams, contoursFilter_RectParams_dropIDs, frozenBufferMaxDT
    global frozenGlobal, imageFolder, g_bublle_type_by_gc_by_type, g_areas_IDs, g_dropIDs, breakLoopInsert, g_splits, l_splits_old, STBubs
    orig0           = dataArchive[index]
    # wanted to replace code below with cv2.subtract, but there are alot of problems with dtypes and results are a bit different
    if useMeanWindow == 1:
        meanImage = masksArr[whichMaskInterval(globalCounter,intervalIndecies)]

    orig            = orig0 -cv2.blur(meanImage, (5,5),cv2.BORDER_REFLECT)
    orig[orig < 0]  = 0                  # if orig0 > mean
    orig            = orig.astype(np.uint8)

    _,err           = cv2.threshold(orig.copy(),thresh0,255,cv2.THRESH_BINARY)
    err             = cv2.morphologyEx(err.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    #if globalCounter == 0: cv2.imshow(f'gc:{0}, binarization test', err)
    #if globalCounter in assistFrames and not os.path.exists(os.path.join(manualMasksFolder, "frame"+str(index).zfill(4)+".png")):
    #    cv2.imwrite(os.path.join(manualMasksFolder, "frame"+str(index).zfill(4)+".png") ,err)
    #    input('Writing assist frame... Modify it and press any key')
    #    breakLoopInsert = False
        
    if workBigArray == 1  and readSingleFromArray == 1: gfx = 1
    else: gfx = 0
    if workBigArray == 0: gfx = 1

    mergeCandidates,mergeCandidatesSubcontourIDs, l_predict_displacement, l_MBub_info, l_splits                                 = {}, {}, {}, {}, {}
    l_bubble_type, l_contours_hull, l_RBub_contours_hull, l_DBub_contours_hull, l_FBub_contours_hull, l_MBub_contours_hull      = {}, {}, {}, {}, {}, {}
    l_RBub_masks, l_RBub_images, l_RBub_rect_parms, l_RBub_centroids, l_RBub_areas_hull, l_RBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    l_DBub_masks, l_DBub_images, l_DBub_rect_parms, l_DBub_centroids, l_DBub_areas_hull, l_DBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    l_RBub_r_masks, l_RBub_r_images, l_RBub_r_rect_parms,l_RBub_r_centroids, l_RBub_r_areas_hull, l_RBub_r_old_new_IDs          = {}, {}, {}, {}, {}, {}
    l_DBub_r_masks, l_DBub_r_images, l_DBub_r_rect_parms,l_DBub_r_centroids, l_DBub_r_areas_hull, l_DBub_r_old_new_IDs          = {}, {}, {}, {}, {}, {}
    l_MBub_masks, l_MBub_images, l_MBub_rect_parms, l_MBub_centroids, l_MBub_areas_hull, l_MBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    l_FBub_masks, l_FBub_images, l_FBub_rect_parms, l_FBub_centroids, l_FBub_areas_hull, l_FBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    
    # get contours from binary image. filter out useless. 
    
    contoursFilter_RectParams_dropIDs,l_centroids_all = [], {}
    topFilter, bottomFilter, leftFilter, rightFilter, minArea    = 80, 40, 100, 100, 180                                                                                           # instead of deleting contours at inlet, i should merge
    # topFilter will drop bubbles on top, they will be resolved with new IDS. removed possible dropped RBs (whereChildrenAreaFiltered) from dropped                                                                                                                                      # them into one. since there is only ~one bubble at a time
    (l_contours,
     whereParentOriginal,
     whereParentAreaFiltered,
     whereChildrenAreaFiltered)             = cntParentChildHierarchy(err,1, 1200,130,0.1)                                                      # whereParentOriginal all non-child contours.
    g_contours[globalCounter]               = l_contours                                                                                        # add contours to global storage.
    contoursFilter_RectParams               = {ID: cv2.boundingRect(l_contours[ID]) for ID in whereParentOriginal}                              # remember bounding rectangle parameters for all primary contours.
    contoursFilter_RectParams_dropIDs       = [ID for ID,params in contoursFilter_RectParams.items() if (sum(params[0:3:2])<topFilter and
                                                                       ID not in whereChildrenAreaFiltered )]                                 # filter out bubbles at left image edge, keep those outside 80 pix boundary. x+w < 80 pix.
    contoursFilter_RectParams_dropIDs       += [ID for ID,params in contoursFilter_RectParams.items() if params[0]> err.shape[1]- bottomFilter]
    contoursFilter_RectParams_dropIDs       += [ID for ID,params in contoursFilter_RectParams.items() if params[1]> err.shape[0]- leftFilter]
    contoursFilter_RectParams_dropIDs       += [ID for ID,params in contoursFilter_RectParams.items() if params[1] + params[3] < rightFilter]
    #contoursFilter_RectParams_dropIDs_inlet = [ID for ID,params in contoursFilter_RectParams.items() if params[0] > err.shape[1]- bottomFilter] # top right corner is within box that starts at len(img)-len(box)
    #contoursFilter_RectParams_dropIDs       = contoursFilter_RectParams_dropIDs + contoursFilter_RectParams_dropIDs_inlet
    l_areas_all_IDs                         = [ID for ID in contoursFilter_RectParams if ID not in contoursFilter_RectParams_dropIDs]
    l_areas_all                             = {key: cv2.contourArea(l_contours[key]) for key in contoursFilter_RectParams if key not in contoursFilter_RectParams_dropIDs } # remember contour areas of main contours that are out of side band.
    l_areas_hull_all                        = {ID:getContourHullArea(l_contours[ID]) for ID in l_areas_all}                                         # convex hull of a single contour. for multiple contrours use getCentroidPosContours.
    contoursFilter_RectParams_dropIDs       = contoursFilter_RectParams_dropIDs + [key for key, area in l_areas_all.items() if area < minArea]      # list of useless contours- inside side band and too small area.
    l_rect_parms_all                        = {key: val for key, val in contoursFilter_RectParams.items() if key in l_areas_all}                    # bounding rectangle parameters for all primary except within a band.
    l_centroids_all                         = {key: getCentroidPosContours(bodyCntrs = [l_contours[key]])[0] for key in l_rect_parms_all}       # centroids of ^
    
    #frozenIDs = []
    
    #frozenLocal = []
 
    fakeBoxW, fakeBoxH      = 176,int(err.shape[0]/3)                                             # generate a fake bubble at inlet and add it to previous frame data.
    fakeBox                 = {-1:[err.shape[1] - fakeBoxW + 1, fakeBoxH, fakeBoxW, fakeBoxH]}    # this will make sure to gather new bubbles at inlet into single cluster.
    _,yFb,_,hFB = fakeBox[-1] 
    if globalCounter >= 1:
        # try to find old contours or cluster of contours that did not move during frame transition.
        # double overlay of old global and old local IDs, remove local & keep global.
        print(f'{globalCounter}:-------- Begin search for frozen bubbles ---------')
        dropKeys                = lambda lib,IDs: {key:val for key,val in lib.items() if key not in IDs}                # function that drops all keys listed in ids from dictionary lib
        deleteOverlaySoloIDs    = [subIDs[0] for _, subIDs in l_old_new_IDs_old.items() if len(subIDs) == 1]            # these are duplicates of global IDs from prev frame. ex: l_old_new_IDs_old= {0: [15]} -> l_Areas_old= {15:A} & joinAreas = {0:A} same object.
        #stuckAreas              = {**dropKeys(l_Areas_hull_old,deleteOverlaySoloIDs),     **{str(key):val for key,val in l_areas_hull_old.items()}} # replaced regular area with fb hull 11/02/23
        #stuckRectParams         = {**dropKeys(l_rect_parms_all_old,deleteOverlaySoloIDs), **{str(key):val for key,val in l_rect_parms_old.items()}}
        #stuckCentroids          = {**dropKeys(l_centroids_old_all,deleteOverlaySoloIDs),  **{str(key):val for key,val in l_centroids_old.items()}}
        for ID,timeDicts in frozenBuffer_old.items():                                                                   
            fTimes = list(timeDicts.keys())                                                                             
            [frozenBuffer_old[ID].pop(t, None) for t in fTimes if t < globalCounter - frozenBufferMaxDT]                # clean buffer from old info
        for ID in list(frozenBuffer_old.keys()).copy():                                                                  
            if len(frozenBuffer_old[ID]) == 0: frozenBuffer_old.pop(ID,None)                                            # delete empty entries

        for ID in l_rect_parms_old:                                                                                     # update info in buffer from old frame
            if ID not in frozenBuffer_old: frozenBuffer_old[ID] = {}                                                    # create new entry if missing
            else:
                fTimes = list(frozenBuffer_old[ID].keys())                                                              # grab times
                if len(fTimes) == frozenBufferSize: frozenBuffer_old[ID].pop(min(fTimes), None)                         # before adding new data check length
            frozenBuffer_old[ID][globalCounter-1] = [l_centroids_old[ID],l_areas_hull_old[ID],l_rect_parms_old[ID]]     # if full pop earliest, then add
        a = 1
        for ID in l_rect_parms_old:                                                                                     # only new information changes buffer. 
            if len(frozenBuffer_old[ID])>=2:                                                                            # no need to recalculate everything each step.
                path    = np.array([centroid for centroid,_,_  in frozenBuffer_old[ID].values()], int)                  # recalculate data based
                areas   = np.array([area     for _,area,_      in frozenBuffer_old[ID].values()], int)                  # on information about 
                displacementFromStart       = path-path[0]                                                              # last frozenBufferSize entries
                displacementFromStartAbs    = np.linalg.norm(displacementFromStart[1:],axis=1)
                displacementFromStartMean   = np.mean(displacementFromStartAbs)                                         # if moving bubble slows down and stops
                displacementFromStartStd    = np.std(displacementFromStartAbs)                                          # it will slowly reduce its mean displ.
                areaMean                    = np.mean(areas).astype(int)
                areaStd                     = np.std(areas).astype(int)                                                 # new frozen bubbles may be triggered faster
                if displacementFromStartMean < 5:                                                                       # moving - depends on frozenBufferSize
                    if ID not in activeFrozen:                                                                          # first time frozen
                        activeFrozen[ID] = []
                        counter = 0
                    else: counter = activeFrozen[ID][2]                                                                 # how many times detected
                    cntrd = list(np.mean(path,axis=0).astype(int))                                                      
                    activeFrozen[ID] = [[displacementFromStartMean,displacementFromStartStd],
                                        [areaMean,areaStd],                                                             # store frozen data
                                        counter+1, cntrd, l_rect_parms_old[ID]]
                    if globalCounter not in allFrozenIDs: allFrozenIDs[globalCounter] = []
                    allFrozenIDs[globalCounter].append(ID)                                                              # add info when detected as frozen
                else: activeFrozen.pop(ID,None)                                                                         # if it fails frozen check try to purge
                                                                                                                        # from active list.
        for ID in list(activeFrozen.keys()).copy():                                                                     # stop tracking IDs that
            if ID not in frozenBuffer_old: activeFrozen.pop(ID,None)                                                    # are out of buffer

        # ==============================================================================================================
        # ===================== Find which old resolved cluster elements did not move ==================================
        declusterNewFRLocals = []
        overlapFRLoc = []
        NStepsLocalFreeze = 2                                                                 # NStepsLocalFreeze steps in total except current
        locFRAreaThreshold = 600
        if globalCounter> NStepsLocalFreeze:
            lastNSteps = range(globalCounter-NStepsLocalFreeze, globalCounter, 1)
            nonFrozenGlobIDsTimes = {time:sum(                                                # these are non frozen global IDs present last steps
                                                [IDs for bType,IDs in g_bublle_type_by_gc_by_type[time].items() if bType != typeFrozen],
                                                []) for time in lastNSteps}
            resolvedLocalIDsTimes = {time:sum(                                                # these are local IDs of non frozen globals
                                                [g_old_new_IDs[subID][time] for subID in gIDs],
                                                []) for time,gIDs in nonFrozenGlobIDsTimes.items()}
            relevantContoursTimes = {time:{subID:g_contours[time][subID] for subID in IDs} for time,IDs in resolvedLocalIDsTimes.items()}

            relevantBRTimes       = {time:                                                    # all boundingRect params
                                            {ID: cv2.boundingRect(contour) for ID, contour in vals.items()}
                                    for time, vals in relevantContoursTimes.items()}
            relevantBRTimesFilter = {time:                                                    # filter out big ones
                                            {ID: [x,y,w,h] for ID, [x,y,w,h] in vals.items() if w*h < locFRAreaThreshold}
                                    for time, vals in relevantBRTimes.items()}
            step1RPtime,step2RPtime = list(lastNSteps)[-2:]                                   # manually select last 2 steps
            step1RP = {str(ID): params for ID,params in relevantBRTimesFilter[step1RPtime].items()}
            step2RP = {ID: params for ID,params in relevantBRTimesFilter[step2RPtime].items()}
            overlap = overlappingRotatedRectangles(step1RP, step2RP)                          # find overlap of small resolved subIds
            allOldLIDs  = [a[0] for a in overlap]
            allNextLIDs = [a[1] for a in overlap]
            oldCAs      = {ID:getCentroidPosContours(bodyCntrs = [relevantContoursTimes[step1RPtime][int(ID)]]) for ID in allOldLIDs}
            nextCAs     = {ID:getCentroidPosContours(bodyCntrs = [relevantContoursTimes[step2RPtime][int(ID)]]) for ID in allNextLIDs}
            calcD = lambda ID1,ID2: np.linalg.norm(np.array(oldCAs[ID1][0]) - np.array(nextCAs[ID2][0]))
            calcA = lambda ID1,ID2: np.abs(oldCAs[ID1][1]-nextCAs[ID2][1])/nextCAs[ID2][1]
            overlapDA   = [[calcD(ID1,ID2),calcA(ID1,ID2)] for ID1,ID2 in overlap]            # find c-c dist and rel area of overlap combos
            overlapPass = [True if a<=3 and b< 0.4 else False for a,b in overlapDA]
            overlapFRLoc = [combo for i,combo in enumerate(overlap) if overlapPass[i] == True]# take ones that are close dist and area

            # ================ continued after global Frozens are resolved and dropped from current locals =================
            # ==============================================================================================================

            #splitCentroidsAreas     = {ID:getCentroidPosContours(bodyCntrs = [hull]) for ID,hull in splitHulls.items()}   
        #l_rect_parms_old
        #cntrRemainingIDs = [cID for cID in whereParentOriginal if cID not in contoursFilter_RectParams_dropIDs ] #+ dropAllRings + dropRestoredIDs
        a  = 1
        #distContours = {ID:contoursFilter_RectParams[ID] for ID in cntrRemainingIDs}
        minAreaInlet                    = minArea
        inletIDsNew, inletIDsTypeNew    = overlappingRotatedRectangles(
                                            {ID:val for ID,val in l_rect_parms_all.items() if ID not in contoursFilter_RectParams_dropIDs},
                                           fakeBox, returnType = 1)          # inletIDs, inletIDsType after frozen part
        inletIDsNew                     = [a[0] for a in inletIDsNew]
        inletFullyInside                = [ID for ID,bType in inletIDsTypeNew.items() if bType == 2]
        #inletIDsNew                     =  [ID for ID in inletIDsNew if cv2.contourArea(l_contours[ID]) > minAreaInlet]
        #[contoursFilter_RectParams_dropIDs.remove(x) for x in inletIDsNew if x in contoursFilter_RectParams_dropIDs]                # remove-drop == which to keep.
        #dropKeysOld                                             = lambda lib: dropKeys(lib,contoursFilter_RectParams_dropIDs_old)   # dropping frozens from inlet since
        dropKeysNew                                             = lambda lib: dropKeys(lib,contoursFilter_RectParams_dropIDs + inletFullyInside)  # bubbles act bad there, might false-positive
        #[stuckRectParams,stuckAreas,stuckCentroids]             = list(map(dropKeysOld,[stuckRectParams,stuckAreas,stuckCentroids] ))
        [fbStoreRectParams2,fbStoreAreas2,fbStoreCentroids2]    = list(map(dropKeysNew,[l_rect_parms_all,l_areas_all,l_centroids_all] ))
        a = 1
        if 1 == 1:
            activeFBub_RP = {str(ID):vals[4] for ID,vals in activeFrozen.items()}
            intersectingCombs = overlappingRotatedRectangles(activeFBub_RP,fbStoreRectParams2)
            cc_unique   = graphUniqueComponents([str(ID) for ID in activeFBub_RP.keys()], intersectingCombs)          # clusters. if one real FB has multiple IDs
            cc_unique   = [comb for comb in cc_unique if len(comb)>1]
            #cc_unique = [['a',1,2],['b','c',3],['d','e',4,5]]
            fbInCC = [[subID for subID in comb if type(subID) == str] for comb in cc_unique ]                         # they will both trigger match. if there is one.
            notfbInCC = [[subID for subID in comb if type(subID) != str] for comb in cc_unique ]                      # split clusters into oldFB and others
            a = 1
            frCulpr = []
            for i,c in enumerate(fbInCC):                                       # fbInCC: [[a],[d],[b,c]], notfbInCC:[[1],[2,3],[4,5]]
                cases = []                                                      #
                for fID in c:                                                   # if [b,c]- > take [b,perms(4,5)] calc best
                    a = 1
                    predictCentroid = np.array(activeFrozen[int(fID)][3],int)   # then sol_b = [b,dist_b,subIDs_b]
                    dist,dStd       = activeFrozen[int(fID)][0]                 # then sol_c = [c,dist_c,subIDs_c]
                    area,aStd          = activeFrozen[int(fID)][1]                 # then sol is lowest dist_ 
                    numCounts       = activeFrozen[int(fID)][2]
                    if numCounts < 3:
                        dist += 2
                    dStd = max(1,dStd)
                    aStd = max(0.1*area, aStd)
                    permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(l_contours,fbStoreRectParams2, activeFBub_RP[fID], notfbInCC[i], fbStoreCentroids2, fbStoreAreas2,
                                                    predictCentroid,dist + 3*dStd, area, relAreaCheck = 0.7, debug = 0)
                    if len(permIDsol2)>0:
                        area2 = cv2.contourArea(cv2.convexHull(np.vstack(l_contours[permIDsol2])))
                        cases.append([int(fID), permIDsol2, permDist2, area, aStd, area2])
                if len(cases)>0:
                    where = np.argmin([b[2] for b in cases])
                    frCulpr.append(cases[where])
        for old,new, dist,areaOld, aStdOld, areaNew in frCulpr:
            areaCrit  = np.abs(areaNew-areaOld)/ aStdOld  
            if areaCrit < 3:
                test666[globalCounter] = []
                test666[globalCounter].append(frCulpr)
   
            # when dist sols
                #a = 1
                    #subPerms = sum([list(itertools.combinations(notfbInCC[i], r)) for r in range(1,len(notfbInCC[i])+1)],[])
                    #allSubCombs = [[fID]+list(elem) for elem in subPerms]
            #    if len(c) > :
            #        1
        #if len(intersectingCombs)>0:

        a = 1
        # frozen bubbles, other than from previous time step should be considered.================================
        # take frozen bubbles from last N steps. frozen bubbles found in old step are already accounted in else/dist data, get all other.
        #lastStepFrozenGlobIDs           = list(l_FBub_old_new_IDs_old.keys())                   # Frozens from last step
        #lastNStepsFrozen                = 10
        #lastNStepsFrozenTimesAll        = [time for time in frozenGlobal.keys() if max(globalCounter-lastNStepsFrozen-1,0)<time<globalCounter] # if time is in [0 1 2 3 4 5], lastNStepsFrozen = 3, globalCounter = 5 -> out = [2, 3, 4]
        #lastNStepsFrozenIDsExceptOld    = set([a for a in sum(frozenGlobal.values(),[]) if a not in lastStepFrozenGlobIDs]) if len(lastNStepsFrozenTimesAll)>0 else []
        #lastNStepsFrozenLatestTimes     = {ID:max([time for time,IDs in frozenGlobal.items() if ID in IDs]) for ID in lastNStepsFrozenIDsExceptOld}
        #lastNStepsFrozenRectParams      = {str(ID):g_FBub_rect_parms[latestTime][ID]     for ID,latestTime in lastNStepsFrozenLatestTimes.items()}
        #lastNStepsFrozenHullAreas       = {str(ID):g_FBub_areas_hull[latestTime][ID]     for ID,latestTime in lastNStepsFrozenLatestTimes.items()}
        #lastNStepsFrozenCentroids       = {str(ID):g_FBub_centroids[latestTime][ID]      for ID,latestTime in lastNStepsFrozenLatestTimes.items()}
        
        #stuckRectParams = {**stuckRectParams,   **lastNStepsFrozenRectParams}
        #stuckAreas      = {**stuckAreas,        **lastNStepsFrozenHullAreas}
        #stuckCentroids  = {**stuckCentroids,    **lastNStepsFrozenCentroids}

        #fields = [stuckRectParams,fbStoreRectParams2,stuckAreas,fbStoreAreas2,stuckCentroids,fbStoreCentroids2]
        
        #frozenIDs,frozenIDsInfo = detectStuckBubs(*fields, l_contours, globalCounter, frozenLocal, relArea = 0.0, relDist = 0, maxAngle = 1, maxArea = 1) # TODO breaks on higher relDist, probly not being split correctly
        #print(f'frozenLocal:{frozenLocal}')

        #[cv2.drawContours( blank, l_contours, ID, cyclicColor(key), -1) for ID in cntrIDlist]    # IMSHOW
        # store frozen bubble info for this step using local old IDs. could use only _old field instead, but this is more consistent with other fields 10/02/23
        #frozenOldGlobNewLoc = {}
        #for oldLocalID, newLocalIDs,_,_,centroid in frozenIDsInfo: #["oldLocID: ", ", newLocID: ",", c-c dist: ",", rArea: ",", centroid: "]

        #    dataSets = [l_FBub_masks,l_FBub_images,l_FBub_old_new_IDs,l_FBub_rect_parms,l_FBub_centroids,l_FBub_areas_hull,l_FBub_contours_hull]
        #    tempStore2(newLocalIDs, l_contours, oldLocalID, err, orig, dataSets, concave = 0)
            
            #findOldGlobIds = []; [findOldGlobIds.append(globID) for globID, locIDs in l_old_new_IDs_old.items() if oldLocalID in locIDs]
            #if type(oldLocalID) == str:                                                                # if old ID already has global ID (str -> global, int -> local)
            #    if int(oldLocalID) in l_old_new_IDs_old or oldLocalID in lastNStepsFrozenRectParams:   # in case oldLocalID is old Else bubble ~='7' 11/02/23 or frozen from N step back 18/02/23
            #        #findOldGlobIds.append(oldLocalID)
            #        frozenOldGlobNewLoc[min(newLocalIDs)] = oldLocalID                                 # min(newLocalIDs) ! kind of correct, might cause problems 10/02/23/ modded 19/02/23
            #else: frozenOldGlobNewLoc[min(newLocalIDs)] = oldLocalID                                   # == if it was single local, then it should have a global, if its part of old cluster, it must get new ID 19/02/23 
            #if type(oldLocalID) == str and int(oldLocalID) in l_old_new_IDs_old: findOldGlobIds.append(oldLocalID) # in case oldLocalID is old Else bubble ~='7' 11/02/23
            #if type(oldLocalID) == str and oldLocalID in lastNStepsFrozenRectParams: findOldGlobIds.append(oldLocalID) # frozen from N step back 18/02/23
            
        
            #frozenKeys = list(frozenOldGlobNewLoc.values()) wrong 11/02/23
            # !!! detectStuckBubs -> centroidAreaSumPermutations when permutation search fails returns empty array. it fucks shit up
        # !!! centroidAreaSumPermutations is used elsewhere, i think there its dealth with. 
        #print(f'frozenIDs:{frozenIDs}')
        #remIDs = [];print(f'remIDs:{remIDs}')
        #for i, a in enumerate(frozenIDs): 
        #    if type(a) == list:
        #        if len(a) == 0 :
        #            print("asdasdsadsad!!!!!!!!!!!!!!!!!!!!!")
        #            remIDs.append(i)
        ##frozenIDs = [a for a in frozenIDs if type(a) != list ]
        #print(f'remIDs:{remIDs}')
        #frozenIDs = [a for i,a in enumerate(frozenIDs) if i not in remIDs ] #
        #frozenIDsInfo = np.array([a for i,a in enumerate(frozenIDsInfo) if i not in remIDs ], dtype=object)
        #print(f'frozenIDs:{frozenIDs}')

        #if  1 == 1:
        #    strRavel = ["oldLocID: ", ", newLocID: ",", c-c dist: ",", rArea: ",", centroid: "]
        #    formats = ["{}", "{}", "{:.2f}", "{:.2f}", "{}"]
        #    frozenIDsInfoFMRT = [[frmt.format(nbr) for nbr,frmt in zip(elem, formats)] for elem in frozenIDsInfo] if len(frozenIDsInfo)> 0 else []
        #    peps = ["".join(np.ravel([strRavel,k], order='F')) for k in frozenIDsInfoFMRT] if len(frozenIDsInfoFMRT)> 0 else []
        #    peps = "\n".join(peps) if len(frozenIDsInfo)> 0 else []
        #    print(f'Detected Frozen bubbles:\n{peps}')
        #print(f'{globalCounter}:------------- Frozen bubbles detected ------------\n') if len(frozenIDsInfo) > 0 else print(f'{globalCounter}:------------- No frozen bubbles found ------------\n')
        
        #for IDS in [bID for bID,bType in l_bubble_type_old.items() if bType == typeFrozen]:
        #    l_predict_displacement[IDS] = [tuple(map(int,stuckCentroids[str(IDS)])), -1]
    
        inletIDs, inletIDsType  = overlappingRotatedRectangles(l_rect_parms_old, fakeBox, returnType = 1, typeAreaThreshold = 0.15)           # inletIDs: which old IDs are close to inlet? 
        #inletIDs                = [a[0] for a in inletIDs if a[0] not in lastStepFrozenGlobIDs]                     # inletIDsType: 1 & 2 - partially & fully inside.
        inletIDs                = [a[0] for a in inletIDs]                     # inletIDsType: 1 & 2 - partially & fully inside.
        inletIDsType            = {ID:inletIDsType[ID] for ID in inletIDs} 
  
    unresolvedOldRB   = list(l_RBub_centroids_old.keys())     # list of global IDs for RB  
    unresolvedOldDB   = list(l_DBub_centroids_old.keys()) + list(l_FBub_centroids_old.keys())  +  list(l_MBub_centroids_old.keys())   # and DB, from previous step. empty if gc = 0
    unresolvedNewRB   = list(whereChildrenAreaFiltered.keys())
 
    print("unresolvedNewRB (new typeRing bubbles):\n",  unresolvedNewRB)
    print("unresolvedOldRB (all old R,rR bubbles):\n",  unresolvedOldRB)
    print("unresolvedOlDRB (all old D,rD bubbles):\n",  unresolvedOldDB)
    # 1.1 ------------- GRAB RING BUBBLES RB ----------------------
    # -----------------recover old-new ring bubble relations--------------------------
    
    #RBOldNewDist    = np.empty((0,2), np.uint16)
    #rRBOldNewDist   = np.empty((0,2), np.uint16)
    #rDBOldNewDist   = np.empty((0,2), np.uint16)
    #FBOldNewDist    = np.empty((0,2), np.uint16)
    #MBOldNewDist    = np.empty((0,2), np.uint16)

    RBOldNewDist    = np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
    rRBOldNewDist   = np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
    rDBOldNewDist   = np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
    FBOldNewDist    = np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
    MBOldNewDist    = np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
    DBOldNewDist    = np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
            
    # 2.1 ====================== RECOVER UNRESOLVED BUBS VIA DISTANCE CLUSTERING ======================= 

    print(f'{globalCounter}:--------- Cluster remaining via distance ---------') 

    # single frame analysis, do this when everything else is resolved.   
    if globalCounter >= 0:      
        # Prepare clusters which hold potential DB bubbles. we are not interested in resolved objects as well as objects that are not DB on prev frame.
        # Drop resolved local ring IDs.  *DELTED* Drop unresolved new rings, they are dealt during DB -> RB part. *WHY* cluster is bridged DB1-(RB)-DB2 through DB(old bub)DB. *RETRACTED*
        #dropAllRings = sum(list(l_RBub_old_new_IDs.values()),[]) + unresolvedNewRB 
        # Drop contour IDs that have been restored using matchTemplate(). 2nd time restored rRB is changed into rDB. drop them 
        #dropRestoredIDs  = sum(list(l_RBub_r_old_new_IDs.values()),[]) + sum(list(l_DBub_r_old_new_IDs.values()),[])
        # remaining contrours: filtered by size or position + everything related to RBs.
        cntrRemainingIDs = [cID for cID in whereParentOriginal if cID not in contoursFilter_RectParams_dropIDs]         
        
        distContours = {ID:contoursFilter_RectParams[ID] for ID in cntrRemainingIDs}                                    # take all current active new contours

        combosSelf = np.array(overlappingRotatedRectangles(distContours,distContours))                                  # check how they overlap with each other.
        
        if len(combosSelf) == 0: # happens if bubble overlays only itself
            combosSelf = np.empty((0,2),np.int16)

        # sometimes nearby cluster elements [n1,n2] do not get connected via rect intersection. but it was clearly part of bigger bubble of prev frame [old].
        # grab that old bubble and find its intersection with new frame elements-> [[old,n1],[old,n2]]-> [n1,n2] are connected now via [old]
        if globalCounter > 0:                                                                                           # they will be used in old-new intersection
            fakeBoxW, fakeBoxH      = 176,int(err.shape[0]/3)                                                           # generate  a fake bubble at inlet and add it to  previous frame data.
            fakeBox                 = {-1:[err.shape[1] - fakeBoxW + 1, fakeBoxH, fakeBoxW, fakeBoxH]}                  # this will make sure to gather new bubbles at inlet into single cluster.
            inletIDs, inletIDsType  = overlappingRotatedRectangles(l_rect_parms_old, fakeBox, returnType = 1)           # inletIDs: which old IDs are close to inlet? 
            #inletIDs                = [a[0] for a in inletIDs if a[0] not in lastStepFrozenGlobIDs]                     # inletIDsType: 1 & 2 - partially & fully inside.
            inletIDs                = [a[0] for a in inletIDs if a[0]]                     # inletIDsType: 1 & 2 - partially & fully inside.
            inletIDsType            = {ID:inletIDsType[ID] for ID in inletIDs}                                                                                                            # note: if edge is shared, then its not fully inside. added 1 pix offset for edge of image
                                                           

            blank = err.copy()
            x,y,w,h = fakeBox[-1]
            cv2.rectangle(blank, (x,y), (x+w,y+h),220,1)
            [cv2.drawContours(  blank,   l_contours, cid, 255, 1) for cid in cntrRemainingIDs]
            fontScale = 0.7;thickness = 4;
            [[cv2.putText(blank, str(cid), l_contours[cid][0][0]-[25,0], font, fontScale, clr, s, cv2.LINE_AA) for s, clr in zip([thickness,1],[255,0])] for cid in cntrRemainingIDs]
            cv2.putText(blank, str(globalCounter), (25,25), font, 0.9, (255,220,195),2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(imageFolder,"99999.png") ,blank)
            #
            #blank = err.copy()
            #cv2.rectangle(blank, (x,y), (x+w,y+h),220,1)
            #for ID in inletIDs:
            #    x,y,w,h = l_rect_parms_old[ID]
            #    cv2.rectangle(blank, (x,y), (x+w,y+h),120,1)
            #    [cv2.putText(blank, str(ID), (x,y), font, fontScale, clr, s, cv2.LINE_AA) for s, clr in zip([thickness,1],[255,0])]
            #cv2.imshow(f'_{globalCounter}',blank)
        dOldNewAll = np.array(overlappingRotatedRectangles({**l_rect_parms_old,**fakeBox},distContours))                # check overlap of old bubbles with new contours.
        
        # ===== graphUniqueComponents() takes connection between nodes and outputs connected components ======
        # ----- argument edgesAux acts as a catalyst by providing additional connections, but it does not appear in output -----
        # ----- in this case edgesAux are old-new bubble overlap connections ------
        cc_unique = graphUniqueComponents(cntrRemainingIDs, combosSelf, dOldNewAll, 0, err.shape,  l_centroids_all, l_centroids_old, l_contours, l_contours_hull_old)

        print('Connected Components unique:', cc_unique) if debugOnly(21) else 0

        # forzen bubs (FB) can be falsely joint for being too close. or it can be false positive FB
            
        #------------------------------------------------------------------------------- deal with  frozen bubbles!! ----------------------------------------------
        # Drop frozen IDs from clusters, add as standalone objects.
        #frozenSeparated = False
        #allFrozenLocalIDs = sum(frozenIDs,[])
        #temp = []
        #for subTemp in cc_unique:
        #    bfr = []
        #    for elem in subTemp:
        #        if elem not in allFrozenLocalIDs:bfr.append(elem)
        #        else: frozenSeparated = True
        #    temp.append(bfr)
        #cc_unique = temp
        #frozenIDs = np.array([a if type(a) != list else min(a) for a in frozenIDs])
        # why [42]-> 42? idk 10/02/23
        #if len(frozenIDs)>0: frozenIDsInfo[:,1] = np.array([a if type(a) != list else min(a) for a in frozenIDsInfo[:,1]])
        #frozenIDsInfo = np.array([a if type(a[1]) != list else [a[0]]+[min(a[1])]+[a[2:]] for a in frozenIDsInfo])

        #jointNeighbors = {min(elem): elem for elem in cc_unique if len(elem)>0}
        jointNeighbors = {ID2S(elem): elem for elem in cc_unique if len(elem)>0}
        unresolvedNewRB = [ID2S(subElem) for subElem in cc_unique if len(subElem)==1 and subElem[0] in unresolvedNewRB.copy()]      # had to remove clustered new RBs
        #unresolvedNewRB = [ID2S([a]) for a in unresolvedNewRB]
        #clusterMsg = 'Cluster groups: ' if frozenSeparated == False else f'Cluster groups (frozenIDs:{frozenIDs} detected): '
        #print(clusterMsg,"\n", list(jointNeighbors.values()))
        # collect multiple contours into one object, store it for further check
        # on first iteration there are no more checks to be done. store it early

        #  ======================= importing manual mask and reclustering =========================
        if (((globalCounter in assistFrames and assistManually == 1) or (globalCounter == 0 and markFirstMaskManually == 1 )) and not
            os.path.exists(os.path.join(manualMasksFolder, "frame"+str(globalCounter).zfill(4)+" - Copy.png"))):

            cv2.imwrite(os.path.join(manualMasksFolder, "frame"+str(globalCounter).zfill(4)+".png") ,err)
            input( "Manual mask is missing. Copy it in format old_name - Copy.png and edit it")
        
        fixedFrame = 0

        if (os.path.exists(os.path.join(manualMasksFolder, "frame"+str(index).zfill(4)+" - Copy.png")) and
            ((globalCounter in assistFrames and assistManually == 1) or (globalCounter == 0 and markFirstMaskManually == 1 ))):
        
            print('fixing assist frame')
            fixedFrame = 1
            contoursMain, group1Params = extractManualMask2(index, debug = 0)
            
            # First approximation. group objects by their bounding box intersection.
            #its fast way to discard combinations that are far away
            combos      = np.array(overlappingRotatedRectangles(group1Params,{ID:vals for ID,vals in contoursFilter_RectParams.items() if ID not in contoursFilter_RectParams_dropIDs}))
            combos      = [(str(a),b) for a,b in combos]
            cc_unique   = graphUniqueComponents([str(ID) for ID in group1Params.keys()], combos)
            olds        = [[subID for subID in IDs if type(subID) == str] for IDs in cc_unique]
            news        = [[subID for subID in IDs if type(subID) != str] for IDs in cc_unique]
            temp        = {}
            OGimg       = err.copy()*0
            newCimg     = err.copy()*0
            for oldIDs,newIDs in zip(olds,news):
                if len(oldIDs) == 1:
                    temp[oldIDs[0]] = newIDs
                else:
                    for oID in oldIDs:
                        temp[oID] = []
                        cv2.drawContours( OGimg, [contoursMain[int(oID)]], -1, 255, -1)
                        
                        for nID in newIDs:
                            cv2.drawContours( newCimg, l_contours, nID, 255, -1)
                            ovrlp = cv2.bitwise_and(OGimg,newCimg)
                            ovrlpA = np.sum(ovrlp)
                            area = cv2.contourArea(l_contours[nID])
                            if ovrlpA > 0.8*area: temp[oID].append(nID)
                            newCimg = newCimg*0
                        OGimg = OGimg*0
            #jointNeighbors = {min(a):a for a in temp.values() if len(a)>0}
            jointNeighbors = {ID2S(a):a for a in temp.values() if len(a)>0}
        
            if (globalCounter == 0 and markFirstMaskManually == 1 ):

                for ID,subIDs in jointNeighbors.items():
                    newRBinTemp = [A for A in unresolvedNewRB if int(A) in subIDs]                                 

                    if len(newRBinTemp) > 0:                                                                       
                        unresolvedNewRB   = [ ID for  ID in unresolvedNewRB if ID not in newRBinTemp]           

                    overlapWORB = [a for a in subIDs if str(a) not in newRBinTemp]                                    
                    # sidenote: i think l_RBub_masks is done second time in the end, so here its done 2 times. no time, not important
                    app = np.array((-1, ID2S(subIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                        
                    if len(overlapWORB) == 0:                                                                       
                        dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                        RBOldNewDist    = np.append(RBOldNewDist,app)   
                    else:
                        dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                        DBOldNewDist    = np.append(DBOldNewDist,app)
                    tempStore2(subIDs, l_contours, ID, err, orig, dataSets, concave = 0)

        if globalCounter == 0 and not os.path.exists(os.path.join(manualMasksFolder, "frame"+str(dataStart).zfill(4)+" - Copy.png")):
            for ID, subIDs in jointNeighbors.items():
                    dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    #tempStore2(subIDs, l_contours, ID, err, orig, dataSets, concave = 0)    
                    tempStore2(subIDs, l_contours, ID2S(subIDs), err, orig, dataSets, concave = 0) 

        # =============================== S E C T I O N - -  0 0 ==================================       
        # ------------------ RECOVER FROZEN BUBBLES BASED ACTIVE FROZEN IDS ---------------------- 
        if globalCounter > 0: 
            print(f'{globalCounter}:--------- Frozen bubble recovery ---------')  
            # --------------- drop data in local storage --------------
            activeFrozenLocalIDs = []
            for oldID, newIDs, dist, areaOld, aStdOld, areaNew in frCulpr:
                newID = min(newIDs)
                areaCrit  = np.abs(areaNew-areaOld)/ aStdOld
                if areaCrit < 3:
                    activeFrozenLocalIDs.append(newIDs)
                    dataSets = [l_FBub_masks,l_FBub_images,l_FBub_old_new_IDs,l_FBub_rect_parms,l_FBub_centroids,l_FBub_areas_hull,l_FBub_contours_hull]
                    tempStore2(newIDs, l_contours, ID2S(newIDs), err, orig, dataSets, concave = 0)
                    if int(old) in unresolvedOldDB: unresolvedOldDB.remove(int(old))
                    app = np.array((oldID, ID2S(newIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                    #FBOldNewDist    = np.append(FBOldNewDist,[[oldID,newID]],axis=0)
                    FBOldNewDist    = np.append(FBOldNewDist,app)
                    if oldID in unresolvedOldDB: unresolvedOldDB.remove(oldID)
                    l_predict_displacement[oldID]                       = [tuple(map(int,dataSets[4][ID2S(newIDs)])), dist]       
                    if oldID in l_FBub_centroids_old:   print(f'-{oldID}:Restored oldFB-newrFB (restored from past): {oldID} & {newID}:{newIDs}.')
                    else:                               print(f'-{oldID}:Restored oldFB-newrFB (from previous frame): {oldID} & {newID}:{newIDs}.')

            # ---- recluster jointNeighbors ----
        
            for subIDsF in activeFrozenLocalIDs:
                jointNeighbors = {ID:[elem for elem in sub if elem not in subIDsF] for ID,sub in jointNeighbors.copy().items()}
                [unresolvedNewRB.remove(str(ID)) for ID in subIDsF if str(ID) in unresolvedNewRB]

            jointNeighbors = {ID2S(subIDs):subIDs for subIDs in jointNeighbors.values() if len(subIDs)>0}
            #if len(activeFrozenLocalIDs) > 0:
            #    print("(updated) unresolvedOlDRB (all old D,rD bubbles):\n",  unresolvedOldDB)
            #    print("(updated) unresolvedNewRB (new typeRing bubbles):\n",  unresolvedNewRB)
            #    print(f'{globalCounter}:--------- Frozen bubble (FB): {[a[0] for a in FBOldNewDist]} recovery ended ---------\n')
            #else: print(f'{globalCounter}:--------- No Frozen bubble (FB) recovered ---------\n')
            
            
            if len(overlapFRLoc) > 0:
                newRelevantLocals   = sum(list(jointNeighbors.values()),[])
                newRelevantRPs      = {ID:l_rect_parms_all[ID] for ID in newRelevantLocals if l_rect_parms_all[ID][2]*l_rect_parms_all[ID][3] < locFRAreaThreshold}
                relevantNextIDs     = [a[1] for a in overlapFRLoc]
                overlapFRLoc2       = overlappingRotatedRectangles({str(ID):step2RP[ID] for ID in relevantNextIDs}, newRelevantRPs) 
                if len(overlapFRLoc2)>0:
                    releveantNextIDs2 = [a[0] for a in overlapFRLoc2]
                    releveantNewIDs = [a[1] for a in overlapFRLoc2]
                    a = 1
                    nextCAs     = {str(ID):vals for ID,vals in nextCAs.copy().items() if str(ID) in releveantNextIDs2}
                    newCAs      = {ID:[l_centroids_all[ID],l_areas_all[ID]] for ID in releveantNewIDs}
                    
                    calcD = lambda ID1,ID2: np.linalg.norm(np.array(nextCAs[ID1][0]) - np.array(newCAs[ID2][0]))
                    calcA = lambda ID1,ID2: np.abs(nextCAs[ID1][1]-newCAs[ID2][1])/newCAs[ID2][1]

                    overlapDA   = [[calcD(ID1,ID2),calcA(ID1,ID2)] for ID1,ID2 in overlapFRLoc2]            # find c-c dist and rel area of overlap combos
                    overlapPass = [True if a<=3 and b< 0.4 else False for a,b in overlapDA]
                    overlapFRLoc = [combo for i,combo in enumerate(overlapFRLoc2) if overlapPass[i] == True]
                    declusterNextFRLocals2  = [int(a[0]) for a in overlapFRLoc]
                    declusterNewFRLocals    = [a[1] for a in overlapFRLoc]
                    if len(declusterNewFRLocals)>0:
                        jointNeighbors  = {ID: [elem for elem in sub if elem not in declusterNewFRLocals]
                                          for ID,sub in jointNeighbors.copy().items()}
                        [unresolvedNewRB.remove(str(ID)) for ID in declusterNewFRLocals if str(ID) in unresolvedNewRB]
                            

                        jointNeighbors = {ID2S(subIDs):subIDs for subIDs in jointNeighbors.values() if len(subIDs)>0}
                        delBuffer = []
                        for oldID,subIDs in l_old_new_IDs_old.items():
                            if len(subIDs) == 1 and subIDs[0] in declusterNextFRLocals2:
                                where = declusterNextFRLocals2.index(subIDs[0])
                                newID = declusterNewFRLocals[where]
                                newIDs = [newID]
                                dist = calcD(str(subIDs[0]),newID)
                                
                                activeFrozenLocalIDs.append(subIDs)
                                dataSets = [l_FBub_masks,l_FBub_images,l_FBub_old_new_IDs,l_FBub_rect_parms,l_FBub_centroids,l_FBub_areas_hull,l_FBub_contours_hull]
                                tempStore2(newIDs, l_contours, ID2S(newIDs), err, orig, dataSets, concave = 0)
                                #if int(oldID) in unresolvedOldDB: unresolvedOldDB.remove(int(oldID))
                                app = np.array((oldID, ID2S(newIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                                FBOldNewDist    = np.append(FBOldNewDist,app)
                                if oldID in unresolvedOldDB: unresolvedOldDB.remove(oldID)
                                l_predict_displacement[oldID]                       = [tuple(map(int,dataSets[4][ID2S(newIDs)])), dist]       
                                if oldID in l_FBub_centroids_old:   print(f'-{oldID}:Restored oldFB-newrFB (restored from past): {oldID} & {newID}:{newIDs}.')
                                else:                               print(f'-{oldID}:Restored oldFB-newrFB (from frozen cluster locals): {oldID} & {newID}:{newIDs}.')
                                delBuffer.append(newID)
                        [declusterNewFRLocals.remove(ID)  for ID in    delBuffer]     #
                        # added to unresolved after
                a = 1
            if len(activeFrozenLocalIDs) > 0:
                print("(updated) unresolvedOlDRB (all old D,rD bubbles):\n",  unresolvedOldDB)
                print("(updated) unresolvedNewRB (new typeRing bubbles):\n",  unresolvedNewRB)
                print(f'{globalCounter}:--------- Frozen bubble (FB): {[a[0] for a in FBOldNewDist]} recovery ended ---------\n')
            elif len(declusterNewFRLocals) > 0:
                print(f'{globalCounter}:--------- Local Frozens: {declusterNewFRLocals} recovery ended ---------\n')
            else: print(f'{globalCounter}:--------- No Frozen bubble (FB) recovered ---------\n')

        # =============================== S E C T I O N - -  0 1 ==================================       
        # ------------------ RECOVER UNRESOLVED BUBS VIA DISTANCE CLUSTERING ----------------------   
        # ------------------ join with unresolvedNewRB and search relations ----------------------
        # jointNeighbors is a rough estimate. In an easy case neighbor bubs will be far enough away
        # so clusters wont overlap. rather strict area/centroid displ restrictions can be satisfied
        # if it fails, take overlapped cluster IDs and start doing permutations and check dist/areas
        if globalCounter >= 1: 
            elseOldNewDoubleCriterium           = []
            elseOldNewDoubleCriteriumSubIDs     = {}
            oldNewDB                            = []
            
            #jointNeighborsWoFrozen              = {mainNewID: subNewIDs for mainNewID, subNewIDs in jointNeighbors.items() if mainNewID not in frozenIDs}   # drop frozen bubbles FB from clusters          e.g {15: [15, 18, 20, 22, 25, 30], 16: [16, 17], 27: [27]} (no frozen in this example)
            jointNeighborsWoFrozen              = {mainNewID: subNewIDs for mainNewID, subNewIDs in jointNeighbors.items()}   # drop frozen bubbles FB from clusters          e.g {15: [15, 18, 20, 22, 25, 30], 16: [16, 17], 27: [27]} (no frozen in this example)
            jointNeighborsWoFrozen_hulls        = {ID: cv2.convexHull(np.vstack(l_contours[subNewIDs])) for ID, subNewIDs in jointNeighborsWoFrozen.items()}# pre-calculate hulls (dims [N, 1, 2]),         e.g {15: array([[[1138,  488]...ype=int32), 16: array([[[395, 550]],...ype=int32), ...}
            jointNeighborsWoFrozen_bound_rect   = {ID: cv2.boundingRect(hull) for ID, hull in jointNeighborsWoFrozen_hulls.items()}                         # bounding rectangles,                          e.g {15: (857, 422, 282, 136), 16: (305, 495, 91, 98), 27: (648, 444, 164, 205)}
            jointNeighborsWoFrozen_c_a          = {ID: getCentroidPosContours(bodyCntrs = [hull]) for ID, hull in jointNeighborsWoFrozen_hulls.items()}     # centrouid and areas for a perfect match test  e.g {15: ((1009, 496), 28955), 16: ((...), 6829), 27: ((...), 21431)}
            
            #resolvedFrozenGlobalIDs         = [int(elem) for elem in frozenOldGlobNewLoc.values() if type(elem) == str]                                     # frozen global IDs are not recovered           e.g []
            #oldDistanceCentroidsWoFrozen    = {key:val for key,val in l_DBub_centroids_old.items()                                                          # drop frozens. not sure what is the difference
            #                                   if key not in list(l_FBub_old_new_IDs_old.keys()) + resolvedFrozenGlobalIDs}                                 # between two lists                             e.g  {1: (366, 539), 2: (927, 505), 3: (741, 539), 4: (1080, 486), 7: (1204, 446)}
            oldDistanceCentroidsWoFrozen    = {key:val for key,val in l_DBub_centroids_old.items()   if key in unresolvedOldDB                                                       # drop frozens. not sure what is the difference
                                               }                                 # between two lists                             e.g  {1: (366, 539), 2: (927, 505), 3: (741, 539), 4: (1080, 486), 7: (1204, 446)}

            oldDBubAndUnresolvedOldRB       = {**oldDistanceCentroidsWoFrozen,**{ID:l_RBub_centroids_old[ID] for ID in unresolvedOldRB}}                      # merge old DB with all old RBs. since 
            oldDBubAndUnresolvedOldRB       = {**oldDBubAndUnresolvedOldRB  ,**{ID:val for ID,val in l_FBub_centroids_old.items() if ID in unresolvedOldDB}}    # if FB is unresolved, try to find it. this way it can continue to live as regular bubble                                                                                                                                           # old-newRB method was merged into this

            print(f'recovering DB bubbles :{list(oldDistanceCentroidsWoFrozen.keys())}') if len(unresolvedOldRB) == 0 else print(f'recovering DB+RB bubbles :{list(oldDBubAndUnresolvedOldRB.keys())}')
            if len([a for a,b in inletIDsType.items() if b == 2])>0:  print(f'except old inlet bubbles: {[a for a,b in inletIDsType.items() if b == 2]}')
            #srtF = lambda x : l_bubble_type_old[x[0]]                                                                                                        # sort by type number-> RBs to front.
            #oldDBubAndUnresolvedOldRB = dict(sorted(oldDBubAndUnresolvedOldRB.items(), key=srtF))                                                            # given low typeIDs are virtually important. 
            #oldDBubAndUnresolvedOldRB = {**{ID:oldDBubAndUnresolvedOldRB[ID] for ID in oldDBubAndUnresolvedOldRB if ID in activeFrozen},                      # think i should do prio for pseudo frozens
            #                             **{ID:oldDBubAndUnresolvedOldRB[ID] for ID in oldDBubAndUnresolvedOldRB if ID not in activeFrozen}}                  # so i can solve them, and drop from cluster.
            partialInletIDs = [a for a,b in inletIDsType.items() if b == 1]
            fullInletIDs    = [a for a,b in inletIDsType.items() if b == 2]
            partialInletIDs = list(sorted(partialInletIDs, key=lambda x: l_centroids_old[x][0]))
            notFrozenIDs = [ID for ID in oldDBubAndUnresolvedOldRB if ID not in activeFrozen]
            oldDBubAndUnresolvedOldRB = {**{ID:oldDBubAndUnresolvedOldRB[ID] for ID in oldDBubAndUnresolvedOldRB if ID in activeFrozen},  # think i should do prio for pseudo frozens
                                         **{ID:oldDBubAndUnresolvedOldRB[ID] for ID in notFrozenIDs if ID in partialInletIDs},
                                         **{ID:oldDBubAndUnresolvedOldRB[ID] for ID in notFrozenIDs if ID not in partialInletIDs}}        # so i can solve them, and drop from cluster.
            STs = sum([vals for time,vals in STBubs.items() if time > globalCounter - 4],[])                            # for see-though bubbles concave hull. its mostly magic
            STIDs, STcounts = np.unique(STs, return_counts = True)                                                      # ST bubbles shed small bubbles when distorted. tracking concave hull gives more bias towards main mass, rather than sheeded bs
            #oldDBubAndUnresolvedOldRB = {ID:a for ID,a in oldDBubAndUnresolvedOldRB.items()}                            # moved if ID not in inletIDsType or (ID in inletIDsType and inletIDsType[ID] < 2) inside loop
            for oldID, oldCentroid in oldDBubAndUnresolvedOldRB.items():                                                # recovering RB, rRB, DB, rDB

                oldType         = l_bubble_type_old[oldID]                                                              # grab old ID bubType                                                   e.g 3  (typeElse)
                if oldType == typeRing or oldType == typeRecoveredRing:                                                 # RB type inheritance should be fragile.
                    newType     = typeRecoveredElse if oldType == typeRecoveredRing else typeRecoveredRing              # recovered RB to rRB, recovered rRB to rDB. for safety...
                else: newType   = typeElse                                                                              # if not (r)RB, then DB. rDB lost its meaning here. maybe on final matchTemplates()
                # ===== GET STATISTICAl DATA ON OLD BUBBLE: PREDICT CENTROID, AREA =====
                if oldID in inletIDsType:
                    (mis,mio,ss,fs) = (10,1,100,0)                                                            # mis = maxInterpSteps; mio = maxInterpOrder
                    maxAreaCrit1 = 3
                    maxAreaCrit = 3 
                else:
                    (mis,mio,ss,fs) = (3,2,0.3,1)                                                                                  # i think i should give inlet IDs more history, but lower order
                    maxAreaCrit1 = 3
                    maxAreaCrit = 5 
                section1STTrigger = False # recluster if ST sheded a bubble   
                section1INTrigger = False # recluster if Inlet stuff... idk
                section1BRTrigger = False
                trajectory                                          = list(g_Centroids[oldID].values())                 # all previous path for oldID                                           e.g [(411, 547), (402, 545), (390, 543), (378, 542), (366, 539)]
                predictCentroid_old, _, distCheck2, distCheck2Sigma = g_predict_displacement[oldID][globalCounter-1]    # g_predict_displacement is an evaluation on how good predictor works.  e.g [(366, 540), 1, 10, 6]
                                                                                                                        # [old predicted value (used only for debug), old predictor error, mean error, stdev] 
                                                                                                                        # you can see than last known position was (366, 539), and it was 
                #if oldID == 40 :
                #    debugVecPredict = 1
                #    ss = 100
                #    fs = 0
                #else:
                #    debugVecPredict = 0
                #    ss = 0.3
                #    fs = 1
                tempArgs                                            = [                                                 # predicted to be at (366, 540), which is an error of 1. 
                                                                        [distCheck2,distCheck2Sigma],                   # dist criterium
                                                                        g_predict_displacement[oldID],5,mis,mio,            # dist prediction history, # of stdevs, max steps for interp, max interp order
                                                                        debugVecPredict,predictVectorPathFolder,        # debug or not, debug save path
                                                                        predictCentroid_old,oldID,globalCounter,[-3,0]  # old centroid, timestep, ID for debug and zeroth displacement. (not used?)
                                                                      ]

                predictCentroid                                     = distStatPredictionVect2(trajectory, *tempArgs,ss = ss, fixSharp = fs)    # predicted centroid for this frame, based on previous history.         e.g array([354, 537])
                
                oldMeanArea                                         = g_predict_area_hull[oldID][globalCounter-1][1]    # mean area                                                             e.g 6591
                oldAreaStd                                          = g_predict_area_hull[oldID][globalCounter-1][2]    # mean area                                                             e.g 620
                areaCheck                                           = oldMeanArea + 3*oldAreaStd                        # top limit                                                             e.g 8451
                l_predict_displacement[oldID]                       = [tuple(map(int,predictCentroid)), -1]             # predictor error will be stored here. acts as buffer storage for predC e.g [(354, 537), -1]
                
                if (oldID in inletIDsType and inletIDsType[oldID] == 2):                                                # i consider bubbles in fully inlet zone. so i get estimate in l_predict_displacement
                    continue                                                                                            # but then i skip this ID and process it in spaghetti code for inlet.
                # ===== REFINE TEST TARGETS BASED ON PROXIMITY =====
                #overlapIDs = np.array(overlappingRotatedRectangles(
                #                                                    {oldID:l_rect_parms_old[oldID]},                    # check rough overlap with clusters.
                #                                                    jointNeighborsWoFrozen_bound_rect),int)             # this drops IDs that are out of reach. [old globalID, cluster localID] e.g array([[ 1, 16]])
                #overlapIDs = {new: jointNeighborsWoFrozen[new] for _, new in overlapIDs}                                # reshape to form [localID, subIDs]                                     e.g {16: [16, 17]}
                overlapIDs = overlappingRotatedRectangles({oldID:l_rect_parms_old[oldID]}, jointNeighborsWoFrozen_bound_rect) # this drops IDs that are out of reach. [old globalID, cluster localID] e.g array([[ 1, 16]])
                overlapIDs = {new: jointNeighborsWoFrozen[new] for _, new in overlapIDs}                                # reshape to form [localID, subIDs]                                     e.g {16: [16, 17]}

                if oldType == typeRing:
                    distCheck2Sigma += 1
                
                tempDistance, tempSol   = 10000000, []                                                                  # tempDistance = min(tempDistance, dist2, permDist2)
                dist2, permDist2        = 10000000, 10000000                                                            # dont want to re-use prev iteration values.
                if oldID in STIDs:
                    STNtimes = STcounts[STIDs.tolist().index(oldID)]                                                    # for see-though bubbles concave hull. its mostly magic
                else: STNtimes = 0
                # ===== TEST A CLUSTER ASSUMING ALL SUBELEMENTS ARE CORRECT (PERFECT MATCH TEST) =====
                clusterElementFailed        = {ID:False for ID in overlapIDs}                                           # track if whole cluster is a solution                                  e.g {16: False}
                clusterCritValues           = {}                                                                        # if cluster generally fails, store criterium values here, for info     e.g {}
                for mainNewID, subNewIDs in overlapIDs.items():                                                         # test only overlapping clusters

                    newCentroid, newArea    = jointNeighborsWoFrozen_c_a[mainNewID]                                     # retrieve pre-computed cluster centroid and hull area                  e.g ((353, 539), 6829)
                    dist2                   = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))         # find distance between (cluster and predicted) centroids               e.g 2.2360 (predicted array([354, 537]))
                    areaCrit                = np.abs(newArea-oldMeanArea)/ oldAreaStd                                   # calculate area difference in terms of stdev                           e.g 0.3838 (old mean 6591, old stdev 620)
                    
                    if dist2 <= distCheck2 + 5*distCheck2Sigma and areaCrit < maxAreaCrit1:                                        # test if criteria within acceptable limits                             e.g dist vs 10+5*6  ( which is kind of alot)      
                        
                        tempID                         = oldID                                                          # l_RBub_r and l_DBub_r have global IDs.
                        hull                           = jointNeighborsWoFrozen_hulls[mainNewID]                        # retrieve pre-computed hull
                        l_predict_displacement[oldID]  = [tuple(map(int,predictCentroid)), int(dist2)]                  # store prediction error
                        
                        if (oldID in partialInletIDs and len(subNewIDs) > 1 and not fixedFrame and len(fullInletIDs)>0):# if ID is partial, it is made of multiple contours
                            _,yFb,_,hFB     = fakeBox[-1]                                                               # and there is fully inside inlet ID, which
                            rectParamsNew   = {ID:l_rect_parms_all[ID]  for ID in subNewIDs}                            # might be part of this cluster
                            rectParamsNewM  = {ID:[x,yFb,w,hFB]         for ID,[x,y,w,h] in rectParamsNew.items()}      # we notice that inlet subIDs have vertical 
                            restMainOverlap = overlappingRotatedRectangles(rectParamsNewM, rectParamsNewM )             # connectivity, which might help split
                            cc_unique       = graphUniqueComponents(list(rectParamsNew.keys()), restMainOverlap) 
                            if len(cc_unique) > 1:                                                                      # if there are multiple clusters
                                section1INTrigger = True
                                subNewIDsIN     = subNewIDs
                                rectParamsNew   = {min(subCluster):np.array([rectParamsNewM[ID] for ID in subCluster]).T for subCluster in cc_unique}
                                rectUnion       = lambda a,b,c,d: list(map(int,[min(a),min(b),max(a+c)-min(a),max(d)])) # 
                                rectParamsNewOM = {ID:rectUnion(a,b,c,d) for ID,[a,b,c,d] in rectParamsNew.items()}     # calc subcluster bounding Rect
                                a = 1                                                                                   # 
                                leftmostID      = min(rectParamsNewOM, key=lambda k: rectParamsNewOM[k][0])             # and take top most as a solution
                                subNewIDs       = sum([a for a in cc_unique if leftmostID in a],[])                     # 
                                hull            = cv2.convexHull(np.vstack(l_contours[subNewIDs]))                      # 
                                newCentroid, newArea = getCentroidPosContours(bodyCntrs = [hull])                       # 
                                dist2           = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))     # 
                                l_predict_displacement[oldID]  = [tuple(map(int,predictCentroid)), int(dist2)]          # 


                        if STNtimes >= 2 and len(subNewIDs) == 1: hull = 1                                                  # for see-though bubbles concave hull. its mostly magic
                        # FIX for ST bubbles shedding small bubbles!!!!
                        if STNtimes >= 1 and len(subNewIDs) > 1:
                            areas = {ID:l_areas_all[ID] for ID in subNewIDs}
                            maxAID = max(areas, key=areas.get)
                            #areasOther = {ID:areas[ID] for ID in subNewIDs if ID != maxAID}
                            areasSmall = {ID:True if areas[ID] < 5000 else False for ID in subNewIDs if ID != maxAID}
                            if all(areasSmall):
                                section1STTrigger = True
                                subNewIDsST = subNewIDs.copy()
                                subNewIDs = [maxAID]
                                hull = 1
                                
                        if len(subNewIDs)>1:                                            # if there are multiple sub-contours 
                            hullTest = cv2.convexHull(np.vstack(l_contours[subNewIDs])) 
                            x0,y0,w0,h0 = cv2.boundingRect(hullTest) 
                            img = np.zeros((h0,w0),np.uint8)                            # check if they are widely spaced 
                            [cv2.drawContours( img, [l_contours[subID]], -1, 255, -1, offset = (-x0,-y0)) for subID in subNewIDs]
                                                                                        # by comparing how much void there is  
                            graysH = cv2.contourArea( hullTest)                      # inside hull area. graysH-> hull area
                            graysM = int(np.sum(img)/255)                               # graysM - cluster area 
                            if graysM < 0.7*graysH:                                     # if there is a lot of void 
                                #subIDs = dataSets[2][ID2S(tempSol)]
                                # modBR: if rect width > side, then nothing changes, else x is offset by side & width diff. width is modifed to max. same for y.
                                def modBR(BR,side):
                                    x,y,w,h  = BR
                                    return [x - int(max((side-w)/2,0)), y - int(max((side-h)/2,0)), max(side,w), max(side,h)]
                                                                                                         # lets check clustering
                                ogRects = {ID:l_rect_parms_all[ID] for ID in subNewIDs}                     # subID bounding rectangles
                                ogRectsM  = {ID:modBR(rec,150) for ID,rec in ogRects.items()}            # expand them to at least 150x150 square
                                                                                                         # 
                                combosSelf = np.array(overlappingRotatedRectangles(ogRectsM,ogRectsM))   # 
                                cc_unique           = graphUniqueComponents(subNewIDs, combosSelf)          # recluster
                                if len(cc_unique)>1:                                                     # if there are multiple clusters
                                    #imgHull = img.copy()*0                                  #  
                                    #cv2.drawContours( imgHull, [hullTest], 0, 255, -1, offset = (-x0,-y0))
                        
                                    #[cv2.rectangle(img, (x-x0,y-y0), (x+w-x0,y+h-y0),200,1) for x,y,w,h in ogRects.values()]
                                    #[cv2.rectangle(img, (x-x0,y-y0), (x+w-x0,y+h-y0),100,1) for x,y,w,h in ogRectsM.values()]
                                    #cv2.imshow(f'gc:{globalCounter}',cv2.hconcat([img,imgHull]))
                                    section1BRTrigger = True
                                    topMostID      = min(ogRects, key=lambda k: ogRects[k][0])  # calc subcluster bounding Rect
                                    subNewIDsBR = subNewIDs                                     # snapshot of old full cluster
                                    subNewIDs = [comb for comb in cc_unique if topMostID in comb][0]
                                    
                                    hull            = cv2.convexHull(np.vstack(l_contours[subNewIDs]))                      # 
                                    newCentroid, newArea = getCentroidPosContours(bodyCntrs = [hull])                       # 
                                    dist2           = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))     # 
                                    l_predict_displacement[oldID]  = [tuple(map(int,predictCentroid)), int(dist2)]

                        newRBinTemp = [A for A in unresolvedNewRB if int(A) in subNewIDs]                                  # all RB types in cluster. im not sure if ever works, since RBs are alone

                        if len(newRBinTemp) > 0:                                                                        # if there are RBs in solution
                            unresolvedNewRB   = [ ID for  ID in unresolvedNewRB if ID not in newRBinTemp]           # drop these RBs from new found list
                        
                        
                        app = np.array((oldID, ID2S(subNewIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                        overlapWORB = [a for a in subNewIDs if str(a) not in newRBinTemp]                                    # drop RBs from solution. subNewIDs always has entries.
                                                                                                                        # empty overlapWORB means subNewIDs consited only of RB
                        if len(overlapWORB) == 0:                                                                       # if solution consists only of RB
                            dataSets        = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                            #newType         = typeRing      #(?)                                                        # change default to RB, keep data in regular RB storage.
                                                                                                                        # any relevant type casts to RB if solution is RB only.
                            #RBOldNewDist    = np.append(RBOldNewDist,[[oldID,newRBinTemp[0]]],axis=0)                   # store RB solution in oldX-newRB relations
                            #tempID          = min(subNewIDs)                                                            # data is stored via local IDs
                            RBOldNewDist    = np.append(RBOldNewDist,app) 
                                                                                
                        elif newType        == typeRecoveredRing:                                                                                                 
                            dataSets        = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                            #rRBOldNewDist    = np.append(rRBOldNewDist,[[oldID,min(subNewIDs)]],axis=0)
                            #tempID          = min(subNewIDs) 
                            rRBOldNewDist    = np.append(rRBOldNewDist,app)

                        elif newType        == typeRecoveredElse:                                                       # rDB is being cast only from rRB                         
                            dataSets        = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                            #rDBOldNewDist    = np.append(rDBOldNewDist,[[oldID,min(subNewIDs)]],axis=0)
                            #tempID          = min(subNewIDs) 
                            rDBOldNewDist    = np.append(rDBOldNewDist, app)
                            print(f'-{oldID}:Swapping bubble type to {typeStrFromTypeID[newType]} because it was recovered second time.')
                        
                        else:
                            dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                            #tempID          = min(subNewIDs)                                                            # data is stored via local IDs
                            #elseOldNewDoubleCriteriumSubIDs[tempID]  = subNewIDs                                        # add to oldDB new DB relations
                            #elseOldNewDoubleCriterium.append([oldID,tempID,np.around(dist2,2),np.around(areaCrit,2)])   # some stats.
                            DBOldNewDist    = np.append(DBOldNewDist, app)
                        
                        
                        #if fixedFrame == 0:
                        #    resolved    = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs,**l_DBub_old_new_IDs,**l_DBub_r_old_new_IDs,**l_MBub_old_new_IDs}
                        #    assert tempID not in resolved, f"image: #{globalCounter+dataStart}=>MERGE ERROR: shared contour is also local IDs, mask by hand :("
                
                        #tempStore2(subNewIDs, l_contours, tempID, err, orig, dataSets, concave = hull)
                        tempStore2(subNewIDs, l_contours, ID2S(subNewIDs), err, orig, dataSets, concave = hull)
                        unresolvedOldRB.remove(oldID) if oldID in unresolvedOldRB   else 0                                  # remove from unresolved IDs
                        unresolvedOldDB.remove(oldID)  if oldID in unresolvedOldDB    else 0                                  # remove from unresolved IDs
                        print(f'-{oldID}:Resolved (first match) old{typeStrFromTypeID[oldType]}-new{typeStrFromTypeID[newType]}: {oldID} & {mainNewID}:{subNewIDs}.')
                        
                        tempDistance = min(tempDistance,dist2)                # do same in radial dist meth, if fails here
                        tempSol = subNewIDs
                    else:
                        clusterElementFailed[mainNewID]     = True                                                      # mark cluster as failed
                        clusterCritValues[mainNewID]        = [dist2,areaCrit]                                          # store crit values for reference
                        print(f'-{oldID}:Failed to resolve (first match) old{typeStrFromTypeID[oldType]}-new{typeStrFromTypeID[newType]}: {oldID} & {subNewIDs}.')
                        
                    print(f'--Distance prediction error: {dist2:0.1f} vs {distCheck2:0.1f} +/- 5* {distCheck2Sigma:0.1f} and area criterium (dA/stev):{areaCrit:0.2f} vs ~3\n')
                        
                # ========== IF PERFECT MATCH FAILED TEST RADIAL DISTRIBUTION METHOD ==========
                #
                # radialStatsXXX() function takes a contour or an image, and calculates how many white pixels are at each radiuss
                # around a point (centroid). disk will have a linear distribution (2pi*r), ring- piecewise linear.
                # IDEA 01: if you split them up into sections and sum up each section's distribution, you will restore OG distribution
                # also you can deduce from segments distribution location if it is a part of OG distribution.
                # IDEA 02: from distribution you can extract lower un upper bound of an interval where most area is located
                # bounds are calculated by finding a minimal interval length that holds to some given % of a total area
                # MODIFICTATION: most bubbles are of elliptical shape, with this loss rotational symmetry, elliptical
                # curves are considered instead of rings
                oneClusterOneElem = True if len(clusterElementFailed) == 1 and len(overlapIDs[list(overlapIDs.keys())[0]]) == 1 else False # if there was only one option of one element. (ussually only 1 choice)
                                                                                                                                           # no point doing permutations, same result as perfect-match.
                if all(clusterElementFailed.values()) and fixedFrame == 0 and not oneClusterOneElem:                    # if all cluster matches have failed + if it is a manual mask frame, dont let it do permutations
                    cvr = 0.90                                                                                          # consider cvr*100% of total area for bounds calculation
                    ellipseParams   = l_ellipse_parms_old[oldID]                                                        # have to check if old bubble was of elliptical shape
                    a,b             = ellipseParams[1]                                                                  # major, minor axis (not semi-axis)
                    isEllipse       = 1 if np.sqrt(1-(min(a,b)/max(a,b))**2) > 0.5 else 0                               # criterium of ellipticity. if its a ring, calculation is simpler.
                    ddd             = 0 if globalCounter != -1 else 1
                    [OGLeft,OGRight], OGDistr = radialStatsImageEllipse(isEllipse, oldCentroid, l_masks_old[oldID], l_rect_parms_old[oldID], ellipseParams, cvr, oldID, globalCounter, debug = ddd)
                    
                    for mainNewID, subNewIDs in overlapIDs.items():

                        cvr2 = 0.8
                        debug = 0 if globalCounter != -1  else 1
                        # ===== function radialAnal() returns best case solution by analysing radial distributions of subcluster =====
                        # if bubble was in a state of merge, modified hull method is applied.
                        dddd = 0 #if not (globalCounter == 150 and oldID == 28) else 1
                        permIDsol2, permDist2, permRelArea2, newCentroid, newArea, hull, newParams = radialAnal( [[OGLeft,OGRight], OGDistr] ,
                                                       [isEllipse,l_contours,subNewIDs,predictCentroid, ellipseParams, cvr2, err],
                                                       [l_rect_parms_all,l_rect_parms_old,l_centroids_all,l_areas_all] ,
                                                       [distCheck2, distCheck2Sigma, areaCheck, oldMeanArea, oldAreaStd, clusterCritValues],
                                                       globalCounter, oldID, mainNewID, l_MBub_info_old, debug = dddd)
                        
                        if len(permIDsol2)>0 and permDist2 < distCheck2 +5* distCheck2Sigma and permRelArea2 < maxAreaCrit:       # check if given solution satisfies criterium

                            if STNtimes >= 2 and len(permIDsol2) == 1: hull = 1
                            if STNtimes >= 1 and len(permIDsol2) > 1:
                                areas = {ID:l_areas_all[ID] for ID in permIDsol2}
                                maxAID = max(areas, key=areas.get)
                                #areasOther = {ID:areas[ID] for ID in subNewIDs if ID != maxAID}
                                areasSmall = {ID:True if areas[ID] < 5000 else False for ID in permIDsol2 if ID != maxAID}
                                if all(areasSmall):
                                    permIDsol2 = [maxAID]
                                    
                                    hull       = alphashapeHullModded(l_contours, permIDsol2, 0.05, 0)
                                    centroid = getCentroidPosContours(bodyCntrs = [hull])[0]
                                    permDist2             = np.linalg.norm(np.array(predictCentroid) - np.array(centroid)).astype(int)
                                    section1STTrigger = True
                                    subNewIDsST = subNewIDs
                            if len(permIDsol2)>1:                                            # if there are multiple sub-contours 
                                hullTest = cv2.convexHull(np.vstack(l_contours[permIDsol2])) 
                                x0,y0,w0,h0 = cv2.boundingRect(hullTest) 
                                img = np.zeros((h0,w0),np.uint8)                            # check if they are widely spaced 
                                [cv2.drawContours( img, [l_contours[subID]], -1, 255, -1, offset = (-x0,-y0)) for subID in permIDsol2]
                                                                                            # by comparing how much void there is  
                                graysH = cv2.contourArea( hullTest)                      # inside hull area. graysH-> hull area
                                graysM = int(np.sum(img)/255)                               # graysM - cluster area 
                                if graysM < 0.7*graysH:                                     # if there is a lot of void 
                                    #subIDs = dataSets[2][ID2S(tempSol)]
                                    # modBR: if rect width > side, then nothing changes, else x is offset by side & width diff. width is modifed to max. same for y.
                                    def modBR(BR,side):
                                        x,y,w,h  = BR
                                        return [x - int(max((side-w)/2,0)), y - int(max((side-h)/2,0)), max(side,w), max(side,h)]
                                                                                                             # lets check clustering
                                    ogRects = {ID:l_rect_parms_all[ID] for ID in permIDsol2}                     # subID bounding rectangles
                                    ogRectsM  = {ID:modBR(rec,150) for ID,rec in ogRects.items()}            # expand them to at least 150x150 square
                                                                                                             # 
                                    combosSelf = np.array(overlappingRotatedRectangles(ogRectsM,ogRectsM))   # 
                                    cc_unique           = graphUniqueComponents(permIDsol2, combosSelf)          # recluster
                                    if len(cc_unique)>1:                                                     # if there are multiple clusters
                                        #imgHull = img.copy()*0                                  #  
                                        #cv2.drawContours( imgHull, [hullTest], 0, 255, -1, offset = (-x0,-y0))
                        
                                        #[cv2.rectangle(img, (x-x0,y-y0), (x+w-x0,y+h-y0),200,1) for x,y,w,h in ogRects.values()]
                                        #[cv2.rectangle(img, (x-x0,y-y0), (x+w-x0,y+h-y0),100,1) for x,y,w,h in ogRectsM.values()]
                                        #cv2.imshow(f'gc:{globalCounter}',cv2.hconcat([img,imgHull]))
                                        section1BRTrigger = True
                                        topMostID      = min(ogRects, key=lambda k: ogRects[k][0])  # calc subcluster bounding Rect
                                        subNewIDsBR = subNewIDs                                     # snapshot of old full cluster
                                        permIDsol2 = [comb for comb in cc_unique if topMostID in comb][0]
                                    
                                        hull            = cv2.convexHull(np.vstack(l_contours[permIDsol2]))                      # 
                                        newCentroid, newArea = getCentroidPosContours(bodyCntrs = [hull])                       # 
                                        permDist2           = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))     # 
                                        l_predict_displacement[oldID]  = [tuple(map(int,predictCentroid)), int(permDist2)]

                            app = np.array((oldID, ID2S(permIDsol2)), dtype=[('integer', '<i4'), ('string', '<U60')])

                            l_predict_displacement[oldID]   = [tuple(map(int,predictCentroid)), np.around(permDist2,2)] # store predictor error
                            tempID                          = oldID                                                     # default to global ID. not important.
                           
                            if oldID in l_MBub_info_old: l_MBub_info[oldID] = [[],newArea,newCentroid] + list(newParams)# if bub merged lately, add merge params
                            
                            newRBinTemp = [A for A in unresolvedNewRB if A in permIDsol2]                             # all RB types in cluster. im not sure if ever works, since RBs are alone

                            if len(newRBinTemp) > 0:                                                                    # if there are RBs in solution
                                unresolvedNewRB   = [ ID for  ID in unresolvedNewRB if ID not in newRBinTemp]       # drop these RBs from new found list


                            overlapWORB = [a for a in permIDsol2 if a not in newRBinTemp]                               # drop RBs from solution. subNewIDs always has entries.
                                                                                                                        # empty overlapWORB means subNewIDs consited only of RB
                            if len(overlapWORB) == 0:                                                                   # if solution consists only of RB
                                dataSets        = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                                #newType         = typeRing                                                              # change default to RB, keep data in regular RB storage.
                                #RBOldNewDist    = np.append(RBOldNewDist,[[oldID,newRBinTemp[0]]],axis=0)               # any relevant type casts to RB if solution is RB only.
                                #tempID          = min(permIDsol2)                                                        # store RB solution in oldX-newRB relations
                                                                                                                        # data is stored via local IDs
                                RBOldNewDist    = np.append(RBOldNewDist,app)
                            elif newType        == typeRecoveredRing:                                                                         
                                dataSets        = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                                #rRBOldNewDist    = np.append(rRBOldNewDist,[[oldID,min(permIDsol2)]],axis=0)
                                #tempID          = min(permIDsol2)
                                rRBOldNewDist    = np.append(rRBOldNewDist,app)
                                                        
                            elif newType        == typeRecoveredElse:                                                   # rDB is being cast only from rRB  
                                dataSets        = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                                #rDBOldNewDist    = np.append(rDBOldNewDist,[[oldID,min(permIDsol2)]],axis=0)
                                #tempID          = min(permIDsol2)
                                rDBOldNewDist    = np.append(rDBOldNewDist,app)
                                print(f'-{oldID}:Swapping bubble type to {typeStrFromTypeID[newType]} because it was recovered second time.')
                        
                            else:
                                dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                                #tempID          = min(permIDsol2)                                                                   # data is stored via local IDs
                                #elseOldNewDoubleCriteriumSubIDs[tempID]  = permIDsol2                                               # add to oldDB new DB relations
                                #elseOldNewDoubleCriterium.append([oldID,tempID,np.around(permDist2,2),np.around(permRelArea2,2)])   # some stats.
                                DBOldNewDist    = np.append(DBOldNewDist,app)
                            
                            #if fixedFrame == 0:
                            #    resolved    = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs,**l_DBub_old_new_IDs,**l_DBub_r_old_new_IDs,**l_MBub_old_new_IDs}
                            #    assert tempID not in resolved, f"image: #{globalCounter+dataStart}=>MERGE ERROR: shared contour is also local IDs, mask by hand :("
                            #tempStore2(permIDsol2, l_contours, tempID, err, orig, dataSets, concave = hull)
                            
                            tempStore2(permIDsol2, l_contours, ID2S(permIDsol2), err, orig, dataSets, concave = hull)

                            #if oldType == typeRing or oldType == typeRecoveredRing: unresolvedOldRB.remove(oldID)
                            
                            unresolvedOldRB.remove(oldID)   if oldID in unresolvedOldRB     else 0                              # remove from unresolved IDs
                            unresolvedOldDB.remove(oldID)   if oldID in unresolvedOldDB     else 0                              # remove from unresolved IDs
                             
                            print(f'-{oldID}:Resolved (via permutations of {subNewIDs}) old{typeStrFromTypeID[oldType]}-new{typeStrFromTypeID[newType]}: {oldID} & {permIDsol2}.')
                            
                            tempDistance    = min(tempDistance,permDist2)
                            tempSol         = permIDsol2
                        else:
                            print(f'-{oldID}:Recovery of oldDB {oldID} via permutations of {subNewIDs} has failed. Solution: {permIDsol2}.')
                            
                        print(f'--Distance prediction error: {permDist2:0.1f} vs {distCheck2:0.1f} +/- 5* {distCheck2Sigma:0.1f} and area criterium (dA/stev):{permRelArea2:0.2f} vs ~5\n')
                a = 1
                
                            
                if ((oldID in activeFrozen and tempDistance < 5 and len(tempSol) < len(subNewIDs)) or                                # i let pseudo frozens pass first
                    (oldID in partialInletIDs and len(partialInletIDs)>1 and oldID == partialInletIDs[0]) or section1STTrigger == True or section1BRTrigger == True or section1INTrigger == True):
                    if section1STTrigger == True:   # triggered by both perfect match and permutations
                        subNewIDs = subNewIDsST
                    if section1INTrigger == True:   # trigged by perfect match inlet stuff
                        subNewIDs = subNewIDsIN
                    if section1BRTrigger == True:
                        subNewIDs = subNewIDsBR     # not sure if needed, looks like it stays. tested only on perm recovery.
                    jointNeighbors.pop(mainNewID,None)                                                                     # if they are recovered with frozen
                    jointNeighbors[ID2S(tempSol)]        = tempSol                                                               # stats, then re-segment clusters
                    otherSet = [ID for ID in subNewIDs if ID not in tempSol]                                                       
                    jointNeighbors[ID2S(otherSet)]       = otherSet                                                              # then you have to recalculate parameters.
                                                                                                                                # but it should be rare.
                    jointNeighborsWoFrozen              = {mainNewID: subNewIDs for mainNewID, subNewIDs in jointNeighbors.items() if len(subNewIDs) > 0}  
                    jointNeighborsWoFrozen_hulls        = {ID: cv2.convexHull(np.vstack(l_contours[subNewIDs])) for ID, subNewIDs in jointNeighborsWoFrozen.items()}
                    jointNeighborsWoFrozen_bound_rect   = {ID: cv2.boundingRect(hull) for ID, hull in jointNeighborsWoFrozen_hulls.items()}                         
                    jointNeighborsWoFrozen_c_a          = {ID: getCentroidPosContours(bodyCntrs = [hull]) for ID, hull in jointNeighborsWoFrozen_hulls.items()}
                    print(f'Reclustering... {[a for a in jointNeighborsWoFrozen.values()]}')
                    if globalCounter not in g_splits: g_splits[globalCounter] = {}
                    g_splits[globalCounter][oldID] = [True,None,[tempSol,otherSet],0]
            if len(elseOldNewDoubleCriterium)>0:
                #print(dropDoubleCritCopies(elseOldNewDoubleCriterium))
                elseOldNewDoubleCriterium = dropDoubleCritCopies(elseOldNewDoubleCriterium)         # i dont think its relevant anymore

            print('end of oldDB/oldRB recovery') 
            # modify jointNeighbors by splitting clusters using elseOldNewDoubleCriteriumSubIDs# l_DBub_old_new_IDs
            # take recovered local IDs: DB, rDB, rRB
            removeFoundSubIDsAll    = sum(l_DBub_old_new_IDs.values(),[]) + sum(l_MBub_old_new_IDs.values(),[]) + sum(l_RBub_r_old_new_IDs.values(),[]) + sum(l_RBub_old_new_IDs.values(),[]) + sum(l_DBub_r_old_new_IDs.values(),[])
            # go though subclusters and delete resolved local IDs. drop recovered old-newRB ( min(elem) is wrong code, have to chech if its single elem)
            #jointNeighborsDelSubIDs = [ [subelem for subelem in elem if subelem not in removeFoundSubIDsAll] for elem in jointNeighborsWoFrozen.values() if min(elem) not in RBOldNewDist[:,1]] # RBO.. may be redundand
            jointNeighborsDelSubIDs = [ [subelem for subelem in elem if subelem not in removeFoundSubIDsAll] for elem in jointNeighborsWoFrozen.values()] # RBO.. may be redundand
            jointNeighborsDelSubIDs = [a for a in jointNeighborsDelSubIDs if len(a)>0]

            # >>> not sure what OG idea was, but i guess reclustering : solved relations are taken out of jointNeighbors and after combined with rest <<<
            retVal = lambda x: [x[ID] for ID in x]
            subClusters             = retVal(l_DBub_old_new_IDs) + retVal(l_MBub_old_new_IDs) + retVal(l_RBub_r_old_new_IDs) + retVal(l_RBub_old_new_IDs) + retVal(l_DBub_r_old_new_IDs)
            jointNeighbors          = {**{ID2S(a):a for a in subClusters},**{ID2S(a):a for a in jointNeighborsDelSubIDs}}
            #jointNeighbors_old      = dict(sorted(jointNeighbors.copy().items()))
            # recombine resolved clusters and filtered cluster. no resolved RB is included in new jointNeighbors
            #jointNeighbors          = {**{min(vals):vals for vals in jointNeighborsDelSubIDs if len(vals) > 0},**elseOldNewDoubleCriteriumSubIDs}
            #jointNeighbors          = dict(sorted(jointNeighbors.items()))
            #print(f'Cluster groups (updated): {jointNeighbors}') if jointNeighbors != jointNeighbors_old else print('Cluster groups unchanged')
            


            # ============================================================================================
            # ============== DROP wrong copies (when numptile old are assigned to 1 new)==================
            # ============================================================================================
            resolvedNewDBs = list([b for _,b in DBOldNewDist]) + list([b for _,b in rDBOldNewDist]) #+ list([b for _,b in FBOldNewDist])
            resolvedNewRBs = list([b for _,b in RBOldNewDist]) + list([b for _,b in rRBOldNewDist]) 
            resolvedNewMBs = list([b for _,b in MBOldNewDist])
            resolvedNewAll = resolvedNewDBs + resolvedNewRBs + resolvedNewMBs
            resolvedOldDBs = list([a for a,_ in DBOldNewDist]) + list([a for a,_ in rDBOldNewDist]) #+ list([a for a,_ in FBOldNewDist])
            resolvedOldRBs = list([a for a,_ in RBOldNewDist]) + list([a for a,_ in rRBOldNewDist])
            resolvedOldMBs = list([a for a,_ in MBOldNewDist])

            resolvedOldAll = resolvedOldDBs + resolvedOldRBs + resolvedOldMBs

            clusterIDS, counts = np.unique(resolvedNewAll, return_counts = True)
            if len(counts)>0 and max(counts) > 1:
                whereCopies = np.argwhere(counts>1).flatten()
                newIDCopies = clusterIDS[whereCopies]
                oldIDCopiesWhere = {ID:np.where(np.array(resolvedNewAll) == ID)[0] for ID in newIDCopies}
                oldIDCopies = {ID:np.array(resolvedOldAll)[where] for ID,where in oldIDCopiesWhere.items()}
                for newID,oldIDs in oldIDCopies.items():
                    predicts = {ID:l_predict_displacement[ID][0] for ID in oldIDs}
                    passID = max(predicts, key=predicts.get)
                    failIDs = [ID for ID in oldIDs if ID != passID]
                    # drop old

                    for ID in failIDs:
                        #for storage in [DBOldNewDist,rDBOldNewDist,RBOldNewDist,rRBOldNewDist,MBOldNewDist]:
                        if ID in [a for a,_ in DBOldNewDist]:
                            DBOldNewDist   = np.array([a for a in DBOldNewDist if a[0] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        elif ID in [a for a,_ in rDBOldNewDist]:
                            rDBOldNewDist   = np.array([a for a in rDBOldNewDist if a[0] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        elif ID in [a for a,_ in RBOldNewDist]:
                            RBOldNewDist   = np.array([a for a in RBOldNewDist if a[0] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        elif ID in [a for a,_ in rRBOldNewDist]:
                            rRBOldNewDist   = np.array([a for a in rRBOldNewDist if a[0] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        elif ID in [a for a,_ in MBOldNewDist]:
                            MBOldNewDist   = np.array([a for a in MBOldNewDist if a[0] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                            
                    a = 1
            print(f'{globalCounter}:--------- Merged bubble recovery ---------')
            print(f'recovering oldMB: {list(l_MBub_centroids_old.keys())}')
            # =============================== S E C T I O N - -  0 2 ==================================       
            # --------------- RECOVER PREVIOUS MERGED BUBBLES VIA DISTANCE CLUSTERING -----------------
            #    
            # conservation of area for merged bubbles is more difficult than others- convex hull
            # is not enough. sum of pre-merge areas will be lower than merged hull. 
            # funciton mergeCrit() uses known pre merge bubble orientation in order to keep
            # concave parts only at merge interface. unfortunately, concave hull is a slow operation
            
            
            delResolvedMB           = []
            leftOverIDs             = []
            leftOver_bound_rect     = {}
            #dontMergeIDs            = []
            if len(l_MBub_centroids_old)>0:
                #leftOverIDs         = sum(list(jointNeighbors.values()),[])                                 # from history- merged bubbles can consist of new unresolved RBs
                leftOverIDs         = [ID for ID in  sum(list(jointNeighbors.values()),[]) if ID not in sum(subClusters,[])]
                leftOver_bound_rect = {ID:l_rect_parms_all[ID] for ID in leftOverIDs}                       # take separately bounding rectangles of cluster elements + unresRB  
            
                
            for oldID, oldCentroid in l_MBub_centroids_old.items():
                predictCentroid_old, _, distCheck2, distCheck2Sigma     = g_predict_displacement[oldID][globalCounter-1]
                subIDs1, subIDs2                = [], []                                                               # split subcluster IDs
                split                           = False
                trajectory                      = list(g_Centroids[oldID].values())
                tempArgs                        = [
                                                    [distCheck2, distCheck2Sigma],g_predict_displacement[oldID],
                                                    5,3,2,debugVecPredict,predictVectorPathFolder,
                                                    predictCentroid_old,oldID,globalCounter,[-3,0]
                                                   ]

                predictCentroid                 = distStatPredictionVect2(trajectory, *tempArgs)
                oldMeanArea                     = g_predict_area_hull[oldID][globalCounter-1][1]
                oldAreaStd                      = g_predict_area_hull[oldID][globalCounter-1][2]
                areaCheck                       = oldMeanArea + 3*oldAreaStd 
                l_predict_displacement[oldID]   = [tuple(map(int,predictCentroid)), -1] 
                
                
                leftOver_overlap                = overlappingRotatedRectangles({oldID:l_rect_parms_old[oldID]}, leftOver_bound_rect)    # calculate overlap of oldMB with others
                leftOver_overlap_new            = [b for _,b in leftOver_overlap]                                                       # all new IDs that overlap with oldMB
                
                if len(leftOver_overlap_new)  > 0 and fixedFrame == 0:

                    # ===== to avoind using concave hull, rough estimate can be made using  convex hull =====
                    oldAreaHull = cv2.contourArea(cv2.convexHull(l_contours_hull_old[oldID], returnPoints = True))                      # we take a convex hull or old concave hull
                    permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(l_contours,l_rect_parms_all, l_rect_parms_old[oldID], toList(leftOver_overlap_new), l_centroids_all, l_areas_all,
                                                    predictCentroid, distCheck2 + 5*distCheck2Sigma, oldAreaHull, relAreaCheck = 0.7, debug = 0)      # find permutation of subIDs that result in a better
                                                                                                                                        #  match with old convex hull area and predicted centroid
                    if len(permIDsol2)>0:                                                                                               # if there is a non-trivial match 
                        mainNewID               = 512512512#min(permIDsol2)
                        #subNewIDs               = permIDsol2
                        previousInfo            = l_MBub_info_old[oldID][3:]      #[0.4*rads,tcc,25]
                        debugs = 0 #if globalCounter != 16 else 1
                        #debugs = 0 if (globalCounter != 20 aor globalCounter != 19) else 1 #if globalCounter < 21 else 1
                        debugs = 1 if globalCounter >14 and globalCounter < 21 else 0
                        a = 1
                        hull, newParams, mCrit  = mergeCrit(permIDsol2, l_contours, previousInfo, pCentr = predictCentroid, alphaParam = 0.05, debug = debugs,debug2 = 0, gc = globalCounter, id = oldID)    # calculated modified hull
                       
                        #if globalCounter == 7:
                        #    blank = err.copy()
                        #    cv2.drawContours(  blank,   [hull], -1, 120, 2)
                        #    cv2.imshow('ads',blank)
                        newCentroid, newArea    = getCentroidPosContours(bodyCntrs = [hull])                                            # centroid and hull of a modified hull
                        dist2                   = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))                     # predictor error
                        areaCrit                = np.abs(newArea-oldMeanArea)/ oldAreaStd                                               # area diffference in terms of stdevs
                        
                        if len(permIDsol2) > 1 and mCrit[0] == True and mCrit[1].shape[0] > 0:
                            direction = np.matmul(np.array([[0, -1],[1, 0]]), newParams[1])
                            debugg = 0 #if oldID != 27 else 1
                            split, subIDs1, subIDs2 = mergeSplitDetect(l_contours,permIDsol2,direction,mCrit[1],globalCounter,oldID, debug = debugg)
                            #if split == True:
                            #    [unresolvedNewRB.remove(x) for x in unresolvedNewRB if x in subIDs1 + subIDs2]

                        if split == False and dist2 <= distCheck2 + 5*distCheck2Sigma and areaCrit < 3:                                                    # a perfect match
                            tempID  = ID2S(permIDsol2)
                            app     = np.array((oldID, tempID), dtype=[('integer', '<i4'), ('string', '<U60')])
                            
                            if mCrit[0] == True:          # still merging
                                dataSets  = [l_MBub_masks, l_MBub_images,l_MBub_old_new_IDs, l_MBub_rect_parms, l_MBub_centroids, l_MBub_areas_hull, l_MBub_contours_hull]
                                #MBOldNewDist    = np.append(MBOldNewDist,[[oldID,min(permIDsol2)]],axis=0) 
                                #tempID          = min(permIDsol2)
                                MBOldNewDist    = np.append(MBOldNewDist,app)
                                
                                l_MBub_info[tempID]                      = [[],newArea,newCentroid] + list(newParams)                        # store new merge params
                                
                            elif len(permIDsol2) == 1 and str(permIDsol2[0]) in unresolvedNewRB:            # if result is one ring bubble and its missing- save as RB
                                dataSets        = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                                #RBOldNewDist    = np.append(RBOldNewDist,[[oldID,tempID]],axis=0) 
                                RBOldNewDist    = np.append(RBOldNewDist,app)
                                                                                       # store RB solution in oldX-newRB relations
                            else:
                                dataSets  = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                                #elseOldNewDoubleCriteriumSubIDs[tempID]  = permIDsol2                                               # add to oldDB new DB relations
                                #elseOldNewDoubleCriterium.append([oldID,tempID,np.around(dist2,2),np.around(areaCrit,2)]) 
                                DBOldNewDist = np.append(DBOldNewDist,app)
                            tempStore2(permIDsol2, l_contours, tempID, err, orig, dataSets, concave = hull)                               # store a solution

                            l_predict_displacement[oldID]            = [tuple(map(int,predictCentroid)), int(dist2)]                    # store predictor error
                            delResolvedMB.append(list(permIDsol2))                                                                      # store merge subIDs
                            [unresolvedNewRB.remove(x) for x in unresolvedNewRB if int(x) in permIDsol2]                                 # remove unresolved RBs in case they are in solution
                            if oldID in unresolvedOldDB: unresolvedOldDB.remove(oldID)
                            if oldID in unresolvedOldRB: unresolvedOldRB.remove(oldID)
                            print(f'-{oldID}:Resolved (via permutations of {leftOver_overlap_new}) oldMB-newMB: {oldID} & {permIDsol2}.')
                        else:
                            print(f'-{oldID}:Recovery of oldDB {oldID} via permutations of {leftOver_overlap_new} has failed. Solution: {permIDsol2}.')
                        print(f'--Distance prediction error: {dist2:0.1f} vs {distCheck2:0.1f} +/- 2* {distCheck2Sigma:0.1f} and area criterium (dA/stev):{areaCrit:0.2f} vs ~5\n')
                # === finalizing DB clusters ===

                if split == True:
                    splitHulls                      = {min(subIDs):cv2.convexHull(np.vstack(l_contours[subIDs])) for subIDs in [subIDs1, subIDs2]} 
                    splitCentroidsAreas             = {ID:getCentroidPosContours(bodyCntrs = [hull]) for ID,hull in splitHulls.items()}         
                    splitCentroidDistances          = {ID:np.linalg.norm(predictCentroid - CA[0]).astype(int) for ID,CA in splitCentroidsAreas.items()}
                    splitSmallestDistID             = min(splitCentroidDistances, key=splitCentroidDistances.get)
                    ccDist = closes_point_contours(splitHulls[min(subIDs1)],splitHulls[min(subIDs2)])[1]
                    l_splits[oldID]                 = [True, splitCentroidsAreas, [subIDs1, subIDs2],ccDist]
                    for subIDs in [subIDs1, subIDs2]:
                        app     = np.array((oldID, ID2S(subIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                        if min(subIDs) == splitSmallestDistID:                  # is solution = closest to previous centroid.
                            dist2                           = splitCentroidDistances[splitSmallestDistID]
                            newArea                         = splitCentroidsAreas[splitSmallestDistID][1]
                            areaCrit                        = np.abs(newArea-oldMeanArea)/ oldAreaStd
                            l_predict_displacement[oldID]   = [tuple(map(int,predictCentroid)), int(dist2)] 
                        jointNeighbors = {ID:[subID for subID in vals if subID not in subIDs ] for ID,vals in jointNeighbors.items()}     # drop subIDs from clusters.
                        subIDsWoRBs     = [a for a in subIDs if str(a) not in unresolvedNewRB]                                                 # drop unresolved RBs from subcluster copy
                        if len(subIDsWoRBs)==0:
                            dataSets        = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                            if min(subIDs) == splitSmallestDistID:
                                #RBOldNewDist    = np.append(RBOldNewDist,[[oldID,min(subIDs)]],axis=0)               
                                RBOldNewDist    = np.append(RBOldNewDist,app)
                               
                        elif len(subIDs) < len(subIDsWoRBs) and len(subIDsWoRBs) > 0:                                                         # if subcluster copy got smaller, then there was RB inside.
                            dataSets        = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                            if min(subIDs) == splitSmallestDistID:
                                #rRBOldNewDist    = np.append(rRBOldNewDist,[[oldID,min(subIDs)]],axis=0)
                                rRBOldNewDist    = np.append(rRBOldNewDist,app)
                        else :
                            dataSets  = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                            if min(subIDs) == splitSmallestDistID:
                                #elseOldNewDoubleCriteriumSubIDs[min(subIDs)]  = subIDs                                               # add to oldDB new DB relations
                                #elseOldNewDoubleCriterium.append([oldID,min(subIDs),np.around(dist2,2),np.around(areaCrit,2)])
                                DBOldNewDist    = np.append(DBOldNewDist,app)
                        [unresolvedNewRB.remove(str(x)) for x in subIDs if str(x) in unresolvedNewRB]
                        #delResolvedMB.append(list(subIDs))
                        tempStore2(subIDs, l_contours, ID2S(subIDs), err, orig, dataSets, concave = splitHulls[min(subIDs)]) 
                        if oldID in unresolvedOldDB: unresolvedOldDB.remove(oldID)
                        if oldID in unresolvedOldRB: unresolvedOldRB.remove(oldID)
                            #tempStore2(subIDs, l_contours, min(subIDs), err, orig, dataSets, concave = 0)                                    # len(subIDsWoRBs) == 0 would mean that there was only one RB inside
                            #[unresolvedNewRB.remove(x) for x in unresolvedNewRB if x in subIDs]                                         # but that one RB should be kept in unresolvedNewRB, so it forms a new RB
                            #jointNeighbors = {ID:[subID for subID in vals if subID not in subIDs ] for ID,vals in jointNeighbors.items()}
                    jointNeighbors = {**{ID2S(vals):vals for vals in jointNeighbors.values() if len(vals) > 0},**{ID2S(sub):sub for sub in [subIDs1, subIDs2] }}
                if fixedFrame == 1:
                    # get rough overlap with existing correct (?) clusters.
                    params = {ID:cv2.boundingRect(np.vstack(l_contours[vals])) for ID,vals in jointNeighbors.items()}
                    # get rough overlap, to drop everything useless.
                    leftOver_overlap                = overlappingRotatedRectangles({oldID:l_rect_parms_old[oldID]}, params)    # calculate overlap of oldMB with others
                    # modify so it looks like ist of permutations
                    leftOver_overlap_new            = [jointNeighbors[b] for _,b in leftOver_overlap]
                    oldAreaHull = cv2.contourArea(cv2.convexHull(l_contours_hull_old[oldID], returnPoints = True))                      # we take a convex hull or old concave hull
                    permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(l_contours,l_rect_parms_all, l_rect_parms_old[oldID], [], l_centroids_all, l_areas_all,
                                                    predictCentroid, oldAreaHull*100, oldAreaHull, relAreaCheck = 100, customPermutations = leftOver_overlap_new ,debug = 0)      # find permutation of subIDs that result in a better
                    
                    if len(permIDsol2) > 0:
                                
                        dataSets  = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                        #elseOldNewDoubleCriteriumSubIDs[min(permIDsol2)]  = permIDsol2                                               # add to oldDB new DB relations
                        #elseOldNewDoubleCriterium.append([oldID,min(permIDsol2),np.around(permDist2,2),np.around(permRelArea2,2)]) 
                        tempID = ID2S(permIDsol2)
                        app     = np.array((oldID, tempID), dtype=[('integer', '<i4'), ('string', '<U60')])
                        DBOldNewDist    = np.append(DBOldNewDist,app)
                        tempStore2(permIDsol2, l_contours, tempID , err, orig, dataSets, concave = 0)                               # store a solution

                        l_predict_displacement[oldID]            = [tuple(map(int,predictCentroid)), int(permDist2)]                    # store predictor error
                            
                        delResolvedMB.append(list(permIDsol2))                                                                      # store merge subIDs
                        [unresolvedNewRB.remove(x) for x in unresolvedNewRB if int(x) in permIDsol2]                                 # remove unresolved RBs in case they are in solution
                        if oldID in unresolvedOldDB: unresolvedOldDB.remove(oldID)
                        if oldID in unresolvedOldRB: unresolvedOldRB.remove(oldID)
                        print(f'-{oldID}:Resolved with fixedFrame (choice of {leftOver_overlap_new}) solution: {oldID} & {permIDsol2}.')

                            
            print('end of oldMB recovery')             

            # --- resolved MB should be deleted from clusters alltogether ---
            # 9999% of the time merge is retrieved from unresolved cluster. but it can intsect resolved cluster.
            # solution based on problem where two clusters (MB and DB) (one near inlet) were split into two. next step MB wanted to share a contour.
            # re-segmentation means that it will take prio and cut out part of resolved DB.
            #MBIntersectCount = {}
            #for ID,subIDs in l_MBub_old_new_IDs.items():
            #    MBIntersectCount[ID] = [[ID2 for ID2,cluster in jointNeighbors.items() if b in cluster] for b in subIDs]
            #if len(delResolvedMB)>0:
            oldNewIdsResolvedD      = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs, **l_DBub_r_old_new_IDs,  **l_MBub_old_new_IDs}

            removeFoundSubIDsAll    = sum(delResolvedMB,[])                                                                                         # get all subIDs of recovered MBs
            jointNeighborsDelSubIDs = []                                                                                                            # 
            jointNeighborsDelSubIDs = [ [subelem for subelem in elem if subelem not in removeFoundSubIDsAll] for elem in jointNeighbors.values()]
            jointNeighborsNoMB        = {ID2S(vals):vals for vals in jointNeighborsDelSubIDs if len(vals) > 0 and ID2S(vals) not in oldNewIdsResolvedD}  # should drop MB out of cluster, only remains left
            jointNeighbors          = {**oldNewIdsResolvedD,**jointNeighborsNoMB}               # basically will pop out resolved MBs out of old clusters.
            #print(MBIntersectCount)
            #if len(MBIntersectCount)>0 and all([len(a) == 1 for a in MBIntersectCount.values()]):  
            #    removeFoundSubIDsAll    = sum(delResolvedMB,[])                                                                                         # get all subIDs of recovered MBs
            #    jointNeighborsDelSubIDs = []                                                                                                            # 
            #    jointNeighborsDelSubIDs = [ [subelem for subelem in elem if subelem not in removeFoundSubIDsAll] for elem in jointNeighbors.values()]   # get all cluster subIDs that are not in removeFoundSubIDsAll
            #    jointNeighbors          = {**{min(vals):vals for vals in jointNeighborsDelSubIDs if len(vals) > 0},**elseOldNewDoubleCriteriumSubIDs}   # not clear. same 2 lists overlayed < EDIT !NOPE! but looks useless
            # --- cluster have twice been recombined ( recover & merge-split), so some connectivity might be lost ---
            # --- must do overlap within clusters and then clusters must be devided based on connectivity --- *** pre delResolvedMB jointNeighbors have holes, might be problematic ***
            print(f'Reclustering jointNeighbors in case of discontinueties due to recovery. old Clusters: {jointNeighbors}')
            bigClusterIDs           = [ID for ID,subIDs in jointNeighbors.items() if len(subIDs)>1]
            for ID in bigClusterIDs:                                                             # does not interact with merge-split = good.
                rectParams = {subID:l_rect_parms_all[subID] for subID in jointNeighbors[ID]}
                combosSelf = np.array(overlappingRotatedRectangles(rectParams,rectParams))
                cc_unique  = graphUniqueComponents(jointNeighbors[ID],combosSelf, edgesAux= dOldNewAll, debug = 0, bgDims = {1e3,1e3}, centroids = [], centroidsAux = [], contours = [], contoursAux = [])
                jointNeighbors.pop(ID, None)
                for subIDs in cc_unique:
                    jointNeighbors[ID2S(subIDs)] = subIDs
            print(f'Reclustering jointNeighbors in case of discontinueties due to recovery. new Clusters: {jointNeighbors}')
            # ===========================================================================================
            # ====================================== INLET STUFF ========================================
            # ===========================================================================================

           #centroidsResolved   = {**l_RBub_centroids,  **l_RBub_r_centroids,   **l_DBub_centroids,   **l_DBub_r_centroids,     **l_FBub_centroids,       **l_MBub_centroids}
           #areasResolved       = {**l_RBub_areas_hull, **l_RBub_r_areas_hull,  **l_DBub_areas_hull,  **l_DBub_r_areas_hull,    **l_FBub_areas_hull,      **l_MBub_areas_hull}
           #oldNewIdsResolved   = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs, **l_DBub_r_old_new_IDs,   **l_FBub_old_new_IDs,     **l_MBub_old_new_IDs}
            centroidsResolved   = {**l_RBub_centroids,  **l_RBub_r_centroids,   **l_DBub_centroids,   **l_DBub_r_centroids,     **l_MBub_centroids,     **l_FBub_centroids}
            areasResolved       = {**l_RBub_areas_hull, **l_RBub_r_areas_hull,  **l_DBub_areas_hull,  **l_DBub_r_areas_hull,    **l_MBub_areas_hull,    **l_FBub_areas_hull}
            oldNewIdsResolved   = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs, **l_DBub_r_old_new_IDs,   **l_MBub_old_new_IDs,   **l_FBub_old_new_IDs}
            rectParamsOld       = {oldID:l_rect_parms_old[oldID] for oldID in unresolvedOldDB + unresolvedOldRB }#if oldID not in inletIDs        # old global IDs with bounding rectangle parameters
            unresolvedClusters  = {ID:subIDs for ID,subIDs in jointNeighbors.items() if ID not in oldNewIdsResolved}
            resolvedContoursNew     = sum(oldNewIdsResolved.values(),[])
            unresolvedInletContours = [a for a in inletIDsNew if a not in resolvedContoursNew + declusterNewFRLocals]
            temp = {}
            fullyOverlappingOldInletIDs = [oldID for oldID, cType in inletIDsType.items() if cType == 2]       # these are fully inside and are not resolved.
            if len(unresolvedClusters)>0:                                                           # they are together with other new inlet elements
                for ID,subIDs in jointNeighbors.items():                                        # determine which cluster is overlapping inlet recangle. might be done conentional way..
                    if ID not in oldNewIdsResolved:
                        match = [a in subIDs for a,tp in inletIDsTypeNew.items() if tp == 2]
                        temp[ID] =  sum(match)                                                      # count matches of inlet new IDs in clusters
                clusterID    = max(temp, key=temp.get)                                          # cmnt: if 2 present more matches takes precidence
            else:
                clusterID = 0
                temp[0]   = 0
            crit1               = temp[clusterID] > 0 and len(unresolvedInletContours) > 0          # drop resolved IDs from inlet cluster, if there are spare elements do your stuff. left part may be redundant
            missingFullInletOld = [ID for ID,cType in inletIDsType.items() if cType == 2]           # fully inside bubbles are not resolved prior, in principle
            crit2               = len(missingFullInletOld) > 0 and len(unresolvedInletContours) > 0 # they are here and we have spare new contours.
            if crit1 or crit2:# inlet cluster  maybe fully resolved by old-newDB, which happens if its poking outside inlet box.
                a = 1                                                                                   # so inletIDsNew = resolved subcluster.
                idsInside   = [ID for ID, cType in inletIDsTypeNew.items() if cType == 2]               # ids fully inside inlet rectangle. might include parts of unresolved neighbor clusters
                idsPartial  = [ID for ID, cType in inletIDsTypeNew.items() if cType == 1 if ID not in sum(list(oldNewIdsResolved.values()),[]) + declusterNewFRLocals]               # ids that are partially inside inlet rectangle. again, might be part of unresolved clusters.
                overlappintPartialOldIDs    = [ID for ID,cType in inletIDsType.items() if cType == 1]   # there are partially overlaying old IDs
                partiallyOverlappingOldInletIDs = [a for a in overlappintPartialOldIDs if a in unresolvedOldRB + unresolvedOldDB] # which of these are missing

                print(f'\nProcessing bubble next to inlet. inside region: {idsInside}, on border: {idsPartial}, unresolved old IDs that overlap inet zone: {partiallyOverlappingOldInletIDs}')
                #if len(partiallyOverlappingOldInletIDs)==0 and : 
                #for oldID, tp in inletIDsType.items():
                #    if tp == 2 and len(idsInside)>0:
                mainInletCluster      = []                                                    # see if else below
                if max(list(temp.values())) == 0:
                    #assert len(idsPartial) <= 1, 'inlet kuku'
                    clusterID = [ID2S(a) for a in jointNeighbors.values() if idsPartial[0] in a][0]
                                                                                        # if a partial old inlet was resolved, it was dropped from big inlet cluster.
                subIDs          = jointNeighbors[clusterID].copy()                      # take IDs of elements in relevant cluster
                includeIDs      = [ID for ID in subIDs if ID        in idsInside]       # extract only those inside. maybe i dont have do this step, but take idsInside. not sure.
                distanceIDs     = []
                if len(partiallyOverlappingOldInletIDs)==0:                                 # partial IDs could be part of unresolved cluster, like MB, which are determined next
                    includeIDs = includeIDs + [ID for ID in subIDs if ID in idsPartial] # or they can be part of inlet bubble. in case there are no missing bubbles, include them into IB
                    distanceIDs = [ID for ID in subIDs if ID not in includeIDs]
                    print(f'no old bubbles in proximity, include border IDs into clusters: {includeIDs}')
                else:
                    print(f'inlet: unresolved partial old ID: {partiallyOverlappingOldInletIDs} might merge with bubbles on top or bottom')
                    # there is half-overlapping unresolved old ID, OLD IDEA:some contours fully in inlet rectangle can be part of it. 
                    # NEW IDEA: unresolved- merged with something. can merge with something on top or something inside inlet.
                    # --------  if on top, should be managed by proper merging algorithm after this. if below, can cheat a bit here.
                    # since partiallyOverlappingOldInletIDs was connected to inlet zone, it will join new frame both top and bottom merged into inlet cluster (through old-new rect)
                    missingOverlapWithInletCluster    = overlappingRotatedRectangles(                           # we have unresolved partially inlet bubble
                                {ID:l_rect_parms_old[ID] for ID in partiallyOverlappingOldInletIDs},            # means it changed too much. which might indicate
                                {ID:l_rect_parms_all[ID] for ID in subIDs})                                     # that it has merged either to top bubble or bottom
                                                                                      
                    mainInletCluster  = np.unique([b for _,b in missingOverlapWithInletCluster]).tolist()       # missing old overlap with this object that part of a merge 
                    # == lets find if old unresolved partial is merging with non-inlet bubbles. ==
                    unresolvedOlds = [a for a in unresolvedOldDB + unresolvedOldRB if a not in inletIDsType]    # unresolved olds which are not related to inlet
                    overlapMissingPartialAndUnresOld    = overlappingRotatedRectangles(                         # merging contour overlaps with these
                                                            {ID:l_rect_parms_old[ID] for ID in unresolvedOlds}, # unresolved (non-inlet) old IDs
                                                            {ID:l_rect_parms_all[ID] for ID in mainInletCluster})     # 
                    if len(overlapMissingPartialAndUnresOld) == 0:                                              
                    # == there is no overlap with non-inlet- then it merges with inlet dudes. ==
                        print(f'inlet: unresolved partial old ID: {partiallyOverlappingOldInletIDs} merges from the bottom!')
                        _,yFb,_,hFB = fakeBox[-1]                                                               # mainInletCluster is the main cluster.
                        rectParamsNew   = {ID:l_rect_parms_all[ID] for ID in subIDs if ID not in mainInletCluster}    # take all rest from inlet cluster but main
                        rectParamsNewM  = {ID:[x-2,yFb,w+2,hFB] for ID,[x,y,w,h] in rectParamsNew.items()}      # modify their boundRect  to vertical columns  <<<
                        if len(rectParamsNewM)>0:                                                               # with old width. specific to our illumination <<<
                            restMainOverlap    = overlappingRotatedRectangles(rectParamsNewM,                   # see which are overlapping with main cluster
                                                {ID:l_rect_parms_all[ID] for ID in mainInletCluster})      
                            mainOverlap = list(map(list,list(itertools.combinations(mainInletCluster, 2))))           # retrieve interconnectedness of main clsuter  
                            restMainOverlap = mainOverlap + restMainOverlap                                     # combine connections
                            cc_unique   = graphUniqueComponents(subIDs, restMainOverlap)                        # recluster inlet baseed on new connections
                            if len(cc_unique) == 1:                                                             # base and candidates merged into one cluster
                                includeIDs = cc_unique[0]                                                       # means all IDs fully inside are not restored 
                                distanceIDs = []                                                                # inside cluster. they wont be recovered
                                fullyOverlappingOldInletIDs = []                                                # drop them
                                print(f'inlet: unresolved partial old ID: {partiallyOverlappingOldInletIDs} merges with inlet IDs, forming one cluster: {includeIDs}!')
                            else:                                                       # multiple clusters
                                # == check if split clusters are overlapping new main cluster
                                print(f'inlet: unresolved partial old ID: {partiallyOverlappingOldInletIDs} merges with inlet IDs, forming multiple clusters: {cc_unique}!')
                                if len(mainInletCluster)==0:
                                    includeIDs = includeIDs
                                    distanceIDs = []
                                else:
                                    mainID = [min(comb) for comb in cc_unique if mainInletCluster[0] in comb][0]                         # gtrab
                                    bRects3 = {min(combs):cv2.boundingRect(np.vstack(l_contours[combs])) for combs in cc_unique}
                                    overlap4    = overlappingRotatedRectangles(                    
                                                {mainID:bRects3[mainID]},
                                                {ID:vals for ID,vals in bRects3.items() if ID != mainID}) 
                                    includeIDs = sum([comb for comb in cc_unique if min(comb) in [mainID] + [b for _,b in overlap4]],[])
                                    if len(overlap4) == (len(cc_unique)-1):
                                        distanceIDs = []
                                        print(f'inlet: all split combs are joined by overlap')
                                    else:
                                        print(f'inlet: failed to join split all split combs  by overlap!')
                                        distanceIDs = sum([comb for comb in cc_unique if min(comb) not in [mainID] + [b for _,b in overlap4]],[])
                                #if len(mainInletCluster)>0: # tried to fix some bug on top, but maybe its was not what i though. different bug. look through log more thoroughly
                                #    print(f'inlet: no overlap
                                #    mainID = [min(comb) for comb in cc_unique if mainInletCluster[0] in comb][0]
                                #    bRects3 = {min(combs):cv2.boundingRect(np.vstack(l_contours[combs])) for combs in cc_unique}
                                #    overlap4    = overlappingRotatedRectangles(                    
                                #                {mainID:bRects3[mainID]},
                                #                {ID:vals for ID,vals in bRects3.items() if ID != mainID}) 
                                #    includeIDs = sum([comb for comb in cc_unique if min(comb) in [mainID] + [b for _,b in overlap4]],[])
                                #    if len(overlap4) == (len(cc_unique)-1):
                                #        distanceIDs = []
                                #        print(f'inlet: all split combs are joined by overlap')
                                #    else:
                                #        print(f'inlet: failed to join split all split combs  by overlap!')
                                #        distanceIDs = sum([comb for comb in cc_unique if min(comb) not in [mainID] + [b for _,b in overlap4]],[])
                                #else:
                                #    includeIDs = sum(cc_unique,[])
                                #    distanceIDs = []
                        else: # subIDs = mainInletCluster -> means there are no merge candidates since OG overlaps whole inelt cluster (WOW! but happens)
                            includeIDs = mainInletCluster
                            distanceIDs = []
                             #assert 1 == 2, 'havent encountered this case'
                        #includeIDs  = [a for a in includeIDs if a not in mainInletCluster]        # drop mainInletCluster IDs from fully inside
                        #leaveIDs    = leaveIDs + mainInletCluster                                 # add mainInletCluster IDs to remaining unresolved cluster
                    else:               # if len(overlapMissingPartialAndUnresOld) > 0 => its merging from top. take all merging clster and drop other stinky inlet IDs
                        print(f'inlet: unresolved partial old ID: {partiallyOverlappingOldInletIDs} DOES merge from the top!')
                        #
                        topMergeCluster0    = overlappingRotatedRectangles(                     
                                                    {ID:l_rect_parms_old[ID] for ID in partiallyOverlappingOldInletIDs + [overlapMissingPartialAndUnresOld[0][0]]},
                                                    {ID:l_rect_parms_all[ID] for ID in subIDs})
                        topMergeCluster    = np.unique([b for _,b in topMergeCluster0]).astype(int)
                        dropIDs             = [ID for ID in subIDs if ID not in topMergeCluster]
                        dropIDs             = {ID2S(dropIDs):dropIDs} if len(dropIDs)>0 else {}
                        mergeCluster        = {ID2S(topMergeCluster):topMergeCluster.tolist()} if len(topMergeCluster)>0 else {}
                        jointNeighbors.pop(clusterID, None)
                        clusterID = -1                                                                  # so next pop does not work lol.
                        jointNeighbors = {**jointNeighbors,**dropIDs,**mergeCluster}
                        [unresolvedNewRB.remove(ID) for ID in [min(subIDs)] if ID in unresolvedNewRB]
                        includeIDs = []
                        distanceIDs = []
                        if len(fullyOverlappingOldInletIDs)>0 and len(dropIDs)>0 : #after merge on top, if any IDs are left below and there was a ~missing ID-> ...
                            tempID = list(dropIDs.keys())[0]
                            oldID = fullyOverlappingOldInletIDs[0]
                            dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                            tempStore2(dropIDs[tempID], l_contours, tempID, err, orig, dataSets, concave = 0)
                            app = np.array((oldID, tempID), dtype=[('integer', '<i4'), ('string', '<U60')])
                            DBOldNewDist = np.append(DBOldNewDist,app)
                            
                            l_predict_displacement[oldID]            = [tuple(map(int,l_DBub_centroids[tempID])), 25]
                        
                    print(f'there are old bubbles in proximity, split inlet clusters into  {includeIDs}')
                if len(includeIDs)>0:
                    includeArea = cv2.contourArea(cv2.convexHull(np.vstack(l_contours[includeIDs])))
                    includeIDs  = includeIDs if includeArea > 550 else []
                jointNeighbors.pop(clusterID, None)
                
                    
                    
                if len(includeIDs)>0:                                                               
                    jointNeighbors  = {**jointNeighbors,**{ID2S(includeIDs):includeIDs}}
                            # there is old ID fully inside inlet zone.
                    hull            = cv2.convexHull(np.vstack(l_contours[includeIDs]))
                    newID           = ID2S(includeIDs)
                    dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    # store solution locally, but dont yet relate it to oldID
                    tempStore2(includeIDs, l_contours, newID, err, orig, dataSets, concave = hull)

                    if len(partiallyOverlappingOldInletIDs) > 0 or len(fullyOverlappingOldInletIDs)>0:              # there were unresolved partial or fully inside oldIDs
                        if len(partiallyOverlappingOldInletIDs) > 0:                                                # partial failed to be resolved via main loop
                            #assert len(partiallyOverlappingOldInletIDs) == 1, 'multiple unresolved inlet IDS'       # lets hope there only one partial :(
                            oldID = partiallyOverlappingOldInletIDs[0]                                              # it should take priority, since below bubbles are flukes
                        elif len(fullyOverlappingOldInletIDs)>0:                                                    # if there are only inside oldIDs, which are not
                           aresOld         = {ID:l_areas_hull_old[ID] for ID in fullyOverlappingOldInletIDs}        # resolved in main loop, just take the 
                           oldID        = max(aresOld, key = aresOld.get)                                           # biggest one and assign cluster to it
                        
                        if len(partiallyOverlappingOldInletIDs) == 1 or len(fullyOverlappingOldInletIDs)>0:         # quick fix-test for multiple inlet IDs
                            app = np.array((oldID, newID), dtype=[('integer', '<i4'), ('string', '<U60')])              # relate old-new to whosever got a priority
                            DBOldNewDist = np.append(DBOldNewDist,app)                                                  #
                            
                            l_predict_displacement[oldID]            = [tuple(map(int,l_DBub_centroids[newID])), 25]
                            unresolvedOldRB.remove(oldID)   if oldID in unresolvedOldRB     else 0                      # remove from unresolved IDs
                            unresolvedOldDB.remove(oldID)   if oldID in unresolvedOldDB     else 0                      # remove from unresolved IDs
                            [unresolvedNewRB.remove(str(a)) for a in includeIDs if str(a) in unresolvedNewRB]
                            rectParamsOld.pop(oldID, None)              # IDK WHY ?!!
                            #print(f'relating inlet cluster to old ID: {oldID}')
                            #print(f'Save inlet cluster: {includeIDs}, distance fail elements: {distanceIDs}.')
                            if len(partiallyOverlappingOldInletIDs) > 0:
                                print(f'-{oldID}:Recovery of old PARTIAL inlet ID: {oldID} from inlet cluster. main IDs: {includeIDs} and secondary {distanceIDs}')
                            else:
                                print(f'-{oldID}:Recovery of old FULLy inside inlet ID: {oldID} from inlet cluster. main IDs: {includeIDs} and secondary {distanceIDs}')
                    else:
                        #print(f'no old ID found, addin as new ID')
                        #print(f'Save inlet cluster: {includeIDs}, distance fail elements: {distanceIDs}.')
                        print(f'-No oldID recovery, forming new clusters. main IDs: {includeIDs} and secondary {distanceIDs}')
                if len(distanceIDs)>0:                                                               
                    jointNeighbors  = {**jointNeighbors,**{ID2S(distanceIDs):distanceIDs}}          # add distance ones as single element clusters
                

            #----------------- consider else clusters are finalized ---------------------

                
            #oldNewDB = np.array([arr[:2] for arr in elseOldNewDoubleCriterium],int)  # array containing recovered DBs old-> new. its used to write global data to storage
            #oldNewDB = oldNewDB if len(oldNewDB) > 0 else np.empty((0,2), np.uint16) 
            # first definition of unresolvedNewDB. consists of resolved DBs, unresolved cluster ids and unresolved RBs
            # holds main local IDs- means min(subIDs) e.g 17:[17,29,31]
            #unresolvedNewDB = np.unique(list(jointNeighbors.keys()) + unresolvedNewRB) #  bad fix for not dropping unresolvedNewRB from merge-split
            #unresolvedNewDB = list(jointNeighbors.keys()) #  bad fix for not dropping unresolvedNewRB from merge-split
            a = 1

            # ==== clearing recovered DB bubbles ====
            #if len(oldNewDB) > 0: # if all distance checks fail (oldNewDB)
            #    #unresolvedNewDB = [A for A in jointNeighbors.keys() if A not in oldNewDB[:,1] and A not in frozenOldGlobNewLoc.keys()];   # drop resolved new DB from unresolved new DB
            #    unresolvedNewDB = [A for A in jointNeighbors.keys() if A not in oldNewDB[:,1]];   # drop resolved new DB from unresolved new DB
            #    #unresolvedOldDB = [A for A in unresolvedOldDB if A not in oldNewDB[:,0] and A not in resolvedFrozenGlobalIDs];              # drop resolved old DB from unresolved old DB
            #    unresolvedOldDB = [A for A in unresolvedOldDB if A not in oldNewDB[:,0]];              # drop resolved old DB from unresolved old DB
            #    for ID in unresolvedNewDB.copy():
            #         subIDs =  jointNeighbors[ID]
            #         if len(subIDs) == 1 and ID in unresolvedNewRB:              # >>>>>>>>>> not sure what it does <<<<<<<<<<<<<
            #             unresolvedNewDB.remove(ID)                              # if cluster is solo bubble and main ID is RB, it should be RB. should add it as RB
            resolvedOldDBs = list([a for a,_ in DBOldNewDist]) + list([a for a,_ in rDBOldNewDist]) #+ list([a for a,_ in FBOldNewDist])
            resolvedOldRBs = list([a for a,_ in RBOldNewDist]) + list([a for a,_ in rRBOldNewDist])
            resolvedOldMBs = list([a for a,_ in MBOldNewDist])

            resolvedOldAll = resolvedOldDBs + resolvedOldRBs + resolvedOldMBs

            unresolvedOldDB = [A for A in unresolvedOldDB if A not in resolvedOldAll];              # drop resolved old DB from unresolved old DB
            unresolvedOldRB = [A for A in unresolvedOldRB if A not in resolvedOldAll];


            resolvedNewDBs = list([b for _,b in DBOldNewDist]) + list([b for _,b in rDBOldNewDist]) #+ list([b for _,b in FBOldNewDist])
            resolvedNewRBs = list([b for _,b in RBOldNewDist]) + list([b for _,b in rRBOldNewDist]) 
            resolvedNewMBs = list([b for _,b in MBOldNewDist])
            
            resolvedNewAll = resolvedNewDBs + resolvedNewRBs + resolvedNewMBs
            resolvedNewAllSubIDs = sum([jointNeighbors[ID] for ID in resolvedNewAll],[])
            
             
            [unresolvedNewRB.remove(str(x)) for x in resolvedNewAllSubIDs if str(x) in unresolvedNewRB]
            #unresolvedNewRB = [A for A in unresolvedNewRB if A not in resolvedNewAllSubIDs];   # these were known from the start by parent-childer stuff. But i ussualy drop them out when i find them inside resolved comb.

            unresolvedNewDB = [A for A in jointNeighbors.keys() if A not in resolvedNewAll and A not in unresolvedNewRB];   # drop resolved new DB from unresolved new DB

            a = 1
            #unresolvedNewDB = unresolvedNewDB if len(unresolvedNewDB) > 0 else np.empty((0,2), np.uint16)
            #if len(unresolvedOldDB)>0: print(f'{globalCounter}:--------- Begin recovery of Else bubbles: {unresolvedOldDB} --------')
            #else:                   print(f'{globalCounter}:----------------- No Else bubble to recover ---------------------')
                
            #jointNeighbors = {ID:subIDs for ID,subIDs in jointNeighbors.items() if ID not in oldNewDB[:,1]} if len(oldNewDB)> 0 else jointNeighbors
            #if len(jointNeighbors)> 0:
            #    for ID,cntrIDs in jointNeighbors.items():
            #        if (ID in unresolvedNewRB and len(cntrIDs) > 1) or (ID not in unresolvedNewRB) :           # if main ID is RB, but cluster has more bubbles.
            #            dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
            #            tempStore2(cntrIDs, l_contours, ID, err, orig, dataSets, concave = 0)
            #            if (ID in unresolvedNewRB and len(cntrIDs) > 1):
            #                unresolvedNewRB.remove(ID)
            
            # idk about OG IDEA but i guess its more about saving clusters with RB+other to DB..

            #jointNeighbors = {ID:subIDs for ID,subIDs in jointNeighbors.items() if ID not in oldNewDB[:,1]} if len(oldNewDB)> 0 else jointNeighbors
            #if len(jointNeighbors)> 0:
            #    clustersWithRB = [[ID for ID,subIDs in jointNeighbors.items() if int(RBID) in subIDs and len(subIDs) > 1] for RBID in unresolvedNewRB]
            #    for ID,cntrIDs in jointNeighbors.items():
            #        if ID in clustersWithRB:           
            #            #dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
            #            #tempStore2(cntrIDs, l_contours, ID, err, orig, dataSets, concave = 0)
            #            if ID in unresolvedNewRB: unresolvedNewRB.remove(ID)
                    
                #unresolvedNewDB = unresolvedNewDB + [elem for elem in list(jointNeighbors.keys()) if elem not in unresolvedNewDB]
                        
    if globalCounter > 0:
        print(f'{globalCounter}:--------- Begin merge detection/processing --------\n')
        # ======== PART 00: test if two resolved contours share a contour on same frame. rare but happens. not completed! =========
        cc_oldIDs, cc_newIDs = [], []
        oldNewIdsResolvedD   = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs, **l_DBub_r_old_new_IDs,   **l_FBub_old_new_IDs,     **l_MBub_old_new_IDs}
        #oldNewIdsResolved   = list(map(lambda x: list(x.values()),[l_RBub_old_new_IDs,l_RBub_r_old_new_IDs, l_DBub_old_new_IDs, l_DBub_r_old_new_IDs,   l_MBub_old_new_IDs]))
        #oldNewIdsResolved   = sum(sum(oldNewIdsResolved,[]),[])
        #oldNewIdsResolved   = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs, **l_DBub_r_old_new_IDs,   **l_MBub_old_new_IDs}
        oldNewIdsResolvedDTotal = sum([a for a in oldNewIdsResolvedD.values()],[])
        # == take a look if resolved bubbles are sharing contours. there some rare cases where flat bubbles dont want to merge but permutations start to include them ==
        copyIDs,numCopies      =  np.unique(oldNewIdsResolvedDTotal, return_counts=True)
        if len(copyIDs)>0 and max(numCopies)>1:  # if some contour has more than one copy- it is shared between some bubbles
            whereShared     = np.argwhere(numCopies>1).flatten()         # where in copyIDs are there IDs with 2+ copies
            sharedContours  = copyIDs[whereShared]                       # which ID have 2+ copies
            sharr           = {ID:[subID for subID,vals in oldNewIdsResolvedD.items() if ID in vals] for ID in sharedContours}         # shared contour: [bubID1,bubID2,..]
            sharr_g         = {}

            resrDB  = list([a[1] for a in rDBOldNewDist])
            resDB   = list([a[1] for a in DBOldNewDist])
            resrRB  = list([a[1] for a in rRBOldNewDist])
            resRB   = list([a[1] for a in RBOldNewDist])
            resMB   = list([a[1] for a in MBOldNewDist])
            
            for ID, vals in sharr.items():
                sharr_g[ID] = []
                for subID in vals:
                    #if subID in {**l_RBub_r_old_new_IDs,**l_DBub_r_old_new_IDs}:                            # unfortunately rRB/rDB  are recovered using global id
                    #    sharr_g[ID].append(subID)                                                           # whick is old method thats not needed anymore
                    if subID in resolvedNewDBs:
                        if subID in resrDB:
                            where = resrDB.index(subID)
                            oldID = list([a[0] for a in rDBOldNewDist])[where]
                        else:
                            where = resDB.index(subID)
                            oldID = list([a[0] for a in DBOldNewDist])[where]
                            
                    elif subID in resolvedNewRBs:
                        if subID in resrRB:
                            where = resrRB.index(subID)
                            oldID = list([a[0] for a in rRBOldNewDist])[where]
                        else:
                            where = resRB.index(subID)
                            oldID = list([a[0] for a in RBOldNewDist])[where]
                    else:
                        where = resMB.index(subID)
                        oldID = list([a[0] for a in MBOldNewDist])[where]
                    sharr_g[ID].append(oldID)
                    
                    #if subID in list([a for a,_  in rRBOldNewDist]): 

                    #    sharr_g[ID].append(rRBOldNewDist[:,0][np.argwhere(rRBOldNewDist[:,1]==subID)[0]][0])
                    #elif subID in list([a for a,_  in rDBOldNewDist]):                                                      # so here a spaghetti code to retrieve
                    #    sharr_g[ID].append(rDBOldNewDist[:,0][np.argwhere(rDBOldNewDist[:,1]==subID)[0]][0])
                    #elif subID in list([a for a,_  in DBOldNewDist]):                                                      # so here a spaghetti code to retrieve
                    #    sharr_g[ID].append(oldNewDB[:,0][np.argwhere(oldNewDB[:,1]==subID)[0]][0])          # {shade contour: [bubGID1,bubGID2,..],..}
                    #else:                                                                                   # 
                    #    sharr_g[ID].append(RBOldNewDist[:,0][np.argwhere(RBOldNewDist[:,1]==subID)[0]][0])  # 
            a = 1
            combs       = sum([[[str(cID),gID] for gID in gIDs] for cID,gIDs in sharr_g.items()],[])
            cc_unique   = graphUniqueComponents(list(map(str,sharedContours)), combs)                       # string is shared contour ID. ints are global bubble IDs
            assert len(cc_unique)==1, f'basic contour share merge, three bubbles are merged: {cc_unique}'
            if len(cc_unique)==1:
                sharrID = list(sharr_g.keys())[0]
                cc_oldIDs.append(sharr_g[sharrID])                                           # idk.. should work.
                cc_newIDs.append(np.unique(sum([oldNewIdsResolved[ID] for ID in sharr[sharrID]],[])).tolist())  #using local ids and old-new IDs combine into cluster
                for ID in sharr[sharrID]:

                    #if ID in rRBOldNewDist[:,1]:
                    #    rRBOldNewDist    = np.array([[a,b] for a,b in rRBOldNewDist if b != ID])
                    #elif ID in rDBOldNewDist[:,1]:
                    #    rDBOldNewDist    = np.array([[a,b] for a,b in rDBOldNewDist if b != ID])
                    #elif ID in RBOldNewDist[:,1]:
                    #    RBOldNewDist    = np.array([[a,b] for a,b in RBOldNewDist if b != ID])
                    #else:
                    #    oldNewDB        = np.array([[a,b] for a,b in oldNewDB if b != ID])
                    if ID in resrRB:
                        #rRBOldNewDist   = np.array([[a,b] for a,b in rRBOldNewDist if b != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        rRBOldNewDist   = np.array([a for a in rRBOldNewDist    if a[1] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                    elif ID in resrDB:
                        #rDBOldNewDist   = np.array([[a,b] for a,b in rDBOldNewDist if b != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        rDBOldNewDist   = np.array([a for a in rDBOldNewDist    if a[1] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                    elif ID in resRB:
                        #RBOldNewDist    = np.array([[a,b] for a,b in RBOldNewDist if b != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        DBOldNewDist    = np.array([a for a in DBOldNewDist     if a[1] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                    else:
                        #DBOldNewDist    = np.array([[a,b] for a,b in DBOldNewDist if b != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        DBOldNewDist    = np.array([a for a in DBOldNewDist     if a[1] != ID], dtype=[('integer', '<i4'), ('string', '<U60')])
                        
                    #RBOldNewDist    = RBOldNewDist  if len(RBOldNewDist) > 0    else  np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
                    #rRBOldNewDist   = rRBOldNewDist if len(rRBOldNewDist) > 0   else  np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
                    #DBOldNewDist    = DBOldNewDist  if len(DBOldNewDist) > 0    else  np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
                    #rDBOldNewDist   = rDBOldNewDist if len(rDBOldNewDist) > 0   else  np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
                    #MBOldNewDist    = MBOldNewDist  if len(MBOldNewDist) > 0    else  np.empty((0, 2), dtype=[('integer', '<i4'), ('string', '<U60')])
            #sharedIDs       = [ID for ID,vals in oldNewIdsResolved.items() if any([True for i in vals if i in sharedContours])]
           # RBOldNewDist
        # ------------------- detect merges by inspecting shared contours ----------------------
        # OG idea was based on fact that if two or more bubbles from previous frame are not recovered and there are new unresolved bubbles, merge might have happened.
        # this case is tested at first part. now its modified
    if globalCounter >= 1 :#and len(unresolvedOldDB)>1 and len(jointNeighbors)>1
        rectParamsOld       = {oldID:l_rect_parms_old[oldID] for oldID in unresolvedOldDB + unresolvedOldRB }
        # ======== PART 01: test if two old bubbles are unresolved and new is unresolved. might hint on merge =========
        #rectParamsNew   = {subID:l_rect_parms_all[subID] for subID in sum(list(jointNeighbors.values()),[])}    # all unresolved cluster elements IDs and bound rect.
        unresolvedNewID69 = [ID for ID in sum(list(jointNeighbors.values()),[]) if ID not in resolvedNewAllSubIDs]  # lets try restricting merge to unresolved contours
        rectParamsNew   = {subID:l_rect_parms_all[subID] for subID in unresolvedNewID69} 
        combosOldNew    = np.array(overlappingRotatedRectangles(rectParamsOld,rectParamsNew),int)               # form [[old1,new1],[old1,new2],[old2,new1]]
        #values          = list(set(combosOldNew[:,1]))
        #counts          = [combosOldNew[:,1].count(a) for a in values]
        if len(combosOldNew)==0: hasShared = False
        else:
            _, counts        = np.unique(combosOldNew[:,1], return_counts=True)                             # get [new1, new2, new1] -> [new1,new2], [2,1]
            hasShared       = any(np.where(counts>1,1,0))                                                   # True if any new has more than one old, implies shared new contour between two olds.
        if hasShared == True:                                                                               # if at least one merge do this
            H = nx.Graph()
            H.add_nodes_from([str(a) for a in rectParamsOld.keys()])
            H.add_edges_from([[str(a),b] for a,b in combosOldNew])                                          # str to extract old IDs later
            cnctd_comp = [set(nx.node_connected_component(H, str(key))) for key in rectParamsOld]           # connected componetets. mix of old and new
            cc_unique = [];[cc_unique.append(x) for x in cnctd_comp if x not in cc_unique]
            cc_unique = [list(A) for A in cc_unique]
            cc_newIDs += [[elem for elem in vec if type(elem) != str] for vec in cc_unique]                  # extract new IDs by ID type
            cc_oldIDs += [[int(elem) for elem in vec if type(elem) == str] for vec in cc_unique]             # same with old


            #print(f'unresolvedNewRB {unresolvedNewRB}, unresolvedNewDB {unresolvedNewDB}')
            #print(f'unresolvedOldRB {unresolvedOldRB}, unresolvedOldDB {unresolvedOldDB}')
            print(f'new local IDs: {cc_newIDs} overlap with old global IDs: {cc_oldIDs}')

        # ======== PART 02: Big bubble merged with small. big is resolved due leeway in criterium ========
        # small old is missing, no new are missing. Target resolved old which has a history of 2+ steps. 
        # resolved centroid will be offset from predicted in derection of missing bubble.
        # if you were to consider total center of mass, of correct combination, it would be closer to resolved centroid.
        #centroidsResolved   = {**l_RBub_centroids,  **l_RBub_r_centroids,   **l_DBub_centroids,   **l_DBub_r_centroids,     **l_MBub_centroids}
        #areasResolved       = {**l_RBub_areas_hull, **l_RBub_r_areas_hull,  **l_DBub_areas_hull,  **l_DBub_r_areas_hull,    **l_MBub_areas_hull}
        oldNewIdsResolved   = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs, **l_DBub_r_old_new_IDs,   **l_MBub_old_new_IDs}

        rectParamsResolved  = {**l_RBub_r_rect_parms,   **l_DBub_r_rect_parms,  **l_RBub_rect_parms,    **l_DBub_rect_parms,    **l_MBub_rect_parms} # first time MB might have str key
        #centroidsResolved   = {**l_RBub_centroids,      **l_RBub_r_centroids,   **l_DBub_centroids,     **l_DBub_r_centroids,   **l_FBub_centroids,       **l_MBub_centroids}
        #areasResolved       = {**l_RBub_areas_hull,     **l_RBub_r_areas_hull,  **l_DBub_areas_hull,    **l_DBub_r_areas_hull,  **l_FBub_areas_hull,      **l_MBub_areas_hull}
        centroidsResolved   = {**l_RBub_centroids,      **l_RBub_r_centroids,   **l_DBub_centroids,     **l_DBub_r_centroids,   **l_MBub_centroids}
        areasResolved       = {**l_RBub_areas_hull,     **l_RBub_r_areas_hull,  **l_DBub_areas_hull,    **l_DBub_r_areas_hull,  **l_MBub_areas_hull}
        rectParamsOld       = {oldID:l_rect_parms_old[oldID] for oldID in unresolvedOldDB + unresolvedOldRB }
        resolvedGlobals      = resolvedOldDBs + resolvedOldRBs #list(oldNewDB[:,0]) + list(RBOldNewDist[:,0])   # holds global IDs of resolved
        resolvedLocals      = resolvedNewDBs + resolvedNewRBs#list(oldNewDB[:,1]) + list(RBOldNewDist[:,1])   # holds local IDs of resolved
        combosOldNew        = overlappingRotatedRectangles(rectParamsOld,{ID:a for ID,a in rectParamsResolved.items() if ID in resolvedLocals}) # old missing (glob) intersects new resolved (loc).
        preCalculated       = {} # concave hull, centroidTest, 
        if len(combosOldNew) > 0:
            #cc_unique           = graphUniqueComponents(list(map(str,unresolvedOldDB)), [[str(a),b] for a,b in combosOldNew])  # clusters: old unresolved intersect resolved old. e.g [[old lobal, old local]]: [['2', 21]]
            #cc_unique           = [a for a in cc_unique if len(a)>1]                                                        # sometimes theres no intersection, cluster of 1 element. drop it.
            #cc_unique           = [[a for a in b if type(a) == str]+[a for a in b if type(a) != str] for b in cc_unique]    # follow convension [str(ID1), ID2]
            #whereOldGlobals     = [np.argwhere(resolvedLocals == b)[0][0] for [_,b] in cc_unique]                           
            cc_unique           = graphUniqueComponents(unresolvedOldDB, combosOldNew)  # clusters: old unresolved intersect resolved old. e.g [[old lobal, old local]]: [['2', 21]]
            cc_unique           = [a for a in cc_unique if len(a)>1]                                                        # sometimes theres no intersection, cluster of 1 element. drop it.
            cc_unique           = [[a for a in b if type(a) != str]+[a for a in b if type(a) == str] for b in cc_unique]    # follow convension [str(ID1), ID2]
            for comb in cc_unique:
                if len(comb)>2: continue
                whereOldGlobals     = resolvedLocals.index(comb[1])   
                oldResolvedGlobals  = resolvedGlobals[whereOldGlobals]                                             # global IDs of resolved old bubbles.               e.g [3]
                oldPreMergeGlobals  = [comb[0],oldResolvedGlobals]                     # global IDs of possible merged bubbles.            e.g [[2, 3]]
                oldResolvedLocals   = resolvedLocals[whereOldGlobals]                                            # local ID of old resolved bubble (on new frame)    e.g [21]
                oldAreas            = np.array([l_areas_hull_old[ID] for ID in oldPreMergeGlobals], int)  # hull areas             of old globals         e.g [array([ 3683, 15179])]
                oldCentroids        = np.array([l_predict_displacement[ID][0] for ID in oldPreMergeGlobals], int)  # predicted centroids    of old globals         e.g [array([[1069,  476], [ 969,  462]])]
                centroidTest        = np.average(oldCentroids, weights = oldAreas,axis = 0).astype(int)  # weighted predicted centroid.                  e.g [array([988, 464])]
                a = 1                                                                                                                # if merge happened, buble will be about here.
                realCentroids       = centroidsResolved[oldResolvedLocals]                                    # actual centroid of new found bubble (local ID)    e.g {21: (986, 464)}
                predictedCentroids  = l_predict_displacement[oldResolvedGlobals][0]                           # expected centroid of that bubble                  e.g {3: (969, 462)}
                previousInfo        = getMergeCritParams(l_ellipse_parms_old, oldPreMergeGlobals, 0.4, 25)    # params to constuct convex-concave hull based on prev bubble orientation.
                mergeCritStuff      = mergeCrit(oldNewIdsResolved[oldResolvedLocals], l_contours, previousInfo, alphaParam = 0.06, debug = 0)
                areasTest           = int(cv2.contourArea(mergeCritStuff[0]))                             # potentially better variant of hull area.          e.g [18373]
                predictedAreas      = l_areas_hull_old[oldResolvedGlobals]                                      # expected old OG bubble area.                      e.g [15179]
                realArea            = areasResolved[oldResolvedLocals]                                         # actual new bubble area                            e.g [20791]
                areaPass            = False                                                           # containers to track state of failure
                distPass            = False                                                           # practical (e.g) example shows that Test centroid and Areas are closer to real vals.
                                 # compare predicted OG vals to (new and recovered (considering merge happened))
                relArea = lambda area: np.abs(realArea-area)/realArea
                if relArea(areasTest) < relArea(predictedAreas): areaPass = True                                             # if relative area change w.r.t new are is smaller for merged than solo->...
            
                dist = lambda centroid: np.linalg.norm(np.array(realCentroids) - np.array(centroid))   
                if dist(centroidTest) <= dist(predictedCentroids): distPass = True                                                  # compare distance from new to (predicted old and recustructed assuming merge)
            
                
                if areaPass == True and distPass == True:
                    allResDBGlobs  = [a[0] for a in DBOldNewDist]
                    allResDBGlobs  = [a[0] for a in DBOldNewDist]
                    DBOldNewDist    = np.array([a for a in DBOldNewDist     if a[0] != oldResolvedGlobals], dtype=[('integer', '<i4'), ('string', '<U60')])
                    rDBOldNewDist   = np.array([a for a in rDBOldNewDist    if a[0] != oldResolvedGlobals], dtype=[('integer', '<i4'), ('string', '<U60')])
                    RBOldNewDist    = np.array([a for a in RBOldNewDist     if a[0] != oldResolvedGlobals], dtype=[('integer', '<i4'), ('string', '<U60')])
                    rRBOldNewDist   = np.array([a for a in rRBOldNewDist    if a[0] != oldResolvedGlobals], dtype=[('integer', '<i4'), ('string', '<U60')])
                    MBOldNewDist    = np.array([a for a in MBOldNewDist     if a[0] != oldResolvedGlobals], dtype=[('integer', '<i4'), ('string', '<U60')])
                    #oldNewDB  = np.array([[a,b] for a,b in oldNewDB  if a != oldResolvedGlobals[i]])            # drop resolved status from bubble.
                    #RBOldNewDist    = np.array([[a,b] for a,b in RBOldNewDist    if a != oldResolvedGlobals[i]])            # easier to re-construct array than looking if ID is in, then where to delete.
                    
                    cc_oldIDs.append(oldPreMergeGlobals)
                    newIDs = oldNewIdsResolved[oldResolvedLocals]
                    cc_newIDs.append(newIDs)
                    preCalculated[ID2S(newIDs)] = [oldAreas, centroidTest, mergeCritStuff[0], mergeCritStuff[1]]
                    #preCalculated[min(newIDs)] = [oldAreas[i], centroidTest[i], mergeCritStuff[i][0], None]                 # new system puts None on first time merge
                    print(f'Merge detected between old:{oldPreMergeGlobals} and new:{newIDs}')
            #whereOldGlobals     = [resolvedLocals.index(b) for [_,b] in cc_unique]    
            #oldResolvedGlobals  = [resolvedGlobals[i] for i in whereOldGlobals]                                             # global IDs of resolved old bubbles.               e.g [3]
            #oldPreMergeGlobals  = [[a,oldResolvedGlobals[i]] for i,[a,_] in enumerate(cc_unique)]                      # global IDs of possible merged bubbles.            e.g [[2, 3]]
            #oldResolvedLocals   = [resolvedLocals[i] for i in whereOldGlobals]                                              # local ID of old resolved bubble (on new frame)    e.g [21]
            #oldAreas            = [np.array([l_areas_hull_old[ID] for ID in IDs], int)          for IDs in oldPreMergeGlobals]  # hull areas             of old globals         e.g [array([ 3683, 15179])]
            #oldCentroids        = [np.array([l_predict_displacement[ID][0] for ID in IDs], int) for IDs in oldPreMergeGlobals]  # predicted centroids    of old globals         e.g [array([[1069,  476], [ 969,  462]])]
            #centroidTest        = [np.average(a, weights = b,axis = 0).astype(int)      for a,b in zip(oldCentroids,oldAreas)]  # weighted predicted centroid.                  e.g [array([988, 464])]
            #a = 1                                                                                                                # if merge happened, buble will be about here.
            #realCentroids       = {ID:centroidsResolved[ID] for ID in oldResolvedLocals}                                    # actual centroid of new found bubble (local ID)    e.g {21: (986, 464)}
            #predictedCentroids  = {ID:l_predict_displacement[ID][0] for ID in oldResolvedGlobals}                           # expected centroid of that bubble                  e.g {3: (969, 462)}
            #previousInfo        = [getMergeCritParams(l_ellipse_parms_old, old, 0.4, 25) for old in oldPreMergeGlobals]     # params to constuct convex-concave hull based on prev bubble orientation.
            #mergeCritStuff      = [mergeCrit(oldNewIdsResolved[lID], l_contours, pInfo, alphaParam = 0.06, debug = 0) for lID,pInfo in zip(oldResolvedLocals,previousInfo)]
            #areasTest           = [int(cv2.contourArea(hull)) for hull,_,_ in mergeCritStuff]                               # potentially better variant of hull area.          e.g [18373]
            #predictedAreas      = [l_areas_hull_old[ID] for ID in oldResolvedGlobals]                                       # expected old OG bubble area.                      e.g [15179]
            #realArea            = [areasResolved[ID] for ID in oldResolvedLocals]                                           # actual new bubble area                            e.g [20791]
            #areaPass            = [False]*len(oldResolvedGlobals)                                                           # containers to track state of failure
            #distPass            = [False]*len(oldResolvedGlobals)                                                           # practical (e.g) example shows that Test centroid and Areas are closer to real vals.
            #for i,[realValue, myVariant, oldVariant] in enumerate(zip(realArea,areasTest,predictedAreas)):                  # compare predicted OG vals to (new and recovered (considering merge happened))
            #    relArea = lambda area: np.abs(realValue-area)/realValue
            #    if relArea(myVariant) < relArea(oldVariant): areaPass[i] = True                                             # if relative area change w.r.t new are is smaller for merged than solo->...
            
            #for i,[realValue, myVariant, oldVariant] in enumerate(zip(list(realCentroids.values()),centroidTest,list(predictedCentroids.values()))):
            #    dist = lambda centroid: np.linalg.norm(np.array(realValue) - np.array(centroid))   
            #    if dist(myVariant) <= dist(oldVariant): distPass[i] = True                                                  # compare distance from new to (predicted old and recustructed assuming merge)
            
            #for i, [passA,passD] in enumerate(zip(areaPass,distPass)):
            #    if passA == True and passD == True:
            #        allResDBGlobs  = [a[0] for a in DBOldNewDist]
            #        allResDBGlobs  = [a[0] for a in DBOldNewDist]
            #        DBOldNewDist    = np.array([a for a in DBOldNewDist     if a[0] != oldResolvedGlobals[i]], dtype=[('integer', '<i4'), ('string', '<U60')])
            #        rDBOldNewDist   = np.array([a for a in rDBOldNewDist    if a[0] != oldResolvedGlobals[i]], dtype=[('integer', '<i4'), ('string', '<U60')])
            #        RBOldNewDist    = np.array([a for a in RBOldNewDist     if a[0] != oldResolvedGlobals[i]], dtype=[('integer', '<i4'), ('string', '<U60')])
            #        rRBOldNewDist   = np.array([a for a in rRBOldNewDist    if a[0] != oldResolvedGlobals[i]], dtype=[('integer', '<i4'), ('string', '<U60')])
            #        MBOldNewDist    = np.array([a for a in MBOldNewDist     if a[0] != oldResolvedGlobals[i]], dtype=[('integer', '<i4'), ('string', '<U60')])
            #        #oldNewDB  = np.array([[a,b] for a,b in oldNewDB  if a != oldResolvedGlobals[i]])            # drop resolved status from bubble.
            #        #RBOldNewDist    = np.array([[a,b] for a,b in RBOldNewDist    if a != oldResolvedGlobals[i]])            # easier to re-construct array than looking if ID is in, then where to delete.
                    
            #        cc_oldIDs.append(oldPreMergeGlobals[i])
            #        newIDs = oldNewIdsResolved[oldResolvedLocals[i]]
            #        cc_newIDs.append(newIDs)
            #        preCalculated[ID2S(newIDs)] = [oldAreas[i], centroidTest[i], mergeCritStuff[i][0], mergeCritStuff[i][1]]
            #        #preCalculated[min(newIDs)] = [oldAreas[i], centroidTest[i], mergeCritStuff[i][0], None]                 # new system puts None on first time merge
            #        print(f'Merge detected between old:{oldPreMergeGlobals[i]} and new:{newIDs}')
        for old,new in zip(cc_oldIDs,cc_newIDs):
            
            # grab new that is known (not stray) EG. is in unresolved new. EDIT 17/03/23 its not clear what next line does. i think error is in unresolvedNewDB containing min(subIDs), which is not used correctly here
            #new = [elem for elem in new if elem in unresolvedNewRB + unresolvedNewDB]
            #print(f'new: {new}')
            #if len(new)<len(old) and len(new)>0: # merge
            if len(old) > 1:                                # 17/03/23 changed condition from top. idk what that ment. now two or more oldIDs is enough evidence for merge.
                # if smallest bubble is merged into big one- keep big one, drop small. rethink case with 3 bubbles.
                # if smaller bubble has at least 20% of big bubble's area, end them both and create new object. set old ones to pre-merge type
                # =========== CASE for 2 bubbles merging =================
                areaList        = np.array([l_areas_hull_old[ID] for ID in old])
                areaLargestID   = np.argmax(areaList)
                areaRatio       = areaList/areaList[areaLargestID]
                areaThreshold   = 0.3
                areaTest        = np.where(areaRatio<areaThreshold,1,0)
                hasSmallBsMerge = any(areaTest)
                inletCase       = False
                if hasSmallBsMerge == True: # ==== its not yet tested !!! ==== 31/03/2023  kind of works for 1+1 merge
                    typeTemp        = typeRing if any(item in unresolvedNewRB for item in new) else typeElse      # maybe drop or change ring+ big area.
                    print(f'-Merge from Old IDs: {old} to new IDs: {new}. Big+Small type merge. Larger bubble of type {typeStrFromTypeID[typeTemp]}')
                    
                    selectID        = old[areaLargestID]    # largest area inherits ID
                    tempID          = ID2S(new)
                    app = np.array((selectID, tempID), dtype=[('integer', '<i4'), ('string', '<U60')])
                    if typeTemp == typeRing:
                        dataSets    = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                        #rRBOldNewDist    = np.append(rRBOldNewDist,[[selectID,min(new)]],axis=0)
                        rRBOldNewDist    = np.append(rRBOldNewDist,app)
                    else: 
                        dataSets    = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                        #rDBOldNewDist    = np.append(rDBOldNewDist,[[selectID,min(new)]],axis=0)
                        rDBOldNewDist    = np.append(rDBOldNewDist,app)
                    
                    if tempID in preCalculated:
                        [areas, predictCentroid, hull, newParams] = preCalculated[tempID]
                    else:
                        previousInfo        = getMergeCritParams(l_ellipse_parms_old, old, 0.4, 25)
                        inletCase = all([oldID in inletIDs for oldID in old])
                        if inletCase == True:                                             # INLET stuff here !!!!! if two inlet zone bubbles merge.
                            hull        = cv2.convexHull(np.vstack(l_contours[new]))
                        else:
                            hull, newParams, _  = mergeCrit(new, l_contours, previousInfo, alphaParam = 0.05, debug = 0, debug2 = 0, gc = globalCounter, id = selectID)
                        areas               = [l_areas_hull_old[IDs] for IDs in old]
                        predictCentroid     = np.average([l_predict_displacement[ID][0] for ID in old], weights = areas, axis = 0).astype(int)  #l_centroids_old[IDs]
                        
                    
                    tempStore2(new, l_contours, tempID, err, orig, dataSets, concave = hull)
                    centroidReal                            = dataSets[4][tempID]                               # centroid from current hull (convex or concave)
                    dist2                                   = np.linalg.norm(np.array(predictCentroid) - np.array(centroidReal))  
                    prevCentroid                            = np.average([l_centroids_old[ID] for ID in old], weights = areas, axis = 0).astype(int)
                    if inletCase == False:
                        l_MBub_info[tempID]              = [old,np.sum(areas).astype(int),prevCentroid] + list(newParams) # [(645, 557), 8.6]
                    g_Centroids[selectID][globalCounter-1]  = prevCentroid                                          # move pre-merge centroid to common center of mass. should be easer to calculate dist predictor.
                    l_predict_displacement[selectID]        = [tuple(map(int,predictCentroid)), int(dist2)]         # predictCentroid not changed idk why. but dist is
                    
                    if globalCounter not in g_merges: g_merges[globalCounter] = {}                                  # drop global storage in form {timestep:{newGlobID1:[oldPremerge1,oldPremerge2],newGlobID2:[],..}
                    g_merges[globalCounter][selectID] = []
                    for ID in [elem for elem in old if elem != selectID]:                                           # looks like its for a case with 2+ buble merge    
                        g_merges[globalCounter][selectID].append(ID)
                        g_bubble_type[ID][globalCounter-1]  = typePreMerge
                        g_Centroids[ID][globalCounter]      = dataSets[4][tempID]                                 # dropped IDs inherit merged bubble centroid, so visuall they merge.
                    
                # ===== IF there are about same size merge bubbles, create new ID, hull is concave
                else:
                    selectID =  ID2S(new)                                          # str to differentiate first time merged IDs
                    
                    previousInfo = getMergeCritParams(l_ellipse_parms_old, old, 0.4, 25)

                    hull, newParams, _  = mergeCrit(new, l_contours, previousInfo, alphaParam = 0.05, debug = 0, debug2 = 0, gc = globalCounter, id = selectID)
                    

                        
                    dataSets  = [l_MBub_masks, l_MBub_images,l_MBub_old_new_IDs, l_MBub_rect_parms, l_MBub_centroids, l_MBub_areas_hull, l_MBub_contours_hull]
                    tempStore2(new, l_contours, selectID, err, orig, dataSets, concave = hull)
                    
                    areas = [l_areas_hull_old[IDs] for IDs in old]
                    prevCentroid = np.average([l_centroids_old[IDs] for IDs in old], weights = areas, axis = 0).astype(int)

                    temp = []
                    temp.append(old)
                    temp.append(np.sum(areas).astype(int))
                    temp.append(prevCentroid)
                    temp.append(newParams[0])   # (refDistance, avgDirection, refAngleThreshold, merge point)
                    temp.append(newParams[1])
                    temp.append(newParams[2])
                    temp.append(newParams[3])
                    temp.append(newParams[4])
                    app = np.array((-1, selectID), dtype=[('integer', '<i4'), ('string', '<U60')])
                    MBOldNewDist    = np.append(MBOldNewDist,app)
                    # not doing g_merge here since newID is not yet known, do it in local-global storage flush
                    #MBOldNewDist    = np.append(MBOldNewDist,[[-1,selectID]],axis=0)         # multiple oldIDs, but we dont need them. we just need 
                                                                                             # to know this is new global ID. tag it impossible ID= -1
                    l_MBub_info[selectID] = temp

                    for ID in old:
                        g_bubble_type[ID][globalCounter-1]  = typePreMerge               # since all olds get RIPped, change their type as a terminator
                        g_Centroids[ID][globalCounter]      = dataSets[4][selectID]
                    
                        
                    
                #print(f'old IDs: {old}, main_old {selectID}, new IDs: {new}, type: {typeTemp}')
                unresolvedNewRB = [elem for elem in unresolvedNewRB if int(elem) not in new]
                unresolvedNewDB = [elem for elem in unresolvedNewDB if elem != ID2S(new)]          # dont like this new stuff. what if its  apart of cluster.
                print(f'unresolvedNewRB, (updated) unrecovered RBs: {unresolvedNewRB}')
                print(f'unresolvedNewDB, (updated) unrecovered RBs: {unresolvedNewDB}')
                [unresolvedOldDB.remove(x) for x in old if x in unresolvedOldDB]
                [unresolvedOldRB.remove(x) for x in old if x in unresolvedOldRB]
        #RBOldNewDist    = RBOldNewDist  if len(RBOldNewDist) > 0    else np.empty((0,2), np.uint16)
        #rRBOldNewDist   = rRBOldNewDist if len(rRBOldNewDist) > 0   else np.empty((0,2), np.uint16)
        #oldNewDB        = oldNewDB      if len(oldNewDB) > 0        else np.empty((0,2), np.uint16)
        #rDBOldNewDist   = rDBOldNewDist if len(rDBOldNewDist) > 0   else np.empty((0,2), np.uint16)

        # =========================================================================================================
        # ===================================== SIMPLE SPLIT DETECTION ============================================
        # =========================================================================================================
        # === gather all relevant resolved clusters ===
        #resLoc              = np.concatenate((oldNewDB[:,1],RBOldNewDist[:,1],rRBOldNewDist[:,1],rDBOldNewDist[:,1]))
        #resGlob             = np.concatenate((oldNewDB[:,0],RBOldNewDist[:,0],rRBOldNewDist[:,0],rDBOldNewDist[:,0]))
        resolvedNewRBs = list([b for _,b in RBOldNewDist]) + list([b for _,b in rRBOldNewDist])     # dropped merged since its not needed for split
        resolvedNewDBs = list([b for _,b in DBOldNewDist]) + list([b for _,b in rDBOldNewDist])
        resolvedNewAll = resolvedNewDBs + resolvedNewRBs 
        resolvedOldDBs = list([a for a,_ in DBOldNewDist]) + list([a for a,_ in rDBOldNewDist])
        resolvedOldRBs = list([a for a,_ in RBOldNewDist]) + list([a for a,_ in rRBOldNewDist])
        resolvedOldAll = resolvedOldDBs + resolvedOldRBs

        resGlob     = resolvedOldAll
        resLoc      = resolvedNewAll
        #resGlob      = resolvedOldDBs + resolvedOldRBs #list(oldNewDB[:,0]) + list(RBOldNewDist[:,0])   # holds global IDs of resolved
        #resLoc      = resolvedNewDBs + resolvedNewRBs
        oldNewIdsResolved2  = {**l_RBub_old_new_IDs,    **l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs,   **l_DBub_r_old_new_IDs} # drop l_MBub_old_new_IDs. no point splitting merge back!!!  drop l_FBub_old_new_IDs
        rectParamsResolved2 = {**l_RBub_r_rect_parms,   **l_DBub_r_rect_parms,  **l_RBub_rect_parms,    **l_DBub_rect_parms}
        # ======================================== Regular 2 element contours ================================================
        resolvedPair        = {ID:subIDs for ID,subIDs in oldNewIdsResolved2.items() if ID in resLoc and len(subIDs) == 2}
        overBox             = overlappingRotatedRectangles(fakeBox,{ID:params for ID,params in rectParamsResolved2.items() if ID in resolvedPair})
        dropBoxIDs          = [b for _,b in overBox]
        # ============================================== inlet clusters ======================================================
        resolvedInletIDs    = overlappingRotatedRectangles(rectParamsResolved2, fakeBox, returnType = 1, typeAreaThreshold = 0.15)[1]
        inletTestSubIDs     = {ID:subIDs for ID,subIDs in oldNewIdsResolved2.items() if ID in resolvedInletIDs and len(subIDs) >= 2 and ID in resLoc}
        inletRectP          = {ID:{subID:cv2.boundingRect(l_contours[subID]) for subID in subIDs} for ID,subIDs in inletTestSubIDs.items()}#[x-2,yFb,w+2,hFB]
        inletRectPModed     = {ID:{subID:[x-5,yFb,w+5,hFB] for subID,[x,_,w,_] in subDict.items()} for ID,subDict in inletRectP.items()}
        resolvedPairInlet   = {}
        for ID,subDict in inletRectPModed.items():
            inletSelfInter  = overlappingRotatedRectangles(subDict,subDict) 
            cc_unique       = graphUniqueComponents(list(subDict.keys()), inletSelfInter)
            if len(cc_unique)==2:
                resolvedPairInlet[ID] = cc_unique
        # first times when cluster splits into two, it registers potential split into "l_splits[gID]"
        # it can continue to stay split but it is being related back to OG ID.

        resolvedPair        = {ID:vals for ID,vals in resolvedPair.items() if ID not in dropBoxIDs}
        # === gather all relevant un-resolved clusters ===
        unresolvedPair      = {ID:subIDs for ID,subIDs in jointNeighbors.items() if len(subIDs) == 2 if ID in unresolvedNewDB}  # unresolved clusters with 2 contours
        unresolvedNewRP     = {ID:cv2.boundingRect(np.vstack(l_contours[jointNeighbors[ID]])) for ID in unresolvedPair}
        missingSplits       = [ID for ID in unresolvedOldDB + unresolvedOldRB if ID in l_splits_old]                            # unresolved old IDs that were registred splitting
        missingSplitsOldRP  = {ID:params for ID,params in l_rect_parms_old.items() if ID in missingSplits}                      # rectangle parametrs of unres Old Split for ovelap
        #missingSplitsNewRP  = {ID:subIDs for ID,subIDs in rectParamsResolved2.items() if ID in unresolvedPair}                 # rectangle parametrs of unres New clust for ovelap  
        #missingSplitOVP     = overlappingRotatedRectangles(missingSplitsOldRP, missingSplitsNewRP)                               
        rectParamsResolved2 = {**rectParamsResolved2,**unresolvedNewRP}
        missingSplitOVP     = overlappingRotatedRectangles(missingSplitsOldRP, unresolvedNewRP) 
        oldUnresID          = -1                                                                                # default to empty list. it will be searched
        if len(missingSplitOVP) == 1:
            oldUnresID      = missingSplitOVP[0][0]
            newUnresID      = missingSplitOVP[0][1]
            resolvedPair    = {**resolvedPair,**{newUnresID:jointNeighbors[newUnresID]}}                        # something related to missing bubs
            resLoc = resLoc + [newUnresID]                                                                      # give it unique ID = -1
            resGlob = resGlob + [oldUnresID]
            #resLoc          = np.concatenate((resLoc,[newUnresID]))
            #resGlob         = np.concatenate((resGlob,[oldUnresID]))
        resolvedPair = {**resolvedPair,**resolvedPairInlet}
        for ID,subIDs in resolvedPair.items(): 
            if type(subIDs[0]) != list: subIDs = [[a] for a in subIDs]
            ID1,ID2 = subIDs                      
            #dist = int(closes_point_contours(l_contours[ID1],l_contours[ID1])[1])
            x0,y0,w0,h0 = rectParamsResolved2[ID]
            blanks      = {min(subID):np.zeros((h0,w0),np.uint8)                                for subID in subIDs}
            splitHulls  = {min(subID):cv2.convexHull(np.vstack(l_contours[subID]))              for subID in subIDs}
            [cv2.drawContours( blanks[min(subID)],   [splitHulls[min(subID)]-[x0,y0]], -1, 255, -1)  for subID in subIDs] 

            inter = cv2.bitwise_and(*list(blanks.values()))
            interArea = int(np.sum(inter/255))
        
            if interArea >0: continue
            gWhere                  = resLoc.index(ID)
            gID                     = resGlob[gWhere]
            predictCentroid         = np.array(l_predict_displacement[gID][0],int)
            splitCentroidsAreas     = {ID:getCentroidPosContours(bodyCntrs = [hull]) for ID,hull in splitHulls.items()}         
            splitCentroidDistances  = {ID:np.linalg.norm(predictCentroid - CA[0]).astype(int) for ID,CA in splitCentroidsAreas.items()}
            # in regular case take centoid closest to predicted. in case of inlet, take one on top. (min centroid x)        
            if ID not in resolvedPairInlet:  splitSmallestDistID     = min(splitCentroidDistances, key=splitCentroidDistances.get)           # local solution ID
            else:                               splitSmallestDistID  = min(splitCentroidsAreas, key=lambda x:  splitCentroidsAreas[x][0][0])
            ccDist = closes_point_contours(l_contours[ID1][0],l_contours[ID2][0])[1]
            l_splits[gID]           = [False, splitCentroidsAreas, [ID1, ID2],ccDist]
            if gID in l_splits_old:
                nLast = 2
                subG = {t:list(a.keys()) for t,a in g_splits.items() if t >= globalCounter-nLast }
                subGgID = [t for t,v in subG.items() if gID in v]
                #numIDSplits = sum(list(subG.values()),[]).count(gID)                # count how many times ID was in splits last nLast steps (except this). nLast + 1 in total
                numIDSplits = len(subGgID)
                distHist = np.array([g_splits[t][gID][3] for t in subGgID] + [ccDist]) # last n step history of contour-contour min distances + this step.
                displ = np.diff(distHist-distHist[0])                               # displ rel to first time: if distance monotonically grows, it will be positive
                if all(np.where(displ>=5,1,0)):                  # check if each step objects get further apart
                    steadyDispl = True
                else: steadyDispl = False
                if all(np.where(distHist>=15,1,0)):
                    bigDispl = True
                else: bigDispl = False
                if (numIDSplits == nLast or gID == oldUnresID) and (steadyDispl or bigDispl):           # it was in split whole time! OR past split went missing due to crit failure!
                    if ID in unresolvedNewDB: unresolvedNewDB.remove(ID)                    # drop cluster from unesolved. recluster it next
                    # ====== drop split ID as unresolved ======
                    jointNeighbors.pop(ID,None)  # i think i need to remove it straight away
                    dropIDs = [IDs for IDs in subIDs if min(IDs) != splitSmallestDistID][0]
                    if gID == oldUnresID: jointNeighbors.pop(ID,None)                       # ???
                    jointNeighbors = {**jointNeighbors,**{ID2S(dropIDs):dropIDs}}           # add to unresolved, i dont think its used anymore

                    if len(dropIDs) == 1 and dropIDs[0] in whereChildrenAreaFiltered:                   # if only sol is a ring
                        unresolvedNewRB.append(str(dropIDs[0]))
                        dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                        if ID in resolvedNewRBs: resolvedNewRBs.remove(ID) 
                    else:                                                                               # if not solo RB, it is a cluster.
                        unresolvedNewDB.append(ID2S(dropIDs))                                           # add as unresolved cluster
                        [unresolvedNewRB.remove(str(x)) for x in dropIDs if str(x) in unresolvedNewRB]  # remove possible RBs since they are inside DB cluster now
                        dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    tempStore2(dropIDs, l_contours, ID2S(dropIDs), err, orig, dataSets, concave = 0)

                       
                    # ====== modify OG contour data =======
                
                    dist2                       = splitCentroidDistances[splitSmallestDistID]
                    newArea                     = splitCentroidsAreas[splitSmallestDistID][1]
                    areaCrit                    = np.abs(newArea-oldMeanArea)/ oldAreaStd
                    l_predict_displacement[gID] = [tuple(map(int,predictCentroid)), int(dist2)]

                    restIDs = [IDs for IDs in subIDs if min(IDs) == splitSmallestDistID][0]

                    jointNeighbors = {**jointNeighbors,**{ID2S(restIDs):restIDs}} 

                    if len(restIDs)==1 and splitSmallestDistID in whereChildrenAreaFiltered: singleRB = True
                    else: singleRB = False

                    [unresolvedNewRB.remove(str(x)) for x in restIDs if str(x) in unresolvedNewRB]
                    # if it was a part of resolved, depending if it turned in RB, it will modify according data storage.
                    # if it was not resolved it will check if it was RB->make new entry or make new DB entry    
                    if gID in [a[0] for a in DBOldNewDist]:                                               
                            
                        if singleRB == False:
                            where = [a[0] for a in DBOldNewDist].index(gID)
                            DBOldNewDist[where] = np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                            dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                        else:
                            DBOldNewDist    = np.array([a for a in DBOldNewDist     if a[0] != gID], dtype=[('integer', '<i4'), ('string', '<U60')])
                            RBOldNewDist    = np.append(RBOldNewDist,np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')]))
                            dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                        
                    elif gID in [a[0] for a in rRBOldNewDist]:
                        if singleRB == False:
                            where = [a[0] for a in rRBOldNewDist].index(gID)                       
                            rRBOldNewDist[where] = np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                            dataSets    = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                        else:
                            rRBOldNewDist    = np.array([a for a in rRBOldNewDist     if a[0] != gID], dtype=[('integer', '<i4'), ('string', '<U60')])
                            RBOldNewDist    = np.append(RBOldNewDist,np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')]))
                            dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                    elif gID in [a[0] for a in rDBOldNewDist]:
                        if singleRB == False:
                            where = [a[0] for a in rDBOldNewDist].index(gID)                       
                            rDBOldNewDist[where] = np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                            dataSets    = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                        else:
                            rDBOldNewDist    = np.array([a for a in rDBOldNewDist     if a[0] != gID], dtype=[('integer', '<i4'), ('string', '<U60')])
                            RBOldNewDist    = np.append(RBOldNewDist,np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')]))
                            dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                    else:
                        if singleRB == True:
                            RBOldNewDist    = np.append(RBOldNewDist,np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')]))
                            dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                        else:
                            DBOldNewDist    = np.append(DBOldNewDist,np.array((gID, ID2S(restIDs)), dtype=[('integer', '<i4'), ('string', '<U60')]))
                            dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    
                   
                    
                
                    [unresolvedOldRB.remove(ID) for ID in [gID] if ID in unresolvedOldRB]        # delete if prev was unresolved.
                    [unresolvedOldDB.remove(ID) for ID in [gID] if ID in unresolvedOldDB]
                    #if ID in resolvedNewDBs: resolvedNewDBs.remove(ID) 
                    #[unresolvedNewDB.remove(ID) for ID in [ID] if ID in unresolvedNewDB]

                    tempStore2(restIDs, l_contours, ID2S(restIDs), err, orig, dataSets, concave = 0)
                    l_splits[gID]           = [True, splitCentroidsAreas, [ID1, ID2], ccDist]
                    
                    
    a = 1
    # ==============================================================================================
    # ==================================== Final cleanup ===========================================
    # ==============================================================================================
    if globalCounter >= 1:
        print('--Performing global cleanup: resolving possible missed relations')
        unresolvedOldAll = unresolvedOldDB + unresolvedOldRB
        unresolvedNewAll = unresolvedNewDB + unresolvedNewRB
        if len(unresolvedNewAll)>0 and len(unresolvedOldAll)>0:                                     # there are missing olds and unresolved new
            print(f'--Performing global cleanup: there are unresolved oldIDs: {unresolvedOldAll} and unresolved new clusters: {unresolvedNewAll}')
            rectParamsUnresOld          = {ID:l_rect_parms_old[ID] for ID in unresolvedOldAll}
            missingNewSubIDs            = {ID: subNewIDs for ID, subNewIDs in jointNeighbors.items() if ID in unresolvedNewAll}  
            missingNewSubIDs_hulls      = {ID: cv2.convexHull(np.vstack(l_contours[subNewIDs])) for ID, subNewIDs in missingNewSubIDs.items()}
            missingNewSubIDs_bound_rect = {ID: cv2.boundingRect(hull) for ID, hull in missingNewSubIDs_hulls.items()}                         
            missingNewSubIDs_c_a        = {ID: getCentroidPosContours(bodyCntrs = [hull]) for ID, hull in missingNewSubIDs_hulls.items()}

            oldNewConnectivity          = overlappingRotatedRectangles( rectParamsUnresOld,                   
                                                                        missingNewSubIDs_bound_rect)      
            cc_unique                   = graphUniqueComponents(unresolvedOldAll, oldNewConnectivity)
            cc_unique                   = [comb for comb in cc_unique if len(comb)>1]
            olds = [[i for i in comb if type(i) != str] for comb in cc_unique]
            news = [[i for i in comb if type(i) == str] for comb in cc_unique]
            for oldIDs,newIDs in zip(olds,news):
                subIDs              = sum([jointNeighbors[newID] for newID in newIDs],[])
                predictCentroid     = np.mean([l_predict_displacement[oldID][0] for oldID in oldIDs],axis = 0).astype(int)
                centroidReal        = np.mean([missingNewSubIDs_c_a[newID][0]   for newID in newIDs],axis = 0).astype(int)
                dist2               = np.linalg.norm(np.array(predictCentroid) - np.array(centroidReal)).astype(int)
                if fixedFrame == 1:
                    dists = {ID:np.linalg.norm(np.array(predictCentroid) - np.array(missingNewSubIDs_c_a[ID][0])) for ID in newIDs}
                    minID = min(dists, key=dists.get)
                    discardIDs = [ID for ID in newIDs if ID != minID]
                    newIDs = [minID]
                    subIDs = sum([jointNeighbors[newID] for newID in newIDs],[])
                    dist2 = dists[minID]
                if dist2 < 100:
                    [unresolvedOldDB.remove(ID) for ID in oldIDs if ID in unresolvedOldDB]
                    [unresolvedOldRB.remove(ID) for ID in oldIDs if ID in unresolvedOldRB]
                    [unresolvedNewDB.remove(ID) for ID in newIDs if ID in unresolvedNewDB]
                    [unresolvedNewRB.remove(ID) for ID in newIDs if ID in unresolvedNewRB]
                    relevantIDs = {ID:l_areas_hull_old[ID] for ID in oldIDs}
                    oldID = max(relevantIDs, key=lambda x: relevantIDs[x])
                    #if oldID in unresolvedOldDB: unresolvedOldDB.remove(oldID)
                    #if oldID in unresolvedOldRB: unresolvedOldRB.remove(oldID)
                    #if newID in unresolvedNewDB: unresolvedNewDB.remove(newID)
                    #if newID in unresolvedNewRB: unresolvedNewRB.remove(newID)
                    app = np.array((oldID, ID2S(subIDs)), dtype=[('integer', '<i4'), ('string', '<U60')])
                    DBOldNewDist    = np.append(DBOldNewDist,app)
                    dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    hull  = 0
                    if len(oldIDs) == 1 and oldIDs[0] in STIDs:                               # recovering See Though bubble with concave hull.
                        if STcounts[STIDs.tolist().index(oldID)] >= 2 and len(newIDs) == 1:
                            hull = 1
                    tempStore2(subIDs, l_contours, ID2S(subIDs), err, orig, dataSets, concave = hull)
                    l_predict_displacement[oldID][1] = int(dist2)
                    print(f'-{oldID}:Resolved (Cleanup) oldID: {oldID} (of {oldIDs}) => {newIDs}. prediction centroid - actual centroid distnace: {int(dist2)}')
                        
        else: print(f'--Performing global cleanup: no candidates ...\n')

     
    if globalCounter > 0 : 
        for newID in unresolvedNewRB:       # looks like its needed- sets are empty before.
            dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
            tempStore2([int(newID)], l_contours, newID, err, orig, dataSets, concave = 0)
        for newID in unresolvedNewDB:       # added dropped frozen cluster element from N last steps
            dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
            tempStore2(jointNeighbors[newID], l_contours, newID, err, orig, dataSets, concave = 0)
        for newID in declusterNewFRLocals:       # added dropped frozen cluster element from N last steps
            dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
            tempStore2([newID], l_contours, str(newID), err, orig, dataSets, concave = 0)
        [unresolvedNewDB.append(str(ID)) for ID in declusterNewFRLocals]
    print(f'l_MBub_info_old: {l_MBub_info_old}\nl_MBub_info: {l_MBub_info}')  
    # ================================= Save first iteration ======================================  
    if globalCounter == 0 :
        startID = 0
        for tempID, newKey in enumerate(l_RBub_masks): # by keys
            l_RBub_masks_old[tempID]            = l_RBub_masks[newKey]
            l_RBub_images_old[tempID]           = l_RBub_images[newKey]
            l_RBub_rect_parms_old[tempID]       = l_RBub_rect_parms[newKey]
            l_RBub_centroids_old[tempID]        = l_RBub_centroids[newKey]
            l_RBub_old_new_IDs_old[tempID]      = l_RBub_old_new_IDs[newKey]
            l_bubble_type[tempID]               = typeRing
            g_bubble_type[tempID]               = {}
            g_bubble_type[tempID][0]            = typeRing
            l_RBub_areas_hull_old[tempID]       = l_RBub_areas_hull[newKey]
            l_contours_hull[tempID]             = l_RBub_contours_hull[newKey]
            
        
        startID = len(l_RBub_masks)
        for globalID,localID in zip(range(startID,startID+len(l_DBub_centroids),1),l_DBub_centroids):
            l_bubble_type[globalID]             = typeElse   #============ wtf here?? ================
            g_bubble_type[globalID]             = {}
            g_bubble_type[globalID][0]          = typeElse
            
                
            l_DBub_masks_old[globalID]          = l_DBub_masks[localID]
            l_DBub_images_old[globalID]         = l_DBub_images[localID]
            l_DBub_rect_parms_old[globalID]     = l_DBub_rect_parms[localID]
            l_DBub_centroids_old[globalID]      = l_DBub_centroids[localID]
            l_DBub_old_new_IDs_old[globalID]    = l_DBub_old_new_IDs[localID]
            l_DBub_areas_hull_old[globalID]     = l_DBub_areas_hull[localID]
            l_contours_hull[globalID]           = l_DBub_contours_hull[localID]
        #l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old = {},{},{},{},{}            
        #l_FBub_old_new_IDs_old = {}
        #print('g_bubble_type',g_bubble_type)
        #l_bubble_type_old = {ID:val[0] for ID,val in g_bubble_type.items()}
        #print('l_bubble_type',l_bubble_type)
          
    # ================================= Save other iterations ====================================== 
    if globalCounter >= 1:
        frozensLastStep = list(l_FBub_masks_old.keys())
        l_RBub_masks_old, l_RBub_images_old, l_RBub_rect_parms_old, l_RBub_centroids_old, l_RBub_areas_hull_old, l_RBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
        l_DBub_masks_old, l_DBub_images_old, l_DBub_rect_parms_old, l_DBub_centroids_old, l_DBub_areas_hull_old, l_DBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
        l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old, l_FBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
        l_MBub_masks_old, l_MBub_images_old, l_MBub_rect_parms_old, l_MBub_centroids_old, l_MBub_areas_hull_old, l_MBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
        l_MBub_info_old = {}
        # print('l_DBub_centroids',l_DBub_centroids)
        
        # ============== frozen tricks ============
        #oldLocIDs2 = frozenIDsInfo[:,0] if len(frozenIDsInfo)>0 else np.array([]) # 10/02/23
        #oldGlobIDs = {}
        #for oldLocID in oldLocIDs2:
        #    if type(oldLocID) == str: oldGlobIDs[oldLocID] = int(oldLocID)
        #    else: oldGlobIDs[oldLocID] = max(g_bubble_type) + 1
        #oldGlobIDs02 = [
        #                    {ii:ID for ID, vals in l_old_new_IDs_old.items() if ii in vals} if type(ii) != str else {ii:int(ii)} 
        #                    for ii in oldLocIDs2 ] # relevant new:old dict   
        #oldGlobIDs = {};[oldGlobIDs.update(elem) for elem in oldGlobIDs02] # basically flatten [{x:a},{y:b}] into {x:a,y:b} or something  10/02/23
        #l_FBub_old_new_IDs_old = {oldGlobIDs[localOldID]:localNewIDs for localOldID, localNewIDs,_,_,_ in frozenIDsInfo}
        #for key in l_FBub_masks: # contains relation indices that satisfy distance
        #    gKey = oldGlobIDs[key]
        #    l_FBub_masks_old[gKey]                  = l_FBub_masks[key]
        #    l_FBub_images_old[gKey]                 = l_FBub_images[key]
        #    l_FBub_rect_parms_old[gKey]             = l_FBub_rect_parms[key]
        #    l_FBub_centroids_old[gKey]              = l_FBub_centroids[key] 
        #    l_FBub_areas_hull_old[gKey]             = l_FBub_areas_hull[key]
        #    #l_bubble_type[gKey]                     = typeFrozen if str(gKey) not in lastNStepsFrozenCentroids else typeRecoveredFrozen
        #    l_bubble_type[gKey]                     = typeFrozen
        #    l_contours_hull[gKey]                   = l_FBub_contours_hull[key]
        #    if globalCounter not in frozenGlobal:
        #        frozenGlobal[globalCounter]             = []
        #        g_FBub_rect_parms[globalCounter]        = {}
        #        g_FBub_centroids[globalCounter]         = {}
        #        g_FBub_areas_hull[globalCounter]        = {}
               
        #    if gKey not in g_bubble_type: g_bubble_type[gKey] = {}
            
        #    #g_bubble_type[gKey][globalCounter]      = typeFrozen if str(gKey) not in lastNStepsFrozenCentroids else typeRecoveredFrozen
        #    g_bubble_type[gKey][globalCounter]      = typeFrozen 
        #    frozenGlobal[globalCounter].append(gKey)
        #    g_FBub_rect_parms[globalCounter][gKey]  = l_FBub_rect_parms[key]
        #    g_FBub_centroids[globalCounter][gKey]   = l_FBub_centroids[key]
        #    g_FBub_areas_hull[globalCounter][gKey]  = l_FBub_areas_hull[key]
        for [old,new] in FBOldNewDist: # contains relation indices that satisfy distance
            bubType = typeFrozen if old in frozensLastStep else typeRecoveredFrozen
            l_bubble_type[old]                  = typeFrozen 
            g_bubble_type[old][globalCounter]   = typeFrozen 
            l_FBub_masks_old[old]               = l_FBub_masks[new]
            l_FBub_images_old[old]              = l_FBub_images[new]
            l_FBub_rect_parms_old[old]          = l_FBub_rect_parms[new]
            l_FBub_centroids_old[old]           = l_FBub_centroids[new]
            l_FBub_old_new_IDs_old[old]         = l_FBub_old_new_IDs[new]
            l_FBub_areas_hull_old[old]          = l_FBub_areas_hull[new]
            l_contours_hull[old]                = l_FBub_contours_hull[new]

        for [old,new] in DBOldNewDist: # contains relation indices that satisfy distance
            l_bubble_type[old]                  = typeElse 
            g_bubble_type[old][globalCounter]   = typeElse 
            l_DBub_masks_old[old]               = l_DBub_masks[new]
            l_DBub_images_old[old]              = l_DBub_images[new]
            l_DBub_rect_parms_old[old]          = l_DBub_rect_parms[new]
            l_DBub_centroids_old[old]           = l_DBub_centroids[new]
            l_DBub_old_new_IDs_old[old]         = l_DBub_old_new_IDs[new]
            l_DBub_areas_hull_old[old]          = l_DBub_areas_hull[new]
            l_contours_hull[old]                = l_DBub_contours_hull[new]
                
        for [old,new] in rDBOldNewDist: # holds global keys
            if new in l_MBub_info:
                l_MBub_info_old[old]            = l_MBub_info[new]
            l_bubble_type[old]                  = typeRecoveredElse
            g_bubble_type[old][globalCounter]   = typeRecoveredElse
            l_DBub_masks_old[old]               = l_DBub_r_masks[new]
            l_DBub_images_old[old]              = l_DBub_r_images[new]
            l_DBub_rect_parms_old[old]          = l_DBub_r_rect_parms[new]
            l_DBub_centroids_old[old]           = l_DBub_r_centroids[new] 
            l_DBub_old_new_IDs_old[old]         = l_DBub_r_old_new_IDs[new]
            l_DBub_areas_hull_old[old]          = l_DBub_r_areas_hull[new]
            l_contours_hull[old]                = l_DBub_contours_hull[new]
                
            
        for [cPi,cCi] in RBOldNewDist: # contains relation indices that satisfy dinstance
            l_bubble_type[cPi]                  = typeRing
            g_bubble_type[cPi][globalCounter]   = typeRing

            l_RBub_masks_old[cPi]               = l_RBub_masks[cCi]
            l_RBub_images_old[cPi]              = l_RBub_images[cCi]
            l_RBub_rect_parms_old[cPi]          = l_RBub_rect_parms[cCi]
            l_RBub_centroids_old[cPi]           = l_RBub_centroids[cCi]
            l_RBub_old_new_IDs_old[cPi]         = l_RBub_old_new_IDs[cCi]
            l_RBub_areas_hull_old[cPi]          = l_RBub_areas_hull[cCi]
            #l_RBub_contours_hull[cPi]             = l_contours_hull[cCi]
            l_contours_hull[cPi]                = l_RBub_contours_hull[cCi]
                
        for [old,new] in rRBOldNewDist: # holds global keys
            if new in l_MBub_info:        # key - glob
                l_MBub_info_old[old]            = l_MBub_info[new]
            g_bubble_type[old][globalCounter]   = typeRecoveredRing
            l_bubble_type[old]                  = typeRecoveredRing
            l_RBub_masks_old[old]               = l_RBub_r_masks[new]
            l_RBub_images_old[old]              = l_RBub_r_images[new]
            l_RBub_rect_parms_old[old]          = l_RBub_r_rect_parms[new]
            l_RBub_centroids_old[old]           = l_RBub_r_centroids[new]
            l_RBub_old_new_IDs_old[old]         = l_RBub_r_old_new_IDs[new]
            l_RBub_areas_hull_old[old]          = l_RBub_r_areas_hull[new]
            l_contours_hull[old]                = l_RBub_contours_hull[new]
                            
        if globalCounter == -1:
            [cv2.imshow(f'c {globalCounter}, i: {i}', img) for i, img in list(l_RBub_masks_old.items())]
        # numRingsPrev, numRingsNow = len(prevIterCentroidsRB_RSTRD),len(tempBubbleTypeRings)
            
            
        # ================STORE REMAINING NEW DISCORVERED BUBS=====================
        # ------------ NEW UNRESOLVED RINGS ADDED TO STORAGE WITH NEW ID --------------
            
        # if len(unresolvedNewRB)>0: # if there is new stray bubble, create new storage index
        startID = max(g_bubble_type) + 1          # changed from g_centroid to g_bubble_type, cause includes frozen 19/02/23
        for gID, localID in zip(range(startID,startID+len(unresolvedNewRB),1), unresolvedNewRB):
            l_RBub_masks_old[gID]               = l_RBub_masks[localID]
            l_RBub_images_old[gID]              = l_RBub_images[localID]
            l_RBub_rect_parms_old[gID]          = l_RBub_rect_parms[localID]
            l_RBub_centroids_old[gID]           = l_RBub_centroids[localID]
            l_RBub_old_new_IDs_old[gID]         = l_RBub_old_new_IDs[localID]
            l_RBub_areas_hull_old[gID]          = l_RBub_areas_hull[localID]
            l_contours_hull[gID]                = l_RBub_contours_hull[localID]
            l_bubble_type[gID]                  = typeRing
            g_bubble_type[gID]                  = {}
            g_bubble_type[gID][globalCounter]   = typeRing
            
                    
        # ------------ NEW UNRESOLVED DIST ADDED TO STORAGE WITH NEW ID --------------
        # if len(unresolvedNewDB)>0: # if there is new stray bubble, create new storage index
        startID += len(unresolvedNewRB)
        for gID,localID in zip(range(startID,startID+len(unresolvedNewDB),1), unresolvedNewDB):#enumerate(unresolvedNewDB): 
            l_bubble_type[gID]                  = typeElse
            g_bubble_type[gID]                  = {}
            #g_bubble_type[gID][globalCounter]   = typeElse if localID not in frozenIDs else typeFrozen
            g_bubble_type[gID][globalCounter]   = typeElse 
            
            l_DBub_masks_old[gID]               = l_DBub_masks[localID]
            l_DBub_images_old[gID]              = l_DBub_images[localID]
            l_DBub_rect_parms_old[gID]          = l_DBub_rect_parms[localID]
            l_DBub_centroids_old[gID]           = l_DBub_centroids[localID]
            l_DBub_old_new_IDs_old[gID]         = l_DBub_old_new_IDs[localID]
            l_DBub_areas_hull_old[gID]          = l_DBub_areas_hull[localID]
            l_contours_hull[gID]                = l_DBub_contours_hull[localID]
        # ------------ NEW UNRESOLVED MERGE ADDED TO STORAGE WITH NEW ID --------------
        #newMbubs = [ID for ID in l_MBub_old_new_IDs if type(ID) == str]
        startID += len(unresolvedNewDB)
        #for gID,localID in zip(range(startID,startID+len(l_MBub_old_new_IDs),1), l_MBub_old_new_IDs):#enumerate(unresolvedNewDB):
        for [old,new] in MBOldNewDist:
            if old == -1:
                gID                 = startID
                g_bubble_type[gID]  = {}
                g_MBub_info[gID]    = {}
            else: gID = old
            if len(l_MBub_info[new][0]) > 0 :                                                               # after first discovery merged old are empty. no need to store.
                if globalCounter not in g_merges: g_merges[globalCounter] = {}                                  # drop global storage in form {timestep:{newGlobID1:[oldPremerge1,oldPremerge2],newGlobID2:[],..}
                g_merges[globalCounter][gID] = l_MBub_info[new][0]
            l_bubble_type[gID]                  = typeMerge
            
            g_bubble_type[gID][globalCounter]   = typeMerge
            g_MBub_info[gID][globalCounter]     = l_MBub_info[new]
            l_MBub_info_old[gID]                = l_MBub_info[new]
            l_MBub_masks_old[gID]               = l_MBub_masks[new]
            l_MBub_images_old[gID]              = l_MBub_images[new]
            l_MBub_rect_parms_old[gID]          = l_MBub_rect_parms[new]
            l_MBub_centroids_old[gID]           = l_MBub_centroids[new]
            l_MBub_old_new_IDs_old[gID]         = l_MBub_old_new_IDs[new]
            l_MBub_areas_hull_old[gID]          = l_MBub_areas_hull[new]
            l_contours_hull[gID]                = l_MBub_contours_hull[new]
        #for localID in l_MBub_old_new_IDs:
        #    if type(localID) == str:
        #        #startID             += 1
        #        gID                 = startID
        #        g_bubble_type[gID]  = {}
        #        g_MBub_info[gID]    = {}
            
        #    else: gID = localID

        #    l_bubble_type[gID]                  = typeMerge
            
        #    g_bubble_type[gID][globalCounter]   = typeMerge
        #    g_MBub_info[gID][globalCounter]     = l_MBub_info[int(localID)]
        #    l_MBub_info_old[gID]                = l_MBub_info[int(localID)]
        #    l_MBub_masks_old[gID]               = l_MBub_masks[localID]
        #    l_MBub_images_old[gID]              = l_MBub_images[localID]
        #    l_MBub_rect_parms_old[gID]          = l_MBub_rect_parms[localID]
        #    l_MBub_centroids_old[gID]           = l_MBub_centroids[localID]
        #    l_MBub_old_new_IDs_old[gID]         = l_MBub_old_new_IDs[localID]
        #    l_MBub_areas_hull_old[gID]          = l_MBub_areas_hull[localID]
        #    l_contours_hull[gID]                = l_MBub_contours_hull[localID]

        #if globalCounter == 6:
        #    with open('concaveContour.pickle', 'wb') as handle:
        #                 pickle.dump(l_MBub_contours_hull[localID], handle)
    
    
    # ========================== DUMP all data in global storage ==============================
    # collect all time step info (except bubbleIDsTypes, use new g_bubble_type )
    l_masks_old         = {**l_RBub_masks_old,        **l_DBub_masks_old,         **l_FBub_masks_old,           **l_MBub_masks_old}
    l_images_old        = {**l_RBub_images_old,       **l_DBub_images_old,        **l_FBub_images_old,          **l_MBub_images_old}
    l_rect_parms_old    = {**l_RBub_rect_parms_old,   **l_DBub_rect_parms_old,    **l_FBub_rect_parms_old,      **l_MBub_rect_parms_old}
    l_centroids_old     = {**l_RBub_centroids_old,    **l_DBub_centroids_old,     **l_FBub_centroids_old,       **l_MBub_centroids_old}
    l_old_new_IDs_old   = {**l_RBub_old_new_IDs_old,  **l_DBub_old_new_IDs_old,   **l_FBub_old_new_IDs_old,     **l_MBub_old_new_IDs_old}
    l_areas_hull_old    = {**l_RBub_areas_hull_old,   **l_DBub_areas_hull_old,    **l_FBub_areas_hull_old,      **l_MBub_areas_hull_old} # l_DBub_areas_hull_old has extra IDs
    l_contours_hull_old = l_contours_hull
    # === fitting ellipse to hulls ====
    # --- if points are not evenly spaced, fit minimization algorithm tries to satisfy more dense regions ---
    # --- so we interpolate points and resample. this makes a better fit ---
    debugg = 0 #if globalCounter != 5 else 1
    # --- Merged Bubbles MB have a concave hull. for them we need to redo regula convex hull ---
    if len(l_MBub_areas_hull_old) == 0:
        l_ellipse_parms_old = {ID:cv2.fitEllipse(interpolateHull(hull, k = 2, s = 0.2, numReturnPoints = 20, debug = debugg)) for ID, hull in l_contours_hull_old.items()}
    else:
        #aa = cv2.convexHull(l_contours_hull[list(l_MBub_areas_hull_old.keys())[0]])
        l_ellipse_parms_old_EDB = {ID:cv2.fitEllipse(interpolateHull(hull, k = 2, s = 0.2, numReturnPoints = 20, debug = debugg)) for ID, hull in l_contours_hull.items() if ID not in l_MBub_areas_hull_old}
        l_ellipse_parms_old_MB  = {ID:cv2.fitEllipse(interpolateHull(cv2.convexHull(hull), k = 2, s = 0.2, numReturnPoints = 20, debug = debugg)) for ID, hull in l_contours_hull.items() if ID in l_MBub_areas_hull_old}
        l_ellipse_parms_old = {**l_ellipse_parms_old_EDB, **l_ellipse_parms_old_MB}
    #if globalCounter == 5:
    #    with open('contour_hulls_gc5.pickle', 'wb') as handle:
    #        pickle.dump(l_contours_hull_old, handle)
    for key in list(l_masks_old.keys()):
        if key not in list(g_Masks.keys()):
            g_Masks[key]            = {}
            g_Images[key]           = {}
            g_Rect_parms[key]       = {}
            g_Ellipse_parms[key]    = {}
            g_Centroids[key]        = {}
            g_old_new_IDs[key]      = {}
            g_contours_hull[key]    = {}
            g_areas_hull[key]       = {}

        g_Masks[key][globalCounter]         = l_masks_old[key]
        g_Images[key][globalCounter]        = l_images_old[key]
        g_Rect_parms[key][globalCounter]    = l_rect_parms_old[key]
        g_Ellipse_parms[key][globalCounter] = l_ellipse_parms_old[key]
        g_Centroids[key][globalCounter]     = l_centroids_old[key]
        g_old_new_IDs[key][globalCounter]   = l_old_new_IDs_old[key]
        g_contours_hull[key][globalCounter] = l_contours_hull_old[key]
        g_areas_hull[key][globalCounter]    = l_areas_hull_old[key]

        if len(l_old_new_IDs_old[key]) == 1:
            grays = np.array(np.where((l_images_old[key] < 200) & (l_images_old[key] > 45),255,0),np.uint8)
            graysA = int(np.sum(grays)/255)
            graysM = int(np.sum(l_masks_old[key])/255)
            if graysA/graysM > 0.7:
                if globalCounter not in STBubs: STBubs[globalCounter] = []
                STBubs[globalCounter].append(key)
        # --------- take care of prediction storage: g_predict_area_hull & g_predict_displacement --------------
        # -- some extra treatment for merged bubbles before regular ones ---
        if  (key not in g_predict_area_hull and key in l_areas_hull_old and key in l_MBub_areas_hull_old):
            localID                         = ID2S(l_MBub_old_new_IDs_old[key])
            prevArea                        = l_MBub_info[localID][1]                                       # sum of pre-merged bubbles
            prevCentroid                    = l_MBub_info[localID][2]                                       # restored from pre-merged area weighted centroids.
            g_predict_area_hull[key]        = {globalCounter-1:[prevArea,  prevArea,  int(prevArea*0.2)]}   # initiate predictors frame earleir
        if globalCounter == 0 or (key not in g_predict_area_hull and key in l_areas_hull_old):              # first time merged bubbles fail this and go to else
            g_predict_area_hull[key]        = {globalCounter:[l_areas_hull_old[key],  l_areas_hull_old[key],  int(l_areas_hull_old[key]*0.2)]}
        else:
            updateValue                     = l_areas_hull_old[key]
            historyLen                      = len(g_predict_area_hull[key])
            #timeStep                        = globalCounter-1 if key not in lastNStepsFrozenLatestTimes else lastNStepsFrozenLatestTimes[key]
            timeStep                        = globalCounter-1
            if historyLen == 1:
                prevVals                        = [g_predict_area_hull[key][timeStep][0],updateValue]
                hullAreaMean                    = np.mean(prevVals)
                hullAreaStd                     = max(np.std(prevVals),hullAreaMean*0.15)                # std cant be very low, if it is, choose as 15% of mean instead
            else:
                #timeStep = globalCounter-1 if key not in frozenKeys else frozenKeys[key];
                numStepsStat                    = 2
                timeSteps                       = list(g_predict_area_hull[key].keys())
                numStepsAvailable               = min(numStepsStat, historyLen)
                vals                            = [g_predict_area_hull[key][time][0] for time in timeSteps[-numStepsAvailable:]] + [updateValue]
                hullAreaMean                    = np.mean(vals)
                hullAreaStd                     = max(np.std(vals),hullAreaMean*0.1)

                #prevMean                        = g_predict_area_hull[key][timeStep][1]
                #prevStd                         = g_predict_area_hull[key][timeStep][2]
                #hullAreaMean, hullAreaStd       = updateStat(historyLen, prevMean, prevStd, updateValue) # just fancy way of updating mean and sdtev w/o recalc whole path data. most likely does not impact anything

            g_predict_area_hull[key][globalCounter] = [updateValue, int(hullAreaMean), int(hullAreaStd)]
        # === subsection for bubbles WITH NO history ===

        # --- some extra treatment for merged bubs with no history ---
        # --- first merged bubble has more info, so it should be taken care of before other first bubbles ---
        if (key not in g_predict_displacement and key in l_MBub_centroids_old):
            # this new merged bubble ID. Because merged bubbles have a history of atleast 1 frame (pre-merge), so predictor can start 1 frame earlier.
            # last centroid can be estimated by area-weighted average centroid, second frame of merged bubble can already be estimated by linear extrapolation.
            offset                                      = np.array([7,0])
            prevCentroidPredict                         = prevCentroid-offset   # define at hull predictor
            distPredictReal                             = int(np.linalg.norm(prevCentroidPredict- l_MBub_centroids_old[key]))
            g_predict_displacement[key]                 = {}
            g_predict_displacement[key][globalCounter]  = [tuple(prevCentroidPredict), distPredictReal, distPredictReal, 4] 
            g_Centroids[key][globalCounter-1]           = tuple(prevCentroid)
            g_Centroids[key]                            = dict(sorted(g_Centroids[key].items()))               # added prev after current. should sort
            #l_predict_displacement[selectID]         = [tuple(prevCentroidPredict), distPredictReal]
        # --- now take care of regular first bubbles. give them some aribrary displacement predictor error that should be lowered with new data ---
        if key not in g_predict_displacement : 
            pCentroid                       = l_centroids_old[key]# g_Centroids[key][globalCounter] # 15/03/23 replace g_ to l_
            pDistByType                     = [14,30]
            updateValue                     = pDistByType[0] if key in l_RBub_masks_old else pDistByType[1]
            pc2CMean                        = updateValue
            pc2CStd                         = 0
            g_predict_displacement[key] = {globalCounter: [pCentroid, updateValue, np.around(pc2CMean,2), np.around(pc2CStd,2)]}
            
        # === subsection for bubbles WITH history ===
        elif key in l_predict_displacement:                                  # l_predict_displacement is filled during old-new frame comparison.
            pCentroid                       = l_predict_displacement[key][0] # current frame prediction based on prev history
            updateValue                     = l_predict_displacement[key][1] # difference between current frame prediction and resolved centroid
            historyLen                      = len(g_predict_displacement[key])
            #timeStep                        = globalCounter-1 if key not in lastNStepsFrozenLatestTimes else lastNStepsFrozenLatestTimes[key]
            timeStep                        = globalCounter-1
            if historyLen == 1:
                prevVals                        = [g_predict_displacement[key][globalCounter-1][1],updateValue]
                pc2CMean                        = np.mean(prevVals)
                pc2CStd                         = np.std(prevVals)
            else:
                numStepsStat                    = 2
                timeSteps                       = list(g_predict_displacement[key].keys())
                numStepsAvailable               = min(numStepsStat, historyLen)
                vals                            = [g_predict_displacement[key][time][2] for time in timeSteps[-numStepsAvailable:]] + [updateValue]
                pc2CMean                        = np.mean(vals)
                pc2CStd                         = np.std(vals)
                
                if key in l_MBub_info_old:                                  # have to account that history is gone.
                    pc2CStd += 2
                
                    #prevMean                        = g_predict_displacement[key][timeStep][2]
                #prevStd                         = g_predict_displacement[key][timeStep][3]
                #pc2CMean, pc2CStd               = updateStat(historyLen, prevMean, prevStd, updateValue)
                
            g_predict_displacement[key][globalCounter] = [pCentroid, int(updateValue), max(int(pc2CMean),5), max(int(pc2CStd),3)]# 15/03/23 [pCentroid, np.around(updateValue,2), np.around(pc2CMean,2), np.around(pc2CStd,2)]

        # --whereChildrenAreaFiltered is reliant on knowing parent. some mumbo-jumbo to work around
        # -  maybe its better to combine RBOldNewDist and oldNewDB and extract global-local IDS there
        parentsWithChildrenIDs = [ID for ID,children in whereChildrenAreaFiltered.items() if len(children) > 0]
        intersection = set(l_old_new_IDs_old[key]).intersection(set(parentsWithChildrenIDs))
        # print('intersection',intersection)
        if len(intersection)>0:
            if key not in list(g_child_contours.keys()): g_child_contours[key] = {}
            g_child_contours[key][globalCounter]    = \
                sum([whereChildrenAreaFiltered[A] for A in intersection],[] ) # pretty general. for ex in case one bubble is two ring bubbles (?!)
        
    # ============================ detect stuck bubbles =======================================    
    # grab all current time IDs, check their traj-s, calculate last nSteps average displacement
    # if its small, collect info. if its that bubble's last live timestep, start investigation by backtracking
           
  

    l_bubble_type_old                       = l_bubble_type
    l_Areas_old                             = l_areas_all
    g_areas_IDs[globalCounter]              = l_areas_all_IDs
    l_Areas_hull_old                        = l_areas_hull_all
    l_centroids_old_all                     = l_centroids_all
    l_rect_parms_all_old                    = l_rect_parms_all
    contoursFilter_RectParams_dropIDs_old   = contoursFilter_RectParams_dropIDs # prepare/store for next time step 
    g_dropIDs[globalCounter]                = contoursFilter_RectParams_dropIDs
    l_splits_old                            = l_splits
    if len(l_splits)>0:
        g_splits[globalCounter]             = l_splits
    #frozenBuffer_old[globalCounter]         = [globalCounter]
    #g_drop_keep_IDs[globalCounter]          = [contoursFilter_RectParams_dropIDs,contoursFilter_RectParams]
    #frozenIDs_old                           = frozenIDs

    g_bublle_type_by_gc_by_type[globalCounter] = {}
    for bType in typeStrFromTypeID.keys():
        g_bublle_type_by_gc_by_type[globalCounter][bType] = [ID for ID,bubT in l_bubble_type_old.items() if bubT == bType]

    #if len(frozenLocal)>0:
    #    if globalCounter not in frozenGlobal_LocalIDs:
    #        #frozenGlobal[globalCounter] = []
    #        frozenGlobal_LocalIDs[globalCounter] = []
    #    #frozenGlobal[globalCounter].append(frozenLocal)
    #    frozenGlobal_LocalIDs[globalCounter].append(frozenLocal)

    #for key in list(l_DBub_r_masks.keys()) + list(l_RBub_r_masks.keys()):
    #    if key in contoursFilter_RectParams_dropIDs:
    #        l_Areas_old[key]        = cv2.contourArea(l_contours[key])
    #        l_centroids_old_all[key]    = getCentroidPosContours(bodyCntrs = [l_contours[key]])[0]
    
    for key in [a for a,_ in rRBOldNewDist] + [a for a,_ in rDBOldNewDist]:
        if key in contoursFilter_RectParams_dropIDs:
            l_Areas_old[key]        = cv2.contourArea(l_contours[key])
            l_centroids_old_all[key]    = getCentroidPosContours(bodyCntrs = [l_contours[key]])[0]

    if exportImagesMHTX == 1  or (drawAfter == 0 and exportImages == 1):
        activeIDs = l_old_new_IDs_old
        blank  = convertGray2RGB(orig)
        if exportImagesMHTX == 1:
            if globalCounter in exportCSVimgsIDs:
                cv2.imwrite(os.path.join(csvImgDirectory,str(dataStart+globalCounter).zfill(4)+".png") ,blank)

    if drawAfter == 0 and exportImages == 1:
        activeIDs = l_old_new_IDs_old
        blank  = convertGray2RGB(orig)

        # ============================
        # ==  for MHT-X blank image ==
        # ============================
        if globalCounter in exportCSVimgsIDs:
            cv2.imwrite(os.path.join(csvImgDirectory,str(dataStart+globalCounter).zfill(4)+".png") ,blank)
        # ============================

        if len(fakeBox) > 0:
            x,y,w,h = fakeBox[-1]
            cv2.rectangle(blank, (x,y), (x+w,y+h),[120,15,55],1)
        typeStrings = ["N","RB", "rRB", "E", "F","rE","pm","rF","M"]
        splitStrings =  {}
        for ID in activeIDs:
            temp            = [0,len(g_old_new_IDs[ID][globalCounter])]
            currentType     = g_bubble_type[ID][globalCounter]#;print('currentType',currentType)
            currentTypeStr  = typeStrings[currentType] 
            currentCntrIds  = g_old_new_IDs[ID][globalCounter]
            # text            = currentTypeStr + '('+','.join(list(map(str,currentCntrIds)))+')'
            temp.append(currentTypeStr + '(')
            cCIstr = list(map(str,currentCntrIds))
            A = [x for y in zip(cCIstr, [","]*len(cCIstr)) for x in y][:-1] #dont ask me
            temp += A
            if min(g_old_new_IDs[ID]) == globalCounter:                 # if current time step is the first in IDs data - mark it "NEW"            
                temp += [f"):{ID} new"]
            else:                                                       # 
                prevTimeStep = globalCounter - 1 if (currentType != typeFrozen and currentType != typeRecoveredFrozen) else [ts for ts in g_bubble_type[ID].keys() if ts < globalCounter][-1]
                temp[0] = 1
                prevType = g_bubble_type[ID][prevTimeStep]
                prevTypeStr = typeStrings[prevType] 
                prevCntrIds  = g_old_new_IDs[ID][prevTimeStep]
                # text += "-" + prevTypeStr + f'({ID})'
                temp += [")"+str(":" + prevTypeStr + f'({ID})')]
            splitStrings[ID] = temp
        
            if ID in g_child_contours:
                if globalCounter in g_child_contours[ID]:
                    [cv2.drawContours(  blank,   g_contours[globalCounter], cid, (0,255,0), 1) for cid in g_child_contours[ID][globalCounter]]
            # aa = [cid for cid in g_child_contours[ID][globalCounter] if globalCounter in g_child_contours[ID]]
            colorSet = [(),(0,0,255),(0,100,190),(255,0,0),(125,0,125),(255,0,125),(0,255,0),(125,0,125),(100,255,100)]
            thcSet = [(), 2, 1, 2, 2, 2 ,2, 2,2]
            clr = colorSet[currentType]
            thc = thcSet[currentType]
            [cv2.drawContours(  blank,   g_contours[globalCounter], cid, clr, thc) for cid in currentCntrIds]
            cv2.ellipse(blank, g_Ellipse_parms[ID][globalCounter], (55,55,55), 1)

            cv2.drawContours(  blank,   [g_contours_hull[ID][globalCounter]], -1, (255,255,255), 1)

        fontScale = 0.7;thickness = 4;

        # ========= overlay organization =============
        # check drawing.py for standalone example. 
        # works the following way. bubbles fight for vertical space. width is modified to text width. bubbles that overlay with other on vertical space are grouped. 
        # that means that text will overlap. group is split into top and bottom groups- less overlay. top and bottom groups are split into vertical positions.
        thickRectangle = {}
        rectParams = {}
        strDimsAll = {}
    
        H,L,_           = blank.shape
        textHeightMax   = 0
        baselineMax     = 0
        for k,ID in enumerate(activeIDs):
            case, numIDs, *strings  = splitStrings[ID]
            strDims  = [cv2.getTextSize(text, font, fontScale, thickness) for text in strings]          # substrings are split to make a line pointing from subIDs to contour
            baseline = max([bl for _,bl in strDims])                                                    # i draw substring seperately. full string drawing is scaled differently
            ySize = max([sz[1] for sz,_ in strDims])
            xSize = [sz[0] for sz,_ in strDims]
            totSize = sum(xSize)
            #[totSize,ySize],baseline  = cv2.getTextSize(''.join(strings), font, fontScale, thickness) # i think text dimensions for partial sum and full string are different.

            textHeightMax           = max(textHeightMax,    ySize)                              # total text height is textHeightMax + baselineMax
            baselineMax             = max(baselineMax,      baseline)                           # baselineMax is height of the part of text under the baseline e.g 'y,p,g,q,..', baseline  = 0 for 'o,k,d,..'
            x,y,w,h                 = g_Rect_parms[ID][globalCounter]
            rectParams[ID]          = [x,y,w,h] 
            dw                      = totSize-w           # if text is thicker, then 
            xS, hS                  = [int(x-0.5*dw),H]
            dx1                     = max(0, -xS)                                               # if BRect is poking out of left side of image dx1 is positive, else zero
            dx2                     = max(0, xS + totSize - L)                                  # if BRect is poking out of right side of image dx2 is positive, else zero
            thickRectangle[ID]      = [xS + dx1 - dx2,0,totSize, hS]
            strDimsAll[ID]          = strDims
            #cv2.rectangle(blank, (xS + dx1 - dx2,y), (xS + dx1 - dx2+totSize,y+h),(255,0,0),2)
        # split into competing vertical columns    
        aa = overlappingRotatedRectangles(thickRectangle,thickRectangle)
        HG = nx.Graph()
        HG.add_nodes_from(rectParams.keys())
        HG.add_edges_from(aa)
        # 
        cntrd = {ID:tuple((x+int(0.5*w),y+int(0.5*h))) for ID,[x,y,w,h] in rectParams.items()}
        posTB = {ID:0 if y < (H - (y+h)) else 1 for ID,[x,y,w,h] in rectParams.items()}

        neighbors = {ID:[] for ID in rectParams}
        neighborsSameSide = {ID:[] for ID in rectParams}
        # dont remember what, but something interesting happens for splitting.
        # top-bottom splitting is not that simple. i think only neighbors are tests wheter they share same side.
        for IDs in rectParams:
            for i in HG.neighbors(IDs):
                neighbors[IDs].append(i)
                if posTB[i] == posTB[IDs]: neighborsSameSide[IDs].append(i) # write connections between neighbors on same side
        #clusters0 = [tuple(np.sort(np.array([ID] +nbrs))) for ID, nbrs in neighborsSameSide.items()]
        a = 1
        edgePairs = sum([[(ID,a) for a in nbrs] for ID,nbrs in neighborsSameSide.items() if len(nbrs) > 0], []) # rewrite connections into pairs, ignore solo connections
        sameSideClusters = graphUniqueComponents(list(rectParams.keys()),edgePairs)                             # get same side cluster, next step order them by centroid y coordinate
    
        sortedByY = [[x for _,x in sorted(zip([cntrd[a][1] for a in X],X))] for X in sameSideClusters] # https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list
        sortedIndexed = [{ID:i for i,ID in enumerate(ID)} for ID in sortedByY]                         # enumerate index is a vertical position index for text
        globSideOrder = {k: v for d in sortedIndexed for k, v in d.items()}                            # combine list of dictionaries into on dict.
        #SameSideClusters = [list(aa) for aa in set(clusters0)]
        #globSideOrder = {ID:0 for ID in rectParams}
        #for arr in SameSideClusters:
        #    srt = np.argsort([cntrd[ID][1] for ID in arr])                                            # old method. did not work, new method on top is simpler.
        #    for i,elem in enumerate(arr):
        #        globSideOrder[elem] = srt[i]
        
        textHeight  = textHeightMax
        textNoGo    = 10
        textSep     = 15
        # text position is calculted for top and bottom part of image separately. different coordinate systems- from edge. duh
        for ID,[x,y,w,h] in thickRectangle.items():
            if posTB[ID] == 1:
                textPos = H - textNoGo - baselineMax - globSideOrder[ID]*(textHeightMax + baselineMax + textSep)
            if posTB[ID] == 0:
                textPos = textNoGo + textHeightMax + globSideOrder[ID]*(textHeightMax + baselineMax + textSep)
        
        
            case, numIDs, *strings  = splitStrings[ID]
            strDims                 = strDimsAll[ID]

            startPos                = np.array((x,textPos),int) 
            ci = 0
            subIDsIndices = list(range(1,len(strings),2))
            # iterating through overlay substrings, on subIDs calculate nearest distance from ID string center to any point on countour with that ID
            for i, string in enumerate(strings):
                [cv2.putText(blank, string, startPos, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
                if i in subIDsIndices:
                
                    startPosMiddle = startPos + np.array((strDims[i][0][0]/2,0),int)        # startPos-> string start, startPosMiddle = string start +half width
                    cntrID = g_old_new_IDs[ID][globalCounter][ci]
                    _, _, p1, p2 = closes_point_contours(startPosMiddle.reshape(-1,1,2),g_contours[globalCounter][cntrID])
                    cv2.polylines(blank, [np.array([p1- np.array((0,(textHeightMax + baselineMax)/2),int),p2])],0, (0,0,255), 1)
                    startPos += np.array((strDims[i][0][0],0),int)
                    ci += 1
                else: startPos += np.array((strDims[i][0][0],0),int)                        # skip, add full width offset.
            
            #string = ''.join(strings)
            #[cv2.putText(blank, string, (x,textPos), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
           
        

        #cv2.imshow('aa',blank)
    
        #--------------- trajectories ---------------------------------
        for i,times in g_Centroids.items(): # key IDs and IDS-> times
            useTimes = [t for t in times.keys() if t <= globalCounter and t>globalCounter - 10]#
            pts = np.array([times[t] for t in useTimes]).reshape(-1, 1, 2)
            if pts.shape[0]>3:
                cv2.polylines(blank, [pts] ,0, (255,255,255), 3)
                cv2.polylines(blank, [pts] ,0, cyclicColor(i), 2)
                [cv2.circle(blank, tuple(p), 3, cyclicColor(i), -1) for [p] in pts]
            else:
                cv2.polylines(blank, [pts] ,0, cyclicColor(i), 1)
                [cv2.circle(blank, tuple(p), 1, cyclicColor(i), -1) for [p] in pts]
            
        cv2.putText(blank, str(globalCounter), (25,25), font, 0.9, (255,220,195),2, cv2.LINE_AA)
         
        cv2.imwrite(os.path.join(imageFolder,str(dataStart+globalCounter).zfill(4)+".png") ,resizeToMaxHW(blank,width = 800))
        #cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\orig_"+str(dataStart+globalCounter).zfill(4)+".png" ,ori)#int(err.shape[0]/3)
        # cv2.imshow(f'{globalCounter}', resizeToMaxHW(blank))
    #if big != 1: cv2.imshow("aas",aas)

    if doIntermediateData == 1:
        if globalCounter > 0 and globalCounter % intermediateDataStepInterval == 0 or globalCounter == min(dataArchive.shape[0]-2,dataStart+dataNum): # why -2 not -1? idk, it just works!
            storeDir = os.path.join(intermediateDataFolder,'data'+postfix+'.pickle')
            with open(storeDir, 'wb') as handle:
                pickle.dump(
                [
                globalCounter, g_contours, g_contours_hull, g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull, g_areas_hull, g_dropIDs , allFrozenIDs,
                g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Areas,g_Masks,g_Images,g_old_new_IDs,g_bubble_type,g_child_contours, frozenBuffer_old,STBubs,
                g_predict_displacement, g_predict_area_hull,  g_MBub_info, frozenGlobal,  g_bublle_type_by_gc_by_type, g_areas_IDs, g_splits, activeFrozen,
                g_merges], handle) 
    # ===========================================================================
    # ===================== for MHT-X append lines into csv file ================
    # ===========================================================================
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for key in list(l_masks_old.keys()):
            row = [globalCounter, min(l_old_new_IDs_old[key]), l_centroids_old[key], l_areas_hull_old[key]]
            writer.writerow(row)

    globalCounter += 1



exportFirstFrame(markFirstExport,dataStart)    
if os.path.exists(os.path.join(intermediateDataFolder,'data'+postfix+'.pickle')) and readIntermediateData: 
    with open(os.path.join(intermediateDataFolder,'data'+postfix+'.pickle'), 'rb') as handle:
                [
                backupStart, g_contours, g_contours_hull, g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull, g_areas_hull, g_dropIDs , allFrozenIDs,
                g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Areas,g_Masks,g_Images,g_old_new_IDs,g_bubble_type,g_child_contours, frozenBuffer_old, STBubs,
                g_predict_displacement, g_predict_area_hull,  g_MBub_info, frozenGlobal,  g_bublle_type_by_gc_by_type, g_areas_IDs, g_splits, activeFrozen,
                g_merges] = pickle.load(handle)
    a = 1
    globalCounter  = min(startIntermediateDataAt,backupStart)
    if startIntermediateDataAt<backupStart:
        def reduceFields(fields):
            output = []
            for field in fields:
                output.append({ID:{t:val for t,val in dic.items() if t <= globalCounter} for ID, dic in field.items() if min(list(dic.keys())) <= globalCounter})
            return output
        def reduceFields2(fields):
            output = []
            for field in fields:
                output.append({t:vals for t,vals in field.items() if t <= globalCounter})
            return output
        [g_areas_hull,g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Masks,g_Images,g_old_new_IDs,
         g_bubble_type,g_predict_displacement,g_predict_area_hull,g_MBub_info,g_contours_hull] = reduceFields([g_areas_hull,g_Centroids,g_Rect_parms,g_Ellipse_parms,
                                                                                               g_Masks,g_Images,g_old_new_IDs,g_bubble_type,
                                                                                               g_predict_displacement,g_predict_area_hull,g_MBub_info,g_contours_hull])

        [g_contours,g_dropIDs,g_bublle_type_by_gc_by_type,g_splits, allFrozenIDs, STBubs, g_merges,
         g_areas_IDs,g_FBub_rect_parms,g_FBub_centroids,g_FBub_areas_hull]     = reduceFields2([g_contours,g_dropIDs,g_bublle_type_by_gc_by_type,g_splits, allFrozenIDs,STBubs, g_merges,
                                                                                                g_areas_IDs,g_FBub_rect_parms,g_FBub_centroids,g_FBub_areas_hull])
     
        
 
    temp = sum([[[bType, subID] for subID in IDS]  for bType,IDS in g_bublle_type_by_gc_by_type[globalCounter].items() if len(IDS)>0],[])
    l_bubble_type_old = {ID:bType for bType,ID in temp}
    l_splits_old      =  g_splits[globalCounter] if globalCounter in g_splits else {}
    allMergedIDs = sum([hist[min(list(hist.keys()))][0] for ID, hist in g_MBub_info.items()],[])                     # OK if you restore on merge frame 
    #allMergedIDs2 = {min(list(hist.keys())):hist[min(list(hist.keys()))][0] for ID, hist in g_MBub_info.items()}     # OK you have to offset grab times
    #allMergedIDs3 = sum([[[b,a] for b in vals] for a,vals in allMergedIDs2.items()],[])                              # OK
    #allMergedIDs4 = {ID:mergeTime for ID,mergeTime in allMergedIDs3}
    frozenBuffer_old2   = {}
    for ID in g_Centroids:
        a = 1
        times = [t for t in list(g_Rect_parms[ID].keys()) if globalCounter - frozenBufferMaxDT <= t < globalCounter ]
        if len(times)>0:
            frozenBuffer_old2[ID] = {}
            #if ID in allMergedIDs and allMergedIDs4[ID] != globalCounter:  times = times[-frozenBufferSize-1:-1]      # ughhh, merge appends on ghost end centroid at merge ;((((
            #elif ID in allMergedIDs and allMergedIDs4[ID] == globalCounter:  times = times[-frozenBufferSize:]        # dont ask me. magic
            #elif ID in g_MBub_info: times = times[-frozenBufferSize+2:]                                               # aaaahhh, and merge ID has first ghost centroid loool
            #else:                   times = times[-frozenBufferSize:]
            times = times[-frozenBufferSize:]
            for t in times:
                frozenBuffer_old2[ID][t] = [g_Centroids[ID][t],g_areas_hull[ID][t],g_Rect_parms[ID][t]]
    activeFrozen2 = {}
    
    allFrozenIDs_numPromotes = sum(list(allFrozenIDs.values()),[])
    for ID in frozenBuffer_old2:                                                                                     
        if len(frozenBuffer_old2[ID])>=2:                                                                               # 
            path    = np.array([centroid for centroid,_,_  in frozenBuffer_old2[ID].values()], int)                     # frozenBuffer_old contains all 
            areas   = np.array([area     for _,area,_      in frozenBuffer_old2[ID].values()], int)                     # info about relevant frozens
            displacementFromStart       = path-path[0]                                                                 # at time of promotion
            displacementFromStartAbs    = np.linalg.norm(displacementFromStart[1:],axis=1)                             #  
            displacementFromStartMean   = np.mean(displacementFromStartAbs)                                            # allFrozenIDs contains
            displacementFromStartStd    = np.std(displacementFromStartAbs)                                             # number of promotions
            areaMean                    = np.mean(areas).astype(int)                                                   # 
            areaStd                     = np.std(areas).astype(int)                                                    # 
            if displacementFromStartMean < 5:                                                                          # 
                activeFrozen2[ID] = []
                counter = allFrozenIDs_numPromotes.count(ID)
                recPar = frozenBuffer_old2[ID][max(list(frozenBuffer_old2[ID].keys()))][2]
                cntrd = list(np.mean(path,axis=0).astype(int))
                activeFrozen2[ID] = [[displacementFromStartMean,displacementFromStartStd],[areaMean,areaStd],counter,cntrd,recPar]
                
         
    #print(activeFrozen2==activeFrozen)
    a = 1;#print(frozenBuffer_old2==frozenBuffer_old)
    activeFrozen = activeFrozen2
    frozenBuffer_old = frozenBuffer_old2
    rbs = [g_bublle_type_by_gc_by_type[globalCounter][bType] for bType in [typeRing,typeRecoveredRing]]
    dbs = [g_bublle_type_by_gc_by_type[globalCounter][bType] for bType in [typeElse,typeRecoveredElse]]
    fbs = [g_bublle_type_by_gc_by_type[globalCounter][bType] for bType in [typeFrozen,typeRecoveredFrozen]]
    mbs = [g_bublle_type_by_gc_by_type[globalCounter][bType] for bType in [typeMerge]]
    l_MBub_info_old = {}
    for mID in sum(mbs,[]):
        if mID in g_MBub_info and globalCounter in g_MBub_info[mID]:
            l_MBub_info_old[mID]    = g_MBub_info[mID][globalCounter]

    def assignFields(field_g, IDs, step):
        output = []
        for subIDs in IDs:
            output.append({ID:field_g[ID][step] for ID in sum(subIDs,[])})
        return output
    #l_RBub_masks_old2 = {ID:g_Masks[ID][globalCounter] for ID in sum(rbs,[])}
    a = 1
    l_contours_hull_old    = {ID:g_contours_hull[ID][globalCounter] for ID in l_bubble_type_old.keys()}
    l_ellipse_parms_old    = {ID:g_Ellipse_parms[ID][globalCounter] for ID in l_bubble_type_old.keys()}
    frozenGlobal            = {t:IDs for t,IDs in frozenGlobal.items() if t <= globalCounter} # first time that ID appears. to prevent future IDs
    #contoursFilter_RectParams_dropIDs_old2, contoursFilter_RectParams_old2 = g_drop_keep_IDs[globalCounter]
    l_Areas_old            = {ID: cv2.contourArea(g_contours[globalCounter][ID])                            for ID in g_areas_IDs[globalCounter]}
    l_Areas_hull_old       = {ID: getContourHullArea(g_contours[globalCounter][ID])                         for ID in g_areas_IDs[globalCounter]}
    l_centroids_old_all    = {ID: getCentroidPosContours(bodyCntrs = [g_contours[globalCounter][ID]])[0]    for ID in g_areas_IDs[globalCounter]}
    l_rect_parms_all_old   = {ID: cv2.boundingRect(g_contours[globalCounter][ID])                           for ID in g_areas_IDs[globalCounter]} 
    contoursFilter_RectParams_dropIDs_old = g_dropIDs[globalCounter]
    a = 1
    [l_RBub_masks_old,         l_DBub_masks_old,          l_FBub_masks_old,      l_MBub_masks_old       ] =  assignFields(g_Masks,        [rbs,dbs,fbs,mbs], globalCounter)
    [l_RBub_images_old,        l_DBub_images_old,         l_FBub_images_old,     l_MBub_images_old      ] =  assignFields(g_Images,       [rbs,dbs,fbs,mbs], globalCounter)
    [l_RBub_rect_parms_old,    l_DBub_rect_parms_old,     l_FBub_rect_parms_old, l_MBub_rect_parms_old  ] =  assignFields(g_Rect_parms,   [rbs,dbs,fbs,mbs], globalCounter)
    [l_RBub_centroids_old,     l_DBub_centroids_old,      l_FBub_centroids_old,  l_MBub_centroids_old   ] =  assignFields(g_Centroids,    [rbs,dbs,fbs,mbs], globalCounter)
    [l_RBub_areas_hull_old,    l_DBub_areas_hull_old,     l_FBub_areas_hull_old, l_MBub_areas_hull_old  ] =  assignFields(g_areas_hull,   [rbs,dbs,fbs,mbs], globalCounter)
    [l_RBub_old_new_IDs_old,   l_DBub_old_new_IDs_old,    l_FBub_old_new_IDs_old,l_MBub_old_new_IDs_old ] =  assignFields(g_old_new_IDs,  [rbs,dbs,fbs,mbs], globalCounter)
    
    l_masks_old         = {**l_RBub_masks_old,        **l_DBub_masks_old,         **l_FBub_masks_old,           **l_MBub_masks_old}
    l_images_old        = {**l_RBub_images_old,       **l_DBub_images_old,        **l_FBub_images_old,          **l_MBub_images_old}
    l_rect_parms_old    = {**l_RBub_rect_parms_old,   **l_DBub_rect_parms_old,    **l_FBub_rect_parms_old,      **l_MBub_rect_parms_old}
    l_centroids_old     = {**l_RBub_centroids_old,    **l_DBub_centroids_old,     **l_FBub_centroids_old,       **l_MBub_centroids_old}
    l_old_new_IDs_old   = {**l_RBub_old_new_IDs_old,  **l_DBub_old_new_IDs_old,   **l_FBub_old_new_IDs_old,     **l_MBub_old_new_IDs_old}
    l_areas_hull_old    = {**l_RBub_areas_hull_old,   **l_DBub_areas_hull_old,    **l_FBub_areas_hull_old,      **l_MBub_areas_hull_old}
    
    globalCounter += 1
    dataStartOffseted = dataStart + globalCounter
    cntr = globalCounter
    # =====================================================================================================
    # ======== for MHT-X generate pre current csv file from histry. append to it during iterations ========
    # =====================================================================================================
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for time in g_bublle_type_by_gc_by_type:
            for ID in sum(list(g_bublle_type_by_gc_by_type[time].values()),[]):
                
                row = [time, min(g_old_new_IDs[ID][time]), g_Centroids[ID][time], g_areas_hull[ID][time]]
                writer.writerow(row)
    # =====================================================================================================
else:
    dataStartOffseted = dataStart
    cntr = 0
    # =====================================================================================================
    # = for MHT-X generate blank csv file in case you start from iter 0. append to it during iterations ===
    # =====================================================================================================
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)

breakLoopInsert     = False

if mode == 1:
    if big == 0:
        mainer(imgNum)
    # ss = bigRun[99]
    # print(ss)
    # mainer(ss)
    else:
        for i in range(dataStartOffseted,min(dataArchive.shape[0]-1,dataStart+dataNum),1):
            print(f'\n==================*IMG:{i}*========================')
            print(f'{timeHMS()}: Time step ID: {cntr} max ID: {dataNum-1}')
            mainer(i)
            cntr += 1
            if breakLoopInsert == True: break  #not used anymore. previosuly for assist mask generation, now it does not stop loop but waits.
if mode == 2:
    mainer(0)
 
if drawAfter == 1 and exportImages == 1:
    for globalCounter in range(globalCounter):
        #==============================================================================================
        #======================== DRAWING STUFF START =================================================
        #==============================================================================================
    
        activeIDs = [ID for ID, timeDict in g_bubble_type.items() if globalCounter in timeDict]#;print(f'globalCounter: {globalCounter} activeIDs: {activeIDs}')
        blank = dataArchive[dataStart+globalCounter] * 1
        #ori = blank.copy()
        blank = cv2.subtract(np.uint8(blank), np.uint8(meanImage))
        blank = convertGray2RGB(blank)
        if len(fakeBox) > 0:
            x,y,w,h = fakeBox[-1]
            cv2.rectangle(blank, (x,y), (x+w,y+h),[120,15,55],1)
        typeStrings = ["N","RB", "rRB", "E", "F","rE","pm","rF","M"]
        splitStrings =  {}
        for ID in activeIDs:
            temp            = [0,len(g_old_new_IDs[ID][globalCounter])]
            currentType     = g_bubble_type[ID][globalCounter]#;print('currentType',currentType)
            currentTypeStr  = typeStrings[currentType] 
            currentCntrIds  = g_old_new_IDs[ID][globalCounter]
            # text            = currentTypeStr + '('+','.join(list(map(str,currentCntrIds)))+')'
            temp.append(currentTypeStr + '(')
            cCIstr = list(map(str,currentCntrIds))
            A = [x for y in zip(cCIstr, [","]*len(cCIstr)) for x in y][:-1] #dont ask me
            temp += A
            if min(g_old_new_IDs[ID]) == globalCounter:                 # if current time step is the first in IDs data - mark it "NEW"            
                temp += [f"):{ID} new"]
            else:                                                       # 
                prevTimeStep = globalCounter - 1 if (currentType != typeFrozen and currentType != typeRecoveredFrozen) else [ts for ts in g_bubble_type[ID].keys() if ts < globalCounter][-1]
                temp[0] = 1
                prevType = g_bubble_type[ID][prevTimeStep]
                prevTypeStr = typeStrings[prevType] 
                prevCntrIds  = g_old_new_IDs[ID][prevTimeStep]
                # text += "-" + prevTypeStr + f'({ID})'
                temp += [")"+str(":" + prevTypeStr + f'({ID})')]
            splitStrings[ID] = temp
        
            if ID in g_child_contours:
                if globalCounter in g_child_contours[ID]:
                    [cv2.drawContours(  blank,   g_contours[globalCounter], cid, (0,255,0), 1) for cid in g_child_contours[ID][globalCounter]]
            # aa = [cid for cid in g_child_contours[ID][globalCounter] if globalCounter in g_child_contours[ID]]
            colorSet = [(),(0,0,255),(0,100,190),(255,0,0),(125,0,125),(255,0,125),(0,255,0),(125,0,125),(100,255,100)]
            thcSet = [(), 2, 1, 2, 2, 2 ,2, 2,2]
            clr = colorSet[currentType]
            thc = thcSet[currentType]
            [cv2.drawContours(  blank,   g_contours[globalCounter], cid, clr, thc) for cid in currentCntrIds]
            cv2.ellipse(blank, g_Ellipse_parms[ID][globalCounter], (55,55,55), 1)

            cv2.drawContours(  blank,   [g_contours_hull[ID][globalCounter]], -1, (255,255,255), 1)
            
        fontScale = 0.7;thickness = 4;

        # ========= overlay organization =============
        # check drawing.py for standalone example. 
        # works the following way. bubbles fight for vertical space. width is modified to text width. bubbles that overlay with other on vertical space are grouped. 
        # that means that text will overlap. group is split into top and bottom groups- less overlay. top and bottom groups are split into vertical positions.
        thickRectangle = {}
        rectParams = {}
        strDimsAll = {}
    
        H,L,_           = blank.shape
        textHeightMax   = 0
        baselineMax     = 0
        for k,ID in enumerate(activeIDs):
            case, numIDs, *strings  = splitStrings[ID]
            strDims  = [cv2.getTextSize(text, font, fontScale, thickness) for text in strings]          # substrings are split to make a line pointing from subIDs to contour
            baseline = max([bl for _,bl in strDims])                                                    # i draw substring seperately. full string drawing is scaled differently
            ySize = max([sz[1] for sz,_ in strDims])
            xSize = [sz[0] for sz,_ in strDims]
            totSize = sum(xSize)
            #[totSize,ySize],baseline  = cv2.getTextSize(''.join(strings), font, fontScale, thickness) # i think text dimensions for partial sum and full string are different.

            textHeightMax           = max(textHeightMax,    ySize)                              # total text height is textHeightMax + baselineMax
            baselineMax             = max(baselineMax,      baseline)                           # baselineMax is height of the part of text under the baseline e.g 'y,p,g,q,..', baseline  = 0 for 'o,k,d,..'
            x,y,w,h                 = g_Rect_parms[ID][globalCounter]
            rectParams[ID]          = [x,y,w,h] 
            dw                      = totSize-w           # if text is thicker, then 
            xS, hS                  = [int(x-0.5*dw),H]
            dx1                     = max(0, -xS)                                               # if BRect is poking out of left side of image dx1 is positive, else zero
            dx2                     = max(0, xS + totSize - L)                                  # if BRect is poking out of right side of image dx2 is positive, else zero
            thickRectangle[ID]      = [xS + dx1 - dx2,0,totSize, hS]
            strDimsAll[ID]          = strDims
            #cv2.rectangle(blank, (xS + dx1 - dx2,y), (xS + dx1 - dx2+totSize,y+h),(255,0,0),2)
        # split into competing vertical columns    
        aa = overlappingRotatedRectangles(thickRectangle,thickRectangle)
        HG = nx.Graph()
        HG.add_nodes_from(rectParams.keys())
        HG.add_edges_from(aa)
        # 
        cntrd = {ID:tuple((x+int(0.5*w),y+int(0.5*h))) for ID,[x,y,w,h] in rectParams.items()}
        posTB = {ID:0 if y < (H - (y+h)) else 1 for ID,[x,y,w,h] in rectParams.items()}

        neighbors = {ID:[] for ID in rectParams}
        neighborsSameSide = {ID:[] for ID in rectParams}
        # dont remember what, but something interesting happens for splitting.
        # top-bottom splitting is not that simple. i think only neighbors are tests wheter they share same side.
        for IDs in rectParams:
            for i in HG.neighbors(IDs):
                neighbors[IDs].append(i)
                if posTB[i] == posTB[IDs]: neighborsSameSide[IDs].append(i) # write connections between neighbors on same side
        #clusters0 = [tuple(np.sort(np.array([ID] +nbrs))) for ID, nbrs in neighborsSameSide.items()]
        a = 1
        edgePairs = sum([[(ID,a) for a in nbrs] for ID,nbrs in neighborsSameSide.items() if len(nbrs) > 0], []) # rewrite connections into pairs, ignore solo connections
        sameSideClusters = graphUniqueComponents(list(rectParams.keys()),edgePairs)                             # get same side cluster, next step order them by centroid y coordinate
    
        sortedByY = [[x for _,x in sorted(zip([cntrd[a][1] for a in X],X))] for X in sameSideClusters] # https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list
        sortedIndexed = [{ID:i for i,ID in enumerate(ID)} for ID in sortedByY]                         # enumerate index is a vertical position index for text
        globSideOrder = {k: v for d in sortedIndexed for k, v in d.items()}                            # combine list of dictionaries into on dict.
        #SameSideClusters = [list(aa) for aa in set(clusters0)]
        #globSideOrder = {ID:0 for ID in rectParams}
        #for arr in SameSideClusters:
        #    srt = np.argsort([cntrd[ID][1] for ID in arr])                                            # old method. did not work, new method on top is simpler.
        #    for i,elem in enumerate(arr):
        #        globSideOrder[elem] = srt[i]
        
        textHeight  = textHeightMax
        textNoGo    = 10
        textSep     = 15
        # text position is calculted for top and bottom part of image separately. different coordinate systems- from edge. duh
        for ID,[x,y,w,h] in thickRectangle.items():
            if posTB[ID] == 1:
                textPos = H - textNoGo - baselineMax - globSideOrder[ID]*(textHeightMax + baselineMax + textSep)
            if posTB[ID] == 0:
                textPos = textNoGo + textHeightMax + globSideOrder[ID]*(textHeightMax + baselineMax + textSep)
        
        
            case, numIDs, *strings  = splitStrings[ID]
            strDims                 = strDimsAll[ID]

            startPos                = np.array((x,textPos),int) 
            ci = 0
            subIDsIndices = list(range(1,len(strings),2))
            # iterating through overlay substrings, on subIDs calculate nearest distance from ID string center to any point on countour with that ID
            for i, string in enumerate(strings):
                [cv2.putText(blank, string, startPos, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
                if i in subIDsIndices:
                
                    startPosMiddle = startPos + np.array((strDims[i][0][0]/2,0),int)        # startPos-> string start, startPosMiddle = string start +half width
                    cntrID = g_old_new_IDs[ID][globalCounter][ci]
                    _, _, p1, p2 = closes_point_contours(startPosMiddle.reshape(-1,1,2),g_contours[globalCounter][cntrID])
                    cv2.polylines(blank, [np.array([p1- np.array((0,(textHeightMax + baselineMax)/2),int),p2])],0, (0,0,255), 1)
                    startPos += np.array((strDims[i][0][0],0),int)
                    ci += 1
                else: startPos += np.array((strDims[i][0][0],0),int)                        # skip, add full width offset.
            
            #string = ''.join(strings)
            #[cv2.putText(blank, string, (x,textPos), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
           
        

        #cv2.imshow('aa',blank)
    
        #--------------- trajectories ---------------------------------
        for i,times in g_Centroids.items(): # key IDs and IDS-> times
            useTimes = [t for t in times.keys() if t <= globalCounter and t>globalCounter - 10]#
            pts = np.array([times[t] for t in useTimes]).reshape(-1, 1, 2)
            if pts.shape[0]>3:
                cv2.polylines(blank, [pts] ,0, (255,255,255), 3)
                cv2.polylines(blank, [pts] ,0, cyclicColor(i), 2)
                [cv2.circle(blank, tuple(p), 3, cyclicColor(i), -1) for [p] in pts]
            else:
                cv2.polylines(blank, [pts] ,0, cyclicColor(i), 1)
                [cv2.circle(blank, tuple(p), 1, cyclicColor(i), -1) for [p] in pts]
            
        cv2.putText(blank, str(globalCounter), (25,25), font, 0.9, (255,220,195),2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(imageFolder,str(dataStart+globalCounter).zfill(4)+".png") ,resizeToMaxHW(blank,width = 800))
        #cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\"+str(dataStart+globalCounter).zfill(4)+".png" ,blank)
        #cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\orig_"+str(dataStart+globalCounter).zfill(4)+".png" ,ori)
        # cv2.imshow(f'{globalCounter}', resizeToMaxHW(blank))
    
    #==============================================================================================
    #======================== DRAWING STUFF  END ==================================================
    #==============================================================================================
k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()
    
# print(type(shareData[1][2][i])==np.ndarray)
# p10=np.array([150,150])
# d = np.array([100,0])
# theta = 90*np.pi/180
# rot=np.array([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]],dtype=float)
# global vec
# vec = np.matmul(rot,d).astype(int)
# print(vec)
# # print(rot)
# pts = np.array([p10,p10+vec])
# # print(pts)
# # print(pts.reshape((-1, 1, 2)))
# pts0=np.array([p10,p10+d])
# cv2.polylines(aas, [pts0],0, (0,0,255), 3)
# cv2.polylines(aas, [pts],0, (0,255,0), 1)


# in mainer

# if 1==12:
#             from skimage.morphology import medial_axis, skeletonize
#             contours, _ = cv2.findContours(err,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#             # print([cv2.contourArea(c) for c in contours])
#             areaMax = np.argmax([cv2.contourArea(c) for c in contours])
#             # print(areaMax)
#             contours = [contours[areaMax]]
#             err = cv2.drawContours(  err.copy(), contours, -1, 0, -1)
#             # cv2.imshow("err",err)
#             # cSel = [i for i in range(len(contours)) if cv2.contourArea(contours[i]) > minArea]
#             blobs = err/255
#             # Compute the medial axis (skeleton) and the distance transform
#             skel, distance = medial_axis(blobs, return_distance=True)
            
#             # Compare with other skeletonization algorithms
#             skeleton = skeletonize(blobs)
#             skeleton_lee = skeletonize(blobs, method='lee')
#             cv2.imwrite("D:\\Alex\\Darbs.exe\\Python_general\\bubble_process\\skel4.png" ,skeleton_lee)
#             # Distance to the background for pixels of the skeleton
#             dist_on_skel = distance * skel
#             fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
#             ax = axes.ravel()
            
#             ax[0].imshow(blobs, cmap=plt.cm.gray)
#             ax[0].set_title('original')
#             ax[0].axis('off')
            
#             ax[1].imshow(dist_on_skel, cmap='magma')
#             ax[1].contour(blobs, [0.5], colors='w')
#             ax[1].set_title('medial_axis')
#             ax[1].axis('off')
            
#             ax[2].imshow(skeleton, cmap=plt.cm.gray)
#             ax[2].set_title('skeletonize')
#             ax[2].axis('off')
            
#             ax[3].imshow(skeleton_lee, cmap=plt.cm.gray)
#             ax[3].set_title("skeletonize (Lee 94)")
#             ax[3].axis('off')
            
#             fig.tight_layout()
#             plt.show()
    
#         if 1==21:
#             def store_evolution_in(lst):
#                 """Returns A callback function to store the evolution of the level sets in
#                 the given list.
#                 """
            
#                 def _store(x):
#                     lst.append(np.copy(x))
            
#                 return _store
        
#             init_ls = checkerboard_level_set(orig.shape, 6)
#             evolution = []
#             callback = store_evolution_in(evolution)
#             ls = morphological_chan_vese(orig, iterations=35, init_level_set=init_ls,
#                                          smoothing=3, iter_callback=callback)
#             fig, axes = plt.subplots(2, 1, figsize=(8, 8))
#             ax = axes.flatten()
            
#             ax[0].imshow(orig, cmap="gray")
#             ax[0].set_axis_off()
#             ax[0].contour(ls, [0.5], colors='r')
#             ax[0].set_title("Morphological ACWE segmentation", fontsize=12)
            
#             ax[1].imshow(ls, cmap="gray")
#             ax[1].set_axis_off()
#             contour = ax[1].contour(evolution[2], [0.5], colors='g')
#             contour.collections[0].set_label("Iteration 2")
#             contour = ax[1].contour(evolution[7], [0.5], colors='y')
#             contour.collections[0].set_label("Iteration 7")
#             contour = ax[1].contour(evolution[-1], [0.5], colors='r')
#             contour.collections[0].set_label("Iteration 35")
#             ax[1].legend(loc="upper right")
#             title = "Morphological ACWE evolution"
#             ax[1].set_title(title, fontsize=12)
            
#         if 1==21:
#             bs = convertGray2RGB(err)
#             contours, _ = cv2.findContours(err, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             minEllipse = [None]*len(contours)
#             for i, c in enumerate(contours):
#                 if c.shape[0] > 5:
#                     minEllipse[i] = cv2.fitEllipse(c)
#                     cv2.ellipse(bs, minEllipse[i], (255,0,0), 2)
           
                
#             cv2.imshow('bs',bs)
# match shapes vi Hu invariants ------------------------
# if 1==2:
#             cnt = parentsCNTRS[1]
#             x,y,w,h = cv2.boundingRect(cnt)
#             asz = cv2.drawContours(
#                                     np.zeros((h,w), dtype=np.uint8)
#                                     ,parentsCNTRS,1,200,-1,cv2.LINE_AA,offset=(-x,-y))
            
#             # if big != 1: cv2.imshow("dasdsada",asz)                       
 
#             with open('matchShape.pickle', 'weightedB') as handle:
#                         pickle.dump(parentsCNTRS[1], handle)
#         if 1 == 1:
#             # gug = aas.copy() * 0
#             gug = convertRGB2Gray(aas.copy() * 0)
#             for i in range(1,3):
#                  cv2.drawContours(gug,parentsCNTRS,i,255,-1,cv2.LINE_AA)
            
#             mp1 = cv2.moments(gug)#;print("moms ",mp1)
            
#             cpr2 = [int(mp1['m10']/mp1['m00']), int(mp1['m01']/mp1['m00'])];print("cpr2 ",cpr2)
#             cv2.circle(gug, tuple(cpr2), 3, 125, -1)
#             # cv2.imshow("gug",gug)
#             # contours5, ir5 = cv2.findContours(gug, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if 1 == 1:
#             with open('matchShape.pickle', 'rb') as handle:
#                 tte = pickle.load(handle)
#             bebs = cv2.drawContours(gug.copy()*0,[tte],-1,255,-1,cv2.LINE_AA)
#             # cv2.imshow("bebs",bebs)
#             # cv2.imshow("gug",gug)
#             # [print(cv2.matchShapes(cntr,tte,1,0.0)) for cntr in parentsCNTRS]
#             print(cv2.matchShapes(bebs,gug,1,0.0))

# ------ moment angles------------------
# for (cntPi,cntP) in enumerate(parentsCNTRS):
#                 temp = list()
#                 mp = cv2.moments(cntP)#;print("moms ",mp)
#                 cpr = [int(mp['m10']/mp['m00']), int(mp['m01']/mp['m00'])]
#                 momentsListPonly.append(cpr)
#                 for cSel in whereChildrenAreaFiltered[cntPi]:
#                     cCntr = l_contours[cSel]
#                     mc = cv2.moments(cCntr)
#                     ccr = [int(mc['m10']/mc['m00']), int(mc['m01']/mc['m00'])]#;print("child ",cSel,"ccr ", ccr)
#                     temp.append([cpr,ccr])
#                     # aas = cv2.circle(aas.copy(), tuple(ccr), radius=3, color=(0, 0, 255), thickness=-1)
                    
#                 (mu00,mu11,mu20,mu02) = (mp['m00'],mp['mu11'],mp['mu20'],mp['mu02'])
#                 mu11x,mu02x,mu20x = mu11/mu00, mu02/mu00, mu20/mu00
#                 ang = 0.5*np.arctan(2*mu11x/(mu20x-mu02x))#;print(ang)
#                 da = 0.5*(mu20x+mu02x)
#                 db = 0.5*np.sqrt(4*mu11x**2 + (mu20x-mu02x)**2)
#                 # d0 = int(max(da+db,da-db))#;print(d0)
#                 d0 = int(np.linalg.norm([da+db,da-db]))#;print(d0)
#                 # d0 = 100
#                 d = np.array([int(d0/10),0])
#                 theta = 90*np.pi/180
#                 rot=np.array([[np.cos(ang),-1*np.sin(ang)],[np.sin(ang),np.cos(ang)]],dtype=float)
#                 vec = np.matmul(rot,d).astype(int)
#                 pts0=np.array([cpr,cpr+vec])
#                 cv2.polylines(aas, [pts0],0, (0,0,255), 2)
#                 childParentCentroids.append(temp)




#-------------- CHECK BOUNDING RECTANGlE INSIDE BOUNDING RECTANGLE-----
# alpha = 0.05
# x,y = int(x - alpha*w), int(y - alpha*h)
# h,w = int((1 + 2*alpha)*h) , int((1 + 2*alpha)*w)
# area0 = np.sum(subMasks_old[i][subMasks_old[i]>0]/255)
# # cv2.rectangle(blank, (300,300), (500,700),255,-1)
# for ci,cntr in enumerate(l_contours):
#     xc,yc,wc,hc = cv2.boundingRect(cntr)
#     if xc>x and yc>y and xc+wc<x+w and y+hc<y+h and cv2.contourArea(cntr)> 0.2*area0:

#         cv2.rectangle(aas, (xc,yc), (xc+wc,yc+hc),(178,60,200),1)
#         cv2.rectangle(aas, (x,y), (x+w,y+h),(60,125,200),1)
#         print('asdasdasdasdas', i)
#         aas = drawContoursGeneral(aas, l_contours, ci, (0,127,127), 2)
#         cpr = getCentroidPos(inp = cntr, offset = (0,0), mode=0, mask=[])
#         # cv2.putText(aas, f' missing bubble id {ci}', tuple(np.array(cpr)+(10,-30)), font, 0.7, (60,125,200),1, cv2.LINE_AA)
#         centroidsByID[i][-1] = cpr
#         if i in unresolvedOldRB: unresolvedOldRB.remove(i)
# cv2.imshow(str(i),err[y:y+h, x:x+w])
# print(y,x)
# [cv2.drawContours( blank,   l_contours, k, (0,255,0), 1) for k in temp]  
# cv2.imshow("asdasdas",blank)



# %timeit -r 4 -n 1000  fuu(d,1)
#print(dataArchive[1] == dataArchive[2])
#dataStart = 1
#dataNum = 51
#cntr = 1
#for i in range(dataStart,dataStart+dataNum,1):
#            #print(f'\n==============================================')
#            print(f'Time step ID: {cntr} max ID: {dataNum-1}. i: {i}')
#            orig0 = dataArchive[i]
#            if cntr >= 1  and cntr <= 25:
#                cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\"+str(dataStart+cntr).zfill(4)+".png" ,orig0)
#                #print('cntr > 1  and cntr <= 10')
#            orig = orig0 -cv2.blur(mean, (5,5),cv2.BORDER_REFLECT)
#            orig[orig < 0] = 0                  # if orig0 > mean
#            orig = orig.astype(np.uint8)
#            if cntr > 25  and cntr <= 50:
#                cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\"+str(dataStart+cntr).zfill(4)+".png" ,orig)
#            cntr += 1