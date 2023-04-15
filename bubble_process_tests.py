# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:30:19 2022

@author: User
"""
#print("\014") #clear spyder console, if you use it
from ast import For
import cv2
# from cv2.ximgproc import anisotropicDiffusion
# from cv2.ximgproc import getDisparityVis
# from skimage.segmentation import (morphological_chan_vese,
#                                   morphological_geodesic_active_contour,
#                                   inverse_gaussian_gradient,
#                                   checkerboard_level_set)
import numpy as np, itertools, networkx as nx
import os#, multiprocessing
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
    return [int(scaled*refRadius),ccPerpendicularDir,angleThreshold,midPoint]                   # notice weights are swapped, merge point will be closer to smaller centroid


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

thresh0             = 1
thresh1             = 15


plotAreaRep         = 0
inspectBubbleDecay  = 211
testThreshold       = 222
testEdgePres        = 3

mode = 1 # mode: 0 - read new images into new array, 1- get one img from existing array, 2- recall pickled image
big = 1
# dataStart = 71+52 ###520
# dataNum = 7
dataStart           = 300 #  53+3+5
dataNum             = 130 # 7+5   
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
markFirstExport = 0 # see exportFirstFrame() lower after X_data import



globalCounter = 0

# debug section numbers: 11- RB IDs, 12- RB recovery, 21- Else IDs, 22- else recovery, 31- merge
# debug on specific steps- empty list, do all steps, or time steps in list
debugSections = [11,21,12,22,31]
debugSteps = [5]
debugVecPredict = 0
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
imgNum = 46

imageMainFolder = r'D:/Alex/Darbs.exe/Python_general/bubble_process/imageMainFolder/'
init(imageMainFolder,imgNum)

X_data, mean, imageLinks = initImport(mode,workBigArray,recalcMean,readSingleFromArray,pickleNewDataLoad,pickleNewDataSave,pickleSingleCaseSave)

def exportFirstFrame(markFirstExport,dataStart):
    if markFirstExport == 1:
        if not os.path.exists(r'./manualMask'):
            os.mkdir(r'./manualMask')
        
        orig0 = X_data[dataStart]
        orig = orig0 -cv2.blur(mean, (5,5),cv2.BORDER_REFLECT)
    
        orig[orig < 0] = 0                  # if orig0 > mean
        orig = np.array(orig, dtype = np.uint8)
        origTH = np.array(cv2.threshold(orig,thresh0,255,cv2.THRESH_BINARY)[1], dtype = np.uint8)
        err = cv2.morphologyEx(origTH.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        #exp = np.maximum.reduce([orig,origTH]) # element wise max() for two matrices. kind of useless
        #cv2.imshow('orig',orig)
        #cv2.imshow('origTH',origTH)
        #cv2.imshow('exp',exp)
        cv2.imwrite("./manualMask/frame"+str(dataStart).zfill(4)+".png" ,err)
        return 1
    else: return 0

def extractManualMask(): # either draw red masks over or using Paint, set bg color to red and freehand select and delete areas.
    manualMask = cv2.imread("./manualMask/frame"+str(dataStart).zfill(4)+" - Copy.png",1)
    manualMask = np.uint8(manualMask[:,:,2])
    _,manualMask = cv2.threshold(manualMask,230,255,cv2.THRESH_BINARY)
    #cv2.imshow('manualMask',manualMask)
    contours = cv2.findContours(manualMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    params = [cv2.boundingRect(c) for c in contours]
    #for x,y,w,h in params:
    #   cv2.rectangle(manualMask,(x,y),(x+w,y+h),128,3)
    #cv2.drawContours(manualMask, contours, -1, 128, 5)
    #cv2.imshow('manualMask',manualMask)
    IDs = np.arange(0,len(params))
    return contours,{ID:param for ID,param in enumerate(params)}

#typeFull,typeRing, typeRecoveredRing, typeElse, typeFrozen,typeRecoveredElse,typePreMerge,typeRecoveredFrozen = np.int8(0),np.int8(1),np.int8(2),np.int8(3), np.int8(4), np.int8(5), np.int8(6), np.int8(7)

[typeFull,typeRing, typeRecoveredRing, typeElse, typeFrozen,typeRecoveredElse,typePreMerge,typeRecoveredFrozen,typeMerge] = np.array([0,1,2,3,4,5,6,7,8])
typeStrFromTypeID = {tID:tStr for tID,tStr in zip(np.array([0,1,2,3,4,5,6,7,8]),['OB','RB', 'rRB', 'DB', 'FB', 'rDB', 'pm', 'rF', 'MB'])}
# g_(...) are global storage variable that contain info from all processed times.
# l_(...) are local storage variable that contain info from current time.
# l_(...)_old are global variables that contain info from previous time. its technically global to acces across two time steps, but are local by idea.
# (...)_r stands for recovered
g_Centroids, g_Rect_parms, g_Ellipse_parms, g_Areas, g_Masks, g_Images, g_old_new_IDs, g_bubble_type, g_child_contours = {}, {}, {}, {}, {}, {}, {}, {}, {}
l_RBub_masks_old, l_RBub_images_old, l_RBub_rect_parms_old, l_RBub_centroids_old, l_RBub_areas_hull_old, l_RBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
l_DBub_masks_old, l_DBub_images_old, l_DBub_rect_parms_old, l_DBub_centroids_old, l_DBub_areas_hull_old, l_DBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old, l_FBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
l_MBub_masks_old, l_MBub_images_old, l_MBub_rect_parms_old, l_MBub_centroids_old, l_MBub_areas_hull_old, l_MBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
frozenGlobal,frozenGlobal_LocalIDs = {},{}; l_MBub_info_old,g_MBub_info = {}, {}

g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull = {},{},{}
g_contours, g_contours_hull,  frozenBubs, frozenBubsTimes,  l_bubble_type_old = {}, {}, {}, {}, {}
l_Centroids_old, l_Areas_old,l_Areas_hull_old, l_rect_parms_all, l_rect_parms_all_old = {}, {}, {}, {}, {}
g_predict_displacement, g_predict_area_hull  = {}, {}; frozenIDs, frozenIDs_old  = [],[]
l_masks_old, l_images_old, l_rect_parms_old, l_ellipse_parms_old, l_centroids_old, l_old_new_IDs_old, l_areas_hull_old, l_contours_hull_old = {}, {}, {}, {}, {}, {}, {}, {}
fakeBox = {}    # inlet area rectangle
def mainer(index):
    global globalCounter, g_contours, g_contours_hull, frozenBubs, frozenBubsTimes, l_Centroids_old, l_Areas_old, l_Areas_hull_old, l_rect_parms_all, l_rect_parms_all_old

    global l_RBub_masks_old, l_RBub_images_old, l_RBub_rect_parms_old, l_RBub_centroids_old, l_RBub_old_new_IDs_old, ringCntrIDs
    global l_DBub_masks_old, l_DBub_images_old, l_DBub_rect_parms_old,  l_DBub_centroids_old, l_DBub_areas_hull_old, l_DBub_old_new_IDs_old
    global l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old, l_FBub_old_new_IDs_old
    global l_MBub_masks_old, l_MBub_images_old, l_MBub_rect_parms_old, l_MBub_centroids_old, l_MBub_areas_hull_old, l_MBub_old_new_IDs_old
    global g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull
    global g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Areas,g_Masks,g_Images,g_old_new_IDs,g_bubble_type,l_bubble_type_old,g_child_contours
    global g_predict_displacement, g_predict_area_hull, frozenIDs, frozenIDs_old, l_MBub_info_old, g_MBub_info
    global l_masks_old, l_images_old, l_rect_parms_old, l_ellipse_parms_old, l_centroids_old, l_old_new_IDs_old, l_areas_hull_old, l_contours_hull_old
    global fakeBox
 
    orig0           = X_data[index]
    # wanted to replace code below with cv2.subtract, but there are alot of problems with dtypes and results are a bit different
    orig            = orig0 -cv2.blur(mean, (5,5),cv2.BORDER_REFLECT)
    orig[orig < 0]  = 0                  # if orig0 > mean
    orig            = orig.astype(np.uint8)

    _,err           = cv2.threshold(orig.copy(),thresh0,255,cv2.THRESH_BINARY)
    err             = cv2.morphologyEx(err.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    #err = cv2.circle(err.copy() , (100,200), 11, 255, -1) if globalCounter<dataNum-1 else err.copy()
    
    if workBigArray == 1  and readSingleFromArray == 1: gfx = 1
    else: gfx = 0
    if workBigArray == 0: gfx = 1

    global debugVecPredict,contoursFilter_RectParams_dropIDs_old
    mergeCandidates,mergeCandidatesSubcontourIDs = {},{}
    l_bubble_type, l_contours_hull, l_RBub_contours_hull, l_DBub_contours_hull, l_FBub_contours_hull, l_MBub_contours_hull      = {}, {}, {}, {}, {}, {}
    l_RBub_masks, l_RBub_images, l_RBub_rect_parms, l_RBub_centroids, l_RBub_areas_hull, l_RBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    l_DBub_masks, l_DBub_images, l_DBub_rect_parms, l_DBub_centroids, l_DBub_areas_hull, l_DBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    l_RBub_r_masks, l_RBub_r_images, l_RBub_r_rect_parms,l_RBub_r_centroids, l_RBub_r_areas_hull, l_RBub_r_old_new_IDs          = {}, {}, {}, {}, {}, {}
    l_DBub_r_masks, l_DBub_r_images, l_DBub_r_rect_parms,l_DBub_r_centroids, l_DBub_r_areas_hull, l_DBub_r_old_new_IDs          = {}, {}, {}, {}, {}, {}
    l_MBub_masks, l_MBub_images, l_MBub_rect_parms, l_MBub_centroids, l_MBub_areas_hull, l_MBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    l_FBub_masks, l_FBub_images, l_FBub_rect_parms, l_FBub_centroids, l_FBub_areas_hull, l_FBub_old_new_IDs                     = {}, {}, {}, {}, {}, {}
    l_predict_displacement = {};l_MBub_info = {}
    # get contours from binary image. filter out useless. 
    global contoursFilter_RectParams,contoursFilter_RectParams_dropIDs
    contoursFilter_RectParams_dropIDs,l_Centroids = [], {}
    topFilter, bottomFilter, minArea    = 80, 10, 180                                                                                           # instead of deleting contours at inlet, i should merge
                                                                                                                                                # them into one. since there is only ~one bubble at a time
    (l_contours,
     whereParentOriginal,
     whereParentAreaFiltered,
     whereChildrenAreaFiltered)             = cntParentChildHierarchy(err,1, 1200,130,0.1)                                                      # whereParentOriginal all non-child contours.
    g_contours[globalCounter]               = l_contours                                                                                        # add contours to global storage.
    contoursFilter_RectParams               = {ID: cv2.boundingRect(l_contours[ID]) for ID in whereParentOriginal}                              # remember bounding rectangle parameters for all primary contours.
    contoursFilter_RectParams_dropIDs       = [ID for ID,params in contoursFilter_RectParams.items() if sum(params[0:3:2])<topFilter]           # filter out bubbles at left image edge, keep those outside 80 pix boundary. x+w < 80 pix.
    #contoursFilter_RectParams_dropIDs_inlet = [ID for ID,params in contoursFilter_RectParams.items() if params[0] > err.shape[1]- bottomFilter] # top right corner is within box that starts at len(img)-len(box)
    #contoursFilter_RectParams_dropIDs       = contoursFilter_RectParams_dropIDs + contoursFilter_RectParams_dropIDs_inlet
    l_Areas                                 = {key: cv2.contourArea(l_contours[key]) for key in contoursFilter_RectParams if key not in contoursFilter_RectParams_dropIDs } # remember contour areas of main contours that are out of side band.
    l_Areas_hull                            = {ID:getContourHullArea(l_contours[ID]) for ID in l_Areas}                                         # convex hull of a single contour. for multiple contrours use getCentroidPosContours.
    contoursFilter_RectParams_dropIDs       = contoursFilter_RectParams_dropIDs + [key for key, area in l_Areas.items() if area < minArea]      # list of useless contours- inside side band and too small area.
    l_rect_parms_all                        = {key: val for key, val in contoursFilter_RectParams.items() if key in l_Areas}                    # bounding rectangle parameters for all primary except within a band.
    l_Centroids                             = {key: getCentroidPosContours(bodyCntrs = [l_contours[key]])[0] for key in l_rect_parms_all}       # centroids of ^
    
    frozenIDs = []
    global frozenGlobal,frozenGlobal_LocalIDs
    frozenLocal = []
 
    fakeBoxW, fakeBoxH      = 176,int(err.shape[0]/3)                                             # generate a fake bubble at inlet and add it to previous frame data.
    fakeBox                 = {-1:[err.shape[1] - fakeBoxW + 1, fakeBoxH, fakeBoxW, fakeBoxH]}    # this will make sure to gather new bubbles at inlet into single cluster.

    if globalCounter >= 1:
        # try to find old contours or cluster of contours that did not move during frame transition.
        # double overlay of old global and old local IDs, remove local & keep global.
        print(f'{globalCounter}:-------- Begin search for frozen bubbles ---------')
        dropKeys                = lambda lib,IDs: {key:val for key,val in lib.items() if key not in IDs}                # function that drops all keys listed in ids from dictionary lib
        deleteOverlaySoloIDs    = [subIDs[0] for _, subIDs in l_old_new_IDs_old.items() if len(subIDs) == 1]            # these are duplicates of global IDs from prev frame. ex: l_old_new_IDs_old= {0: [15]} -> l_Areas_old= {15:A} & joinAreas = {0:A} same object.
        stuckAreas              = {**dropKeys(l_Areas_hull_old,deleteOverlaySoloIDs),     **{str(key):val for key,val in l_areas_hull_old.items()}} # replaced regular area with fb hull 11/02/23
        stuckRectParams         = {**dropKeys(l_rect_parms_all_old,deleteOverlaySoloIDs), **{str(key):val for key,val in l_rect_parms_old.items()}}
        stuckCentroids          = {**dropKeys(l_Centroids_old,deleteOverlaySoloIDs),      **{str(key):val for key,val in l_centroids_old.items()}}


        #cntrRemainingIDs = [cID for cID in whereParentOriginal if cID not in contoursFilter_RectParams_dropIDs ] #+ dropAllRings + dropRestoredIDs
        #
        #distContours = {ID:contoursFilter_RectParams[ID] for ID in cntrRemainingIDs}
        minAreaInlet                    = 150
        inletIDsNew, inletIDsTypeNew    = overlappingRotatedRectangles(contoursFilter_RectParams, fakeBox, returnType = 1)          # inletIDs, inletIDsType after frozen part
        inletIDsNew                     = [a[0] for a in inletIDsNew]
        inletIDsNew                     =  [ID for ID in inletIDsNew if cv2.contourArea(l_contours[ID]) > minAreaInlet]
        [contoursFilter_RectParams_dropIDs.remove(x) for x in inletIDsNew if x in contoursFilter_RectParams_dropIDs]                # remove-drop == which to keep.
        dropKeysOld                                             = lambda lib: dropKeys(lib,contoursFilter_RectParams_dropIDs_old)   # dropping frozens from inlet since
        dropKeysNew                                             = lambda lib: dropKeys(lib,contoursFilter_RectParams_dropIDs        + inletIDsNew)  # bubbles act bad there, might false-positive
        [stuckRectParams,stuckAreas,stuckCentroids]             = list(map(dropKeysOld,[stuckRectParams,stuckAreas,stuckCentroids] ))
        [fbStoreRectParams2,fbStoreAreas2,fbStoreCentroids2]    = list(map(dropKeysNew,[l_rect_parms_all,l_Areas,l_Centroids] ))
        
        

        # frozen bubbles, other than from previous time step should be considered.================================
        # take frozen bubbles from last N steps. frozen bubbles found in old step are already accounted in else/dist data, get all other.
        lastStepFrozenGlobIDs           = list(l_FBub_old_new_IDs_old.keys())
        lastNStepsFrozen                = 10
        lastNStepsFrozenTimesAll        = [time for time in frozenGlobal.keys()if max(globalCounter-lastNStepsFrozen-1,0)<time<globalCounter] # if time is in [0 1 2 3 4 5], lastNStepsFrozen = 3, globalCounter = 5 -> out = [2, 3, 4]
        lastNStepsFrozenIDsExceptOld    = set([a for a in sum(frozenGlobal.values(),[]) if a not in lastStepFrozenGlobIDs]) if len(lastNStepsFrozenTimesAll)>0 else []
        lastNStepsFrozenLatestTimes     = {ID:max([time for time,IDs in frozenGlobal.items() if ID in IDs]) for ID in lastNStepsFrozenIDsExceptOld}
        lastNStepsFrozenRectParams      = {str(ID):g_FBub_rect_parms[latestTime][ID]     for ID,latestTime in lastNStepsFrozenLatestTimes.items()}
        lastNStepsFrozenHullAreas       = {str(ID):g_FBub_areas_hull[latestTime][ID]     for ID,latestTime in lastNStepsFrozenLatestTimes.items()}
        lastNStepsFrozenCentroids       = {str(ID):g_FBub_centroids[latestTime][ID]      for ID,latestTime in lastNStepsFrozenLatestTimes.items()}
        
        stuckRectParams = {**stuckRectParams,   **lastNStepsFrozenRectParams}
        stuckAreas      = {**stuckAreas,        **lastNStepsFrozenHullAreas}
        stuckCentroids  = {**stuckCentroids,    **lastNStepsFrozenCentroids}

        fields = [stuckRectParams,fbStoreRectParams2,stuckAreas,fbStoreAreas2,stuckCentroids,fbStoreCentroids2]
        #fields = [stuckRectParams,fbStoreRectParams2,stuckAreas,fbStoreAreas2,stuckCentroids,fbStoreCentroids2,fbStoreCulprits,frozenIDs_old,allLastFrozenParamsIDs]
        frozenIDs,frozenIDsInfo = detectStuckBubs(*fields, l_contours, globalCounter, frozenLocal, relArea = 0.5, relDist = 3, maxAngle = 10, maxArea = 1000) # TODO breaks on higher relDist, probly not being split correctly
        print(f'frozenLocal:{frozenLocal}')

        #[cv2.drawContours( blank, l_contours, ID, cyclicColor(key), -1) for ID in cntrIDlist]    # IMSHOW
        # store frozen bubble info for this step using local old IDs. could use only _old field instead, but this is more consistent with other fields 10/02/23
        frozenOldGlobNewLoc = {}
        for oldLocalID, newLocalIDs,_,_,centroid in frozenIDsInfo: #["oldLocID: ", ", newLocID: ",", c-c dist: ",", rArea: ",", centroid: "]

            dataSets = [l_FBub_masks,l_FBub_images,l_FBub_old_new_IDs,l_FBub_rect_parms,l_FBub_centroids,l_FBub_areas_hull,l_FBub_contours_hull]
            tempStore2(newLocalIDs, l_contours, oldLocalID, err, orig, dataSets, concave = 0)
            
            #findOldGlobIds = []; [findOldGlobIds.append(globID) for globID, locIDs in l_old_new_IDs_old.items() if oldLocalID in locIDs]
            if type(oldLocalID) == str:                                                                # if old ID already has global ID (str -> global, int -> local)
                if int(oldLocalID) in l_old_new_IDs_old or oldLocalID in lastNStepsFrozenRectParams:   # in case oldLocalID is old Else bubble ~='7' 11/02/23 or frozen from N step back 18/02/23
                    #findOldGlobIds.append(oldLocalID)
                    frozenOldGlobNewLoc[min(newLocalIDs)] = oldLocalID                                 # min(newLocalIDs) ! kind of correct, might cause problems 10/02/23/ modded 19/02/23
            else: frozenOldGlobNewLoc[min(newLocalIDs)] = oldLocalID                                   # == if it was single local, then it should have a global, if its part of old cluster, it must get new ID 19/02/23 
            #if type(oldLocalID) == str and int(oldLocalID) in l_old_new_IDs_old: findOldGlobIds.append(oldLocalID) # in case oldLocalID is old Else bubble ~='7' 11/02/23
            #if type(oldLocalID) == str and oldLocalID in lastNStepsFrozenRectParams: findOldGlobIds.append(oldLocalID) # frozen from N step back 18/02/23
            
        
            #frozenKeys = list(frozenOldGlobNewLoc.values()) wrong 11/02/23
            # !!! detectStuckBubs -> centroidAreaSumPermutations when permutation search fails returns empty array. it fucks shit up
        # !!! centroidAreaSumPermutations is used elsewhere, i think there its dealth with. 
        print(f'frozenIDs:{frozenIDs}')
        remIDs = [];print(f'remIDs:{remIDs}')
        for i, a in enumerate(frozenIDs): 
            if type(a) == list:
                if len(a) == 0 :
                    print("asdasdsadsad!!!!!!!!!!!!!!!!!!!!!")
                    remIDs.append(i)
        #frozenIDs = [a for a in frozenIDs if type(a) != list ]
        print(f'remIDs:{remIDs}')
        frozenIDs = [a for i,a in enumerate(frozenIDs) if i not in remIDs ] #
        frozenIDsInfo = np.array([a for i,a in enumerate(frozenIDsInfo) if i not in remIDs ], dtype=object)
        print(f'frozenIDs:{frozenIDs}')

        if  1 == 1:
            strRavel = ["oldLocID: ", ", newLocID: ",", c-c dist: ",", rArea: ",", centroid: "]
            formats = ["{}", "{}", "{:.2f}", "{:.2f}", "{}"]
            frozenIDsInfoFMRT = [[frmt.format(nbr) for nbr,frmt in zip(elem, formats)] for elem in frozenIDsInfo] if len(frozenIDsInfo)> 0 else []
            peps = ["".join(np.ravel([strRavel,k], order='F')) for k in frozenIDsInfoFMRT] if len(frozenIDsInfoFMRT)> 0 else []
            peps = "\n".join(peps) if len(frozenIDsInfo)> 0 else []
            print(f'Detected Frozen bubbles:\n{peps}')
        print(f'{globalCounter}:------------- Frozen bubbles detected ------------\n') if len(frozenIDsInfo) > 0 else print(f'{globalCounter}:------------- No frozen bubbles found ------------\n')
        
        #for IDS in [bID for bID,bType in l_bubble_type_old.items() if bType == typeFrozen]:
        #    l_predict_displacement[IDS] = [tuple(map(int,stuckCentroids[str(IDS)])), -1]
    
        inletIDs, inletIDsType  = overlappingRotatedRectangles(l_rect_parms_old, fakeBox, returnType = 1)           # inletIDs: which old IDs are close to inlet? 
        inletIDs                = [a[0] for a in inletIDs if a[0] not in lastStepFrozenGlobIDs]                     # inletIDsType: 1 & 2 - partially & fully inside.
        inletIDsType            = {ID:inletIDsType[ID] for ID in inletIDs} 
  
    unresolvedOldRB   = list(l_RBub_centroids_old.keys())     # list of global IDs for RB  
    unresolvedOldDB   = list(l_DBub_centroids_old.keys())     # and DB, from previous step. empty if gc = 0
    unresolvedNewRB   = list(whereChildrenAreaFiltered.keys())
 
    print("unresolvedNewRB (new typeRing bubbles):\n",  unresolvedNewRB)
    print("unresolvedOldRB (all old R,rR bubbles):\n",  unresolvedOldRB)
    print("unresolvedOldRB (all old R,rR bubbles):\n",  unresolvedOldDB)
    # 1.1 ------------- GRAB RING BUBBLES RB ----------------------
    # -----------------recover old-new ring bubble relations--------------------------

    RBOldNewDist = np.empty((0,2), np.uint16)
            
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
        cntrRemainingIDs = [cID for cID in whereParentOriginal if cID not in contoursFilter_RectParams_dropIDs] #+ dropAllRings + dropRestoredIDs
        
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
            inletIDs                = [a[0] for a in inletIDs if a[0] not in lastStepFrozenGlobIDs]                     # inletIDsType: 1 & 2 - partially & fully inside.
            inletIDsType            = {ID:inletIDsType[ID] for ID in inletIDs}                                                                                                            # note: if edge is shared, then its not fully inside. added 1 pix offset for edge of image
                                                           

            blank = err.copy()
            x,y,w,h = fakeBox[-1]
            cv2.rectangle(blank, (x,y), (x+w,y+h),220,1)
            [cv2.drawContours(  blank,   l_contours, cid, 255, 1) for cid in cntrRemainingIDs]
            fontScale = 0.7;thickness = 4;
            [[cv2.putText(blank, str(cid), l_contours[cid][0][0]-[25,0], font, fontScale, clr, s, cv2.LINE_AA) for s, clr in zip([thickness,1],[255,0])] for cid in cntrRemainingIDs]
            cv2.putText(blank, str(globalCounter), (25,25), font, 0.9, (255,220,195),2, cv2.LINE_AA)
            cv2.imwrite(".\\imageMainFolder_output\\rawContours\\"+str(dataStart+globalCounter).zfill(4)+".png" ,blank)
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
        cc_unique = graphUniqueComponents(cntrRemainingIDs, combosSelf, dOldNewAll, 0, err.shape,  l_Centroids, l_centroids_old, l_contours, l_contours_hull_old)

        print('Connected Components unique:', cc_unique) if debugOnly(21) else 0

        # forzen bubs (FB) can be falsely joint for being too close. or it can be false positive FB
            
        #------------------------------------------------------------------------------- deal with  frozen bubbles!! ----------------------------------------------
        # Drop frozen IDs from clusters, add as standalone objects.
        frozenSeparated = False
        allFrozenLocalIDs = sum(frozenIDs,[])
        temp = []
        for subTemp in cc_unique:
            bfr = []
            for elem in subTemp:
                if elem not in allFrozenLocalIDs:bfr.append(elem)
                else: frozenSeparated = True
            temp.append(bfr)
        cc_unique = temp
        frozenIDs = np.array([a if type(a) != list else min(a) for a in frozenIDs])
        # why [42]-> 42? idk 10/02/23
        #if len(frozenIDs)>0: frozenIDsInfo[:,1] = np.array([a if type(a) != list else min(a) for a in frozenIDsInfo[:,1]])
        #frozenIDsInfo = np.array([a if type(a[1]) != list else [a[0]]+[min(a[1])]+[a[2:]] for a in frozenIDsInfo])

        jointNeighbors = {min(elem): elem for elem in cc_unique if len(elem)>0}
        clusterMsg = 'Cluster groups: ' if frozenSeparated == False else f'Cluster groups (frozenIDs:{frozenIDs} detected): '
        print(clusterMsg,"\n", list(jointNeighbors.values()))
        # collect multiple contours into one object, store it for further check
        # on first iteration there are no more checks to be done. store it early
        if globalCounter == 0:
            blank  = np.zeros(err.shape,np.uint8)
            blank = convertGray2RGB(blank)                                                              # IMSHOW
            
            # 
            if  markFirstMaskManually == 1 and os.path.exists("./manualMask/frame"+str(dataStart).zfill(4)+" - Copy.png"):
                # *EDITED* not dropping----i am dropping RBs out of search, assuming they behave well. i do it to avoid doing it after.
                cntrRemainingIDsMOD = [cID for cID in whereParentOriginal if cID not in  contoursFilter_RectParams_dropIDs ]
                cntrRemainingMOD = {ID:l_contours[ID] for ID in cntrRemainingIDsMOD}

                contoursMain, group1Params = extractManualMask()
                group2Params = {ID:cv2.boundingRect(c) for ID,c in cntrRemainingMOD.items()}
                blank = orig.copy()

                # First approximation. group objects by their bounding box intersection.
                #its fast way to discard combinations that are far away
                combos = np.array(overlappingRotatedRectangles(group1Params,group2Params))
                mainIDs = combos[:,0]
                mainUniques = np.unique(mainIDs, return_counts=False)
                # group in library by main ID : {1:[0,1],2:[2,3,4],3:[5]}
                # THIS DOES WORKS ONLY ONE WAY. MAIN-> Secondary. copies may be expected. but should be ok for only first frame.
                groupedByMain = {singleUniq:[b for a,b in combos if a == singleUniq] for singleUniq in mainUniques}
                print(groupedByMain)
                # Second appox, more refined. check if secondary contour enters main contour. 

                for mainID in mainUniques:
                    x,y,w,h                         = group1Params[mainID]
                    mainMask                        = np.zeros((h,w),np.uint8)
                    cv2.drawContours( mainMask, contoursMain, mainID, 255, -1, offset = (-x,-y))
                    secondaryCntrs                  = groupedByMain[mainID]
                    (tempMask, [xt,yt,wt,ht], overlapingContourIDList) = \
                    overlapingContours(l_contours, secondaryCntrs, mainMask,
                                       (x,y,w,h), 0, prefix = f'{mainID}')
                    
                    dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    tempStore2(overlapingContourIDList, l_contours, min(overlapingContourIDList), err, orig, dataSets, concave = 0)
                    
                
                
                
            else:
                for key, cntrIDlist in jointNeighbors.items():
                    dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    tempStore2(cntrIDlist, l_contours, min(cntrIDlist), err, orig, dataSets, concave = 0)

                                    
                    
        # =============================== S E C T I O N - -  0 1 ==================================       
        # ------------------ RECOVER UNRESOLVED BUBS VIA DISTANCE CLUSTERING ----------------------   
        # ------------------ join with unresolvedNewRB and search relations --------------------- 
        # jointNeighbors is a rough estimate. In an easy case neighbor bubs will be far enough away
        # so clusters wont overlap. rather strict area/centroid displ restrictions can be satisfied
        # if it fails, take overlapped cluster IDs and start doing permutations and check dist/areas
        if globalCounter >= 1: 
            elseOldNewDoubleCriterium           = []
            elseOldNewDoubleCriteriumSubIDs     = {}
            oldNewDB                            = []
            
            jointNeighborsWoFrozen              = {mainNewID: subNewIDs for mainNewID, subNewIDs in jointNeighbors.items() if mainNewID not in frozenIDs}   # drop frozen bubbles FB from clusters          e.g {15: [15, 18, 20, 22, 25, 30], 16: [16, 17], 27: [27]} (no frozen in this example)
            #jointNeighborsWoFrozen              = {**jointNeighborsWoFrozen,**{ID:[ID] for ID in unresolvedNewRB}}                                        # add new RBs to clusters, since methods merged e.g {15: [... same, no rb here
            jointNeighborsWoFrozen_hulls        = {ID: cv2.convexHull(np.vstack(l_contours[subNewIDs])) for ID, subNewIDs in jointNeighborsWoFrozen.items()}# pre-calculate hulls (dims [N, 1, 2]),         e.g {15: array([[[1138,  488]...ype=int32), 16: array([[[395, 550]],...ype=int32), ...}
            jointNeighborsWoFrozen_bound_rect   = {ID: cv2.boundingRect(hull) for ID, hull in jointNeighborsWoFrozen_hulls.items()}                         # bounding rectangles,                          e.g {15: (857, 422, 282, 136), 16: (305, 495, 91, 98), 27: (648, 444, 164, 205)}
            jointNeighborsWoFrozen_c_a          = {ID: getCentroidPosContours(bodyCntrs = [hull]) for ID, hull in jointNeighborsWoFrozen_hulls.items()}     # centrouid and areas for a perfect match test  e.g {15: ((1009, 496), 28955), 16: ((...), 6829), 27: ((...), 21431)}
            
            resolvedFrozenGlobalIDs         = [int(elem) for elem in frozenOldGlobNewLoc.values() if type(elem) == str]                                     # frozen global IDs are not recovered           e.g []
            oldDistanceCentroidsWoFrozen    = {key:val for key,val in l_DBub_centroids_old.items()                                                          # drop frozens. not sure what is the difference
                                               if key not in list(l_FBub_old_new_IDs_old.keys()) + resolvedFrozenGlobalIDs}                                 # between two lists                             e.g  {1: (366, 539), 2: (927, 505), 3: (741, 539), 4: (1080, 486), 7: (1204, 446)}

            oldDBubAndUnresolvedOldRB       = {**oldDistanceCentroidsWoFrozen,**{ID:l_RBub_centroids_old[ID] for ID in unresolvedOldRB}}                      # merge old DB with all old RBs. since 
                                                                                                                                                            # old-newRB method was merged into this

            print(f'recovering DB bubbles :{list(oldDistanceCentroidsWoFrozen.keys())}') if len(unresolvedOldRB) == 0 else print(f'recovering DB+RB bubbles :{list(oldDBubAndUnresolvedOldRB.keys())}')
            if len(inletIDs)>0:  print(f'except old inlet bubbles: {inletIDs}')
            srtF = lambda x : l_bubble_type_old[x[0]]                                                                                                        # sort by type number-> RBs to front.
            oldDBubAndUnresolvedOldRB = dict(sorted(oldDBubAndUnresolvedOldRB.items(), key=srtF))                                                            # given low typeIDs are virtually important. 

            oldDBubAndUnresolvedOldRB = {ID:a for ID,a in oldDBubAndUnresolvedOldRB.items()}                            # moved if ID not in inletIDsType or (ID in inletIDsType and inletIDsType[ID] < 2) inside loop
            for oldID, oldCentroid in oldDBubAndUnresolvedOldRB.items():                                                # recovering RB, rRB, DB, rDB
                oldType         = l_bubble_type_old[oldID]                                                              # grab old ID bubType                                                   e.g 3  (typeElse)
                if oldType == typeRing or oldType == typeRecoveredRing:                                                 # RB type inheritance should be fragile.
                    newType     = typeRecoveredElse if oldType == typeRecoveredRing else typeRecoveredRing              # recovered RB to rRB, recovered rRB to rDB. for safety...
                else: newType   = typeElse                                                                              # if not (r)RB, then DB. rDB lost its meaning here. maybe on final matchTemplates()

                # ===== GET STATISTICAl DATA ON OLD BUBBLE: PREDICT CENTROID, AREA =====
                trajectory                                          = list(g_Centroids[oldID].values())                 # all previous path for oldID                                           e.g [(411, 547), (402, 545), (390, 543), (378, 542), (366, 539)]
                predictCentroid_old, _, distCheck2, distCheck2Sigma = g_predict_displacement[oldID][globalCounter-1]    # g_predict_displacement is an evaluation on how good predictor works.  e.g [(366, 540), 1, 10, 6]
                                                                                                                        # [old predicted value (used only for debug), old predictor error, mean error, stdev] 
                                                                                                                        # you can see than last known position was (366, 539), and it was 
                tempArgs                                            = [                                                 # predicted to be at (366, 540), which is an error of 1. 
                                                                        [distCheck2,distCheck2Sigma],                   # dist criterium
                                                                        g_predict_displacement[oldID],5,3,2,            # dist prediction history, # of stdevs, max steps for interp, max interp order
                                                                        debugVecPredict,predictVectorPathFolder,        # debug or not, debug save path
                                                                        predictCentroid_old,oldID,globalCounter,[-3,0]  # old centroid, timestep, ID for debug and zeroth displacement. (not used?)
                                                                      ]

                predictCentroid                                     = distStatPredictionVect2(trajectory, *tempArgs)    # predicted centroid for this frame, based on previous history.         e.g array([354, 537])
                
                oldMeanArea                                         = g_predict_area_hull[oldID][globalCounter-1][1]    # mean area                                                             e.g 6591
                oldAreaStd                                          = g_predict_area_hull[oldID][globalCounter-1][2]    # mean area                                                             e.g 620
                areaCheck                                           = oldMeanArea + 3*oldAreaStd                        # top limit                                                             e.g 8451
                l_predict_displacement[oldID]                       = [tuple(map(int,predictCentroid)), -1]             # predictor error will be stored here. acts as buffer storage for predC e.g [(354, 537), -1]
                
                if (oldID in inletIDsType and inletIDsType[oldID] == 2):                                                # i consider bubbles in fully inlet zone. so i get estimate in l_predict_displacement
                    continue                                                                                            # but then i skip this ID and process it in spaghetti code for inlet.
                # ===== REFINE TEST TARGETS BASED ON PROXIMITY =====
                overlapIDs = np.array(overlappingRotatedRectangles(
                                                                    {oldID:l_rect_parms_old[oldID]},                    # check rough overlap with clusters.
                                                                    jointNeighborsWoFrozen_bound_rect),int)             # this drops IDs that are out of reach. [old globalID, cluster localID] e.g array([[ 1, 16]])
                overlapIDs = {new: jointNeighborsWoFrozen[new] for _, new in overlapIDs}                                # reshape to form [localID, subIDs]                                     e.g {16: [16, 17]}

                if oldType == typeRing:
                    distCheck2Sigma += 1

                # ===== TEST A CLUSTER ASSUMING ALL SUBELEMENTS ARE CORRECT (PERFECT MATCH TEST) =====
                clusterElementFailed        = {ID:False for ID in overlapIDs}                                           # track if whole cluster is a solution                                  e.g {16: False}
                clusterCritValues           = {}                                                                        # if cluster generally fails, store criterium values here, for info     e.g {}
                
                for mainNewID, subNewIDs in overlapIDs.items():                                                         # test only overlapping clusters

                    newCentroid, newArea    = jointNeighborsWoFrozen_c_a[mainNewID]                                     # retrieve pre-computed cluster centroid and hull area                  e.g ((353, 539), 6829)
                    dist2                   = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))         # find distance between (cluster and predicted) centroids               e.g 2.2360 (predicted array([354, 537]))
                    areaCrit                = np.abs(newArea-oldMeanArea)/ oldAreaStd                                   # calculate area difference in terms of stdev                           e.g 0.3838 (old mean 6591, old stdev 620)
                    
                    if dist2 <= distCheck2 + 5*distCheck2Sigma and areaCrit < 3:                                        # test if criteria within acceptable limits                             e.g dist vs 10+5*6  ( which is kind of alot)      
                        
                        tempID                         = oldID                                                          # l_RBub_r and l_DBub_r have global IDs.
                        hull                           = jointNeighborsWoFrozen_hulls[mainNewID]                        # retrieve pre-computed hull
                        l_predict_displacement[oldID]  = [tuple(map(int,predictCentroid)), int(dist2)]                  # store prediction error
                        
                        newRBinTemp = [A for A in unresolvedNewRB if A in subNewIDs]                                  # all RB types in cluster. im not sure if ever works, since RBs are alone

                        if len(newRBinTemp) > 0:                                                                        # if there are RBs in solution
                            unresolvedNewRB   = [ ID for  ID in unresolvedNewRB if ID not in newRBinTemp]           # drop these RBs from new found list


                        overlapWORB = [a for a in subNewIDs if a not in newRBinTemp]                                    # drop RBs from solution. subNewIDs always has entries.
                                                                                                                        # empty overlapWORB means subNewIDs consited only of RB
                        if len(overlapWORB) == 0:                                                                       # if solution consists only of RB
                            dataSets        = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                            newType         = typeRing      #(?)                                                        # change default to RB, keep data in regular RB storage.
                                                                                                                        # any relevant type casts to RB if solution is RB only.
                            RBOldNewDist    = np.append(RBOldNewDist,[[oldID,newRBinTemp[0]]],axis=0)                   # store RB solution in oldX-newRB relations
                            tempID          = min(subNewIDs)                                                            # data is stored via local IDs
                                                                                
                        elif newType        == typeRecoveredRing:                                                                                                 
                            dataSets        = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]

                        elif newType        == typeRecoveredElse:                                                       # rDB is being cast only from rRB                         
                            dataSets        = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                            print(f'-{oldID}:Swapping bubble type to {typeStrFromTypeID[newType]} because it was recovered second time.')
                        
                        else:
                            dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                            tempID          = min(subNewIDs)                                                            # data is stored via local IDs
                            elseOldNewDoubleCriteriumSubIDs[tempID]  = subNewIDs                                        # add to oldDB new DB relations
                            elseOldNewDoubleCriterium.append([oldID,tempID,np.around(dist2,2),np.around(areaCrit,2)])   # some stats.
                            
                        tempStore2(subNewIDs, l_contours, tempID, err, orig, dataSets, concave = hull)
                        unresolvedOldRB.remove(oldID) if oldID in unresolvedOldRB   else 0                                  # remove from unresolved IDs
                        unresolvedOldDB.remove(oldID)  if oldID in unresolvedOldDB    else 0                                  # remove from unresolved IDs
                        print(f'-{oldID}:Resolved (first match) old{typeStrFromTypeID[oldType]}-new{typeStrFromTypeID[newType]}: {oldID} & {mainNewID}:{subNewIDs}.')

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
                
                if all(clusterElementFailed.values()):                                                                  # if all cluster matches have failed
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
                        dddd = 0 #if not (globalCounter == 62 and oldID == 13) else 1
                        permIDsol2, permDist2, permRelArea2, newCentroid, newArea, hull, newParams = radialAnal( [[OGLeft,OGRight], OGDistr] ,
                                                       [isEllipse,l_contours,subNewIDs,predictCentroid, ellipseParams, cvr2, err],
                                                       [l_rect_parms_all,l_rect_parms_old,l_Centroids,l_Areas] ,
                                                       [distCheck2, distCheck2Sigma, areaCheck, oldMeanArea, oldAreaStd, clusterCritValues],
                                                       globalCounter, oldID, mainNewID, l_MBub_info_old, debug = dddd)
                        
                        if len(permIDsol2)>0 and permDist2 < distCheck2 +5* distCheck2Sigma and permRelArea2 < 5:       # check if given solution satisfies criterium
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
                                newType         = typeRing                                                              # change default to RB, keep data in regular RB storage.
                                RBOldNewDist    = np.append(RBOldNewDist,[[oldID,newRBinTemp[0]]],axis=0)               # any relevant type casts to RB if solution is RB only.
                                tempID          = min(permIDsol2)                                                        # store RB solution in oldX-newRB relations
                                                                                                                        # data is stored via local IDs
                            elif newType        == typeRecoveredRing:                                                                         
                                dataSets        = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                                                        
                            elif newType        == typeRecoveredElse:                                                   # rDB is being cast only from rRB  
                                dataSets        = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                                print(f'-{oldID}:Swapping bubble type to {typeStrFromTypeID[newType]} because it was recovered second time.')
                        
                            else:
                                dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                                tempID          = min(permIDsol2)                                                                   # data is stored via local IDs
                                elseOldNewDoubleCriteriumSubIDs[tempID]  = permIDsol2                                               # add to oldDB new DB relations
                                elseOldNewDoubleCriterium.append([oldID,tempID,np.around(permDist2,2),np.around(permRelArea2,2)])   # some stats.
                            
                            tempStore2(permIDsol2, l_contours, tempID, err, orig, dataSets, concave = hull)

                            #if oldType == typeRing or oldType == typeRecoveredRing: unresolvedOldRB.remove(oldID)
                            
                            unresolvedOldRB.remove(oldID)   if oldID in unresolvedOldRB     else 0                              # remove from unresolved IDs
                            unresolvedOldDB.remove(oldID)   if oldID in unresolvedOldDB     else 0                              # remove from unresolved IDs
                             
                            print(f'-{oldID}:Resolved (via permutations of {subNewIDs}) old{typeStrFromTypeID[oldType]}-new{typeStrFromTypeID[newType]}: {oldID} & {permIDsol2}.')
                            
                        else:
                            print(f'-{oldID}:Recovery of oldDB {oldID} via permutations of {subNewIDs} has failed. Solution: {permIDsol2}.')
                            
                        print(f'--Distance prediction error: {permDist2:0.1f} vs {distCheck2:0.1f} +/- 5* {distCheck2Sigma:0.1f} and area criterium (dA/stev):{permRelArea2:0.2f} vs ~5\n')

            if len(elseOldNewDoubleCriterium)>0:
                #print(dropDoubleCritCopies(elseOldNewDoubleCriterium))
                elseOldNewDoubleCriterium = dropDoubleCritCopies(elseOldNewDoubleCriterium)         # i dont think its relevant anymore

            print('end of oldDB/oldRB recovery') 
            # modify jointNeighbors by splitting clusters using elseOldNewDoubleCriteriumSubIDs# l_DBub_old_new_IDs
            # take recovered local IDs: DB, rDB, rRB
            removeFoundSubIDsAll    = sum(l_DBub_old_new_IDs.values(),[]) + sum(l_MBub_old_new_IDs.values(),[]) + sum(l_RBub_r_old_new_IDs.values(),[]) + sum(l_DBub_r_old_new_IDs.values(),[])
            # go though subclusters and delete resolved local IDs. drop recovered old-newRB ( min(elem) is wrong code, have to chech if its single elem)
            jointNeighborsDelSubIDs = [ [subelem for subelem in elem if subelem not in removeFoundSubIDsAll] for elem in jointNeighborsWoFrozen.values() if min(elem) not in RBOldNewDist[:,1]]
            jointNeighbors_old      = dict(sorted(jointNeighbors.copy().items()))
            # recombine resolved clusters and filtered cluster. no resolved RB is included in new jointNeighbors
            jointNeighbors          = {**{min(vals):vals for vals in jointNeighborsDelSubIDs if len(vals) > 0},**elseOldNewDoubleCriteriumSubIDs}
            jointNeighbors          = dict(sorted(jointNeighbors.items()))
            print(f'Cluster groups (updated): {jointNeighbors}') if jointNeighbors != jointNeighbors_old else print('Cluster groups unchanged')
            
            
            
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
            
            if len(l_MBub_centroids_old)>0:
                leftOverIDs         = sum(list(jointNeighbors.values()),[])                                 # from history- merged bubbles can consist of new unresolved RBs
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
                
                if len(leftOver_overlap_new)  > 0:

                    # ===== to avoind using concave hull, rough estimate can be made using  convex hull =====
                    oldAreaHull = cv2.contourArea(cv2.convexHull(l_contours_hull_old[oldID], returnPoints = True))                      # we take a convex hull or old concave hull
                    permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(l_contours,l_rect_parms_all, l_rect_parms_old[oldID], toList(leftOver_overlap_new), l_Centroids, l_Areas,
                                                    predictCentroid, oldAreaHull*0.15, oldAreaHull, relAreaCheck = 0.7, debug = 0)      # find permutation of subIDs that result in a better
                                                                                                                                        #  match with old convex hull area and predicted centroid
                    if len(permIDsol2)>0:                                                                                               # if there is a non-trivial match 
                        mainNewID               = 512512512#min(permIDsol2)
                        #subNewIDs               = permIDsol2
                        previousInfo            = l_MBub_info_old[oldID][3:]      #[0.4*rads,tcc,25]
                        #debugs = 0 if globalCounter != 16 else 1
                        debugs = 0 #if globalCounter != 7 else 1 #if globalCounter < 21 else 1
                        hull, newParams, mCrit  = mergeCrit(permIDsol2, l_contours, previousInfo, alphaParam = 0.05, debug = debugs,debug2 = 0, gc = globalCounter, id = oldID)    # calculated modified hull
                       
                        #if globalCounter == 7:
                        #    blank = err.copy()
                        #    cv2.drawContours(  blank,   [hull], -1, 120, 2)
                        #    cv2.imshow('ads',blank)
                        newCentroid, newArea    = getCentroidPosContours(bodyCntrs = [hull])                                            # centroid and hull of a modified hull
                        dist2                   = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))                     # predictor error
                        areaCrit                = np.abs(newArea-oldMeanArea)/ oldAreaStd                                               # area diffference in terms of stdevs
                        
                        if len(permIDsol2) > 1 and mCrit[0] == True and mCrit[1].shape[0] > 0:
                            direction = np.matmul(np.array([[0, -1],[1, 0]]), newParams[1])
                            split, subIDs1, subIDs2 = mergeSplitDetect(l_contours,permIDsol2,direction,mCrit[1],globalCounter,oldID, debug = 0)
                            #if split == True:
                            #    [unresolvedNewRB.remove(x) for x in unresolvedNewRB if x in subIDs1 + subIDs2]

                        if split == False and dist2 <= distCheck2 + 5*distCheck2Sigma and areaCrit < 3:                                                    # a perfect match

                            if mCrit[0] == True:
                                dataSets  = [l_MBub_masks, l_MBub_images,l_MBub_old_new_IDs, l_MBub_rect_parms, l_MBub_centroids, l_MBub_areas_hull, l_MBub_contours_hull]
                                l_MBub_info[oldID]                      = [[],newArea,newCentroid] + list(newParams)                        # store new merge params
                                tempID          = oldID
                            
                            elif len(permIDsol2) == 1 and permIDsol2[0] in unresolvedNewRB:
                                tempID          = min(permIDsol2) 
                                dataSets        = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
                                RBOldNewDist    = np.append(RBOldNewDist,[[oldID,tempID]],axis=0)               
                                                                                       # store RB solution in oldX-newRB relations
                            else:
                                tempID          = min(permIDsol2)                                                                   # data is stored via local IDs
                                
                                dataSets  = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                                elseOldNewDoubleCriteriumSubIDs[tempID]  = permIDsol2                                               # add to oldDB new DB relations
                                elseOldNewDoubleCriterium.append([oldID,tempID,np.around(dist2,2),np.around(areaCrit,2)]) 

                            tempStore2(permIDsol2, l_contours, tempID, err, orig, dataSets, concave = hull)                               # store a solution

                            l_predict_displacement[oldID]            = [tuple(map(int,predictCentroid)), int(dist2)]                    # store predictor error
                            
                            delResolvedMB.append(list(permIDsol2))                                                                      # store merge subIDs
                            [unresolvedNewRB.remove(x) for x in unresolvedNewRB if x in permIDsol2]                                 # remove unresolved RBs in case they are in solution

                            print(f'-{oldID}:Resolved (via permutations of {leftOver_overlap_new}) oldMB-newMB: {oldID} & {permIDsol2}.')
                        else:
                            print(f'-{oldID}:Recovery of oldDB {oldID} via permutations of {leftOver_overlap_new} has failed. Solution: {permIDsol2}.')
                        print(f'--Distance prediction error: {dist2:0.1f} vs {distCheck2:0.1f} +/- 2* {distCheck2Sigma:0.1f} and area criterium (dA/stev):{areaCrit:0.2f} vs ~5\n')
                # === finalizing DB clusters ===

                if split == True:
                    for subIDs in [subIDs1, subIDs2]:
                        jointNeighbors = {ID:[subID for subID in vals if subID not in subIDs ] for ID,vals in jointNeighbors.items()}
                        #subIDsWoRBs     = [a for a in subIDs if a not in unresolvedNewRB]                                                 # drop unresolved RBs from subcluster copy
                        #if len(subIDs) < len(subIDsWoRBs) and len(subIDsWoRBs) > 0:                                                         # if subcluster copy got smaller, then there was RB inside.
                        #    dataSets        = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                        #    tempStore2(subIDs, l_contours, min(subIDs), err, orig, dataSets, concave = 0)                                    # len(subIDsWoRBs) == 0 would mean that there was only one RB inside
                        #    [unresolvedNewRB.remove(x) for x in unresolvedNewRB if x in subIDs]                                         # but that one RB should be kept in unresolvedNewRB, so it forms a new RB
                        #    jointNeighbors = {ID:[subID for subID in vals if subID not in subIDs ] for ID,vals in jointNeighbors.items()}
                    jointNeighbors = {**{min(vals):vals for vals in jointNeighbors.values() if len(vals) > 0},**{min(sub):sub for sub in [subIDs1, subIDs2] }}
                    
            print('end of oldMB recovery')             

            # --- resolved MB should be deleted from clusters alltogether ---
                      
            removeFoundSubIDsAll    = sum(delResolvedMB,[])                                                                                         # get all subIDs of recovered MBs
            jointNeighborsDelSubIDs = []                                                                                                            # 
            jointNeighborsDelSubIDs = [ [subelem for subelem in elem if subelem not in removeFoundSubIDsAll] for elem in jointNeighbors.values()]   # get all cluster subIDs that are not in removeFoundSubIDsAll
            jointNeighbors          = {**{min(vals):vals for vals in jointNeighborsDelSubIDs if len(vals) > 0},**elseOldNewDoubleCriteriumSubIDs}   # not clear. same 2 lists overlayed
            # --- cluster have twice been recombined ( recover & merge-split), so some connectivity might be lost ---
            # --- must do overlap within clusters and then clusters must be devided based on connectivity --- *** pre delResolvedMB jointNeighbors have holes, might be problematic ***
            print(f'Reclustering jointNeighbors in case of discontinueties due to recovery. old Clusters: {jointNeighbors}')
            bigClusterIDs           = [ID for ID,subIDs in jointNeighbors.items() if len(subIDs)>1]
            for ID in bigClusterIDs:
                rectParams = {subID:l_rect_parms_all[subID] for subID in jointNeighbors[ID]}
                combosSelf = np.array(overlappingRotatedRectangles(rectParams,rectParams))
                cc_unique  = graphUniqueComponents(jointNeighbors[ID],combosSelf, edgesAux= dOldNewAll, debug = 0, bgDims = {1e3,1e3}, centroids = [], centroidsAux = [], contours = [], contoursAux = [])
                jointNeighbors.pop(ID, None)
                for subIDs in cc_unique:
                    jointNeighbors[min(subIDs)] = subIDs
            print(f'Reclustering jointNeighbors in case of discontinueties due to recovery. new Clusters: {jointNeighbors}')
            
            centroidsResolved   = {**l_RBub_centroids,  **l_RBub_r_centroids,   **l_DBub_centroids,   **l_DBub_r_centroids,     **l_FBub_centroids,       **l_MBub_centroids}
            areasResolved       = {**l_RBub_areas_hull, **l_RBub_r_areas_hull,  **l_DBub_areas_hull,  **l_DBub_r_areas_hull,    **l_FBub_areas_hull,      **l_MBub_areas_hull}
            oldNewIdsResolved   = {**l_RBub_old_new_IDs,**l_RBub_r_old_new_IDs, **l_DBub_old_new_IDs, **l_DBub_r_old_new_IDs,   **l_FBub_old_new_IDs,     **l_MBub_old_new_IDs}
            rectParamsOld       = {oldID:l_rect_parms_old[oldID] for oldID in unresolvedOldDB + unresolvedOldRB }#if oldID not in inletIDs        # old global IDs with bounding rectangle parameters
        
            resolvedContoursNew     = sum(oldNewIdsResolved.values(),[])
            unresolvedInletContours = [a for a in inletIDsNew if a not in resolvedContoursNew]
            temp = {}
            oldInsideIDs = [oldID for oldID, cType in inletIDsType.items() if cType == 2]
            for ID,subIDs in jointNeighbors.items():                                        # determine which cluster is overlapping inlet recangle. might be done conentional way..
                match = [a in subIDs for a,tp in inletIDsTypeNew.items() if tp == 2]
                temp[ID] =  sum(match)                                                      # count matches of inlet new IDs in clusters
            clusterID    = max(temp, key=temp.get)                                          # best match wins !
            crit1               = temp[clusterID] > 0 and len(unresolvedInletContours) > 0         # drop resolved IDs from inlet cluster, if there are spare elements do your stuff. left part may be redundant
            crit2               = len([ID for ID,cType in inletIDsType.items() if cType == 2]) > 0 # fully inside bubbles are not resolved prior in principle. resolve them here
            if crit1 or crit2:# inlet cluster  maybe fully resolved by old-newDB, which happens if its poking outside inlet box.
                a = 1                                                                                   # so inletIDsNew = resolved subcluster.
                idsInside   = [ID for ID, cType in inletIDsTypeNew.items() if cType == 2]               # ids fully inside inlet rectangle. might include parts of unresolved neighbor clusters
                idsPartial  = [ID for ID, cType in inletIDsTypeNew.items() if cType == 1]               # ids that are partially inside inlet rectangle. again, might be part of unresolved clusters.
                overlappintPartialOldIDs    = [ID for ID,cType in inletIDsType.items() if cType == 1]   # there are partially overlaying old IDs
                overlappintUnresolvedOldIDs = [a for a in overlappintPartialOldIDs if a in unresolvedOldRB + unresolvedOldDB] # which of these are missing

                print(f'Processing bubble next to inlet. inside region: {idsInside}, on border: {idsPartial}, unresolved old IDs that overlap inet zone: {overlappintUnresolvedOldIDs}')
                #if len(overlappintUnresolvedOldIDs)==0 and : 
                #for oldID, tp in inletIDsType.items():
                #    if tp == 2 and len(idsInside)>0:
                overlapNew      = []                                                    # see if else below
                if max(list(temp.values())) == 0:
                    assert len(idsPartial) <= 1, 'inlet kuku'
                    clusterID = [min(a) for a in jointNeighbors.values() if idsPartial[0] in a][0]
                subIDs          = jointNeighbors[clusterID].copy()                      # take IDs of elements in relevant cluster
                includeIDs      = [ID for ID in subIDs if ID        in idsInside]       # extract only those inside. maybe i dont have do this step, but take idsInside. not sure.
                leaveIDs        = [ID for ID in subIDs if ID not    in idsInside]       # part of big cluster that is not fully inside.
                if len(overlappintUnresolvedOldIDs)==0:                                 # partial IDs could be part of unresolved cluster, like MB, which are determined next
                    includeIDs = includeIDs + [ID for ID in subIDs if ID in idsPartial] # or they can be part of inlet bubble. in case there are no missing bubbles, include them into IB
                    leaveIDs = []
                    print(f'no old bubbles in proximity, include border IDs into clusters: {includeIDs}')
                else:                                                                   # in case there is overlapping unresolved old ID, some contours fully in inlet rectangle
                    overlap     = overlappingRotatedRectangles(                         # can be part of it. 
                                {ID:l_rect_parms_old[ID] for ID in overlappintUnresolvedOldIDs},
                                {ID:l_rect_parms_all[ID] for ID in includeIDs}          # check intersection between old partially overlapping unresolved old bubble and 
                                                                )                       # objects fully  inside inlet rectangle.
                    overlapNew  = [a for _,a in overlap]                                # get local IDs of new object ids
                    includeIDs  = [a for a in includeIDs if a not in overlapNew]        # drop overlapNew IDs from fully inside
                    leaveIDs    = leaveIDs + overlapNew                                 # add overlapNew IDs to remaining unresolved cluster
                        
                    print(f'there are old bubbles in proximity, split inlet clusters into  {includeIDs} and {leaveIDs}')
                if len(includeIDs)>0:
                    includeArea = cv2.contourArea(cv2.convexHull(np.vstack(l_contours[includeIDs])))
                    includeIDs  = includeIDs if includeArea > 550 else []
                jointNeighbors.pop(clusterID, None)
                if len(includeIDs)>1:
                    neighborDists = closestDistancesContours({ID:l_contours[ID] for ID in includeIDs})  # get closest distances between contours. format: {ID1:np.array([[ID2,d12],[ID3,d13]]),ID2:...}
                    overlapNew = []
                    if len(oldInsideIDs) > 0:                                           # 
                        overlap     = overlappingRotatedRectangles(                     # want to check which inlet cluster elements overlap fully inside old. 
                                {ID:l_rect_parms_old[ID] for ID in oldInsideIDs},
                                {ID:l_rect_parms_all[ID] for ID in includeIDs}          
                                                                )
                        a=1 
                        overlapNew  = [a for _,a in overlap]                            # dont expect multiple oldInsideIDs. take overlapNew as a new base.
                        print(f'old , fully inside inlet region bubble present. setting inlet cluster base IDs to old-new overlap: {overlapNew}')
                        huhuh = np.array(sum([[[a,b] for a,b in neighborDists[ID] if a not in overlapNew] for ID in overlapNew],[]),int) # distances from all elements in overlapNew to non-overlapNew elements.
                        neighb  = np.empty((0,2), int) if huhuh.shape[0] == 0 else np.unique(huhuh[:,0])                           # IDs of neighbors
                        dists   = {ID:np.array([b for a,b in huhuh if a == ID], int) for ID in neighb}                             # distances : {IDx:[dix,djx,..]} where dix distance to neighbor ID = x from element i in overlapNew
                        IDfail2 = {ID:all(np.where(vals > 150,1,0)) for  ID,vals in dists.items()}                                  # if all dists are too large for IDx -> overlapNew, it should be separated
                        print(f'testing distances to nearby neighbors: dists IDi:[IDj, Dij]: {dists}; dist fails: {IDfail2}')
                        _, _, w0, _ = l_rect_parms_old[oldInsideIDs[0]]
                        _,_,w1,_ = cv2.boundingRect(np.vstack(l_contours[overlapNew]))
                        distElems = [ID for ID, failed in IDfail2.items() if failed == False]
                        testCluster = overlapNew + distElems
                        _,_,w2,_ = cv2.boundingRect(np.vstack(l_contours[testCluster]))
                        avgWidth = 0.5*(w0+w1)
                        #if avgWidth/fakeBoxW > 0.75*fakeBoxW
                        if w2/(max(avgWidth,0.3*fakeBoxW)) > 2.3:                      # smallwidth clusters can increase their size easier. limit smallest cluster width with 0.3*fakeBoxW
                            includeIDs = overlapNew
                            distanceIDs = distElems
                        else:
                            includeIDs  = overlapNew + [ID for ID, failed in IDfail2.items() if failed == False]
                            distanceIDs = [ID for ID, failed in IDfail2.items() if failed == True] 
                        #includeIDs  = overlapNew + [ID for ID, failed in IDfail2.items() if failed == False]  
                        #distanceIDs = [ID for ID, failed in IDfail2.items() if failed == True]   
                    else:
                        IDfail  = {ID:all(np.where(vals[:,1] > 150,1,0)) for  ID,vals in neighborDists.items()} # all distances are greater than threshold
                        
                        if len(IDfail) == 2 and IDfail[list(neighborDists.keys())[0]] == True:                # if two contours remaining and are far apart, both will fail, since d12=d21>threshold
                            areas = {ID:cv2.contourArea(l_contours[ID]) for ID in neighborDists.keys()}
                            includeIDs  = [max(areas, key=areas.get)]
                            distanceIDs = [min(areas, key=areas.get)]
                            if areas[includeIDs[0]]     < 550: includeIDs   = []                              # there are cases with two tiny contours
                            if areas[distanceIDs[0]]    < 550: distanceIDs  = []                              # 
    
                            print(f'checking for widely spaced elements in clusters. from two choices separating smallest: {distanceIDs}')
                        else:
                            includeIDs  = [ID for ID in includeIDs if IDfail[ID] == False]                       # keep inside contours that are reasonably clustered
                            distanceIDs = [ID for ID, failed in IDfail.items() if failed == True]               # and take out those that are too farm away from each other
                            print(f'checking for widely spaced elements in clusters. IDs that fail dist check {IDfail}')
                    areasPass   = [ID  for ID in distanceIDs if cv2.contourArea(l_contours[ID]) > 550]
                    if len(areasPass)<len(distanceIDs): print(f'dropping smaller distanceIDs')
                    distanceIDs = [ID for ID in distanceIDs if ID in areasPass]
                    
                else: distanceIDs = []
                    
                    
                if len(includeIDs)>0:                                                               
                    jointNeighbors  = {**jointNeighbors,**{min(includeIDs):includeIDs}}
                            # there is old ID fully inside inlet zone.
                    print(f'old ID inside inlet zone: {oldInsideIDs}')
                    aresOld         = {ID:l_areas_hull_old[ID] for ID in oldInsideIDs}
                    hull            = cv2.convexHull(np.vstack(l_contours[includeIDs]))
                    newID           = min(includeIDs)
                    dataSets        = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                    tempStore2(includeIDs, l_contours, newID, err, orig, dataSets, concave = hull)
                    
                    
                    if len(oldInsideIDs)>0:                                                               # if multiple olds inside, get one with largest area
                        oldID        = max(aresOld, key = aresOld.get)
                    
                        elseOldNewDoubleCriteriumSubIDs[newID]  = includeIDs                                               # add to oldDB new DB relations
                        elseOldNewDoubleCriterium.append([oldID, newID, 15, 1])   # some stats.
                            
                        l_predict_displacement[oldID]            = [tuple(map(int,l_DBub_centroids[newID])), 15]
                        unresolvedOldRB.remove(oldID)   if oldID in unresolvedOldRB     else 0                              # remove from unresolved IDs
                        unresolvedOldDB.remove(oldID)   if oldID in unresolvedOldDB     else 0                              # remove from unresolved IDs
                        print(f'relating inlet cluster to old ID: {oldID}')
                        print(f'Save inlet cluster: {includeIDs}, distance fail elements: {distanceIDs}, rest of cluster: {leaveIDs}, which might be a merge.')
                    else:
                        print(f'no old ID found, addin as new ID')
                        print(f'Save inlet cluster: {includeIDs}, distance fail elements: {distanceIDs}, rest of cluster: {leaveIDs}, which might be a merge.')
                if len(distanceIDs)>0:                                                               
                    jointNeighbors  = {**jointNeighbors,**{ID:[ID] for ID in distanceIDs}}          # add distance ones as single element clusters
                if len(leaveIDs)>0:
                    jointNeighbors  = {**jointNeighbors,**{min(leaveIDs):leaveIDs}}                 # rest of the cluster, not only partials
                    

            #----------------- consider else clusters are finalized ---------------------
            #tempJoinNeighbors = jointNeighbors.copy()
            #[tempJoinNeighbors.pop(min(key)) for key in frozenLocal] # dropped resolved local frozen IDs 10/02/23
            #for key, cntrIDlist in jointNeighbors.items():
            #    if key not in l_DBub_old_new_IDs:
            #        dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
            #        tempStore2(cntrIDlist, l_contours, key, err, orig, dataSets, concave = 0)
                
            oldNewDB = np.array([arr[:2] for arr in elseOldNewDoubleCriterium],int)  # array containing recovered DBs old-> new. its used to write global data to storage
            oldNewDB = oldNewDB if len(oldNewDB) > 0 else np.empty((0,2), np.uint16) 
            # first definition of unresolvedNewDB. consists of resolved DBs, unresolved cluster ids and unresolved RBs
            # holds main local IDs- means min(subIDs) e.g 17:[17,29,31]
            unresolvedNewDB = np.unique(list(jointNeighbors.keys()) + unresolvedNewRB) #  bad fix for not dropping unresolvedNewRB from merge-split
            
                
            #if debugOnly(21):
            #    #print("oldNewDB (Else IDs pass dist)\n", list(map(list,oldNewDB)))
            #    print(f"oldNewDB (Else IDs pass dist, /w subIDs)\n{[[oldID,jointNeighbors[mainNewID]] for oldID,mainNewID in oldNewDB]}")
            #    print("unresolvedNewDB (new Else IDs)\n",unresolvedNewDB)
            #    print("unresolvedOldDB (all old Else IDs)\n",unresolvedOldDB)
            
            # ==== clearing recovered DB bubbles ====
            if len(oldNewDB) > 0: # if all distance checks fail (oldNewDB)
                unresolvedNewDB = [A for A in jointNeighbors.keys() if A not in oldNewDB[:,1] and A not in frozenOldGlobNewLoc.keys()];   # drop resolved new DB from unresolved new DB
                unresolvedOldDB = [A for A in unresolvedOldDB if A not in oldNewDB[:,0] and A not in resolvedFrozenGlobalIDs];              # drop resolved old DB from unresolved old DB
                for ID in unresolvedNewDB.copy():
                     subIDs =  jointNeighbors[ID]
                     if len(subIDs) == 1 and ID in unresolvedNewRB:
                         unresolvedNewDB.remove(ID)
                #if debugOnly(21):
                #    print("unresolvedNewDB (unresolved new IDs)\n",unresolvedNewDB)
                #    print("unresolvedOldDB (unresolved old Else IDs)\n",unresolvedOldDB)
                
            # if RB is "recovered" from E, add it to RB data, remove from unresolved RB
            #elseToRingBubIDs = [i for i,[_,ID] in enumerate(oldNewDB) if ID in unresolvedNewRB] # if new local resolved DB id is a unresolved RB
            #unresolvedNewRB = [elem for elem in unresolvedNewRB if elem not in oldNewDB[:,1]] # drop that local unresolved RB ID
            #for i in elseToRingBubIDs:
            #    RBOldNewDist = np.vstack((RBOldNewDist,oldNewDB[i]))                              # swap old-newDB entry into old-newRB list
            #oldNewDB = np.array([ elem for i,elem in enumerate(oldNewDB) if i not in elseToRingBubIDs]) # remove swapped entry from old-newDB
            #oldNewDB = oldNewDB if len(oldNewDB) > 0 else np.empty((0,2), np.uint16) 
            #if debugOnly(21) == True and len(elseToRingBubIDs)>0:
            #    print(f'unresolvedNewRB, (found RB removed) unrecovered RBs: {unresolvedNewRB}')
            #    print("RBOldNewDist, ( new RB added) (distance relations < distCheck):\n",RBOldNewDist)
            #    print("oldNewDB, (found RB removed) (Else IDs pass dist)\n", oldNewDB)
    
               
            #if len(unresolvedOldDB)>0: print(f'{globalCounter}:--------- Begin recovery of Else bubbles: {unresolvedOldDB} --------')
            #else:                   print(f'{globalCounter}:----------------- No Else bubble to recover ---------------------')
                
            jointNeighbors = {ID:subIDs for ID,subIDs in jointNeighbors.items() if ID not in oldNewDB[:,1]} if len(oldNewDB)> 0 else jointNeighbors
            gfx = 1 if debugOnlyGFX(23) else 0
            if 1 == -1:
            #for i in unresolvedOldDB.copy(): 
                print(f'Trying to recover old else {i}')
                #gfx = 1 if globalCounter == 22 and i == 26 else 0
                #gfx = 1 if globalCounter == 12 else 0
                (x,y,w,h) = matchTemplateBub(err,l_masks_old[i],l_rect_parms_old[i],graphics=gfx,prefix = f'MT: Time: {globalCounter}, E_old_ID: {i} ')
                # some partial reflections change too much btwn frames and cannot be locatied as binary templates. try to search grayscale img if binary fails.
                cornerDist = np.linalg.norm(np.array(l_rect_parms_old[i][:2]) - np.array([x,y])).astype(np.uint32) # matchTemplateBub might give wrong results, finds sol-n somewhere deep in bounding rect.
                if cornerDist > 10:
                    print(f'cornerDist: {cornerDist} > 10 !!!')
                    blank = err.copy()*0
                    blank[err == 255] = orig[err == 255]
                    #cv2.imshow('asd',blank)
                    (x2,y2,w2,h2) = matchTemplateBub(blank,l_images_old[i],l_rect_parms_old[i],graphics=gfx,prefix = f'MT: Time: {globalCounter}, E_old_ID: {i}  #2')
                    cornerDist2 = np.linalg.norm(np.array(l_rect_parms_old[i][:2]) - np.array([x2,y2])).astype(np.uint32) 
                    if cornerDist2 > 10: print(f'cornerDist2: {cornerDist2} > 10 !!!')
                    if cornerDist2 < cornerDist: (x,y,w,h) = (x2,y2,w2,h2)
                (tempMask, [xt,yt,wt,ht], overlapingContourIDList) = \
                    overlapingContours(l_contours, whereParentOriginal, l_masks_old[i], (x,y,w,h), gfx, prefix = f'OC: Time: {globalCounter}, RB_old_ID: {i} ')
                overlapingContourIDListTemp = [ID for ID in overlapingContourIDList if ID not in frozenIDs] # remove frozen IDs
                if len(overlapingContourIDListTemp)>0:
                    mergeCandidatesSubcontourIDs[i] = overlapingContourIDListTemp
                    for localID in overlapingContourIDListTemp:                                         # mergeCandidates-> {localID:[oldGlobal1,oldGlobal2,..],..}
                        if localID not in mergeCandidates: mergeCandidates[localID]  = []
                        mergeCandidates[localID].append(i)
                baseSubMask,baseSubImage = err.copy()[yt:yt+ht, xt:xt+wt], orig.copy()[yt:yt+ht, xt:xt+wt]
                baseSubImage[tempMask == 0] = 0
                shapePass = compareMoments(big=err,
                                                shape1=baseSubImage,shape2=l_images_old[i],
                                                coords1=[xt,yt,wt,ht],coords2 = l_rect_parms_old[i],debug = debugOnly(22))
                print(f'recovery of {i} {overlapingContourIDList} completed') if shapePass == 1 else print(f'recovery of {i} {overlapingContourIDList} failed.')
                if shapePass == 1:
                    # account for frozenIDs: keep resolvedEBub, modify mask. frozenID will be left in unresolvedNewDB, if it stay there it will bw asigned typeFrozen
                    overlapingContourIDList     = [ID for ID in overlapingContourIDList if ID not in frozenIDs] 
                    dataSets = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                    tempStore2(overlapingContourIDList, l_contours, i, err, orig, dataSets, concave = 0)
                    
                    pCentr = l_predict_displacement[i][0]                       # pre computed
                    dist2 = np.linalg.norm(np.array(pCentr) - np.array(l_DBub_r_centroids[i])).astype(np.uint32)
                    l_predict_displacement[i] = [pCentr, dist2]
                   
                        
    
                    unresolvedOldDB.remove(i)
                    [[unresolvedNewRB.remove(ii) for ii in toList(inter)] for inter in np.intersect1d(overlapingContourIDList,unresolvedNewRB) if len(toList(inter))>0]
                    # print('overlapingContourIDList',overlapingContourIDList)
                    # resolveElseIDs - get main local cluster IDs which own contours in overlapingContourIDList
                    resolveElseIDs = set(sum([[ID for ID,subIDs in jointNeighbors.items() if foundID in subIDs] for foundID in overlapingContourIDList],[]))
                    unresolvedNewDB = [elem for elem in unresolvedNewDB if elem not in resolveElseIDs] # drop resolved
                    jointNeighbors2 = {A:[ID for ID in subIDs if ID not in overlapingContourIDList] for A,subIDs in jointNeighbors.items()}
                    jointNeighbors = {min(subIDs): subIDs for _,subIDs in jointNeighbors2.items() if len(subIDs) > 0}
                    if debugOnly(22):
                        print(f'resolveElseIDs {resolveElseIDs}')
                        print(f'jointNeighbors {jointNeighbors}')
                        print(f'jointNeighbors2 {jointNeighbors2}')
                        #print(f'jointNeighbors3 {jointNeighbors3}')
                        print(f'C: {globalCounter}, IDE: {i}, l_DBub_r_old_new_IDs[{i}]: {l_DBub_r_old_new_IDs[i]}')
                        print(f'Recovered old else {i}, remaining unresolvedOldDB: {unresolvedOldDB}')
                        print("unresolvedNewDB (unresolved  (updated) new IDs)\n",unresolvedNewDB) if len(resolveElseIDs)>0 else 0
        
            if len(jointNeighbors)> 0:
                for ID,cntrIDs in jointNeighbors.items():
                    if ID not in unresolvedNewRB:
                        dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull,l_DBub_contours_hull]
                        tempStore2(cntrIDs, l_contours, ID, err, orig, dataSets, concave = 0)
                    
                #unresolvedNewDB = unresolvedNewDB + [elem for elem in list(jointNeighbors.keys()) if elem not in unresolvedNewDB]
                        

    print(f'{globalCounter}:--------- Begin merge detection/processing --------\n')
        
    # ------------------- detect merges by inspecting shared contours ----------------------
    # OG idea was based on fact that if two or more bubbles from previous frame are not recovered and there are new unresolved bubbles, merge might have happened.
    # this case is tested at first part. now its modified
    if globalCounter >= 1 :#and len(unresolvedOldDB)>1 and len(jointNeighbors)>1
        cc_oldIDs, cc_newIDs = [], []
        # ======== PART 01: test if two old bubbles are unresolved and new is unresolved. might hint on merge =========
        rectParamsNew   = {subID:l_rect_parms_all[subID] for subID in sum(list(jointNeighbors.values()),[])}    # all unresolved cluster elements IDs and bound rect.
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
            cc_newIDs = [[elem for elem in vec if type(elem) != str] for vec in cc_unique]                  # extract new IDs by ID type
            cc_oldIDs = [[int(elem) for elem in vec if type(elem) == str] for vec in cc_unique]             # same with old


            #print(f'unresolvedNewRB {unresolvedNewRB}, unresolvedNewDB {unresolvedNewDB}')
            #print(f'unresolvedOldRB {unresolvedOldRB}, unresolvedOldDB {unresolvedOldDB}')
            print(f'new local IDs: {cc_newIDs} overlap with old global IDs: {cc_oldIDs}')

        # ======== PART 02: Big bubble merged with small. big is resolved due leeway in criterium ========
        # small old is missing, no new are missing. Target resolved old which has a history of 2+ steps. 
        # resolved centroid will be offset from predicted in derection of missing bubble.
        # if you were to consider total center of mass, of correct combination, it would be closer to resolved centroid.
        rectParamsResolved  = {**l_RBub_r_rect_parms,**l_DBub_r_rect_parms,**l_RBub_rect_parms,**l_DBub_rect_parms, **l_MBub_rect_parms} # first time MB might have str key
        
        resolvedGlobals      = list(oldNewDB[:,0]) + list(RBOldNewDist[:,0])   # holds global IDs of resolved
        resolvedLocals      = list(oldNewDB[:,1]) + list(RBOldNewDist[:,1])   # holds local IDs of resolved
        combosOldNew        = np.array(overlappingRotatedRectangles(rectParamsOld,{ID:a for ID,a in rectParamsResolved.items() if ID in resolvedLocals}),int)  # old missing (glob) intersects new resolved (loc).
        preCalculated       = {} # concave hull, centroidTest, 
        if len(combosOldNew) > 0:
            cc_unique           = graphUniqueComponents(list(map(str,unresolvedOldDB)), [[str(a),b] for a,b in combosOldNew])  # clusters: old unresolved intersect resolved old. e.g [[old lobal, old local]]: [['2', 21]]
            cc_unique           = [a for a in cc_unique if len(a)>1]                                                        # sometimes theres no intersection, cluster of 1 element. drop it.
            cc_unique           = [[a for a in b if type(a) == str]+[a for a in b if type(a) != str] for b in cc_unique]    # follow convension [str(ID1), ID2]
            whereOldGlobals     = [np.argwhere(resolvedLocals == b)[0][0] for [_,b] in cc_unique]                           
            oldResolvedGlobals  = [resolvedGlobals[i] for i in whereOldGlobals]                                             # global IDs of resolved old bubbles.               e.g [3]
            oldPreMergeGlobals  = [[int(a),oldResolvedGlobals[i]] for i,[a,_] in enumerate(cc_unique)]                      # global IDs of possible merged bubbles.            e.g [[2, 3]]
            oldResolvedLocals   = [resolvedLocals[i] for i in whereOldGlobals]                                              # local ID of old resolved bubble (on new frame)    e.g [21]
            oldAreas            = [np.array([l_areas_hull_old[ID] for ID in IDs], int)          for IDs in oldPreMergeGlobals]  # hull areas             of old globals         e.g [array([ 3683, 15179])]
            oldCentroids        = [np.array([l_predict_displacement[ID][0] for ID in IDs], int) for IDs in oldPreMergeGlobals]  # predicted centroids    of old globals         e.g [array([[1069,  476], [ 969,  462]])]
            centroidTest        = [np.average(a, weights = b,axis = 0).astype(int)      for a,b in zip(oldCentroids,oldAreas)]  # weighted predicted centroid.                  e.g [array([988, 464])]
                                                                                                                            # if merge happened, buble will be about here.
            realCentroids       = {ID:centroidsResolved[ID] for ID in oldResolvedLocals}                                    # actual centroid of new found bubble (local ID)    e.g {21: (986, 464)}
            predictedCentroids  = {ID:l_predict_displacement[ID][0] for ID in oldResolvedGlobals}                           # expected centroid of that bubble                  e.g {3: (969, 462)}
            previousInfo        = [getMergeCritParams(l_ellipse_parms_old, old, 0.4, 25) for old in oldPreMergeGlobals]     # params to constuct convex-concave hull based on prev bubble orientation.
            mergeCritStuff      = [mergeCrit(oldNewIdsResolved[lID], l_contours, pInfo, alphaParam = 0.06, debug = 0) for lID,pInfo in zip(oldResolvedLocals,previousInfo)]
            areasTest           = [int(cv2.contourArea(hull)) for hull,_,_ in mergeCritStuff]                               # potentially better variant of hull area.          e.g [18373]
            predictedAreas      = [l_areas_hull_old[ID] for ID in oldResolvedGlobals]                                       # expected old OG bubble area.                      e.g [15179]
            realArea            = [areasResolved[ID] for ID in oldResolvedLocals]                                           # actual new bubble area                            e.g [20791]
            areaPass            = [False]*len(oldResolvedGlobals)                                                           # containers to track state of failure
            distPass            = [False]*len(oldResolvedGlobals)                                                           # practical (e.g) example shows that Test centroid and Areas are closer to real vals.
            for i,[realValue, myVariant, oldVariant] in enumerate(zip(realArea,areasTest,predictedAreas)):                  # compare predicted OG vals to (new and recovered (considering merge happened))
                relArea = lambda area: np.abs(realValue-area)/realValue
                if relArea(myVariant) < relArea(oldVariant): areaPass[i] = True                                             # if relative area change w.r.t new are is smaller for merged than solo->...
            
            for i,[realValue, myVariant, oldVariant] in enumerate(zip(list(realCentroids.values()),centroidTest,list(predictedCentroids.values()))):
                dist = lambda centroid: np.linalg.norm(np.array(realValue) - np.array(centroid))   
                if dist(myVariant) <= dist(oldVariant): distPass[i] = True                                                  # compare distance from new to (predicted old and recustructed assuming merge)
            
            for i, [passA,passD] in enumerate(zip(areaPass,distPass)):
                if passA == True and passD == True:
                    oldNewDB  = np.array([[a,b] for a,b in oldNewDB  if a != oldResolvedGlobals[i]])            # drop resolved status from bubble.
                    RBOldNewDist    = np.array([[a,b] for a,b in RBOldNewDist    if a != oldResolvedGlobals[i]])            # easier to re-construct array than looking if ID is in, then where to delete.
                    cc_oldIDs.append(oldPreMergeGlobals[i])
                    newIDs = oldNewIdsResolved[oldResolvedLocals[i]]
                    cc_newIDs.append(newIDs)
                    preCalculated[min(newIDs)] = [oldAreas[i], centroidTest[i], mergeCritStuff[i][0], mergeCritStuff[i][1]]
                    print(f'Merge detected between old:{oldPreMergeGlobals[i]} and new:{newIDs}')
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
                
                    if typeTemp == typeRing:
                        dataSets    = [l_RBub_r_masks,l_RBub_r_images,l_RBub_r_old_new_IDs,l_RBub_r_rect_parms,l_RBub_r_centroids,l_RBub_r_areas_hull,l_RBub_contours_hull]
                    else: 
                        dataSets    = [l_DBub_r_masks,l_DBub_r_images,l_DBub_r_old_new_IDs,l_DBub_r_rect_parms,l_DBub_r_centroids,l_DBub_r_areas_hull,l_DBub_contours_hull]
                
                    selectID        = old[areaLargestID]    # largest area inherits ID
                    
                    if min(new) in preCalculated:
                        [areas, predictCentroid, hull, newParams] = preCalculated[min(new)]
                    else:
                        previousInfo        = getMergeCritParams(l_ellipse_parms_old, old, 0.4, 25)
                        inletCase = all([oldID in inletIDs for oldID in old])
                        if inletCase == True:                                             # INLET stuff here !!!!! if two inlet zone bubbles merge.
                            hull        = cv2.convexHull(np.vstack(l_contours[new]))
                        else:
                            hull, newParams, _  = mergeCrit(new, l_contours, previousInfo, alphaParam = 0.05, debug = 0, debug2 = 0, gc = globalCounter, id = selectID)
                        areas               = [l_areas_hull_old[IDs] for IDs in old]
                        predictCentroid     = np.average([l_predict_displacement[ID][0] for ID in old], weights = areas, axis = 0).astype(int)  #l_centroids_old[IDs]
                        
                    
                    tempStore2(new, l_contours, selectID, err, orig, dataSets, concave = hull)
                    centroidReal                            = dataSets[4][selectID]                               # centroid from current hull (convex or concave)
                    dist2                                   = np.linalg.norm(np.array(predictCentroid) - np.array(centroidReal))  
                    prevCentroid                            = np.average([l_centroids_old[ID] for ID in old], weights = areas, axis = 0).astype(int)
                    if inletCase == False:
                        l_MBub_info[int(selectID)]              = [old,np.sum(areas).astype(int),prevCentroid] + list(newParams) # [(645, 557), 8.6]
                    g_Centroids[selectID][globalCounter-1]  = prevCentroid                                          # move pre-merge centroid to common center of mass. should be easer to calculate dist predictor.
                    l_predict_displacement[selectID]        = [tuple(map(int,predictCentroid)), int(dist2)]         # predictCentroid not changed idk why. but dist is
                    for ID in [elem for elem in old if elem != selectID]:                                           # looks like its for a case with 2+ buble merge    
                        g_bubble_type[ID][globalCounter-1]  = typePreMerge
                        g_Centroids[ID][globalCounter]      = dataSets[4][selectID]                                 # dropped IDs inherit merged bubble centroid, so visuall they merge.
                    
                # ===== IF there are about same size merge bubbles, create new ID, hull is concave
                else:
                    selectID =  str(min(new))                                           # str to differentiate first time merged IDs
                    
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
                    #temp.append(newParams[3])
                    
                    l_MBub_info[int(selectID)] = temp

                    for ID in old:
                        g_bubble_type[ID][globalCounter-1]  = typePreMerge               # since all olds get RIPped, change their type as a terminator
                        g_Centroids[ID][globalCounter]      = dataSets[4][selectID]
                # new object will have an id of merged bubble for N frames, during which hull will be done via alphashape. think what crit can say that shape became static
                    # hasSmallBsMerge = True cases will stored in RBub and DBub, i think there is no need to analyze alphashape
                    
                        
                    
                #print(f'old IDs: {old}, main_old {selectID}, new IDs: {new}, type: {typeTemp}')
                unresolvedNewRB = [elem for elem in unresolvedNewRB if elem not in new]
                unresolvedNewDB = [elem for elem in unresolvedNewDB if elem not in new]
                print(f'unresolvedNewRB, (updated) unrecovered RBs: {unresolvedNewRB}')
                print(f'unresolvedNewDB, (updated) unrecovered RBs: {unresolvedNewDB}')
    for newID in unresolvedNewRB:
        dataSets = [l_RBub_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull,l_RBub_contours_hull]
        tempStore2([newID], l_contours, newID, err, orig, dataSets, concave = 0)
    print(f'l_MBub_info_old: {l_MBub_info_old}\nl_MBub_info: {l_MBub_info}')  
    # ================================= Save first iteration ======================================  
    if globalCounter == 0 :
        startID = 0
        for tempID, newKey in enumerate(l_RBub_masks): # by keys
            l_RBub_masks_old[tempID]            = l_RBub_masks[newKey]
            l_RBub_images_old[tempID]           = l_RBub_images[newKey]
            l_RBub_rect_parms_old[tempID]       = l_RBub_rect_parms[newKey]
            l_RBub_centroids_old[tempID]        = l_RBub_centroids[newKey]
            l_RBub_old_new_IDs_old[tempID]      = [newKey]
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
        l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old = {},{},{},{},{}            
        l_FBub_old_new_IDs_old = {}
        #print('g_bubble_type',g_bubble_type)
        #l_bubble_type_old = {ID:val[0] for ID,val in g_bubble_type.items()}
        #print('l_bubble_type',l_bubble_type)
          
    # ================================= Save other iterations ====================================== 
    if globalCounter >= 1:
        l_RBub_masks_old, l_RBub_images_old, l_RBub_rect_parms_old, l_RBub_centroids_old, l_RBub_old_new_IDs_old = {}, {}, {}, {}, {}
        l_DBub_masks_old, l_DBub_images_old, l_DBub_rect_parms_old, l_DBub_centroids_old, l_DBub_old_new_IDs_old = {}, {}, {}, {}, {}
        l_FBub_masks_old, l_FBub_images_old, l_FBub_rect_parms_old, l_FBub_centroids_old, l_FBub_areas_hull_old = {},{},{},{},{}
        l_MBub_masks_old, l_MBub_images_old, l_MBub_rect_parms_old, l_MBub_centroids_old, l_MBub_areas_hull_old, l_MBub_old_new_IDs_old = {}, {}, {}, {}, {}, {}
        l_MBub_info_old = {}
        # print('l_DBub_centroids',l_DBub_centroids)
        
        # ============== frozen tricks ============
        oldLocIDs2 = frozenIDsInfo[:,0] if len(frozenIDsInfo)>0 else np.array([]) # 10/02/23
        oldGlobIDs = {}
        for oldLocID in oldLocIDs2:
            if type(oldLocID) == str: oldGlobIDs[oldLocID] = int(oldLocID)
            else: oldGlobIDs[oldLocID] = max(g_bubble_type) + 1
        #oldGlobIDs02 = [
        #                    {ii:ID for ID, vals in l_old_new_IDs_old.items() if ii in vals} if type(ii) != str else {ii:int(ii)} 
        #                    for ii in oldLocIDs2 ] # relevant new:old dict   
        #oldGlobIDs = {};[oldGlobIDs.update(elem) for elem in oldGlobIDs02] # basically flatten [{x:a},{y:b}] into {x:a,y:b} or something  10/02/23
        l_FBub_old_new_IDs_old = {oldGlobIDs[localOldID]:localNewIDs for localOldID, localNewIDs,_,_,_ in frozenIDsInfo}
        for key in l_FBub_masks: # contains relation indices that satisfy distance
            gKey = oldGlobIDs[key]
            l_FBub_masks_old[gKey]                  = l_FBub_masks[key]
            l_FBub_images_old[gKey]                 = l_FBub_images[key]
            l_FBub_rect_parms_old[gKey]             = l_FBub_rect_parms[key]
            l_FBub_centroids_old[gKey]              = l_FBub_centroids[key] 
            l_FBub_areas_hull_old[gKey]             = l_FBub_areas_hull[key]
            l_bubble_type[gKey]                     = typeFrozen if str(gKey) not in lastNStepsFrozenCentroids else typeRecoveredFrozen
            l_contours_hull[gKey]                   = l_FBub_contours_hull[key]
            if globalCounter not in frozenGlobal:
                frozenGlobal[globalCounter]             = []
                g_FBub_rect_parms[globalCounter]        = {}
                g_FBub_centroids[globalCounter]         = {}
                g_FBub_areas_hull[globalCounter]        = {}
               
            if gKey not in g_bubble_type: g_bubble_type[gKey] = {}
            
            g_bubble_type[gKey][globalCounter]      = typeFrozen if str(gKey) not in lastNStepsFrozenCentroids else typeRecoveredFrozen
            frozenGlobal[globalCounter].append(gKey)
            g_FBub_rect_parms[globalCounter][gKey]  = l_FBub_rect_parms[key]
            g_FBub_centroids[globalCounter][gKey]   = l_FBub_centroids[key]
            g_FBub_areas_hull[globalCounter][gKey]  = l_FBub_areas_hull[key]

        for [old,new] in oldNewDB: # contains relation indices that satisfy distance
            l_bubble_type[old]                  = typeElse 
            g_bubble_type[old][globalCounter]   = typeElse 
            l_DBub_masks_old[old]               = l_DBub_masks[new]
            l_DBub_images_old[old]              = l_DBub_images[new]
            l_DBub_rect_parms_old[old]          = l_DBub_rect_parms[new]
            l_DBub_centroids_old[old]           = l_DBub_centroids[new]
            l_DBub_old_new_IDs_old[old]         = l_DBub_old_new_IDs[new]
            l_DBub_areas_hull_old[old]          = l_DBub_areas_hull[new]
            l_contours_hull[old]                = l_DBub_contours_hull[new]
                
        for key in l_DBub_r_masks.keys(): # holds global keys
            if key in l_MBub_info:
                l_MBub_info_old[key]            = l_MBub_info[int(key)]
            l_bubble_type[key]                  = typeRecoveredElse
            g_bubble_type[key][globalCounter]   = typeRecoveredElse
            l_DBub_masks_old[key]               = l_DBub_r_masks[key]
            l_DBub_images_old[key]              = l_DBub_r_images[key]
            l_DBub_rect_parms_old[key]          = l_DBub_r_rect_parms[key]
            l_DBub_centroids_old[key]           = l_DBub_r_centroids[key] 
            l_DBub_old_new_IDs_old[key]         = l_DBub_r_old_new_IDs[key]
            l_DBub_areas_hull_old[key]          = l_DBub_r_areas_hull[key]
            l_contours_hull[key]                = l_DBub_contours_hull[key]
                
            
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
                
        for key in l_RBub_r_masks.keys(): # holds global keys
            if key in l_MBub_info:
                l_MBub_info_old[key]            = l_MBub_info[int(key)]
            g_bubble_type[key][globalCounter]   = typeRecoveredRing
            l_bubble_type[key]                  = typeRecoveredRing
            l_RBub_masks_old[key]               = l_RBub_r_masks[key]
            l_RBub_images_old[key]              = l_RBub_r_images[key]
            l_RBub_rect_parms_old[key]          = l_RBub_r_rect_parms[key]
            l_RBub_centroids_old[key]           = l_RBub_r_centroids[key]
            l_RBub_old_new_IDs_old[key]         = l_RBub_r_old_new_IDs[key]
            l_RBub_areas_hull_old[key]          = l_RBub_r_areas_hull[key]
            l_contours_hull[key]                = l_RBub_contours_hull[key]
                            
        if globalCounter == 1222:
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
            g_bubble_type[gID][globalCounter]   = typeElse if localID not in frozenIDs else typeFrozen
            
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
        for localID in l_MBub_old_new_IDs:
            if type(localID) == str:
                #startID             += 1
                gID                 = startID
                g_bubble_type[gID]  = {}
                g_MBub_info[gID]    = {}
            
            else: gID = localID

            l_bubble_type[gID]                  = typeMerge
            
            g_bubble_type[gID][globalCounter]   = typeMerge
            g_MBub_info[gID][globalCounter]     = l_MBub_info[int(localID)]
            l_MBub_info_old[gID]                = l_MBub_info[int(localID)]
            l_MBub_masks_old[gID]               = l_MBub_masks[localID]
            l_MBub_images_old[gID]              = l_MBub_images[localID]
            l_MBub_rect_parms_old[gID]          = l_MBub_rect_parms[localID]
            l_MBub_centroids_old[gID]           = l_MBub_centroids[localID]
            l_MBub_old_new_IDs_old[gID]         = l_MBub_old_new_IDs[localID]
            l_MBub_areas_hull_old[gID]          = l_MBub_areas_hull[localID]
            l_contours_hull[gID]                = l_MBub_contours_hull[localID]

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
        g_Masks[key][globalCounter]         = l_masks_old[key]
        g_Images[key][globalCounter]        = l_images_old[key]
        g_Rect_parms[key][globalCounter]    = l_rect_parms_old[key]
        g_Ellipse_parms[key][globalCounter] = l_ellipse_parms_old[key]
        g_Centroids[key][globalCounter]     = l_centroids_old[key]
        g_old_new_IDs[key][globalCounter]   = l_old_new_IDs_old[key]
        g_contours_hull[key][globalCounter] = l_contours_hull_old[key]
        # --------- take care of prediction storage: g_predict_area_hull & g_predict_displacement --------------
        # -- some extra treatment for merged bubbles before regular ones ---
        if  (key not in g_predict_area_hull and key in l_areas_hull_old and key in l_MBub_areas_hull_old):
            localID                         = min(l_MBub_old_new_IDs_old[key])
            prevArea                        = l_MBub_info[localID][1]                                       # sum of pre-merged bubbles
            prevCentroid                    = l_MBub_info[localID][2]                                       # restored from pre-merged area weighted centroids.
            g_predict_area_hull[key]        = {globalCounter-1:[prevArea,  prevArea,  int(prevArea*0.2)]}   # initiate predictors frame earleir
        if globalCounter == 0 or (key not in g_predict_area_hull and key in l_areas_hull_old):              # first time merged bubbles fail this and go to else
            g_predict_area_hull[key]        = {globalCounter:[l_areas_hull_old[key],  l_areas_hull_old[key],  int(l_areas_hull_old[key]*0.2)]}
        else:
            updateValue                     = l_areas_hull_old[key]
            historyLen                      = len(g_predict_area_hull[key])
            timeStep                        = globalCounter-1 if key not in lastNStepsFrozenLatestTimes else lastNStepsFrozenLatestTimes[key]
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
            timeStep                        = globalCounter-1 if key not in lastNStepsFrozenLatestTimes else lastNStepsFrozenLatestTimes[key]
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
                
            g_predict_displacement[key][globalCounter] = [pCentroid, int(updateValue), max(int(pc2CMean),5), max(int(pc2CStd),2)]# 15/03/23 [pCentroid, np.around(updateValue,2), np.around(pc2CMean,2), np.around(pc2CStd,2)]

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
    l_Areas_old                             = l_Areas
    l_Areas_hull_old                        = l_Areas_hull
    l_Centroids_old                         = l_Centroids
    l_rect_parms_all_old          = l_rect_parms_all
    contoursFilter_RectParams_dropIDs_old   = contoursFilter_RectParams_dropIDs # prepare/store for next time step   
    frozenIDs_old                           = frozenIDs
    if len(frozenLocal)>0:
        if globalCounter not in frozenGlobal_LocalIDs:
            #frozenGlobal[globalCounter] = []
            frozenGlobal_LocalIDs[globalCounter] = []
        #frozenGlobal[globalCounter].append(frozenLocal)
        frozenGlobal_LocalIDs[globalCounter].append(frozenLocal)

    for key in list(l_DBub_r_masks.keys()) + list(l_RBub_r_masks.keys()):
        if key in contoursFilter_RectParams_dropIDs:
            l_Areas_old[key]        = cv2.contourArea(l_contours[key])
            l_Centroids_old[key]    = getCentroidPosContours(bodyCntrs = [l_contours[key]])[0]



    
    
    #if big != 1: cv2.imshow("aas",aas)

    globalCounter += 1



exportFirstFrame(markFirstExport,dataStart)    

cntr = 0
if mode == 1:
    if big == 0:
        mainer(imgNum)
    # ss = bigRun[99]
    # print(ss)
    # mainer(ss)
    else:
        for i in range(dataStart,dataStart+dataNum,1):
            print(f'\n==============================================')
            print(f'Time step ID: {cntr} max ID: {dataNum-1}')
            mainer(i)
            cntr += 1
if mode == 2:
    mainer(0)
       
for globalCounter in range(globalCounter):
    #==============================================================================================
    #======================== DRAWING STUFF START ================================================
    #==============================================================================================
    
    activeIDs = [ID for ID, timeDict in g_bubble_type.items() if globalCounter in timeDict]#;print(f'globalCounter: {globalCounter} activeIDs: {activeIDs}')
    blank = X_data[dataStart+globalCounter] * 1
    #ori = blank.copy()
    blank = cv2.subtract(np.uint8(blank), np.uint8(mean))
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
    cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\"+str(dataStart+globalCounter).zfill(4)+".png" ,blank)
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
#print(X_data[1] == X_data[2])
#dataStart = 1
#dataNum = 51
#cntr = 1
#for i in range(dataStart,dataStart+dataNum,1):
#            #print(f'\n==============================================')
#            print(f'Time step ID: {cntr} max ID: {dataNum-1}. i: {i}')
#            orig0 = X_data[i]
#            if cntr >= 1  and cntr <= 25:
#                cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\"+str(dataStart+cntr).zfill(4)+".png" ,orig0)
#                #print('cntr > 1  and cntr <= 10')
#            orig = orig0 -cv2.blur(mean, (5,5),cv2.BORDER_REFLECT)
#            orig[orig < 0] = 0                  # if orig0 > mean
#            orig = orig.astype(np.uint8)
#            if cntr > 25  and cntr <= 50:
#                cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\"+str(dataStart+cntr).zfill(4)+".png" ,orig)
#            cntr += 1