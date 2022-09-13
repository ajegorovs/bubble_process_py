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
                              getContourHullArea, centroidAreaSumPermutations, listFormat,
                              distStatPredictionVect,distStatPredictionVect2,updateStat)
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


def partialBubble(image, graphics = 0, m=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    w,h = image.shape[0],image.shape[1]
    dlt = int(np.sqrt(w**2+h**2)/2 + 1)
    tempImg0 = cv2.morphologyEx(image.copy(), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    tempImg0 = cv2.copyMakeBorder(tempImg0, dlt, dlt, dlt, dlt, cv2.BORDER_CONSTANT, None, 0)
    tempImg = tempImg0.copy()
    temp = list()
    temp.append(tempImg)
    maxIter = 19
    di = 15
    for i in range(5,maxIter*di,di):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i,i))
        A = cv2.morphologyEx(tempImg0, cv2.MORPH_CLOSE, kernel, iterations = 1)
        tempCntr, _ = cv2.findContours(A,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        A = cv2.drawContours(A.copy(), tempCntr, -1, 255, -1)
        typeCheck, _, _ = bubbleTypeCheck(image = A, index = str(m)+" "+str(i),erode = 4,close = 1,areaRemainingThreshold = 0.75, graphics = 0)
        # cv2.imshow("adsad",A)
        
        temp2 = cv2.addWeighted(A, 0.5, tempImg0, 1 - 0.5,0)
        temp2 = cv2.putText(temp2.copy(), str(typeCheck), (dlt+10,dlt+10), font, 0.4, 204, 1, cv2.LINE_AA)
        temp.append(temp2)
        if typeCheck >= 0.7: break
    tempCntr, _ = cv2.findContours(A[dlt:-dlt,dlt:-dlt],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # if graphics == 1:
    #     tempRGB = cv2.drawContours(imageRGB.copy(), tempCntr, -1, (255,0,255), 1)
    # else: tempRGB = None
    return A, dlt
    # imgMrphCls = [img[dlt:-dlt,dlt:-dlt] for img in temp]
    # output[y:y+h,x:x+w] = imgMrphCls[len(imgMrphCls)-1]
    # reportList[4][y:y+h,x:x+w] = A[dlt:-dlt,dlt:-dlt]

def hullGeneral(image):
    cntrHull, _ = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cntrHull)>0:
    # print(len(cntrHull))
        cont = np.vstack(cntrHull)
        hull = cv2.convexHull(cont)
        return cv2.drawContours(  image.copy() * 0,   [hull], -1, 255, -1)
    else:
        return image.copy() * 0
    
def removeBorderComponents(img,mode=1,px=1):    
    # remove border components https://stackoverflow.com/questions/65534370/remove-the-element-attached-to-the-image-border
    image = convertRGB2Gray(img)
    if mode == 1:
        pad = cv2.copyMakeBorder(image, px,px,px,px, cv2.BORDER_CONSTANT, value=255)
        h, w = pad.shape
        mask = np.zeros([h + 2, w + 2], np.uint8)
        img_floodfill = cv2.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1]
        out = img_floodfill[px:h-px, px:w-px]
        
    else:
        pad = np.full(image.shape,255,dtype=np.uint8)
        h, w = pad.shape
        pad[px:h-px, px:w-px] = image[px:h-px, px:w-px]
        mask = np.zeros([h + 2, w + 2], np.uint8)
        out = cv2.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1]    
    
    
    return out
    

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
def maskProcSegments(img, mask, lowThreshold, graphics, exportGraphics, drawBG, name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    dilate = 5
    img = convertRGB2Gray(img.copy())
    imgBG = convertRGB2Gray(drawBG.copy())
    if graphics == 1 or exportGraphics == 1:
        imgRGB = convertGray2RGB(imgBG.copy())
        overlayColor = [0,255,255]
        textColor = [0,0,255]
        textColorFull = [0,165,255]
        textColorEmpty = [0,255,0]
        overlayOpacityMultiplier = 0.55
        nReports = 4
        reportList = np.zeros((nReports,mask.shape[0],mask.shape[1]),dtype=np.uint8)
        # reportListRGB = np.zeros((nReports,mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        reportListRGB = [imgRGB for A in range(nReports)]
        # cv2.imshow('aa',reportListRGB2[0])
        
            
    # print(reportList[0].shape)
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    hulls = list()
    m = 0
    M = np.full((mask.shape[0],mask.shape[1],3),overlayColor,dtype=np.uint8)
    
        
    if plotAreaRep == 1:
        fig, ax = plt.subplots()
        ax.set_facecolor('#eafff5')
        ax.set_title(' arRemList')
        # clrs = ['b','g','r','c','m','y']
        cmap = plt.get_cmap('gnuplot')
        clrs = [cmap(i) for i in np.linspace(0, 1, len(contours))]   
    # flowchart:
    # f-n gets rough mask where clusters are joined using hull
    # each body is examined:
    # 1) on an expanded selection
    # 2) rough submask 10->255 where more detailed
    # 3) advanced thresholding is conducted or submasked region
    # 4) dilate-erode morphology is used to close gaps
    cSel = [i for i in range(len(contours)) if cv2.contourArea(contours[i]) > 200]
    # print(cSel)
    contours = [contours[i] for i in cSel]
    # print(f'# conrs= {len(contours)}')
    for c in contours:
        # print(cv2.contourArea(c))
        # 1) Full size blanks for bubble and mask. full beacause selection will be dilated.
        maskSelection = mask.copy() * 0 
        selFull = mask.copy() * 0
        x, y, w, h = cv2.boundingRect(c)
        # in case dilate goes out of border, selection should be modified.
        if dilate > 0 : 
            # x0, y0, w0, h0 = x, y, w, h
            x, y, w, h = getPaddedSegmentCoords(mask, x, y, w, h, dilate)
        if graphics == 1 or exportGraphics == 1: tempRGB = imgRGB[y:y+h,x:x+w]
        # on full size black blank draw A single controur and dilate it    
        maskSelection = cv2.drawContours(maskSelection.copy(), [c], -1, 255, -1)
        
        # maskSelection = cv2.dilate(maskSelection.copy(),np.ones((dilate,dilate),np.uint8),iterations = 1)
        # redraw bubble only on expanded (dialated) mask area
            
        selFull[maskSelection == 255] = img[maskSelection == 255]
        # take only working area
        subImage = selFull[y:y+h,x:x+w]
        temp1 = subImage.copy() * 0

        # cv2.imshow(f'm {m}',   maskedBlend(convertGray2RGB(img),(255,255,0),mask,0.1))

        # 2) rough secondary mask- intensity higher than 10 -> 255. works for us
        _,th0 = cv2.threshold(subImage.copy(),lowThreshold,255,cv2.THRESH_BINARY)
        # th0test = cv2.morphologyEx(th0.copy(), cv2.MORPH_OPEN, np.ones((13,13)), iterations = 1)
        # cv2.imshow(f'm {m}',th0)
        # reportList[1][y:y+h,x:x+w] = th0
        # add extra closing to remove sharp corners and small holes
        kerSize = 11
        # if np.sum(th0test) == 0: #higher threshold can delete object
        #     m += 1
        #     break
        # if m==5:cv2.imshow(f'm {m}',th0test)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kerSize,kerSize))
        th03 = cv2.copyMakeBorder(th0.copy(), kerSize, kerSize, kerSize, kerSize, cv2.BORDER_CONSTANT, None, 0)
        th03 = cv2.morphologyEx(th03, cv2.MORPH_CLOSE, kernel, iterations = 1)
        th03 = th03[kerSize:-kerSize,kerSize:-kerSize]
         
        # temp1[th0 == 255] = subImage[th0 == 255] # without closing
        temp1[th03 == 255] = subImage[th03 == 255]
        alpha = 0.5
        th00 = cv2.addWeighted(th03, alpha, th0, 1 - alpha,0)
        
        
        # 3) image thresholding is performed
        locBinsAll = [cv2.ximgproc.BINARIZATION_NIBLACK,cv2.ximgproc.BINARIZATION_SAUVOLA,cv2.ximgproc.BINARIZATION_WOLF,cv2.ximgproc.BINARIZATION_NICK]
        globBin = cv2.THRESH_BINARY | cv2.THRESH_OTSU
        thrInput = np.uint8(temp1)
        # th = cv2.ximgproc.niBlackThreshold(thrInput,255,globBin,31,0.1, _, locBinsAll[2])
        th = cv2.ximgproc.niBlackThreshold(thrInput,255,globBin,37,0.79, _, locBinsAll[2])
        if 1==122:
            aa = convertGray2RGB(thrInput)
            thrInput = cv2.edgePreservingFilter(aa,None,1,200,0.5)
            thrInput2 = convertRGB2Gray(thrInput)
            th = cv2.ximgproc.niBlackThreshold(thrInput2,255,globBin,37,0.79, _, locBinsAll[2])    
            # cv2.imshow('orig',aa)
            # uu = cv2.hconcat([cv2.edgePreservingFilter(aa,None,1,200,0.5),aa.copy()])
        # cv2.imshow(f'm{m}', th)
        global runTh
        # print(f'm= {m}, sum= {np.sum(th)}')
        if m == testThreshold:
            runTh = True
            global xxx
            xxx = imgBG[y:y+h,x:x+w]
            # xxx = thrInput2
        if m == 21:
            # c = 255/(np.log(1 + np.max(subImage)))
            gamma = 1.4
            uu = orig[y:y+h,x:x+w]
            tr = np.array(255*( uu/ 255) ** gamma, dtype = 'uint8')
            # cv2.imshow('00', np.uint8(log_transformed))
            init_ls = checkerboard_level_set(thrInput.shape, 4)

            ls = morphological_chan_vese(uu, iterations=2, init_level_set=init_ls,
                                         smoothing=3)
            cv2.imshow('aa', cv2.hconcat([(ls*255).astype(np.uint8),uu,tr]))
            
        if 1==21:
            ss = cv2.morphologyEx(th.copy(), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            cntrHull, _ = cv2.findContours(ss,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cont = np.vstack(cntrHull)
            hull = cv2.convexHull(cont)
            uni_hull = []
            uni_hull.append(hull) # <- array as first element of list
            xx = cv2.drawContours(  th.copy() * 0,   uni_hull, -1, 255, -1)
            yy = cv2.drawContours(  th.copy() * 0,   cntrHull, -1, 255, -1)
            zz = xx-yy
            
            zz = cv2.addWeighted(zz, 0.5, xx, 1 - 0.5,0)
            hulls.append(zz)
        if 11==1:
            zz = cv2.morphologyEx(th.copy(), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            # dlt = 200
            # zz = cv2.copyMakeBorder(zz.copy(), dlt, dlt, dlt, dlt, cv2.BORDER_CONSTANT, None, 0)
            cntrHull, _ = cv2.findContours(zz,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            minRect = [cv2.minAreaRect(c) for i,c in enumerate(cntrHull)]
            

            for i,c in enumerate(cntrHull):
                box = cv2.boxPoints(minRect[i])
                # print(minRect[i])
                box = np.intp(box)
                cv2.drawContours(zz, [box], 0, 124)
                print(cv2.contourArea(box))
                print(box)


            # [cv2.drawContours(zz, [np.intp(cv2.boxPoints(A))], 0, 124) for A in minRect]
            # [cv2.polylines(zz, c, True, 125, 1) for c in minRect]

            hulls.append(zz)
        # remove small white speckles
        # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        
        # 4) REMOVED (joining neighbor bubbles) dilate-erode morphology is used to close gaps. 
        # sub-image is expanded to prevent growing wall bound objects
        # dilate = 3
        # dilIter = 3
        # dlt = dilate * dilIter + 1
        # th2 = cv2.copyMakeBorder(th.copy(), dlt, dlt, dlt, dlt, cv2.BORDER_CONSTANT, None, 0)
        # th2 = cv2.dilate(th2.copy(),np.ones((dilate,dilate),np.uint8),iterations = dilIter)
        # th2 = cv2.erode(th2.copy(),np.ones((dilate,dilate),np.uint8),iterations = dilIter)
        
        # th2 = th2[dlt:-dlt,dlt:-dlt]
        # reportList[3][y:y+h,x:x+w] = th2
        
        # if plotAreaRep == 1 and  drw ==0:
        # drw = 1
        checkErd = 4
        # temp = np.sum(cv2.erode(th.copy(),np.ones((checkErd,checkErd),np.uint8),iterations = 1))
        # print(f'm= {m}, sum= {temp}')
        
        if m == inspectBubbleDecay:  drw = 1
        else:       drw = 0 
        bubbleTypeAVG, arRemList, coefs = bubbleTypeCheck(image = th, index = m,erode = checkErd,close = 1,
                                        areaRemainingThreshold = 0.75, graphics = drw)
        # print(coefs)
        if coefs[0]== None:# bubbleTypeCheck fails if there is only 1 iteration
            m += 1
            break
        # print(f'm= {m}, arRemList= {arRemList}')
        # print(coefs)
        # print(arRemList)
        if plotAreaRep == 1:
            ax.plot(arRemList,linewidth  = 3,c = clrs[m],label= '#{}; {:.4f}'.format(m,coefs[0]))
            # fx = arRemList[1:]
            xx = np.arange(1,len(arRemList),1)
            # zz = np.polyfit(xx, fx, 1)#list(range(1,len(y)+1,1))
            ax.plot(xx,coefs[0]*xx +coefs[1],'--',c = clrs[m])
            ax.legend(loc ="lower left")
            ax.set_title('Relative area loss per iteration')
            ax.set_ylim(0,1)
            # print(np.full(len(arRemList),np.average(arRemList)))
            # plt.legend([f'obj #{m}'])
        if bubbleTypeAVG <= 0.1:
            # bp, dlt = partialBubble(image = th, graphics = 0, m=m)
            # reportList[4][y:y+h,x:x+w] = bp[dlt:-dlt,dlt:-dlt]
            thcl = cv2.morphologyEx(th.copy(), cv2.MORPH_OPEN,
                                                   np.ones((3,3),np.uint8))
            tempFull  = hullGeneral(thcl)
            
         
        if bubbleTypeAVG >= 0.9:# can join reflection with bubble or erode it
            tempFull = cv2.morphologyEx(th.copy(), cv2.MORPH_OPEN,
                                                   np.ones((3,3),np.uint8))
            tempFull = maskDilateErode(img = tempRGB,mask = tempFull,
                                       kernel = 13,iters = 1, order = 2, graphics = 0,
                                       name = 'global '+str(m))
            # cv2.imshow(str(m)+" tempFull",tempFull)
            cntrHull, _ = cv2.findContours(tempFull,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if len(cntrHull) > 1:
                areas = [cv2.contourArea(cntr) for cntr in cntrHull]
                cntrHull = [cntrHull[np.argmax(areas)]]
                tempFull = cv2.drawContours(tempFull.copy() *0, cntrHull, -1, 255,-1)
                # cv2.imshow(str(m)+" tempFull",tempFull)
            if 1 == 1:
                # check if main contur takes up most of the frame. if not detect hull parasites
                x1, y1, w1, h1 = cv2.boundingRect(cntrHull[0])
                # cv2.rectangle(tempFull,(x1,y1),(x1+w1,y1+h1),127,1)
                # cv2.rectangle(tempFull,(0,0),(x+w,y+h),127,1)
                # print(f'{1/np.sqrt(h*w/(h1*w1))}')
                # print(f'h1/h = {h1/h}, w1/w = {w1/w}')
                scale = min(h1/h,w1/w)
                if scale <= 0.65:
                    dd          = 26
                    origCntBrdr   = cv2.copyMakeBorder(tempFull.copy(), dd, dd, dd, dd, cv2.BORDER_CONSTANT, None, 0)
                    origCntBrdrDil   = cv2.dilate(origCntBrdr.copy(),np.ones((dd,dd),np.uint8),iterations = 1)
                    origRawBrdr          = cv2.copyMakeBorder(th.copy(), dd, dd, dd, dd, cv2.BORDER_CONSTANT, None, 0)
                    isolatedMain         = cv2.bitwise_and(origCntBrdrDil.copy(),origRawBrdr.copy())
                    isolatedRest         = cv2.absdiff( origRawBrdr , isolatedMain)
                    isolatedRestOpn         = cv2.morphologyEx(isolatedRest, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
                    isolatedRestCls         = cv2.morphologyEx(isolatedRestOpn, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
                    isolatedRestHull         = hullGeneral(image=isolatedRestCls)
                    # cv2.imshow('asd',cv2.hconcat([origCntBrdrDil,origRawBrdr,isolatedMain,isolatedRest,isolatedRestOpn,isolatedRestCls,isolatedRestHull]))
                    # tt7         = isolatedRestHull[dd:-dd,dd:-dd]
                    combineAllBrdr    = cv2.bitwise_or(origCntBrdr,isolatedRestHull)
                    if 1==12:
                        nms = ['origCntBrdrDil','origRawBrdr','isolatedMain','isolatedRest','isolatedRestOpn','isolatedRestCls','isolatedRestHull','combineAllBrdr']
                        imgs = [origCntBrdrDil,origRawBrdr,isolatedMain,isolatedRest,isolatedRestOpn,isolatedRestCls,isolatedRestHull,combineAllBrdr]
                        imgs = [cv2.putText(img, nms[i], (4,10), font, 0.3, 127, 1, cv2.LINE_AA) for i,img in enumerate(imgs)]
                        cv2.imshow('closed multi '+str(m),cv2.hconcat(imgs))
                    tempFull = combineAllBrdr[dd:-dd,dd:-dd]
                # print(f'area old  = {w * h}')
                # print(f'area new  = {w1 * h1}')
        if 0.1<bubbleTypeAVG<0.9:
            # print('0.1<criterium<0.9')
            # tempFull = cv2.putText(th.copy(), '0.1<crt<0.9', (30,30), font, 0.2, 128, 1, cv2.LINE_AA)
            tempFull = th.copy()
        # print(coefs[0])    
        if coefs[0] >= -0.01:
            init_ls = checkerboard_level_set(th.shape, 4)
            temp = orig[y:y+h,x:x+w]
            ls = morphological_chan_vese(temp, iterations=10, init_level_set=init_ls, smoothing=3)
            ls = np.uint8(ls*255)
            # sometimes chain returns inverted image + some defects in the corners. sol-ns: 
            # 1) extract biggest contour
            # 2) (our) check mean border color, most likely w/o A bubble
            maskLs = np.zeros(th.shape,dtype=np.uint8)
            maskLs[10:h-10,10:w-10]=1
            ma = np.ma.array(ls, mask=maskLs)
            if ma.mean()<=128:
                tempFull = ls
            else:
                tempFull = removeBorderComponents(cv2.bitwise_not(ls),1,1)
            #----tempFull---------------------
            # tempFull = tempFull *0
                
            # cv2.imshow(f'{m}', cv2.hconcat([tempFull,temp,ls,cv2.bitwise_not(ls)]))
            # cv2.imshow(f'2m {m}', maskLs)
            # print( ma.mean()) 
         
        if m==testEdgePres:
            aa = convertGray2RGB(orig[y:y+h,x:x+w])
            # cv2.imshow('orig',aa)
            gg = cv2.edgePreservingFilter(aa,None,1,200,0.5)
            gg = cv2.blur(gg, (3,3),cv2.BORDER_REFLECT)
            # uu = cv2.hconcat([gg,aa.copy()])
            uu = convertRGB2Gray(gg)
            # ss = resizeToMaxHW(uu,1600,900)
            maskLs = np.zeros(uu.shape,dtype=np.uint8)
            maskLs[10:h-10,10:w-10]=1
            ma = np.ma.array(uu, mask=maskLs)
            mn = np.mean(ma)
            print(mn)
            d = 6
            _,hh = cv2.threshold(uu,int(mn)+d,255,cv2.THRESH_BINARY)
            bb = cv2.hconcat([uu,hh])
            
            _,hh2 = cv2.threshold(orig[y:y+h,x:x+w],int(mn)+d,255,cv2.THRESH_BINARY)
            bb2 = cv2.hconcat([orig[y:y+h,x:x+w],hh2])
            gugu = resizeToMaxHW(cv2.vconcat([bb2,bb]),1200,600)
            cv2.imshow('asdasstyl',gugu)
        
        if graphics == 1 or exportGraphics == 1:
            
            if m ==0:
                alpha = 0.5
                msk = cv2.addWeighted(err.copy(), alpha, mask, 1 - alpha,0)
                reportListRGB[0] = maskedBlend(reportListRGB[0],overlayColor,msk.copy(),overlayOpacityMultiplier)
            if coefs[0] >= -0.01: textColor = textColorFull
            else: textColor = textColorEmpty
        
            for c, subMask in enumerate([th00,th,tempFull]):
                
                blank = mask.copy() * 0
                c += 1
                blank[y:y+h,x:x+w] = subMask
                reportListRGB[c] = maskedBlend(reportListRGB[c],overlayColor,blank.copy(),overlayOpacityMultiplier)
                reportListRGB[c] = cv2.putText(reportListRGB[c], str(m), (x+4,y+12), font, 0.3, textColor, 1, cv2.LINE_AA)
                reportListRGB[c] = cv2.rectangle(reportListRGB[c],(x-1,y-1),(x+w,y+h),textColor,1)
                if c >= 2:
                    reportListRGB[c] = cv2.putText(reportListRGB[c], "{:01.2f}".format(bubbleTypeAVG), (x+4,y+h-4), font, 0.3, textColor, 1, cv2.LINE_AA)
                    reportListRGB[c] = cv2.putText(reportListRGB[c], "{:01.4f}".format(coefs[0]), (x+2,y-4), font, 0.3, textColor, 1, cv2.LINE_AA)
                    
                    
        m += 1
    if graphics == 1 or exportGraphics == 1:
        reportNames = [f'{thresh0}+ => 255 and hull', f' >{lowThreshold}+ => 255 threshold, closing by r{kerSize}','local threshold','new']

        reportSet = [cv2.putText(c, reportNames[A], (30,30), font, 0.8, (0,255,255), 2, cv2.LINE_AA) for A,c in enumerate(reportListRGB)]
        
        c = [resizeToMaxHW(img,1200,900) for img in reportSet] 
        def on_trackbar2(var):
            on_trackbar(var,reportSet)
    if graphics == 1:
        if drawFileName == 1:
            # reportSet[0] = cv2.putText(reportSet[0], os.path.basename(imageLinks[imgNum]) , (30,55), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
            title_window = f'file: { os.path.basename(imageLinks[imgNum])}'
        else: title_window = f'image # {imgNum}'
        def on_trackbar(val,array):
            cv2.imshow(title_window, array[val])
        cv2.namedWindow(title_window,cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("report #", title_window , 0, nReports-1, on_trackbar2)
        on_trackbar2(0)
        [cv2.imshow('hulls '+str(i), resizeToMaxHW(hulls[i],400,400)) for i in range(len(hulls))]
    
        
    # cntrHull, _ = cv2.findContours(reportList[nReports-1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # return reportSet[nReports-1]
    # return cv2.drawContours( mean,   cntrHull, -1, (0,255,255), 1)
    return reportList[nReports-1]

def cyclicColor(index):
    colors = [(255,0,0),(0,255,0),(125,125,0),(0,125,125),(0,0,255),(125,0,125),(255,125,0),(255,0,125),(125,255,0)]
    colors = np.array(colors,dtype=np.uint8)
    # np.random.shuffle(colors)
    return colors[index % len(colors)].tolist()

thresh0             = 1
thresh1             = 15

import pickle
plotAreaRep         = 0
inspectBubbleDecay  = 211
testThreshold       = 222
testEdgePres        = 3

mode = 1 # mode: 0 - read new images into new array, 1- get one img from existing array, 2- recall pickled image
big = 1
# dataStart = 71+52 ###520
# dataNum = 7
dataStart = 57#46
dataNum = 3
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


globalCounter = 0

# debug section numbers: 11- RB IDs, 12- RB recovery, 21- Else IDs, 22- else recovery, 31- merge
# debug on specific steps- empty list, do all steps, or time steps in list
debugSections = [11,21,12,22,31]
debugSteps = [1]
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
debugStepsGFX = [-1]
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


typeFull,typeRing, typeRecoveredRing, typeElse, typeFrozen,typeRecoveredElse,typePreMerge = np.int8(0),np.int8(1),np.int8(2),np.int8(3), np.int8(4), np.int8(5), np.int8(6)
centroidsByID2,recParamsByID, areasByID, masksByID,imagesByID,cntrIDsByIDs,bubTypesByID = {},{},{},{},{},{},{}
cntrChildrenIDsByIDs = {}
ringBubMasks_old, ringBubImages_old, ringBubRectParams_old, ringBubCentroids_old,  ringCntrIDs = {}, {}, {}, {}, {}
ringOldNewIDs_old = {}
distanceBubMasks_old, distanceBubImages_old, distanceBubRectParams_old,  distanceBubCentroids_old = {},{},{},{}
distanceOldNewIDs_old = {}
distanceBubMasks, distanceBubImages, distanceBubRectParams,  distanceBubCentroids = {},{},{},{}
distanceOldNewIDs = {}
contoursAll, contours_old, frozenBubs, frozenBubsTimes, bubTypesLocal, bubTypesLocal_old, bubTypesLocal = {}, {}, {}, {}, {}, {} ,[]
fbStoreCentroids_old, fbStoreAreas_old, fbStoreRectParams, fbStoreRectParams_old, fbStoreCulprits = {}, {}, {}, {}, {}
predictCentroidDiff = {}
def mainer(index):
    global shareData, globalCounter, contoursAll, contours_old, frozenBubs, frozenBubsTimes, fbStoreCentroids_old, fbStoreAreas_old, fbStoreRectParams, fbStoreRectParams_old, fbStoreCulprits, contours_old
    global ringBubMasks_old, ringBubImages_old, ringBubRectParams_old, ringBubCentroids_old, ringCntrIDs, ringOldNewIDs_old
    global distanceBubMasks_old, distanceBubImages_old, distanceBubRectParams_old,  distanceBubCentroids_old, distanceOldNewIDs_old
    global distanceBubMasks, distanceBubImages, distanceBubRectParams,  distanceBubCentroids, distanceOldNewIDs
    global centroidsByID2,recParamsByID,areasByID,masksByID,imagesByID,cntrIDsByIDs,bubTypesByID,bubTypesLocal,bubTypesLocal_old,cntrChildrenIDsByIDs
    global predictCentroidDiff
    orig0 = X_data[index]
    if 1==12:
        # orig = cv2.circle(orig.copy() * 0, (300,500), 250, 255, -1)
        orig0 = cv2.circle(orig0.copy() , (700,300), 20, 255, -1)
        # orig = cv2.circle(orig.copy() , (700,359), 30, 255, -1)
        # orig0 = orig.copy()
        # cv2.imshow('orig',orig)
    origRGB = convertGray2RGB(orig0)
    if 1==1:
        global orig
        orig = orig0 -cv2.blur(mean, (5,5),cv2.BORDER_REFLECT)
    
        orig[orig < 0] = 0                  # if orig0 > mean
        orig = orig.astype(np.uint8)
        # blank = orig0.copy() * 0
        # cv2.imshow('mean',mean.astype(np.uint8))
     
    if 1==1:
        global err
        # subMasks_old, subImages_old = {}, list()    
        #err0 = err.copy()
        _,err = cv2.threshold(orig.copy(),thresh0,255,cv2.THRESH_BINARY)
        err = cv2.morphologyEx(err.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        # err = removeBorderComponents(err,2,10)    
        
        
    # if globalCounter>0:
        # global orig_prev, err_prev
        # orig_prev, err_prev = orig.copy() ,err.copy()
        
        
        # err = maskDilateErode(img = origRGB,mask = err,kernel = 14,iters = 1, order = 1, graphics = 1, name = 'global')
    
    # remove border components https://stackoverflow.com/questions/65534370/remove-the-element-attached-to-the-image-border
    
       
    # qq =  maskSegmentAndHull(img = orig, mask = err, minArea = 350, graphics = 0, name = 1)
    # cv2.imshow("qq",qq)
    if workBigArray == 1  and readSingleFromArray == 1: gfx = 1
    else: gfx = 0
    if workBigArray == 0: gfx = 1

    global contours4, mergeCandidates, bubTypesLocal, bubTypesLocal_old
    mergeCandidates,mergeCandidatesSubcontourIDs = {},{}
    recoveredMaskElse, recoveredImgElse, recoverRectParamsElse,recoverCentroidsElse, recoverOldNewIDsElse = {} , {} , {} , {} ,{}
    bubTypesLocal, bubTypesLocal, bubTypesLocal_old = {}, {}, {}
    ringBubMasks, ringBubImages, ringBubRectParam, ringBubCentoids, ringOldNewIDs = {}, {}, {}, {}, {} # temp placeholders, redefined in the end
    recoveredMaskRB, recoveredImgRB, recoverRectParamsRB,recoveredCentroidsRB, recoverOldNewIDsRB = {} , {} , {} , {} , {} # moved A copy for sake of globalCounter > 0
    
    try:
        (contours4, whereParentOriginal,
         whereParentAreaFiltered,whereChildrenAreaFiltered) = cntParentChildHierarchy(err,1, 1200,100,0.1) # whereParentOriginal all non-child contours
        contoursAll[globalCounter] = contours4
        print(f'whereChildrenAreaFiltered {whereChildrenAreaFiltered}')
        global contoursFilter_RectParams,contoursFilter_RectParams_dropIDs
        contoursFilter_RectParams_dropIDs,fbStoreCentroids = [], {}
        contoursFilter_RectParams = {ID: cv2.boundingRect(contours4[ID]) for ID in whereParentOriginal} # maybe mod if for all contours
        # Bubbles boundingRect: x+w < 80 pix <-> filter out bubbles at left image edge
        contoursFilter_RectParams_dropIDs = [ID for ID,params in contoursFilter_RectParams.items() if sum(params[0:3:2])<80]
        # Additional size filter
        minArea = 160 # used in distance bub filtering
        fbStoreAreas = {key: cv2.contourArea(contours4[key]) for key in contoursFilter_RectParams if key not in contoursFilter_RectParams_dropIDs } # dropped since never used
        contoursFilter_RectParams_dropIDs = contoursFilter_RectParams_dropIDs + [key for key, area in fbStoreAreas.items() if area < minArea]
        

        jointMasks = {**ringBubMasks_old, **distanceBubMasks_old}  # possibly no need for dist bubs here!
        jointRectParams = {**ringBubRectParams_old, **distanceBubRectParams_old}
        jointImages = {**ringBubImages_old, **distanceBubImages_old}
        jointCentroids = {**ringBubCentroids_old,**distanceBubCentroids_old}
        jointSubIDs = {**ringOldNewIDs_old,**distanceOldNewIDs_old}
           

        fbStoreAreas = {key: area for key, area in fbStoreAreas.items() if area > minArea}
        fbStoreRectParams = {key: val for key, val in contoursFilter_RectParams.items() if key in fbStoreAreas}
        fbStoreCentroids = {key: getCentroidPosContours(bodyCntrs = [contours4[key]])[0] for key in fbStoreRectParams}
        frozenIDs = []
        if globalCounter >= 1:
            print(f'{globalCounter}:-------- Begin search for frozen bubbles ---------')
        
            joinAreas = {oldID: sum([fbStoreAreas_old[subID] for subID in subIDs]) for oldID, subIDs in jointSubIDs.items()}# no children- bigger area for rings, no hull- fails if type changes
            stuckRectParams = {**fbStoreRectParams_old,**{str(key):val for key,val in jointRectParams.items()}}
            stuckAreas = {**fbStoreAreas_old,**{str(key):val for key,val in joinAreas.items()}}
            stuckCentroids = {**fbStoreCentroids_old,**{str(key):val for key,val in jointCentroids.items()}}
            #joinAreas2 = {oldID: np.sum(mask)/255 for oldID, mask in jointMasks.items()}
            # I want to join old singles and clusters , check them vs new singles. see if there are multiple singles/+cluster point at new single, select most viable, compare to global fb storage and add to old or new
            fields = [stuckRectParams,fbStoreRectParams,stuckAreas,fbStoreAreas,stuckCentroids,fbStoreCentroids,fbStoreCulprits]
            frozenIDs,frozenIDsInfo = detectStuckBubs(*fields, globalCounter, relArea = 0.2, relDist = 5)
            #_,_ = detectStuckBubs(fbStoreRectParams_old,fbStoreRectParams,fbStoreAreas_old,fbStoreAreas,fbStoreCentroids_old,fbStoreCentroids,fbStoreCulprits,globalCounter)
            #frozenIDs,frozenIDsInfo = detectStuckBubs(jointRectParams,fbStoreRectParams,joinAreas,fbStoreAreas,jointCentroids,fbStoreCentroids,fbStoreCulprits,globalCounter)
            # IDs are held in fbStoreCulprits (appended each time)
            print(f'fbStoreCulprits:{fbStoreCulprits}')
            #frozenIDs = []
            #frozenIDsInfo = frozenIDsInfo + frozenIDsInfo2
            if  1 == 1:
                strRavel = ["oldLocID: ", ", newLocID: ",", c-c dist: ",", rArea: ",", centroid: "]
                formats = ["{}", "{}", "{:.2f}", "{:.2f}", "{}"]
                frozenIDsInfoFMRT = [[frmt.format(nbr) for nbr,frmt in zip(elem, formats)] for elem in frozenIDsInfo] if len(frozenIDsInfo)> 0 else []
                peps = ["".join(np.ravel([strRavel,k], order='F')) for k in frozenIDsInfoFMRT] if len(frozenIDsInfoFMRT)> 0 else []
                peps = "\n".join(peps) if len(frozenIDsInfo)> 0 else []
                print(f'Detected Frozen bubbles:\n{peps}')
            print(f'{globalCounter}:------------- Frozen bubbles detected ------------\n') if len(frozenIDsInfo) > 0 else print(f'{globalCounter}:------------- No frozen bubbles found ------------\n')
        
        
         # 1.1 ------------- GRAB RING BUBBLES RB ----------------------
        #for (cntPi,cntP) in enumerate(contours4[whereParentAreaFiltered]): #subset contours
        for (newID,cntP) in zip(whereParentAreaFiltered,contours4[whereParentAreaFiltered]):
            #temp = list()
            #newID = whereParentAreaFiltered[cntPi]#;print(newID);print(whereParentAreaFiltered)
            #cpr = getCentroidPos(inp = cntP, offset = (0,0), mode=0, mask=[])
            cpr,_ = getCentroidPosContours(bodyCntrs = [cntP],holesCntrs=[])
            if newID in whereChildrenAreaFiltered:
                ringBubCentoids[newID] = cpr 
                # save outer ring bubble mask to temp storage for next iteration
                x,y,w,h = cv2.boundingRect(cntP)
                baseSubMask,baseSubImage = err.copy()[y:y+h, x:x+w], orig.copy()[y:y+h, x:x+w]
                # could have filled children 0 value on top of 255 parent. but code structure currently bad.
                subSubMask = cv2.drawContours( np.zeros((h,w),np.uint8),   [cntP], -1, 255, -1,offset = (-x,-y))
                baseSubMask[subSubMask == 0] = 0
                baseSubImage[subSubMask == 0] = 0
                    
                ringBubMasks[newID]       = baseSubMask
                ringBubImages[newID]      = baseSubImage
                ringBubRectParam[newID]   = [x,y,w,h]
                ringOldNewIDs[newID]      = [newID]
                    
                
                # for cSel in whereChildrenAreaFiltered[newID]:
                #     ccr = getCentroidPos(inp = contours4[cSel], offset = (0,0), mode=0, mask=[])
                #     temp.append([cpr,ccr])
    
                # childParentCentroids.append(temp)
               
        # parentChildNorms = [
        #                     [np.linalg.norm(np.array(p1) - np.array(p2)) for (p1,p2) in A]
        #                     for A in childParentCentroids
        #                     ]
        
     

        # -----  calculate centroid-centroid distance for old and new ring bubbles
        # -----  take only ones that satisfy criteria
        
        # intiate predictCentroidDiff values as initial guess
        timeZeroDistCritRing = 8
        timeZeroDistCritElse = 30
        timeZeroStd = 3 
        for ID in list(ringBubCentroids_old.keys()):
            if ID not in predictCentroidDiff: predictCentroidDiff[ID] = {globalCounter-1:[[0,0],timeZeroDistCritRing,timeZeroDistCritRing,timeZeroStd]} # [vec, val, mean, std]
        for ID in list(distanceBubCentroids_old.keys()):
            if ID not in predictCentroidDiff: predictCentroidDiff[ID] = {globalCounter-1:[[0,0],timeZeroDistCritElse,timeZeroDistCritElse,0]}
        
        #tempDistListTest = list()
        # -----------------recover old-new ring bubble relations--------------------------
        RBOldNewDist = np.empty((0,2), np.uint16)
        RBOldNewDist2 = RBOldNewDist.copy()
        if globalCounter >= 1: # << uses centroids as it goes, any changes due to back-tracking are not accounted.
            for oldID, oldCentroid in ringBubCentroids_old.items(): 
                cntrds = np.array(list(centroidsByID2[oldID].values()))
                _,_, distCheck2, distCheck2Sigma  = predictCentroidDiff[oldID][globalCounter-1]
                sigmasDeltas = [a[2:] for a in predictCentroidDiff[oldID].values()]
                predVec_old = predictCentroidDiff[oldID][globalCounter-1][0]
                predictCentroid = distStatPredictionVect2(cntrds, sigmasDeltas = sigmasDeltas[-1], sigmasDeltasHist = predictCentroidDiff[oldID],
                                            numdeltas = 5, maxInterpSteps = 3, maxInterpOrder = 2, debug =  1, savePath = predictVectorPathFolder,
                                            predictvec_old = predVec_old, bubID = oldID, timestep = globalCounter, zerothDisp = [-3,0])
                print(f'oldID: {oldID}, distCheck2: {distCheck2}, distCheck2Sigma: {distCheck2Sigma}')
                pathLen = len(predictCentroidDiff[oldID])
                for newID, newCentroid in list(ringBubCentoids.items()):
                    dist2 = np.linalg.norm(np.array(predictCentroid) - np.array(newCentroid)).astype(np.uint32)
                    oldArea = sum([fbStoreAreas_old[subOldID] for subOldID in ringOldNewIDs_old[oldID]])
                    newArea = fbStoreAreas[newID] # i hope new RB has only 1 contour
                    relArea = abs(oldArea - newArea)/oldArea
                    print(f'oldID:{oldID}; newID:{newID}, dist2:{dist2:0.1f}, relArea:{relArea:0.2f}')
                    if dist2 <= distCheck2 + 5*distCheck2Sigma and relArea < 0.2: #predictCentroidDiff may be missing if failed, probly resolved because ID gon be dropped and wont appear in next cycle
                        newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, dist2)
                        predictCentroidDiff[oldID][globalCounter] = [predictCentroid, dist2, newMean, newStd]
                        RBOldNewDist = np.append(RBOldNewDist,[[oldID,newID]],axis=0)
                    #fbStoreAreas  fbStoreAreas_old ringOldNewIDs_old
            print(f'RBOldNewDist:{list(map(list,RBOldNewDist))}')    
            print('**analyzing new Ring and old Ring/Recover bubbles**')
            newFoundBubsRings = list(ringBubCentoids.keys())
            oldFoundRings = list(ringBubCentroids_old.keys())
            if debugOnly(11):
                print("newFoundBubsRings (new typeRing bubbles):\n",newFoundBubsRings)
                print("oldFoundRings (all old R,rR bubbles):\n",oldFoundRings)
                # print('tempDistListTest (distance relation all dist, distCheck):\n',tempDistListTest)
                print("RBOldNewDist (distance relations < distCheck):\n",list(map(list,RBOldNewDist)))
            
            if len(RBOldNewDist) > 0: # if all distance checks fail (elseOldNewDist)
                newFoundBubsRings = [A for A in newFoundBubsRings if A not in RBOldNewDist[:,1]]
                oldFoundRings = [A for A in oldFoundRings if A not in RBOldNewDist[:,0]]
                if debugOnly(11):
                    print("newFoundBubsRings (unresolved new indices)\n",newFoundBubsRings)
                    print("oldFoundRings (updated, unresolved old R,rR IDs)\n",oldFoundRings)
            # recoveredMaskRB, recoveredImgRB, recoverRectParamsRB, recoverOldNewIDsRB, recoveredCentroidsRB = {} , {} , {} , {} , {}
            
            # 1.2 ------------- RECOVER old unresolved RB and recovred RB USING matchTemplate() -----------------------
            
            if len(oldFoundRings)>0:    print(f'{globalCounter}:--------- Begin recovery of Ring bubbles ---------\n')
            else:                       print(f'{globalCounter}:-------------- No RBubs to recover ---------------\n')

            recoveredBubsRelation = {}
            gfx = 1 if debugOnlyGFX(12) else 0
            for i in oldFoundRings.copy(): # take unresolved R,rR old global IDs (i)
                print(f'{globalCounter}:Trying to recover old ring {i}')
                (x,y,w,h) = matchTemplateBub(err,jointMasks[i],jointRectParams[i],graphics=gfx,prefix = f'MT: Time: {globalCounter}, RB_old_ID: {i} ')
                (tempMask, [xt,yt,wt,ht], overlapingContourIDList) = \
                    overlapingContours(contours4, whereParentOriginal, jointMasks[i], (x,y,w,h), gfx, prefix = f'OC: Time: {globalCounter}, RB_old_ID: {i} ')
                
                if len(overlapingContourIDList)>0:
                    
                    mergeCandidatesSubcontourIDs[i] = overlapingContourIDList
                    for ID in overlapingContourIDList:
                        if ID not in mergeCandidates: mergeCandidates[ID]  = []
                        mergeCandidates[ID].append(i)
                        
                    baseSubMask,baseSubImage = err.copy()[yt:yt+ht, xt:xt+wt], orig.copy()[yt:yt+ht, xt:xt+wt]
                    baseSubImage[tempMask == 0] = 0
                    shapePass = compareMoments(big=err,
                                                   shape1=baseSubImage,shape2=jointImages[i],
                                                   coords1=[xt,yt,wt,ht],coords2 = jointRectParams[i],debug = debugOnly(12))
                    print(f'recovery of {i} {overlapingContourIDList} completed') if shapePass == 1 else print(f'recovery of {i} {overlapingContourIDList} failed.')
                    if shapePass == 1:# i- globals
                        tempC = getCentroidPos(inp = tempMask, offset = (xt,yt), mode=0, mask=[])
                        recoveredMaskRB[i]      = tempMask 
                        recoveredImgRB[i]       = baseSubImage
                        recoverRectParamsRB[i]  = [xt,yt,wt,ht]
                        recoveredCentroidsRB[i] = tempC
                        recoverOldNewIDsRB[i]   = overlapingContourIDList
                        # bubTypesLocal[i] = typeRecoveredRing
                        bubTypesByID[i][globalCounter] = typeRecoveredRing
                        # centroidsByID2[i][globalCounter] = tempC

                        oldC = jointCentroids[i];print(f'**** oldC:{oldC}, tempC:{tempC}')
                        pathLen = len(centroidsByID2[i])
                        _,_, distCheck2, distCheck2Sigma  = predictCentroidDiff[i][globalCounter-1]
                        print(f'i: {i}, distCheck2: {distCheck2}, distCheck2Sigma: {distCheck2Sigma}')
                        dist = np.linalg.norm(np.diff([tempC,oldC],axis=0),axis=1)[0]
                        newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, dist)
                        predictCentroidDiff[i][globalCounter] = [oldC, dist, newMean, newStd]

                        #oldC = jointCentroids[i];print(f'**** oldC:{oldC}, tempC:{tempC}')
                        #newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, permDist2)
                        #predictCentroidDiff[i][globalCounter] = [permDist2, newMean, newStd]
                        #predictCentroidDiff[i][globalCounter] = np.linalg.norm(np.diff([tempC,oldC],axis=0),axis=1)[0]
                        oldFoundRings.remove(i)
                        print(f'Recovered old ring {i}, oldFoundRings (remaining unresolved): {oldFoundRings}')  if debugOnly(12) else 0
                        
                        # if i in oldFoundRings: oldFoundRings.remove(i)
                        # <<<<< i dont expect to recover multiple new from one old, but in case check this block! (*EDIT: IDK IF RELEVANT)
                        newRBinTemp = [A for A in newFoundBubsRings if A in overlapingContourIDList]
                        if len(newRBinTemp) > 1: print('<<<<<<<<<<<<len(newRBinTemp) > 1 !!!!!!!!!!!!!!!!!!!!!!!')
                        if len(newRBinTemp) > 0:
                            recoveredBubsRelation[i] = newRBinTemp[0] # storage is mapped global key->global key instead of local-> global. this to recover index
                            newFoundBubsRings = [ ID for  ID in newFoundBubsRings if ID not in newRBinTemp]
                            print(f'Recovered new rings {newRBinTemp}, newFoundBubsRings (remaining unresolved): {newFoundBubsRings}') if debugOnly(12) else 0
                        else: recoveredBubsRelation[i] = min(overlapingContourIDList)
                        
                print('recoveredBubsRelation',recoveredBubsRelation) if debugOnly(21) else 0
            print('mergeCandidates',mergeCandidates) if debugOnly(21) else 0
        
            
        # 2.1 ====================== RECOVER UNRESOLVED BUBS VIA DISTANCE CLUSTERING ======================= 
        print(f'{globalCounter}:--------- Cluster remaining via distance ---------') 
        # single frame analysis, do this when everything else is resolved.   
        if globalCounter >= 0:      
            #  GRAB ALL contours except resolved RBs and size-location filtered
            dropResolvedRingIDs = list(ringBubRectParam.keys())+sum(list(recoverOldNewIDsRB.values()),[]) # <<< ringBubRectParam does not hold info on children | idk if relevant
            # whereParentOriginal all non-child contours 
            cntrRemainingIDs = [cID for cID in whereParentOriginal if cID not in dropResolvedRingIDs + contoursFilter_RectParams_dropIDs ]
            cntrRemaining = {ID:contours4[ID] for ID in cntrRemainingIDs}
             # calculate centroid to neighbor contour distance and filter by some asumptions
            strgD_anis_IDs = []
            edges = []
            selfCombs = list(itertools.combinations(cntrRemaining,2))
            charBubSize =  100
            x_weight, y_weight = 4, 1 # elongates vectors in x direction. horizontal vecs fail more.
            smx = np.array( [[x_weight,0],[0,y_weight]])
            for i,j in selfCombs:
                vec, dist, p1, p2 = closes_point_contours(cntrRemaining[i],cntrRemaining[j])
                vec_scaled = np.matmul(smx,np.array(vec))
                dist_scaled = np.linalg.norm(vec_scaled)
                strgD_anis_IDs.append([vec_scaled, dist_scaled, i, j, p1, p2])
                if dist_scaled < charBubSize : edges.append(tuple([i,j]))
            # ------- draw distance bubbles with connections ---------
            if debugOnlyGFX(21):
                blank  = np.zeros(err.shape,np.uint8)
                blank = convertGray2RGB(blank)
                [cv2.drawContours( blank, contours4, i, (125,125,220), -1) for i in cntrRemaining]
                [cv2.polylines(blank, [np.array([p1,p2])],0, (0,255,255), 2) for _, dist_scaled, _, _, p1, p2 in strgD_anis_IDs if dist_scaled < charBubSize]
                [cv2.polylines(blank, [np.array([p1,p2])],0, (0,0,255), 2) for _, dist_scaled, _, _, p1, p2 in strgD_anis_IDs if dist_scaled > charBubSize]
                frac = 0.7                                                                                  # IMSHOW
                dims = (np.array(err.shape)*frac).astype(np.uint16)#;print(dims)
                cv2.imshow(f'32 {globalCounter}',cv2.resize(blank,tuple((1*dims[1],1*dims[0])), interpolation= cv2.INTER_LINEAR) )    
            
            # make A graph of contours connections specified by anisatropic distance.
                        
            H = nx.Graph()
            H.add_nodes_from(cntrRemaining)
            H.add_edges_from(edges)
            # ----- visualize  netrworkx graph with background contrours
            if globalCounter == 123120: 
                pos = {i:getCentroidPos(inp = vec, offset = (0,0), mode=0, mask=[]) for i, vec in cntrRemaining.items()}
                for n, p in pos.items():
                        H.nodes[n]['pos'] = p
                plt.figure(1)
                for cntr in list(cntrRemaining.values()):
                    [x,y] = np.array(cntr).reshape(-1,2).T
                    plt.plot(x,y)
                nx.draw(H, pos)
            
            # extract clusters that are connected (undirected)    
            cnctd_comp = [list(sorted(nx.node_connected_component(H, key))) for key in cntrRemaining]
            cc_unique = [];[cc_unique.append(x) for x in cnctd_comp if x not in cc_unique] # remove duplicates
            print('Connected Components: ',cnctd_comp,'\nCC unique:', cc_unique) if debugOnly(21) else 0

            # forzen bubs (FB) can be falsely joint for being too close. or it can be false positive FB
            
            #------------------------------------------------------------------------------- deal with  frozen bubbles!! ----------------------------------------------
            frozenSeparated = False
            for i,comb in enumerate(cc_unique.copy()):
                for fID in frozenIDs:
                    if fID in comb:
                        cc_unique[i].remove(fID)
                        cc_unique.append([fID])
                        frozenSeparated = True
                        

            jointNeighbors = {min(elem): elem for elem in cc_unique if len(elem)>0}
            clusterMsg = 'Cluster groups: ' if frozenSeparated == False else 'Cluster groups (frozenIDs detected): '
            print(clusterMsg,"\n", jointNeighbors)
            # collect multiple contours into one object, store it for further check
            # on first iteration there are no more checks to be done. store it early
            if globalCounter == 0:
                blank  = np.zeros(err.shape,np.uint8)
                blank = convertGray2RGB(blank)                                                              # IMSHOW
                distanceBubMasks, distanceBubImages, distanceBubRectParams, distanceBubCentroids, distanceOldNewIDs = {},{},{},{},{}
    
                for key, cntrIDlist in jointNeighbors.items():
                    #[cv2.drawContours( blank, contours4, ID, cyclicColor(key), -1) for ID in cntrIDlist]    # IMSHOW
                    distCntrSubset = np.vstack([contours4[ID] for ID in cntrIDlist])#;print(distCntrSubset.shape)
                    x,y,w,h = cv2.boundingRect(distCntrSubset)#;print([x,y,w,h])
                    tempID = min(cntrIDlist)  #<<<<<<<<<<< maybe check if some of these contours are missing elsewhere
                
                    baseSubMask,baseSubImage = err.copy()[y:y+h, x:x+w], orig.copy()[y:y+h, x:x+w]
                    subSubMask = np.zeros((h,w),np.uint8)
                    [cv2.drawContours( subSubMask, contours4, ID, 255, -1, offset = (-x,-y)) for ID in cntrIDlist]
    
                    baseSubMask[subSubMask == 0] = 0
                
                    baseSubImage[subSubMask == 0] = 0
                
                    distanceBubMasks[tempID] = baseSubMask
                    distanceBubImages[tempID] = baseSubImage
                    distanceOldNewIDs[tempID] = cntrIDlist
                    distanceBubRectParams[tempID] = ([x,y,w,h])
                    distanceBubCentroids[tempID] = getCentroidPos(inp = baseSubMask, offset = (x,y), mode=0, mask=[])
                #print('distanceBubCentroids',distanceBubCentroids)
                #cv2.imshow('gfx debug 22: jointNeighbors',blank) if debugOnlyGFX(22) else 0

                
            # -------------- RECOVER UNRESOLVED BUBS VIA DISTANCE CLUSTERING ---------------------   
            # -------------- join with newFoundBubsRings and search relations ----------------- 
            # jointNeighbors is a rough estimate. In an easy case neighbor bubs will be far enough away
            # so clusters wont overlap. rather strict area/centroid displ restrictions can be satisfied
            # if it fails, take overlapped cluster IDs and start doing permutations and check dist/areas
            if globalCounter >= 1: # unresolved new + distance bubs
                elseOldNewDoubleCriterium = []
                elseOldNewDoubleCriteriumSubIDs = {}
                elseOldNewDist = []
                jointNeighborsWoFrozen = {mainNewID: subNewIDs for mainNewID, subNewIDs in jointNeighbors.items() if mainNewID not in frozenIDs}
                jointNeighborsOnlyFrozen = {mainNewID: subNewIDs for mainNewID, subNewIDs in jointNeighbors.items() if mainNewID in frozenIDs}
                    
                for oldID, oldCentroid in distanceBubCentroids_old.items():
                    cntrds = list(centroidsByID2[oldID].values())
                    distCheck = distStatPrediction(trajectory = cntrds,startAmp0 = 30, expAmp = 10, halfLife = 2, numsigmas = 2, plot=0,extraStr = f'E:ID {oldID} ')
                    _,_, distCheck2, distCheck2Sigma  = predictCentroidDiff[oldID][globalCounter-1]
                    sigmasDeltas = [a[2:] for a in predictCentroidDiff[oldID].values()]
                    #predictCentroid = distStatPredictionVect(cntrds, zerothDisp = [-1,0], sigmasDeltas = sigmasDeltas, numdeltas = 5, maxInterpSteps = 3, maxInterpOrder = 1, mode = 1,debug = testPred, maxNumPlots = 4)
                    print(f'oldID: {oldID}, distCheck: {distCheck}, distCheck2: {distCheck2}, distCheck2Sigma: {distCheck2Sigma}')
                    pathLen = len(predictCentroidDiff[oldID])
                    predVec_old = predictCentroidDiff[oldID][globalCounter-1][0]
                    predictCentroid = distStatPredictionVect2(cntrds, sigmasDeltas = sigmasDeltas[-1], sigmasDeltasHist = predictCentroidDiff[oldID],
                                            numdeltas = 5, maxInterpSteps = 3, maxInterpOrder = 2, debug =  1, savePath = predictVectorPathFolder,
                                            predictvec_old = predVec_old, bubID = oldID, timestep = globalCounter, zerothDisp = [-3,0])
                    #print(f'old vs new predict. predictCentroid:{predictCentroid}, oldCentroid:{oldCentroid}')
                    areaCheck = getContourHullArea([contours_old[subOldID] for subOldID in distanceOldNewIDs_old[oldID]])
                    
                    #----------------- looking for new else bubs related to old else bubs ---------------------
                    for mainNewID, subNewIDs in jointNeighborsWoFrozen.items():
                        newCentroid, newArea = getCentroidPosContours(bodyCntrs = [contours4[k] for k in subNewIDs], hullArea = 1)
                        relArea = abs(areaCheck - newArea)/areaCheck
                        dist = np.linalg.norm(np.array(newCentroid) - np.array(oldCentroid)).astype(np.uint32)
                        dist2 = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid)).astype(np.uint32)
                        #print(f'old vs new dist. dist:{dist}, dist2:{dist2}')
                        if dist <= distCheck:
                            #print(tuple([oldID,mainNewID,dist,relArea]))
                            elseOldNewDist.append([oldID,mainNewID])
                        
                        if dist2 <= distCheck2 + 5*distCheck2Sigma and relArea <= 0.1:
                            print(f'\ndist 2. Perfect match: {oldID} <-> mainNewID: {mainNewID}, dist2: {dist2:0.1f}, relArea: {relArea:0.1f}')
                            newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, dist2)
                            predictCentroidDiff[oldID][globalCounter] = [predictCentroid, dist2, newMean, newStd]
                            #print(f'oldID:{oldID}; mainNewID:{mainNewID}, dist2:{dist2}, newMean: {newMean}, newStd: {newStd}')
                            elseOldNewDoubleCriterium.append([oldID,mainNewID,dist,relArea])
                            elseOldNewDoubleCriteriumSubIDs[mainNewID] = subNewIDs
                        elif dist2 > 70 or relArea > 1:
                            0#;print('dist 2. soft break')
                        else:
                            debug = 1 if (globalCounter == 1 and oldID == 3) else 0
                            permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(contours4, subNewIDs, fbStoreCentroids, fbStoreAreas,
                                                        predictCentroid, distCheck2 + 5*distCheck2Sigma, areaCheck, relAreaCheck = 0.7, debug = debug)
                            if len(permIDsol2)>0:
                                print(f'\ndist 2. Permutation reconstruct. oldID: {oldID}, subNewIDs: {subNewIDs}, permIDsol2: {permIDsol2}, permDist2: {permDist2}')
                                newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, permDist2)
                                predictCentroidDiff[oldID][globalCounter] = [predictCentroid, permDist2, newMean, newStd]
                                print(f'oldID:{oldID}; mainNewID:{mainNewID}, permDist2:{permDist2}, newMean: {newMean}, newStd: {newStd}')
                                elseOldNewDoubleCriterium.append([oldID,min(permIDsol2),dist,permRelArea2])
                                elseOldNewDoubleCriteriumSubIDs[min(permIDsol2)] = permIDsol2

                    #----------------- looking for new rings related to old else bubs ---------------------
                    for unresNewRBID in newFoundBubsRings:
                        assert 11<0, "going into untested code!!!!! (for unresNewRBID in newFoundBubsRings)"
                        subNewIDs = unresNewRBID#jointSubIDs[]
                        newCentroid, newArea = getCentroidPosContours(bodyCntrs = [contours4[unresNewRBID]], hullArea = 1)
                        relArea = abs(areaCheck - newArea)/areaCheck
                        dist2 = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid)).astype(np.uint32)
                        if dist2 <= distCheck2 + 5*distCheck2Sigma and relArea <= 0.1:
                            print(f'\ndist 2. Perfect match: {oldID} <-> mainNewID: {mainNewID}, dist2: {dist2:0.1f}, relArea: {relArea:0.1f}')
                            newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, dist2)
                            predictCentroidDiff[oldID][globalCounter] = [predictCentroid, dist2, newMean, newStd]
                            print(f'oldID:{oldID}; mainNewID:{mainNewID}, dist2:{dist2}, newMean: {newMean}, newStd: {newStd}')
                            elseOldNewDoubleCriterium.append([oldID,mainNewID,dist,relArea])
                            elseOldNewDoubleCriteriumSubIDs[mainNewID] = subNewIDs
                        elif dist2 > 70 or relArea > 1:
                            0#;print('dist 2. soft break')
                        else:
                            debug = 1 if (globalCounter == 1123123 and oldID == 3) else 0
                            permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(contours4, subNewIDs, fbStoreCentroids, fbStoreAreas,
                                                        predictCentroid, distCheck2 + 5*distCheck2Sigma, areaCheck, relAreaCheck = 0.7, debug = debug)
                            if len(permIDsol2)>0:
                                print(f'\ndist 2. Permutation reconstruct. oldID: {oldID}, subNewIDs: {subNewIDs}, permIDsol2: {permIDsol2}, permDist2: {permDist2}')
                                newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, permDist2)
                                predictCentroidDiff[oldID][globalCounter] = [predictCentroid, permDist2, newMean, newStd]
                                print(f'oldID:{oldID}; mainNewID:{mainNewID}, permDist2:{permDist2}, newMean: {newMean}, newStd: {newStd}')
                                elseOldNewDoubleCriterium.append([oldID,min(permIDsol2),dist,permRelArea2])
                                elseOldNewDoubleCriteriumSubIDs[min(permIDsol2)] = permIDsol2

                print(f'elseOldNewDoubleCriterium: {[listFormat(data, formatList = ["{:.0f}", "{:0.0f}", "{:.2f}","{:0.2f}"], outType = float) for data in elseOldNewDoubleCriterium]}')
                print(f'elseOldNewDoubleCriteriumSubIDs: {elseOldNewDoubleCriteriumSubIDs}')
                
                # modify jointNeighbors by splitting clusters using elseOldNewDoubleCriteriumSubIDs
                removeFoundSubIDsAll = sum(elseOldNewDoubleCriteriumSubIDs.values(),[])
                # delete new cluster IDs from original jointNeighbors, gather remaining and new found in jointNeighbors22
                jointNeighborsDelSubIDs = [ [subelem for subelem in elem if subelem not in removeFoundSubIDsAll] for elem in jointNeighborsWoFrozen.values()]
                jointNeighbors_old = jointNeighbors.copy()
                jointNeighbors = {**{min(vals):vals for vals in jointNeighborsDelSubIDs if len(vals) > 0},**elseOldNewDoubleCriteriumSubIDs}
                jointNeighbors = {**jointNeighbors, **jointNeighborsOnlyFrozen}
                # forozen bubs offer only old local IDs. find local-global relation in jointSubIDs and relpace them.
                oldLocIDs = frozenIDsInfo[:,0]
                asd = list(map(type,oldLocIDs));print(f'asd:{asd}')
                oldGlobIDs0 = [
                                {ii:ID for ID, vals in jointSubIDs.items() if ii in vals} if type(ii) != str else {ii:int(ii)} 
                               for ii in oldLocIDs ] # relevant new:old dict
                print(f'oldGlobIDs0: {oldGlobIDs0}')
                oldGlobIDs = {};[oldGlobIDs.update(elem) for elem in oldGlobIDs0]
                for oldLocID, newLocID, dist, rArea, centroid in frozenIDsInfo:
                    oldID = oldGlobIDs[oldLocID]
                    pathLen = len(predictCentroidDiff[oldID])
                    _,_, distCheck2, distCheck2Sigma  = predictCentroidDiff[oldID][globalCounter-1]
                    newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, dist)
                    predictCentroidDiff[oldID][globalCounter] = [centroid, dist, newMean, newStd]
                    elseOldNewDoubleCriterium.append([oldGlobIDs[oldLocID], newLocID, dist, rArea])
                #[elseOldNewDoubleCriterium.append([oldGlobIDs[oldLocID], newLocID, dist, rArea]) for oldLocID, newLocID, dist, rArea, *_ in frozenIDsInfo]
                print(f'Cluster groups (updated): {jointNeighbors}') if jointNeighbors != jointNeighbors_old else print('Cluster groups unchanged')
                
                #----------------- consider else clusters are finalized ---------------------
                distanceBubMasks, distanceBubImages, distanceBubRectParams, distanceBubCentroids, distanceOldNewIDs = {},{},{},{},{}
                for key, cntrIDlist in jointNeighbors.items():
                    #[cv2.drawContours( blank, contours4, ID, cyclicColor(key), -1) for ID in cntrIDlist]    # IMSHOW
                    distCntrSubset = np.vstack([contours4[ID] for ID in cntrIDlist])#;print(distCntrSubset.shape)
                    x,y,w,h = cv2.boundingRect(distCntrSubset)#;print([x,y,w,h])
                    tempID = min(cntrIDlist)  #<<<<<<<<<<< maybe check if some of these contours are missing elsewhere
                
                    baseSubMask,baseSubImage = err.copy()[y:y+h, x:x+w], orig.copy()[y:y+h, x:x+w]
                    subSubMask = np.zeros((h,w),np.uint8)
                    [cv2.drawContours( subSubMask, contours4, ID, 255, -1, offset = (-x,-y)) for ID in cntrIDlist]
    
                    baseSubMask[subSubMask == 0] = 0
                
                    baseSubImage[subSubMask == 0] = 0
                
                    distanceBubMasks[tempID] = baseSubMask
                    distanceBubImages[tempID] = baseSubImage
                    distanceOldNewIDs[tempID] = cntrIDlist
                    distanceBubRectParams[tempID] = ([x,y,w,h])
                    distanceBubCentroids[tempID] = getCentroidPos(inp = baseSubMask, offset = (x,y), mode=0, mask=[])
                #print('distanceBubCentroids',distanceBubCentroids)
                #cv2.imshow('gfx debug 22: jointNeighbors',blank) if debugOnlyGFX(22) else 0
                
                #elseOldNewDist = np.array(elseOldNewDist);
                elseOldNewDist = np.array(elseOldNewDoubleCriterium,np.uint32)[:,:2]#;print(f'elseOldNewDist2: {elseOldNewDist2}')
                # IN CASE there are duplicates!!
                #if len(np.unique(elseOldNewDist[:,0]))!=len(np.unique(elseOldNewDist[:,1])):
                #    sol = centroidSumPermutationsMOD(elseOldNewDist, fbStoreCentroids, fbStoreAreas, distanceBubCentroids_old, 15)
                #    print(f'sol: {sol}')

                newFoundBubsElse = list(jointNeighbors.keys()) + newFoundBubsRings #list(centroidsJoin.keys()); 
                oldFoundElse = list(distanceBubCentroids_old.keys());
                
                if debugOnly(21):
                    #print("elseOldNewDist (Else IDs pass dist)\n", list(map(list,elseOldNewDist)))
                    print(f"elseOldNewDist (Else IDs pass dist, /w subIDs)\n{[[oldID,jointNeighbors[mainNewID]] for oldID,mainNewID in elseOldNewDist]}")
                    print("newFoundBubsElse (new Else IDs)\n",newFoundBubsElse)
                    print("oldFoundElse (all old Else IDs)\n",oldFoundElse)
                if len(elseOldNewDist) > 0: # if all distance checks fail (elseOldNewDist)
                    newFoundBubsElse = [A for A in newFoundBubsElse if A not in elseOldNewDist[:,1]]; 
                    oldFoundElse = [A for A in oldFoundElse if A not in elseOldNewDist[:,0]]; 
                    if debugOnly(21):
                        print("newFoundBubsElse (unresolved new IDs)\n",newFoundBubsElse)
                        print("oldFoundElse (unresolved old Else IDs)\n",oldFoundElse)
                
                        # if RB is "recovered" from E, add it to RB data, remove from unresolved RB
                elseToRingBubIDs = [i for i,[_,ID] in enumerate(elseOldNewDist) if ID in newFoundBubsRings] #index for elseOldNewDist
                newFoundBubsRings = [elem for elem in newFoundBubsRings if elem not in elseOldNewDist[:,1]]
                for i in elseToRingBubIDs:
                    RBOldNewDist = np.vstack((RBOldNewDist,elseOldNewDist[i]))
                elseOldNewDist = np.array([ elem for i,elem in enumerate(elseOldNewDist) if i not in elseToRingBubIDs])
                
                if debugOnly(21) == True and len(elseToRingBubIDs)>0:
                    print(f'newFoundBubsRings, (found RB removed) unrecovered RBs: {newFoundBubsRings}')
                    print("RBOldNewDist, ( new RB added) (distance relations < distCheck):\n",RBOldNewDist)
                    print("elseOldNewDist, (found RB removed) (Else IDs pass dist)\n", elseOldNewDist)
    
                newFoundBubsElse = [elem for elem in newFoundBubsElse if elem not in newFoundBubsRings] # <<<<<<<<<<<   added in case missing RB is not resolved. should rething structure
                
                if len(oldFoundElse)>0: print(f'{globalCounter}:--------- Begin recovery of Else bubbles: {oldFoundElse} --------')
                else:                   print(f'{globalCounter}:----------------- No Else bubble to recover ---------------------')
                
                jointNeighbors = {ID:subIDs for ID,subIDs in jointNeighbors.items() if ID not in elseOldNewDist[:,1]} if len(elseOldNewDist)> 0 else jointNeighbors
                gfx = 1 if debugOnlyGFX(23) else 0
                for i in oldFoundElse.copy(): 
                    print(f'Trying to recover old else {i}')

                    (x,y,w,h) = matchTemplateBub(err,jointMasks[i],jointRectParams[i],graphics=gfx,prefix = f'MT: Time: {globalCounter}, E_old_ID: {i} ')
                    (tempMask, [xt,yt,wt,ht], overlapingContourIDList) = \
                        overlapingContours(contours4, whereParentOriginal, jointMasks[i], (x,y,w,h), gfx, prefix = f'OC: Time: {globalCounter}, RB_old_ID: {i} ')
                    overlapingContourIDListTemp = [ID for ID in overlapingContourIDList if ID not in frozenIDs] # remove frozen IDs
                    if len(overlapingContourIDListTemp)>0:
                        mergeCandidatesSubcontourIDs[i] = overlapingContourIDListTemp
                        for localID in overlapingContourIDListTemp:
                            if localID not in mergeCandidates: mergeCandidates[localID]  = []
                            mergeCandidates[localID].append(i)
                    baseSubMask,baseSubImage = err.copy()[yt:yt+ht, xt:xt+wt], orig.copy()[yt:yt+ht, xt:xt+wt]
                    baseSubImage[tempMask == 0] = 0
                    shapePass = compareMoments(big=err,
                                                    shape1=baseSubImage,shape2=jointImages[i],
                                                    coords1=[xt,yt,wt,ht],coords2 = jointRectParams[i],debug = debugOnly(22))
                    print(f'recovery of {i} {overlapingContourIDList} completed') if shapePass == 1 else print(f'recovery of {i} {overlapingContourIDList} failed.')
                    if shapePass == 1:
                        # account for frozenIDs: keep resolvedEBub, modify mask. frozenID will be left in newFoundBubsElse, if it stay there it will bw asigned typeFrozen
                        if any(item in frozenIDs for item in overlapingContourIDList):
                            overlapingContourIDList = [ID for ID in overlapingContourIDList if ID not in frozenIDs] # no need to copy(), it seems
                            tempMask, baseSubImage, [xt,yt,wt,ht], tempC = getMasksParams(contours4,overlapingContourIDList,err,orig)
                            
                        else:
                            tempC = getCentroidPos(inp = tempMask, offset = (xt,yt), mode=0, mask=[])
                           

                        recoveredMaskElse[i]        = tempMask  # recoveredMaskElse keys are used for _old/global storage
                        recoveredImgElse[i]         = baseSubImage
                        recoverRectParamsElse[i]    = [xt,yt,wt,ht]
                        recoverCentroidsElse[i]     = tempC
                        recoverOldNewIDsElse[i]     = overlapingContourIDList
                        # bubTypesLocal[i]            = typeRecoveredElse
                        bubTypesByID[i][globalCounter] = typeRecoveredElse
                        # centroidsByID2[i][globalCounter] = tempC
                        oldC = jointCentroids[i];print(f'**** oldC:{oldC}, tempC:{tempC}')
                        pathLen = len(centroidsByID2[i])
                        _,_, distCheck2, distCheck2Sigma  = predictCentroidDiff[i][globalCounter-1]
                        print(f'i: {i}, distCheck2: {distCheck2}, distCheck2Sigma: {distCheck2Sigma}')
                        dist = np.linalg.norm(np.diff([tempC,oldC],axis=0),axis=1)[0]
                        newMean, newStd = updateStat(pathLen, distCheck2, distCheck2Sigma, dist)
                        predictCentroidDiff[i][globalCounter] = [oldC,dist, newMean, newStd]
                        
    
                        oldFoundElse.remove(i)

                        # print('overlapingContourIDList',overlapingContourIDList)
                        # resolveElseIDs - get main local cluster IDs which own contours in overlapingContourIDList
                        resolveElseIDs = set(sum([[ID for ID,subIDs in jointNeighbors.items() if foundID in subIDs] for foundID in overlapingContourIDList],[]))
                        newFoundBubsElse = [elem for elem in newFoundBubsElse if elem not in resolveElseIDs] # drop resolved
                        jointNeighbors2 = {A:[ID for ID in subIDs if ID not in overlapingContourIDList] for A,subIDs in jointNeighbors.items()}
                        jointNeighbors = {min(subIDs): subIDs for _,subIDs in jointNeighbors2.items() if len(subIDs) > 0}
                        if debugOnly(22):
                            print(f'resolveElseIDs {resolveElseIDs}')
                            print(f'jointNeighbors {jointNeighbors}')
                            print(f'jointNeighbors2 {jointNeighbors2}')
                            #print(f'jointNeighbors3 {jointNeighbors3}')
                            print(f'C: {globalCounter}, IDE: {i}, recoverOldNewIDsElse[{i}]: {recoverOldNewIDsElse[i]}')
                            print(f'Recovered old else {i}, remaining oldFoundElse: {oldFoundElse}')
                            print("newFoundBubsElse (unresolved  (updated) new IDs)\n",newFoundBubsElse) if len(resolveElseIDs)>0 else 0
        
                if len(jointNeighbors)> 0:
                    for ID,cntrIDs in jointNeighbors.items():
                        tempMask, baseSubImage, [xt,yt,wt,ht], tempC = getMasksParams(contours4,cntrIDs,err,orig)
                        distanceBubMasks[ID] = tempMask
                        distanceBubImages[ID] = baseSubImage
                        distanceBubRectParams[ID] = [xt,yt,wt,ht]
                        distanceBubCentroids[ID] = tempC
                        distanceOldNewIDs[ID] = cntrIDs
                    newFoundBubsElse = newFoundBubsElse + [elem for elem in list(jointNeighbors.keys()) if elem not in newFoundBubsElse]
                        
                print('mergeCandidates',mergeCandidates) if debugOnly(22) else 0
        
        print(f'{globalCounter}:--------- Begin merge detection/processing --------\n')
        
        # ------------------- detect merges by inspecting shared contours ----------------------
        if globalCounter >= 1:
            graphConnections2 = sum([[[str(key), elem] for elem in vals] for key, vals in mergeCandidates.items()], [])# change key to string to prevent int-int overlap
            # print(graphConnections2)
            H = nx.Graph()
            H.add_nodes_from([str(key) for key in mergeCandidates])
            H.add_edges_from(graphConnections2)
            # ----- visualize  netrworkx graph with background contrours
            if debugOnlyGFX(31) == True:
                blank  = np.full(err.shape,128, np.uint8)
                blank = convertGray2RGB(blank)
                [cv2.drawContours( blank, contours4, i, (6,125,220), -1) for i in mergeCandidates]
                [[cv2.drawContours( blank, contoursAll[globalCounter-1], i, (225,125,125), 2) for i in cntrIDsByIDs[oldID][globalCounter-1]] for _,oldID in graphConnections2]
                pos1 = {str(ID):getCentroidPos(inp = contours4[ID], offset = (0,0), mode=0, mask=[]) for ID in mergeCandidates}
                pos2 = {oldID:centroidsByID2[oldID][globalCounter-1] for _,oldID in graphConnections2}
                pos = {**pos1,**pos2}#;print('pos2',pos2)
                for n, p in pos.items():
                        H.nodes[n]['pos'] = p
                plt.figure(1)
                plt.imshow(blank)
                uniqNodes2 = set([oldID for _,oldID in graphConnections2])
                clrs = ['#00FFFF' for _ in mergeCandidates]+ ['#FFB6C1' for _ in uniqNodes2]
                nx.draw(H, pos, with_labels = True, node_color = clrs)#, node_size= 90
                plt.show()
            cnctd_comp = [set(nx.node_connected_component(H, str(key))) for key in mergeCandidates]
            cc_unique = [];[cc_unique.append(x) for x in cnctd_comp if x not in cc_unique]
            cc_unique = [list(A) for A in cc_unique]#;print('Connected Components: ',cnctd_comp,cc_unique)
            cc_newIDs = [[int(elem) for elem in vec if type(elem) == str] for vec in cc_unique]
            cc_oldIDs = [[elem for elem in vec if type(elem) != str] for vec in cc_unique]
            print(f'newFoundBubsRings {newFoundBubsRings}, newFoundBubsElse {newFoundBubsElse}')
            print(f'oldFoundRings {oldFoundRings}, oldFoundElse {oldFoundElse}')
            print(f'cc_newIDs IDs: {cc_newIDs}\ncc_oldIDs IDs: {cc_oldIDs}')
            for old,new in zip(cc_oldIDs,cc_newIDs):
                print(f'old {old}, new {new}')
                # grab new that is known (not stray) EG. is in unresolved new
                new = [elem for elem in new if elem in newFoundBubsRings + newFoundBubsElse]
                print(f'new: {new}')
                if len(new)<len(old) and len(new)>0: # merge
                    typeTemp = typeRing if any(item in newFoundBubsRings for item in new) else typeElse
                    print(f'typeTemp {typeTemp}')
                    # smallest x centroid survives. ID:Cx -> smallest Cx ID
                    centroidsTemp = {ID: centroidsByID2[ID][globalCounter-1][0] for ID in old}
                    selectID = min(centroidsTemp, key=centroidsTemp.get)
                    
                    contourStack = np.vstack([contours4[k] for k in new])
                    x,y,w,h = cv2.boundingRect(contourStack);print([x,y,w,h])
                    baseSubMask,baseSubImage = err.copy()[y:y+h, x:x+w], orig.copy()[y:y+h, x:x+w]
                    
                    subSubMask = np.zeros((h,w),np.uint8)
                    [cv2.drawContours( subSubMask, contours4, ID, 255, -1, offset = (-x,-y)) for ID in new]
                    baseSubMask[subSubMask == 0] = 0
                    
                    baseSubImage[subSubMask == 0] = 0
                    
                    tempC = getCentroidPos(inp = baseSubMask, offset = (x,y), mode=0, mask=[])
                    if typeTemp == typeRing:
                        recoveredMaskRB[selectID]    = baseSubMask 
                        recoveredImgRB[selectID]     = baseSubImage
                        recoverRectParamsRB[selectID]    = [x,y,w,h]
                        recoveredCentroidsRB[selectID]     = tempC
                        recoverOldNewIDsRB[selectID]     = new
                        # centroidsByID2[selectID][globalCounter] = tempC
                    else:
                        # recoveredMaskElse[i]    = tempMask 
                        # recoveredImgElse[i]     = baseSubImage
                        # recoverRectParamsElse[i]    = [xt,yt,wt,ht]
                        # recoverCentroidsElse[i] = getCentroidPos(inp = tempMask, offset = (xt,yt), mode=0, mask=[])
                        # recoverOldNewIDsElse[i]     = overlapingContourIDList; 
                        recoveredMaskElse[selectID] = baseSubMask
                        recoveredImgElse[selectID] = baseSubImage
                        recoverRectParamsElse[selectID] = [x,y,w,h]
                        recoverCentroidsElse[selectID] = tempC
                        recoverOldNewIDsElse[selectID] = new
                        
                        # distanceBubMasks[selectID] = baseSubMask
                        # distanceBubImages[selectID] = baseSubImage
                        # distanceBubRectParams[selectID] = [x,y,w,h]
                        # distanceOldNewIDs[selectID] = new
                        # distanceBubCentroids[selectID] = tempC
                    bubTypesLocal[selectID] = typeTemp   #<<<< inspect here for duplicates
                    bubTypesByID[selectID][globalCounter] = typeTemp
                    
                    for ID in [elem for elem in old if elem != selectID]:
                        bubTypesByID[ID][globalCounter-1] = typePreMerge
                        centroidsByID2[ID][globalCounter] = tempC # <<<<< ALERT! ends dead trajectory at merge center.
                        # <<< this is only visual stuff, might break something, if centroidsByID2 used as ref.
                    
                    print(f'old IDs: {old}, main_old {selectID}, new IDs: {new}, type: {typeTemp}')
                    newFoundBubsRings = [elem for elem in newFoundBubsRings if elem not in new]
                    newFoundBubsElse = [elem for elem in newFoundBubsElse if elem not in new]
                    print(f'newFoundBubsRings, (updated) unrecovered RBs: {newFoundBubsRings}')
                    print(f'newFoundBubsElse, (updated) unrecovered RBs: {newFoundBubsElse}')
                        
                
                        
                    
                    # law = {elem:0 if elem in newFoundBubsRings else 1 for elem in new}
                    # print(f'law {law}')
                    # print(['typeRing','typeElse'][min(law, key=law.get)])
                    
                    
            # jointNeighbors = {min(elem): elem for elem in cc_unique}
            # RBOldNewDist = np.vstack((RBOldNewDist,[4,33]))
            # newFoundBubsRings.remove(33)
            # bubTypesByID[5][globalCounter-1] = typePreMerge
            
        # overlapCntrIDs = {cntrID: objIDs for cntrID,objIDs in mergeCandidates.items() if len(objIDs)>1}
        # for cntrID, objIDs in overlapCntrIDs.items():
        #     isElse = [ID for ID,subIDs in jointNeighbors.items() if cntrID in subIDs]; print('isElse',isElse)
        #     isRing = [ID for ID in newFoundBubsRings if ID == cntrID];print('newFoundBubsRings',newFoundBubsRings)
        #     assert len(isElse) <= 1, f"contour {cntrID} is part of more than 1 EBub \n jointNeighbors: {jointNeighbors}"
        #     if len(isRing) == 1:
        #         newFoundBubsRings.remove(cntrID)
        #     print('newFoundBubsRings',newFoundBubsRings)
        #     print('newFoundBubsElse',newFoundBubsElse)
        #     print('oldFoundElse',oldFoundElse)
        #     print('oldFoundRings',oldFoundRings)
        #     for ID in objIDs:
        #         if ID in oldFoundElse: oldFoundElse.remove(ID);print('oldFoundElse',oldFoundElse)
        #         if ID in oldFoundRings: oldFoundRings.remove(ID);print('oldFoundRings',oldFoundRings)
                
        #    
        #     mergeIDs = [ID for ID in objIDs if centroidsTemp[ID][0] != closestXCentroid]
        #     tempID = [ID for ID in objIDs if centroidsTemp[ID][0] == closestXCentroid][0]
        #     for ID in mergeIDs:
        #         bubTypesByID[ID][globalCounter] = typePreMerge
            
        #     ci = cntrID
        #     if len(isRing) == 1:
        #         RBOldNewDist = np.vstack((RBOldNewDist,[tempID,cntrID]))
        #         # ringBubMasks_old[tempID] = ringBubMasks[ci]
        #         # ringBubImages_old[tempID] = ringBubImages[ci]
        #         # ringBubRectParams_old[tempID] = ringBubRectParam[ci]
        #         # ringOldNewIDs_old[tempID] = ringOldNewIDs[ci]
        #         # bubTypesByID[tempID][globalCounter] = typeRing
        #         # centroidsByID2[tempID][globalCounter] = ringBubCentoids[ci]
        # # mergeCandidatesSubcontourIDs
        
        # ================================= Save first iteration ======================================  
        if globalCounter == 0 :
            startID = 0
            for tempID, newKey in enumerate(ringBubMasks): # by keys
                ringBubMasks_old[tempID]        = ringBubMasks[newKey]
                ringBubImages_old[tempID]       = ringBubImages[newKey]
                ringBubRectParams_old[tempID]   = ringBubRectParam[newKey]
                ringBubCentroids_old[tempID]   = ringBubCentoids[newKey]
                ringOldNewIDs_old[tempID]       = [newKey]
                bubTypesLocal[tempID]           = typeRing
                bubTypesByID[tempID]            = {}
                bubTypesByID[tempID][0]         = typeRing
                # centroidsByID2[tempID]          = {}
                # centroidsByID2[tempID][0]         = ringBubCentoids[newKey]
            # print(f'distanceBubCentroids,{distanceBubCentroids}')
            startID = len(ringBubMasks)+1
            for globalID,localID in zip(range(startID,startID+len(distanceBubCentroids),1),distanceBubCentroids):
                # tempID = len(centroidsByID2)      # time zero- arbitrary ID
                bubTypesLocal[tempID]                 = typeElse
                bubTypesByID[globalID]                = {}
                bubTypesByID[globalID][0]             = typeElse
                # print('centroidsByID2',centroidsByID2)
                # centroidsByID2[tempID]              = {}
                # centroidsByID2[tempID][0]           = cntrd
                
                distanceBubMasks_old[globalID]        = distanceBubMasks[localID]
                distanceBubImages_old[globalID]       = distanceBubImages[localID]
                distanceBubRectParams_old[globalID]   = distanceBubRectParams[localID]
                distanceBubCentroids_old[globalID]    = distanceBubCentroids[localID]
                distanceOldNewIDs_old[globalID]       = distanceOldNewIDs[localID]
                    
            # print('distanceBubCentroids_old',distanceBubCentroids_old)
            
        # ================================= Save other iterations ====================================== 
        if globalCounter >= 1:
            
            distanceBubMasks_old, distanceBubImages_old, distanceBubRectParams_old,  distanceBubCentroids_old, distanceOldNewIDs_old = {}, {}, {}, {}, {}
            # print('distanceBubCentroids',distanceBubCentroids)
            for [old,new] in elseOldNewDist: # contains relation indices that satisfy distance
                bubTypesLocal[old]                  = typeElse if new not in frozenIDs else typeFrozen
                bubTypesByID[old][globalCounter]    = typeElse if new not in frozenIDs else typeFrozen
                # centroidsByID2[old][globalCounter]  = distanceBubCentroids[new]
                distanceBubMasks_old[old]           = distanceBubMasks[new]
                distanceBubImages_old[old]          = distanceBubImages[new]
                distanceBubRectParams_old[old]      = distanceBubRectParams[new]
                distanceBubCentroids_old[old]       = distanceBubCentroids[new]
                distanceOldNewIDs_old[old]          = distanceOldNewIDs[new]
                
            for key in recoveredMaskElse.keys(): # holds global keys
                distanceBubMasks_old[key]       = recoveredMaskElse[key]
                distanceBubImages_old[key]      = recoveredImgElse[key]
                distanceBubRectParams_old[key]  = recoverRectParamsElse[key]
                distanceBubCentroids_old[key]   = recoverCentroidsElse[key] 
                distanceOldNewIDs_old[key]      = recoverOldNewIDsElse[key] 
                bubTypesLocal[key]              = typeRecoveredElse
                
            ringBubMasks_old, ringBubImages_old, ringBubRectParams_old, ringBubCentroids_old, ringOldNewIDs_old = {},{},{},{},{}
            
            for [cPi,cCi] in RBOldNewDist: # contains relation indices that satisfy dinstance
                bubTypesLocal[cPi] = typeRing
                bubTypesByID[cPi][globalCounter] = typeRing
                # centroidsByID2[cPi][globalCounter] = ringBubCentoids[cCi]
                # Add ney keys of IDs to prev T dictionary of cropped masks and images
                ringBubMasks_old[cPi]       = ringBubMasks[cCi]
                ringBubImages_old[cPi]      = ringBubImages[cCi]
                ringBubRectParams_old[cPi]  = ringBubRectParam[cCi]
                ringBubCentroids_old[cPi]  = ringBubCentoids[cCi]
                ringOldNewIDs_old[cPi]      = ringOldNewIDs[cCi]
                
            for key in recoveredMaskRB.keys(): # holds global keys
                ringBubMasks_old[key]       = recoveredMaskRB[key]
                ringBubImages_old[key]      = recoveredImgRB[key]
                ringBubRectParams_old[key]  = recoverRectParamsRB[key]
                ringBubCentroids_old[key]   = recoveredCentroidsRB[key]
                ringOldNewIDs_old[key]      = recoverOldNewIDsRB[key]
                bubTypesLocal[key]          = typeRecoveredRing
                
            if globalCounter == 1222:
                [cv2.imshow(f'c {globalCounter}, i: {i}', img) for i, img in list(ringBubMasks_old.items())]
            # numRingsPrev, numRingsNow = len(prevIterCentroidsRB_RSTRD),len(tempBubbleTypeRings)
            
            
            # ================STORE REMAINING NEW DISCORVERED BUBS=====================
            # ------------ NEW UNRESOLVED RINGS ADDED TO STORAGE WITH NEW ID --------------
            
            # if len(newFoundBubsRings)>0: # if there is new stray bubble, create new storage index
            startID = max(centroidsByID2)+1
            for gID, localID in zip(range(startID,startID+len(newFoundBubsRings),1), newFoundBubsRings):
            # for i,ci in enumerate(newFoundBubsRings): 
                # tempID = max(centroidsByID2)+1 # added last
                ringBubMasks_old[gID] = ringBubMasks[localID]
                ringBubImages_old[gID] = ringBubImages[localID]
                ringBubRectParams_old[gID] = ringBubRectParam[localID]
                ringBubCentroids_old[gID] = ringBubCentoids[localID]
                ringOldNewIDs_old[gID] = ringOldNewIDs[localID]
                bubTypesLocal[gID] = typeRing
                bubTypesByID[gID] = {}
                bubTypesByID[gID][globalCounter] = typeRing
                # centroidsByID2[gID] = {}
                # centroidsByID2[gID][globalCounter] = ringBubCentoids[ci]
                    
            # ------------ NEW UNRESOLVED DIST ADDED TO STORAGE WITH NEW ID --------------
            # if len(newFoundBubsElse)>0: # if there is new stray bubble, create new storage index
            startID += (len(newFoundBubsRings)+1)
            for gID,localID in zip(range(startID,startID+len(newFoundBubsElse),1), newFoundBubsElse):#enumerate(newFoundBubsElse): 
                # gID = max(centroidsByID2)+1 # added last
                bubTypesLocal[gID] = typeElse
                bubTypesByID[gID] = {}
                bubTypesByID[gID][globalCounter] = typeElse if localID not in frozenIDs else typeFrozen
                # centroidsByID2[gID] = {}
                # centroidsByID2[gID][globalCounter] = distanceBubCentroids[localID]
                
                distanceBubMasks_old[gID]        = distanceBubMasks[localID]
                distanceBubImages_old[gID]       = distanceBubImages[localID]
                distanceBubRectParams_old[gID]   = distanceBubRectParams[localID]
                distanceBubCentroids_old[gID]    = distanceBubCentroids[localID]
                distanceOldNewIDs_old[gID]       = distanceOldNewIDs[localID]
        
        bubTypesLocal_old = {}
        bubTypesLocal_old = bubTypesLocal
        # ========================== DUMP all data in global storage ==============================
        # collect all time step info (except bubbleIDsTypes, use new bubTypesByID )
        dumpMasks = {**ringBubMasks_old,**distanceBubMasks_old}
        dumpImages = {**ringBubImages_old,**distanceBubImages_old}
        dumpRecParams = {**ringBubRectParams_old,**distanceBubRectParams_old}#;print('dumpRecParams',dumpRecParams)
        dumpCentroids = {**ringBubCentroids_old,**distanceBubCentroids_old}#;print(f'dumpCentroids,{dumpCentroids}')
        dumpOldNewIDs = {**ringOldNewIDs_old,**distanceOldNewIDs_old}
        for key in list(dumpMasks.keys()):
            if key not in list(masksByID.keys()):
                masksByID[key]                      = {}
                imagesByID[key]                     = {}
                recParamsByID[key]                  = {}
                centroidsByID2[key]                  = {}
                cntrIDsByIDs[key]                   = {}
            masksByID[key][globalCounter]       = dumpMasks[key]
            imagesByID[key][globalCounter]      = dumpImages[key]
            recParamsByID[key][globalCounter]   = dumpRecParams[key]
            centroidsByID2[key][globalCounter]   = dumpCentroids[key]
            cntrIDsByIDs[key][globalCounter]    = dumpOldNewIDs[key]
            # --whereChildrenAreaFiltered is reliant on knowing parent. some mumbo-jumbo to work around
            # -  maybe its better to combine RBOldNewDist and elseOldNewDist and extract global-local IDS there
            parentsWithChildrenIDs = [ID for ID,children in whereChildrenAreaFiltered.items() if len(children) > 0]
            intersection = set(dumpOldNewIDs[key]).intersection(set(parentsWithChildrenIDs))
            # print('intersection',intersection)
            if len(intersection)>0:
                if key not in list(cntrChildrenIDsByIDs.keys()): cntrChildrenIDsByIDs[key] = {}
                cntrChildrenIDsByIDs[key][globalCounter]    = \
                    sum([whereChildrenAreaFiltered[A] for A in intersection],[] ) # pretty general. for ex in case one bubble is two ring bubbles (?!)
        
        # ============================ detect stuck bubbles =======================================    
        # grab all current time IDs, check their traj-s, calculate last nSteps average displacement
        # if its small, collect info. if its that bubble's last live timestep, start investigation by backtracking
           
  
                    
        # if globalCounter == 0:
        fbStoreAreas_old = fbStoreAreas
        fbStoreCentroids_old = fbStoreCentroids
        fbStoreRectParams_old = fbStoreRectParams
        contours_old = contours4
        
        if globalCounter >= 1231231:
            investigateBubIDs = []
            nSteps = 4 
            activeBubsIDs = list(dumpMasks.keys())      # get all current IDs
            # print('activeBubsIDs',activeBubsIDs)
            for activeID in activeBubsIDs:
                totalSteps = len(centroidsByID2[activeID])#;print('activeID',activeID,'totalSteps',totalSteps)
                if totalSteps > 1:
                    nSteps2 = min(totalSteps,nSteps) # for lists smaller than nSteps
                    cntrList = list(centroidsByID2[activeID].values())[-nSteps2:]
                    diffs = np.linalg.norm(np.diff(cntrList,axis=0),axis=1)
                    meanDiff,stdDiff = np.mean(diffs), np.std(diffs)
                    if meanDiff - stdDiff <= 6:
                        # print('activeID',activeID,'cntrList',cntrList,'diffs',diffs,'meanDiff',meanDiff,'stdDiff',stdDiff)
                        if activeID not in frozenBubs: frozenBubs[activeID] = []                        # outer keys as IDs
                        if globalCounter not in frozenBubsTimes: frozenBubsTimes[globalCounter] = []    # outer keys as times
                        frozenBubsTimes[globalCounter].append(activeID)
                        frozenBubs[activeID].append(globalCounter)
            if globalCounter not in frozenBubsTimes and globalCounter-1 in frozenBubsTimes: # time step does not exist. <<< if multiple steps are missing, breaks
                investigateBubIDs = frozenBubsTimes[globalCounter-1]
            if globalCounter in frozenBubsTimes and globalCounter-1 in frozenBubsTimes:
                investigateBubIDs = [elem for elem in frozenBubsTimes[globalCounter-1] if elem not in frozenBubsTimes[globalCounter]]
            print(f'investigateBubIDs, {investigateBubIDs}')
            
            # find earliest time for frozen bubble, take its bounding box and check if it intersects any bubbles form prev frame
            # testBub = 9
            for testBub in investigateBubIDs:
                # print('\n\n\n', f'{testBub} in investigateBubIDs', '\n\n\n')
                earliestTimeInit = min(cntrIDsByIDs[testBub])
                for earliestTime in range(earliestTimeInit,1,-1):
                    # blank = np.zeros(err.shape,np.uint8)
                    # blank = cv2.cvtColor(blank,cv2.COLOR_GRAY2RGB)
                    # for i in cntrIDsByIDs[testBub][earliestTime]:
                    #     cv2.drawContours(blank, contoursAll[earliestTime], i, (255,0,0), -1)
                    x,y,w,h = recParamsByID[testBub][earliestTime]
                    rotatedRectangle = ((x+w/2, y+h/2), (w, h), 0) # first tuple is center by def
                    prevTimeStep = earliestTime - 1
                    prevStepActiveIDs = [ID for ID, timeDict in bubTypesByID.items() if prevTimeStep in timeDict]
                    storeOverlappingIDs = []
                    for ID in prevStepActiveIDs:
                        x2,y2,w2,h2 = recParamsByID[ID][prevTimeStep]
                        rotatedRectangle_old = ((x2+w2/2, y2+h2/2), (w2, h2), 0)
                        interType,interPoints = cv2.rotatedRectangleIntersection(rotatedRectangle, rotatedRectangle_old)
                        
                        if interType > 0:
                            # [cv2.drawContours(blank, contoursAll[prevTimeStep], i, (0,255,255), 1) for i in cntrIDsByIDs[ID][prevTimeStep]]
                            linePts = cv2.convexHull(interPoints, returnPoints=True)
                            # cv2.drawContours(blank,np.int32([linePts]),0,(255,0,255),1)
                            storeOverlappingIDs.append(ID)
                    # cv2.putText(blank, str(prevTimeStep), (25,25), font, 0.9, (255,220,195),2, cv2.LINE_AA)
                    prevStepOverlapCntrIDs = {idx:cntrIDsByIDs[idx][prevTimeStep] for idx in storeOverlappingIDs}
                    # print('\n\n\n', f'prevStepOverlapCntrIDsin {prevStepOverlapCntrIDs}',f'storeOverlappingIDs {storeOverlappingIDs}', '\n\n\n')
                    # box = cv2.boxPoints(rotatedRectangle)
                    # box = np.int0(box)
                    # cv2.drawContours(blank,[box],0,(0,0,255),1)
                    # cv2.imshow(str(prevTimeStep),blank)
                    # store in sol
                    sol = {}
                    # print('lcentr',list(centroidsByID2[testBub].values()))
                    origC = np.mean(list(centroidsByID2[testBub].values()),axis = 0)#;print('origC',origC)
                    # origC =   centroidsByID2[testBub][earliestTime]
                    for globID,cntrLocalIDs in prevStepOverlapCntrIDs.items():
                        for locID in cntrLocalIDs:
                            cp = getCentroidPos(inp = contoursAll[prevTimeStep][locID], offset = (0,0), mode=0, mask=[], value= 0)
                            dist =  np.linalg.norm(np.diff([origC,cp],axis=0),axis=1)[0]#;print('dist',dist)
                            if dist< 20:
                                if globID not in sol: sol[globID] = [] 
                                sol[globID].append(locID)
                    print('sol',sol)
                    if len(sol)>0:
                        # Add new data to frozen bubs
                        recoveredContours  = sum(sol.values(),[])
                        cntrIDsByIDs[testBub][prevTimeStep] = recoveredContours
                        contourStack = np.vstack([contoursAll[prevTimeStep][k] for k in recoveredContours])
                        x,y,w,h = cv2.boundingRect(contourStack)
                        orig0_old = np.uint8(X_data[dataStart+prevTimeStep].copy()[y:y+h, x:x+w])
                        mean_old = np.uint8(mean.copy()[y:y+h, x:x+w])
                        orig_old = cv2.subtract(orig0_old, mean_old)
                        _,err_old = cv2.threshold(orig_old.copy(),thresh0,255,cv2.THRESH_BINARY)
                        err_old = cv2.morphologyEx(err_old.copy(), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
                        baseSubMask, baseSubImage = err_old, orig0_old
                        subSubMask = np.zeros((h,w),np.uint8)
                            
                        if len(recoveredContours) == 1:
                            cp = getCentroidPos(inp = contoursAll[prevTimeStep][recoveredContours[0]], offset = (0,0), mode=0, mask=[], value= 0)
                        else:
                            [[cv2.drawContours( subSubMask, contoursAll[prevTimeStep], locID, 255, -1, offset = (-x,-y)) for locID in locIDs] for globID, locIDs in sol.items()]
                            cp = getCentroidPos(inp = subSubMask, offset = (x,y), mode=0, mask=[])
                        centroidsByID2[testBub][prevTimeStep] = cp
                    
                    
                        baseSubMask[subSubMask == 0] = 0
                        baseSubImage[subSubMask == 0] = 0
                        
                        masksByID[testBub][prevTimeStep] = baseSubMask
                        imagesByID[testBub][prevTimeStep] = baseSubImage
                        recParamsByID[testBub][prevTimeStep] = [x,y,w,h]
                        # bubTypesLocal[testBub]
                        bubTypesByID[testBub][prevTimeStep] = typeElse
                        print('checkpoint')
                        # remove data from old owners
                        for oldID, locIDs in sol.items():
                            locCntsCopy = cntrIDsByIDs[oldID][prevTimeStep].copy()
                            [locCntsCopy.remove(x) for x in locIDs]
                            cntrIDsByIDs[oldID][prevTimeStep] = locCntsCopy
                            # subSubMask =  masksByID[oldID][prevTimeStep].copy()
                            x,y,w,h = recParamsByID[oldID][prevTimeStep]
                            baseSubMask,baseSubImage =  masksByID[oldID][prevTimeStep],  imagesByID[oldID][prevTimeStep]
                            [cv2.drawContours( baseSubMask, contoursAll[prevTimeStep], locID, 0, -1, offset = (-x,-y)) for locID in locIDs]
                            [cv2.drawContours( baseSubImage, contoursAll[prevTimeStep], locID, 0, -1, offset = (-x,-y)) for locID in locIDs]
                            cp = getCentroidPos(inp = baseSubMask, offset = (x,y), mode=0, mask=[])
                            centroidsByID2[oldID][prevTimeStep] = cp
                            masksByID[oldID][prevTimeStep] = baseSubMask
                            imagesByID[oldID][prevTimeStep] = baseSubImage
                    else:
                        bubTypesByID[testBub] = {key:typeFrozen for key in bubTypesByID[testBub]}
                        print('break')
                        break
                    
            # print(f'cntrIDsByIDs: {cntrIDsByIDs}')
                
            # grab all IDs from prev time step except of frozen type. find ones that stopped existing
            prevTimeStepActiveIDs = [ID for ID, timeDict in bubTypesByID.items() if globalCounter-1 in timeDict and bubTypesByID[ID][globalCounter-1] not in [typeFrozen, typePreMerge]]
               
            investigateVanishedBubIDs = [elem for elem in prevTimeStepActiveIDs if elem not in activeBubsIDs ] #and len(bubTypesByID[elem])>1
            print('investigateVanishedBubIDs',investigateVanishedBubIDs)
            for testBub in investigateVanishedBubIDs:
                prevTimeStep = globalCounter - 1
                x,y,w,h = recParamsByID[testBub][prevTimeStep]
                rotatedRectangle_old = ((x+w/2, y+h/2), (w, h), 0) # first tuple is center by def
                storeOverlappingIDs = []
                for ID in activeBubsIDs:
                    x2,y2,w2,h2 = recParamsByID[ID][globalCounter]
                    rotatedRectangle = ((x2+w2/2, y2+h2/2), (w2, h2), 0)
                    interType,interPoints = cv2.rotatedRectangleIntersection(rotatedRectangle, rotatedRectangle_old)
                    if interType > 0:
                        # [cv2.drawContours(blank, contoursAll[prevTimeStep], i, (0,255,255), 1) for i in cntrIDsByIDs[ID][prevTimeStep]]
                        linePts = cv2.convexHull(interPoints, returnPoints=True)
                        # cv2.drawContours(blank,np.int32([linePts]),0,(255,0,255),1)
                        storeOverlappingIDs.append([])
                        storeOverlappingIDs[-1].append(ID)
                        storeOverlappingIDs[-1].append(np.int32(linePts))
                
                if len(storeOverlappingIDs)>1:
                    maxAreaPos = np.argmax(np.array([cv2.contourArea(A) for [_,A] in storeOverlappingIDs]))
                    storeOverlappingIDs = [storeOverlappingIDs[maxAreaPos]]
                if len(storeOverlappingIDs)>0:
                    storeOverlappingIDs = storeOverlappingIDs[0][0]
                    print('testBub',testBub,'storeOverlappingIDs',storeOverlappingIDs)
                    
                    # remove data from old owners
                    donorID, recieverID = testBub, storeOverlappingIDs
                    # for donorID, recieverID in zip([testBub],[storeOverlappingIDs]):
                    recieverTimes = set(bubTypesByID[recieverID])
                    donorTimes = set(bubTypesByID[donorID])
                    commonTimes = recieverTimes.intersection(donorTimes) # <<<< dont know any problems yet
                    for timeStep in commonTimes:
                        cntrIDsByIDs[recieverID][timeStep] += cntrIDsByIDs[donorID][timeStep]
                        combinedContourIDs = cntrIDsByIDs[recieverID][timeStep]
                        x,y,w,h = cv2.boundingRect(np.vstack([contoursAll[timeStep][c] for c in combinedContourIDs]))
                        orig0_old = np.uint8(X_data[dataStart+timeStep].copy()[y:y+h, x:x+w])
                        mean_old = np.uint8(mean.copy()[y:y+h, x:x+w])
                        orig_old = cv2.subtract(orig0_old, mean_old)
                        baseSubMask, baseSubImage = np.zeros((h,w),np.uint8), orig0_old
                        [cv2.drawContours( baseSubMask, contoursAll[timeStep], ID, 255, -1, offset = (-x,-y)) for ID in combinedContourIDs]
                        baseSubImage[baseSubMask == 0] = 0
                        cp = getCentroidPos(inp = baseSubMask, offset = (x,y), mode=0, mask=[])
                        centroidsByID2[recieverID][timeStep]    = cp
                        masksByID[recieverID][timeStep]         = baseSubMask
                        imagesByID[recieverID][timeStep]        = baseSubImage
                        del cntrIDsByIDs[donorID][timeStep], centroidsByID2[donorID][timeStep], masksByID[donorID][timeStep], imagesByID[donorID][timeStep], bubTypesByID[donorID][timeStep]
    
                    if len(cntrIDsByIDs[donorID]) == 0:
                        # print(f'cntrIDsByIDs[donorID],{cntrIDsByIDs[donorID]}')
                        del cntrIDsByIDs[donorID], centroidsByID2[donorID], masksByID[donorID], imagesByID[donorID], bubTypesByID[donorID]
    
    except:
        print('yolo')
        # print('ringOldNewIDs',ringOldNewIDs)
        failImg = convertGray2RGB(err.copy())
        [[cv2.drawContours(  failImg,  contours4, c, (255,0,0), 2) for c in jointNeighbors[cx]] for cx in elseOldNewDist[:,1]]
        poses = [centroidsByID2[ID][globalCounter] for ID in elseOldNewDist[:,0]]
        # [cv2.putText(failImg, string, tuple(pos), 7, 0.5, (0,0,255),s, cv2.LINE_AA) for string, pos in zip(,poses)]
        
        [cv2.drawContours(  failImg,  contours4, c, (0,0,255), 2) for c in RBOldNewDist[:,1]]
        [cv2.drawContours(  failImg,  contours4, c, (0,255,0), 2) for c in newFoundBubsElse]
        [[cv2.drawContours(  failImg,  contours4, c, (125,60,125), 2) for c in recoverOldNewIDsRB[globID]]  for globID in recoveredBubsRelation]
        cv2.imshow('fail',failImg)
    if globalCounter == 3123123:
        import alphashape
        from descartes import PolygonPatch
        thresh  = masksByID[1][globalCounter]
        img  = convertGray2RGB(thresh)
        # global contours
        contours, hierarchy = cv2.findContours(thresh,2,1)
        print(np.array(contours[0]).shape)
        cnt = np.vstack(contours)

        points_2d = np.array(cnt).reshape(-1,2)
   
        alpha_shape = alphashape.alphashape(points_2d, 0.04)
        x,y = alpha_shape.exterior.coords.xy
        hull = [np.array(list(zip(x,y)),np.int32).reshape((-1,1,2))]
        
        cv2.drawContours(  img,  hull, -1, (125,0,180), 1)
        cv2.imshow('asd',img)
       
    
    if globalCounter == 212312312312:
        print('adsadsadsadas -----------------')
        keys = list(ringBubImages_old.keys())
        i,j = keys[1],keys[0]
        compareMoments(big=err,
                                                   shape1=ringBubImages_old[i],shape2=ringBubImages_old[j],
                                                   coords1=ringBubRectParams_old[i],coords2 = ringBubRectParams_old[j])
    
    
    if big != 1: cv2.imshow("aas",aas)

    globalCounter += 1


    # from fil_finder import FilFinder2D
    # import astropy.units as u
    # # fil = FilFinder2D(orig, distance=250 * u.pc, mask=orig)
    # # fil.preprocess_image(flatten_percent=95)
    
    # # ax[0,0].imshow(fil.image.value, origin='lower',cmap='gray')
    # # ax[0,1].imshow(fil.flat_img.value, origin='lower',cmap='gray')
    # # ax[1,0].imshow(err, origin='lower',cmap='gray')
    # # print(len(shareData[0]["masks"]))
    # img = shareData[1]["masks"][1]
    # dlt = 10
    # img = cv2.copyMakeBorder(img, dlt, dlt, dlt, dlt, cv2.BORDER_CONSTANT, None, 0)
    # fil = FilFinder2D(img, distance=50 * u.pc, mask=img)
    # fil.preprocess_image(skip_flatten=True)
    # fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    # # fil.create_mask(verbose=True)
    # fil.medskel(verbose=False)
    # fil.analyze_skeletons(branch_thresh=100* u.pix, skel_thresh=50 * u.pix, prune_criteria='length')#'intensity'
    
    # # fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    # # ax[0,0].imshow(fil.image.value, origin='lower',cmap='gray')  
    # # ax[0,1].imshow(fil.skeleton, origin='lower',cmap='gray')
    # # ax[1,0].imshow(fil.image.value, origin='lower',cmap='gray')
    # # ax[1,0].contour(fil.skeleton_longpath, colors='r')
    # # fil1 = fil.filaments[0]
    # # fil1.skeleton_analysis(fil.image, verbose=False, branch_thresh=5 * u.pix, prune_criteria='length')
    # # fil.exec_rht()
    # # fil1.plot_rht_distrib()
    # # fil1 = fil.filaments[0]
    # # fil.skeleton_analysis(fil.image, verbose=False, branch_thresh=5 * u.pix, prune_criteria='length')
    # # fil.exec_rht()
    # fil.exec_rht(branches=True, min_branch_length=5 * u.pix)
    # plt.plot()
    # _ = plt.hist(fil.orientation_branches[0].value, bins=10)
    # # _ = plt.hist(fil.orientation_branches[0].value, bins=10)
    # plt.xlabel("Orientation (rad)")


# import fil_finder  as fil

    

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
    
    activeIDs = [ID for ID, timeDict in bubTypesByID.items() if globalCounter in timeDict]#;print(f'globalCounter: {globalCounter} activeIDs: {activeIDs}')
    blank = X_data[dataStart+globalCounter] * 1
    blank = cv2.subtract(np.uint8(blank), np.uint8(mean))
    blank = convertGray2RGB(blank)
    typeStrings = ["N","RB", "rRB", "E", "F","rE","pm"]
    splitStrings =  {}
    for ID in activeIDs:
        temp = [0,len(cntrIDsByIDs[ID][globalCounter])]
        currentType     = bubTypesByID[ID][globalCounter]#;print('currentType',currentType)
        currentTypeStr  = typeStrings[currentType] 
        currentCntrIds  = cntrIDsByIDs[ID][globalCounter]
        # text            = currentTypeStr + '('+','.join(list(map(str,currentCntrIds)))+')'
        temp.append(currentTypeStr + '(')
        cCIstr = list(map(str,currentCntrIds))
        A = [x for y in zip(cCIstr, [","]*len(cCIstr)) for x in y][:-1] #dont ask me
        temp += A
        # if len(cntrIDsByIDs[ID]) == 1:
        if globalCounter-1 not in cntrIDsByIDs[ID]:
            # text += " new"
            temp += [f"):{ID} new"]
        else:
            temp[0] = 1
            prevType = bubTypesByID[ID][globalCounter-1]
            prevTypeStr = typeStrings[prevType] 
            prevCntrIds  = cntrIDsByIDs[ID][globalCounter-1]
            # text += "-" + prevTypeStr + f'({ID})'
            temp += [")"+str(":" + prevTypeStr + f'({ID})')]
        splitStrings[ID] = temp
        
        if ID in cntrChildrenIDsByIDs:
            if globalCounter in cntrChildrenIDsByIDs[ID]:
                [cv2.drawContours(  blank,   contoursAll[globalCounter], cid, (0,255,0), 1) for cid in cntrChildrenIDsByIDs[ID][globalCounter]]
        # aa = [cid for cid in cntrChildrenIDsByIDs[ID][globalCounter] if globalCounter in cntrChildrenIDsByIDs[ID]]
        colorSet = [(),(0,0,255),(0,100,190),(255,0,0),(125,0,125),(255,0,125),(0,255,0)]
        thcSet = [(), 2, 1, 2, 2,2,2]
        clr = colorSet[currentType]
        thc = thcSet[currentType]
        [cv2.drawContours(  blank,   contoursAll[globalCounter], cid, clr, thc) for cid in currentCntrIds]
        if len(cntrIDsByIDs[ID][globalCounter])> 1:
            cnt = np.vstack([contoursAll[globalCounter][A] for A in cntrIDsByIDs[ID][globalCounter]])
    
            points_2d = np.array(cnt).reshape(-1,2)
            # print(len(points_2d))
            # initAlpha = 0.03
            for alpha in np.arange(0.03, 0,-0.01):
                
                alpha_shape = alphashape.alphashape(points_2d, alpha)
                if alpha_shape.geom_type == 'Polygon':
                    x,y = alpha_shape.exterior.coords.xy
                    hull = np.array(list(zip(x,y)),np.int32).reshape((-1,1,2))
                    cv2.drawContours(  blank,   [hull], -1, (125,0,180), 1)
                    break
            #     for elem in alpha_shape.geoms:
            # # mycoordslist = [list(x.exterior.coords) for x in geom.geoms]
            #         x,y = elem.exterior.coords.xy
            #         hull = np.array(list(zip(x,y)),np.int32).reshape((-1,1,2))
            #         cv2.drawContours(  blank,   [hull], -1, (125,0,180), 1)
            #     cv2.imshow('asd',blank)
        
    fontScale = 0.7;thickness = 4;
    for k,ID in enumerate(activeIDs):
        moveX = 0
        case, numIDs, *strings = splitStrings[ID]
        strDims  = [cv2.getTextSize(text, font, fontScale, thickness) for text in strings]
        baseline = max([bl for _,bl in strDims])
        ySize = max([sz[1] for sz,_ in strDims])
        xSize = [sz[0] for sz,_ in strDims]
        totSize = sum(xSize)
        # distToIDs = [sum(xSize[:(2*A)] ) for A in range(numIDs)]#;print('distToIDs',distToIDs)
        ci = 0
        for i, string in enumerate(strings):
            origPos = np.array(centroidsByID2[ID][globalCounter])
            centroidX, halfTotSize = origPos[0],int(totSize/2)
            leftSideOffset = max(0, 0 - (centroidX-halfTotSize)) # in case text is out of frame add offset.
            rightSideOffset = min(0,err.shape[1] - (centroidX+halfTotSize)) 
            # ladder = (k%4)*(ySize+baseline + 5) # vertical offset in steps
            ladder = int(np.ceil(0.5*(k+1)))*(ySize+baseline + 5) # ceil to get 1,1,2,2,3,3,...
            if k%2 == 0:
                topBottom = err.shape[0]-baseline - 30;flip = -1
            else:
                topBottom = 30 + ySize; flip  = 1
            pos = np.array(
                (centroidX -1*halfTotSize + moveX + leftSideOffset + rightSideOffset
                 ,
                 topBottom + flip* ladder)
                        )
            if i in range(1,len(strings),2):
                cntrID = cntrIDsByIDs[ID][globalCounter][ci]
                topID = map(int,(pos+ (strDims[i][0][0]/2,-strDims[i][0][1]/2)))
                _, _, p1, p2 = closes_point_contours([[list(topID)]],contoursAll[globalCounter][cntrID])
                cv2.polylines(blank, [np.array([p1,p2])],0, (0,0,255), 1)
                ci += 1
            
            [cv2.putText(blank, string, tuple(pos), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
            moveX += xSize[i]
    
    #--------------- trajectories ---------------------------------
    for i,times in centroidsByID2.items(): # key IDs and IDS-> times
        useTimes = [t for t in times.keys() if t <= globalCounter and t>globalCounter - 12]
        pts = np.array([times[t] for t in useTimes]).reshape(-1, 1, 2)
        if pts.shape[0]>1:
            cv2.polylines(blank, [pts] ,0, cyclicColor(i), 1)
        
        [cv2.circle(blank, tuple(p), 3, cyclicColor(i), -1) for [p] in pts]
            
    cv2.putText(blank, str(globalCounter), (25,25), font, 0.9, (255,220,195),2, cv2.LINE_AA)
    cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\"+str(dataStart+globalCounter).zfill(4)+".png" ,blank)
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
#                     cCntr = contours4[cSel]
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
# for ci,cntr in enumerate(contours4):
#     xc,yc,wc,hc = cv2.boundingRect(cntr)
#     if xc>x and yc>y and xc+wc<x+w and y+hc<y+h and cv2.contourArea(cntr)> 0.2*area0:

#         cv2.rectangle(aas, (xc,yc), (xc+wc,yc+hc),(178,60,200),1)
#         cv2.rectangle(aas, (x,y), (x+w,y+h),(60,125,200),1)
#         print('asdasdasdasdas', i)
#         aas = drawContoursGeneral(aas, contours4, ci, (0,127,127), 2)
#         cpr = getCentroidPos(inp = cntr, offset = (0,0), mode=0, mask=[])
#         # cv2.putText(aas, f' missing bubble id {ci}', tuple(np.array(cpr)+(10,-30)), font, 0.7, (60,125,200),1, cv2.LINE_AA)
#         centroidsByID[i][-1] = cpr
#         if i in oldFoundRings: oldFoundRings.remove(i)
# cv2.imshow(str(i),err[y:y+h, x:x+w])
# print(y,x)
# [cv2.drawContours( blank,   contours4, k, (0,255,0), 1) for k in temp]  
# cv2.imshow("asdasdas",blank)



# %timeit -r 4 -n 1000  fuu(d,1)