# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:43:30 2022

@author: User
"""
import cv2
from cv2.ximgproc import anisotropicDiffusion
# from cv2.ximgproc import getDisparityVis

import numpy as np
import os, multiprocessing
import glob
from matplotlib import pyplot as plt
from skimage.filters import (threshold_li, threshold_local, threshold_otsu, rank,try_all_threshold)
# https://stackoverflow.com/questions/33091755/bradley-roth-adaptive-thresholding-algorithm-how-do-i-get-better-performance/37459940
from scipy import ndimage
from PIL import Image
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import random as rng
mapXY = (np.load('./mapx.npy'), np.load('./mapy.npy'))
def resizeImage(img,frac):
    width = int(img.shape[1] * frac)
    height = int(img.shape[0] * frac)
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

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
def createOutputFolders():
    if saveOutput == 1:
        # print(imageMainFolder[:-1])
        outputpath = imageMainFolder[:-1]+postFix
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
            print("output Folder created")
        for folder in folderNames:
            subFolderPath = os.path.join(outputpath,folder)
            if not os.path.exists(subFolderPath):
                os.makedirs(subFolderPath)
                
# if doRun == 1 and saveOutput == 1:
#     createOutputFolders()

def getSaveLink(mainString,searchStringList,postFix):
    for searchString in searchStringList:
        if (mainString.find(searchString) != -1):
            path = mainString.replace(searchString,searchString+postFix)
            path = os.path.splitext(path)[0] + extension
    return path


def saveFile(path,file):
    if doRun == 1 and saveOutput == 1:
       cv2.imwrite(path,file)
       
# def generateMaks():
#     if generateMasks == 1:
#         for folder in folderNames:
#             path = os.path.join(imageMainFolder,folder)
#             fileName = os.listdir(path)[0]
#             imgLink = os.path.join(path,fileName)
#             img = cv2.imread(imgLink)
#             img = adjustBrightness(img)
#             img = undistort(img)
#             maskPath = os.path.join('.','masks',folder+'.bmp')
#             cv2.imwrite(maskPath,img)

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
# generateMaks()

def setupBlur(image, blurKernalMax, numVariatons):
    x       = blurKernalMax[0]
    y       = blurKernalMax[1]
    blurSet = []

    shape   =  list(image.shape)       
    blur    = [
                np.linspace(start = 1, stop = int(x), num = int(numVariatons)),
                np.linspace(start = 1, stop = int(y), num = int(numVariatons))
            ]
    blur = np.array(blur).T.astype(int)
    
    for blurVariant in blur:
        blurVariant = cv2.blur(image.copy(),tuple(blurVariant),cv2.BORDER_REFLECT)
        blurSet     = np.append( blurSet, blurVariant )
    shape.insert(0,-1)
    blurSet = np.reshape(blurSet, shape)

    blurTitles      = []
    for a in blur: blurTitles = np.append(blurTitles,'kernel value: {}'.format(a))
    drawRectangles('gray', (13,9) ,blurSet, blurTitles)
    
def debugBlur(image):
    if determineBlurKernel == 1:
        img = setupBlur(image = image, blurKernalMax = useBlurKernalMax, numVariatons = 4)

def modifiedOtsu(image, alpha): #get threshold by otsu, modify it with alpha.
    ret,th          = cv2.threshold(image.copy(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    desired_th      = ret*alpha
    _ , th_new      = cv2.threshold(image.copy(), desired_th, 255, cv2.THRESH_BINARY)
    return ret , th_new

def setupAlpha(image, alphaMin, alphaMax, numDifferentAlphas):
    tempImage       = image.copy()
    
    alphaImageRGB      = []
    shape           =  list(tempImage.shape)       
    alphas          = np.linspace(start = alphaMin, stop = alphaMax, num = numDifferentAlphas)
    for alphaVariant in alphas:
        _, thresholdVariant = modifiedOtsu(image=tempImage, alpha=alphaVariant)

        ovrl = overlay(image = image, mask = cv2.bitwise_not(thresholdVariant), color = maskColor, alpha = overlayOpacity)
        # print(ovrl)
        alphaImageRGB  = np.append( alphaImageRGB, ovrl)
    shape.insert(0,-1)
    shape.insert(len(shape),3)

    alphaImageRGB = np.reshape(alphaImageRGB,shape )
    
    alphasTitles = []
    for a in alphas: alphasTitles = np.append(alphasTitles,'Parameter value: {0:.2f}'.format(a))
    drawRectangles('viridis', (13,9), alphaImageRGB.astype(np.uint8), alphasTitles)
    
def debugBinarization(image):
    if determineAlpha == 1:
        setupAlpha(image                = image,
                   alphaMin             = useAlphaMin,
                   alphaMax             = useAlphaMax,
                   numDifferentAlphas   = useNumDifferentAlphas)

# def faster_bradley_threshold(image, threshold=bradleyThresold, window_r=bradleyWindow):
#     percentage = threshold / 100.
#     window_diam = 2*window_r + 1
#     # convert image to numpy array of grayscale values
#     img = image.astype(np.float)
#     # img = np.array(image.convert('L')).astype(np.float) # float for mean precision 
#     # matrix of local means with scipy
#     means = ndimage.uniform_filter(img, window_diam)
#     # result: 0 for entry less than percentage*mean, 255 otherwise 
#     height, width = img.shape[:2]
#     result = np.zeros((height,width), np.uint8)   # initially all 0
#     result[img >= percentage * means] = 255       # numpy magic :)
#     # convert back to PIL image
#     tmp = Image.fromarray(result)
#     return tmp.convert('L')
# ----------------------------------------------------------------------------

def binarization(image,mode):
    if mode == 1: # Otsu
        _ , alpha = modifiedOtsu(image, useAlpha)
    elif mode == 2: # useBradley
        alpha = faster_bradley_threshold(image)
    elif mode == 3: # useLi
        thresh = threshold_li(image)
        alpha = (image > thresh)*255
    elif 1 == 0:
        block_size = 41
        local_thresh = threshold_local(image, block_size, offset=31)
        alpha = (image > local_thresh)*255
    elif 1 ==0:
        radius = 15
        footprint = disk(radius)
        
        local_otsu = rank.otsu(image, footprint)
        alpha = (image > local_otsu)*255
    else:
        alpha = image
    return np.array(alpha,dtype=np.uint8)

if 1 == 2:
    if useOtsu == 1:
        print("use Otsu")
    elif useBradley == 1:
        print("use Bradley")
    elif useLi == 1:
        print("use li")        
        
def closingOpening(image, kernelWhite,kernelBlack, order, debugImages):
    img = image.copy()
    kernelWhite     = np.ones(kernelWhite,np.uint8)
    kernelBlack     = np.ones(kernelBlack,np.uint8)
    
    if order == 0:
        closeWhite      = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelWhite)
        closeBlack      = cv2.morphologyEx(closeWhite, cv2.MORPH_CLOSE, kernelBlack)
        out             = closeBlack
    else:
        closeBlack      = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelBlack)
        closeWhite      = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelWhite)
        out             = closeWhite

    if debugImages == 1: 
        cv2.imshow("closingOpening: original", image)
        cv2.imshow("closingOpening: closeWhite", closeWhite)
        cv2.imshow("closingOpening: closeBlack", closeBlack)
        return         image 
    else:
        return          out 
    
def undistort(image):
    return cv2.remap(image,mapXY[0],mapXY[1],cv2.INTER_LINEAR)
    # cv2.imshow("distorted image" , img)
    # cv2.imshow("undistorted image" , dst)
# ---------------------------------------------   

def maskedBlend(image1,image2,mask,multiplier):
    # you can pass rgb color
    if len(image1)>3: A = image1.copy()
    else: A = np.full((mask.shape[0],mask.shape[1],3),image1,dtype=np.uint8)
    if len(image2)>3: B = image2.copy()
    else: B = np.full((mask.shape[0],mask.shape[1],3),image2,dtype=np.uint8)    
    
    alpha = mask.copy().astype(float)
    alpha = cv2.normalize(alpha.copy(), None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
    # multiplier had problems with C. order reveresed..
    alpha = alpha * multiplier

    beta = np.ones(alpha.shape)-alpha
    C = np.zeros(image1.shape)
    for i in range(3):
        C[:,:,i] =  np.multiply(A[:,:,i],beta) + np.multiply(alpha,B[:,:,i])
    return np.uint8(C)

# / examples for ----- maskedBlend------------     
# alpha = np.zeros((100,100),dtype=np.uint8)

# alpha = cv2.circle(alpha, (50,50), 20, 255, -1)
# alpha = cv2.blur(alpha,(20,20))

# A = np.zeros((100,100,3),dtype = np.uint8)
# B = np.zeros((100,100,3),dtype = np.uint8)
# B[:,:] = (0,0,255)

# C = maskedBlend(image1 = A,image2 = B,mask = alpha)

# cv2.imshow('A',A)
# cv2.imshow('B',B)
# cv2.imshow('C',C)
#  / examples for ----- maskedBlend------------ 

# expand segment selection by some offset, takes into account border zone.
# consider np.clip(x, 0, 255)
def getPaddedSegmentCoords(img, x, y, w, h, offset):
    (imgH,imgW) = img.shape
    # define global coordinates and crop them with minmax(img.dims)
    (x00,y00,x11,y11) = (max(x-offset,0),max(y-offset,0),min(x+w+offset,imgW),min(y+h+offset,imgH))
    x2,y2,w2,h2 = x00,y00,x11-x00,y11-y00
    return x2, y2, w2, h2



def convertGray2RGB(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image.copy(),cv2.COLOR_GRAY2RGB)
    else:
        return image
    
def convertRGB2Gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image.copy(),cv2.COLOR_RGB2GRAY)
    else:
        return image

def overlay(image, mask, color, alpha):
    base = convertGray2RGB(image.copy())
    overlay = base.copy()
    overlay[mask == 255] = color
    cv2.addWeighted(overlay, alpha, base, 1 - alpha,0, base)
    return base

    
def maskDilateErode(img,mask,kernel = 3,iters = 2, order = 1, graphics = 0,name = 'default '): #external contours
    maskBase        = mask.copy()
    ker = np.ones((kernel,kernel),np.uint8)
    border          = kernel * iters
    maskBasePad     = cv2.copyMakeBorder(maskBase, border, border, border, border, cv2.BORDER_CONSTANT, None, 0)
    if order == 1:
        maskDilate  = cv2.dilate(maskBasePad, ker, iterations = iters)
        maskErode   = cv2.erode(maskDilate.copy(), ker, iterations = iters)
    else:
        maskErode   = cv2.erode(maskBasePad, ker, iterations = iters)
        maskDilate  = cv2.dilate(maskErode.copy(), ker, iterations = iters)

        
    maskDilate      = maskDilate[border:-border,border:-border]
    maskErode       = maskErode[border:-border,border:-border]
    if order == 1:  cMask, _   = cv2.findContours(maskErode,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    else:           cMask, _   = cv2.findContours(maskDilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # if doHull == 1:
    #     cont = np.vstack(cMaskErode)
    #     hull = cv2.convexHull(cont)
    #     uni_hull = []
    #     uni_hull.append(hull) # <- array as first element of list
    #     outputMask  = cv2.drawContours(  outputMask.copy(),   uni_hull, -1, 255, -1)
    # else:
    #     outputMask = cv2.drawContours(  outputMask.copy(),   cMaskErode, -1, 255, -1)
    outputMask = cv2.drawContours(  mask.copy() * 0,   cMask, -1, 255, -1)    
    if graphics == 1:
        clrMsk = (0,0,255)
        clrMrph = (255,128,128)
        # clrErd = clrDlt
        clrEnd = (0,255,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        base        = convertGray2RGB(img.copy())
        if order == 1:  
            stage1      = overlay(image = base, mask = maskDilate, color = clrMrph , alpha=0.5)
            stage2      = overlay(image = stage1, mask = mask, color = clrMsk, alpha=0.9)
        else:
            stage1      = overlay(image = base, mask = mask, color = clrMsk, alpha=0.9)
            stage2      = overlay(image = stage1, mask = maskErode, color = clrMrph, alpha=1)
        # if doHull == 1:
        #     stage3  = cv2.drawContours(  stage2.copy(),   uni_hull, -1, (0,255,0), 2)
            # cv2.imshow("sdfsd",cv2.drawContours(  stage2.copy(),   hull, -1, (0,255,0), 2))
        
        # else:
        stage3      = cv2.drawContours(  stage2.copy(),   cMask, -1, clrEnd, 1)
        stage3      = cv2.putText(stage3, 'orig mask', (10,30), font, 0.5, clrMsk, 1, cv2.LINE_AA)
        if order == 1:  clr1,clr2 = clrMrph,clrEnd
        else:           clr1,clr2 = clrEnd,clrMrph
        stage3      = cv2.putText(stage3, f'dilate {kernel} px * {iters}', (10,60), font, 0.5, clr1, 1, cv2.LINE_AA)
        stage3      = cv2.putText(stage3, f'erode {kernel} px * {iters}', (10,90), font, 0.5, clr2, 1, cv2.LINE_AA)
        
        if order == 1:  cv2.imshow('maskDilateErode (D->E): '+str(name),stage3)
        else:           cv2.imshow('maskDilateErode (E->D): '+str(name),stage3)
    return outputMask

def maskSegmentAndHull(img, mask, minArea, graphics, name):
    # get object contours => filter them by min area
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    cSel = [i for i in range(len(contours)) if cv2.contourArea(contours[i]) > minArea]
    contours = [contours[i] for i in cSel]
        
    blank = mask.copy() * 0 
    blankLinesOrig = mask.copy() * 0 
    for c in contours:
        blankLinesOrig = cv2.drawContours(  blankLinesOrig.copy(),   [c], -1, 255, 1)
        blank = cv2.drawContours(  blank.copy(),   [cv2.convexHull(c)], -1, 255, -1)
        
    if graphics == 1:
            gfx = overlay(image = img, mask = blank, color = (0,0,255), alpha=0.5)
            gfx = overlay(image = gfx, mask = blankLinesOrig, color = (0,255,255), alpha=1.0)
            cv2.putText(gfx, 'orig mask outline', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(gfx, 'orig mask hull', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('maskSegmentAndHull: '+str(name),gfx)
    return blank

 # how to know if segment contains see-though bubble or partial bubble?
        # possible sol-n is to check area/perimeter ratio. but alternatively
        # you can see how erosion/closing affects total area. bigger perimeter 
        # "consumes" area on erosion. high area loss is a clear indicator.
        # if bubble is partial, connect reflections using dilate
        # pad image with black background of thickness diagonal/2,
        # furthest possible case. dilate reflections iteratively until
        # only one controur is left. then erode shape back.
         # if bubbleTypeAVG <= 0.1:
         #    dilate = 3
         #    imgas = list()
         #    dlt = int(np.sqrt(w**2+h**2)/2 + 1)
         #    # dlt = dilate * dilIter + 1
         #    dilIter = int(dlt/dilate)
         #    dilCounter = 1
         #    bs = cv2.copyMakeBorder(th.copy(), dlt, dlt, dlt, dlt, cv2.BORDER_CONSTANT, None, 0)
         #    imgas.append(bs.copy())
         #    for i in range(dilIter):
               
         #        bs = cv2.dilate(bs.copy(),np.ones((dilate,dilate),np.uint8),iterations = 1)
         #        # print(dilCounter)
                
         #        cntrDilErd, _ = cv2.findContours(bs,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                
         #        imgas.append(bs)
         #        # print(len(cntrDilErd))
         #        if len(cntrDilErd) == 1:
         #            bs = cv2.erode(bs.copy(),np.ones((dilate,dilate),np.uint8),iterations = dilCounter)
         #            imgas.append(cv2.addWeighted(imgas[0], 0.5, bs, 1 - 0.5,0))
         #            # print(cv2.isContourConvex(cntrDilErd[0]))
         #            break
         #        dilCounter += 1
         #    imgDilErdSet = [img[dlt:-dlt,dlt:-dlt] for img in imgas]
            # cv2.imshow('part dil:'+str(m),resizeToMaxHW(cv2.hconcat(imgas)))
            # cv2.imshow('part dil:'+str(m),resizeToMaxHW(cv2.hconcat(imgDilErdSet)))
            
            
# TEST thresholds
# locBinsAll = [cv2.ximgproc.BINARIZATION_NIBLACK,cv2.ximgproc.BINARIZATION_SAUVOLA,cv2.ximgproc.BINARIZATION_WOLF,cv2.ximgproc.BINARIZATION_NICK]
# globBin = cv2.THRESH_BINARY | cv2.THRESH_OTSU
# huuu = [cv2.ximgproc.niBlackThreshold(thrInput,255,globBin,31, 0.1, _,a) for a in locBinsAll]
# huuu.insert(0, thrInput)
# imgSet = [resizeImage(a,2) for a in huuu]
# # yee = cv2.hconcat(imgSet)
# # cv2.imshow('threshold types:'+str(m),cv2.hconcat(imgSet))
# k = 0
# for i in 2*np.arange(3,16,3)+1:
#     huuu = [cv2.ximgproc.niBlackThreshold(thrInput,255,globBin,i, 0.1, _,a) for a in locBinsAll]
#     huuu.insert(0, thrInput)
#     imgSet  = [resizeImage(a,2) for a in huuu]
#     bebe    = cv2.hconcat(imgSet)
#     bebe    = cv2.putText(bebe, str(i), (20,30), font, 0.5, 127, 1, cv2.LINE_AA)
#     if k==0:
#         yee = bebe
#     else:
#         yee =  cv2.vconcat([yee,bebe])
#     k += 1
# 17/03/23 removed cause it took up much space and wasnt used at all
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
    
