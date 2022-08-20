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