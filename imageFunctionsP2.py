# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:17:57 2022

@author: User
"""
import glob, pickle, numpy as np, cv2, os, itertools, networkx as nx
from scipy import interpolate
from matplotlib import pyplot as plt
import alphashape

def opossiteAngles(p1,p2p3):
    # given a point p1 return oposite to p1 angles
    # point oredr anti-clockwise, angles too
    p1,[p2,p3] = np.array(p1), np.array(p2p3)
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
 
    angle1 = np.rad2deg(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
    angle2 = np.rad2deg(np.arccos((a**2 + b**2 - c**2) / (2 * b * a)))
    
    return angle1,angle2
def init(folder,imageNumber): # initialize some globals, so i dont have to pass them. EDIT: IDK wth does it do, looks like nothing
    global imageMainFolder,imgNum
    imageMainFolder = folder
    imgNum = imageNumber
    # print(imageMainFolder,imgNum)
    #   
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

def initImport(mode, workBigArray,recalcMean,readSingleFromArray,pickleNewDataLoad,pickleNewDataSave,pickleSingleCaseSave):
    if mode == 0: # start new array with new pictures
        workBigArray = 1
        recalcMean = 1
        readSingleFromArray = 0
        pickleNewDataLoad = 1
        pickleNewDataSave = 1
        pickleSingleCaseSave = 0
    if mode == 1: # read one from array
        workBigArray = 1
        recalcMean = 0
        readSingleFromArray = 1
        pickleNewDataLoad = 0
        pickleNewDataSave = 0
        pickleSingleCaseSave = 0
    if mode == 2: # read one from saved pickles
        workBigArray = 0
        recalcMean = 0
        readSingleFromArray = 0
        pickleNewDataLoad = 0
        pickleNewDataSave = 0
        pickleSingleCaseSave = 0
          
    imageLinks = glob.glob(imageMainFolder + "**/*.bmp", recursive=True)    
    imageLinks = [a.replace("\\","/") for a in imageLinks]
    # folderNames = os.listdir(imageMainFolder)
    # countImages = ["".join(imageLinks).count(x) for x in folderNames]
    global X_data
    global mean
    if workBigArray == 1:
        print("get data array, begin...")
        X_data = []
        if pickleNewDataLoad == True:       # read imgs in temp array
            print("readding image files")   
            for myFile in imageLinks:
                # print(myFile)
                img     = cv2.imread (myFile,0)
                bright  = adjustBrightness(img,adjustBrightness)
                dist    = undistort(bright)
                masks   = glob.glob('./masks'+'/*.bmp')
                cropped = cropImage(image = dist,importMaskLink=  masks[0], cropUsingMask = cropUsingMask)
                X_data.append(cropped)
                
            if pickleNewDataSave == True:   # store temp array on drive
                print("saving image files on drive")
                with open('X_data.pickle', 'wb') as handle:
                    pickle.dump(X_data, handle)       
        else:                               # read stored array into python
            with open('X_data.pickle', 'rb') as handle:
                X_data = pickle.load(handle)
                
        print("get data array, competed:")
        print('X_data shape:', np.array(X_data).shape)
        
        if pickleSingleCaseSave     == 1:
            with open('./picklesImages/img'+str(imgNum).zfill(4)+'.pickle', 'wb') as handle:
                pickle.dump(X_data[imgNum], handle) 
        
        meanFileExists = os.path.exists('./mean.pickle')        
        if recalcMean == 1 or not meanFileExists:
            if meanFileExists: print('mean.pickle not found, recalculating')
            print("mean image, begin...")
            mean = np.mean(X_data, axis=0)
            cv2.blur(mean, (3,3),cv2.BORDER_REFLECT)
            print("mean image, calculated")
    
            with open('./mean.pickle', 'wb') as handle:
                print('saving mean.pickle')
                pickle.dump(mean, handle) 
                    
            
        else:
            with open('./mean.pickle', 'rb') as handle:
                print('loading mean.pickle')
                mean = pickle.load(handle)
        
    if workBigArray == 0:
        with open('./picklesImages/img'+str(imgNum).zfill(4)+'.pickle', 'rb') as handle:
                X_data = [pickle.load(handle)]
        with open('./mean.pickle', 'rb') as handle:
                mean = pickle.load(handle)
    return X_data, mean,imageLinks

def bubbleTypeCheck(image, index = 0,erode = 5,close = 3,areaRemainingThreshold = 0.75, graphics = 0):
    kernerlErode    = np.ones((erode,erode),np.uint8)
    kernerlClose    = np.ones((close,close),np.uint8)
    borderWidth     = int(np.sqrt((image.shape[0])**2+(image.shape[1])**2)/2 + 1)
    # maxErosionIter  = int(borderWidth/erode)
    maxErosionIter  = max([int(image.shape[a]/erode * 0.2) for a in range(2)])
    # print(maxErosionIter)
    temp = image.copy()
    # area0 = np.sum((temp/255).flatten())
    
    bubbleType = list()

    if graphics == 1:
        graphicsList = list()
        graphicsList.append(temp)
    

    arRemList = list()
    arRemList.append(1)
    for i in range(maxErosionIter):
       area0 = np.sum((temp/255).flatten())
       temp = cv2.erode(temp.copy(),kernerlErode,iterations = 1)
       temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernerlClose)
       area = np.sum((temp/255).flatten())
       if area == 0:
           print01('break', graphics)
           break
       areaRemaining = 1.0-(area0-area)/area0
       if areaRemaining < areaRemainingThreshold:
           # "({}) areaRemaining: {}, most likely partial bubble".format(str(index).zfill(3),areaRemaining)
           msg = "({}) areaRemaining: {:.2f}, most likely partial bubble".format(str(index).zfill(3),areaRemaining)
           print01(msg, graphics)
           bubbleType.append(0)
       else:
           msg = "({}) areaRemaining: {:.4f}, most likely full bubble".format(str(index).zfill(3),areaRemaining)
           print01(msg, graphics)
           # print01("areaRemaining: "+str(areaRemaining)+", most likely full bubble", graphics)
           bubbleType.append(1)
          
       if graphics == 1:
           txt = " {:.2f} ".format(areaRemaining)
           temp = cv2.putText(temp.copy(), txt, (4,14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 180, 1, cv2.LINE_AA)
           graphicsList.append(temp)
           
       arRemList.append(areaRemaining)
    # print(arRemList)
    y = arRemList[1:]#;print(y)
    x = np.arange(1,len(y)+1,1)#;print(x)
    if len(y)>1:
        coefs = np.polyfit(x, y, 1) #list(range(1,len(y)+1,1))
    else: coefs = (None,None)

    if graphics == 1:
        print("\n")
        cv2.imshow('bubbleTypeCheck:'+str(index),resizeToMaxHW(cv2.hconcat(graphicsList),1200,200))
    return np.round_(np.mean(bubbleType), decimals = 1) , arRemList, coefs

def convertGray2RGB(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image.copy(),cv2.COLOR_GRAY2RGB)
    else:
        return image
    
def drawContoursGeneral(img,contours,selection, color, thickness):
    imgRGB = convertGray2RGB(img)
    img = cv2.drawContours(imgRGB, contours, selection, color, thickness)
    return img

def boundingRect2Contour(contourList,xOffset=0,yOffset=0,xAdd=0,yAdd=0):
    params = list()
    contour = list()
    rectParamsList = [cv2.boundingRect(contour) for contour in contourList]
    for (x,y,w,h) in rectParamsList:
        params.append(np.array(
                                 [
                                 [x+xOffset-xAdd,     y+yOffset-yAdd],
                                 [x+xOffset-xAdd,     y+yOffset+h+yAdd],
                                 [x+xOffset+w+xAdd,   y+yOffset+h+yAdd],
                                 [x+xOffset+w+xAdd,   y+yOffset-yAdd]
                                 ]).reshape(-1,1,2))    
        contour.append(np.array([x,y,w,h]))
    return params, contour

def boundRectAngle(rect1,rect2, maxAngle, debug  = 0, info= ''):
    _,_,w,h = rect1
    _,_,w2,h2 = rect2
    angleDiff = np.abs(np.arctan(h/w) - np.arctan(h2/w2))*180/np.pi
    if debug == 1: print(f'{info} rect1: {rect1}, rect2: {rect2}, angleDiff: {angleDiff}')
    return [True,angleDiff] if angleDiff < maxAngle else [False,angleDiff]

def cntParentChildHierarchy(image, mode, minArea, minAreaChild, CPareaRatio):
    childrenCNTRS, whereChildrenAreaFiltered = [],np.array([],dtype=object)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE
    whereParents0 = np.argwhere(hierarchy[:,:,3]==-1)[:,1] #flag -1> cnt /w no parents (no owner)
    
    parentSizeSel = [i for i in whereParents0 if cv2.contourArea(contours[i]) > minArea]
    # parentsCNTRS = [contours[i] for i in parentSizeSel]
    pIDsAreaFiltered = []
    cIDsAreaFiltered = {} # really keeps p-c with similar aspect ratios, not area
        
    if mode >0:
        # find children of good parents. seach sub-contours with right parent ID
        whereChildren0 = [np.argwhere(hierarchy[:,:,3]==parent)[:,1] for parent in parentSizeSel]
        whereChildren0Dict0 = {parent: np.argwhere(hierarchy[:,:,3]==parent)[:,1] for parent in parentSizeSel}
        whereChildren0Dict0 = {parent: childList for parent,childList in whereChildren0Dict0.items() if len(childList)>0}
        # print(f'whereChildren0Dict0, {whereChildren0Dict0}')
        whereChildren0Dict = {}
        for i,childSet in enumerate(whereChildren0):
            temp = [child for child in childSet if cv2.contourArea(contours[child]) > minAreaChild]
            whereChildren0Dict[parentSizeSel[i]] = temp

        for parentID,childrenIDs in whereChildren0Dict0.items():
            parentArea = cv2.contourArea(contours[parentID])
            childAreas = {cID: cv2.contourArea(contours[cID]) for cID in childrenIDs}
            childID = max(childAreas, key=childAreas.get)
            if childAreas[childID]/parentArea > CPareaRatio:
                _,_,w,h = cv2.boundingRect(contours[parentID])
                _,_,w2,h2 = cv2.boundingRect(contours[childID])
                angleDiff = np.abs(np.arctan(h/w) - np.arctan(h2/w2))*180/np.pi
                # print(f'pID: {parentID}, cID: {childID} angle diff: {angleDiff}')
                if angleDiff < 10:
                    pIDsAreaFiltered.append(parentID)
                    cIDsAreaFiltered[parentID] = [cID for cID in childrenIDs if cv2.contourArea(contours[cID]) > minAreaChild]
        
    return np.array(contours,dtype=object), whereParents0,  pIDsAreaFiltered,  cIDsAreaFiltered

def maskByValue(img, mask, value=0):
    if len(img.shape) > 2 and len(value)==1:
        value =  (value, value, value)
    base = np.full(img.shape,value)
    base[mask == 255] = img[mask == 255]
    return np.uint8(base)

def getCentroidPos(inp, offset, mode, mask,value=0):
    if type(inp)==np.ndarray:
        if mode == 0:shape = inp
        if mode == 1:shape = maskByValue(inp, mask, value)
        if mode == 2:shape = mask
    else:
        shape = inp
    mp = cv2.moments(shape)#;print("moms ",mp)]
    return tuple([int(mp['m10']/mp['m00'] + offset[0]), int(mp['m01']/mp['m00'] + offset[1])])

def getCentroidPosContours(bodyCntrs,holesCntrs=[],hullArea = 0):
    areas1      = [cv2.contourArea(cntr) for cntr in bodyCntrs]
    areas0      = [cv2.contourArea(cntr) for cntr in holesCntrs]
    moms1       = [cv2.moments(cntr) for cntr in bodyCntrs]
    moms0       = [cv2.moments(cntr) for cntr in holesCntrs]
    centroids1  = [np.uint32([m['m10']/m['m00'], m['m01']/m['m00']]) for m in moms1]
    centroids0  = [np.uint32([m['m10']/m['m00'], m['m01']/m['m00']]) for m in moms0]
    totalArea   = np.sum(areas1) - np.sum(areas0)   
    endCentroid = sum([w*a for w,a in zip(areas1,centroids1)]) - sum([w*a for w,a in zip(areas0,centroids0)])      
    endCentroid /= totalArea
    returnArea  = cv2.contourArea(cv2.convexHull(np.vstack(bodyCntrs))) if hullArea == 1 else totalArea
    return  tuple(map(int,np.ceil(endCentroid))), int(returnArea)

def getContourHullArea(bodyCntrs):
    return cv2.contourArea(cv2.convexHull(np.vstack(bodyCntrs))) if len(bodyCntrs)>0 else -1 

def getCentroidPosCentroidsAndAreas(centroids1,areas1,centroids0=[],areas0=[]):
    centroids1 =  np.float32(centroids1)
    areas1 =  np.float32(areas1)
    totalMass = np.sum(areas1) - np.sum(areas0)   
    endCentroid = sum([w*a for w,a in zip(areas1,centroids1)]) - sum([w*a for w,a in zip(areas0,centroids0)])      
    endCentroid /= totalMass
    return  tuple(map(int,np.ceil(endCentroid)))

def centroidSumPermutations(IDsOfInterest, centroidDict, areaDict, refCentroid,distCheck):
    permutations = sum([list(itertools.combinations(IDsOfInterest, r)) for r in range(1,len(IDsOfInterest)+1)],[])
    cntrds2 =  np.array([getCentroidPosCentroidsAndAreas([centroidDict[k] for k in vec],[areaDict[m] for m in vec]) for vec in permutations])
    print(f'permutations,{permutations}')
    print(f'refC: {refCentroid}, cntrds2: {list(map(list,cntrds2))}')
    index, dist = closest_point(refCentroid, cntrds2)
    sol = np.sqrt(dist[index])
    #print(f'result min: {permutations[index]}',['{:.1f}'.format(np.sqrt(a)) for a in dist])
    output = [permutations[index], sol] if sol < distCheck else []
    return output

def doubleCritMinimum(setA,setB, mode = 0, debug = 0, printPrefix=''):
    print(printPrefix) if len(printPrefix)>0 and debug == 1 else 0
    
    if mode == 0:
        print('rescaling data to {0, max(set)}->{0, 1}') if debug == 1 else 0
        weightedA = np.array(setA)/max(setA)
        weightedB = np.array(setB)/max(setB)
    else:
        print('rescaling data to {min(set, max(set)}->{0, 1}') if debug == 1 else 0
        weightedA = (np.array(setA)-min(setA))/(max(setA) - min(setA))
        weightedB = (np.array(setB)-min(setB))/(max(setB) - min(setB))
    
    res = [np.mean([a,b]) for a,b in zip(weightedA,weightedB)]
    
    resIndex = np.argmin(res)
    if debug == 1:
        #print(f'setA: {[np.round(a, 2) for a in setA]}') 
        #print(f'setB: {[np.round(a, 2) for a in setB]}') 
        print(f'weightedA: {[np.round(a, 2) for a in weightedA]}') 
        print(f'weightedB: {[np.round(a, 2) for a in weightedB]}') 
        print(f'mean of weight pairs: {[np.round(a, 2) for a in res]}')
        print(f'smallest index: {resIndex}')
    return resIndex

def dropDoubleCritCopies(elseOldNewDoubleCriterium):
    IDs = [a[0] for a in elseOldNewDoubleCriterium]
    u, c = np.unique(IDs, return_counts=True);#print(f'c',c)
    if(len(c[c>1])>0): # which have more than 1 copy? if there is one or more, do code below
        dropCopies = []
        for ID, cnt in zip(u,c):    # select first uniq ID
            if cnt > 1:             # if it has morethan one copy do stuff
                where = np.argwhere(IDs==ID).flatten()
                #print(f'ID:{ID} where',where)
                # select dist and rel area sets for double crit check
                setA = [elseOldNewDoubleCriterium[here][2] for here in where]
                setB = [elseOldNewDoubleCriterium[here][3] for here in where]
                here = doubleCritMinimum(setA ,setB, mode = 0, debug = 0, printPrefix='')
                #print(f'ID:{ID} good index',here)
                deleteThese = [a for a in where if a not in [where[here]]] # drop these array IDs
                #print(f'ID:{ID} del these',deleteThese)
                dropCopies = dropCopies + deleteThese
        #print(dropCopies)
        return [entry for ID,entry in enumerate(elseOldNewDoubleCriterium) if ID not in dropCopies] # keep only useful
    else:
        return elseOldNewDoubleCriterium

def rescaleTo255(rmin,rmax,x):
    return int(255*(rmin-x)/(rmin-rmax))

def clusterPerms(refCentroid, mask, rectParams, ID, globalCounter, debug = 0):#oldCentroid, l_masks_old[oldID], l_rect_parms_old[oldID]
    #with open('./clusterPerms.pickle', 'wb') as handle:
    #            pickle.dump([refCentroid,mask,rectParams,ID,globalCounter,debug], handle) 
    x,y,w,h         = rectParams
    localCentroid   = np.array(refCentroid,dtype = int) - np.array([x,y],dtype = int)
    xs, ys          = np.meshgrid(np.arange(0,w,1), np.arange(0,h,1), sparse=True)  # all x,y pairs. hopefully its faster using meshgrid + numpy
    zs              = np.sqrt((xs-localCentroid[0])**2 + (ys-localCentroid[1])**2).astype(int)
    #rmin, rmax      = np.min(zs), np.max(zs)
    dic = {rad:0 for rad in np.sort(np.unique(zs.flatten()))} 
    for i,xses in enumerate(xs[0]):                                                     # get radius of each pixel, add to counter. 
            for j,yses in enumerate(ys):
                if mask[yses[0],xses] == 255:                                     # count only those inside contour (color = 255)
                    radi = zs[j][i]
                    dic[radi] += 1
    
    if debug == 1:
        xvals, weights  = np.array(list(dic.keys())), np.array(list(dic.values())) 
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 16), sharex=False, sharey=False)
        axes[0].scatter(xvals,weights, label=f'Radial pixel distribution ID:{ID}')
        axes[1].scatter(np.arange(len(xvals)),xvals, label=f'Radial pixel distribution ID:{ID}')
        dmin, dmax      = np.min(weights), np.max(weights)
        mask2 = mask.copy()
        for i,xses in enumerate(xs[0]):
            for j,yses in enumerate(ys):
                if mask[yses[0],xses] == 255:
                    radi                    = zs[j][i]
                    clr                     = rescaleTo255(dmin,dmax,dic[radi])             # select a grayscale value based on number of pixel at that radius
                    mask2[yses[0],xses]   = clr
        cv2.circle(mask2, localCentroid, 3,  190, -1)
        cv2.imshow('a',mask2)
        plt.show()

# findMajorInterval() - given some function data [x,f(x)] eg [[0,1,5,8],[9,4,1,5]], calculates smallest x interval with  cover_area % area. from duplicate results, one closer to weighted mean(x) is selected.
def findMajorInterval(x, fx, meanVal=None, cover_area=0.8, debug=0):
    w_nonzero   = fx.nonzero()
    x           = x[w_nonzero]
    fx          = fx[w_nonzero]
    if meanVal is None:
        meanVal = np.average(x, weights=fx)
    fx_c        = np.cumsum(fx)                # assuming x is integer, fx_c is offset by 1 from x, either offset x id by -1 or consider fx_c as sum up to and including value at x[ID]
    totalArea   = fx_c[-1]
    fx_c        = fx_c/totalArea                # normalize to 0- 1
    #fx_c = np.concatenate(([0],fx_c))   # first entry 0 area, bit of an offset.
    #print(np.vstack((x,fx_c)))#;print(fx_c - (1-cover_area))    
    # fx_c = [0.1,..., 0.7, 0.8, 0.9, 1. ]; cover_area 0.25, (fx_c - 0.75) = [-0.65 ... -0.15 -0.05  0.05  0.15  0.25] 
    # means  that 2 closest options of x at which remaining areas are 0.30 (-0.05) and 0.20 (0.05). but there is no interval from (1-0.2) = 0.8 to (1-0.2) + cover_area = 1.05
    x_right_max_index = np.argmin(np.abs(fx_c - (1-cover_area))) + 1   # abs(-0.05,0.05)- > (0.05,0.05) -> just take first since fx_c is monotone increasing f-n
    #print(f'x_right = {x[x_right_max_index]},{fx_c - (1-cover_area)}\n')
    #print(f'cumulative area at x = {x[x_right_max_index] } is {fx_c[x_right_max_index]} and x-1 = {x[x_right_max_index-1]} is {fx_c[x_right_max_index-1]} and x+1 = {x[x_right_max_index+1]} is {fx_c[x_right_max_index+1]}')
    solsIntevals2 = np.zeros(x_right_max_index+1, int)
    solsAreas2 = np.zeros(x_right_max_index+1, int)
    #print(f'cover Area %: {cover_area:.2f}')
    x_left              = x[0]                                      
    targetArea          = cover_area  
    findMin             = np.abs(fx_c - targetArea)#;print(findMin)            
    tarIndex0           = np.array(np.where(findMin == findMin.min()))[0]   
    #subTarIndex0        = np.argmin(np.abs(tarIndex0-meanVal)) # take one closer to x = 0. it is a bias. biasing towards meanVal selects bigger intervals in region [x[0],meanVal] and smaller in rest. i dont want it here.
    tarIndex0           = tarIndex0[0]                               # apply prev IDs to og tarIndex0.
    #tarIndex            = np.argmin(np.abs(fx_c - targetArea)) + 1
    x_right             = x[tarIndex0 ]
    solsIntevals2[0]    = x_right - x_left          
    solsAreas2[0]       = np.round(np.abs(cover_area-(fx_c[tarIndex0])),5) 
    if debug == 1:
        print(str(0).zfill(2)+f', x:[{x[0]:.2f} , {x[tarIndex0 ]:.2f}], x_diff: {(x[tarIndex0+1 ]-x[0]):.2f}, diff: {(fx_c[tarIndex0]):.3f}')
        print(f'cA: {0:.3f}, tarArea: {targetArea:.3f}, existingArea: {fx_c[tarIndex0]:.3f}, solAreas: {solsAreas2[0]}')
    if x_right_max_index > 0:
        for i in range(1,x_right_max_index+1,1):                            # does not reach x_right_max_index, so +1
            x_left              = x[i]                                      # area betwen x[i] and x[i+n] is (fx_c[i+n] - fx_c[i])
            prevArea            = fx_c[i-1]    
            targetArea          = cover_area + prevArea                     # fx_c[i-1] is staggered to the left. so x[i = 0] has area fx_c[i=0] of zero.
            #tarIndex            = np.argmin(np.abs(fx_c - targetArea))     # considers target value closest to target, from both top and bottom. top- wider interval. might not be best soln
            findMin             = np.round(np.abs(fx_c - targetArea),4)     #;print(findMin)  
            tarIndex0           = np.array(np.where(findMin == findMin.min()))[0]   # in case there are same entries eg. min(abs([-1,1]), take on closer to mean value.
            tarIndex            = tarIndex0[0]
            x_right             = x[tarIndex]
            solsIntevals2[i]    = x_right - x_left             
            solsAreas2[i]       = np.round(np.abs(cover_area-(fx_c[tarIndex]-fx_c[i-1])),7) # can be relative dA/A0, but all A0 same for all.
            if debug == 1:
                print(str(i).zfill(2)+f', x: [{x[i]} , {x[tarIndex]}], x_diff: {(x[tarIndex]-x[i])}, diff: {(fx_c[tarIndex]-fx_c[i-1]):.3f}')#, diff -1: {(fx_c[tarIndex-1]-fx_c[i-1]):.3f}, diff+1: {0 if tarIndex == x_right_max_index else (fx_c[tarIndex+1]-fx_c[i-1]):.3f}
                print(f'cA: {fx_c[i-1]:.3f}, tarArea: {targetArea:.3f}, existingArea: {fx_c[tarIndex]:.3f}, solAreas: {solsAreas2[i]}\n')
    
    # take the shortest interval [x[i],x[i+n]] that has area close to cover_area
    checkIntr_G = np.argwhere(solsIntevals2 == solsIntevals2.min()).flatten()           # multiple intervals of this length can be recovered (due to discrete distribution)
    minRelArea  = np.min(solsAreas2[checkIntr_G])   #take first min area                                                                    # 
    minAreas_L     = np.argwhere(solsAreas2[checkIntr_G]  == minRelArea).flatten()                              # search in subset of IDs, solution is subset ID
    minAreas_G      = checkIntr_G[minAreas_L]
    #print(f'x: {x[minAreas_G]} , 0.5 int: {0.5*solsIntevals2[minAreas_G]},mn: {meanVal}, dx:{(x[minAreas_G]+0.5*solsIntevals2[minAreas_G])-meanVal}')
    closestToMean_L     = np.argmin(np.abs((x[minAreas_G]+0.5*solsIntevals2[minAreas_G])-meanVal))
    globalMin    = minAreas_G[closestToMean_L]
    
    if debug == 1:
        _, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=False)
        axes[1].scatter(x,fx/totalArea)
        #axes[1].fill_between(x,fx,0,where=(x>=minKey2) & (x<=minKey2+solsIntevals[minKey2]),color='b')
        axes[1].fill_between(x,fx/totalArea,0,where=(x>=x[globalMin]) & (x<=x[globalMin]+solsIntevals2[globalMin]),color='g')
        axes[1].set_xlabel('radius, pix')
        axes[1].set_ylabel('density')
        #axes[1].set_xticks(x)
        axes[0].scatter(x,fx_c)
        plt.show()
    return x[globalMin],solsIntevals2[globalMin]
    # possible rework: higher f(x) values whould lead to smaller interval, but because of  discrete dx, area gained there might be too larger than $cover_area,
    # so other interval for smaller f(x) is selected, which is not physically correct. fix: add small acceptable area deviation when its checked 

# radialStatsImage() - given binary submask $mask, its position on main pictire $rectParams, calculate number of white pixels on concentric circles centered on global point $refCentroid

def radialStatsImageEllipse(isEllipse,refCentroid, mask, rectParams, ellipseParams, cover_area, ID, globalCounter, debug = 0):
    x,y,w,h         = rectParams
    localCentroid   = np.array(refCentroid,dtype = int) - np.array([x,y],dtype = int)
    xs, ys          = np.meshgrid(np.arange(0,w,1), np.arange(0,h,1), sparse=True)  # all x,y pairs. hopefully its faster using meshgrid + numpy
    xsl,ysl         = xs-localCentroid[0], ys-localCentroid[1]
    if isEllipse == 0:
        zs              = np.sqrt((xs-localCentroid[0])**2 + (ys-localCentroid[1])**2).astype(int)
        dic = {rad:0 for rad in np.sort(np.unique(zs.flatten()))} 
    else:
        a2,b2           = ellipseParams[1]                          # a2 = major diameter = 2* half diameters
        theta           = np.radians(180-ellipseParams[2])          # empirically, might be wrong? hope not.
        c, s            = np.cos(theta), np.sin(theta)

        xsl2  = c*xsl-s*ysl;ysl2  = s*xsl+c*ysl                     # rotate points w.r.t ellipse orientation. Cant find a proper way to matrix muliply with meshgrid

        alphas          = np.sqrt((xsl2*2/a2)**2 + (ysl2*2/b2)**2)  # finds at what scaled ellipse point (x,y) is located: (x/(alpha*a))^2 + (y/(alpha*b))^2 = 1 => (x/a)^2 + (y/b)^2 = alpha^2
        zs              = (alphas*a2/2).astype(int)                 # circle has only r, ellipse major and minor axis, ref major.

    dic = {rad:0 for rad in np.sort(np.unique(zs.flatten()))}       # sort all radiusses
    for i,xses in enumerate(xs[0]):                                 # get radius of each pixel, add to counter. 
        for j,yses in enumerate(ys):
            if mask[yses[0],xses] == 255:                           # count only those inside contour (color = 255)
                radi = zs[j][i]
                dic[radi] += 1
               # radi2 = alphasR[j][i]
                #dic2[radi2] += 1

    xvals, weights  = np.array(list(dic.keys())), np.array(list(dic.values()))
    #xvals2, weights2  = np.array(list(dic2.keys())), np.array(list(dic2.values()))
    avg             = np.average(xvals, weights=weights).astype(int)
    rmin, dr        = findMajorInterval(xvals,weights,avg,cover_area,debug= 0)

    if debug == 1:
        #fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False, sharey=False)
        #axes[0].scatter(xvals,weights, label=f'Radial pixel distribution ID:{ID}')
        #axes[0].fill_between(xvals,weights,0,where= (xvals<=rmin +dr) & (xvals>=rmin))
        #axes[1].scatter(np.arange(len(xvals)),xvals, label=f'Radial pixel distribution ID:{ID}')
        dmin, dmax      = np.min(weights), np.max(weights)
        mask2 = mask.copy()
        for i,xses in enumerate(xs[0]):
            for j,yses in enumerate(ys):
                if mask[yses[0],xses] == 255:
                    radi                    = zs[j][i]
                    clr                     = rescaleTo255(dmin,dmax,dic[radi])             # select a grayscale value based on number of pixel at that radius
                    mask2[yses[0],xses]   = clr

        cv2.imshow(f'gc: {globalCounter}, oldID: {ID}',mask2)
        #fig.suptitle(f'gc: {globalCounter}, bID: {ID}', fontsize=16)
        plt.show()
    return np.array([rmin, rmin + dr],int), dic

# radialStatsContours() -  same as radialStatsImage() but for collection of $IDsOfInterest contours from list $bodyCntrs


def radialStatsContoursEllipse(isEllipse,bodyCntrs,IDsOfInterest,refCentroid, ellipseParams, cover_area, img, pltID, globalCounter, debug = 0):

    output = {ID:np.zeros(3,int) for ID in IDsOfInterest}                                          # future return dict {ID:[avg_r,stdev_r]}
    output_dist = {}
    if debug == 1:
        imgGray = img.copy()*0 #cv2.cvtColor(img.copy()*0, cv2.COLOR_BGR2GRAY) 
        n = len(IDsOfInterest)
        axes = plt.subplots(int(np.ceil(np.sqrt(n))), int(np.ceil(n/np.ceil(np.sqrt(n)))), figsize=(16, 9), sharex=True, sharey=False)[1]
        if n>1:
            axes = axes.reshape(-1)
        else: axes = [axes]
 
    for k,ID in enumerate(IDsOfInterest):
        #area0           = cv2.contourArea(bodyCntrs[ID])
        x,y,w,h         = cv2.boundingRect(bodyCntrs[ID])
        xs, ys          = np.meshgrid(np.arange(x,x+w,1), np.arange(y,y+h,1), sparse=True)  # all x,y pairs. hopefully its faster using meshgrid + numpy
        xsl,ysl         = xs-refCentroid[0], ys-refCentroid[1]
        if isEllipse == 0:
            zs              = np.sqrt(xsl**2 + ysl**2).astype(int)      # calculate L2 norms from reference centroid.
            dic = {rad:0 for rad in np.sort(np.unique(zs.flatten()))} 
        else:
            a2,b2           = ellipseParams[1]                          # a2 = major diameter = 2* half
            theta           = np.radians(180-ellipseParams[2])          # empirically, might be wrong? 
            c, s            = np.cos(theta), np.sin(theta)
        
            xsl2  = c*xsl-s*ysl;ysl2  = s*xsl+c*ysl                     # rotate points w.r.t ellipse o
        
            alphas          = np.sqrt((xsl2*2/a2)**2 + (ysl2*2/b2)**2)  # finds at what scaled ellipse 
            zs              = (alphas*a2/2).astype(int)                 # circle has only r, ellipse ma
        #zs              = np.sqrt((xs-refCentroid[0])**2 + (ys-refCentroid[1])**2).astype(int)  
        rmin, rmax      = np.min(zs), np.max(zs)                                            # rmin/max of whole image, so 0 pixel count for some r values.
        dic = {rad:0 for rad in np.arange(rmin,rmax+1,1)}                                   # if there is a discontinuity, because of casting to int, 0-count rads will be dropped anyway. EG (1.05, 1.99, 3.01)- > (1,1,3)
        
        #dic = {rad:0 for rad in np.sort(np.unique(zs.flatten()))}                          # order (sort) should not be important if not drawing a continious relation [r1,n1],[r2,n2],...
        subSubMask      = np.zeros((h,w),np.uint8)                                          # stencil for determining if pixel is part of a bubble
        cv2.drawContours( subSubMask, bodyCntrs, ID, 255, -1, offset = (-x,-y))
        
        for i,xses in enumerate(xs[0]):                                                     # get radius of each pixel, add to counter. 
            for j,yses in enumerate(ys):
                if subSubMask[yses[0]-y,xses-x] == 255:                                     # count only those inside contour (color = 255)
                    radi = zs[j][i]
                    dic[radi] += 1
                    #clr = rescaleTo255(rmin,rmax,radi)
                    #imgGray[yses[0],xses] = clr
    
        xvals, weights  = np.array(list(dic.keys())), np.array(list(dic.values()))           # cast to numpy to do statistics
        avg             = np.average(xvals, weights=weights)     .astype(int)                            # weighted average
        #stdev           = np.sqrt(numpy.average((xvals-avg)**2, weights=weights))            # stdev of weighted data. theres no ready function in numpy.
        rmin, dr        = findMajorInterval(xvals,weights,avg,cover_area,debug= 0)      #;print(a,b)(x,fx,meanVal,cover_area,debug= 1)
        dmin, dmax      = min(dic.values()),max(dic.values())
        output[ID]      = np.array([ rmin, rmin + dr],int)
        output_dist[ID] = dic
        if debug == 1:
            
            axes[k].plot(xvals,weights)
            axes[k].vlines(avg, min(dic.values()), max(dic.values()), linestyles ="dashed", colors ="k")
            axes[k].fill_between(xvals,weights,0,where= (xvals<=rmin +dr) & (xvals>=rmin))
            axes[k].set_xlabel('radius, pix')
            axes[k].set_ylabel('sum of pixels')
            axes[k].set_title(f'Radial pixel distribution ID:{ID}')
            
            for i,xses in enumerate(xs[0]):
                for j,yses in enumerate(ys):
                    if subSubMask[yses[0]-y,xses-x] == 255:
                        radi                    = zs[j][i]
                        clr                     = rescaleTo255(dmin,dmax,dic[radi])             # select a grayscale value based on number of pixel at that radius
                        imgGray[yses[0],xses]   = clr
            x0,y0 = bodyCntrs[ID][0][0]
            cv2.putText(imgGray, str(ID), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 3, cv2.LINE_AA)
            cv2.putText(imgGray, str(ID), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
            
    if debug == 1:
        cv2.circle(imgGray, tuple(map(int,refCentroid)), 3, (255,0,0), -1)
        cv2.imshow(f'gc: {globalCounter}, oldID: {pltID}',imgGray)
        plt.tight_layout()
        plt.show()
    return output, output_dist

def compareRadial(OGband, OGDistr, SlaveBand, SlaveDistr,solution,cyclicColor,globalCounter,oldID):
    subNewIDs = np.array(list(SlaveDistr.keys()))
    rmin  = 10000;rmax = 0
    for ID,distr in SlaveDistr.items():
        arr = np.array(list(distr.keys()))
        dMin = np.min(arr);dMax = np.max(arr)
        rmin = min(rmin,dMin);rmax = max(rmax,dMax)
    #print(rmin,rmax)
    domain = np.arange(rmin,rmax+1,1)
    vals = np.zeros(len(domain))
    for ID,distr in SlaveDistr.items():
        if ID in solution:
            for r,subVal in distr.items():
                vals[r-rmin] += subVal # if dom = [rmin,rmin+1,...] distr: {{rmin:val},..}, val should go to vals[rmin-rmin] => vals[0]
        

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False, sharey=False)
    t = 8
    ogDom = list(OGDistr.keys())
    ogVals = list(OGDistr.values())
    axes[0].plot(ogDom,ogVals,lw = 2,color=np.array(cyclicColor(0))/255,linestyle='dashed')
    #axes[0].plot([r,r+dr],[max(ogVals)+ t,max(ogVals)+ t],lw = 3,color=np.array(cyclicColor(0))/255)
    axes[0].fill_between(ogDom,ogVals,0,where= (ogDom>=OGband[0]) & (ogDom<=OGband[1]),color=np.array(cyclicColor(0))/255)  # OGband = [left,right] boundaries
    r1 = np.array([rr+0.5*(rr2-rr) for rr,rr2 in SlaveBand.values()])
    ordr = np.argsort(r1)
    for i,subID in enumerate(subNewIDs[ordr]):
        i += 1
        dom = list(SlaveDistr[subID].keys())
        val = list(SlaveDistr[subID].values())
        axes[0].plot(dom,val,c=np.array(cyclicColor(i))/255, lw = 3, label = str(subID))
        r1,r2 = SlaveBand[subID]
        axes[0].plot([r1,r2],[-t*(i+1),-t*(i+1)],color=np.array(cyclicColor(i))/255, lw = 4)
    axes[0].legend()
    axes[1].scatter(ogDom,ogVals,s = 12,label = "original")
    axes[1].scatter(domain,vals,s = 12,label = str(solution))
    axes[1].legend()
    fig.suptitle(f'gc: {globalCounter}, oldID: {oldID}, ids: {subNewIDs}')
    plt.show()

def centroidAreaSumPermutations(bodyCntrs,rectParams,rectParams_old, IDsOfInterest, centroidDict, areaDict, refCentroid, distCheck, refArea=10, relAreaCheck = 10000000, doHull = 1,customPermutations = [], debug = 0):
    #bodyCntrs is used only if doHull == 1. i dont use it for frozen bubs permutations
    #print(rectParams)
    #with open('./cntr.pickle', 'wb') as handle:
    #            pickle.dump(bodyCntrs, handle) 
    if len(customPermutations) == 0:
        permutations = sum([list(itertools.combinations(IDsOfInterest, r)) for r in range(1,len(IDsOfInterest)+1)],[])                              # different combinations of size 1 to max cluster size.
    else:permutations = customPermutations

    cntrds2 =  np.array([getCentroidPosCentroidsAndAreas([centroidDict[k] for k in vec],[areaDict[m] for m in vec]) for vec in permutations])   # NOT OF HULLS !!!
    if doHull == 1:
        hullAreas = np.array([cv2.contourArea(cv2.convexHull(np.vstack([bodyCntrs[k] for k in vec]))) for vec in permutations])
    else:
        hullAreas = np.array([sum([areaDict[m] for m in vec]) for vec in permutations])
    print(f'permutations,{permutations}') if debug == 1 else 0
    print(f'refC: {refCentroid}, cntrds2: {list(map(list,cntrds2))}') if debug == 1 else 0
    distances = np.linalg.norm(cntrds2-refCentroid,axis=1)
    distPassIndices = np.where(distances<distCheck)[0]
    relAreas = np.abs(refArea-hullAreas)/refArea
    relAreas2 = (refArea-hullAreas)/refArea 
    relAreasPassIndices = np.where(relAreas<relAreaCheck)[0]
    passBothIndices = np.intersect1d(distPassIndices, relAreasPassIndices)
    # added aspect ratio check - angle difference check 24/02/23
    rects = [cv2.boundingRect(np.vstack([bodyCntrs[k] for k in perm])) for perm in permutations]
    angles = np.array([boundRectAngle(rect,rectParams_old, maxAngle = 10, debug  = 0, info= '')[1] for rect in rects],np.int16)
    anglePassIndices = np.where(angles<20)[0]
    passBothIndices = np.intersect1d(passBothIndices, anglePassIndices)

    if len(passBothIndices)>0:
        if len(passBothIndices) == 1:
            print(f'only 1 comb after stage 1: {list(permutations[passBothIndices[0]])}') if debug == 1 else 0 
            return list(permutations[passBothIndices[0]]), distances[passBothIndices[0]], relAreas[passBothIndices[0]]
        
        remainingPermutations = np.array(permutations, dtype=object)[passBothIndices]
        print(f'remainingPermutations: {remainingPermutations}') if debug == 1 else 0
        #remainingCentroids = np.array(cntrds2)[passBothIndices]
        remainingDistances = np.array(distances)[passBothIndices]
        remainingRelAreas = np.array(relAreas)[passBothIndices]
        remainingRelAreas2 = np.array(relAreas2)[passBothIndices]
        print(f'remainingDistances: {remainingDistances}') if debug == 1 else 0
        print(f'remainingRelAreas: {remainingRelAreas}') if debug == 1 else 0
        remainingCentroids = np.array(cntrds2)[passBothIndices]

        # ==== permutations blindly pick combinations that suit criteria the best, sometimes that means that lone elements are left inside clusters, which is non-physical. ====
        if debug == 1:
            candidates = []
            x0,y0,w0,h0 = rectParams_old
            remainingPermutationsList = [list(a) for a in remainingPermutations]

            for i,ID in enumerate(passBothIndices):
                perm = permutations[ID]
                if 1 < len(perm) < len(IDsOfInterest) and remainingRelAreas[i] < 0.3:
                    
                    xt,yt,wt,ht = cv2.boundingRect(np.vstack([bodyCntrs[k] for k in perm]))                             # small permutation in question
                    rotatedRectangle_perm = ((xt+wt/2, yt+ht/2), (wt, ht), 0)

                    for elem in [a for a in IDsOfInterest if a not in perm]:                                            # elem = subID: check all ids thats not in perm

                        x1,y1,w1,h1 = rectParams[elem]                                                                  # subID, take its parameters
                        rotatedRectangle_elem = ((x1+w1/2, y1+h1/2), (w1, h1), 0)        
                        interType,aa = cv2.rotatedRectangleIntersection(rotatedRectangle_perm, rotatedRectangle_elem)   # check if elem (SubID) intersects with small perm

                        join = list(perm) + [elem] ;join.sort() #(15, 18, 20, 22, 25, 30))                              # form big perm from small perm and elem (subID)
                        if interType > 0 and join in remainingPermutationsList:                                         # if there there intersection and big per was not discarded previously
                            idBig       = remainingPermutationsList.index(join)                                         # idBig: find ID of big perm in all releveant perm IDs
                            cBig        = remainingCentroids[idBig]                                                     # grab big perm centroid
                            cSmall      = remainingCentroids[i]                                                         # grab small perm centroid
                            deltaD      = cBig-cSmall
                            #ddist   = np.linalg.norm(,axis=0)
                            candidates.append(join)
                            img = np.zeros((800,1200,3),np.uint8)
                            cv2.rectangle(img, (x0,y0), (x0+w0,y0+h0), (0,125,255), 1)
                            cv2.rectangle(img, (xt,yt), (xt+wt,yt+ht), (255,0,0), 1)
                            #cv2.rectangle(img, (x1,y1), (x1+w1,y1+h1), (120,255,0), 1)
                            cv2.drawContours(img, [aa.astype(int)], -1, (0,0,255), -1)
                            [cv2.drawContours(  img,   bodyCntrs, cid, (125,250,120), 1) for cid in perm]
                            cv2.drawContours(  img,   bodyCntrs, elem, (250,120,120), 1)
                            cv2.circle(img, tuple(cBig), 1,  (0,0,255), -1)
                            cv2.circle(img, tuple(cSmall), 1, (255,0,0), -1)
                            cv2.putText(img, "big rA: "+str(np.around(remainingRelAreas2[idBig],4)), (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
                            cv2.putText(img, "sml rA: "+str(np.around(remainingRelAreas2[i],4)), (200,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
                            cv2.imshow(str(i)+": "+str(perm)+" bonus: "+str(elem) ,img)
                            1
        resIndex = doubleCritMinimum(remainingDistances,remainingRelAreas, mode = 0, debug = debug, printPrefix='')
        return list(remainingPermutations[resIndex]), remainingDistances[resIndex], remainingRelAreas[resIndex]#,  remainingCentroids[resIndex]
    else: 
        return [],-1,-1#,[]

def centroidAreaSumPermutations2(bodyCntrs,rectParams,rectParams_old, permutations, centroidDict, areaDict, refCentroid, distCheck, refArea=10, relAreaCheck = 10000000, doHull = 1, debug = 0):
    #bodyCntrs is used only if doHull == 1. i dont use it for frozen bubs permutations
    #print(rectParams)
    #with open('./cntr.pickle', 'wb') as handle:
    #            pickle.dump(bodyCntrs, handle) 
    cntrds2 =  np.array([getCentroidPosCentroidsAndAreas([centroidDict[k] for k in vec],[areaDict[m] for m in vec]) for vec in permutations])   # NOT OF HULLS !!!
    if doHull == 1:
        hullAreas = np.array([cv2.contourArea(cv2.convexHull(np.vstack([bodyCntrs[k] for k in vec]))) for vec in permutations])
    else:
        hullAreas = np.array([sum([areaDict[m] for m in vec]) for vec in permutations])
    print(f'permutations,{permutations}') if debug == 1 else 0
    print(f'refC: {refCentroid}, cntrds2: {list(map(list,cntrds2))}') if debug == 1 else 0
    distances = np.linalg.norm(cntrds2-refCentroid,axis=1)
    distPassIndices = np.where(distances<distCheck)[0]
    relAreas = np.abs(refArea-hullAreas)/refArea
    relAreas2 = (refArea-hullAreas)/refArea 
    relAreasPassIndices = np.where(relAreas<relAreaCheck)[0]
    passBothIndices = np.intersect1d(distPassIndices, relAreasPassIndices)
    # added aspect ratio check - angle difference check 24/02/23
    rects = [cv2.boundingRect(np.vstack([bodyCntrs[k] for k in perm])) for perm in permutations]
    angles = np.array([boundRectAngle(rect,rectParams_old, maxAngle = 10, debug  = 0, info= '')[1] for rect in rects],np.int16)
    anglePassIndices = np.where(angles<20)[0]
    passBothIndices = np.intersect1d(passBothIndices, anglePassIndices)

    if len(passBothIndices)>0:
        if len(passBothIndices) == 1:
            print(f'only 1 comb after stage 1: {list(permutations[passBothIndices[0]])}') if debug == 1 else 0 
            return list(permutations[passBothIndices[0]]), distances[passBothIndices[0]], relAreas[passBothIndices[0]]
        
        remainingPermutations = np.array(permutations, dtype=object)[passBothIndices]
        print(f'remainingPermutations: {remainingPermutations}') if debug == 1 else 0
        #remainingCentroids = np.array(cntrds2)[passBothIndices]
        remainingDistances = np.array(distances)[passBothIndices]
        remainingRelAreas = np.array(relAreas)[passBothIndices]
        remainingRelAreas2 = np.array(relAreas2)[passBothIndices]
        print(f'remainingDistances: {remainingDistances}') if debug == 1 else 0
        print(f'remainingRelAreas: {remainingRelAreas}') if debug == 1 else 0
        remainingCentroids = np.array(cntrds2)[passBothIndices]


        resIndex = doubleCritMinimum(remainingDistances,remainingRelAreas, mode = 0, debug = debug, printPrefix='')
        return list(remainingPermutations[resIndex]), remainingDistances[resIndex], remainingRelAreas[resIndex]#,  remainingCentroids[resIndex]
    else: 
        return [],-1,-1#,[]

def centroidSumPermutationsMOD(oldNewRelationArray, centroidDict, areaDict, refCentroidsDict,distCheck):
    u, c = np.unique(oldNewRelationArray[:,0], return_counts=True)
    dup = u[c > 1]
    print(f'duplicates detected: {dup}')
    output = {}
    for i,d in enumerate(dup):
        dupPos = np.where(oldNewRelationArray[:,0] == d)[0]
        nums = [oldNewRelationArray[k,1] for k in dupPos]
        out = centroidSumPermutations(nums, centroidDict, areaDict, refCentroidsDict[d],distCheck)
        output[d] = out
    return output

from skimage.metrics import structural_similarity
# global gg
gg = 0;
def compareMoments(big,shape1,shape2,coords1,coords2, debug):
    global gg
    (x,y,w,h) = coords1
    diag1 = np.sqrt(w**2+h**2)
    maskedFull1 = np.zeros(big.shape,np.uint8)
    maskedFull1[y:y+h, x:x+w] = shape1
    (x,y,w,h) = coords2
    diag2 = np.sqrt(w**2+h**2)
    maskedFull2 = np.zeros(big.shape,np.uint8)
    maskedFull2[y:y+h, x:x+w] = shape2
    moms1,moms2 = cv2.moments(maskedFull1),cv2.moments(maskedFull2)
    areas = []
    for img in [shape1,shape2]:
        contours = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        hull = cv2.convexHull(np.vstack(contours))
        areas.append(cv2.contourArea(hull))
    # print('areas',areas)
    # pass centroid-centroid dist
    cntds = np.array([
                        [int(moms1['m10']/moms1['m00']),int(moms1['m01']/moms1['m00'])]
                        ,
                        [int(moms2['m10']/moms2['m00']),int(moms2['m01']/moms2['m00'])]
                    ])#;print('cntds',cntds)
    # df = np.diff(cntds,axis=0);print('df',df)
    dist = np.linalg.norm(np.diff(cntds,axis=0))
    charDist = 0.5*(diag1+diag2)  # <<<<< change if it fails
    # area1, area2 = moms1['m00'] , moms2['m00']
    area1, area2 = areas
    if area1 > area2: areaChangePrec = 1- area2/area1
    else: areaChangePrec = 1- area1/area2
    if debug:
        print(f'dist: {dist:.1f}; charDist: {charDist:.1f}; areaChangePrec: {areaChangePrec:.2f}')
    
    if dist< charDist and areaChangePrec< 0.3: 
    # if 1 == 1: 
        #useKeys = ('m20','m11','m02','m30',
        #           'm21','m12','m03','mu20')#,'mu11','mu02','mu30'
        #moms1a = [moms1[x] for x in useKeys]
        #moms2a = [moms2[x] for x in useKeys]
        #ratio = [a/b for a,b in zip(moms1a,moms2a)]
        #a =     np.abs(1-np.array(ratio))
        #weights = [3, 6, 3, 3, 3, 3, 3, 1]
        #w_avg = np.average(a, weights = weights)
        #std_dev = np.sum(weights * (a - w_avg)**2)/np.sum(weights)#;print(std_dev)
        #zero_based = np.abs(a - w_avg)#;print(zero_based)
        #max_deviations = 2
        #outliers = a[zero_based > max_deviations * std_dev]#;print('outliers',outliers)
        #outlierNames = [ i for i,c in enumerate(zero_based) if c >  max_deviations * std_dev]
        #outlierNames = [useKeys[i] for i in outlierNames]#;print('outlierNames',outlierNames)
        
        #stree = f'mean = {w_avg:.3f}; '+f'stdev = {std_dev:.3f}; '+" ".join([f'{a}= {b:1.3f};' for a,b in zip(outlierNames,outliers)])
    
        #print(stree)
        #print(" ".join([f'{a}= {b:1.3f};' for a,b in zip(useKeys,ratio)]))
        # if w_avg>= 0.2:
            # cv2.imshow(f'{gg}_1 mean: {w_avg:.3f}, std_dev: {std_dev:.3f}',maskedFull1)
            # cv2.imshow(f'{gg}_2 mean: {w_avg:.3f}, std_dev: {std_dev:.3f}',maskedFull2)
            # (score, diff) = structural_similarity(maskedFull1, maskedFull2, full=True)
            # print("Image Similarity: {:.4f}%".format(score * 100))
            # cv2.imwrite("D:\\Alex\\Darbs.exe\\Python_general\\bubble_process\\imageMainFolder_output\\compareMoments\\"+f"{gg}_bub01"+".png" ,maskedFull1)
            # cv2.imwrite("D:\\Alex\\Darbs.exe\\Python_general\\bubble_process\\imageMainFolder_output\\compareMoments\\"+f"{gg}_bub02"+".png" ,maskedFull2)
        gg+=1
        return 1
    else:
        return 0
    

def checkMoments(err,gs1,gs2,masks1,masks2,recPars1,recPars2):
    shape = (len(masks1),len(masks2))
    momsRatio = np.ones((24,shape[0],shape[1]),np.double)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            mask1,mask2 = masks1[i],masks2[j]
            masked1 = maskByValue(img = gs1[i], mask = mask1 , value = 0)
            masked2 = maskByValue(img = gs2[j], mask = mask2 , value = 0)
            
            (x,y,w,h) = recPars1[i]
            maskedFull1 = np.zeros(err.shape,np.uint8)
            maskedFull1[y:y+h, x:x+w] = masked1
            (x,y,w,h) = recPars2[j]
            maskedFull2 = np.zeros(err.shape,np.uint8)
            maskedFull2[y:y+h, x:x+w] = masked2
            moms1,moms2 = cv2.moments(maskedFull1),cv2.moments(maskedFull2)
            moms1a = [moms1[x] for x in moms1.keys()]#;print(f'i: {i}, j: {j}',moms1)
            moms2a = [moms2[x] for x in moms2.keys()]#;print(f'i: {i}, j: {j}',moms2)
            ratio2 = [a/b for a,b in zip(moms1a,moms2a)]#;print(f'RATIO: i: {i}, j: {j}',ratio2)
            # print(moms1.keys())
            for s,val in enumerate(ratio2):
                momsRatio[s,i,j] = val
            


    workdata = momsRatio
    workShape  = workdata.shape
    temp = list()
    for i in range(workShape[0]):
        beb = list()
        for j in range(workShape[1]):
            arr = np.array(workdata[i,j])
            arrDifAbs = np.abs(arr-1)
            mn = min(arrDifAbs)
            where = np.where(arrDifAbs==mn)
            beb.append(where)
        beb = np.array(beb).flatten()
        temp.append(beb)
    
    fig, axes = plt.subplots(3, 8, figsize=(12, 6), sharex=True, sharey=True)
    momNames = list(moms1.keys())
    for i in range(3):
        for j in range(8):
            axes[i,j].plot(range(workShape[1]),temp[int(8*i + j)])
            axes[i,j].set_title(momNames[int(8*i + j)])
            
            
def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance


# https://stackoverflow.com/questions/72109163/how-to-find-nearest-points-between-two-contours-in-opencv 
def closes_point_contours(c1,c2):
    # initialize some variables
    min_dist = 4294967294
    chosen_point_c2 = None
    chosen_point_c1 = None
    # iterate through each point in contour c1
    for point in c1:
        t = point[0][0], point[0][1] 
        index, dist = closest_point(t, c2[:,0]) 
        if dist[index] < min_dist :
            min_dist = dist[index]
            chosen_point_c2 = c2[index]
            chosen_point_c1 = t
    d_vec = tuple(np.diff((chosen_point_c1,chosen_point_c2[0]),axis=0)[0])
    d_mag = np.int32(np.sqrt(min_dist))
    return d_vec, d_mag , tuple(chosen_point_c1), tuple(chosen_point_c2[0])
    # # draw the two points and save
    # cv2.circle(img,(chosen_point_c1), 4, (0,255,255), -1)
    # cv2.circle(img,tuple(chosen_point_c2[0]), 4, (0,255,255), -1)

def closestDistancesContours(contoursDict):
    contourIDs = list(contoursDict.keys())
    IDcombinations = np.unique(np.sort(np.array(list(itertools.permutations(contourIDs, 2)))), axis = 0)
    output = {ID:np.zeros((len(contourIDs)-1,2),int) for ID in contourIDs}                                  # every ID has N-1 connections
    counter  = {ID:0 for ID in contourIDs}
    for i,j in IDcombinations:
       distance = int(closes_point_contours(contoursDict[i],contoursDict[j])[1])                            # distance between contour i and j , i != j
       ii = counter[i];jj = counter[j]                                                                      # maybe better to pre-reserve memory, so this is the pos counter
       output[i][ii] = np.array((j,distance), int)                                                          # since dist i->j is same as j-> fill both
       output[j][jj] = np.array((i,distance), int)
       counter[i] += 1;counter[j] += 1
    return output 

showDistDecay = True    
def distStatPrediction(trajectory, startAmp0 = 20, expAmp = 20, halfLife = 1, numsigmas = 2, plot=0, extraStr = ''):
    global showDistDecay
    numSteps = len(trajectory) - 1
    decay = 0;mags = []
    lam = 0.693/halfLife  # decay coef based on half-life. ln(2)/t_1/2
    cutoff = int(2.30/lam)      # at which step value reaches 1/10 of E(0)
    if numSteps == 0: distCheck = startAmp0
    if numSteps >= 1 and numSteps <= cutoff + 1: decay = expAmp*np.exp(-lam * (numSteps-1)) 
    if numSteps > 0: distCheck = decay 
    if numSteps >= 1:                  
        mags = np.linalg.norm(np.diff(trajectory,axis=0),axis=1) # magnitude of centroid displacement
        distCheck += np.mean(mags)
    if numSteps >= 2: 
        distCheck += numsigmas* np.std(mags) 
        
    if (plot == 1 and numSteps >=  cutoff+1 and showDistDecay == True) or plot == 2:
        fig, axes = plt.subplots(1, 1, figsize=(16, 9), sharex=False, sharey=False)
        fig.suptitle(f'A0 {startAmp0}, A {expAmp}, cutoff {cutoff}, total Traj len {len(trajectory)}\n')
        axes.axvspan(xmin = 2, xmax = cutoff+1,alpha = 0.1 , label = 'exp zone')
        addedBoost = np.concatenate(([startAmp0] , expAmp*np.exp(-lam * ( np.arange(1,cutoff+1,1)-1) ))) 
        steps = np.arange(1, len(addedBoost)+1 ,1)
        axes.plot(steps,addedBoost,label=f' A0 and Aexp(N) ',linestyle = '--',marker="o")
        
        maxStepsDraw = min(numSteps,10)
        magsVals = mags[:maxStepsDraw]
        stepsNum = np.arange(2,maxStepsDraw+2,1)
        axes.plot(stepsNum,magsVals,linestyle = '-',marker="o",label='delta d -> d_i- d_(i-1)')
        
        means = [np.mean(magsVals[:i]) for i in range(1, len(magsVals)+1,1)]# if i < len(magsVals)
        meansSteps = np.arange(2,len(means)+2,1)
        axes.plot(meansSteps,means,linestyle = '--',marker="o",label= 'runing delta d mean')
        
        sigmas = np.array([np.std(magsVals[:i+1]) for i in range(1,len(magsVals),1)])*numsigmas
        sigmas = np.pad(sigmas, (1, 0), 'constant')
        sigmasSteps = np.arange(3,len(sigmas)+3,1)#[print(magsVals[:i+1]) for i in range(1,len(magsVals),1) ]
        axes.errorbar(meansSteps, means, yerr=sigmas, label = f' running d {numsigmas}*stdev',  lolims=True, linestyle='')
        # axes.plot(sigmasSteps,sigmas,linestyle = '-.',label=f'running mean + {numsigmas}*stdev')
        # axes.fill_between(sigmasSteps, sigmas, means[1:],label = f' running d {numsigmas}*stdev', alpha = 0.1, color='r')
        
        if cutoff + 1 > maxStepsDraw + 1:
            sumAll = addedBoost[:maxStepsDraw+1]
        else:
            sumAll = np.pad(addedBoost, (0, maxStepsDraw+1 - len(addedBoost)), 'constant')
            
        sumAll += np.pad(means, (1, 0), 'constant')
        sumAll += np.pad(sigmas, (1, 0), 'constant') # it was padded once previously
        
        stepsAll = np.arange(1, len(sumAll)+1,1)
        axes.plot(stepsAll,sumAll,linestyle = '-',marker="o",label=' A0+Aexp+mean1+meanStd',linewidth = 2.6)
        axes.set_title(extraStr+'Distance progression')
        axes.legend()
        axes.set_ylim(bottom=0)
        axes.set_xlim(left=1)
        axes.set_xlabel('number of steps N in trajectory')
        axes.set_ylabel('displacement magnitude, px')
        showDistDecay = False
    return distCheck

showDistDecay = True   
from matplotlib.patches import Circle
def distStatPredictionVect(trajectory, zerothDisp, sigmasDeltas = [], numdeltas = 5, maxInterpSteps = 3, maxInterpOrder = 2, mode = 1,debug = 0, maxNumPlots = 4):
    global showDistDecay
    predictPoints = []
    numPointsInTraj = len(trajectory)
    numStepsInTraj = numPointsInTraj - 1 
    #sigmasDeltas = [[10,3]]*numPointsInTraj
    if mode  ==  1:
        numStepsFor = list(range(max(0,numPointsInTraj-maxNumPlots),len(trajectory)))
    else:
        numStepsFor = [numStepsInTraj]
    numPlots = len(numStepsFor) if mode  ==  1 else 2
    if debug == 1:
        fig, axes = plt.subplots(1,numPlots , figsize=( numPlots*5,5), sharex=True, sharey=True)
    for numSteps,numSteps2 in enumerate(numStepsFor):
        numSteps = 1 if mode == 0  else numSteps

        start = 0 if numSteps2 < maxInterpSteps else numSteps2-maxInterpSteps
        x = np.array([a[0] for a in trajectory[start:numSteps2+1]])
        y = np.array([a[1] for a in trajectory[start:numSteps2+1]])
        t = np.arange(0,len(x),1)
        t1 = np.arange(0,len(x)+1,1)
        
        if numSteps2 == 0:
            predictPoints.append([trajectory[0][0]+zerothDisp[0],trajectory[0][1]+zerothDisp[1]])
            if debug == 1:
                axes[numSteps].plot(x, y, 'o',c='green', label = 'traj')
                axes[numSteps].plot([x[0],predictPoints[0][0]], [y[0],predictPoints[0][1]], '--o', label = 'forecast')
                
        if numSteps2 > 0:
            k = min(numSteps2,maxInterpOrder); print(f' interpOrder = {k}') if debug == 1 else 0
            spline, _ = interpolate.splprep([x, y], u=t, s=0,k=k)
            new_points = interpolate.splev(t1, spline,ext=0)
            if debug == 1:
                axes[numSteps].plot(new_points[0][-2:],new_points[1][-2:], '--o', label = 'forecast')
            if mode != 0:
                if numSteps > 0 and debug == 1:
                    axes[numSteps].plot([x[-2],predictPoints[-1][0]],[y[-2],predictPoints[-1][1]], '--o', label = 'prev forecast')
                    if len(sigmasDeltas)>0:
                        circleMean = Circle(tuple(predictPoints[-1]), sigmasDeltas[numSteps][0] , alpha=0.1) 
                        circleNStd = Circle(tuple(predictPoints[-1]), sigmasDeltas[numSteps][1]*numdeltas , alpha=0.1,fill=False) 
                        axes[numSteps].add_patch(circleMean)
                        axes[numSteps].add_patch(circleNStd)
                        axes[numSteps].text(*predictPoints[-1], s = f'm: {sigmasDeltas[numSteps][0]:0.2f}, s:{sigmasDeltas[numSteps][1]:0.1f}')
                predictPoints.append([new_points[0][-1],new_points[1][-1]])
                
            else: predictPoints.append([new_points[0][-1],new_points[1][-1]])
        if debug == 1: axes[numSteps].plot(x, y, '-o',c='green', label = 'traj')

    if debug == 1:
        plt.legend(loc=(1.1, 0.5))
        plt.show()
    return np.array(predictPoints[-1],np.uint32)

def extrapolate(data, maxInterpSteps = 3, maxInterpOrder = 2, smoothingScale = 0, zerothDisp = [],fixSharp = 0, angleLimit = 30, debug = 0, axes=[], pltString = ''):
    data = np.array(data).reshape(len(data),-1) #[1,2]->[[1],[2]]; [[11,12],[21,22]]-> itself
    numPointsInTraj = data.shape[0]
    numStepsInTraj = numPointsInTraj - 1 #;print(f'numPointsInTraj:{numPointsInTraj}, numStepsInTraj:{numStepsInTraj}')
    numDims = data.shape[1]
    # start-> take last maxInterpSteps
    start = 0 if numStepsInTraj < maxInterpSteps else numStepsInTraj-maxInterpSteps
    if numStepsInTraj == 0:
        zeroth = [0]*numDims if len(zerothDisp) == 0 else zerothDisp
        return data[0] + zeroth
    k = min(numStepsInTraj,maxInterpOrder)
    t = np.arange(0,numPointsInTraj-start,1)#;print(f't:{t},(numPointsInTraj-star):{numPointsInTraj-start}')
    if numDims == 1:data = np.hstack([t.reshape(len(t),-1),data])
    splitSubsetComponents = data[start:].T#;print(data)
    #if debug == 1: tDebug = np.arange(0,numPointsInTraj-start+0.01,0.2)#;print(f'tDebug:{tDebug}')
    sMod = numPointsInTraj+np.sqrt(2*numPointsInTraj) # max proper range from docs.
    if fixSharp == 1:
        alphaMax = 1.2
        alphas = np.arange(smoothingScale,alphaMax+0.0001,0.2)
        for alpha in alphas:
            sMod2 = alpha*sMod
            spline0, _ = interpolate.splprep(splitSubsetComponents, u = t, s = sMod2, k = k)

            extrapolatedPoint0 = np.array(interpolate.splev([t[-1],t[-1]+1], spline0,ext=0),int)
            dv0 = np.diff(extrapolatedPoint0).reshape(max(2,numDims))
            z = np.polyfit(*splitSubsetComponents[:,-3:], 1);#print(f'z:{z}')
            x0,x1 = splitSubsetComponents[0,-min(numPointsInTraj,3)], splitSubsetComponents[0,-1]
            p0,p1 = np.array([x0,np.dot([x0,1],z)]),np.array([x1,np.dot([x1,1],z)]) #[ x0,y(x0)], or use poly1d lol
            dv = (lambda x: x/np.linalg.norm(x)) (p1-p0)*np.linalg.norm(dv0)
            angleDeg = (lambda x,y: np.degrees(np.arccos(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1.0, 1.0))))(dv0,dv)
            if angleDeg <= angleLimit or alpha == alphas[-1]:
                returnVec =  extrapolatedPoint0[:,-1]
                break
            
    else:
        alpha = smoothingScale
        sMod2 = alpha*sMod
        spline0, _ = interpolate.splprep(splitSubsetComponents, u = t, s = sMod2, k = k)
        extrapolatedPoint0 = np.array(interpolate.splev([t[-1],t[-1]+1], spline0,ext=0),int)
        returnVec = extrapolatedPoint0[:,-1]

    if debug == 1:
        tDebug = np.arange(0,numPointsInTraj-start+0.01,0.2)
        extrapolatedPointsDebug2 = np.array(interpolate.splev(tDebug, spline0,ext=0))
        axes.plot(*extrapolatedPointsDebug2,'-o',label = f'order: {k}, smoothing: {alpha:0.1f}*sMax ({sMod2:0.1f})',ms= 3)
        if fixSharp == 1:
            axes.plot(*np.array([p0,p1]).T,'--',label = f'3 step linear fit (*)')
            axes.plot([p0[0]+dv[0],p0[0],p0[0]+dv0[0]],[p0[1]+dv[1],p0[1],p0[1]+dv0[1]],'-',label = f'angleDeg between (*) and exterp:{angleDeg:0.1f}',lw=3)
        axes.plot(*splitSubsetComponents,'x',label = 'Original pts (subset)',ms= 13);print(f'splitSubsetComponents:{splitSubsetComponents}')
        axes.legend(prop={'size': 6})
        axes.set_title(pltString+"extrapolate() debug")
        #plt.show()
    return returnVec if numDims > 1 else returnVec[1]

def distStatPredictionVect2(trajectory, sigmasDeltas = [],sigmasDeltasHist = [], numdeltas = 5, maxInterpSteps = 3, maxInterpOrder = 2, debug = 0, savePath = r'./', predictvec_old = [], bubID = 1, timestep = 0, zerothDisp = [0,0],ss = .5, fixSharp = 1):
    #global showDistDecay
    returnVec = []
    #debug = 1 if timestep >= 0 and bubID == 5 else 0
    if debug == 1: _, axes = plt.subplots(1,2 , figsize=( 13,5), sharex=False, sharey=False) # a = x if else y-> does not work for some reason
    else: _, axes = 1,[1]
    returnVec = extrapolate(trajectory, maxInterpSteps = maxInterpSteps, maxInterpOrder = 1, smoothingScale = ss, zerothDisp = zerothDisp, fixSharp = fixSharp, angleLimit = 10, debug = debug,axes = axes[0], pltString = f'bubID:{bubID},timestep:{timestep}|')
    trajectory = trajectory[-1*maxInterpSteps:] #may redo to -min something
    if debug == 1:
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
        if len(trajectory)>1: axes[0].plot([trajectory[-1][0],predictvec_old[0]],[trajectory[-1][1],predictvec_old[1]], '--o', label = 'pC-C dist')
        if len(sigmasDeltas)>0:
            circleMean = Circle(tuple(predictvec_old), sigmasDeltas[0] , alpha=0.1) 
            circleNStd = Circle(tuple(predictvec_old), sigmasDeltas[1]*numdeltas , alpha=0.1,fill=False) 
            axes[0].add_patch(circleMean)
            axes[0].autoscale_view()
            axes[0].set_autoscale_on(False)
            axes[0].add_patch(circleNStd)
            axes[0].text(*predictvec_old, s = f'm: {sigmasDeltas[0]:0.2f}, s:{sigmasDeltas[1]:0.1f}')
            axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, fancybox=True, shadow=True)
            axes[0].set_aspect('equal')
        if len(sigmasDeltasHist)>0:
            sigmasDeltasHistNP = np.array([v[1:] for v in sigmasDeltasHist.values()])
            timeSteps = list(sigmasDeltasHist.keys())
            axes[1].plot(timeSteps,sigmasDeltasHistNP[:,0], '--o', label = 'vals')
            axes[1].plot(timeSteps,sigmasDeltasHistNP[:,1], '--o', label = 'running mean')
            axes[1].plot(timeSteps,sigmasDeltasHistNP[:,2], '--o', label = 'running stdev')
            axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            axes[1].grid()
        plt.tight_layout()
        #plt.show()
        filename = os.path.join(savePath, f"ID_{str(bubID).zfill(3)}_t_{str(timestep).zfill(3)}.png")
        plt.savefig(filename)
    #debug = 0
    #numPointsInTraj = len(trajectory)
    #numStepsInTraj = numPointsInTraj - 1 
    #start = 0 if numStepsInTraj < maxInterpSteps else numStepsInTraj-maxInterpSteps
    #x = np.array([a[0] for a in trajectory[start:]])
    #y = np.array([a[1] for a in trajectory[start:]])
    #t = np.arange(0,len(x),1)
    #t1 = np.arange(0,len(x)+1,1)
    
            
    #if numStepsInTraj == 0:
    #    returnVec = np.array([x[0]+zerothDisp[0],y[0]+zerothDisp[1]])
    #    if debug == 1:
    #        axes[0].plot([x[0],x[0]+zerothDisp[0]], [y[0],y[0]+zerothDisp[1]], '--o', label = 'forecast', ms= 3)
    #        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
                
    #if numStepsInTraj > 0:
    #    k = min(numStepsInTraj,maxInterpOrder)
    #    # reduce iterp order if prediction does sharp turns
    #    #(numPointsInTraj-np.sqrt(2*numPointsInTraj),numPointsInTraj+np.sqrt(2*numPointsInTraj))
    #    for kMod in range(k,0,-1): # smoothing did not help on 3 point data, k=2 extepolation gives a 120 degree turn on one occasion
    #   # maxSmoothing = numPointsInTraj+np.sqrt(2*numPointsInTraj)
    #    #for sMod in np.arange(0,maxSmoothing+0.0001,maxSmoothing/5):
    #        sMod = 0
    #        spline, _ = interpolate.splprep([x, y], u=t, s=sMod,k=kMod)
    #        new_points = np.array(interpolate.splev(t1, spline,ext=0))
    #        #v1,v2 = zip(new_points[0][-2:],new_points[1][-2:])
    #        vs = new_points[:,-2:] - new_points[:,-3:-1] #deltas are displ components
    #        angleDeg = (lambda x,y: np.degrees(np.arccos(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1.0, 1.0))))(*np.transpose(vs))
    #        print(f' interpOrder = {k},smoothing = {sMod}, angle = {angleDeg}') if debug == 1 else 0
            
    #        if angleDeg<= 30: break
    #        else:
    #            if debug == 1: axes[0].plot(new_points[0],new_points[1], '-o', label = f'forecast (failed): s:{sMod:0.1f},k:{kMod:0.1f}', ms= 2,linewidth = 1)
    #    if debug == 1:
    #        axes[0].plot(new_points[0,-2:],new_points[1,-2:], '--o', label = 'forecast', ms= 3)
    #        axes[0].plot([x[-2],predictvec_old[0]],[y[-2],predictvec_old[1]], '--o', label = 'prev forecast')
    #        if len(sigmasDeltas)>0:
    #            circleMean = Circle(tuple(predictvec_old), sigmasDeltas[0] , alpha=0.1) 
    #            circleNStd = Circle(tuple(predictvec_old), sigmasDeltas[1]*numdeltas , alpha=0.1,fill=False) 
    #            axes[0].add_patch(circleMean)
    #            axes[0].add_patch(circleNStd)
    #            axes[0].text(*predictvec_old, s = f'm: {sigmasDeltas[0]:0.2f}, s:{sigmasDeltas[1]:0.1f}')
    #            axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
    #    returnVec = np.array([new_points[0,-1],new_points[1,-1]])

    #if debug == 1:
    #    if len(sigmasDeltasHist)>0:
    #        sigmasDeltasHistNP = np.array([v[1:] for v in sigmasDeltasHist.values()])
    #        timeSteps = list(sigmasDeltasHist.keys())
    #        axes[1].plot(timeSteps,sigmasDeltasHistNP[:,0], '--o', label = 'vals')
    #        axes[1].plot(timeSteps,sigmasDeltasHistNP[:,1], '--o', label = 'running mean')
    #        axes[1].plot(timeSteps,sigmasDeltasHistNP[:,2], '--o', label = 'running stdev')
    #        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
    #        axes[1].grid()
    #    #plt.legend(loc=(1.1, 0.5))
        
    #    filename = os.path.join(savePath, f"ID_{str(bubID).zfill(3)}_t_{str(timestep).zfill(3)}.png")
    #    plt.savefig(filename)
    #    #plt.suptitle( f"ID_{str(bubID).zfill(3)}_t_{str(timestep).zfill(3)}")
    #    #plt.show()
    return returnVec

def overlappingRotatedRectangles(group1Params, group2Params, returnType = 0, typeAreaThreshold = 0):
    #group1IDs, group2IDs = list(group1Params.keys()),list(group2Params.keys())
    #allCombs = list(itertools.product(group1Params, group1Params))#;print(f'allCombs,{allCombs}')
    intersectionType    = {}
    intersectingCombs   = []
    if group1Params == group2Params:
        allCombs = np.unique(np.sort(np.array(list(itertools.permutations(group1Params, 2)))), axis = 0)
    elif len(group2Params) ==0:
        return intersectingCombs
    else:
        allCombs = list(itertools.product(group1Params, group2Params))#;print(f'allCombs,{allCombs}')
    for (keyOld,keyNew) in allCombs:
        x1,y1,w1,h1 = group2Params[keyNew]#;print(allCombs)
        rotatedRectangle_new = ((x1+w1/2, y1+h1/2), (w1, h1), 0)
        x2,y2,w2,h2 = group1Params[keyOld]
        rotatedRectangle_old = ((x2+w2/2, y2+h2/2), (w2, h2), 0)
        interType, points  = cv2.rotatedRectangleIntersection(rotatedRectangle_new, rotatedRectangle_old)
        if interType > 0:
            if returnType == 1:
                if typeAreaThreshold > 0 and interType == 1:                                   # consider contours with partial overlap
                    intersection_area = cv2.contourArea( np.array(points, dtype=np.int32))     # group1 and group2 intersection area
                    relArea = intersection_area/(h2*w2)                           # partial intersection area vs group1 area
                    if relArea < typeAreaThreshold: continue                                      # if group1 is weakly intersection group2 drop him
                intersectionType[keyOld] = interType
                intersectingCombs.append([keyOld,keyNew]) # rough neighbors combinations
            else: intersectingCombs.append([keyOld,keyNew])
            
    if returnType == 0: return intersectingCombs
    else: return intersectingCombs, intersectionType

def filterPermutations(intersectingCombs,rectParams,rectParams_Old,areas_Old,areas,centroids,centroids_Old,relArea,relDist,maxAngle,maxArea,globalCounter):
    intersectingCombs_stage2 = []
    for (keyOld,keyNew) in intersectingCombs:
        #if keyOld in frozenIDs_old: relArea2 = 1; relDist2 = np.linalg.norm(rectParams_Old[keyOld][-2:]) # in case of FB split, weaken contrains 
        #else: relArea2 = relArea; relDist2 = relDist
        relArea2 = relArea; relDist2 = relDist
        areaOld, areaNew = areas_Old[keyOld], areas[keyNew]
        relativeAreaChange = abs(areaOld-areaNew)/areaOld
        centroidOld,centroidNew = centroids_Old[keyOld], centroids[keyNew]
        dist = np.linalg.norm(np.diff([centroidOld,centroidNew],axis=0),axis=1)[0]
        [anglePass, angleDiff] = boundRectAngle(rectParams[keyNew],rectParams_Old[keyOld], maxAngle, debug  = 0, info= '')
        if  relativeAreaChange < 2*relArea2 and dist < 1 and anglePass == True and areaNew < maxArea: # due to hulls area might change alot, but if displacement is really low, relax relArea crit 11/02/23
              intersectingCombs_stage2.append([keyOld,keyNew,relativeAreaChange,dist]) # these objects did not move or change area in-between last time steps
        elif  relativeAreaChange < relArea2 and dist < relDist2 and anglePass == True and areaNew < maxArea:
              intersectingCombs_stage2.append([keyOld,keyNew,relativeAreaChange,dist])
    return intersectingCombs_stage2

def detectStuckBubs(rectParams_Old,rectParams,areas_Old,areas,centroids_Old,centroids,contours,globalCounter,frozenLocal,relArea,relDist,maxAngle,maxArea):
    # analyze current and previous frame: old controur-new contour, old cluster-> new contour
    # search rough neighbors combination by detecting overlapping bounding rectangles

    intersectingCombs = overlappingRotatedRectangles(rectParams_Old,rectParams)

    #print(f'intersectingCombs,{intersectingCombs}')
    # fiter these combinations based on centroid dist and relative area change
    # !!!! for some reason '6' = 32 has different centroid coordinates (here by 1 pixel) !!! should be the same 'cause same object !!!!
    
    intersectingCombs_stage2 = filterPermutations(intersectingCombs,rectParams,rectParams_Old,areas_Old,areas,centroids,centroids_Old,relArea,relDist,maxAngle,maxArea,globalCounter)
    #print(f'intersectingCombs_stage2,{intersectingCombs_stage2}')
    # if constraints are weak single new contour will be related to multiple old cntrs/clusters.
    # find these duplicate combinations. must be very rare case. can check it by setting high rel area and dist
    # ====== MERGE ======= 
    # must check this out more thorougly 12/02/23. had problem with split, array of mixed types, argwhere did not work, produced []
    keysNew = [a[1] for a in intersectingCombs_stage2]
    keysNewVals = np.array([a[0] for a in intersectingCombs_stage2],dtype=object)
    values, counts = np.unique(keysNew, return_counts=True)
    dupMerge = [ a for a,b in zip(values, counts) if b>1]
    dupWhereIndicies = {a:np.argwhere(keysNew == a).reshape(-1).tolist() for a in dupMerge}
    dupVals = {ID:keysNewVals[lst] for ID,lst in dupWhereIndicies.items()}
    print(f'dupVals:{dupVals}')
    # perform two-criteria  (dist/area) minimization task
    dupSubset = []
    onlyOldIDs = [aa for aa in rectParams_Old if type(aa) == str ]
    for ID, subIDs in dupVals.items():
        # in cases when duplicates consist of global and local ID of same object, give prio to global ID. havent seen yet otherwise cases- else permutations.
        simpleCopy = False
        
        for aa in subIDs:
            if len(subIDs) == 2 and aa in onlyOldIDs:
                tempCombs = [bb for bb in intersectingCombs_stage2 if bb[0] == aa]
                intersectingCombs_stage2 = [a for a in intersectingCombs_stage2 if a[1] not in [ID]]    # drop duplicates altogether
                intersectingCombs_stage2 = intersectingCombs_stage2 + tempCombs                                     # add solution
                simpleCopy = True

        if simpleCopy == False:
            assert 2 == 3, "dupVals consists of more than 2 elements- global and local IDs, but some other object "
            permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(contours, rectParams, rectParams_Old[ID], subIDs, centroids_Old, areas_Old,
                                         centroids[ID], relDist, areas[ID], relAreaCheck = 0.7, doHull = 1, debug = 0)
            print(f'pIDs (merge): {permIDsol2}; pDist: {permDist2:0.1f}; pRelA: {permRelArea2:0.2f}')
            assert len(permIDsol2) < 2, f"detectStuckBubs-> centroidAreaSumPermutations  resulted in strange solution for new ID:{ID} - permIDsol2"
            dupSubset.append([permIDsol2[0],ID,permDist2,permRelArea2]) # 
            
            intersectingCombs_stage2 = [a for a in intersectingCombs_stage2 if a[1] not in dupWhereIndicies]    # drop duplicates altogether
            intersectingCombs_stage2 = intersectingCombs_stage2 + dupSubset                                     # add solution

    # ========================== Detect Splits ===========================
    # non overlapping old local and old global (else/clusters) intersections with new local contours <-> intersectingCombs
    # take those old that intersect multiple new. try to find match of permutations using double criteria weights
    keysOld = [a[0] for a in intersectingCombs]
    keysNewVals = np.array([a[1] for a in intersectingCombs],dtype=object)
    values = list(set(keysOld))
    counts = [keysOld.count(a) for a in values]
    #values, counts = np.unique(keysOld, return_counts=True) # unique does not want to deal with mixed type
    dupSplit = [ a for a,b in zip(values, counts) if b>1]
    keysOld = np.array(keysOld,dtype = object)
    dupWhereIndicies = {a:np.argwhere(keysOld == a).reshape(-1).tolist() for a in dupSplit}
    dupVals = {ID:keysNewVals[lst] for ID,lst in dupWhereIndicies.items()}
    print(f'dupVals Split:{dupVals}')
    dupSubset = []
    for ID, subIDs in dupVals.items():
        permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations(contours, rectParams, rectParams_Old[ID], subIDs, centroids, areas,
                                     centroids_Old[ID], relDist, areas_Old[ID], relAreaCheck = 2, doHull = 1, debug = 0) # !! new and _old swapped palces , relArea  should be around 1 !!
        cntrs = contours[permIDsol2]
        newArea = getContourHullArea(cntrs)
        if  0 < permRelArea2 <  relArea and 0 < permDist2 < relDist and 0 < newArea <  maxArea:
            dupSubset.append([ID, permIDsol2, permDist2,permRelArea2])
            print(f'pIDs (split): {permIDsol2}; pDist: {permDist2:0.1f}; pRelA: {permRelArea2:0.2f}')
        
    asd = [ e[0] for e in dupSubset]  
    if len(dupSubset) > 0:
        intersectingCombs_stage2 = [a for a in intersectingCombs_stage2  if a[0] not in asd]#not in dupWhereIndicies
    intersectingCombs_stage2 = [[a,[b],c,d] for a,b,c,d in intersectingCombs_stage2] # newIDs as a list, in case there are multiple IDs, which happens.
    intersectingCombs_stage2 = intersectingCombs_stage2 + dupSubset        
    # compare these two-frame combinations with global list of stuck bubbles
    toList = lambda x: [x] if type(x) != list else x
    returnInfo = []
    for keyOld, keyNew, relArea, dist  in intersectingCombs_stage2:
        returnInfo.append([keyOld,keyNew, dist, relArea, centroids_Old[keyOld]])
        frozenLocal.append(keyNew)
    returnInfo = np.array(returnInfo, dtype=object) # kind weird that second element goes from [54] to list([54]), but looks like it makes not diff
    returnNewIDs = returnInfo[:,1] if len(returnInfo)>0 else np.array([])
    return returnNewIDs, returnInfo

def stuckBubHelp(fbStoreCulprits,searchCentroid):

    dists = {centroid:np.linalg.norm(np.diff([centroid,searchCentroid],axis=0),axis=1)[0] for centroid in fbStoreCulprits}
    minKey = min(dists, key=dists.get)
    return minKey

def multiContourBoundingRect(contours):
    tempCntrJoin = np.vstack(contours)
    xt,yt,wt,ht = cv2.boundingRect(tempCntrJoin)
    return xt,yt,wt,ht

def getMasksParams(contours,IDs,err,orig):
    contourSubset = [contours[k] for k in IDs]
    contourStack = np.vstack(contourSubset)   
    x,y,w,h = cv2.boundingRect(contourStack)#;print([x,y,w,h])
    baseSubMask,baseSubImage = err.copy()[y:y+h, x:x+w], orig.copy()[y:y+h, x:x+w]
                    
    subSubMask = np.zeros((h,w),np.uint8)
    [cv2.drawContours( subSubMask, contours, ID, 255, -1, offset = (-x,-y)) for ID in IDs]
    baseSubMask[subSubMask == 0] = 0
                    
    baseSubImage[subSubMask == 0] = 0
                    
    tempC = getCentroidPosContours(contourSubset)[0]
    return baseSubMask, baseSubImage, [x,y,w,h] , tempC

# listFormat(dataList = [3, 18, 10.63014581273465, 0.0763888888888889],
#            formatList = ["{:.0f}", "{:0.0f}", "{:.2f}","{:0.2f}"], outType = float)
def listFormat(dataList, formatList, outType):
    # if outType = float, then map ints to ints, not float. intFloat = int->int/float->float else float->str
    intFloat = lambda x: int(x) if x.isdigit() else outType(x) if outType != str else lambda x: x
    return [intFloat(frmt.format(nbr)) for nbr,frmt in zip(dataList, formatList)]

def updateStat(count, mean, std, newValue):
    M2 = count * std**2
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    if count < 2:
        return float("nan")
    else:
        (mean, variance) = (mean, M2 / count)
        return (mean, np.sqrt(variance))





#typeTemp                        = typeRing if any(item in newFoundBubsRings for item in new) else typeElse
#print(f'typeTemp {typeTemp}: {typeStrFromTypeID[typeTemp]}')
#centroidsTemp                   = {ID: g_Centroids[ID][globalCounter-1][0] for ID in old} # ID is inteherted to bubble closer to free surface (minimal x coord)
#selectID                        = min(centroidsTemp, key=centroidsTemp.get)

#if typeTemp == typeRing:
#     dataSets = [l_RBub_r_masks,l_RBub_images,l_RBub_old_new_IDs,l_RBub_rect_parms,l_RBub_centroids,l_RBub_areas_hull]
#else: 
#    dataSets = [l_DBub_masks,l_DBub_images,l_DBub_old_new_IDs,l_DBub_rect_parms,l_DBub_centroids,l_DBub_areas_hull]
 
def alphashapeHullModded(contours, contourIDs, param, debug):
    # finds a concave hull of cloud of points given a control parameter 'param'.
    # when applied to contours, might perform incorrectly, since there are no points inside
    # solution- add extra points inside contours.
    points_2d   = np.vstack(contours[contourIDs]).reshape(-1,2)
    x,y,w,h     = cv2.boundingRect(points_2d)                   # grab a subImage to reduce computational resources

    fillContour = np.zeros((h,w),np.uint8)                      # mask of contour/-s
    [cv2.drawContours( fillContour,   [contours[i]-[x,y]], -1, 255, -1) for i in contourIDs]
    
    fillPoints  = np.zeros((h,w),np.uint8)                      # for extra points
    fillPoints[0:h:15,0:w:15] = 255                             # grid of evenly spaced points
    
    fillAndOp   = cv2.bitwise_and(fillContour,fillPoints)               # only points left are inside contour/-s
    wherePoints = np.flip(np.array(np.where(fillAndOp == 255),int)).T   # local coordinates of inside points. some mumbo jambo with flipped x and y
    wherePoints += [x,y]                                                # offset to global space
    
    allPonts = np.vstack((points_2d,wherePoints))
    alpha_shape = alphashape.alphashape(allPonts, param)
    if alpha_shape.geom_type == 'Polygon':
            xx,yy                   = alpha_shape.exterior.coords.xy
            hull                    = np.array(list(zip(xx,yy)),np.int32).reshape((-1,1,2))
            #centroid, area          =  getCentroidPosContours(bodyCntrs = [hull])
    else:
        print(f'alphashapeHullModded() subIDs: {contourIDs}-> OG geom type{alpha_shape.geom_type}')
        for prm in np.append(np.arange(param-0.01,0,-0.01),0):
            print(f'alphashapeHullModded() subIDs: {contourIDs}-> reducing alpha parameter from {param} to {np.around(prm,2)}, to resolve single polygon')
            alpha_shape2 = alphashape.alphashape(allPonts, prm)
            print(f'alphashapeHullModded() subIDs: {contourIDs}-> new geom type{alpha_shape2.geom_type}')
            if alpha_shape2.geom_type == 'Polygon':
                xx,yy                   = alpha_shape2.exterior.coords.xy
                hull                    = np.array(list(zip(xx,yy)),np.int32).reshape((-1,1,2))
                print(f'alphashapeHullModded() subIDs: {contourIDs}-> resoled single poly with parameter: {np.around(prm,2)}')
                break
            
        #cv2.imshow('1',fillPoints)
        #cv2.imshow('2',fillContour)
        #cv2.imshow('3',fillAndOp)
    if debug == 1:
        cv2.imshow('1',fillPoints)
        cv2.imshow('2',fillContour)
        cv2.imshow('3',fillAndOp)
        imageDebug = np.zeros((h,w,3),np.uint8)
        [cv2.circle(imageDebug , c  - [x,y], 1, (0,255,0), -1) for c in allPonts]
        cv2.drawContours( imageDebug,   [hull- [x,y]], -1, (255,255,0), 1)
        cv2.imshow('4',imageDebug)

            
    return hull

def tempStore(contourIDs, contours, globalID, mask, image, dataSets):
    gatherPoints                    = np.vstack([contours[ID] for ID in contourIDs])#;print(distCntrSubset.shape)
    x,y,w,h                         = cv2.boundingRect(gatherPoints)#;print([x,y,w,h])
    #tempID                          = min(contourIDs)  #<<<<<<<<<<< maybe check if some of these contours are missing elsewhere
                
    baseSubMask,baseSubImage        = mask.copy()[y:y+h, x:x+w], image.copy()[y:y+h, x:x+w]
    subSubMask                      = np.zeros((h,w),np.uint8)
    [cv2.drawContours( subSubMask, contours, ID, 255, -1, offset = (-x,-y)) for ID in contourIDs]
    
    baseSubMask[subSubMask == 0]    = 0
                
    baseSubImage[subSubMask == 0]   = 0

    hullArea = getCentroidPosContours(bodyCntrs = [contours[k] for k in contourIDs], hullArea = 1)[1]
    centroid = getCentroidPos(inp = baseSubMask, offset = (x,y), mode=0, mask=[])

    for storage, data in zip(dataSets,[baseSubMask , baseSubImage, contourIDs, ([x,y,w,h]), centroid, hullArea]):
        storage[globalID] = data

def tempStore2(contourIDs, contours, globalID, mask, image, dataSets,concave = 0,param = 0.05):
    # smallest x centroid survives. ID:Cx -> smallest Cx ID
                    
    gatherPoints                    = np.vstack([contours[k] for k in contourIDs])
    x,y,w,h                         = cv2.boundingRect(gatherPoints)#;print([x,y,w,h])
    baseSubMask,baseSubImage        = mask.copy()[y:y+h, x:x+w], image.copy()[y:y+h, x:x+w]
                    
    subSubMask                      = np.zeros((h,w),np.uint8)
    [cv2.drawContours( subSubMask, contours, ID, 255, -1, offset = (-x,-y)) for ID in contourIDs]
    baseSubMask[subSubMask == 0]    = 0
                    
    baseSubImage[subSubMask == 0]   = 0
                    
    #centroid = getCentroidPos(inp = baseSubMask, offset = (x,y), mode=0, mask=[])

    if (type(concave) == int and concave == 0):
        hull                            = cv2.convexHull(np.vstack(contours[contourIDs]))
        hullArea                        = getCentroidPosContours(bodyCntrs = [hull])[1]
        #hullArea = getCentroidPosContours(bodyCntrs = [contours[k] for k in contourIDs], hullArea = 1)[1]
    elif (type(concave) == int and concave == 1):
        hull = alphashapeHullModded(contours, contourIDs, param, 0)

    else:
        hull = concave
    centroid, hullArea    = getCentroidPosContours(bodyCntrs = [hull])
    for storage, data in zip(dataSets,[baseSubMask , baseSubImage, contourIDs, ([x,y,w,h]), centroid, int(hullArea), hull]):
        storage[globalID] = data

    #l_bubble_type[selectID]                 = typeTemp   #<<<< inspect here for duplicates
    #g_bubble_type[selectID][globalCounter]  = typeTemp
def mergeCrit(contourIDs, contours, previousInfo, pCentr = None, alphaParam = 0.05, debug = 0,debug2 = 0, gc = 0, id = 0):
    #baseContour, _, _       = alphashapeHullCentroidArea(contours, contourIDs, alphaParam)  
    baseContour             = alphashapeHullModded(contours, contourIDs, alphaParam, debug2) # find concave hull for a cluster using alphashape lib.
    hullCoordinateIndices   = cv2.convexHull(baseContour, returnPoints = False)             # indicies of original contour points that form a convex hull
    hullDefects             = cv2.convexityDefects(baseContour, hullCoordinateIndices)      # find concave defects of convex hull
    hullDefects             = hullDefects if hullDefects is not None else np.array([])
    refDistance             = previousInfo[0]                                               # limits length of defects
    refPlaneTangent         = previousInfo[1]                                               # tangent vector to centroid-centroid v. merge interface ~ along it
    refAngleThreshold       = previousInfo[2]                                               # limits angle of defects
    refMergePoint           = previousInfo[3]
    baseDefects             = previousInfo[4]
    baseDefectsOut          = None
    calculateAngle          = lambda vector: np.arccos( min(np.dot(refPlaneTangent, vector)/(np.linalg.norm(refPlaneTangent)*np.linalg.norm(refPlaneTangent)),1))
                                                                                            # chech angle between reference tangent and a vector.>> problems with 0 deg angle...
    defectsIndicies         = []                                                            # large defects will be stored here
    defectsDirection        = []                                                            # direction from 'cave' furthest point normal to contour.
    defectsLength           = []
    defectsFarpoints        = []
    defectsDiscard          = []
    farpointMean            = np.empty((0,2),int)
    inPlaneDefectsPresent   = True
    if debug == 1:
        x,y,w,h             = cv2.boundingRect(baseContour)
        imageDebug          = np.zeros((h,w,3),np.uint8)
        cv2.drawContours( imageDebug, [baseContour - [x,y]], -1, (255,0,0), 2)
        [cv2.drawContours(imageDebug, [contours[i]-[x,y]], -1, (40,40,40), -1) for i in contourIDs]
    for i in range(hullDefects.shape[0]):
        s,e,f,d = hullDefects[i,0]                                              # s,e,f are indicies of original concave hull, d is weird distance normal to contour
        if d >= (refDistance)*256:                                              # filter out defects smaller than refDistance. times 256 - it just works!
            start,end   = [np.array(tuple(baseContour[i][0])) for i in [s,e]]   # contour hull edge adjacent to a defect
            tangent     = (lambda x: x/np.linalg.norm(x))(end-start)            # make it unit length
            normal      = np.matmul(np.array([[0, -1],[1, 0]]), tangent)        # rotate tangent by 90 c-clockwise do find a normal. opencv travels contours clockwise!?
            defectsIndicies.append([s,e])                                       # store defects indicies where it begins to cave and ends.
            defectsDirection.append(normal)
            defectsLength.append(d)
            defectsFarpoints.append(f)
            if debug == 1:
                farPointStart   = np.array(tuple(baseContour[f][0])) - [x,y]
                farPointEnd     = np.array(farPointStart+normal*d/256, int)
                cv2.line(imageDebug, start - [x,y], end - [x,y], [0,0,255], 2)
                cv2.line(imageDebug, farPointStart, farPointEnd, [255,125,0], 1)
                cv2.circle(imageDebug,farPointStart, 5, [0,0,255], -1)
    defectsIndicies2     = np.array(defectsIndicies,int)
    defectsLength2       = np.array(defectsLength,int)
    defectsDirection2    = np.array(defectsDirection)
    defectsFarpoints2    = np.array(defectsFarpoints,int)
    defectsFarpoints2p   = np.array([baseContour[ID] for ID in defectsFarpoints2],int).reshape(-1,2)
    if refMergePoint is not None: 
        
                                                      # ~ point where interfaces meet. limits defects position.
        ccDir                   = np.matmul(np.array([[0, -1],[1, 0]]), refPlaneTangent)                             # unit vector in c-c direction
        pts                 = defectsFarpoints2p - refMergePoint                                                     # move coord system to one centered at refMergePoint
        projections         = np.abs(np.einsum('ij,j->i', pts, ccDir).astype(int))                                   # get dist from tangent plane to defect point
        defectsDiscardL2     = np.array(np.where(projections>2*refDistance)).flatten().astype(int)                   # which are to far?
        defectsAngles2      = np.array([int(min(np.pi-calculateAngle(v),calculateAngle(v))*180/np.pi) for v in defectsDirection2])  # smallest angle to tangent plane
        defectsDiscardA2    = np.array(np.where(np.abs(defectsAngles2)>40)).flatten()                                # which are too big?
        defectsDiscard2     = np.union1d(defectsDiscardL2, defectsDiscardA2)                                         # combine drop lists.
        baseDefectsOut      = [stats for i, stats in enumerate(zip(defectsFarpoints2p,defectsDirection2,defectsLength2)) if i not in defectsDiscard2]
    elif (baseDefects is not None and len(baseDefects) >0 ) and len(defectsIndicies)>0:
        currentCentroid = np.array(getCentroidPosContours(bodyCntrs = [baseContour])[0],int)
        displacement    = currentCentroid - pCentr                                                                   # translate base defects by that amount
        baseDefects2     = [[a+displacement,b,int(c/256)] for a,b,c in baseDefects]
        #defectsIndicies2     = np.array(defectsIndicies,int)
        #defectsLength2       = np.array(defectsLength,int)
        #defectsDirection2    = np.array(defectsDirection)
        #defectsFarpoints2    = np.array(defectsFarpoints,int)
        #defectsFarpoints2p   = np.array([baseContour[ID] for ID in defectsFarpoints2],int).reshape(-1,2)
        oldDefNewDefRelation = []
        for i,[defPointPred, defDirOld, defDistOld] in enumerate(baseDefects2):
            defOldNewDist   = [np.linalg.norm(np.array(defCurrent) - np.array(defPointPred)) for defCurrent in defectsFarpoints2p]
            defOldNewAng    = [int(min(
                                np.pi-np.arccos(np.dot(defDirOld,defDir)), np.arccos(np.dot(defDirOld,defDir))
                                        )*180/np.pi) for defDir in defectsDirection2]
            tt = np.array([defOldNewDist,defOldNewAng],int).T                                                       # [[dist_old_1,ang_old_1],[dist_old_2,ang_old_2]]
            tt2 = np.argmin(tt, axis=0)                                                                             # [whereMin([dist_old_1,dist_old_2]),whereMin([ang_old_1,ang_old_2])]  
            uniqtt = np.unique(tt2)                                                                                 # say both d1,a1 < d2,a2, so returns [0,0]-> [0]
            if len(uniqtt) == 1:                                                                                    # if only 1 entry then defect wins in both comparisons
                oldDefNewDefRelation.append([i,uniqtt[0]])                                                          # predicted /w index i matches current defect /w idx 0 -> [i,j]
        passNewDef1 = [b for _,b in oldDefNewDefRelation]
        passNewDef2 = [i for i in passNewDef1 if int(defectsLength2[i]/256) > 0.5*refDistance]
        twoAngles = {}                                                                                              # added removal criterium for wide defect.
        for i in passNewDef2:                                                                                       # wide defects may persist for too long
            twoAngles[i] = np.mean(opossiteAngles(baseContour[defectsFarpoints2[i]].reshape(1,2), baseContour[defectsIndicies2[i]].reshape(2,2)))
        passNewDef3 = [i for i in passNewDef2 if twoAngles[i]>25]
        baseDefectsOut = [stats for i, stats in enumerate(zip(defectsFarpoints2p,defectsDirection2,defectsLength2)) if i in passNewDef3]
        defectsDiscard = [i for i in range(len(defectsFarpoints2p)) if i not in passNewDef2]
    defectsIndicies = defectsIndicies2
    defectsLength = defectsLength2
    defectsDirection    = [vector*np.sign(np.dot(refPlaneTangent,vector)) for vector in defectsDirection]
    defectsDirection    = np.array(defectsDirection)
    defectsAngles       = np.array([int(calculateAngle(v)*180/np.pi) for v in defectsDirection]) 
    defectsFarpoints    = np.array(defectsFarpoints,int)
    defectsDiscard      = np.array(defectsDiscard)
    #defectsIndicies     = np.array(defectsIndicies,int)                         # start end of defect edge                                
    #defectsLength       = np.array(defectsLength,int)
    #defectsDirection    = [vector*np.sign(np.dot(refPlaneTangent,vector)) for vector in defectsDirection]       # invert defectDirections if projection to refVector is negative.
    #defectsDirection    = np.array(defectsDirection)
    #defectsAngles       = np.array([int(calculateAngle(v)*180/np.pi) for v in defectsDirection])                # find defect angle to ref. min case where |angle| =< 90  
    #defectsDiscard      = np.array(np.where(np.abs(defectsAngles)>refAngleThreshold)).flatten()                 # which angle is higher than threshold? not sure if i need abs
    #defectsFarpoints    = np.array(defectsFarpoints,int)                 
    #if refMergePoint.shape[0]>0:
    #    pts                 = np.array([baseContour[i][0] for i in defectsFarpoints]).reshape(-1,2) - refMergePoint # move global point coordinates into local, placed into midline center
    #    projections         = np.abs(np.einsum('ij,j->i', pts, ccDir).astype(int))
    #    defectsDiscardL     = np.array(np.where(projections>2*refDistance)).flatten() 
    #    defectsDiscard      = np.union1d(defectsDiscardA, defectsDiscardL)                                          # which failed angle or distance check
    #else:
    #    defectsDiscard = defectsDiscardA
                                                                                                        
    if len(defectsDiscard)>0:                                                                           # usually discard defect start-end IDs do not intersect ID = 0
        discardIndices              = [defectsIndicies[i] for i in defectsDiscard]                      # # array([ [60,  75], [140, 155]])
        if discardIndices[0][0] < discardIndices[0][1]:                                                 
            removeTheseIntevals     = [[0,0]] + discardIndices + [[-1,-1]]                              # array([ [0,  0], [60,  75], [140, 155], [-1, -1]])
        else:                                                                                           # ------------- but 
            if len(discardIndices) > 1:
                removeTheseIntevals = np.concatenate((                                                  # array([[160,   3], [60,  75], [140, 155]]) might happen (first elem)
                                            [   [0,discardIndices[0][1]]   ],                           # in this case contour will be reconstruceted doint ~ two loops 
                                                defectsIndicies[1:]          ,                          # have to split first interval into two
                                            [   [discardIndices[0][0],-1]  ]                            # array([[  0,   3], [ 60,  75], [140, 155], [160,  -1]])
                                        )                                    , axis=0)                      
            else:
                removeTheseIntevals = np.concatenate((                                                      # array([[160,   3], [60,  75], [140, 155]]) might happen (first elem)
                                            [   [0,discardIndices[0][1]]   ],                              # in this case contour will be reconstruceted doint ~ two loops 
                                                                          
                                            [   [discardIndices[0][0],-1]  ]                               # array([[  0,   3], [ 60,  75], [140, 155], [160,  -1]])
                                        )                                    , axis=0) 
                                                                                                        # combine contour slices -> last defect end: next defect start
        invertedIndicies        = [(removeTheseIntevals[i][1],removeTheseIntevals[i+1][0]) for i in range(len(removeTheseIntevals)-1)] # like  array([ [0,  60], [75,  140], [155, -1]])
        hullOutput              = np.array(sum([list(baseContour[start:end]) for start,end in invertedIndicies] ,[]))                  # or    array([ [3,  60], [75,  140], [155, 160]])
    else:
        hullOutput              = baseContour
                        
    numUsefulDefects    = len(defectsAngles) - len(defectsDiscard)                                      # these specify a plane (line) between merged bubbles.
    if numUsefulDefects >= 1:                                                                           # ive seen single deffect.
        defectsUseful       = [i for i in range(len(defectsAngles)) if i not in list(defectsDiscard)]   # do average defect direction weighted by defect length
        avgDirection        = np.average(defectsDirection[defectsUseful], weights = defectsLength[defectsUseful],axis = 0)
    else:                                                                                               # lack of defects in tangent plane might indicate end of merge.
        avgDirection = refPlaneTangent                                                                  # at least keep old ref plane.
        inPlaneDefectsPresent = False
    
    if numUsefulDefects > 1:
        farpointsUseful = np.array([baseContour[i][0] for i in defectsFarpoints[defectsUseful]])
        farpointMean    = np.average(farpointsUseful, weights = defectsLength[defectsUseful],axis = 0).astype(int)
        defectsLenSum   = np.sum(defectsLength[defectsUseful]/256)
        p1,p2           = [np.array(farpointMean + i*0.5*defectsLenSum*avgDirection,int) for i in [-1,1]]
    else: p1,p2 = [],[]
    
    if debug ==1:
        #cv2.circle(imageDebug,refMergePoint - [x,y], 3, [0,255,255], -1)
        
        if numUsefulDefects > 1:
            cv2.circle(imageDebug,farpointMean - [x,y], 5, [255,0,255], -1)
            cv2.line(imageDebug, p1 - [x,y], p2 - [x,y], [255,0,255], 1)
        cv2.drawContours( imageDebug,   [hullOutput- [x,y]], -1, (255,0,0), 2) 
        cv2.imshow(f'GC {gc}, ID {id}',imageDebug) 
    return hullOutput, (refDistance, avgDirection, refAngleThreshold, None, baseDefectsOut), (inPlaneDefectsPresent, farpointMean)
    #return hullOutput, (refDistance, avgDirection, refAngleThreshold), (inPlaneDefectsPresent, farpointMean)


colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
np.random.seed(2);np.random.shuffle(colorList);np.random.seed()

def cyclicColor(index):
    return colorList[index % len(colorList)].tolist()

def plot_histogram_multiple(data_sets,ax, IDs, colors):
    
    
    offsets0 = [0] + [max(data[1]) for data in data_sets]
    offsets = np.cumsum(offsets0)

    # Plot the bars for each data set
    for i, data in enumerate(data_sets):
        values, counts = data
        ax.bar(values, counts, width=1.3, bottom = offsets[i], align='center', label=f'ID: {IDs[i]}', color = colors[i])

    ax.set_xlabel('Values')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of Multiple Data Sets')
    ax.legend()
 

def mergeSplitDetect(contours,contourIDs, direction, position, gc = 0, id = 0, debug = 0):
    sideOneContourIDs, sideTwoContourIDs = [], []
    isThereASplit   = False
    pointsGather    = {}
    distsGather     = {}
    intervalsGather = {}
    saveHist        = {}
    points_all  = np.vstack(contours[contourIDs]).reshape(-1,2)
    x0,y0,w0,h0 = cv2.boundingRect(points_all)
    blank = np.zeros((h0,w0,3),np.uint8) 
    for ID in contourIDs:
        points_2d   = contours[ID].reshape(-1,2)
        x,y,w,h     = cv2.boundingRect(points_2d) 
        fillContour = np.zeros((h,w),np.uint8)                              # mask of contour/-s
        cv2.drawContours( fillContour,   [contours[ID]-[x,y]], -1, 255, -1)
        totalArea = np.sum(fillContour)/255                                 # total area of a contour
        numPoints = 36                                                      # specify how many points you want inside contour
        step = int(np.sqrt(totalArea/numPoints))                            # assuming isotropic point density, and square region, calculate linear point density
        fillPoints  = np.zeros((h,w),np.uint8)                              # for extra points
        fillPoints[::step,::step] = 255                                     # grid of evenly spaced points, with step dependant on contour area.
    
        fillAndOp   = cv2.bitwise_and(fillContour,fillPoints)               # only points left are inside contour/-s
        #cv2.imshow(f'ID:{ID}',fillAndOp)
        wherePoints = np.flip(np.array(np.where(fillAndOp == 255),int)).T   # local coordinates of inside points. some mumbo jambo with flipped x and y
        wherePoints += [x,y]                                                # grab a subImage to reduce computational resources
        pointsGather[ID] = wherePoints
        
    
    for ID in contourIDs:
        
        pts = pointsGather[ID] - position                                       # move global point coordinates into local, placed into midline center
        projections = np.einsum('ij,j->i', pts, direction).astype(int)          # take projections onto normal to midline basis vector, this shows how far and which side of midline point is in.

        unique, counts = np.unique(projections, return_counts = True)           # kind of histogram
        result = list(zip(unique, counts))                                      # to sort histogram based on distanc values
        result.sort(key=lambda x: x[0])                                         # so i can get f=f(x): [[xmin,f[xmin],...,[xmax,f[xmax]]]
        result = np.array(result,int)                                           # cast to numpy array so its easier and faster to work with
        saveHist[ID] = result if debug == 1 else []
        r,dr = findMajorInterval(result[:,0],result[:,1], meanVal=None, cover_area=0.9, debug=debug)
        #print(r,r+dr)
        intervalsGather[ID] = np.array([[np.min(result[:,0]),np.max(result[:,0])],[r, r +dr],[np.diff([np.min(result[:,0]),np.max(result[:,0])])[0],dr]], int)
        distsGather[ID]     = projections
        if debug == 1:
            ptsLoc = pointsGather[ID] - [x0,y0]
            blank[pointsGather[ID][:,1]-y0,pointsGather[ID][:,0]-x0] = cyclicColor(ID)
        
            [cv2.line(blank, ps, np.array(ps-dst*direction,int), cyclicColor(ID), 1) for ps,dst in zip(ptsLoc[::2],projections[::2])]#[0:-1:1]
            cv2.drawContours(blank, contours, ID, [0,0,0], 2, offset= (-x0,-y0))
            cv2.drawContours(blank, contours, ID, cyclicColor(ID), 1, offset= (-x0,-y0))
    #intervalWidths = {ID:np.diff(intervalsGather[ID][1]) for ID in contourIDs}
    sideID = {}
    if len(contourIDs)>1:                                                                                          # possible split implies existanc of one or contours.
        for ID in contourIDs:
            # determine if interval lives on one of the half-lines. split total interval into 2 parts at zero
            # if one of subintervals is empty- whole interval lives on "+" ir "-" half-plane
            # if one whole interval is relatively small, we dont bother. if one sub interval is small check how much smaller and discard it.
            weightedInterval    = intervalsGather[ID][1]
            width               = intervalsGather[ID][2,1]
            print(f'width:{width}')
            meanWidth           = np.average([intervalsGather[ids][2,1] for ids in contourIDs if ids != ID])    # mean of all except current ID
            print(f'meanWidth:{meanWidth}')
            overlapLeft         = interval_overlap(weightedInterval, [-max(w0, h0),0])                          # split big interval into two. split at zero
            overlapRight        = interval_overlap(weightedInterval, [0, max(w0,h0)])                           # length of test interval is set to be the largest relevant for this problem.
            overlapPartsLengths = np.array([np.diff(a)[0] for a in [overlapLeft,overlapRight] if len(a)>0])     # abs values of half interval lengths
            if any([len(interval)==0 for interval in [overlapLeft,overlapRight] ]):                             # if one of intervals is empty (no overlap with half line)
                oneSided = np.sign(*[np.sum(x) for x in [overlapLeft,overlapRight] if len(x)>0])                # sum(x) produces sign*intervalAra: [-a,0]-> -a, [0,a] -> a
            elif (width >= 0.2*meanWidth) and (np.min(overlapPartsLengths) <= 0.2*np.max(width)):               # if width is bugger than fraction of mean width, and smaller side is a small fraction of total width
                whereLargest    = np.argmax(overlapPartsLengths)                                                # where is largest by abs value
                oneSided        = np.sign(np.sum([overlapLeft,overlapRight][whereLargest]))                     # get a sign of its position
            else: oneSided = 0                                                                                  # 0 means not 1, nor -1. or abs(+/-1) > 0
            sideID[ID] = oneSided
            print(oneSided)
            print(f'ID:{ID}, left overlap : {overlapLeft}, right overlap: {overlapRight}, side identified: {oneSided}')
        sideIDs = np.array(list(sideID.values()))
        if np.count_nonzero(sideIDs == 0) == 0:                                                        # there are only contours that live on either side. no inbetween
            sideOneContourIDs = [ID for ID in contourIDs if sideID[ID] == -1]
            sideTwoContourIDs = [ID for ID in contourIDs if sideID[ID] ==  1]
            if len(sideOneContourIDs)>0 and len(sideTwoContourIDs)>0:
                blank2 = np.zeros((h0,w0),np.uint8)
                blank3 = np.zeros((h0,w0),np.uint8)
                hull1 = cv2.convexHull(np.vstack(contours[sideOneContourIDs]))
                hull2 = cv2.convexHull(np.vstack(contours[sideTwoContourIDs]))
                cv2.drawContours( blank2,   [hull1-[x0,y0]], -1, 255, -1)
                cv2.drawContours( blank3,   [hull2-[x0,y0]], -1, 255, -1)
                #area2 = int(np.sum(blank2/255))
                #area3 = int(np.sum(blank3/255))
                inter = cv2.bitwise_and(blank2, blank3)
            
                interArea = int(np.sum(inter/255))
                print(f'interArea: {interArea}')
                if interArea == 0:
                    dist = closes_point_contours(hull1,hull2)[1]
                    if dist > 5:
                        isThereASplit = True
                if debug == 1:
                    deb = np.zeros((h0,w0,3),np.uint8)
                    [cv2.drawContours( deb,   [contours[ID]-[x0,y0]], -1, [120,120,120], -1) for ID in contourIDs]
                    cv2.drawContours( deb,   [hull1-[x0,y0]], -1, [255,0,0], 2)
                    cv2.drawContours( deb,   [hull2-[x0,y0]], -1, [0,0,255], 2)
                    cv2.imshow(f'{gc,id}', deb)
                    cv2.imshow(f'inter {gc,id}', inter)
        
            
            #[cv2.drawContours( blank2,   [contours[ID]-[x0,y0]], -1, [120,120,120], -1) for ID in contourIDs]

            
    #resized = cv2.resize(blank, (w0*3, h0*3), interpolation = cv2.INTER_AREA)
    #cv2.imshow('asdas',resized)
    if debug == 1:
        resized = cv2.resize(blank, (w0*3, h0*3), interpolation = cv2.INTER_AREA)
        cv2.imshow(f'projections {gc,id}',resized)
        _, ax = plt.subplots()
        plot_histogram_multiple([saveHist[ID].T for ID in contourIDs], ax, contourIDs, [np.array(np.flip(cyclicColor(ID)))/255 for ID in contourIDs])
        plt.savefig(f'.\\picklesImages\\splitMerge\\{gc}_{id}.png')
        plt.close()

    return isThereASplit , sideOneContourIDs, sideTwoContourIDs


def sortMixed(arr):
    return sorted([i for i in arr if type(i) != str]) + sorted([i for i in arr if type(i) == str])
    #string_list = [str(x) for x in arr]
    #string_list.sort()
    #return [int(x) if x.isdigit() else x for x in string_list]

def graphUniqueComponents(nodes, edges, edgesAux= [], debug = 0, bgDims = {1e3,1e3}, centroids = [], centroidsAux = [], contours = [], contoursAux = []):
    # generally provide nodes and pairs of connections (edges), then retreave unique interconnected clusters.
    # small modification where second set of connections is introduced. like catalyst, it may create extra connections, but is after deleted.
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
            
    if len(edgesAux)>0:
        edgesAux = [v for v in edgesAux if v[1] in nodes]                                       # to avoid introducing new nodes via edgesAux. for all contour analysis you dont care because they are already in. in partial its a problem.
        edgesAux = np.array(edgesAux, int) #if len(edgesAux) >0 else np.empty((0,2),np.int16)    # not sure if needed. just in case. (copied from combosSelf)
        H.add_edges_from([[str(i),j] for i,j in edgesAux])
        connected_components_all = [list(nx.node_connected_component(H, key)) for key in nodes]
        connected_components_main = [sorted([subID for subID in IDs if type(subID) != str]) for IDs in connected_components_all]
    else:   
        #connected_components_main = [list(sorted(nx.node_connected_component(H, key), key=int)) for key in nodes]       # sort for mixed type [2, 1, '1'] -> ['1', 1, 2]
        connected_components_main = [sortMixed(list(nx.node_connected_component(H, key))) for key in nodes]
       # connected_components_main = [list(nx.node_connected_component(H, key)) for key in nodes] 

    connected_components_unique = []
    [connected_components_unique.append(x) for x in connected_components_main if x not in connected_components_unique]  # remove duplicates

    # debug can set node position if centroids dictionary is supplied, might require background dimensions
    # also contour dictionary can be used to visualise node objects.
    if debug == 1:
        plt.figure(1)
        idsAux = np.unique(edgesAux[:,0]) if len(edgesAux) > 0 else []
        clrs    = ['#00FFFF']*len(nodes)+ ['#FFB6C1']*len(idsAux)
        if len(centroids)>0:
            pos     = {ID:centroids[ID] for ID in nodes}
            posAux  = {str(ID):centroidsAux[ID] for ID in idsAux}
            posAll  = {**pos, **posAux}
            for n, p in posAll.items():
                H.nodes[n]['pos'] = p
            blank   = np.full((bgDims[0],bgDims[1],3),128, np.uint8)
            if len(contours)>0:
                [cv2.drawContours( blank,  [contours[i]], -1, (6,125,220), -1) for i in nodes]
                [cv2.drawContours( blank, [contoursAux[i]], -1, (225,125,125), 2) for i in idsAux]
            plt.imshow(blank)
            nx.draw(H, posAll, with_labels = True, node_color = clrs)
        else:
            nx.draw(H, with_labels = True, node_color = clrs)
        plt.show()
    return connected_components_unique


from scipy.interpolate import UnivariateSpline

def interpolateHull(points,k,s,numReturnPoints,debug):
    points      = points.reshape(-1,2)
    arclength   = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    arclength   = np.insert(arclength, 0, 0)/arclength[-1]                                  # normalize to % of total.
    splines     = [UnivariateSpline(arclength, coords, k=k, s=s) for coords in points.T]
    output      = np.array(np.vstack( spl(np.linspace(0, 1, numReturnPoints)) for spl in splines ).T,int)
    if debug == 1:
        _, ax = plt.subplots(ncols=2,sharex=False, sharey=True)
        ax[0].plot(*points.T,'-' ,label='original points', color = 'black');
        ax[0].scatter(*points.T,s=13, color = 'black');
        ax[1].plot(*output.T,'-' ,label='resampled', color = 'red');
        ax[1].scatter(*output.T,s=23, color = 'red');
        ax[1].scatter(*points.T,s=13, color = 'black');
        ax[0].set_aspect('equal', 'box')
        ax[1].set_aspect('equal', 'box')
        plt.show()
    return output

def getOverlap(a, b, mode): # if mode = 1, return intersection interval width, if mode  = 0, intersection right coordinate.
    return np.maximum(0, np.minimum(a[:,1], b[:,1]) - mode*np.maximum(a[:,0], b[:,0]))

def interval_overlap(interval1, interval2):
    overlap_start = max(interval1[0], interval2[0])
    overlap_end = min(interval1[1], interval2[1])
    if overlap_start < overlap_end:
        return [overlap_start, overlap_end]
    else:
        return []

def radialAnal( OGparams, SLparams, PermParams, StatParams, globalCounter, oldID, mainNewID, l_MBub_info_old, debug=0):
    [[OGLeft,OGRight], OGDistr]   = OGparams
    [isEllipse,l_contours,subNewIDs,predictCentroid, ellipseParams, cvr2, err]  = SLparams
    [l_rect_parms_all,l_rect_parms_old,l_Centroids,l_Areas] = PermParams
    [distCheck2, distCheck2Sigma, areaCheck, oldMeanArea, oldAreaStd, clusterCritValues] = StatParams
    SlaveBand, SlaveDistr       = radialStatsContoursEllipse(isEllipse,l_contours,subNewIDs,predictCentroid, ellipseParams, cvr2, err, oldID, globalCounter, debug = 0)
    ogInterval  = np.array([0,OGRight]).reshape(1,2)                                        # set min radius to 0, interval = [0,rmin+dr]. so all inside contours are automatically incorporated.
    slIntervals = np.array([np.array( [b[0], b[1]] ) for b in SlaveBand.values()])          # [[rmin1, rmin1 + dr1],[..],..]
    slWidths    = np.array([b[1]-b[0] for b in SlaveBand.values()],int)                     # [dr1,dr2,..]
    overlap     = getOverlap(ogInterval,slIntervals,1)                                      # how much area inside 
    rOverlap    = np.divide(overlap,slWidths)                                               # overlap area/interval area-> 1: fully inside ogInterval, 0: no overlap.
    passBase    = np.where(rOverlap >= 0.9)[0]                                              # returns tuple of dim (x,)
    passRest    = np.where((rOverlap > 0.05) & (rOverlap < 0.9))[0]
    baseIDs     = np.array(subNewIDs,int)[passBase]
                        
    # left band means r from [0,OG interval right coord], right band means  [OG interval right coord, rest]
    restIDs     = np.array(subNewIDs,int)[passRest]
    permIDsol2  = baseIDs.tolist()                                                          # base solution. can be empty. will be modified by restIDs
    # if there are prime candidates inside trusted region (baseIDs) and possible cadidates in buffer region (restIDs)
    # there might be a need to include buffer cadidates into baseIDs if prime cadidate is larger than buffer zone.
    if len(restIDs) > 0 and len(baseIDs) > 0:
        usefulIDs       = np.array(subNewIDs,int)[np.concatenate((passRest,passBase))]                                      # grab resolved IDs and ones in question
        bandRightRs     = {ID:  [r for r,_ in SlaveDistr[ID].items()              if r > OGRight]   for ID in usefulIDs}    # take Rs past OG right band coord
        bandRightCumSum = {ID:  np.cumsum([a for r,a in SlaveDistr[ID].items()    if r > OGRight])  for ID in usefulIDs}    # cum sum of right band
        bandRightCumSum = {ID:  arr for ID,arr in bandRightCumSum.items()         if len(arr) > 0}                          # interval fully within OG will be empty.
                            
        if len(np.intersect1d(list(bandRightCumSum.keys()), baseIDs)) > 0:                                                  # testing. only baseIDs is inside OG band. so no other accepted IDs can contain other stray IDs.
            aa = {ID:bandRightRs[ID][np.argmin(np.abs(cumSumArea- cvr2*cumSumArea[-1]))] for ID,cumSumArea in bandRightCumSum.items()}  # grab r at which area reaches 80%, in case contour is stretched thin at dist.
            maxRightInterval    = max([br - OGRight for ID,br in aa.items() if ID in baseIDs])                                 # take widest resolved right band*0.8 interval as a reference
            preResolvedRest     = [ID for ID,br in aa.items() if ID not in baseIDs and (br - OGRight)*0.8 < maxRightInterval]  # if Rest interval width is within 80% of maxRightInterval, consider it resolved.
            restIDs             = [ID for ID in restIDs if ID not in preResolvedRest]
            baseIDs             = np.concatenate((baseIDs,preResolvedRest)).astype(int)
            permIDsol2          = baseIDs.copy().tolist()               
    # baseIDs have been re-evaluted. do permutations and double criterium for restIDs+baseIDs. baseIDs might be empty.
    if len(restIDs) > 0:
        combs                   = [tuple(baseIDs)] if len(baseIDs) > 0 else []                                              # this is the base list of IDs, they are 100% included eg [1,2]
        restIDsPerms            = sum([list(itertools.combinations(restIDs, r)) for r in range(1,len(restIDs)+1)],[])       # these are permuations possible neighbors [3,4]-> [[3],[4],[3,4]]
        [combs.append(tuple(baseIDs) + perm) for perm in restIDsPerms]                                                      # add base to combos of neighbors -> [[1,2],[1,2,3],[1,2,4],[1,2,3,4]]

        permIDsol2, permDist2, permRelArea2 = centroidAreaSumPermutations2(l_contours,l_rect_parms_all, l_rect_parms_old[oldID], combs, l_Centroids, l_Areas,
                            predictCentroid, distCheck2 + 5*distCheck2Sigma, areaCheck, relAreaCheck = 0.7, debug = 0, doHull = 1)

    if debug == 1: compareRadial([OGLeft,OGRight], OGDistr, SlaveBand, SlaveDistr, permIDsol2, cyclicColor, globalCounter, oldID) # debug on top
    # solution is either empty, or its consist of elelents of baseIDs and restIDs
    # if its is not empty, calculate concave or convex hull based on whether bubble had a merge lately.
    newParams,hull,permRelArea2,newCentroid,newArea, = [],[],-1,-1, -1
    if len(permIDsol2)>0:
        if oldID in l_MBub_info_old:                                                    # it was part of a merge (big or small)
            ddd = 0 #if globalCounter != 3 and oldID != 4 else 1
            previousInfo        = l_MBub_info_old[oldID][3:] 
            hull, newParams, _  = mergeCrit(permIDsol2, l_contours, previousInfo, pCentr = predictCentroid, alphaParam = 0.05, debug = ddd ,debug2 = ddd, gc = globalCounter, id = oldID)
        else:
            hull                = cv2.convexHull(np.vstack(l_contours[permIDsol2]))
                                
        newCentroid, newArea    = getCentroidPosContours(bodyCntrs = [hull])
        permDist2               = np.linalg.norm(np.array(newCentroid) - np.array(predictCentroid))                                              
        permRelArea2            = np.abs(newArea-oldMeanArea)/ oldAreaStd
                        
    # no solution, revert to criterium vs original cluster # 02/04/23 had to recover from prefect cluster match. since dist2, areaCrit are inhereted wrongly.
    else:
        permDist2, permRelArea2 = clusterCritValues[mainNewID]
    return permIDsol2, permDist2, permRelArea2, newCentroid, newArea, hull, newParams