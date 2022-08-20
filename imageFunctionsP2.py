# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:17:57 2022

@author: User
"""
import glob, pickle, numpy as np, cv2
from matplotlib import pyplot as plt

def init(folder,imageNumber): # initialize some globals, so i dont have to pass them
    global imageMainFolder,imgNum
    imageMainFolder = folder
    imgNum = imageNumber
    # print(imageMainFolder,imgNum)  
    
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
        pickleSingleCaseSave = 1
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
                img = cv2.imread (myFile,0)
                bright  = adjustBrightness(img,adjustBrightness)
                dist    = undistort(bright)
                masks = glob.glob('./masks'+'/*.bmp')
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
                
        if recalcMean ==1 :
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

def cntParentChildHierarchy(image, mode, minArea, minAreaChild, CPareaRatio):
    childrenCNTRS, whereChildrenAreaFiltered = [],np.array([],dtype=object)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    whereParents0 = np.argwhere(hierarchy[:,:,3]==-1)[:,1] #flag -1> cnt /w no parents (no owner)
    
    parentSizeSel = [i for i in whereParents0 if cv2.contourArea(contours[i]) > minArea]
    # parentsCNTRS = [contours[i] for i in parentSizeSel]
    pIDsAreaFiltered = []
    cIDsAreaFiltered = {}
        
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

def getCentroidPosContours(bodyCntrs,holesCntrs=[]):
    areas1 = [cv2.contourArea(cntr) for cntr in bodyCntrs]
    areas0 = [cv2.contourArea(cntr) for cntr in holesCntrs]
    moms1 = [cv2.moments(cntr) for cntr in bodyCntrs]
    moms0 = [cv2.moments(cntr) for cntr in holesCntrs]
    centroids1 = [np.uint32([m['m10']/m['m00'], m['m01']/m['m00']]) for m in moms1]
    centroids0 = [np.uint32([m['m10']/m['m00'], m['m01']/m['m00']]) for m in moms0]
    totalMass = np.sum(areas1) - np.sum(areas0)   
    endCentroid = sum([w*a for w,a in zip(areas1,centroids1)]) - sum([w*a for w,a in zip(areas0,centroids0)])      
    endCentroid /= totalMass
    # print(endCentroid)
    return  tuple(map(int,np.ceil(endCentroid)))


from skimage.metrics import structural_similarity
# global gg
gg = 0;
def compareMoments(big,shape1,shape2,coords1,coords2):
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
    print(f'dist: {dist:.1f}; charDist: {charDist:.1f}')
    # area1, area2 = moms1['m00'] , moms2['m00']
    area1, area2 = areas
    if area1 > area2: areaChangePrec = 1- area2/area1
    else: areaChangePrec = 1- area1/area2
    print(f'areaChangePrec: {areaChangePrec:.2f}')

    if dist< charDist and areaChangePrec< 0.3: 
    # if 1 == 1: 
        useKeys = ('m20','m11','m02','m30',
                   'm21','m12','m03','mu20')#,'mu11','mu02','mu30'
        moms1a = [moms1[x] for x in useKeys]
        moms2a = [moms2[x] for x in useKeys]
        ratio = [a/b for a,b in zip(moms1a,moms2a)]
        a =     np.abs(1-np.array(ratio))
        weights = [3, 6, 3, 3, 3, 3, 3, 1]
        w_avg = np.average(a, weights = weights)
        std_dev = np.sum(weights * (a - w_avg)**2)/np.sum(weights)#;print(std_dev)
        zero_based = np.abs(a - w_avg)#;print(zero_based)
        max_deviations = 2
        outliers = a[zero_based > max_deviations * std_dev]#;print('outliers',outliers)
        outlierNames = [ i for i,c in enumerate(zero_based) if c >  max_deviations * std_dev]
        outlierNames = [useKeys[i] for i in outlierNames]#;print('outlierNames',outlierNames)
        
        stree = f'mean = {w_avg:.3f}; '+f'stdev = {std_dev:.3f}; '+" ".join([f'{a}= {b:1.3f};' for a,b in zip(outlierNames,outliers)])
    
        print(stree)
        print(" ".join([f'{a}= {b:1.3f};' for a,b in zip(useKeys,ratio)]))
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
