# openCV contour rotation and scale about either center of mass of arbitrary point
# https://math.stackexchange.com/questions/3245481/rotate-and-scale-A-point-around-different-origins
# i wanted to scale up A contour w/o having to use images and morphological dilate. 

import sqlite3
from time import process_time_ns
import numpy as np
import numpy
import cv2

#from bubble_process_tests import print01
#dic1 = {1:1,2:2}
#dic2 = {3:3,4:4}
#print(list(dic1.keys()+dic2.keys()))

#tempDistListTest = []
                
                #for oldID, oldCentroid in distanceBubCentroids_old.items():
                #    cntrds = list(centroidsByID2[oldID].values())
                #    distCheck = distStatPrediction(trajectory = cntrds,startAmp0 = 30, expAmp = 10, halfLife = 2, numsigmas = 2, plot=0,extraStr = f'E:ID {oldID} ')
                #    #areaCheck = 
                #    for mainNewID, subNewIDs in jointNeighbors.items():

#def closest_point(point, array):
#    diff = array - point
#    distance = np.einsum('ij,ij->i', diff, diff)
#    return np.argmin(distance), distance

#print(closest_point([4],np.array([1,2,4,7])))

#array = np.array([[1,3,tuple([1,1])],[2,2,tuple([2,2])],[3,1,tuple([3,3])]],dtype=object)
#A = array
#print(f'A =\n{A}')
#print(f'A[:, 1] = {A[:, 1]}')
#print(f'A[:, 1].argsort() = {A[:, 1].argsort()}')
#print(f'A[A[:, 1].argsort()] =\n{A[A[:, 1].argsort()]}')
#print('Arrays:')
#A = [1,4,9,13];print(f'A: {A}');print(f'A (deltas): {np.array(A)-min(A)}')
#B = [45,45,29,47];print(f'B: {B}');print(f'B (deltas): {np.array(B)-min(B)}')


#print('\nNormalization weights:')
#deltaA = max(A) - min(A)
#deltaB = max(B) - min(B)
#weightedA = (np.array(A)-min(A))/deltaA;print(f'weightedA: {[np.round(a, 2) for a in weightedA]}')
#weightedB = (np.array(B)-min(B))/deltaB;print(f'weightedB: {[np.round(a, 2) for a in weightedB]}')
#print('\nElement positions in order:')
#sortedA = np.argsort(np.argsort(A));print(f'sortedA (position): {[np.round(a, 2) for a in sortedA]}')
#sortedB = np.argsort(np.argsort(B));print(f'sortedB (position): {[np.round(a, 2) for a in sortedB]}')
#print('\nRescaled element positions:')
#rescaledArgSortA = np.matmul(np.diag(weightedA),sortedA);print(f'rescaledArgSortA (position): {[np.round(a, 2) for a in rescaledArgSortA]}')
#rescaledArgSortB = np.matmul(np.diag(weightedB),sortedB);print(f'rescaledArgSortB (position): {[np.round(a, 2) for a in rescaledArgSortB]}')

#print('\nMean of couples:')
#res = [np.mean([a,b]) for a,b in zip(rescaledArgSortA,rescaledArgSortB)];print(f'mean rescaled positions :\n{[np.round(a, 2) for a in res]}')
#resIndex = np.argmin(res);print(f'resIndex: {resIndex}') 
#elseOldNewDoubleCriteriumSubIDs = {18: [18, 28], 33: [33], 34: [34, 35]}
#aa=sum(elseOldNewDoubleCriteriumSubIDs.values(),[])
#print(aa)
#jn = {18: [18, 28, 32], 27: [27, 34, 35], 33: [33]}
#jn2 = [ [subelem for subelem in elem if subelem not in aa] for elem in jn.values()]
#jn3 = {min(vals):vals for vals in jn2 if len(vals) > 0}
#jn4 = {**jn3,**elseOldNewDoubleCriteriumSubIDs}
##jn2  = [subelem for subelem in elem if elem not in aa]
#print(jn2)
#print(jn3)
#print(jn4)
#from scipy import interpolate
#from scipy.interpolate import BSpline
#traj0 = {0: (728, 624), 1: (717, 625), 2: (704, 626), 3: (691, 627), 4: (677, 627), 5: (662, 625)}
#traj0 = {0: (222, 565), 1: (212, 559), 2: (201, 555), 3: (190, 549), 4: (179, 541), 5: (169, 535)}

#num = 5
#tmax = 5
#traj = {t:(3*t,np.cos(2*t)) for t in np.linspace(0,tmax,num)}

#x = [a[0] for a in traj0.values()]
#y = [a[1] for a in traj0.values()]

##------ fit a spline to the coordinates, x and y axis are interpolated towards t
#t = np.array(list(traj.keys()),np.float32) #t is # of values

#t1 = np.linspace(0, tmax + 1, 2*(len(traj)+int(num/tmax)))



#splBx = BSpline(t, x, 1, extrapolate = True)
#splBy = BSpline(t, y, 1, extrapolate = True)
#x2 = [splBx(a) for a in t1]
#y2 = [splBy(a) for a in t1]

#fsx2 = interpolate.interp1d(t, x, kind = 'cubic', fill_value="extrapolate")
#fsy2 = interpolate.interp1d(t, y, kind = 'cubic', fill_value="extrapolate")
#fx2 = fsx2(t1)
#fy2 = fsy2(t1)
#import matplotlib.pyplot as plt
#from scipy.interpolate import BSpline
        
#plt.plot(x2, y2, 'o', x, y, 'o')
#plt.plot(t, x, 'o', t1, fx2,'o')
#plt.plot(fx2, fy2, 'o',label='interp1d')
#plt.plot( x, y, '-.',label = 'orig')
#plt.legend()

#plt.show()

#showDistDecay = True    
#def distStatPredictionVect(trajectory, zerothDisp, maxInterpSteps = 3, maxInterpOrder = 2, mode = 1,debug = 0, maxNumPlots = 4):
#    global showDistDecay
#    predictPoints = []
#    numPointsInTraj = len(trajectory)
#    numStepsInTraj = numPointsInTraj - 1 
#    if mode  ==  1:
#        numStepsFor = list(range(max(0,numPointsInTraj-maxNumPlots),len(trajectory)))
#    else:
#        numStepsFor = [numStepsInTraj]
#    numPlots = len(numStepsFor) if mode  ==  1 else 2
#    if debug == 1:
#        fig, axes = plt.subplots(1,numPlots , figsize=( numPlots*5,5), sharex=True, sharey=True)
#    for numSteps,numSteps2 in enumerate(numStepsFor):
#        numSteps = 1 if mode == 0  else numSteps

#        start = 0 if numSteps2 < maxInterpSteps else numSteps2-maxInterpSteps
#        x = np.array([a[0] for a in trajectory[start:numSteps2+1]])
#        y = np.array([a[1] for a in trajectory[start:numSteps2+1]])
#        t = np.arange(0,len(x),1)
#        t1 = np.arange(0,len(x)+1,1)
        
#        if numSteps2 == 0:
#            predictPoints.append([trajectory[0][0]+zerothDisp[0],trajectory[0][1]+zerothDisp[1]])
#            if debug == 1:
#                axes[numSteps].plot(x, y, 'o',c='green', label = 'traj')
#                axes[numSteps].plot([x[0],predictPoints[0][0]], [y[0],predictPoints[0][1]], '--o', label = 'forecast')
#        if numSteps2 > 0:
#            k = min(numSteps2,maxInterpOrder); print(f' interpOrder = {k}') if debug == 1 else 0
#            spline, _ = interpolate.splprep([x, y], u=t, s=0,k=k)
#            new_points = interpolate.splev(t1, spline,ext=0)
#            if debug == 1:
#                axes[numSteps].plot(new_points[0][-2:],new_points[1][-2:], '--o', label = 'forecast')
#            if mode != 0:
#                if numSteps > 0 and debug == 1:
#                    axes[numSteps].plot([x[-2],predictPoints[-1][0]],[y[-2],predictPoints[-1][1]], '--o', label = 'prev forecast')
#                predictPoints.append([new_points[0][-1],new_points[1][-1]])
                
#            else: predictPoints.append([new_points[0][-1],new_points[1][-1]])
#        if debug == 1: axes[numSteps].plot(x, y, '-o',c='green', label = 'traj')

#    if debug == 1:
#        plt.legend(loc=(1.1, 0.5))
#        plt.show()
#    return np.array(predictPoints[-1],np.uint32)
#print(distStatPredictionVect(list(traj0.values())[:], zerothDisp = [0,-7], maxInterpOrder = 1, mode = 1, debug = 1))
#print([1,2][:0+1])
#print(f'trajectory: {trajectory}')   
#    if (plot == 1 and numSteps >=  cutoff+1 and showDistDecay == True) or plot == 2:
#        fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False, sharey=False)
#        fig.suptitle(f'A0 {startAmp0}, A {expAmp}, cutoff {cutoff}, total Traj len {len(trajectory)}\n')
#        axes[0].axvspan(xmin = 2, xmax = cutoff+1,alpha = 0.1 , label = 'exp zone')
#        addedBoost = np.concatenate(([startAmp0] , expAmp*np.exp(-lam * ( np.arange(1,cutoff+1,1)-1) ))) 
#        steps = np.arange(1, len(addedBoost)+1 ,1)
#        axes[0].plot(steps,addedBoost,label=f' A0 and Aexp(N) ',linestyle = '--',marker="o")
        
#        maxStepsDraw = min(numSteps,10)
#        magsVals = mags[:maxStepsDraw]
#        stepsNum = np.arange(2,maxStepsDraw+2,1)
#        axes[0].plot(stepsNum,magsVals,linestyle = '-',marker="o",label='delta d -> d_i- d_(i-1)')
        
#        means = [np.mean(magsVals[:i]) for i in range(1, len(magsVals)+1,1)]# if i < len(magsVals)
#        meansSteps = np.arange(2,len(means)+2,1)
#        axes[0].plot(meansSteps,means,linestyle = '--',marker="o",label= 'runing delta d mean')
        
#        sigmas = np.array([np.std(magsVals[:i+1]) for i in range(1,len(magsVals),1)])*numsigmas
#        sigmas = np.pad(sigmas, (1, 0), 'constant')
#        sigmasSteps = np.arange(3,len(sigmas)+3,1)#[print(magsVals[:i+1]) for i in range(1,len(magsVals),1) ]
#        axes[0].errorbar(meansSteps, means, yerr=sigmas, label = f' running d {numsigmas}*stdev',  lolims=True, linestyle='')
#        # axes[0].plot(sigmasSteps,sigmas,linestyle = '-.',label=f'running mean + {numsigmas}*stdev')
#        # axes[0].fill_between(sigmasSteps, sigmas, means[1:],label = f' running d {numsigmas}*stdev', alpha = 0.1, color='r')
        
#        if cutoff + 1 > maxStepsDraw + 1:
#            sumAll = addedBoost[:maxStepsDraw+1]
#        else:
#            sumAll = np.pad(addedBoost, (0, maxStepsDraw+1 - len(addedBoost)), 'constant')
            
#        sumAll += np.pad(means, (1, 0), 'constant')
#        sumAll += np.pad(sigmas, (1, 0), 'constant') # it was padded once previously
        
#        stepsAll = np.arange(1, len(sumAll)+1,1)
#        axes[0].plot(stepsAll,sumAll,linestyle = '-',marker="o",label=' A0+Aexp+mean1+meanStd',linewidth = 2.6)
#        axes[0].set_title(extraStr+'Distance progression')
#        axes[0].legend()
#        axes[0].set_ylim(bottom=0)
#        axes[0].set_xlim(left=1)
#        axes[0].set_xlabel('number of steps N in trajectory')
#        axes[0].set_ylabel('displacement magnitude, px')
                           
#        x = [a[0] for a in trajectory]
#        y = [a[1] for a in trajectory]
#        axes[1].plot( x, y, '-.', x, y, 'o',label = 'orig')
#        showDistDecay = False
#        plt.show()

#def updateSTDEV(meanOld, stdevOld, setLengthOld, meanNew, elementNew):
#    stdevSqr = 1/setLengthOld * ((setLengthOld-1) * (stdevOld)**2 + (elementNew - meanNew)*(elementNew - meanOld))
#    return np.sqrt(stdevSqr)

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
#def updateStat(count, mean, std, newValue):
#    M2 = count * std**2
#    count += 1
#    delta = newValue - mean
#    mean += delta / count
#    delta2 = newValue - mean
#    M2 += delta * delta2
#    if count < 2:
#        return float("nan")
#    else:
#        (mean, variance) = (mean, M2 / count)
#        return (mean, np.sqrt(variance))


# Retrieve the mean, variance and sample variance from an aggregate
#def finalize(existingAggregate):
#    (count, mean, M2) = existingAggregate
#    if count < 2:
#        return float("nan")
#    else:
#        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
#        return (mean, variance, sampleVariance)
#set0 = [1]
#mean0 = np.mean(set0)
#std0 = np.std(set0)
#print(f'mean0: {mean0}, std0: {std0}')
#set1 = set0 + [6]
#mean1 = np.mean(set1)
#std1 = np.std(set1)
#print(f'mean1: {mean1}, var: {std1}')
#aggregate = updateStat(len(set0),mean0, std0, 6)

#aa = updateSTDEV(mean0,std0, len(set0), mean1, 6)
#print(aggregate)
#lst = [[1,2]]*3;print(lst)
#import os
#predictVectorPathFolder = r'./debugImages/predictVect'
#if not os.path.exists(predictVectorPathFolder): os.makedirs(predictVectorPathFolder)
#intersectingCombs_stage2 = [[1,1],["1",1],[3,1],[4,2],[5,2],[6,3]]
#keysNew = [a[1] for a in intersectingCombs_stage2]
#keysNewVals = np.array([a[0] for a in intersectingCombs_stage2],dtype=object)
#values, counts = np.unique(keysNew, return_counts=True)
#duplicates = [ a for a,b in zip(values, counts) if b>1]
#duplicates2 = {a:np.argwhere(keysNew == a).reshape(-1).tolist() for a in duplicates}
#dupVals = {ID:keysNewVals[lst] for ID,lst in duplicates2.items()}
##keysNewUniq = set(keysNew)
##whereduplicates = [i for i, x in enumerate(keysNew) if keysNew.count(x) > 1]
##duplicates = [ newKey for newKey in set(keysNew) if keysNew.count(newKey) > 1]
#intersectingCombs_stage22 = np.array(intersectingCombs_stage2,dtype=object)
#print(intersectingCombs_stage22[:,0])
#print(duplicates,duplicates2,dupVals)
#debug = 1
#A = [29.42787794,  6.32455532, 30, 5]
#deltaA = max(A) - min(A)
#print(f'A: (centroid diff  min pos) {A}') if debug == 1 else 0
#print(f'A (deltas): {deltaA}') if debug == 1 else 0
#B = [0.50992063, 0.5546398 , 0.2, 0.6]
#deltaB = max(B) - min(B)
#print(f'B: (area ratio  min pos) {B}') if debug == 1 else 0 
#print(f'B (deltas): {deltaB}') if debug == 1 else 0
#if deltaB == 0 or deltaA == 0: # in detectStuckBubs() i pass both identical objects cluster and separate bubs : {"1":[1,2], 1:[1], 2:[2]}. so permutation "1"  = 1 + 2. hard to throw it out of function
#    deltaB = 1;deltaA = 1
#weightedA = np.array(A)/max(A);print(f'weightedA: {[np.round(a, 2) for a in weightedA]}') if debug == 1 else 0
#weightedB = np.array(B)/max(B);print(f'weightedB: {[np.round(a, 2) for a in weightedB]}') if debug == 1 else 0
#sortedA = np.argsort(np.argsort(A));print(f'sortedA (position): {[np.round(a, 2) for a in sortedA]}') if debug == 1 else 0
#sortedB = np.argsort(np.argsort(B));print(f'sortedB (position): {[np.round(a, 2) for a in sortedB]}') if debug == 1 else 0
#rescaledArgSortA = np.matmul(np.diag(weightedA),sortedA);print(f'rescaledArgSortA (position): {[np.round(a, 2) for a in rescaledArgSortA]}') if debug == 1 else 0
#rescaledArgSortB = np.matmul(np.diag(weightedB),sortedB);print(f'rescaledArgSortB (position): {[np.round(a, 2) for a in rescaledArgSortB]}') if debug == 1 else 0
#res = [np.mean([a,b]) for a,b in zip(rescaledArgSortA,rescaledArgSortB)];print(f'mean rescaled positions :\n{[np.round(a, 2) for a in res]}') if debug == 1 else 0
#resIndex = np.argmin(res);print(f'resIndex: {resIndex}') if debug == 1 else 0
#print(f'v2: {[np.mean([a,b]) for a,b in zip(weightedA,weightedB)]}')

#def doubleCritMinimum(setA,setB, mode = 0, debug = 0, printPrefix=''):
#    print(printPrefix) if len(printPrefix)>0 and debug == 1 else 0
    
#    if mode == 0:
#        print('rescaling data to {0, max(set)}->{0, 1}') if debug == 1 else 0
#        weightedA = np.array(setA)/max(setA)
#        weightedB = np.array(setB)/max(setB)
#    else:
#        print('rescaling data to {min(set, max(set)}->{0, 1}') if debug == 1 else 0
#        weightedA = (np.array(setA)-min(setA))/(max(setA) - min(setA))
#        weightedB = (np.array(setB)-min(setB))/(max(setB) - min(setB))
    
#    res = [np.mean([a,b]) for a,b in zip(weightedA,weightedB)]
    
#    resIndex = np.argmin(res)
#    if debug == 1:
#        print(f'setA: {[np.round(a, 2) for a in setA]}') 
#        print(f'setB: {[np.round(a, 2) for a in setB]}') 
#        print(f'weightedA: {[np.round(a, 2) for a in weightedA]}') 
#        print(f'weightedB: {[np.round(a, 2) for a in weightedB]}') 
#        print(f'mean of weight pairs: {[np.round(a, 2) for a in res]}')
#        print(f'smallest index: {resIndex}')
#    return resIndex

#doubleCritMinimum(setA = [29.42787794,  6.32455532, 30, 5],setB = [0.50992063, 0.5546398 , 0.2, 0.6] , mode = 0, debug = 1, printPrefix='')

#dic1 = {1:[1],2:[3,4]}
#dic2 = {2:[3,4],1:[1]}
#print(dic1==dic2)
#jointSubIDs = {0: [11], 1: [16], 3: [24, 32], 4: [25], 5: [28], 6: [29], 7: [45]}
#ie = [16,45]
#gg = [{ii:ID for ID, vals in jointSubIDs.items() if ii in vals} for ii in ie]
#gg2 = {};[gg2.update(elem) for elem in gg]
#print(gg2)

#print(map(**,gg))
#oldLocIDs = ['5'];print(type(oldLocIDs[0])== str)
#jointSubIDs = {}
#oldGlobIDs0 = [
#                                'str' if type(ii) == str else 'int'
#                               for ii in oldLocIDs ] 
#print(oldGlobIDs0)
#tup = (1,1,3,4)
#print(np.linalg.norm(tup[-2:]))
#jointNeighbors = {32: [32, 39], 37: [37, 40], 22: [22, 30, 33], 35: [35, 38], 54: [54]}
#frozenIDs = np.array([54, list([35, 38])], dtype=object)
#[print(a) for a in jointNeighbors.values() if a in frozenIDs.tolist()]
#frozenIDsInfo = np.array([[45, 54, 2.23606797749979, 0.2, (325, 48)],
#       [28, list([35, 38]), 2.23606797749979, 0.2, (835, 445)]],
#      dtype=object)
#print(frozenIDsInfo[:,1])
#print( np.array([a if type(a) != list else min(a) for a in frozenIDsInfo[:,1]]))
#frozenIDsInfo[:,1] =  np.array([a if type(a) != list else min(a) for a in frozenIDsInfo[:,1]])
#print(frozenIDsInfo)
#print(1) if [1] in [[1],[2]] else 0

#fID = np.array([39], dtype=np.int64).tolist()
#comb = [24, 34, 39]
#difference = np.setdiff1d(comb, fID);print(difference)
#distanceOldNewIDs_old = {3: [29], 4: [24, 32], 5: [25], 6: [28], 8: [45]}
#frozenIDs_old	= np.array([28,32], dtype=np.int64)
#frozenIDs_old_glob = [oldGlobID for elem in frozenIDs_old for oldGlobID,oldLocIDs in  distanceOldNewIDs_old.items() if elem in oldLocIDs ]
#print(frozenIDs_old_glob)
#returnInfo = np.array([], dtype=object)
#print(len(returnInfo))
#print(
#    (lambda x,y: np.degrees(np.arccos(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1.0, 1.0))))([1,-1],[1,1])
#)
#import matplotlib.pyplot as plt
#new_points = [np.array([920., 906., 901., 905.]),np.array([476., 483., 482., 473.])]
#new_points = [np.array([0,1,1]),np.array([0,0,1])]
#new_points2 = np.array(new_points)
#print(new_points2)
#minxy = np.min(new_points2, axis=1)
#print(minxy)
#print(new_points2.T - minxy)
#new_points3 = new_points2.T - new_points2[:,0]
#new_points3 = np.transpose(new_points3)
#plt.plot(*new_points3,'-o')

##print(np.min(new_points2, axis=1))
#vs = new_points2[:,-2:] - new_points2[:,-3:-1]
##vs = vs.T
#aa = np.ravel([[0,0],vs[:,0]], order='F')
#bb = np.ravel([[0,0],vs[:,1]], order='F')
#print(aa,bb)
#plt.plot(aa,bb,'-o')
##vs2 = np.ravel([vs,[[0,0],[0,0]], order='F');print(vs2)          
##vs = vs.T

#print(vs)

#angleDeg = (lambda x,y: np.degrees(np.arccos(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1.0, 1.0))))(*np.transpose(vs))
#print(angleDeg)
#plt.show()
extrapolatedPointsDebug = np.array([[1. , 1.4, 1.8, 2.2, 2.6, 3. , 3.4, 3.8, 4.2, 4.6, 5. , 5.4, 5.8,
        6.2, 6.6, 7. , 6.4, 5.8, 5.2, 4.6, 4. , 3.6, 3.2, 2.8, 2.4, 2. ,
        2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. ]])
tDebug = np.array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4,
       2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2, 4.4, 4.6, 4.8, 5. ,
       5.2, 5.4, 5.6, 5.8, 6. , 6.2, 6.4, 6.6, 6.8, 7. ])

aa = np.vstack([extrapolatedPointsDebug[0],tDebug])
from scipy import interpolate
import matplotlib.pyplot as plt

#from bubble_process_tests import print01
#x = np.array([920, 906, 901])
#y = np.array([476, 483, 482])
#t = np.array([0, 1, 2])
#t1 = np.array([0, 1, 2, 3])
#k = 2
#sMod = 212
#spline, _ = interpolate.splprep([x, y], u=t, s=sMod,k=k)
#new_points = np.array(interpolate.splev(t1, spline,ext=0))
#fig, axes = plt.subplots(1,2 , figsize=( 13,5), sharex=False, sharey=False)
#axes[0].plot(x, y, '-o',c='green', label = 'traj')
#axes[0].plot(new_points[0],new_points[1], '-o', label = f'forecast: s:{sMod:0.1f}, k:{k:0.1f}', ms= 2,linewidth = 1)
#axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
#axes[0].grid()
#plt.show()
def extrapolate(data, maxInterpSteps = 3, maxInterpOrder = 2, smoothingScale = 0, zerothDisp=[],fixSharp = 0, angleLimit = 30, debug = 0, axes=[], pltString = ''):
    data = np.array(data).reshape(len(data),-1) #[1,2]->[[1],[2]]; [[11,12],[21,22]]-> itself
    numPointsInTraj = data.shape[0]
    numStepsInTraj = numPointsInTraj - 1 #;print(f'numPointsInTraj:{numPointsInTraj}, numStepsInTraj:{numStepsInTraj}')
    numDims = data.shape[1]
    # start-> take last maxInterpSteps
    start = 0 if numStepsInTraj < maxInterpSteps else numStepsInTraj-maxInterpSteps
    if numStepsInTraj == 0:
        zeroth = [0]*numPointsInTraj if len(zerothDisp) == 0 else zerothDisp
        return data[:,-1]+ zeroth, []
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

            extrapolatedPoint0 = np.array(interpolate.splev([t[-1],t[-1]+1], spline0,ext=0))
            dv0 = np.diff(extrapolatedPoint0).reshape(max(2,numDims))
            z = np.polyfit(*splitSubsetComponents[:,-3:], 1);print(f'z:{z}')
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
        extrapolatedPoint0 = np.array(interpolate.splev([t[-1],t[-1]+1], spline0,ext=0))
        returnVec = extrapolatedPoint0[:,-1]

    if debug == 1:
        tDebug = np.arange(0,numPointsInTraj-start+0.01,0.2)
        extrapolatedPointsDebug2 = np.array(interpolate.splev(tDebug, spline0,ext=0))
        axes.plot(*extrapolatedPointsDebug2,'-o',label = f'order: {k}, smoothing: {alpha:0.1f}*sMax ({sMod2:0.1f})',ms= 3)
        if fixSharp == 1:
            axes.plot(*np.array([p0,p1]).T,'--',label = f'3 step linear fit (*)')
            axes.plot([p0[0]+dv[0],p0[0],p0[0]+dv0[0]],[p0[1]+dv[1],p0[1],p0[1]+dv0[1]],'-',label = f'angleDeg between (*) and exterp:{angleDeg:0.1f}',lw=3)
        axes.plot(*splitSubsetComponents,'X',label = 'Original pts (subset)',ms= 7)
        axes.legend(prop={'size': 6})
        axes.set_title(pltString+"extrapolate() debug")
        #plt.show()
    return returnVec if numDims > 1 else returnVec[1]

    
        #returnVec = np.array([x[0]+zerothDisp[0],y[0]+zerothDisp[1]])
    #splitSubsetComponents = [np.array([a[0] for a in data[start:]])

#print(np.hstack([data,t.reshape(len(t),-1)]))
#extrapolate([[11,21,31,41,51],[12,22,32,42,52]])
#data = np.array([[1,0],[2,0],[3,0],[3.3,0.7]])
#data = np.array([1,3,5,7,4,2,3])
#fig, axes = plt.subplots(1,1 , figsize=( 13,5), sharex=False, sharey=False)
#pred = extrapolate(data, smoothingScale = 1, maxInterpSteps = 15, maxInterpOrder = 3,fixSharp = 0,debug = 1,axes = axes)
#plt.show()
#a = np.array([[2, 82.14, np.array([2.46666667, 0.95      ])],[2, 82.14, np.array([3.46666667, 1.95      ])]], dtype=object)
#print(f'pred: {pred}, a:{a}')

#def updateStat(count, mean, std, newValue):
#    M2 = count * std**2
#    count += 1
#    delta = newValue - mean
#    mean += delta / count
#    delta2 = newValue - mean
#    M2 += delta * delta2
#    if count < 2:
#        return float("nan")
#    else:
#        (mean, variance) = (mean, M2 / count)
#        return (mean, np.sqrt(variance))
#ss = []
##print(updateStat(len(ss), np.mean(ss), np.std(ss), 1))

#if 1==1 and type('a') == str and 1 + 1 == 2 and 1/0 == 0: print('great success!')
#else: print('fail!')
#timeLibs = {0: [-1, '6'], 1: ['6', 28]}
#print((max(timeLibs, key=timeLibs.get)))
#dicc = {1:1,2:2,3:3}
#dicc2 = {11:11,21:21,31:31}
#drop = [1,3,21]
#dropKeys = lambda x: {key:val for key,val in x.items() if key not in drop}
#print(dicc,dicc2)
#print(f'drop: {drop}')
#[dicc,dicc2] = list(map(dropKeys,[dicc,dicc2]))
#print(dicc,dicc2)
#toList = lambda x: [x] if type(x) != list else x
#overlapingContourIDList = [22, 26, 28]
#newFoundBubsRings = [21,22]
#[
#    [newFoundBubsRings.remove(ii) for ii in toList(inter)]
#   for inter in np.intersect1d(overlapingContourIDList,newFoundBubsRings) if len(toList(inter))>0]
#print(newFoundBubsRings)

#aa = [1,1,3,5,6,6,7]
#u, c = np.unique(aa, return_counts=True)

#dup = u[c > 1]
#print(1)

#m1 = np.array([[1,12],[3,4]])
#m2 = np.array([[11,2],[13,14]])
#print(np.maximum.reduce([m1,m2]))


#arr = ['4', (967, 605),3,'4', (967, 605)]
#unq = list(set(arr))
#cnt = [arr.count(a) for a in unq]
#print(cnt)
#def doubleCritMinimum(setA,setB, mode = 0, debug = 0, printPrefix=''):
#    print(printPrefix) if len(printPrefix)>0 and debug == 1 else 0
    
#    if mode == 0:
#        print('rescaling data to {0, max(set)}->{0, 1}') if debug == 1 else 0
#        weightedA = np.array(setA)/max(setA)
#        weightedB = np.array(setB)/max(setB)
#    else:
#        print('rescaling data to {min(set, max(set)}->{0, 1}') if debug == 1 else 0
#        weightedA = (np.array(setA)-min(setA))/(max(setA) - min(setA))
#        weightedB = (np.array(setB)-min(setB))/(max(setB) - min(setB))
    
#    res = [np.mean([a,b]) for a,b in zip(weightedA,weightedB)]
    
#    resIndex = np.argmin(res)
#    if debug == 1:
#        #print(f'setA: {[np.round(a, 2) for a in setA]}') 
#        #print(f'setB: {[np.round(a, 2) for a in setB]}') 
#        print(f'weightedA: {[np.round(a, 2) for a in weightedA]}') 
#        print(f'weightedB: {[np.round(a, 2) for a in weightedB]}') 
#        print(f'mean of weight pairs: {[np.round(a, 2) for a in res]}')
#        print(f'smallest index: {resIndex}')
#    return resIndex

#elseOldNewDoubleCriterium = [[8, 39, 15.033296378372908, 1.6082997301997093], [3,3,3,3],[8, 42, 116.40017182117903, 1.1947117842512815], [9, 12, 116.40017182117903, 1.1947117842512815], [9, 22, 16.40017182117903, 5.1947117842512815]]
#print(elseOldNewDoubleCriterium)

#if len(elseOldNewDoubleCriterium)>0:
#    IDs = [a[0] for a in elseOldNewDoubleCriterium]
#    u, c = np.unique(IDs, return_counts=True)
#    if(len(c[c>1])>0):
#        dropCopies = []
#        for ID, cnt in zip(u,c):
#            if cnt > 1:
#                where = np.argwhere(IDs==ID).flatten()
#                print(f'ID:{ID} where',where)
#                setA = [elseOldNewDoubleCriterium[here][2] for here in where]
#                setB = [elseOldNewDoubleCriterium[here][3] for here in where]
#                hehe = doubleCritMinimum(setA ,setB, mode = 0, debug = 0, printPrefix='')
#                print(f'ID:{ID} good index',hehe)
#                deleteThese = [a for a in where if a not in [where[hehe]]]
#                print(f'ID:{ID} del these',deleteThese)
#                dropCopies = dropCopies + deleteThese
#        print(dropCopies)
#        elseOldNewDoubleCriterium = [entry for ID,entry in enumerate(elseOldNewDoubleCriterium) if ID not in dropCopies]
#print(elseOldNewDoubleCriterium)

#arr = np.array([[12, 16]])
#arr0 = np.empty((0,2))
#arr1 = np.vstack((arr0,arr))
#print(arr1)


#arr = np.array(['6','7'], dtype=object)
##mask = [True if a in []]

#arr2 = np.array(['6',32], dtype=object)
#print([True if a in arr else False for a in arr2])



    
#arr = ['0', '1', '4', '5', '7', '8', '6']
#print()
#print(np.argwhere(np.array(arr,np.int8)==int('6'))[0,0])


#print(np.argwhere(arr == 22).reshape(-1).tolist())
#arr = {1: [0,6],2:[1,2],3:[0,2,3]}
#
#srch = [2,6]
#sol = srch * 0
#aa = {ID:max([time for time,IDs in arr.items() if ID in IDs]) for ID in srch}
#print(aa)
#a = list(filter(lambda x: x[1].index(srch) if srch in x[1]), arr.items()))
#print(a)

#allFrozenLocalIDs0 = [[32,14], [28],[13]]
#allFrozenLocalIDs = sum(allFrozenLocalIDs0,[])
#print(f'drop {allFrozenLocalIDs}')
#temp = [[11,9,32,14],[4], [28,24],[0,13]]
#print(f'full {temp}')
#temp2 = []
#for subTemp in temp:
#    bfr = []
#    for elem in subTemp:
#        if elem not in allFrozenLocalIDs: bfr.append(elem)
#    temp2.append(bfr)
##[for subTemp in temp2]
#print(temp2)#subTemp.remove(elem) for elem in allFrozenLocalIDs if allFrozenLocalIDs in subTemp 
#
#keysNewVals = np.array([24, 34, 24, 34, 31, 15, 24, 34, 39, 41, 34, 39], dtype=object)
#dupWhereIndicies = {'5': [10, 11], '4': [6, 7, 8], 18: [0, 1], 28: [2, 3]}
#keysOld = np.array([18, 18, 28, 28, '1', '0', '4', '4', '4', '6', '5', '5'], dtype=object)
#dupSplit = ['5', '4', 18, 28]
#dupWhereIndicies = {a:np.argwhere(keysOld == a) for a in dupSplit}
#dupVals = {ID:keysNewVals[lst] for ID,lst in dupWhereIndicies.items()}





#cv2.imshow('a',img)

#print(bodyCntrs[15]) #[[[1009  518]],[[a,b]]]
#doHull = 1;debug = 0;distCheck = 59.67;relAreaCheck = 0.7;refArea = 18170
#permutations = sum([list(itertools.combinations(IDsOfInterest, r)) for r in range(1,len(IDsOfInterest)+1)],[])          # different combinations of size 1 to max cluster size.
#cntrds2 =  np.array([getCentroidPosCentroidsAndAreas([centroidDict[k] for k in vec],[areaDict[m] for m in vec]) for vec in permutations])
#if doHull == 1:
#    hullAreas = np.array([cv2.contourArea(cv2.convexHull(np.vstack([bodyCntrs[k] for k in vec]))) for vec in permutations])
#else:
#    hullAreas = np.array([sum([areaDict[m] for m in vec]) for vec in permutations])
#print(f'permutations,{permutations}') if debug == 1 else 0
#print(f'refC: {refCentroid}, cntrds2: {list(map(list,cntrds2))}') if debug == 1 else 0
#distances = np.linalg.norm(cntrds2-refCentroid,axis=1)
#distPassIndices = np.where(distances<distCheck)[0]
#relAreas = np.abs(refArea-hullAreas)/refArea 
#relAreasPassIndices = np.where(relAreas<relAreaCheck)[0]
#passBothIndices = np.intersect1d(distPassIndices, relAreasPassIndices)


#cv2.INTERSECT_FULL
#group1Params = {'1': [0, 605, 86, 179], '2': [369, 500, 127, 30], '3': [931, 434, 114, 119], '4': [713, 432, 135, 207], '5': [1093, 429, 115, 98]}
#group2Params = {26: (371, 506, 23, 17), 34: (1064, 465, 19, 22), 36: (911, 441, 115, 114), 41: (704, 436, 136, 206), 42: (1079, 424, 119, 112)}
#allCombs =  list(itertools.product(group1Params, group2Params))
#for (keyOld,keyNew) in allCombs:
#    x1,y1,w1,h1 = group2Params[keyNew]
#    rotatedRectangle_new = ((x1+w1/2, y1+h1/2), (w1, h1), 0)
#    x2,y2,w2,h2 = group1Params[keyOld]
#    rotatedRectangle_old = ((x2+w2/2, y2+h2/2), (w2, h2), 0)
#    interType,aa = cv2.rotatedRectangleIntersection(rotatedRectangle_new, rotatedRectangle_old)
#    if interType > 0 and keyOld == '2':
#        print(keyOld,keyNew,interType)#,aa.reshape((-1,2)),aa.reshape((-1,1,2))
#        cv2.rectangle(img, (x1,y1), (x1+w1,y1+h1), (255,0,0), 1)
#        cv2.rectangle(img, (x2,y2), (x2+w2,y2+h2), (120,255,0), 1)
#
#        img = cv2.drawContours(img, [aa.astype(int)], -1, (0,0,255), -1)

def cyclicColor(index):
    colors = [(255,0,0),(0,255,0),(125,125,0),(0,125,125),(0,0,255),(125,0,125),(255,125,0),(255,0,125),(125,255,0)]
    colors = np.array(colors,dtype=np.uint8)
    # np.random.shuffle(colors)
    return colors[index % len(colors)].tolist()

dm = 500
numP = 110
dPhi = 2*np.pi/numP
img = np.zeros((dm,dm,3),np.uint8)

if 1 == 1:
    import itertools, pickle
    img = np.zeros((1000,1400,3),np.uint8)
    with open('cntr.pickle', 'rb') as handle:
                    bodyCntrs = pickle.load(handle)
    #rectParams = {0: (1177, 895, 5, 5), 2: (265, 890, 6, 5), 3: (123, 880, 5, 5), 9: (944, 675, 7, 6), 10: (955, 610, 6, 5), 11: (963, 598, 16, 9), 12: (1030, 589, 5, 5), 13: (1041, 583, 8, 7), 14: (352, 545, 13, 9), 15: (1005, 518, 124, 33), 16: (316, 498, 21, 21), 17: (305, 495, 91, 98), 18: (1091, 493, 35, 26), 19: (615, 484, 14, 10), 20: (857, 482, 25, 29), 21: (1153, 473, 9, 7), 22: (866, 457, 101, 101), 25: (1119, 454, 20, 53), 26: (1183, 452, 15, 10), 27: (648, 444, 164, 205), 30: (971, 422, 154, 91), 32: (924, 311, 6, 7), 36: (752, 94, 6, 6), 37: (307, 48, 19, 13), 38: (373, 43, 6, 8), 39: (611, 35, 5, 5), 41: (285, 3, 5, 7)}
    IDsOfInterest = [15, 18, 20, 22, 25, 30]
    refCentroid = np.array([1061.78233365,  491.36451101]).astype(np.int16)
    [x,y,w,h] = [996, 421, 151, 126]
#    cv2.circle(img, tuple(map(int,refCentroid)), 3, (255,0,0), -1)
#    cntrsOfInterest = {ID:bodyCntrs[ID].reshape((-1,2)).astype(np.int16)-refCentroid for ID in IDsOfInterest}
#    cntrsOfInterest_Polar = {}
#    for ID in IDsOfInterest:
#        x,y = np.transpose(cntrsOfInterest[ID])
#        #print([np.arctan2(y[0],x[0]),np.sqrt(x[0]**2+y[0]**2)])
#        cntrsOfInterest_Polar[ID] = np.transpose(np.array([np.arctan2(y,x),numpy.linalg.norm([x,y], axis=0)]))
#        cntrsOfInterest_Polar[ID] = np.vstack((cntrsOfInterest_Polar[ID],cntrsOfInterest_Polar[ID][0]))
#    area = {}
#    Ravg = {}
#    diffs = {ID:np.diff(arr,axis = 0) for ID,arr in cntrsOfInterest_Polar.items()}
#    print(1)
#    for ID in IDsOfInterest:
#        area[ID] = 0
#        Ravg[ID] = 0
#        for i,[dAng,dR] in enumerate(diffs[ID]):
#            R = cntrsOfInterest_Polar[ID][i][1]
#            Reff = (R+0.5*dR)
#            dA = 0.5*Reff**2*np.sin(dAng)
#            area[ID] = area[ID] + dA
#            Ravg[ID] = Ravg[ID] + 2/3* Reff * dA
#        Ravg[ID] = Ravg[ID]/area[ID]
#cntOG = {ID:bodyCntrs[ID].reshape((-1,2)) for ID in IDsOfInterest}
#areasOG = {ID: cv2.contourArea(cntr) for ID, cntr in cntOG.items()}

#print(f'area2 = {area}')
#print(f'areasOG = {areasOG}')
#print(f'Ravg = {Ravg}')


#[cv2.drawContours(  img,   bodyCntrs, cid, cyclicColor(i), -1) for i,cid in enumerate(IDsOfInterest)]
#[cv2.circle(img, refCentroid, int(Ravg[ID]), cyclicColor(i), 2) for i,ID in enumerate(IDsOfInterest)]
#cv2.imshow('a',img)

refCentroid = np.array([1061.78233365,  491.36451101]).astype(np.int16)
imgGrayFinal = cv2.cvtColor(img.copy()*0, cv2.COLOR_BGR2GRAY)
def rescaleTo255(rmin,rmax,x):
    return int(255*(rmin-x)/(rmin-rmax))
from matplotlib import pyplot as plt
#for ID in IDsOfInterest:#IDsOfInterest
#    imgGray = cv2.cvtColor(img.copy()*0, cv2.COLOR_BGR2GRAY)
#    x,y,w,h = cv2.boundingRect(bodyCntrs[ID])
#    subSubMask = np.zeros((h,w),np.uint8)
#    cv2.drawContours( subSubMask, bodyCntrs, ID, 255, -1, offset = (-x,-y))
#    #np.transpose(np.array([np.arctan2(y,x),numpy.linalg.norm([x,y], axis=0)]))
#    xs, ys = np.meshgrid(np.arange(x,x+w,1), np.arange(y,y+h,1), sparse=True) # all x,y pairs. hopefully its faster using meshgrid + numpy
#    #xs, ys = np.meshgrid(x, y, sparse=True)
#    zs = np.sqrt((xs-refCentroid[0])**2 + (ys-refCentroid[1])**2).astype(int)
#    rmin, rmax = np.min(zs), np.max(zs)
#    dic ={rad:0 for rad in np.arange(rmin,rmax+1,1)}                              # all possible radiusses
#    imgGray[y:y+h,x:x+w] = subSubMask
#    imgGray2 = imgGray.copy()
#    #print( rescaleTo255(0,10,0),rescaleTo255(0,10,10))
#    for i,xses in enumerate(xs[0]):
#        for j,yses in enumerate(ys):
#            #ptClr = imgGray[yses[0],xses] 
#            if imgGray[yses[0],xses] == 255:
#                radi = zs[j][i]
#                dic[radi] += 1
#                #clr = rescaleTo255(rmin,rmax,radi)
#                #imgGray[yses[0],xses] = clr
    
#    xvals = np.array(list(dic.keys()))
#    weights = np.array(list(dic.values()))
#    avg = np.average(xvals, weights=weights)
#    stdev = np.sqrt(numpy.average((xvals-avg)**2, weights=weights))
#    dmin,dmax = min(dic.values()),max(dic.values())
#    if ID == 15:
#        plt.plot(dic.keys(),dic.values())
#        plt.xlabel('radius')
#        plt.ylabel('# pixels')
#        plt.vlines(avg, min(dic.values()), max(dic.values()), linestyles ="dashed", colors ="k")
#        plt.vlines(avg-stdev, dmin,dmax, linestyles ="dashed", colors ="b")
#        plt.vlines(avg+stdev, dmin,dmax, linestyles ="dashed", colors ="b")
    
    
#    for i,xses in enumerate(xs[0]):
#        for j,yses in enumerate(ys):
#            if imgGray[yses[0],xses] == 255:
#                radi = zs[j][i]
#                clr = rescaleTo255(dmin,dmax,dic[radi])
#                imgGrayFinal[yses[0],xses] = clr
#    #for ypos in yy:
#    #    imgGray[yy,x] = 128
#cv2.imshow('a',imgGrayFinal)
#plt.show()
##print('asdasd',rescaleTo255(27,86,64))

def findMajorInterval(x,fx,cover_area,uniformStep,debug):
    if uniformStep == 1:
        ddx = x[1]-x[0]                 # uniform step
        fx_c = np.cumsum(fx)*ddx        # stacking object heights, then multipying by width.
    else:
        dx = np.diff(x)
        dx = np.append(dx,dx[-1])
        fx_c2 = np.multiply(fx,dx)      # calculating each object area
        fx_c = np.cumsum(fx_c2)         # then adding

    fx_c = fx_c/fx_c[-1]                # normalize to 0- 1
    fx_c = np.concatenate(([0],fx_c))   # first entry 0 area, bit of an offset.
    x_right_max_index = np.argmax(fx_c >= (1-cover_area))-1   # at which x cum_sum reaches (1-cover_area), so next x-ses wont cover remaining cover_area. 
    # i have to reduce x_right_max_index because of indexing problems.
    #print(f'cumulative area at x = {x[x_right_max_index] } is {fx_c[x_right_max_index]} and x-1 = {x[x_right_max_index-1]} is {fx_c[x_right_max_index-1]} and x+1 = {x[x_right_max_index+1]} is {fx_c[x_right_max_index+1]}')
    solsIntevals2 = np.zeros(x_right_max_index)
    solsAreas2 = np.zeros(x_right_max_index)
    #print(f'cover Area %: {cover_area:.2f}')
    for i in range(0,x_right_max_index,1): 
        x_left              = x[i]                                  # area betwen x[i] and x[i+n] is (fx_c[i+n] - fx_c[i])
        targetArea          = cover_area + fx_c[i]                  # fx_c[i] is staggered to the left. so x[i = 0] has area fx_c[i=0] of zero.
        tarIndex            = np.argmin(np.abs(fx_c - targetArea))  # considers target value closest to target, from both top and bottom. top- wider interval. might not be best soln
        x_right             = x[tarIndex]
        solsIntevals2[i]    = np.round(x_right - x_left,3)          # precision oscillations can mess with min max, thus rounding.
        solsAreas2[i]       = np.round(np.abs(cover_area-(fx_c[tarIndex]-fx_c[i])),5) # can be relative dA/A0, but all A0 same for all.
        if debug == 1:
            print(str(i).zfill(2)+f', from: {x[i]:.2f}, to: {x[tarIndex]:.2f}, x_diff: {(x[tarIndex]-x[i]):.2f}, diff: {(fx_c[tarIndex]-fx_c[i]):.3f}, diff -1: {(fx_c[tarIndex-1]-fx_c[i]):.3f}, diff+1: {(fx_c[tarIndex+1]-fx_c[i]):.3f}')
            print(f'cA: {fx_c[i]:.3f}, tarArea: {targetArea:.3f}, existingArea: {fx_c[tarIndex]:.3f}, solAreas: {solsAreas2[i]}')
    
    min1Pos     = np.argmin(solsIntevals2);min1Val = solsIntevals2[min1Pos] # take the shortest interval [x[i],x[i+n]] that has area close to cover_area
    where1      = np.argwhere(solsIntevals2 == min1Val).flatten()           # multiple intervals of this length can be recovered (due to discrete distribution)
                                                                            # 
    min2Pos     = np.argmin(solsAreas2[where1])                             # search in subset of IDs, solution is subset ID
    min2PosG    = where1[min2Pos]                                           # refine with respect of subset.
    if debug == 1:
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=False)
        axes[1].plot(x,fx)
        #axes[1].fill_between(x,fx,0,where=(x>=minKey2) & (x<=minKey2+solsIntevals[minKey2]),color='b')
        axes[1].fill_between(x,fx,0,where=(x>=x[min2PosG]) & (x<=x[min2PosG]+solsIntevals2[min2PosG]),color='g')
        axes[1].set_xlabel('radius, pix')
        axes[1].set_ylabel('density')
        axes[1].set_xticks(x)
        plt.show()
    return x[min2PosG],solsIntevals2[min2PosG]

if 12 == 122:
    sigma  = 1.5;sigma2  = 0.8;
    step = 0.2
    cover_area = 0.8
    end = 10
    x = np.around(np.arange(0,end,step),4)
    #x = np.around(np.arange(2,8,step),4)
    #x = np.around(np.concatenate((np.arange(0,end/4,step/2),np.arange(end/4,3/4*end,step/4),np.arange(3/4*end,end,step/2))),4)
    #fx = np.mean(x)
    #fx = np.around(1/sigma/np.sqrt(2*np.pi)*np.exp(-1/2/sigma**2*(x-x_mean)**2)+1/sigma2/np.sqrt(2*np.pi)*np.exp(-1/2/sigma2**2*(x-0.55*x_mean)**2),4) #
    #fx = np.piecewise(x, [x < 5, ((x >= 5) & (x <= 12)), x > 12], [0, 1, 0])
    #fx = np.full(len(x),1)
    fx = np.sqrt((end/2)**2-(x- end/2)**2)
    fx = np.piecewise(x, [x < end/2, x >= end/2], [lambda x: x, lambda x: -x+end])
    #fx = x
    #fx = -(x-end/2)**2+25
    findMajorInterval(x,fx,cover_area,uniformStep =1,debug= 1)

def radialStats(bodyCntrs,IDsOfInterest,refCentroid,img, oneContour = True,debug = 0):

    output = {ID:[0,0] for ID in IDsOfInterest}                                             # future return dict {ID:[avg_r,stdev_r]}
    imgGray = cv2.cvtColor(img.copy()*0, cv2.COLOR_BGR2GRAY) if debug > 0 else 1

    for ID in IDsOfInterest:
        area0           = cv2.contourArea(bodyCntrs[ID])
        x,y,w,h         = cv2.boundingRect(bodyCntrs[ID])
        xs, ys          = np.meshgrid(np.arange(x,x+w,1), np.arange(y,y+h,1), sparse=True)  # all x,y pairs. hopefully its faster using meshgrid + numpy
        zs              = np.sqrt((xs-refCentroid[0])**2 + (ys-refCentroid[1])**2).astype(int)  # calculate L2 norms from reference centroid.
        rmin, rmax      = np.min(zs), np.max(zs)
        
        if oneContour == True:                                                              # count all pixel at radius 'rad'-> {r1:n1, r2:n2, ...}
            dic = {rad:0 for rad in np.arange(rmin,rmax+1,1)}                               # if only one pixel, 'rad' should be continous, exept casting to int can disrupt it.
        else:
            dic = {rad:0 for rad in np.sort(np.unique(np.flatten(zs)))}                     # order (sort) should not be important if not drawing a continious relation [r1,n1],[r2,n2],...
        
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
        avg             = np.average(xvals, weights=weights)                                 # weighted average
        stdev           = np.sqrt(numpy.average((xvals-avg)**2, weights=weights))            # stdev of weighted data. theres no ready function in numpy.
        a,b             = findMajorInterval(xvals,weights,0.7,uniformStep =1,debug= 0)      ;print(a,b)
        dmin, dmax      = min(dic.values()),max(dic.values())
        output[ID]      = [avg, stdev]
        if debug == 1:
            plt.plot(xvals,weights, label=f'Radial pixel distribution ID:{ID}')
            plt.vlines(avg, min(dic.values()), max(dic.values()), linestyles ="dashed", colors ="k")
            #plt.vlines(avg-stdev, dmin,dmax, linestyles ="dashed", colors ="b")
            #plt.vlines(avg+stdev, dmin,dmax, linestyles ="dashed", colors ="b")
            plt.fill_between(xvals,weights,0,where= (xvals<=a+b) & (xvals>=a))
            plt.xlabel('radius, pix')
            plt.ylabel('sum of pixels')
        
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
        cv2.imshow('a',imgGray)
        plt.show()
    return output
ids = np.array(IDsOfInterest)[[1,3,4]] ;print(ids)
aa = radialStats(bodyCntrs,ids,refCentroid,img, oneContour = True,debug = 1)
#print(aa)
k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()
