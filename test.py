# openCV contour rotation and scale about either center of mass of arbitrary point
# https://math.stackexchange.com/questions/3245481/rotate-and-scale-A-point-around-different-origins
# i wanted to scale up A contour w/o having to use images and morphological dilate. 

import sqlite3
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
def extrapolate(data, maxInterpSteps = 3, maxInterpOrder = 2,zerothDisp=[],fixSharp = 0, angleLimit = 30):
    data = np.array(data).reshape(len(data),-1) #[1,2]->[[1],[2]]; [[11,12],[21,22]]-> itself
    numPointsInTraj = data.shape[0]
    numStepsInTraj = numPointsInTraj - 1 ;print(f'numPointsInTraj:{numPointsInTraj}, numStepsInTraj:{numStepsInTraj}')
    
    # start-> take last maxInterpSteps
    start = 0 if numStepsInTraj < maxInterpSteps else numStepsInTraj-maxInterpSteps
    if numStepsInTraj == 0:
        zeroth = [0]*numPointsInTraj if len(zerothDisp) == 0 else zerothDisp
        return data[:,-1]+ zeroth, []
    k = min(numStepsInTraj,maxInterpOrder)
    splitSubsetComponents = data[start:].T
    t = np.arange(0,numPointsInTraj-start,1)
    #t1 = np.arange(0,numPointsInTraj+1,1)
    if fixSharp == 1:
        failedAngleCrit = []
        for kMod in range(k,0,-1):
            tSubFull = np.arange(0,numPointsInTraj - start + 1,1)
            spline, _ = interpolate.splprep(splitSubsetComponents, u=t, s=0,k=kMod)
            extrapolatedPoints = np.array(interpolate.splev(tSubFull, spline,ext=0))
            vs = extrapolatedPoints[:,-2:] - extrapolatedPoints[:,-3:-1] #deltas are displ components
            angleDeg = (lambda x,y: np.degrees(np.arccos(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1.0, 1.0))))(*np.transpose(vs))
            if angleDeg<= angleLimit: break
            else: failedAngleCrit.append([extrapolatedPoints[:,-1],np.round(angleDeg,2),kMod])
        return extrapolatedPoints[:,-1],np.array(failedAngleCrit,dtype=object)

    spline, _ = interpolate.splprep(splitSubsetComponents, u=t, s=0,k=k)
    extrapolatedPoint = np.array(interpolate.splev(t[-1]+1, spline,ext=0))
    return extrapolatedPoint, []

    
        #returnVec = np.array([x[0]+zerothDisp[0],y[0]+zerothDisp[1]])
    #splitSubsetComponents = [np.array([a[0] for a in data[start:]])

#extrapolate([[11,21,31,41,51],[12,22,32,42,52]])
data = np.array([[0,0],[1,0],[2,0],[3,0],[3,0.7]])
pred,a = extrapolate(data,maxInterpOrder = 3,fixSharp = 1)
#a = np.array([[2, 82.14, np.array([2.46666667, 0.95      ])],[2, 82.14, np.array([3.46666667, 1.95      ])]], dtype=object)
print(f'pred: {pred}, a:{a}')
if len(a)>0:
    a2 = np.array(list(a[:,0]),dtype=np.float32);print(f'a2: {a2},\na2.T:{a2.T}')
    plt.plot(*(a2.T),'o')

plt.plot(*data.T,'o')
plt.plot(*(pred.reshape(2,-1)),'o')
plt.show()
last = [3,0.7]
sol1 = [1. , 2.8]
sol2 = [1.83333333, 2.21666667]
#solf = [3.,  1.4]
vOrig = [0,0.7]
v1 = np.array(sol1) - last
v2 = np.array(sol2) - last
ang = (lambda x,y: np.degrees(np.arccos(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1.0, 1.0))))
print(ang(vOrig,v1))
print(ang(vOrig,v2))

#extrapolate([1,2])
#extrapolate([[1],[1]], zerothDisp = [1,1])
#def distStatPredictionVect2(trajectory, sigmasDeltas = [],sigmasDeltasHist = [], numdeltas = 5, maxInterpSteps = 3, maxInterpOrder = 2, debug = 0, savePath = r'./', predictvec_old = [], bubID = 1, timestep = 0, zerothDisp = [0,0]):
#    returnVec = []
#    numPointsInTraj = len(trajectory)
#    numStepsInTraj = numPointsInTraj - 1 
#    start = 0 if numStepsInTraj < maxInterpSteps else numStepsInTraj-maxInterpSteps
#    x = np.array([a[0] for a in trajectory[start:]])
#    y = np.array([a[1] for a in trajectory[start:]])
#    t = np.arange(0,len(x),1)
#    t1 = np.arange(0,len(x)+1,1)
#    if debug == 1:
#        fig, axes = plt.subplots(1,2 , figsize=( 13,5), sharex=False, sharey=False)
#        axes[0].plot(x, y, '-o',c='green', label = 'traj')
            
#    if numStepsInTraj == 0:
#        returnVec = np.array([x[0]+zerothDisp[0],y[0]+zerothDisp[1]])
#        if debug == 1:
#            axes[0].plot([x[0],x[0]+zerothDisp[0]], [y[0],y[0]+zerothDisp[1]], '--o', label = 'forecast', ms= 3)
#            axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
                
#    if numStepsInTraj > 0:
#        k = min(numStepsInTraj,maxInterpOrder)
#        # reduce iterp order if prediction does sharp turns
#        (numPointsInTraj-np.sqrt(2*numPointsInTraj),numPointsInTraj+np.sqrt(2*numPointsInTraj))
#        for kMod in range(k,0,-1): # smoothing did not help on 3 point data, k=2 extepolation gives a 120 degree turn on one occasion
#       # maxSmoothing = numPointsInTraj+np.sqrt(2*numPointsInTraj)
#        #for sMod in np.arange(0,maxSmoothing+0.0001,maxSmoothing/5):
#            sMod = 0
#            spline, _ = interpolate.splprep([x, y], u=t, s=sMod,k=kMod)
#            new_points = np.array(interpolate.splev(t1, spline,ext=0))
#            #v1,v2 = zip(new_points[0][-2:],new_points[1][-2:])
#            vs = new_points[:,-2:] - new_points[:,-3:-1] #deltas are displ components
#            angleDeg = (lambda x,y: np.degrees(np.arccos(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1.0, 1.0))))(*np.transpose(vs))
#            print(f' interpOrder = {k},smoothing = {sMod}, angle = {angleDeg}') if debug == 1 else 0
            
#            if angleDeg<= 30: break
#            else:
#                if debug == 1: axes[0].plot(new_points[0],new_points[1], '-o', label = f'forecast (failed): s:{sMod:0.1f},k:{kMod:0.1f}', ms= 2,linewidth = 1)
#        if debug == 1:
#            axes[0].plot(new_points[0,-2:],new_points[1,-2:], '--o', label = 'forecast', ms= 3)
#            axes[0].plot([x[-2],predictvec_old[0]],[y[-2],predictvec_old[1]], '--o', label = 'prev forecast')
#            if len(sigmasDeltas)>0:
#                circleMean = Circle(tuple(predictvec_old), sigmasDeltas[0] , alpha=0.1) 
#                circleNStd = Circle(tuple(predictvec_old), sigmasDeltas[1]*numdeltas , alpha=0.1,fill=False) 
#                axes[0].add_patch(circleMean)
#                axes[0].add_patch(circleNStd)
#                axes[0].text(*predictvec_old, s = f'm: {sigmasDeltas[0]:0.2f}, s:{sigmasDeltas[1]:0.1f}')
#                axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
#        returnVec = np.array([new_points[0,-1],new_points[1,-1]])

#    if debug == 1:
#        if len(sigmasDeltasHist)>0:
#            sigmasDeltasHistNP = np.array([v[1:] for v in sigmasDeltasHist.values()])
#            timeSteps = list(sigmasDeltasHist.keys())
#            axes[1].plot(timeSteps,sigmasDeltasHistNP[:,0], '--o', label = 'vals')
#            axes[1].plot(timeSteps,sigmasDeltasHistNP[:,1], '--o', label = 'running mean')
#            axes[1].plot(timeSteps,sigmasDeltasHistNP[:,2], '--o', label = 'running stdev')
#            axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
#            axes[1].grid()
#        #plt.legend(loc=(1.1, 0.5))
        
#        filename = os.path.join(savePath, f"ID_{str(bubID).zfill(3)}_t_{str(timestep).zfill(3)}.png")
#        plt.savefig(filename)
#        #plt.suptitle( f"ID_{str(bubID).zfill(3)}_t_{str(timestep).zfill(3)}")
#        #plt.show()
#    return returnVec