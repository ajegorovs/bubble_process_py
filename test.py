# openCV contour rotation and scale about either center of mass of arbitrary point
# https://math.stackexchange.com/questions/3245481/rotate-and-scale-A-point-around-different-origins
# i wanted to scale up A contour w/o having to use images and morphological dilate. 

import numpy as np
import numpy
import cv2
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
from scipy import interpolate
from scipy.interpolate import BSpline
traj0 = {0: (728, 624), 1: (717, 625), 2: (704, 626), 3: (691, 627), 4: (677, 627), 5: (662, 625)}
traj0 = {0: (222, 565), 1: (212, 559), 2: (201, 555), 3: (190, 549), 4: (179, 541), 5: (169, 535)}

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
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
        
#plt.plot(x2, y2, 'o', x, y, 'o')
#plt.plot(t, x, 'o', t1, fx2,'o')
#plt.plot(fx2, fy2, 'o',label='interp1d')
#plt.plot( x, y, '-.',label = 'orig')
#plt.legend()

#plt.show()

showDistDecay = True    
def distStatPrediction(trajectory, startAmp0 = 20, expAmp = 20, halfLife = 1, numsigmas = 2, plot=0, extraStr = '',mode = 1):
    global showDistDecay
    print(f'trajectory:{trajectory}')
    predictPoints = []
    distCheck = 0
    interpOrder = ['zero', 'slinear']#['zero', 'slinear', 'quadratic', 'cubic']
    if mode  ==  1:
        numStepsFor = list(range(len(trajectory)))#;print(numSteps2)
    else:
        numStepsFor = [len(trajectory) - 1]
    print(f'numStepsFor:{numStepsFor}')
    numPlots = numStepsFor[-1]+1 +1
    fig, axes = plt.subplots(1,numPlots , figsize=( numPlots*5,5), sharex=True, sharey=True)
    for numSteps in numStepsFor:

        interpKind = interpOrder[numSteps] if numSteps < len(interpOrder) else interpOrder[-1]
        print(f'interpKind: {interpKind}')
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
        
        x = np.array([a[0] for a in trajectory[:numSteps+1]])
        y = np.array([a[1] for a in trajectory[:numSteps+1]])
        t = np.arange(0,len(x),1)
        
        if numSteps == 0:
            interp_fill_value_x = 0
            interp_fill_value_y = trajectory[-1][0]
            plusTime = t[-1]
        else:
            interp_fill_value_x = "extrapolate"
            interp_fill_value_y = "extrapolate"
            plusTime  = t[-1]+1
        
        t1 = np.arange(0,plusTime+0.00001,1);print(t1)
        fsx2 = interpolate.interp1d(t, x, kind = interpKind, fill_value=interp_fill_value_x)
        fsy2 = interpolate.interp1d(t, y, kind = interpKind, fill_value=interp_fill_value_y)
        fx2 = fsx2(t1)
        fy2 = fsy2(t1)
        predictPoints.append([fsx2(plusTime),fsy2(plusTime)])
        if numSteps > 1:
            #t, c, k = interpolate.splrep(x, y, s=0, k=1)
            spline, _ = interpolate.splprep([x, y], u=t, s=0,k=2)
            #spline = interpolate.BSpline(t, c, k, extrapolate=False)
            #x_i, y_i = interpolate.splev(t, spline)
            #axes[numSteps].plot(x_i, y_i, '-o')
            new_points = interpolate.splev(t1, spline,ext=0)
            axes[numSteps].plot(new_points[0],new_points[1], '-o')
        
        #axes[numSteps].plot(fx2, fy2, '-',linewidth=0.5)
        #axes[numSteps].plot(fx2[-2:], fy2[-2:], '-o',ms=5)
        axes[numSteps].plot(x, y, 'o')
        #axes[numSteps].set_aspect('equal', 'box')
    predictPoints = np.array(predictPoints).reshape(-1,2)
    axes[numSteps+1].plot(predictPoints[:,0], predictPoints[:,1], '-.')
    axes[numSteps+1].plot(x, y, 'o')
    plt.show()
    return distCheck
distStatPrediction(list(traj0.values()), startAmp0 = 20, expAmp = 20, halfLife = 1, numsigmas = 2, plot=22, extraStr = '')
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