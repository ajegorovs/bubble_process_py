
# =========================
# get thinnest domain slice that holds cover_area *100% of the data.

uniformStep = 1;
import numpy as np
from matplotlib import pyplot as plt
sigma  = 1.5;
sigma2  = 0.8;
step = 0.2
cover_area = 0.8
end = 10
x = np.around(np.arange(0,end,step),4)
#x = np.around(np.arange(2,8,step),4)
#x = np.around(np.concatenate((np.arange(0,end/4,step/2),np.arange(end/4,3/4*end,step/4),np.arange(3/4*end,end,step/2))),4)

#x_mean = np.mean(x)
#fx = np.around(1/sigma/np.sqrt(2*np.pi)*np.exp(-1/2/sigma**2*(x-x_mean)**2)+1/sigma2/np.sqrt(2*np.pi)*np.exp(-1/2/sigma2**2*(x-0.55*x_mean)**2),4) #
#fx = np.piecewise(x, [x < 5, ((x >= 5) & (x <= 12)), x > 12], [0, 1, 0])
#fx = np.full(len(x),1)
fx = np.sqrt((end/2)**2-(x- end/2)**2)
fx = np.piecewise(x, [x < end/2, x >= end/2], [lambda x: x, lambda x: -x+end])
#fx = x
#fx = -(x-end/2)**2+25
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
findMajorInterval(x,fx,cover_area,uniformStep =1,debug= 0)
1+1
print(1)
#axes[0].vlines(0, min(fx), max(fx), linestyles ="dashed", colors ="k")

