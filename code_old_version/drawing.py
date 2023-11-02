import cv2, itertools, networkx as nx, numpy as np
from matplotlib import pyplot as plt
def overlappingRotatedRectangles(group1Params,group2Params):
    group1IDs, group2IDs = list(group1Params.keys()),list(group2Params.keys())
    allCombs = np.unique(np.sort(np.array(list(itertools.permutations(rectParams, 2)))), axis = 0)#;print(f'allCombs,{allCombs}')
    intersectingCombs = []
    for (keyOld,keyNew) in allCombs:
        x1,y1,w1,h1 = group2Params[keyNew]
        rotatedRectangle_new = ((x1+w1/2, y1+h1/2), (w1, h1), 0)
        x2,y2,w2,h2 = group1Params[keyOld]
        rotatedRectangle_old = ((x2+w2/2, y2+h2/2), (w2, h2), 0)
        interType,_ = cv2.rotatedRectangleIntersection(rotatedRectangle_new, rotatedRectangle_old)
        if interType > 0:
            intersectingCombs.append([keyOld,keyNew]) # rough neighbors combinations
    return intersectingCombs
if 1 == -1:
    img = cv2.imread ('.\\manualMask\\frame0051.png',0)*0
    H,L = img.shape
    rectParams = {0:[0.15*L,0.4*H,200,200],1:[0.45*L,0.35*H,200,200],2:[0.75*L,0.35*H,200,200],3:[0.55*L,0.73*H,200,100],4:[0.7*L,0.60*H,100,100],5:[0.45*L,0.26*H,100,70],6:[0.52*L,0.17*H,100,70],7:[0.48*L,0.63*H,100,70],8:[0.18*L,0.65*H,100,70],9:[0.25*L,0.75*H,100,70]}
    rectParams = {i:list(map(int,a)) for i,a in rectParams.items()}; print(rectParams)
    rectParamsVert = {ID:[x,0,w,H] for ID,[x,y,w,h] in rectParams.items()}
    [cv2.rectangle(img, (x,y), (x+w,y+h), 128, -1) for [x,y,w,h] in rectParamsVert.values()]
    [cv2.rectangle(img, (x,y), (x+w,y+h), 255, -1) for [x,y,w,h] in rectParams.values()]
    #cv2.imshow('a', img)

    #aa = np.unique(np.sort(np.array(list(itertools.permutations(rectParams, 2)))), axis = 0)
    #print(aa)
    aa = overlappingRotatedRectangles(rectParamsVert,rectParamsVert);print(aa)
    HG = nx.Graph()
    HG.add_nodes_from(rectParams.keys())
    HG.add_edges_from(aa)
    cntrd = {ID:tuple((x+int(0.5*w),y+int(0.5*h))) for ID,[x,y,w,h] in rectParams.items()}
    # ----- visualize  netrworkx graph with background contrours

    
    posTB = {ID:0 if y < (H - (y+h)) else 1 for ID,[x,y,w,h] in rectParams.items()};print(f'posTB:{posTB}')
    k = cv2.waitKey(0)
    if k == 27:  # close on ESC key
        cv2.destroyAllWindows()
    q = 1
    neighbors = {ID:[] for ID in rectParams}
    neighborsSameSide = {ID:[] for ID in rectParams}
    for IDs in rectParams:
        for i in HG.neighbors(IDs):
            neighbors[IDs].append(i)
            if posTB[i] == posTB[IDs]: neighborsSameSide[IDs].append(i)
    print(neighbors)
    print(neighborsSameSide)
    clusters0 = [tuple(np.sort(np.array([ID] +nbrs))) for ID, nbrs in neighborsSameSide.items()];print(clusters0)
    SameSideClusters = [list(aa) for aa in set(clusters0)]
    print(f'SameSideCluster:{SameSideClusters}')
    #print(np.argsort(list(hehe.values())))
    globSideOrder = {ID:0 for ID in rectParams}
    print(f'globSideOrder   :{globSideOrder}')
    for arr in SameSideClusters:
        srt = np.argsort([cntrd[ID][1] for ID in arr]);print(srt)
        for i,elem in enumerate(arr):
            globSideOrder[elem] = srt[i]
    print(f'globSideOrder   :{globSideOrder}')
    print(f'posTB:{posTB}')

    textHeight  = 25
    textNoGo    = 10
    textSep     = 15
    q = 0
    #cv2.rectangle(img, (x,y), (x+w,y+h), 255, -1)
    for ID,[x,y,w,h] in rectParams.items():
        if posTB[ID] == 1:
            start = H - textNoGo - globSideOrder[ID]*(textHeight + textSep)
            stop = start - textHeight
            cv2.rectangle(img, (x,start), (x+w,stop), 255, 5) 
        if posTB[ID] == 0:
            start = textNoGo + globSideOrder[ID]*(textHeight + textSep)
            stop = start + textHeight
            cv2.rectangle(img, (x,start), (x+w,stop), 255, 5)

    if 1 == 1: 
        #pos = {i:getCentroidPos(inp = vec, offset = (0,0), mode=0, mask=[]) for i, vec in cntrRemaining.items()}
        for n, p in cntrd.items():
                HG.nodes[n]['pos'] = p
        plt.figure(1)
        plt.imshow(img)
        #for cntr in list(cntrRemaining.values()):
        #    [x,y] = np.array(cntr).reshape(-1,2).T
        #    plt.plot(x,y)
        #[cv2.rectangle(img, (x,y), (x+w,y+h), 255, -1) for [x,y,w,h] in rectParams.values()]
        nx.draw(HG, cntrd, with_labels=True)
    plt.show()
