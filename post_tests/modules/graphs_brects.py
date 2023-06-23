import numpy as np, cv2, networkx as nx, itertools, pickle
from matplotlib import pyplot as plt

def sortMixed(arr):
    return sorted([i for i in arr if type(i) != str]) + sorted([i for i in arr if type(i) == str])

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
        #rotatedRectangle_new = ((int(x1+w1/2), int(y1+h1/2)), (w1, h1), 0)
        rotatedRectangle_new = (tuple(map(int,(x1+w1/2,y1+h1/2))), tuple(map(int,(w1, h1))), 0)
        x2,y2,w2,h2 = group1Params[keyOld]
        #rotatedRectangle_old = ((int(x2+w2/2), int(y2+h2/2)), (w2, h2), 0)
        rotatedRectangle_old = (tuple(map(int,(x2+w2/2,y2+h2/2))), tuple(map(int,(w2, h2))), 0)
        #with open('./asd.pickle', 'wb') as handle: 
        #    pickle.dump([rotatedRectangle_new,rotatedRectangle_old], handle) 

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
