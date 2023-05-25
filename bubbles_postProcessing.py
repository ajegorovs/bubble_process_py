

# =========================
# get thinnest domain slice that holds cover_area *100% of the data.

uniformStep = 1;
import numpy as np, os, glob, pickle, cv2, itertools, networkx as nx, re
from scipy import stats as sc_stat
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob, csv, cv2, numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
[typeFull,typeRing, typeRecoveredRing, typeElse, typeFrozen,typeRecoveredElse,typePreMerge,typeRecoveredFrozen,typeMerge] = np.array([0,1,2,3,4,5,6,7,8])
typeStrFromTypeID = {tID:tStr for tID,tStr in zip(np.array([0,1,2,3,4,5,6,7,8]),['OB','RB', 'rRB', 'DB', 'FB', 'rDB', 'pm', 'rF', 'MB'])}

def getHullCA(contours):
    hullContour = cv2.convexHull(np.vstack(contours))
    m = cv2.moments(hullContour)
    area = m['m00']
    centroids  = np.uint32([m['m10']/m['m00'], m['m01']/m['m00']])
    return  tuple(map(int,np.ceil(centroids))), int(area)
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


def sortMixed(arr):
    return sorted([i for i in arr if type(i) != str]) + sorted([i for i in arr if type(i) == str])
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
colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
np.random.seed(2);np.random.shuffle(colorList);np.random.seed()

def cyclicColor(index):
    return colorList[index % len(colorList)].tolist()

def sortDictEntry(dic,ID):
    times = sorted(list(dic[ID].keys()))
    temp = {t:dic[ID][t] for t in times}
    dic.pop(ID,None)
    dic[ID] = temp

# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =====================================================================  hi  ==============================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================

mainIntermediateDataFolder  = r'.\intermediateData'                              # these are main themed folders, sub-projects go inside.
mainDataArchiveFolder       = r'.\archives'
mainTestFolder              = r'.\test'

mainOutputSubFolders= ['Field OFF Series 7','sccm250-meanFix', "00001-05000"]    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#mainOutputSubFolders= ['HFS 200 mT Series 4','sccm150-meanFix', "00001-05000"]  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                                                 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
maxStepsForTest = 3000                                                           # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                                                 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
overWriteStart  = 0                   # -1 to start from existing max           # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
overWriteEnd    = maxStepsForTest#200                                            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


dataTestFolder          = mainTestFolder
dataArchiveFolder       = os.path.join(mainDataArchiveFolder, *mainOutputSubFolders)
csvImgDirectory         = os.path.join(dataArchiveFolder,"-".join(["csv_img"]+mainOutputSubFolders))
intermediateDataFolder  = os.path.join(mainIntermediateDataFolder, *mainOutputSubFolders)
archivePathOrig         = os.path.join(intermediateDataFolder, 'data-'+mainOutputSubFolders[-1]+'.pickle')
archivePathNoImg        = os.path.join(intermediateDataFolder, 'data-'+mainOutputSubFolders[-1]+'-noImg.pickle')
for folderName in mainOutputSubFolders:                                          # or creates heararchy of folders "subfolder/subsub/.."
    dataTestFolder  = os.path.join(dataTestFolder,  folderName)
    if not os.path.exists(dataTestFolder): os.mkdir(dataTestFolder)

createNoImgArchive = 1             # create and use archive w/o images and masks. x6 times smaller size
#if os.path.exists(archivePathNoImg):    archiveDir = archivePathNoImg
#else:                                   archiveDir = archivePathOrig
if createNoImgArchive == 1 and not os.path.isfile(archivePathNoImg):

    with open(archivePathOrig, 'rb') as handle:
                    [
                    backupStart, g_contours, g_contours_hull, g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull, g_areas_hull, g_dropIDs , allFrozenIDs,
                    g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Areas,g_Masks,g_Images,g_old_new_IDs,g_bubble_type,g_child_contours, frozenBuffer_old, STBubs,
                    g_predict_displacement, g_predict_area_hull,  g_MBub_info, frozenGlobal,  g_bublle_type_by_gc_by_type, g_areas_IDs, g_splits, activeFrozen,
                    g_merges] = pickle.load(handle)
    g_Masks,g_Images = {},{}
    with open(archivePathNoImg, 'wb') as handle:
        pickle.dump(
                    [
                    backupStart, g_contours, g_contours_hull, g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull, g_areas_hull, g_dropIDs , allFrozenIDs,
                    g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Areas,g_Masks,g_Images,g_old_new_IDs,g_bubble_type,g_child_contours, frozenBuffer_old, STBubs,
                    g_predict_displacement, g_predict_area_hull,  g_MBub_info, frozenGlobal,  g_bublle_type_by_gc_by_type, g_areas_IDs, g_splits, activeFrozen,
                    g_merges]
            , handle) 
    importPath = archivePathNoImg
elif createNoImgArchive == 1 and  os.path.isfile(archivePathNoImg):
    importPath = archivePathNoImg
else:
    importPath = archivePathOrig
with open(importPath, 'rb') as handle:
    [
    backupStart, g_contours, g_contours_hull, g_FBub_rect_parms, g_FBub_centroids, g_FBub_areas_hull, g_areas_hull, g_dropIDs , allFrozenIDs,
    g_Centroids,g_Rect_parms,g_Ellipse_parms,g_Areas,g_Masks,g_Images,g_old_new_IDs,g_bubble_type,g_child_contours, frozenBuffer_old, STBubs,
    g_predict_displacement, g_predict_area_hull,  g_MBub_info, frozenGlobal,  g_bublle_type_by_gc_by_type, g_areas_IDs, g_splits, activeFrozen,
    g_merges] = pickle.load(handle)

imageLinks = glob.glob(csvImgDirectory + "**/*.png", recursive=True) 
img = np.uint8(cv2.imread(imageLinks[0],0))
fakeBoxW, fakeBoxH      = 176,int(img.shape[0]/3)                                             
fakeBox                 = {-1:[img.shape[1] - fakeBoxW + 1, fakeBoxH, fakeBoxW, fakeBoxH]}

# =========================================================================================================================================================
# ========= get times and IDs of bubbles intersecting with box, these are problematic and might be avoided ============
# =========================================================================================================================================================
dropBoxIDs = {}
for ID,t_rectParams in g_Rect_parms.items():
    t_times = sorted(list(t_rectParams.keys()))
    for t in t_times:
        ovlp = overlappingRotatedRectangles(fakeBox,{0:t_rectParams[t]})
        if len(ovlp)>0:
            if t not in dropBoxIDs: dropBoxIDs[t] = []
            dropBoxIDs[t].append(ID)
        else: break

a = 1
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================

detectedSplits = {}
# g_splits holds confirmed and unconfirmed splits, take confirmed
for time,stats in g_splits.items():
    if time > maxStepsForTest: break
    splitsTrue = [ID for ID,params in stats.items() if params[0] == True ]
    if len(splitsTrue) > 0 : detectedSplits[time] = splitsTrue              # detectedSplits holds global IDs of confirmed splits
# g_merges = timestep: {newGlobID:[oldID1, oldID2]}, merge all globals
agag = {}
for time,stats in g_splits.items():
    if time > maxStepsForTest: break
    splitsTrue = [ID for ID,params in stats.items() if params[0] == True and params[1] is not None]
    if len(splitsTrue)==0: continue
    agag[time] = splitsTrue
detectedSplits = agag




#detectedSplits = agag
mergingGlobals = {time:sum(list(stats.values()),[]) for time,stats in g_merges.items() if time <= maxStepsForTest}  # times + list of active merge globals. small-big merge inherit ID so 2 contour 
                                                                                                                    #  merge is ID1:[ID2] not IDX:[ID1,ID2] its fixed in mergingGlobalsSmallBig

# small + big merges do not create new global ID, they use biggest area ID. have to gather them properly
mergingGlobalsSmallBig0 = {time: [[ID]+ subIDs for ID,subIDs in stats.items() if g_bubble_type[ID][time] != typeMerge] for time,stats in g_merges.items() if time <= maxStepsForTest}
mergingGlobalsSmallBig = {time:sum(IDs,[]) for time,IDs in mergingGlobalsSmallBig0.items() if len(IDs)>0}
# reformat into into oldID1:[oldID1,oldID2] instead of oldID1:[oldID2]
for time,subIDs in mergingGlobalsSmallBig.items():
    dics = g_merges[time].copy()
    for ID in subIDs:
        if ID in dics:
            new = sum([[ID],dics[ID]],[]) # flatten
            g_merges[time].pop(ID,None)
            g_merges[time][ID] = new
   
for t in g_merges:
    for i in g_merges[t]:
        g_merges[t][i] = sorted(g_merges[t][i])

# ==== these globals split at these times e.g 53: [999, 1000, 1051]
g0 = sorted(list(set(sum(list(agag.values()),[]))))                                 # isolate globals
resortSpit = {i:[key for key,vals in agag.items() if i in vals] for i in g0}        # global: [times] to see if there is one with close frames e.g t = 10:[T151,T152]

agagMerge = {int(i):list(dic.keys()) for i,dic in g_merges.items()}                      # time: globals of merges
g1 = sorted(set(sum(list(agagMerge.values()),[])))                                  # isolate globals

# ==== these globals merge at these times e.g 53: [1000, 1001, 1052]
resortMerge = {int(i):[key for key,vals in agagMerge.items() if i in vals] for i in g1}  # global: times

allSplitMergeGIDs = sorted(set(list(resortMerge.keys()) + list(resortSpit.keys())))
a = 1
resortTogether = {}
for i in allSplitMergeGIDs:
    if i not in resortTogether: resortTogether[i] = []
    if i in resortSpit:     resortTogether[i] += resortSpit[i]
    if i in resortMerge:    resortTogether[i] += resortMerge[i]
    resortTogether[int(i)] = sorted(resortTogether[i] )                                  # gather times of split/merge events for every main ID
# ==== these globals have split and merge events on these times
resortTogetherFailTest = {int(i):vals for i, vals in resortTogether.items()}
resortTogetherFailTest2 = {i:np.diff(vals) for i, vals in resortTogetherFailTest.items() if len(np.diff(vals))>0} 
resortTogetherFail = {int(i):sum(np.where(np.diff(vals)>=0,1,0)) for i, vals in resortTogether.items()} 
resortTogetherFailIDs = [int(i) for i,val in resortTogetherFail.items() if val >= 2]
a = 1

for t,v in mergingGlobalsSmallBig.items():            # fix mergingGlobals to include small IDs from small-big merge
    if t not in mergingGlobals: mergingGlobals[t] = v
    else: mergingGlobals[t] =  list(np.unique(mergingGlobals[t] + v))
   
allSplitGlobals     = sorted(list(map(int,set(sum(list(detectedSplits.values()),[])))))
allMergedGlobals    = sorted(list(map(int,set(sum(list(mergingGlobals.values()),[])))))

splitMergeCulprits = sorted(list(set([ID for ID in allSplitGlobals if ID in allMergedGlobals]))) # global IDs that both in split and in merge
splitMergeCulpritsCompanions0 = [{time:group for time,group in mergingGlobals.items() if ID in group} for ID in splitMergeCulprits]

splitMergeCulpritsCompanions = {};
for dic in splitMergeCulpritsCompanions0:
    for k1,v1 in dic.items():
        splitMergeCulpritsCompanions[k1] = sorted(v1)

splitMergeCulpritsCompanions2 ={}
for t,subIDs in splitMergeCulpritsCompanions.items():
    a = [i for i in subIDs if i in resortTogetherFailIDs]
    if len(a) == 0: splitMergeCulpritsCompanions2[t] = subIDs

splitMergeCulpritsCompanions = splitMergeCulpritsCompanions2


# ==================================================== DETECT SPLIT TO MERGE ==========================================================
# ============================================ MAIN ID SPLITS == ITS IN g_splits ======================================================
# =====================================================================================================================================

allSplitGlobals     = sorted(list(map(int,set(sum(list(detectedSplits.values()),[])))))
allMergedGlobals    = sorted(list(map(int,set(sum(list(mergingGlobals.values()),[])))))

splitMergeCulprits = sorted(list(set([ID for ID in allSplitGlobals if ID in allMergedGlobals]))) # IDs that have both split and merge
splitMergeBuffer = {}

for i in splitMergeCulprits:                                             

    splitTimes = [t for t, IDS in detectedSplits.items() if i in IDS]   # accounting for multiple events -> [s1,s2,m1,s3,m2,m3] 
    mergeTimes = [t for t, IDS in mergingGlobals.items() if i in IDS]   # should form [[s2,m1],[s3,m2]] as split-merge culprits
    
    for st in splitTimes:
        m = 0
        for mt in mergeTimes:
            m = mt
            if mt > st:
                if (st in dropBoxIDs and i in dropBoxIDs[st]) or (mt in dropBoxIDs and i in dropBoxIDs[mt]):
                    continue                                            # skip if events take place in restricted box
                if i not in splitMergeBuffer: splitMergeBuffer[i] = []
                splitMergeBuffer[i].append([st,m])
                break

# == drop large time diffs between split-merge
splitMergeBuffer2 = {} #splitMergeBuffer[-1] = [[1,50]] # test drop, works

for i, vals in splitMergeBuffer.items():
    for pair in vals:
        if np.diff(pair)[0] < 8:
            if i not in splitMergeBuffer2: splitMergeBuffer2[i] = []
            splitMergeBuffer2[i].append(pair)

indexedDSplitMergeBufferIDs     = {i:vals for i,vals in enumerate(splitMergeBuffer2.keys())}
indexedDSplitMergeBufferTimes   = {i:vals for i,vals in enumerate(splitMergeBuffer2.values())}

fixSplitMerge = []
stage1RemovedGIDs = []
removedSplitMergeIDs = []
inheritIDs = {}

for i in tqdm(range(len(splitMergeBuffer2))):
    splitID = indexedDSplitMergeBufferIDs[i]
    for [splitTime,mergeTime] in indexedDSplitMergeBufferTimes[i]:
        swappedIDCase = False
        key = [mergeID for mergeID,subIDs in g_merges[mergeTime].items() if splitID in subIDs][0]
        culpritPair = g_merges[mergeTime][key]
        times = {subID: min(list(g_areas_hull[subID].keys())) for subID in culpritPair} # grab first time steps of culprits
        latestTime = max(list(times.values()))                                          # take one most recent (zero history)
        leadingID   = min(times,key=times.get)                                          # is present earlier
        s = [ j for j in culpritPair if j in g_merges[mergeTime].keys()]                # check if its bit-small merge
        if len(s) == 1 and leadingID != s[0]:
            swappedIDCase   = True
            leadingID       = s[0]
        leadingLatestAreas = [area for time,area in g_areas_hull[leadingID].items() if latestTime - 3 <= time < latestTime]
        fillTimes = np.arange(latestTime,mergeTime,1) # lost times between split and merge
        fillAreas,fillSubIDs,fillCentroids = [], [], []
        for time in fillTimes:                                                         # iterate through lost times 
            subIDs = sum([g_old_new_IDs[IDs][time] for IDs in culpritPair],[])         # get local contor IDs
            centroid, area = getHullCA(g_contours[time][subIDs])                       # get area and centroid for each lost timestep
            #area = cv2.contourArea(cv2.convexHull(np.vstack(g_contours[time][subIDs])))
            fillAreas.append(area)
            fillCentroids.append(centroid)
            fillSubIDs.append(subIDs)
        mergeSubIDs                     = g_old_new_IDs[key][mergeTime]                                    # merged global ID
        if swappedIDCase == True:   leadingID2 = [i for i in culpritPair if i != leadingID][0]      # other bubble has history.
        else:                       leadingID2 = leadingID
        predict_area_stats              = g_predict_area_hull[leadingID2][latestTime-1]              # hull hitory pre split
        predict_centroid                = g_predict_displacement[leadingID2][latestTime][0]          # 0 holds predict centroid
        predict_centroid_mean_std       = g_predict_displacement[leadingID2][latestTime-1][2:]       # others hold stats
        preSplitCentroid                = g_Centroids[leadingID2][latestTime-1]                      # previous to split

        centroidMerge, areaMerge        = getHullCA(g_contours[mergeTime][mergeSubIDs])
        meanMissingArea = np.mean(fillAreas)
    
        distPredictAndLost  =  np.linalg.norm(np.array(predict_centroid) - np.array(fillCentroids[0])).astype(int)
    
        diffStartEnd        = np.array(centroidMerge) - np.array(preSplitCentroid)                 # full length without missing time
        fullCentroidPath    = [preSplitCentroid] + fillCentroids + [centroidMerge]                 # include missing
        diffs               = np.diff(np.array(fullCentroidPath),axis = 0)                         # include displacements
        diffsTest           = diffStartEnd/(len(fillCentroids)+1)                                  # estime displ from full length
        diffsDiffs          = [np.linalg.norm(np.array(d) - np.array(diffsTest)) for d in diffs]   # check length of diff estimate vs include
        passShortEvent  = True if len(fillTimes) <= 8 else False
        passPredC   = True if distPredictAndLost < predict_centroid_mean_std[0] + 3* predict_centroid_mean_std[1] else False       # just [mean,std]
        passDisplC  = True if all([1 if a < 15 else 0 for a in diffsDiffs]) else False              # check if differenec lenghts are reasonable
        relArea = np.abs(meanMissingArea-areaMerge)/meanMissingArea
        passArea    = True if relArea < ( 5*predict_area_stats[2]/predict_area_stats[1] ) else False             # mean area + stdev
        #passArea    = True if meanMissingArea < (predict_area_stats[1] + 3*predict_area_stats[2]) else False             # mean area + stdev
        if passShortEvent and passPredC and passDisplC and passArea:
            print('\n')
    
            fixSplitMerge.append([leadingID,latestTime,key])
            lastErrors  = [a for t,[_,a,_,_] in g_predict_displacement[leadingID2].items() if latestTime  - 4 < t  <= latestTime - 1 ] + distPredictAndLost
            meanError   = max(int(np.mean(lastErrors)),3)
            stdError    = max(int(np.std(lastErrors)),1)
            g_predict_displacement[leadingID][latestTime] = [predict_centroid,distPredictAndLost,meanError,stdError]

            if swappedIDCase == True:                        # other bubble holds history, copy it to main bubble.
                print(f'{latestTime}: swapped IDs case')
                startTIme   = times[leadingID2]
                endTime     = times[leadingID]
                inheritTimes = np.arange(startTIme,endTime,1)
                for t in inheritTimes:
                    g_old_new_IDs[leadingID][t]             = g_old_new_IDs[leadingID2][t]         
                    g_Centroids[leadingID][t]               = g_Centroids[leadingID2][t]           
                    g_areas_hull[leadingID][t]              = g_areas_hull[leadingID2][t]          
                    g_bubble_type[leadingID][t]             = g_bubble_type[leadingID2][t]         
                    g_Rect_parms[leadingID][t]              = g_Rect_parms[leadingID2][t]          
                    g_Ellipse_parms[leadingID][t]           = g_Ellipse_parms[leadingID2][t]       
                    g_predict_displacement[leadingID][t]    = g_predict_displacement[leadingID2][t]
                    g_predict_area_hull[leadingID][t]       = g_predict_area_hull[leadingID2][t]
                    g_contours_hull[leadingID][t]           = g_contours_hull[leadingID2][t]
                
                    sortDictEntry(g_old_new_IDs,            leadingID)
                    sortDictEntry(g_Centroids,              leadingID)
                    sortDictEntry(g_areas_hull,             leadingID)
                    sortDictEntry(g_bubble_type,            leadingID)
                    sortDictEntry(g_Rect_parms,             leadingID)
                    sortDictEntry(g_Ellipse_parms,          leadingID)
                    sortDictEntry(g_predict_displacement,   leadingID)
                    sortDictEntry(g_predict_area_hull,      leadingID)
        
            for i,time in enumerate(fillTimes):
                g_old_new_IDs[leadingID][time]      = fillSubIDs[i]
                g_areas_hull[leadingID][time]       = fillAreas[i]
                g_Centroids[leadingID][time]        = fillCentroids[i]
                g_contours_hull[leadingID][time]    = cv2.convexHull(np.vstack(g_contours[time][fillSubIDs[i]]))
            

            falseMergeTimesAll = [t for t in  list(g_areas_hull[key].keys()) if t >= mergeTime] # steal false new merge IDs history and give it to main ID
            if swappedIDCase == False:                                                          # looks like swapped case main bubble holds post merge history
                for i,time in enumerate(falseMergeTimesAll):                                    # just copy
                    subIDs                          = g_old_new_IDs[key][time]
                    centroid, area                  = getHullCA(g_contours[time][subIDs]) 
                    g_old_new_IDs[leadingID][time]  = subIDs
                    g_areas_hull[leadingID][time]   = area
                    g_Centroids[leadingID][time]    = centroid   
                    g_contours_hull[leadingID][time]= cv2.convexHull(np.vstack(g_contours[time][subIDs]))

            allTimeSteps = sorted(list(np.unique(list(fillTimes) + falseMergeTimesAll)))       # go though whole history of dropped new merge ID
            for time in allTimeSteps:
                lastCentroids   = np.array([c for t,c in g_Centroids[leadingID].items() if time - 3 < t <= time ])
                centroid        = g_Centroids[leadingID][time]
                hullA           = g_areas_hull[leadingID][time]
                a = 1
                slope           = sc_stat.linregress(*lastCentroids.T)[0]
                stepLen         = np.linalg.norm(diffsTest)
                vector          =  stepLen * np.array([1, slope]) / np.linalg.norm([1, slope])              # build a vector from slope
                vector          = np.sign(np.dot(diffsTest,np.array([1, slope]))) * vector                  # change sign if needed
                predictVector   = tuple(map(int,(vector + centroid)))
                predictedOld    = g_predict_displacement[leadingID][time][0]

                distPredictOld  = np.linalg.norm(np.array(predictedOld) - np.array(centroid)).astype(int)
                lastErrors  = [a for t,[_,a,_,_] in g_predict_displacement[leadingID].items() if time - 3 < t <= time - 1 ] + distPredictOld
                meanError   = max(int(np.mean(lastErrors)),3)
                stdError    = max(int(np.std(lastErrors)),1)
            
                if time+1 not in g_predict_displacement[leadingID]: g_predict_displacement[leadingID][time + 1] = []
                g_predict_displacement[leadingID][time+1] = [predictVector,distPredictOld,meanError,stdError]

                lastErrorsA  = [a for t,a in g_areas_hull[leadingID].items() if time  - 4 < t  <= time - 1 ] 
                meanErrorA   = int(np.mean(lastErrorsA))
                stdErrorA    = max(int(np.std(lastErrorsA)),int(0.1*meanErrorA))
                if time not in g_predict_area_hull[leadingID]: g_predict_area_hull[leadingID][time] = []
                g_predict_area_hull[leadingID][time] = [hullA,meanErrorA,stdErrorA]
                if time not in g_bubble_type[leadingID]: g_bubble_type[leadingID][time] = typeElse
                g_bubble_type[leadingID][time] = typeElse
                if time not in g_Rect_parms[leadingID]: g_Rect_parms[leadingID][time] = []
                g_Rect_parms[leadingID][time] = cv2.boundingRect(g_contours_hull[leadingID][time])

                if leadingID != key:
                    # === drop merged status from bubble types for single frame. since i know merge time. might remove iterations. old stuff
                    if key in list(sum(g_bublle_type_by_gc_by_type[time].values(),[])):          # all active global IDs on that timeStep/frame
                        where = [ID for ID,vals in g_bublle_type_by_gc_by_type[time].items() if key in vals][0]
                        g_bublle_type_by_gc_by_type[time][where].remove(key)
                        g_bublle_type_by_gc_by_type[time][typeElse].append(leadingID)
                    if time in g_splits and key in g_splits[time]:                     #  check each step of history of dropped merge ID 
                        tempVals  = g_splits[time][key]                                # and modify g_splits to new ID
                        g_splits[time].pop(key,None)
                        g_splits[time][leadingID] = tempVals
                        
                        a = 1

            if leadingID != key:
                g_MBub_info.pop(            key,    None)
                g_old_new_IDs.pop(          key,    None)
                g_Centroids.pop(            key,    None)
                g_areas_hull.pop(           key,    None)
                g_bubble_type.pop(          key,    None)
                g_Rect_parms.pop(           key,    None)
                g_Ellipse_parms.pop(        key,    None)
                g_predict_displacement.pop( key,    None)
                g_predict_area_hull.pop(    key,    None)
                stage1RemovedGIDs.append(key)
                print(f'{mergeTime}: removing false new merge ID: {key}, keep original: {leadingID}')
            else: print(f'{mergeTime}: no new merge ID, keep old ID: {key}')
        
            if mergeTime in g_merges:
                if key in g_merges[mergeTime]:
                    if len(g_merges[mergeTime]) == 1:  g_merges.pop(mergeTime,None)
                    else: g_merges[mergeTime].pop(key,None)
        
            if latestTime in g_splits:
                if leadingID in g_splits[latestTime]:
                    if len(g_splits[latestTime]) == 1:  g_splits.pop(latestTime,None)
                    else: g_splits[latestTime].pop(leadingID,None)

            otherIDs = [ID for ID in culpritPair if ID != leadingID]                                    # kind of assume that there only one second split bubble ID
            print(f'{mergeTime}: removing false secondary IDs: {otherIDs} (merged into {leadingID})')
            for a in otherIDs:
            
                g_old_new_IDs.pop(          a,    None)
                g_Centroids.pop(            a,    None)
                g_areas_hull.pop(           a,    None)
                g_bubble_type.pop(          a,    None)
                g_Rect_parms.pop(           a,    None)
                g_Ellipse_parms.pop(        a,    None)
                g_predict_displacement.pop( a,    None)
                g_predict_area_hull.pop(    a,    None)
                stage1RemovedGIDs.append(a)
            removedSplitMergeIDs = removedSplitMergeIDs + otherIDs
            inheritIDs[key] = leadingID

            indexedDSplitMergeBufferIDs = {k:inheritIDs[ID] if ID in inheritIDs else ID for k,ID in indexedDSplitMergeBufferIDs.items()}

            whereKey = [t for t,vals in splitMergeCulpritsCompanions.items() if key in vals]
            for t in whereKey:
                splitMergeCulpritsCompanions[t].remove(key)
                splitMergeCulpritsCompanions[t].append(leadingID)
                splitMergeCulpritsCompanions[t] = sorted(splitMergeCulpritsCompanions[t])           # cheking if updating works
            whereKey2 = {t:{i:v for i,v in vals.items() if key in v} for t,vals in g_merges.items() }
            whereKey3 = {t:list(v.keys())[0] for t,v in whereKey2.items() if len(v)>0}
            for t,i in whereKey3.items():
                g_merges[t][i].remove(key)
                g_merges[t][i].append(leadingID)
                g_merges[t][i] = sorted(g_merges[t][i])
            #g_merges 

            a = 1


a = 1
#fixSplitMerge = []
#inheritIDs = {}
#removedSplitMergeIDs = []
#temp = list(splitMergeCulpritsCompanions.keys())
#stage1RemovedGIDs = []
#for i in tqdm(range(len(splitMergeCulpritsCompanions))):
#    swappedIDCase = False
#    mergeTime, culpritPair = temp[i], splitMergeCulpritsCompanions[temp[i]]

#    pos = list(g_merges[mergeTime].values()).index(culpritPair)
#    key = list(g_merges[mergeTime].keys())[pos]                                     # new global ID
#    culpritPair = [inheritIDs[a] if a in inheritIDs else a for a in culpritPair]
#    times = {subID: min(list(g_areas_hull[subID].keys())) for subID in culpritPair} # grab first time steps of culprits
#    latestTime = max(list(times.values()))                                          # take one most recent (zero history)
#    leadingID   = min(times,key=times.get)                                          # is present earlier
#    s = [ j for j in culpritPair if j in g_merges[mergeTime].keys()]                # check if its bit-small merge
#    if len(s) == 1 and leadingID != s[0]:
#        swappedIDCase   = True
#        leadingID       = s[0]                                                # in case big-small merge main can be younger, but bigger bubble
#    #xPos = {i:g_Centroids[i][mergeTime][0] for i in culpritPair}
#    #leadingIDc  = min(xPos,key=xPos.get) 
#    leadingLatestAreas = [area for time,area in g_areas_hull[leadingID].items() if latestTime - 3 <= time < latestTime]
#    fillTimes = np.arange(latestTime,mergeTime,1) # lost times between split and merge
#    fillAreas = []
#    fillSubIDs = []
#    fillCentroids = []
#    for time in fillTimes:                                                         # iterate through lost times 
#        subIDs = sum([g_old_new_IDs[IDs][time] for IDs in culpritPair],[])         # get local contor IDs
#        centroid, area = getHullCA(g_contours[time][subIDs])                       # get area and centroid for each lost timestep
#        #area = cv2.contourArea(cv2.convexHull(np.vstack(g_contours[time][subIDs])))
#        fillAreas.append(area)
#        fillCentroids.append(centroid)
#        fillSubIDs.append(subIDs)
#    mergeSubIDs                     = g_old_new_IDs[key][mergeTime]                                    # merged global ID
#    if swappedIDCase == True:   leadingID2 = [i for i in culpritPair if i != leadingID][0]      # other bubble has history.
#    else:                       leadingID2 = leadingID
#    predict_area_stats              = g_predict_area_hull[leadingID2][latestTime-1]              # hull hitory pre split
#    predict_centroid                = g_predict_displacement[leadingID2][latestTime][0]          # 0 holds predict centroid
#    predict_centroid_mean_std       = g_predict_displacement[leadingID2][latestTime-1][2:]       # others hold stats
#    preSplitCentroid                = g_Centroids[leadingID2][latestTime-1]                      # previous to split

#    centroidMerge, areaMerge        = getHullCA(g_contours[mergeTime][mergeSubIDs])
#    meanMissingArea = np.mean(fillAreas)
    
#    distPredictAndLost  =  np.linalg.norm(np.array(predict_centroid) - np.array(fillCentroids[0])).astype(int)
    
#    diffStartEnd        = np.array(centroidMerge) - np.array(preSplitCentroid)                 # full length without missing time
#    fullCentroidPath    = [preSplitCentroid] + fillCentroids + [centroidMerge]                 # include missing
#    diffs               = np.diff(np.array(fullCentroidPath),axis = 0)                         # include displacements
#    diffsTest           = diffStartEnd/(len(fillCentroids)+1)                                  # estime displ from full length
#    diffsDiffs          = [np.linalg.norm(np.array(d) - np.array(diffsTest)) for d in diffs]   # check length of diff estimate vs include
#    passShortEvent  = True if len(fillTimes) <= 8 else False
#    passPredC   = True if distPredictAndLost < predict_centroid_mean_std[0] + 3* predict_centroid_mean_std[1] else False       # just [mean,std]
#    passDisplC  = True if all([1 if a < 15 else 0 for a in diffsDiffs]) else False              # check if differenec lenghts are reasonable
#    relArea = np.abs(meanMissingArea-areaMerge)/meanMissingArea
#    passArea    = True if relArea < ( 5*predict_area_stats[2]/predict_area_stats[1] ) else False             # mean area + stdev
#    #passArea    = True if meanMissingArea < (predict_area_stats[1] + 3*predict_area_stats[2]) else False             # mean area + stdev
#    if passShortEvent and passPredC and passDisplC and passArea:
#        print('\n')
    
#        fixSplitMerge.append([leadingID,latestTime,key])
#        lastErrors  = [a for t,[_,a,_,_] in g_predict_displacement[leadingID2].items() if latestTime  - 4 < t  <= latestTime - 1 ] + distPredictAndLost
#        meanError   = max(int(np.mean(lastErrors)),3)
#        stdError    = max(int(np.std(lastErrors)),1)
#        g_predict_displacement[leadingID][latestTime] = [predict_centroid,distPredictAndLost,meanError,stdError]

#        if swappedIDCase == True:                        # other bubble holds history, copy it to main bubble.
#            print(f'{latestTime}: swapped IDs case')
#            startTIme   = times[leadingID2]
#            endTime     = times[leadingID]
#            inheritTimes = np.arange(startTIme,endTime,1)
#            for t in inheritTimes:
#                g_old_new_IDs[leadingID][t]             = g_old_new_IDs[leadingID2][t]         
#                g_Centroids[leadingID][t]               = g_Centroids[leadingID2][t]           
#                g_areas_hull[leadingID][t]              = g_areas_hull[leadingID2][t]          
#                g_bubble_type[leadingID][t]             = g_bubble_type[leadingID2][t]         
#                g_Rect_parms[leadingID][t]              = g_Rect_parms[leadingID2][t]          
#                g_Ellipse_parms[leadingID][t]           = g_Ellipse_parms[leadingID2][t]       
#                g_predict_displacement[leadingID][t]    = g_predict_displacement[leadingID2][t]
#                g_predict_area_hull[leadingID][t]       = g_predict_area_hull[leadingID2][t]
#                g_contours_hull[leadingID][t]           = g_contours_hull[leadingID2][t]
                
#                sortDictEntry(g_old_new_IDs,            leadingID)
#                sortDictEntry(g_Centroids,              leadingID)
#                sortDictEntry(g_areas_hull,             leadingID)
#                sortDictEntry(g_bubble_type,            leadingID)
#                sortDictEntry(g_Rect_parms,             leadingID)
#                sortDictEntry(g_Ellipse_parms,          leadingID)
#                sortDictEntry(g_predict_displacement,   leadingID)
#                sortDictEntry(g_predict_area_hull,      leadingID)
        
#        for i,time in enumerate(fillTimes):
#            g_old_new_IDs[leadingID][time]      = fillSubIDs[i]
#            g_areas_hull[leadingID][time]       = fillAreas[i]
#            g_Centroids[leadingID][time]        = fillCentroids[i]
#            g_contours_hull[leadingID][time]    = cv2.convexHull(np.vstack(g_contours[time][fillSubIDs[i]]))
            

#        falseMergeTimesAll = [t for t in  list(g_areas_hull[key].keys()) if t >= mergeTime]
#        if swappedIDCase == False:                                                          # looks like swapped case main bubble holds post merge history
#            for i,time in enumerate(falseMergeTimesAll):
#                subIDs                          = g_old_new_IDs[key][time]
#                centroid, area                  = getHullCA(g_contours[time][subIDs]) 
#                g_old_new_IDs[leadingID][time]  = subIDs
#                g_areas_hull[leadingID][time]   = area
#                g_Centroids[leadingID][time]    = centroid   
#                g_contours_hull[leadingID][time]= cv2.convexHull(np.vstack(g_contours[time][subIDs]))

#        allTimeSteps = sorted(list(np.unique(list(fillTimes) + falseMergeTimesAll)))
#        for time in allTimeSteps:
#            lastCentroids   = np.array([c for t,c in g_Centroids[leadingID].items() if time - 3 < t <= time ])
#            centroid        = g_Centroids[leadingID][time]
#            hullA           = g_areas_hull[leadingID][time]
#            a = 1
#            slope           = sc_stat.linregress(*lastCentroids.T)[0]
#            stepLen         = np.linalg.norm(diffsTest)
#            vector          =  stepLen * np.array([1, slope]) / np.linalg.norm([1, slope])              # build a vector from slope
#            vector          = np.sign(np.dot(diffsTest,np.array([1, slope]))) * vector                  # change sign if needed
#            predictVector   = tuple(map(int,(vector + centroid)))
#            predictedOld    = g_predict_displacement[leadingID][time][0]

#            distPredictOld  = np.linalg.norm(np.array(predictedOld) - np.array(centroid)).astype(int)
#            lastErrors  = [a for t,[_,a,_,_] in g_predict_displacement[leadingID].items() if time - 3 < t <= time - 1 ] + distPredictOld
#            meanError   = max(int(np.mean(lastErrors)),3)
#            stdError    = max(int(np.std(lastErrors)),1)
            
#            if time+1 not in g_predict_displacement[leadingID]: g_predict_displacement[leadingID][time + 1] = []
#            g_predict_displacement[leadingID][time+1] = [predictVector,distPredictOld,meanError,stdError]

#            lastErrorsA  = [a for t,a in g_areas_hull[leadingID].items() if time  - 4 < t  <= time - 1 ] 
#            meanErrorA   = int(np.mean(lastErrorsA))
#            stdErrorA    = max(int(np.std(lastErrorsA)),int(0.1*meanErrorA))
#            if time not in g_predict_area_hull[leadingID]: g_predict_area_hull[leadingID][time] = []
#            g_predict_area_hull[leadingID][time] = [hullA,meanErrorA,stdErrorA]
#            if time not in g_bubble_type[leadingID]: g_bubble_type[leadingID][time] = typeElse
#            g_bubble_type[leadingID][time] = typeElse
#            if time not in g_Rect_parms[leadingID]: g_Rect_parms[leadingID][time] = []
#            g_Rect_parms[leadingID][time] = cv2.boundingRect(g_contours_hull[leadingID][time])

#            if leadingID != key:
#                if key in list(sum(g_bublle_type_by_gc_by_type[time].values(),[])):
#                    where = [ID for ID,vals in g_bublle_type_by_gc_by_type[time].items() if key in vals][0]
#                    g_bublle_type_by_gc_by_type[time][where].remove(key)
#                    g_bublle_type_by_gc_by_type[time][typeElse].append(leadingID)
#                if time in g_splits and key in g_splits[time]:
#                    tempVals  = g_splits[time][key]
#                    g_splits[time].pop(key,None)
#                    g_splits[time][leadingID] = tempVals
#                    a = 1

#        if leadingID != key:
#            g_MBub_info.pop(            key,    None)
#            g_old_new_IDs.pop(          key,    None)
#            g_Centroids.pop(            key,    None)
#            g_areas_hull.pop(           key,    None)
#            g_bubble_type.pop(          key,    None)
#            g_Rect_parms.pop(           key,    None)
#            g_Ellipse_parms.pop(        key,    None)
#            g_predict_displacement.pop( key,    None)
#            g_predict_area_hull.pop(    key,    None)
#            stage1RemovedGIDs.append(key)
#            print(f'{mergeTime}: removing false new merge ID: {key}, keep origina;: {leadingID}')
#        else: print(f'{mergeTime}: no new merge ID, keep old ID: {key}')
        
#        if mergeTime in g_merges:
#            if key in g_merges[mergeTime]:
#                if len(g_merges[mergeTime]) == 1:  g_merges.pop(mergeTime,None)
#                else: g_merges[mergeTime].pop(key,None)
        
#        if latestTime in g_splits:
#            if leadingID in g_splits[latestTime]:
#                if len(g_splits[latestTime]) == 1:  g_splits.pop(latestTime,None)
#                else: g_splits[latestTime].pop(leadingID,None)

#        otherIDs = [ID for ID in culpritPair if ID != leadingID]
#        print(f'{mergeTime}: removing false secondary IDs: {otherIDs} (merged into {leadingID})')
#        for a in otherIDs:
            
#            g_old_new_IDs.pop(          a,    None)
#            g_Centroids.pop(            a,    None)
#            g_areas_hull.pop(           a,    None)
#            g_bubble_type.pop(          a,    None)
#            g_Rect_parms.pop(           a,    None)
#            g_Ellipse_parms.pop(        a,    None)
#            g_predict_displacement.pop( a,    None)
#            g_predict_area_hull.pop(    a,    None)
#            stage1RemovedGIDs.append(a)
#        removedSplitMergeIDs = removedSplitMergeIDs + otherIDs
#        inheritIDs[key] = leadingID
#        whereKey = [t for t,vals in splitMergeCulpritsCompanions.items() if key in vals]
#        for t in whereKey:
#            splitMergeCulpritsCompanions[t].remove(key)
#            splitMergeCulpritsCompanions[t].append(leadingID)
#            splitMergeCulpritsCompanions[t] = sorted(splitMergeCulpritsCompanions[t])           # cheking if updating works
#        whereKey2 = {t:{i:v for i,v in vals.items() if key in v} for t,vals in g_merges.items() }
#        whereKey3 = {t:list(v.keys())[0] for t,v in whereKey2.items() if len(v)>0}
#        for t,i in whereKey3.items():
#            g_merges[t][i].remove(key)
#            g_merges[t][i].append(leadingID)
#            g_merges[t][i] = sorted(g_merges[t][i])
#        #g_merges 

#        a = 1
#stage1RemovedGIDs = sorted(list(map(int,stage1RemovedGIDs)))
#print(f'fixed false split-merges:\n{fixSplitMerge}')

## =========================================================================================================================================================
## =========================================================================================================================================================
## ==================================================== Gather small +  big merge earlier ==================================================================
## =========================================================================================================================================================
#for time, subIDs in mergingGlobalsSmallBig.items():
#    #test = [1 if ID not in removedSplitMergeIDs + resortTogetherFailIDs else 0 for ID in subIDs]
#    test = [1 if ID not in stage1RemovedGIDs +  list(inheritIDs.values()) else 0 for ID in subIDs]
#    if not all(test): continue
#    areas = {ID:g_areas_hull[ID][time-1] for ID in subIDs if time-1 in g_areas_hull[ID] }
#    if len(areas) == 2:
#        times = {ID: len(list(g_areas_hull[ID].keys())) for ID in subIDs}
#        minArea, maxArea = min(areas.values()), max(areas.values())
#        if minArea/maxArea < 0.25 and min(list(times.values())) <= 2:
#            smallID = min(areas,key=areas.get)
#            bigID   = max(areas,key=areas.get)
#            earlestTimeSmall = min(g_areas_hull[smallID])
#            for t in range(earlestTimeSmall,time):
#                subIDs                   = g_old_new_IDs[bigID][t] + g_old_new_IDs[smallID][t]
#                centroid, area           = getHullCA(g_contours[t][subIDs]) 
#                g_old_new_IDs[bigID][t]  = subIDs
#                g_areas_hull[bigID][t]   = area
#                g_Centroids[bigID][t]    = centroid   
#                g_contours_hull[bigID][t]= cv2.convexHull(np.vstack(g_contours[t][subIDs]))
#            g_old_new_IDs.pop(          smallID,    None)
#            g_Centroids.pop(            smallID,    None)
#            g_areas_hull.pop(           smallID,    None)
#            g_bubble_type.pop(          smallID,    None)
#            g_Rect_parms.pop(           smallID,    None)
#            g_Ellipse_parms.pop(        smallID,    None)
#            g_predict_displacement.pop( smallID,    None)
#            g_predict_area_hull.pop(    smallID,    None)
#            if len(g_merges[time]) == 1:   g_merges.pop(time,None)
#            else:                          g_merges[time].pop(bigID,None)
#            g_bubble_type[bigID][time] = typeElse
#            g_Rect_parms[bigID][time] = cv2.boundingRect(g_contours_hull[bigID][time])
#            removedSplitMergeIDs += [smallID]
#            # === not doing prediction recalculation... yet ===
#            a = 1
#removedSplitMergeIDs = sorted(set(removedSplitMergeIDs))
#detectedSplitsUPD = {}
#for time,stats in g_splits.items():
#    if time > maxStepsForTest: break
#    splitsTrue = [ID for ID,params in stats.items() if params[0] == True and params[1] is not None]
#    splitsTrue2 = [ID for ID in splitsTrue if ID not in removedSplitMergeIDs]
#    if len(splitsTrue2) > 0 :
#        detectedSplitsUPD[time] = splitsTrue2
## =========================================================================================================================================================
## =========================================================================================================================================================
## ======================================================== Separate detected splits earler ================================================================
## ========================================================================================================================================================= 
#for time, splitMainIDs in detectedSplitsUPD.items():
#    subMainIDs = {ID:g_splits[time][ID] for ID in splitMainIDs}
#    for ID in splitMainIDs:
#        if subMainIDs[ID][1] is not None and ID not in inheritIDs.values(): # added check if ID was resolved on stage 1. means some fixes are not done. but its hard to troubleshoot
            
#            ovlp = overlappingRotatedRectangles(fakeBox,{0:g_Rect_parms[ID][time]})
#            if len(ovlp)>0: continue                                        # problems in inlet zone.

#            localIDs = subMainIDs[ID][2]
#            activeIDs = sum(list(g_bublle_type_by_gc_by_type[time].values()),[])    # no global IDs avalabile since they were generated after.
#            #activeIDs = [ID if ID not in inheritIDs else inheritIDs[ID] for ID in activeIDs]
#            temp = {i:g_old_new_IDs[i][time] for i in activeIDs if i != ID}         # active global IDs that are not main ID
#            otherID = [i for i, subIDs in temp.items() if subIDs in localIDs][0]    # check which of remaining active globl IDs has same local IDS 
#            startTime = time
#            while startTime-1 in g_splits and ID in g_splits[startTime-1]:          # use failed split history to count back to first split check.
#                startTime -= 1
#            timeSteps = np.arange(startTime,time+1,1)                               # times from first split detection up to split confirmation
#            localIDsPrev = {t:g_splits[t][ID][2] for t in timeSteps}                # grab local IDs from history
#            centr = {}

#            for t, subIDs in localIDsPrev.items():
#                centr[t] = [getHullCA(g_contours[t][subIDs[i]])[0] for i in [0,1]]  # generate centroid from local IDs
           
#            relations2 = {0:{time:[centr[t][0],localIDsPrev[t][0]]},1:{time:[centr[t][1],localIDsPrev[t][1]]}}
#            for t in np.flip(timeSteps[1:]):                                        # propagate from confirmed split time back and restore connections
#                for i in [0,1]:
#                    p1 = centr[t][i]
#                    ps2 = centr[t-1]
#                    dist = np.linalg.norm(np.array(ps2) - np.array(p1), axis = 1).astype(int)
#                    minPos = np.argmin(dist)
#                    #relations2[i][t]= [centr[t][i],localIDsPrev[t][i]]
#                    relations2[minPos][t-1]= [ps2[minPos],localIDsPrev[t-1][i]]
#            cntrOG = g_Centroids[ID][time]                                         # grab original  centroid since i regenerated it anew

#            pickMain = {i: np.linalg.norm(np.array(cntrOG) - relations2[i][time][0]) for i in [0,1]}
#            mainIDIndex = min(pickMain, key=pickMain.get)                          # see which connection path is main bubble
#            otherIDIndex = max(pickMain, key=pickMain.get)
#            globIDs = {mainIDIndex:ID,otherIDIndex:otherID}

#            bubParams = {}
            
#            for t in timeSteps[:-1]:                                               # rewrite separated stats from first detection time to OG split
#                for i in [mainIDIndex,otherIDIndex]:
#                    pickID                      = globIDs[i]
#                    subIDs                      = relations2[i][t][1]
#                    centroid, area              = getHullCA(g_contours[t][subIDs]) 
#                    g_old_new_IDs[pickID][t]    = subIDs
#                    g_areas_hull[pickID][t]     = area
#                    g_Centroids[pickID][t]      = centroid   
#                    g_contours_hull[pickID][t]  = cv2.convexHull(np.vstack(g_contours[t][subIDs]))
#                    g_bubble_type[pickID][t]    = typeElse
#                    g_Rect_parms[pickID][t]     = cv2.boundingRect(g_contours_hull[ID][t]) 
#            for i in list(globIDs.values()):                                      # have to resort other bubble since added eariler time steps to end.
#                 sortDictEntry(g_old_new_IDs,i)
#                 sortDictEntry(g_areas_hull,i)
#                 sortDictEntry(g_areas_hull,i)
#                 sortDictEntry(g_Centroids,i)
#                 sortDictEntry(g_contours_hull,i)
#                 sortDictEntry(g_bubble_type,i)
#                 sortDictEntry(g_Rect_parms,i)
#            # dont think that you need predictor stuff since we reslice already happened events
#            a = 1
#    a = 1
# =========================================================================================================================================================
# =========================================================================================================================================================
# ====================================Detect same frozen bubble with different global IDs==================================================================
# =========================================================================================================================================================

u, c = np.unique(sum(list(allFrozenIDs.values()),[]), return_counts=True) # c counts only frozen states, not recovered (E)
fDictC = {} # mean pos, max displ, mean dist, stde disp, num copies
fDictBR = {}
for i in range(len(u)):
    fID, count = u[i],c[i]
    if fID in removedSplitMergeIDs or fID not in g_areas_hull: continue
    fcs         = np.array([ g_Centroids[fID][time] for time in g_areas_hull[fID]])
    fcMean      = np.mean(fcs,axis = 0)
    fcDispl     = fcs - fcMean
    fcDisplMag  = np.linalg.norm(fcDispl, axis = 1).astype(int)
    fDictC[fID]  = [fcMean,max(fcDisplMag),np.mean(fcDisplMag),np.std(fcDisplMag),count]
    fBRmaxPrms  = np.array([ g_Rect_parms[fID][time] for time in g_areas_hull[fID]])
    fBRmaxPts   = np.array([[(x,y),(x+w,y),(x,y+h),(x+w,y+h)] for x,y,w,h in fBRmaxPrms]).reshape((-1,2))
    fDictBR[fID]= cv2.boundingRect(fBRmaxPts)

relations0 = overlappingRotatedRectangles(fDictBR,fDictBR)
relations = []
for [a,b] in relations0:
    ca,mda,mna,stda,cna = fDictC[a]
    cb,mdb,mnb,stdb,cnb = fDictC[b]
    #distAB = np.linalg.norm(ca - cb).astype(int)
    maxDispl = max(mna,mnb)
    #maxStd = max(stda,stdb)
    if maxDispl < 20: relations.append([a,b]) #distAB < maxDispl + maxStd and

unique_cc = graphUniqueComponents(u,relations) 
unique_cc = [a for a in unique_cc if len(a)>1]
fRepresent = {min(a):sorted(a) for a in unique_cc}

for ID,subIDs in fRepresent.items():
    subIDs = subIDs[1:]
    for subID in subIDs:
        g_Centroids[ID]     = {**g_Centroids[ID],       **g_Centroids[subID]    }
        g_old_new_IDs[ID]   = {**g_old_new_IDs[ID],     **g_old_new_IDs[subID]  }
        g_areas_hull[ID]    = {**g_areas_hull[ID],      **g_areas_hull[subID]   }
        g_bubble_type[ID]   = {**g_bubble_type[ID],     **g_bubble_type[subID]  }
        g_contours_hull[ID] = {**g_contours_hull[ID],   **g_contours_hull[subID]}
        g_Rect_parms[ID]    = {**g_Rect_parms[ID],      **g_Rect_parms[subID]   }
        g_old_new_IDs.pop(  subID, None)
        g_Centroids.pop(    subID, None)
        g_areas_hull.pop(   subID, None)
        g_bubble_type.pop(  subID, None)
        g_contours_hull.pop(subID, None)
        g_Rect_parms.pop(   subID, None)
        a= 1
    g_bubble_type[ID] = {time:typeFrozen for time in g_bubble_type[ID]}



print(f'fixed multi ID frozens :\n{fRepresent}')
# =========================================================================================================================================================
# =====================================================Remove single case IDs ============================================================
# =========================================================================================================================================================
soloIDs = [ID for ID,vals in g_areas_hull.items() if len(vals) == 1]
for ID in soloIDs:
    g_old_new_IDs.pop(  ID, None)
    g_Centroids.pop(    ID, None)
    g_areas_hull.pop(   ID, None)
    g_bubble_type.pop(  ID, None)
    g_contours_hull.pop(ID, None)
    g_Rect_parms.pop(   ID, None)
print(f'Remove solo IDs :\n{soloIDs}')
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================




a = 1
data = []
fakeBox = []
dataStart = 0
dataTestFolder
imageLinksBlank             = glob.glob(dataTestFolder + "**/*.png", recursive=True) 
extractIntergerFromFileName = lambda x: int(re.findall('\d+', os.path.basename(x))[0])
csvImageLinksNumbers        = [extractIntergerFromFileName(x) for x in imageLinksBlank]
allRelevantFrames           = np.arange(min(maxStepsForTest,backupStart))
if len(csvImageLinksNumbers) == 0: latestFrame = 0
else: latestFrame                 = max(csvImageLinksNumbers)

if overWriteStart == -1: start = latestFrame
else:                    start = min(overWriteStart,maxStepsForTest)
exportCSVimgsIDs            = [x for x in allRelevantFrames if start <= x < min(overWriteEnd,maxStepsForTest)]
a = 1
if 1 == 1:
    for globalCounter in tqdm(exportCSVimgsIDs):
        #==============================================================================================
        #======================== DRAWING STUFF START ================================================
        #==============================================================================================
    
        activeIDs = [ID for ID, timeDict in g_bubble_type.items() if globalCounter in timeDict]#;print(f'globalCounter: {globalCounter} activeIDs: {activeIDs}')

        blank = np.uint8(cv2.imread(imageLinks[globalCounter],1))
        if len(fakeBox) > 0:
            x,y,w,h = fakeBox[-1]
            cv2.rectangle(blank, (x,y), (x+w,y+h),[120,15,55],1)
        typeStrings = ["N","RB", "rRB", "E", "F","rE","pm","rF","M"]
        splitStrings =  {}
        for ID in activeIDs:
            temp            = [0,len(g_old_new_IDs[ID][globalCounter])]
            currentType     = g_bubble_type[ID][globalCounter]#;print('currentType',currentType)
            currentTypeStr  = typeStrings[currentType] 
            currentCntrIds  = g_old_new_IDs[ID][globalCounter]
            # text            = currentTypeStr + '('+','.join(list(map(str,currentCntrIds)))+')'
            temp.append(currentTypeStr + '(')
            cCIstr = list(map(str,currentCntrIds))
            A = [x for y in zip(cCIstr, [","]*len(cCIstr)) for x in y][:-1] #dont ask me
            temp += A
            if min(g_old_new_IDs[ID]) == globalCounter:                 # if current time step is the first in IDs data - mark it "NEW"            
                temp += [f"):{ID} new"]
            else:                                                       # 
                prevTimeStep = globalCounter - 1 if (currentType != typeFrozen and currentType != typeRecoveredFrozen) else [ts for ts in g_bubble_type[ID].keys() if ts < globalCounter][-1]
                temp[0] = 1
                prevType = g_bubble_type[ID][prevTimeStep]
                prevTypeStr = typeStrings[prevType] 
                prevCntrIds  = g_old_new_IDs[ID][prevTimeStep]
                # text += "-" + prevTypeStr + f'({ID})'
                temp += [")"+str(":" + prevTypeStr + f'({ID})')]
            splitStrings[ID] = temp
        
            if ID in g_child_contours:
                if globalCounter in g_child_contours[ID]:
                    [cv2.drawContours(  blank,   g_contours[globalCounter], cid, (0,255,0), 1) for cid in g_child_contours[ID][globalCounter]]
            # aa = [cid for cid in g_child_contours[ID][globalCounter] if globalCounter in g_child_contours[ID]]
            colorSet = [(),(0,0,255),(0,100,190),(255,0,0),(125,0,125),(255,0,125),(0,255,0),(125,0,125),(100,255,100)]
            thcSet = [(), 2, 1, 2, 2, 2 ,2, 2,2]
            clr = colorSet[currentType]
            thc = thcSet[currentType]
            [cv2.drawContours(  blank,   g_contours[globalCounter], cid, clr, thc) for cid in currentCntrIds]
            #cv2.ellipse(blank, g_Ellipse_parms[ID][globalCounter], (55,55,55), 1)

            cv2.drawContours(  blank,   [g_contours_hull[ID][globalCounter]], -1, (255,255,255), 1)
            
        fontScale = 0.7;thickness = 4;

        # ========= overlay organization =============
        # check drawing.py for standalone example. 
        # works the following way. bubbles fight for vertical space. width is modified to text width. bubbles that overlay with other on vertical space are grouped. 
        # that means that text will overlap. group is split into top and bottom groups- less overlay. top and bottom groups are split into vertical positions.
        thickRectangle = {}
        rectParams = {}
        strDimsAll = {}
    
        H,L,_           = blank.shape
        textHeightMax   = 0
        baselineMax     = 0

        for k,ID in enumerate(activeIDs):
            case, numIDs, *strings  = splitStrings[ID]
            strDims  = [cv2.getTextSize(text, font, fontScale, thickness) for text in strings]          # substrings are split to make a line pointing from subIDs to contour
            baseline = max([bl for _,bl in strDims])                                                    # i draw substring seperately. full string drawing is scaled differently
            ySize = max([sz[1] for sz,_ in strDims])
            xSize = [sz[0] for sz,_ in strDims]
            totSize = sum(xSize)
            #[totSize,ySize],baseline  = cv2.getTextSize(''.join(strings), font, fontScale, thickness) # i think text dimensions for partial sum and full string are different.

            textHeightMax           = max(textHeightMax,    ySize)                              # total text height is textHeightMax + baselineMax
            baselineMax             = max(baselineMax,      baseline)                           # baselineMax is height of the part of text under the baseline e.g 'y,p,g,q,..', baseline  = 0 for 'o,k,d,..'
            x,y,w,h                 = g_Rect_parms[ID][globalCounter]
            rectParams[ID]          = [x,y,w,h] 
            dw                      = totSize-w           # if text is thicker, then 
            xS, hS                  = [int(x-0.5*dw),H]
            dx1                     = max(0, -xS)                                               # if BRect is poking out of left side of image dx1 is positive, else zero
            dx2                     = max(0, xS + totSize - L)                                  # if BRect is poking out of right side of image dx2 is positive, else zero
            thickRectangle[ID]      = [xS + dx1 - dx2,0,totSize, hS]
            strDimsAll[ID]          = strDims
            #cv2.rectangle(blank, (xS + dx1 - dx2,y), (xS + dx1 - dx2+totSize,y+h),(255,0,0),2)
        # split into competing vertical columns    
        aa = overlappingRotatedRectangles(thickRectangle,thickRectangle)
        HG = nx.Graph()
        HG.add_nodes_from(rectParams.keys())
        HG.add_edges_from(aa)
        # 
        cntrd = {ID:tuple((x+int(0.5*w),y+int(0.5*h))) for ID,[x,y,w,h] in rectParams.items()}
        posTB = {ID:0 if y < (H - (y+h)) else 1 for ID,[x,y,w,h] in rectParams.items()}

        neighbors = {ID:[] for ID in rectParams}
        neighborsSameSide = {ID:[] for ID in rectParams}
        # dont remember what, but something interesting happens for splitting.
        # top-bottom splitting is not that simple. i think only neighbors are tests wheter they share same side.
        for IDs in rectParams:
            for i in HG.neighbors(IDs):
                neighbors[IDs].append(i)
                if posTB[i] == posTB[IDs]: neighborsSameSide[IDs].append(i) # write connections between neighbors on same side
        #clusters0 = [tuple(np.sort(np.array([ID] +nbrs))) for ID, nbrs in neighborsSameSide.items()]
        a = 1
        edgePairs = sum([[(ID,a) for a in nbrs] for ID,nbrs in neighborsSameSide.items() if len(nbrs) > 0], []) # rewrite connections into pairs, ignore solo connections
        sameSideClusters = graphUniqueComponents(list(rectParams.keys()),edgePairs)                             # get same side cluster, next step order them by centroid y coordinate
    
        sortedByY = [[x for _,x in sorted(zip([cntrd[a][1] for a in X],X))] for X in sameSideClusters] # https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list
        sortedIndexed = [{ID:i for i,ID in enumerate(ID)} for ID in sortedByY]                         # enumerate index is a vertical position index for text
        globSideOrder = {k: v for d in sortedIndexed for k, v in d.items()}                            # combine list of dictionaries into on dict.

        
        textHeight  = textHeightMax
        textNoGo    = 10
        textSep     = 15
        # text position is calculted for top and bottom part of image separately. different coordinate systems- from edge. duh
        for ID,[x,y,w,h] in thickRectangle.items():
            if posTB[ID] == 1:
                textPos = H - textNoGo - baselineMax - globSideOrder[ID]*(textHeightMax + baselineMax + textSep)
            if posTB[ID] == 0:
                textPos = textNoGo + textHeightMax + globSideOrder[ID]*(textHeightMax + baselineMax + textSep)
        
        
            case, numIDs, *strings  = splitStrings[ID]
            strDims                 = strDimsAll[ID]

            startPos                = np.array((x,textPos),int) 
            ci = 0
            subIDsIndices = list(range(1,len(strings),2))
            # iterating through overlay substrings, on subIDs calculate nearest distance from ID string center to any point on countour with that ID
            for i, string in enumerate(strings):
                [cv2.putText(blank, string, startPos, font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
                if i in subIDsIndices:
                
                    startPosMiddle = startPos + np.array((strDims[i][0][0]/2,0),int)        # startPos-> string start, startPosMiddle = string start +half width
                    cntrID = g_old_new_IDs[ID][globalCounter][ci]
                    _, _, p1, p2 = closes_point_contours(startPosMiddle.reshape(-1,1,2),g_contours[globalCounter][cntrID])
                    cv2.polylines(blank, [np.array([p1- np.array((0,(textHeightMax + baselineMax)/2),int),p2])],0, (0,0,255), 1)
                    startPos += np.array((strDims[i][0][0],0),int)
                    ci += 1
                else: startPos += np.array((strDims[i][0][0],0),int)                        # skip, add full width offset.
            
            #string = ''.join(strings)
            #[cv2.putText(blank, string, (x,textPos), font, fontScale, clr,s, cv2.LINE_AA) for s, clr in zip([thickness,1],[(255,255,255),(0,0,0)])]
           

    
        #--------------- trajectories ---------------------------------
        for i,times in g_Centroids.items(): # key IDs and IDS-> times
            useTimes = [t for t in times.keys() if t <= globalCounter and t>globalCounter - 15]#
            pts = np.array([times[t] for t in useTimes]).reshape(-1, 1, 2)
            if pts.shape[0]>3:
                cv2.polylines(blank, [pts] ,0, (255,255,255), 3)
                cv2.polylines(blank, [pts] ,0, cyclicColor(i), 2)
                [cv2.circle(blank, tuple(p), 3, cyclicColor(i), -1) for [p] in pts]
            else:
                cv2.polylines(blank, [pts] ,0, cyclicColor(i), 1)
                [cv2.circle(blank, tuple(p), 1, cyclicColor(i), -1) for [p] in pts]
            
        cv2.putText(blank, str(globalCounter), (25,25), font, 0.9, (255,220,195),2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(dataTestFolder,str(dataStart+globalCounter).zfill(4)+".png") ,blank)
        #cv2.imwrite(".\\imageMainFolder_output\\ringDetect\\orig_"+str(dataStart+globalCounter).zfill(4)+".png" ,ori)
        # cv2.imshow(f'{globalCounter}', resizeToMaxHW(blank))
    
    #==============================================================================================
    #======================== DRAWING STUFF  END ==================================================
    #==============================================================================================


#a = 1
#_, ax = plt.subplots(ncols = 1)
#for ID in range(0,5):
#    traj = np.array(list(g_Centroids[ID].values())) 
#    ax.plot(*traj.T)
#ax.set_ylim(bottom=0,top = 800)
#ax.set_xlim(left=0, right = 1200)
#plt.show()
#a = 1