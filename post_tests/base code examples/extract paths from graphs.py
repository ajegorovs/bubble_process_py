
import networkx as nx, itertools
import matplotlib.pyplot as plt, numpy as np

# assume graph represents bubble temporal evolution: node (t1,t2) denotes object at t1 connected to object at t2 (t2>t1)
# chain or path is graph with edges [(t1,t2),(t2,t3),...]
# merges and splits are represented by edges [(t1a,t2),(t1b,t2)] and [(t1,t2a),(t1,t2b)]
# i want to extract all chains/path between merges and splits. chain starts from nothing or from merge
# chain terminates at nothing, split (not including) or before merge. 


colorList  = np.array(list(itertools.permutations(np.arange(0,255,255/5, dtype= np.uint8), 3)))
np.random.seed(2);np.random.shuffle(colorList);np.random.seed()

def cyclicColor(index):
    return colorList[index % len(colorList)].tolist()

# nodes may not be time explicitly, they may be tuples of information that contain time. e.g node1 = (t1,ID1),...
# thats why time can be extracted from node name using argument function "f" e.g f = lambda x: x[0]
def graph_extract_paths(H,f):
    nodeCopy = list(H.nodes()).copy()
    segments2 = {a:[] for a in nodeCopy}
    resolved = []
    skipped = []
    for node in nodeCopy:
        goForward = True if node not in resolved else False
        nextNode = node
        prevNode = None
        while goForward == True:
            neighbors = list(H.neighbors(nextNode))
            nextNodes = [a for a in neighbors if f(a) > f(nextNode)]
            prevNodes = [a for a in neighbors if f(a) < f(nextNode)]
            # find if next node exists and its single
            soloNext    = True if len(nextNodes) == 1 else False
            soloPrev    = True if len(prevNodes) == 1 else False # or prevNode is None
            soloPrev2   = True if soloPrev and (prevNode is None or prevNodes[0] == prevNode) else False

            # if looking one step ahead, starting node can have one back and/or forward connection to split/merge
            # this would still mean that its a chain and next/prev node will be included.
            # to fix this, check if next/prev are merges/splits
            # find if next is not merge:
            nextNotMerge = False
            if soloNext:
                nextNeighbors = list(H.neighbors(nextNodes[0]))
                nextPrevNodes = [a for a in nextNeighbors if f(a) < f(nextNodes[0])]
                if len(nextPrevNodes) == 1: 
                    nextNotMerge = True

            nextNotSplit = False
            if soloNext:
                nextNeighbors = list(H.neighbors(nextNodes[0]))
                nextNextNodes = [a for a in nextNeighbors if f(a) > f(nextNodes[0])]
                if len(nextNextNodes) <= 1:   # if it ends, it does not split. (len = 0)
                    nextNotSplit = True

            prevNotSplit = False
            if soloPrev2:
                prevNeighbors = list(H.neighbors(prevNodes[0]))
                prevNextNodes = [a for a in prevNeighbors if f(a) > f(prevNodes[0])]
                if len(prevNextNodes) == 1:
                    prevNotSplit = True


            saveNode = False
            # test if it is a chain start point:
            # if prev is a split, implies only one prevNode 
            if prevNode is None:                # starting node
                if len(prevNodes) == 0:         # if no previos node =  possible chain start
                    if nextNotMerge:            # if it does not change into merge, it is good
                        saveNode = True
                    else:
                        skipped.append(node)
                        goForward = False
                elif not prevNotSplit:
                    if nextNotMerge:
                        saveNode = True
                    else: 
                        skipped.append(node)
                        goForward = False
                else:
                    skipped.append(node)
                    goForward = False
            else:
            # check if its an endpoint
                # dead end = zero forward neigbors
                if len(nextNodes) == 0:
                    saveNode = True
                    goForward = False
                # end of chain =  merge of forward neigbor
                elif not nextNotMerge:
                    saveNode = True
                    goForward = False
                elif not nextNotSplit:
                    saveNode = True
                
                # check if it is part of a chain
                elif nextNotMerge:
                    saveNode = True


            if saveNode:
                segments2[node].append(nextNode)
                resolved.append(nextNode)
                prevNode = nextNode
                if goForward :
                    nextNode = nextNodes[0]

    
    
    return segments2, skipped

edges = [(0,1.15),(1.15,2.15),(1.16,2.15),(1, 2), (2, 3), (3, 4), (4,4.1), (4.1,5),(4,4.2),  (6, 7), (7, 8), (8, 9.1), (8,9.2), (9.1, 9.5),(9.5,10), (9.2, 10),(10,11),(11,12),(1.1,2.1),(2.1,3.1),(3.1,4),(1.2,2.2),(2.2,4)]
f = lambda x : x
#edges = [[tuple([u]),tuple([v])] for u,v in edges]
#f = lambda x : x[0]
H = nx.Graph()
H.add_edges_from(edges)


segments2, skipped = graph_extract_paths(H,f)

# Draw extracted segments with bold lines and different color.
segments2 = [a for _,a in segments2.items() if len(a) > 0]
paths = {i:vals for i,vals in enumerate(segments2)}
colors = {i:np.array(cyclicColor(i))/255 for i in paths}

colors_edges2 = {}
width2 = {}
# set colors to different chains. iteratively sets default color until finds match. slow but whatever
for u, v in H.edges():
    for i, path in enumerate(segments2):
        if u in path and v in path:

            colors_edges2[(u,v)] = colors[i]
            width2[(u,v)] = 5
            break
        else:

            colors_edges2[(u,v)] = np.array((0,0,0))
            width2[(u,v)] = 2

nx.draw(H, with_labels=True, edge_color=list(colors_edges2.values()), width = list(width2.values()))
plt.show()