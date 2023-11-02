# bubble_process_py
series of images with raising bubbles -> time series analysis
Algorithm is designed to extract contours from a series of imges depicting bubbles flow. Original case works with images that are of high contrast images as can seen on image below
![My Image](README_FILES/img0027.bmp)
fish-eye correction is applied to images using predefined (mapx.npy,mapy.npy) parameter array (openCV image calibration):
![My Image](README_FILES/fish-eye-correction-overlay-krita.jpg)
images are cropped by either reading manually edited crop masp, which is an image with red filled rectangle overlay, which specifies remaining image rectangle.
or, if mask is missing allows you to crop via GUI

mean sequence image is calculated and its subtracted from each image. result is the following:
![My Image](README_FILES/proc_mean.jpg)

alternative implementation of mean calculation is using more local (time-narrower) sample sequence

images are binarized and small elements are removed
![My Image](README_FILES/binar.jpg)

we merge nearby contours using dilate-erode morphology operation to join fractured binarry objects (read below) (*)
contours are extracted and filtered out by size and from FoV periphiry.

Clustering on single frame:
some bubbles are fractured into smaller visual elements, which are not connected by (*) they are clustered together by proximity. our implementation is simple and fast
we find bounding boxes for all elements. those that are small, we increase to specific size 100x100 pix.
check overlaps between all pairs of bounding boxes. collect those that have overlap. to connect these clusters we gather overlap connections on a graph and find connected components
nodes on graph are IDs of contours on image of specific time steps. and edges are connections between two nodes which have the geometric overlap.
this is a rough estimate, which will refined after.

analyzing time evolution/connectivity of contours.
we analyze all pairs of frames and test similar boundary box overlap between previous-next frame bubbles. those who overlap are the same bubble on both frames.
by doing this analysis on all frames and combining on a 'relation' graph, we can trace which contours on each frames correspond to which progenitor bubble.
![My Image](README_FILES/rect_families_merge.jpg)

by analyzing connected components on the rough time-overlap-relation graph, we can split families of bubble trails, which dont interact with other families.
We can analyze each familry as isolated partition of graph with more detail. We repeat clustering on each single frame. This time without expanding bounding rectangles for small elements.
Its possible to replace rectangle overlap with analysis of closest distance between contours. but its slow, so i have disabled it.

Basically, for our case we can say that for most part bubble on each frame is represented as one contour (1C). so when we analyze frame pair,
we see an overlap between 1C object, on one frame, with other 1C object on next frame. we are pretty sure that if these connections
evolve as 1C objects, we are observing raise of a single bubble. In code i refer for these chains of single-contour bubble connections as a segments (or a branch).

interactions (merge/split) between two bubbles on relation graph is represented by an event where multiple segments merge or split into other segments.

merge/split events can be classified by connectivity of branches -- i.e many branches merge into one branch <=> merge event.

straight segments-chains is the only thing we are certain at this time, but...

we see cases where bubble, represented as a binary object, splits (is fragmented) into many pieces and joins back again. 
we know that this is still the same single bubble and splitting is an optical artifact. Although this is still a single bubble evolution, this event is not overlap between one contour objects. 
instead bubble after optical split creates a cluster, where every fragment of cluster is a separate node on graph.

![My Image](README_FILES/fake_event_1.jpg)

Ideally, we want to connect all solo bubble segments with are interrupted by this 'fake' event into one true chain. But for this we have to, at least gather clusters back together
into one object, and patch holes by joining split nodes into one composit node, which will tell that object consists of many contours.

Implemented solution is not that simple, instead we evaluate, wheter we can, actually, include all cluster elements together. instead we have to check different partitions of cluster and determine if this or that
partition fits better previous history of bubble trajectory and if area is conserved.

once these fake events are patched out, we are left again with interaction of long segments. 

by analyzing connectivity we can deduce type of event. merge and split involve one outgoing or incoming branch.
![My Image](README_FILES/branch_extend.jpg)

in practice geometric overlap condition is on overestimate and begins to detect events prematurely.
this means we have to refine merge event graphs with higher precision than oberlap condition.

for merges and splits we can extend branches closer to merge node by progressively extrapolating trajectory and redistributing nodes for each branch

other type of merges is when multiple branches come into event and multiple leave it.
![My Image](README_FILES/mixed.jpg)

due to proximity overestimation its hard to say straight away wheter this is real merge or simply bubbles passing close by.

to deal with it, we assume that splits are very rare, and they are. and extrapolate these event branches as merges.
and we check if extrapolation has led on of exit branch.

when we do this some nodes may be dropped from event. sometimes these node reveal that there is another, previously hidden branch, is located inside event.

if we do a second iteration of this whole graph refinement process, we can take a second glance at events and achieve additional refinement





