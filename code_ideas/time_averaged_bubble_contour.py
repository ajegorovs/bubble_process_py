import os, pickle, cv2, numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw 

def x_y_w_h_to_contour(rectangle_params, offset=(0, 0), expand=(0, 0)):
    # from bounding rectangle parameters x,y,height,width create list of rectangle vertex cooridnate
    # add optional x,y offset and vertical and horizontal expansion.
    x, y, w, h = rectangle_params
    w_add, h_add = expand

    vertices = np.array([
        [x      - w_add  , y     - h_add],
        [x      - w_add  , y + h + h_add],
        [x + w  + w_add  , y + h + h_add],
        [x + w  + w_add  , y     - h_add]
    ])

    return np.array(vertices + offset).reshape(-1,1,2)


def pad_image(image, width, height, draw_internal_border = True):
    # center image on blank image of size width x height. its not scaling
     
    img_width, img_height = image.size
    left_padding = (width - img_width) // 2
    top_padding = (height - img_height) // 2

    padded_img = Image.new(image.mode, (width, height), 0)  # white background
    padded_img.paste(image, (left_padding, top_padding))

    if draw_internal_border:
        draw = ImageDraw.Draw(padded_img)
        draw.rectangle([(1,1),(width - 1,height - 1)], width = 1, outline=128)

    return padded_img

def collage_pick_cols_rows_num_aspect(N, width, height, R = 16/9):
    #R = 16/9
    #N = 4
    #width = 50
    #height = 50
    # at least 1 columns, at max N columns
    columns_test    = range(1,N + 1)  
    # N//c = total number of columns N%c num of images in additional column
    #rows_test       = [N//c + (0 if N%c == 0 else 1) for c in columns_test]
    rows_test       = [np.ceil(N/c).astype(int) for c in columns_test]
    # convert to ratios via image dimensions
    ratios_test     = [(width*c)/(height*r) for c,r in zip(columns_test, rows_test)]
    #for c,r in zip(columns_test, rows_test):
    #    cv2.imshow(f'{c,r}', np.zeros((height*r,width*c),np.uint8))
    # test difference to target ratio
    ratios_diff     = [abs(r - R) for r in ratios_test]
    best_fit_ratio  = ratios_diff.index(min(ratios_diff))
    return (columns_test[best_fit_ratio],rows_test[best_fit_ratio])



def collage_pick_cols_rows_num_max_width(N, width, maxWidth):
    # N = 4
    # maxWidth = 1600
    # width = 800
    # pick col & row based on maximum available screen width
    if maxWidth < width:raise ValueError("Image wider than maximum set width!"  )
    if N == 0:          raise ValueError("List of images is empty!"             )
    # generate column number from highest to lowest and take first which is at or beyold maxWidth
    num_cols = next(
                    (cols for cols in 
                        (i for i in reversed(range(1,N + 1))) 
                    if cols*width <= maxWidth), 
                  None)
    #return (num_cols, N//num_cols + (0 if N%num_cols == 0 else 1))
    return (num_cols, np.ceil(N/num_cols).astype(int))

def create_collage(images, default_mode = 'width_max', width_max = 1600, aspect = 16/9):  
    # create a collage from list of images
    # images are padded to max size and organized in 
    pil_images = [Image.fromarray(img, mode= 'L') for img in images]
                
    max_width   = max(img.size[0] for img in pil_images)
    max_height  = max(img.size[1] for img in pil_images)
    
    padded_images = [pad_image(img, max_width, max_height) for img in pil_images]
                
    N = len(images)

    if default_mode == 'width_max':
        (C,R) = collage_pick_cols_rows_num_max_width(N, max_width, width_max)
    else:
        (C,R) = collage_pick_cols_rows_num_aspect(N, max_width, max_height, R = 16/9)
 
    #R = int(N**0.5)
    #C = (N + R - 1) // R

    # Dimensions of each image (assuming they are all the same size)
                
    # Calculate the dimensions of the collage
    collage_width = C * max_width
    collage_height = R * max_height

                
    collage = Image.new("RGB", (collage_width, collage_height))

    # Loop through the images and arrange them in rows and columns
    for r in range(R):
        for c in range(C):
            index = r * C + c
            if index < N:
                img = padded_images[index]
                x = c * max_width
                y = r * max_height
                collage.paste(img, (x, y))

    #collage.save("collage.jpg")
    collage.show()


def split_into_bins_of_max_size(array, max_size):
    array = list(array)
    return [array[i:i+max_size] for i in range(0, len(array), max_size)]



#import numpy as np
#import matplotlib.pyplot as plt

## Parameters
#center = 4
#std_dev_values = [1,2,3]  # Test different std_dev values
#x =  np.array([2, 3, 4, 5, 6])

## Plot Gaussian distributions for different std_dev values
#plt.figure(figsize=(10, 6))
#for std_dev in std_dev_values:
#    y = np.exp(-0.5 * ((x - center) / std_dev) ** 2) 
#    y /= sum(y)
#    label = f"std_dev = {std_dev}"
#    plt.plot(x, y, label=label)

#plt.title('Gaussian Distributions with Different std_dev Values')
#plt.xlabel('x')
#plt.ylabel('Probability Density')
#plt.legend()
#plt.grid()
#plt.ylim(bottom=0)
#plt.show()


#gg = collage_pick_cols_rows_num_aspect(7, 100, 50, R = 16/9)
#gg2 = collage_pick_cols_rows_num_max_width(4, 600, 1600)
a = 1
#aa = bounding_rectangle_to_contour((0,0,1,1), offset = (1,1), expand = (1,1))

r_folder = r'F:\UL Data\bubble_process_py_copy_02_11_2023\post_tests'
dict_glob = {'Field OFF Series 7':{}} # HFS 125 mT Series 1 ; Field OFF Series 7; VFS 125 mT Series 5; HFS 200 mT Series 4
pref = ["segments", "graphs", "ms-events","contorus"]
for p_folder in dict_glob:
    folder0 = os.path.join(r_folder,p_folder)
    dirs0_all = os.listdir(folder0)
    dirs0 = [item for item in dirs0_all if os.path.isdir(os.path.join(folder0, item))]
    dirs0 = [dirs0[0]]
    for s_folder in dirs0:
        N = s_folder[4:7]#'sccm100-meanFix'
        root1 = os.path.join(folder0,s_folder)
        dirs1_all = os.listdir(root1)
        dirs1 = [item for item in dirs1_all if os.path.isdir(os.path.join(root1, item))]
        #for root1, dirs1, files in os.walk(os.path.join(root0,s_folder)):
        folder1 = dirs1[0]#'00001-03000'
        #K = folder1[:5]
        #L = folder1[-5:]
        root2 = os.path.join(root1,folder1,'archives')
        for root3, dirs3, files3 in os.walk(os.path.join(root2)):
            dict_glob[p_folder][int(N)] = {prefix: os.path.join(root3, file) for file in files3 for prefix in pref if file.startswith(prefix)}



# i want to overlay (stack) short, consequtive, sequence of bubbles on top of each other
# this way i can take an average shape.
# 1)    i have to translate bubbles to one spot (first bubble centroid)
#       for it i need offsets between centroids.
# 2)    in order to overlay i have to isolate a part of an image
#       cropped part is as big as whole contour bounding boxes stacked together
for proj,proj_d in dict_glob.items():
    for sccm, params_d in proj_d.items():
        max_length = 0
        path_segments = params_d["segments"]
        with open(path_segments, 'rb') as handle:
            segments_d = pickle.load(handle)

        path_graphs = params_d["graphs"]
        with open(path_graphs, 'rb') as handle:
            graphs_d = pickle.load(handle)

        path_contorus = params_d["contorus"]
        with open(path_contorus, 'rb') as handle:
            contorus_d = pickle.load(handle)
        
        families = list(segments_d.keys())
        #families = [families[0]]
        centroids_fam = []
        for family in families:
            #centroid_fam = []
            G1, G2  = graphs_d[family]
            segments = segments_d[family]
            centroid        = lambda node: (G1.nodes[node]['cx'],G1.nodes[node]['cy'])
            contours        = lambda node: [contorus_d[node[0]][ID] for ID in node[1:]]
            contour_hull    = lambda node: cv2.convexHull(np.vstack(contours(node))) 

            segments_non_empty = [k for k,nodes in enumerate(segments) if len(nodes) > 0 ]
            for k in segments_non_empty:
                # all nodes in a segment, their centroids and bounding boxes
                nodes           = segments[k]
                centroids       = {node:centroid(node) for node in nodes}
                boundingBoxes = {t:np.array(cv2.boundingRect(contour_hull(t)), float) for t in nodes}
                
                side = 2    # take this many neighbor nodes. at max 2 + 1 + 2 in total
                # partition nodes into segments. ideally in form of center node and its few neighbors
                # like [left,center, right] <=>  [[0,1],[2],[3,4]]
                # on edges partitions should only take as much is avalable
                slices = [
                            [
                                nodes[max(0,i - side):i],
                                [nodes[i]],
                                nodes[i + 1:min(len(nodes),i + side + 1)]
                            ] 
                            
                            for i in range(len(nodes))
                        ]

                offsets = {}                    # holds centroid offsets
                images = []
                for s,nodes_slice in enumerate(slices):
                    # i have to overlay neighbors on top of center node.
                    # so i calculate centroid offset and translate
                    # additionally, you might require bigger image to fit all contours
                    # so calculate bounding box for all translated contours (bounding boxes)
                    node_center = nodes_slice[1][0]
                    nodes_all   = sum(nodes_slice,[])

                    rects_offset            = []      # store rects here
                    centroid_node_center    = np.array(centroids[node_center])

                    #blank0 = np.zeros((600,600), dtype=np.uint8)
                    #cv2.circle(blank0, centroid_first.astype(int), 3, 255, -1)
                    #[cv2.drawContours( blank0,   [contour], -1, 160, 2) for contour in contours(node_first)]

                    for node in nodes_all:
                        # find centroid offsets for contours, translate bounding boxes and store
                        offset = np.array(centroids[node]) - centroid_node_center
                        offsets[node] = offset.astype(int)
                        offset_rect = np.concatenate((offset, [0,0]))
                        rects_offset.append(boundingBoxes[node] - offset_rect)

                    #    cv2.circle(blank0, np.array(centroids[node]) .astype(int), 2, 180, -1)
                    #    [cv2.drawContours( blank0,   [contour], -1, 120, 2) for contour in contours(node)]
                    #plt.imshow(blank)
                    #plt.show()

                    # calc overlapping image dimensions and create a primer image: 
                    all_verticies   = np.vstack([x_y_w_h_to_contour(t) for t in rects_offset])
                    xt,yt,wt,ht     = cv2.boundingRect(all_verticies.astype(int))
                    offset_total    = -1* np.array([xt,yt])

                    # in fact there is separate primer for each time step, because we have to avg them
                    blank       = np.zeros((len(nodes_all) ,ht ,wt), dtype=np.uint8)

                    #[ cv2.drawContours( blank[0],   [contour], -1, 255, -2, offset = offset_total)
                    #    for contour in contours(node_center)]

                    # fill primers with offset contours
                    for t,node in enumerate(nodes_all):
                        offset = -1 * offsets[node] + offset_total
                        [
                            cv2.drawContours( blank[t],   [contour], -1, 255, -1, offset = offset)
                            for contour in contours(node)]

                    # calculate normalized weights for time steps based on gaussian distribution
                    t_center    = node_center[0]
                    t_all       = [t[0] for t in nodes_all]
                    t_std_dev   = side - 1
                    weights     = np.exp(-0.5 * ((t_all - t_center) / t_std_dev) ** 2)
                    weights     /= np.sum(weights)
                    # calculate weighted average for image of all time steps and binarize
                    blank0      = np.average(blank, axis = 0, weights = weights).astype(np.uint8)

                    _,th1 = cv2.threshold(blank0,100,255,cv2.THRESH_BINARY)
                    th1 = th1.astype(np.uint8)

                    # for debugging draw combined averaged img and threshold. original contour for ref
                    [ cv2.drawContours( blank0,   [contour], -1, 255, 2, offset = offset_total)
                        for contour in contours(node_center)]
                    [ cv2.drawContours( blank0,   [contour], -1, 0, 1, offset = offset_total)
                        for contour in contours(node_center)]

                    [ cv2.drawContours( th1,   [contour], -1, 128, 2, offset = offset_total)
                        for contour in contours(node_center)]
                    
                    # add current slice to collection for this segment.
                    images.append(cv2.hconcat((blank0,th1)))

                create_collage(images)



            a = 1
                

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()