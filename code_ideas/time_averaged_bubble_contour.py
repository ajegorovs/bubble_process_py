import os, pickle, cv2
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw 
def bounding_rectangle_to_contour(rectangle_params, offset=(0, 0), expand=(0, 0)):
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


def pad_image(image, width, height):
                    
    img_width, img_height = image.size
    left_padding = (width - img_width) // 2
    top_padding = (height - img_height) // 2
    #right_padding = width - img_width - left_padding
    #bottom_padding = height - img_height - top_padding


    padded_img = Image.new(image.mode, (width, height), 0)  # white background
    padded_img.paste(image, (left_padding, top_padding))

                    
    #new_width = width - 2
    #new_height = height - 2
    draw = ImageDraw.Draw(padded_img)
    draw.rectangle([(1,1),(width - 1,height - 1)], width = 1, outline=128)
    #border_image = Image.new(image.mode, (new_width, new_height),255)
    #border_image.paste(padded_img, (1, 1,new_width,new_height))

    return padded_img 

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
        families = [families[4]]
        centroids_fam = []
        for family in families:
            #centroid_fam = []
            G1, G2  = graphs_d[family]
            segments = segments_d[family]
            centroid        = lambda node: (G1.nodes[node]['cx'],G1.nodes[node]['cy'])
            contours        = lambda node: [contorus_d[node[0]][ID] for ID in node[1:]]
            contour_hull    = lambda node: cv2.convexHull(np.vstack(contours(node))) 
            boundingBoxes = {}
            segments_non_empty = [k for k,nodes in enumerate(segments) if len(nodes) > 0 ]
            for k in segments_non_empty:
                nodes = segments[k]
                boundingBoxes[k] = {t:np.array(cv2.boundingRect(contour_hull(t)), float) for t in nodes}
                # define averaging slices
                N = 7
                N = min(N, len(nodes))   # if less nodes than N it will put all nodes as a subset.
                slices = [nodes[i:i + N] for i in range(len(nodes) - N + 1)]
                #slices = [[nodes[10],nodes[16]]]
                slices = slices[::3]
                centroids = {node:centroid(node) for node in nodes}
                subsets_rect = []
                offsets = {}                    # holds centroid offsets
                #f, axarr = plt.subplots(len(slices),1) 
                images = []
                for s,nodes_slice in enumerate(slices):
                    node_first,*node_rest = nodes_slice
                    rect_first = boundingBoxes[k][node_first]
                    rects_offset = [rect_first] # holds offset rectangles
                    centroid_first = np.array(centroids[node_first])

                    #blank0 = np.zeros((600,600), dtype=np.uint8)
                    #cv2.circle(blank0, centroid_first.astype(int), 3, 255, -1)
                    #[cv2.drawContours( blank0,   [contour], -1, 160, 2) for contour in contours(node_first)]

                    for node in node_rest:
                        offset = np.array(centroids[node]) - centroid_first
                        offsets[node] = offset.astype(int)
                        offset_rect = np.concatenate((offset, [0,0]))
                        rects_offset.append(boundingBoxes[k][node] - offset_rect)

                    #    cv2.circle(blank0, np.array(centroids[node]) .astype(int), 2, 180, -1)
                    #    [cv2.drawContours( blank0,   [contour], -1, 120, 2) for contour in contours(node)]
                    
                    #plt.imshow(blank)
                    #plt.show()

                    all_verticies   = np.vstack([bounding_rectangle_to_contour(t) for t in rects_offset])
                    xt,yt,wt,ht     = cv2.boundingRect(all_verticies.astype(int))
                    offset_total    = -1* np.array([xt,yt])
                    blank           = np.zeros((len(nodes_slice),ht,wt), dtype=np.uint8)

                    [ cv2.drawContours( blank[0],   [contour], -1, 255, -2, offset = offset_total)
                        for contour in contours(node_first)]

                    #cv2.imshow('a', blank)
                    
                    for t,node in enumerate(node_rest):
                        offset = -1 * offsets[node] + offset_total
                        [
                            cv2.drawContours( blank[t +1],   [contour], -1, 255, -1, offset = offset)
                            for contour in contours(node)]
                    blank0 = np.mean(blank, axis = 0).astype(np.uint8)
                    _,th1 = cv2.threshold(blank0,100,255,cv2.THRESH_BINARY)
                    th1 = th1.astype(np.uint8)
                    #plt.figure()
                    images.append(cv2.hconcat((blank0,th1)))
                    #subplot(r,c) provide the no. of rows and columns
                    

                    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                    #axarr[s].imshow(cv2.hconcat((blank0,th1)))
                    

                    plt.show()
            

                
                pil_images = [Image.fromarray(img, mode= 'L') for img in images]
                


                # create a list of padded images
                max_width   = max(img.size[0] for img in pil_images)
                max_height  = max(img.size[1] for img in pil_images)
                #pil_images[0].show()
                padded_images = [pad_image(img, max_width, max_height) for img in pil_images]
                #padded_images[0].show()
                #for img in padded_images:
                #    img.show()
                # Calculate the number of rows (R) and columns (C) for the collage
                N = len(images)
                R = int(N**0.5)
                C = (N + R - 1) // R

                # Dimensions of each image (assuming they are all the same size)
                
                # Calculate the dimensions of the collage
                collage_width = C * max_width
                collage_height = R * max_height

                # Create an empty canvas for the collage
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

                # Save or display the collage
                #collage.save("collage.jpg")
                collage.show()
                a = 1
                #max_width = max(img.shape[0] for img in images)
                #max_height = max(img.shape[1] for img in images)

                #sqrt_N = np.sqrt(len(images))

                #R = int(sqrt_N)
                #C = int(sqrt_N)

                #collage_width = C * max_width
                #collage_height = R * max_height

                #collage = Image.new("RGB", (collage_width, collage_height))

                #def pad_image(image, width, height):
                #    img_width, img_height = image.size
                #    left_padding = (width - img_width) // 2
                #    top_padding = (height - img_height) // 2
                #    #right_padding = width - img_width - left_padding
                #    #bottom_padding = height - img_height - top_padding

                #    # Pad the image
                #    padded_img = Image.new(image.mode, (width, height), (255, 255, 255))  # White background
                #    padded_img.paste(image, (left_padding, top_padding))

                #    return padded_img

                ## Create a list of padded images
                #padded_images = [pad_image(img, max_width, max_height) for img in images]


                ## Save or display the collage
                ##collage.save("collage.jpg")
                #collage.show()

                a = 1

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()