import glob, csv, cv2, numpy as np, pickle, os
#imgPath = r'test\csv_img-Field OFF Series 7-sccm100-meanFix-00001-05000'
#imageLinks = glob.glob(imgPath + "**/*.png", recursive=True) 
#a = 1
#data = []
#with open(r'test\csv-Field OFF Series 7-sccm100-meanFix-00001-05000.csv', 'r', newline='') as file:
#    # Create a CSV writer
#    reader = csv.reader(file)
#    for row in reader:
#        temp = int(row[0]),int(row[1]),eval(row[2]),int(row[3])
#        data.append(temp)
#a = 1
#maxFrame = data[-1][0]
#cntr = 0
#locCntr = 0
#for frame, localID, centroid, area in data:
#    if frame > cntr:
#        locCntr = 0
#        cntr = frame
#        cv2.imwrite(".\\test\\output\\"+str(frame-1).zfill(4)+".png" ,pic)
#    if locCntr == 0:
#        pic = np.uint8(cv2.imread(imageLinks[cntr],1))
#    if frame == cntr:
#        cv2.circle(pic, centroid, 10, (0,0,255), 2)
#        locCntr +=1
#    #if frame > 3: break
#    a = 1
#lnk = r'C:\Users\mhd01\Desktop\mag\traj family\off'
#imageLinks = glob.glob(lnk + "**/*.png", recursive=True) 
#imgs = [np.uint8(cv2.imread(lnk,1)) for lnk in imageLinks]
#imgShapes =  np.array([a.shape[:-1] for a in imgs])
#maxX,maxY = [max(a) for a in imgShapes.T]
#new_image_height = maxX
#new_image_width = maxY
#channels = 3
#color = (255,255,255)
#for i in range(len(imgs)):
#    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
#    img = imgs[i]
#    old_image_height,old_image_width = imgShapes[i]
#    # compute center offset
#    x_center = (new_image_width - old_image_width) // 2
#    y_center = (new_image_height - old_image_height) // 2

#    # copy img image into center of result image
#    result[y_center:y_center+old_image_height, 
#           x_center:x_center+old_image_width] = img
#    cv2.imwrite(os.path.join(lnk,f"padded_{os.path.basename(imageLinks[i])}.png"), result)
# view result
#cv2.imshow("result", result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Import the required libraries
import os
import sys
import cv2

import numpy as np, pickle

#with open('./asd.pickle', 'rb') as handle:
#    a,b = pickle.load(handle)
#c,s,an = a
#rect1 = (tuple(map(int,c)), tuple(map(int,s)), int(an))

#c,s,an = b
#rect2 = (tuple(map(int,c)), tuple(map(int,s)), int(an))
#print(rect1 == a)
#print(rect2 == b)
#interType, points  = cv2.rotatedRectangleIntersection(rect1, rect2)
#interType, points  = cv2.rotatedRectangleIntersection(a, b)


## Create the graph (example)
#G = nx.Graph()
#G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)])

## Define the path
#path = [1, 2, 3, 4, 5]

## Create a dictionary to store edge colors
#edge_colors = []

## Iterate over the edges and check if they belong to the path
#for u, v in G.edges():
#    if u in path and v in path:
#        edge_colors.append((1,0,0))
#    else:
#        edge_colors.append('black')


## Draw the graph with colored edges
#pos = nx.spring_layout(G)
#nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color=edge_colors)
#plt.show()
#a =1





k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()