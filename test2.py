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

#def centroid_area_cmomzz(contour):
#    m = cv2.moments(contour)
#    area = int(m['m00'])
#    cx0, cy0 = m['m10'], m['m01']
#    centroid = np.array([cx0,cy0])/area
#    c_mom_xx, c_mom_yy = m['mu20'], m['mu02']
#    c_mom_zz = int((c_mom_xx + c_mom_yy))
#    return  centroid, area, c_mom_zz

#import cv2
#import numpy as np

## Create a black image as the background
#image0 = np.zeros((200, 200), dtype=np.uint8)

## Define the center and radius of the circle
#center = (100, 100)
#radius = 50

## Draw the filled circle on the image
#cv2.circle(image0, center, radius, (255, 255, 255), -1)

## Find contours in the image
#contours, hierarchy = cv2.findContours(image0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

## Extract the contour of the filled circle (assuming there is only one contour)
#circle_contour = contours[0]

#mask = image0.copy()
#M = cv2.moments(mask)

## Calculate the centroid coordinates
#cx = int(M['m10'] / M['m00'])
#cy = int(M['m01'] / M['m00'])

## Calculate the moment of inertia
#moment_of_inertia = 0.0
#for y in range(mask.shape[0]):
#    for x in range(mask.shape[1]):
#        if mask[y, x] == 255:  # Inside the object
#            squared_distance = ((x - cx) ** 2 + (y - cy) ** 2)
#            moment_of_inertia += squared_distance

#print("Moment of inertia:", moment_of_inertia)

#moment_of_inertia_circle = moment_of_inertia

## Create a black image as the background
#image = np.zeros((200, 200), dtype=np.uint8)

## Define the top-left and bottom-right corners of the square
#top_left = (75, 75)
#bottom_right = (125, 125)

## Draw the filled square on the image
#cv2.rectangle(image, top_left, bottom_right, 255, -1)
##cv2.imshow("Image", image)
## Find contours in the image
#contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

## Extract the contour of the filled square (assuming there is only one contour)
#square_contour = contours[0]

## Calculate the moments of the filled shape
#mask = image.copy()
#M = cv2.moments(mask)

## Calculate the centroid coordinates
#cx = int(M['m10'] / M['m00'])
#cy = int(M['m01'] / M['m00'])

## Calculate the moment of inertia
#moment_of_inertia = 0.0
#for y in range(mask.shape[0]):
#    for x in range(mask.shape[1]):
#        if mask[y, x] == 255:  # Inside the object
#            squared_distance = ((x - cx) ** 2 + (y - cy) ** 2)
#            moment_of_inertia += squared_distance

#print("Moment of inertia:", moment_of_inertia)
#moment_of_inertia_rect = moment_of_inertia
#s=  50
#hs = 50/2
#ar0 = s**2
## 4 x quater segment
#cm0 = s**4/6

#aa0 = centroid_area_cmomzz(square_contour)


## Display the image with the filled circle
##cv2.imshow("Image", image)
#ar = np.pi*radius**2
#aa = centroid_area_cmomzz(circle_contour)
#cm = 1/4*radius**4 * 2* np.pi 
#moment_of_inertia_circle
#a = 1


from collections import deque
import numpy as np

class TrajectoryBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def append(self, trajectory):
        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (2,):
            raise ValueError("Trajectory should be a 1x2 NumPy array.")
        
        if len(self.buffer) == self.max_size:
            self.buffer.popleft()
        self.buffer.append(trajectory)

    def get_data(self):
        return np.array(self.buffer)

    def get_last_trajectory(self):
        if len(self.buffer) > 0:
            return self.buffer[-1]
        else:
            return None

d = deque(maxlen=5)
d.extend(np.array([[0,0],[1,1],[2,2],[3,3],[4,4]]))
print(d)
# deque([1, 2, 3, 4, 5], maxlen=5)
d.append([5,5])
print(d)
# deque([2, 3, 4, 5, 10], maxlen=5)

# Create the TrajectoryBuffer with a maximum size of 3
trajectory_buffer = TrajectoryBuffer(max_size=3)

# Append 1x2 arrays to the buffer
trajectory_buffer.append(np.array([1, 2]))
trajectory_buffer.append(np.array([3, 4]))
trajectory_buffer.append(np.array([5, 6]))

# Check the buffer content
print(trajectory_buffer.get_data())

# Append a new 1x2 array, which will trigger the shift in the buffer
trajectory_buffer.append(np.array([7, 8]))

# Check the buffer content after the shift
print(trajectory_buffer.get_data())

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()