
import cv2, numpy as np


# ive tested c_mom_zz in centroid_area_cmomzz. it looks correct from definition m_zz = sum_x,y[r^2] = sum_x,y[(x-x0)^2 + (y-y0)^2]
# which is the same as sum_x,y[(x-x0)^2] + sum_x,y[(y-y0)^2] = mu(2,0) + mu(0,2), and from tests on images.

def centroid_area_cmomzz(contour):
    m = cv2.moments(contour)
    area = int(m['m00'])
    cx0, cy0 = m['m10'], m['m01']
    centroid = np.array([cx0,cy0])/area
    c_mom_xx, c_mom_yy = m['mu20'], m['mu02']
    c_mom_zz = int((c_mom_xx + c_mom_yy))
    return  centroid, area, c_mom_zz

def centroid_area(contour):
    m = cv2.moments(contour)
    area = int(m['m00'])
    cx0, cy0 = m['m10'], m['m01']
    centroid = np.array([cx0,cy0])/area
    return  centroid, area



# not used ============

def compositeRectArea(rects, rectsParamsArr):
    if len(rects) > 1:
        # find region where all rectangles live. working smaller region should be faster.
        minX, minY = 100000, 100000                 # Initialize big low and work down to small
        maxX, maxY = 0, 0                           # Initialize small high  and work up to big

        for ID in rects:
            x,y,w,h = rectsParamsArr[ID]
            minX = min(minX,x)
            minY = min(minY,y)
            maxX = max(maxX,x+w)
            maxY = max(maxY,y+h)

        width = maxX - minX                         # Get composite width
        height = maxY - minY                        # Get composite height
        blank = np.zeros((height,width),np.uint8)   # create a smaller canvas
        for ID in rects:                            # draw masks with offset to composite edge
            x,y,w,h = rectsParamsArr[ID]
            cntr = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],int).reshape(-1,1,2)
            cv2.drawContours( blank, [cntr], -1, 255, -1, offset = (-minX,-minY))
         
        return int(np.count_nonzero(blank))         # count pixels = area

    else:
        x,y,w,h = rectsParamsArr[rects[0]]
        return int(w*h)