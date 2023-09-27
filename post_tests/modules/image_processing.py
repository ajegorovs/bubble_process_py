

import cv2

def convertGray2RGB(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image.copy(),cv2.COLOR_GRAY2RGB)
    else:
        return image

def undistort(image, mapXY):
    return cv2.remap(image,mapXY[0],mapXY[1],cv2.INTER_LINEAR)
