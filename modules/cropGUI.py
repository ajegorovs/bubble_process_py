import cv2, numpy as np
from threading import Thread

# The coordinates defining the square selected will be kept in this list.
select_coords = [(0,0),(0,0)]
# While we are in the process of selecting a region, this flag is True.
selecting = False
image  = 1; clone = 1

def resizeImage(img,frac):
        width = int(img.shape[1] * frac)
        height = int(img.shape[0] * frac)
        return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def resizeToMaxHW(image,width=1200,height=600):
    w0 = image.shape[1] 
    h0 = image.shape[0]
    alpha,beta = width/w0,height/h0
    if w0*beta > width:
        return resizeImage(image,alpha)
    else:
        return resizeImage(image,beta)

def cropGUI(img, width=1200, height=600):
    global select_coords, selecting, image, clone
    basename = 'cropGUI: drag crop rectangle. press "c" to crop'
    cropped_basename = 'cropGUI: press "Esc" to close'
    image = img
    h, w = image.shape[:2]
    scale = 1
    if h > width or w > width:
        alpha,beta = width/w,height/h
        scale = min(alpha, beta)
    image = resizeImage(image,scale)

    # Store a clone of the original image (without selected region annotation).
    clone = image.copy() 
    #print('Drag mouse to expand rectangle diagonal. PRESS C to CROP! Esc to close after')
    def region_selection(event, x, y, flags, param): 
        """Callback function to handle mouse events related to region selection."""
        global select_coords, selecting, image, clone

        if event == cv2.EVENT_LBUTTONDOWN: 
            # Left mouse button down: begin the selection.
            # The first coordinate pair is the centre of the square.
            select_coords[0] = (x, y)
            selecting = True

        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            image = clone.copy()
            select_coords[1] = (x, y)
            cv2.rectangle(image, select_coords[0], select_coords[1], (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP: 
            # Left mouse button up: the selection has been made.
            select_coords[1] = (x, y)
            selecting = False
    
    # Name the main image window after the image filename.
    cv2.namedWindow(basename) 
    cv2.setMouseCallback(basename, region_selection)

    # Keep looping and listening for user input until 'c' is pressed.
    while True: 
        # Display the image and wait for a keypress 
        cv2.imshow(basename, image) 
        key = cv2.waitKey(1) & 0xFF
        # If 'c' is pressed, break from the loop and handle any region selection.
        if key == ord("c"): 
            break

    # Did we make a selection?
    if select_coords[1] != (0,0): 

        # Crop the image to the selected region and display in a new window.
        xes     = [a[0] for a in select_coords]
        yses    = [a[1] for a in select_coords]

        xmin, xmax = min(xes)   , max(xes)
        ymin, ymax = min(yses)  ,   max(yses)
        cropped_image = clone[ymin:ymax, xmin:xmax]
        cv2.imshow(cropped_basename, cropped_image) 
    
    k = cv2.waitKey(0)
    if k == 27:  # close on ESC key
        cv2.destroyAllWindows()
    xyhw = np.array([xmin, ymin, xmax - xmin, ymax - ymin])/scale
    p1p2 = np.array([[xmin, ymin],[xmax, ymax]])/scale
    
    return np.uint(xyhw), np.uint(p1p2)
    #return [tuple([int(x/scale) for x in a]) for a in  select_coords]

class crop_gui_thread(Thread):
    def __init__(self, image, width = 1200, height = 600):
        Thread.__init__(self)
        self.image = image
        self.width = width
        self.height = height
        self.result = None
    # function executed in a new thread
    def run(self):
        self.result = cropGUI(self.image, self.width, self.height)
# >>>> example
# base_img            = cv2.imread(cropMaskPath, 1)
# check_ready_thread  = crop_gui_thread(base_img)#Thread(target=cropGUI, args = (base_img))
# check_ready_thread.start()
# for i in range(1,5):
#     time.sleep(1)
#     print('asd')

# check_ready_thread.join()
# [X, Y, W, H], (p1,p2) = check_ready_thread.result