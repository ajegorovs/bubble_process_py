import os
import sys
import cv2

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

def cropGUI(filename, width=1200, height=600):
    global select_coords, selecting, image, clone
    #filename = r'post_tests\cropTest.png'#sys.argv[1]
    #basename = os.path.basename(filename)
    basename = 'cropGUI'
    cropped_basename = 'cropped'
    image = cv2.imread(filename)
    h, w = image.shape[:2]
    scale = 1
    if h > 600 or w > 1200:
        alpha,beta = width/w,height/h
        scale = min(alpha, beta)
    image = resizeImage(image,scale)

    # The cropped image will be saved with this filename.
    #cropped_filename = os.path.splitext(filename)[0] + '_sq.png'
    #cropped_basename = os.path.basename(cropped_filename)
    # Store a clone of the original image (without selected region annotation).
    clone = image.copy() 
    print('PRESS C to CROP! Esc to close after')
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
    # Load the image and get its filename without path and dimensions.
    
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
        x0,y0 = select_coords[0]
        x1,y1 = select_coords[1]
        cropped_image = clone[y0:y1, x0:x1]
        cv2.imshow(cropped_basename, cropped_image) 
        #cv2.imwrite(cropped_filename, cropped_image)
        # Wait until any key press.
    
    k = cv2.waitKey(0)
    if k == 27:  # close on ESC key
        cv2.destroyAllWindows()

    return [tuple([int(x/scale) for x in a]) for a in  select_coords]
