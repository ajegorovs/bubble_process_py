import cv2, numpy as np, sys, time
from multiprocessing import shared_memory, Manager, Pool, Event
from threading import Thread


path_modues = r'.\modules'      # os.path.join(mainOutputFolder,'modules')
sys.path.append(path_modues) 

from cropGUI import (cropGUI, crop_gui_thread)
from image_processing import (convertGray2RGB)

cropMaskPath        = r'C:\Users\Hot Mexican\source\repos\bubble_process_py\post_tests\cropTest.png'
cropMaskPath_edit   = r'C:\Users\Hot Mexican\source\repos\bubble_process_py\post_tests\cropTest_edit.png'

base_img            = cv2.imread(cropMaskPath, 1)
check_ready_thread  = crop_gui_thread(base_img)
check_ready_thread.start()


while check_ready_thread.is_alive():
    for i in range(4):
        print(f"Thread is still running{'.' * i}{' ' * (4 - i)}", end='\r')
        time.sleep(0.5)
check_ready_thread.join()
[X, Y, W, H], (p1,p2) = check_ready_thread.result

cropMask        = np.zeros_like(base_img)

cv2.rectangle(base_img  , p1, p2, [0,0,255], -1)
cv2.imwrite(  cropMaskPath  , cropMask)

a = 1


k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()


# def foo(x):
#     time.sleep(x)
#     return x

# thread = Thread(target=foo, args = (1,))