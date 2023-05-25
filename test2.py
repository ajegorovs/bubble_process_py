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

outPath = r'.\archives\HFS 200 mT Series 4\sccm100-meanFix\00001-05000\csv_img-HFS 200 mT Series 4-sccm100-meanFix-00001-05000'
inPath  = r'.\archives\HFS 200 mT Series 4\sccm100-meanFix\00001-05000\HFS 200 mT Series 4-sccm100-meanFix-00001-05000-00001-05000.pickle'

with open(inPath, 'rb') as handle:
    data = pickle.load(handle)

for i in range(data.shape[0]):
    cv2.imwrite(os.path.join(outPath,str(i).zfill(4)+".png") ,data[i])

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()