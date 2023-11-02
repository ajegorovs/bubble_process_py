import itertools, pickle, cv2
import numpy as np
dm = 500
numP = 9
dPhi = 2*np.pi/numP
img = np.zeros((dm,dm,3),np.uint8)
#t = np.arange(0,2*np.pi+0.001,2*np.pi/numP)
t1 = np.arange(0,2*np.pi+0.0000000001,dPhi)#;print(t1/np.pi)
t2 =np.flip(t1)
#t = np.vstack((t1,t2)).flatten()
rad = int(0.7*dm/2)
rad2 = int(0.2*dm/2)
#crvP = [[[angl,rad]] for angl in t]
crvP1 = [[[angl,rad]] for angl in t1]
crvP2 = [[[angl,rad2]] for angl in t2]
crvP = crvP1 + crvP2
crvP2 = crvP + [[[0,rad]]]

crvD = np.array([[[int(rad*np.cos(angl)+dm/2),int(rad*np.sin(angl)+dm/2)]] for [[angl,rad]] in crvP],dtype=int).reshape((-1,2))
cv2.drawContours(  img,   [crvD], -1, (125,250,120), 1)
print(f'area = {int(np.pi*(rad)**2-np.pi*(rad2)**2)}')
area = 0
Ravg = 0
for i,[[angl,rad]] in enumerate(crvP):
    [[angl2,rad2]] = crvP2[i+1]
    dAng = angl2 - angl
    #dAngDeg = dAng * 180/np.pi;print(np.around(dAngDeg,0))
    #area = area + 0.5*(rad**2 *dAng)
    dA = 0.5*rad*rad2*np.sin(dAng)
    area = area + dA
    Ravg = Ravg + 2/3*(0.5*(rad + rad2)) * dA
Ravg = Ravg/area
print(f'area2 = {area}')
print(f'Ravg = {Ravg/(dm/2)}')
cv2.circle(img, (int(Ravg+dm/2),int(dm/2)), 6, (255,0,0), -1)
cv2.imshow('a',img)

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()
