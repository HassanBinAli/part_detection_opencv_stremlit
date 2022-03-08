import cv2
import numpy as np

def imageread():
    img = cv2.imread("obj1(4-circles).png")
    img1 = cv2.imread("obj2(2-circles).png")
    img2 = cv2.imread("obj3(2-circles-N).png")
    img3 = cv2.imread("obj4(0-circles).png")
    return img, img1, img2, img3




def get_contours(imgGC):
    imgGray = cv2.cvtColor(imgGC, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCorner = cv2.Canny(imgGray, 10, 10)
    #imgBlank = np.zeros_like(imgGray)
    
    imgContour = imgGC.copy()
    #imgStacked = np.hstack((imgGray, imgBlur, imgCorner))

    contours, hierarchy = cv2.findContours(imgCorner, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   # last argument shows that
    # all contours will be stored in contours
    # second argument shows that contours are considered from outermost stroke of the shape
    area_dict = {
        'Square_Area': 0,
        'Circle_Area': 0
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            peri = cv2.arcLength(cnt, True)   # used to find perimeter (length of contour boundary)
    
            # to find number of corners in our contour
            approx = cv2.approxPolyDP(cnt, 0.005*peri, True)   # 0.2 is the resolution
            # this will return the coordinate points of each corner related to specific contour
        
            obj_cor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            
            if obj_cor == 4:
                aspratio = w/float(h)
                if (aspratio > 0.95) and (aspratio < 1.05):
                    obj_type = "Sqr"
                    area_dict['Square_Area'] += area
                else:
                    obj_type = "rect"
                    area_dict['Square_Area'] += area
            elif obj_cor > 4:
                obj_type = "Circle"
                area_dict['Circle_Area'] += area
            else:
                obj_type = "None"

    area_dict['Square_Area'] /= 2
    area_dict['Circle_Area'] /= 2
    return area_dict['Square_Area'], area_dict['Circle_Area']

    
