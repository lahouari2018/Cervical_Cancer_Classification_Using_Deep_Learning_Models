import cv2
import numpy as np 
from skimage.feature import local_binary_pattern

def lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    out  = local_binary_pattern(gray,8,1,method='uniform')
    return np.uint8((out / np.max(out)) * 255)

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 50, 100)
    return edge 

def CANNY(img):
    divs = [1, 2, 4, 8]
    k = 0 
    hist = None
    
    #resize
    img = cv2.resize(img, (300, 250))
    edg = canny(img)
    w,h,_ = img.shape
    
    for level in range(0,4):
        for i in range (0, divs[k]):
            for j in range (0, divs[k]):
                mask_size = (w//divs[k], h//divs[k])    
                sub_img = edg[i*mask_size[0]:(i+1)*mask_size[0], j*mask_size[0]:(j+1)*mask_size[0]] 
                l = cv2.calcHist([sub_img], [0], None, [8], [0, 256])
                if hist is None:
                    hist = l
                else:
                    hist = np.concatenate([hist, l], axis = 0)
        k+= 1
    return hist

def PLAB(img, bins = 16):
     
    divs = [1, 2, 4]
    k = 0 
    hist = None
    img = cv2.resize(img, (300, 250))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    w,h,_ = img.shape
    for level in range(0,3):
        for i in range (0, divs[k]):
            for j in range (0, divs[k]):
                mask_size = (w//divs[k], h//divs[k]) 
                sub_img = img[i*mask_size[0]:(i+1)*mask_size[0], j*mask_size[0]:(j+1)*mask_size[0], :]
                
                #calculate in each channel 
                l = cv2.calcHist([sub_img], [0], None, [bins], [0, 256])
                a = cv2.calcHist([sub_img], [1], None, [bins], [0, 256])
                b = cv2.calcHist([sub_img], [2], None, [bins], [0, 256])
                
                if hist is None:
                    hist = np.concatenate([l, a, b], axis = 0)
                else:
                    hist = np.concatenate([hist, l, a, b], axis = 0)
        k+= 1
    return hist

def sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    mag = np.sqrt(dx**2 + dy**2)
    return np.uint8((mag / np.max(mag)) * 255)

def PHOG(img, bins = 8):
     
    divs = [1, 2, 4, 8]
    k = 0 
    hist = None
    
    #resize
    img = cv2.resize(img, (300, 250))
    edg = sobel(img)
    w,h,_ = img.shape
    
    for level in range(0,4):
        for i in range (0, divs[k]):
            for j in range (0, divs[k]):
                mask_size = (w//divs[k], h//divs[k])    
                sub_img = edg[i*mask_size[0]:(i+1)*mask_size[0], j*mask_size[0]:(j+1)*mask_size[0]] 
                l = cv2.calcHist([sub_img], [0], None, [bins], [0, 256])
                if hist is None:
                    hist = l
                else:
                    hist = np.concatenate([hist, l], axis = 0)
        k+= 1
    return hist

def PLBP(img, bins = 10):
     
    divs = [1, 2, 4, 8]
    k = 0 
    hist = None
    #resize
    img = cv2.resize(img, (300, 250))
    lbp_image= lbp(img)
    w,h,_ = img.shape
    
    for level in range(0,4):
        for i in range (0, divs[k]):
            for j in range (0, divs[k]):
                mask_size = (w//divs[k], h//divs[k])     
                sub_img = lbp_image[i*mask_size[0]:(i+1)*mask_size[0], j*mask_size[0]:(j+1)*mask_size[0]] 
                l = cv2.calcHist([sub_img], [0], None, [bins], [0, 256])
                if hist is None:
                    hist = l
                else:
                    hist = np.concatenate([hist, l], axis = 0)
        k+= 1
    return hist