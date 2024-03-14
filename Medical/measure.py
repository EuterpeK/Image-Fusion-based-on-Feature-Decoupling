import cv2
import skimage.measure 
import numpy as np
import math
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


def avgGradient(path):
    image = Image.open(path).convert('L')
    image = np.array(image)/255.0
    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0

    for i in range(width):
        for j in range(heigt):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds

    imageAG = tmp / (width * heigt)
    return round(imageAG,3)

def spatialF(path):
    image = Image.open(path).convert('L')
    image = np.array(image)/255
    M = image.shape[0]
    N = image.shape[1]

    cf = 0
    rf = 0
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2

    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)

    return round(SF,2)

def getMI(im1,im2):
    
    #im1 = im1.astype('float')
    #im2 = im2.astype('float')

    hang, lie = im1.shape
    count = hang*lie
    N = 256

    h = np.zeros((N,N))

    for i in range(hang):
        for j in range(lie):
            h[im1[i,j],im2[i,j]] = h[im1[i,j],im2[i,j]]+1

    h = h/np.sum(h)

    im1_marg = np.sum(h,axis=0)
    im2_marg = np.sum(h, axis=1)

    H_x = 0
    H_y = 0

    for i in range(N):
        if(im1_marg[i]!=0):
            H_x = H_x + im1_marg[i]*math.log2(im1_marg[i])

    for i in range(N):
        if(im2_marg[i]!=0):
            H_x = H_x + im2_marg[i]*math.log2(im2_marg[i])

    H_xy = 0

    for i in range(N):
        for j in range(N):
            if(h[i,j]!=0):
                H_xy = H_xy + h[i,j]*math.log2(h[i,j])

    MI = H_xy-H_x-H_y

    return MI


# 数组旋转180度
def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

#相当于matlab的Conv2
def convolution(k, data):
    k = flip180(k)
    data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    n,m = data.shape
    img_new = []
    for i in range(n-2):
        line = []
        for j in range(m-2):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    return np.array(img_new)


#用h3对strA做卷积并保留原形状得到SAx，再用h1对strA做卷积并保留原形状得到SAy
#matlab会对图像进行补0，然后卷积核选择180度
#gA = sqrt(SAx.^2 + SAy.^2);
#定义一个和SAx大小一致的矩阵并填充0定义为aA，并计算aA的值
def getArray(img):
    #Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    SAx = convolution(h3,img)
    SAy = convolution(h1,img)
    gA = np.sqrt(np.multiply(SAx,SAx)+np.multiply(SAy,SAy))
    n, m = img.shape
    aA = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(SAx[i,j]==0):
                aA[i,j] = math.pi/2
            else:
                aA[i, j] = math.atan(SAy[i,j]/SAx[i,j])
    return gA,aA


#the relative strength and orientation value of GAF,GBF and AAF,ABF;
def getQabf(aA,gA,aF,gF):
    #model parameters 模型参数
    L = 1;
    Tg = 0.9994;
    kg = -15;
    Dg = 0.5;
    Ta = 0.9879;
    ka = -22;
    Da = 0.8;
    
    n, m = aA.shape
    GAF = np.zeros((n,m))
    AAF = np.zeros((n,m))
    QgAF = np.zeros((n,m))
    QaAF = np.zeros((n,m))
    QAF = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(gA[i,j]>gF[i,j]):
                GAF[i,j] = gF[i,j]/gA[i,j]
            elif(gA[i,j]==gF[i,j]):
                GAF[i, j] = gF[i, j]
            else:
                GAF[i, j] = gA[i,j]/gF[i, j]
            AAF[i,j] = 1-np.abs(aA[i,j]-aF[i,j])/(math.pi/2)

            QgAF[i,j] = Tg/(1+math.exp(kg*(GAF[i,j]-Dg)))
            QaAF[i,j] = Ta/(1+math.exp(ka*(AAF[i,j]-Da)))

            QAF[i,j] = QgAF[i,j]*QaAF[i,j]

    return QAF

def getQABF(vi, ir, fusion):
    #if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    #如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = cv2.imread(vi, 0).astype(np.float32)
    strB = cv2.imread(ir, 0).astype(np.float32)
    strF = cv2.imread(fusion, 0).astype(np.float32)

    #对strB和strF进行相同的操作
    gA,aA = getArray(strA)
    gB,aB = getArray(strB)
    gF,aF = getArray(strF)

    QAF = getQabf(aA,gA,aF,gF)
    QBF = getQabf(aB,gB,aF,gF)


    #计算QABF
    deno = np.sum(gA+gB)
    nume = np.sum(np.multiply(QAF,gA)+np.multiply(QBF,gB))
    output = nume/deno
    
    return output

if __name__ == '__main__':
	# 这样读取后已经是灰度图
    en = []
    sf = []
    mi = []
    mg = []
    qabf = []
    for i in range(3):
        if i<10:
            fusionpath = 'result/epoch0009/F9_0'+str(i) + '.bmp'
        else: 
            fusionpath = 'result/epoch0009/F9_'+str(i) + '.bmp'
        sourceVIpath = 'Test_vi/' + str(i+1) + '.bmp'
        sourceIRpath = 'Test_ir/' + str(i+1) + '.bmp'
        fusionImage = cv2.imread(fusionpath, 0)
        sourceImage = cv2.imread(sourceVIpath, 0)
        en.append(skimage.measure.shannon_entropy(fusionImage, base=2))
        sf.append(spatialF(fusionpath))
        mi.append(getMI(fusionImage, sourceImage))
        mg.append(avgGradient(fusionpath))
        qabf.append(getQABF(sourceVIpath, sourceIRpath, fusionpath))

    en = np.array(en)
    sf = np.array(sf)
    mi = np.array(mi)
    mg = np.array(mg)
    qabf = np.array(qabf)
    print('epoch:9')
    print('en:',en.mean())
    print('sf:',sf.mean())
    print('mi:',mi.mean())
    print('mg:',mg.mean())
    print('qabf:',qabf.mean())
    # print('SF:',spatialF(image))
