import cv2
import skimage.measure 
from sklearn import metrics
import numpy as np
import math
from PIL import Image
import argparse
import os
from skimage.metrics import structural_similarity
import time

def SCD(ir, vi, fusion):
    # SCD = r1 + r2
    r1 = np.corrcoef(fusion-vi, ir)
    r2 = np.corrcoef(fusion-ir, vi)
    return r1[0,1]+r2[0,1]

def CC(a,b): 
    ret = np.corrcoef(a,b) 
    return ret[0,1]


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

def getQabf(aA,gA,aF,gF):
    #model parameters 模型参数
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    
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


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type=str, default='auto')
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--epoch', type=int, default=0)

  args, _ = parser.parse_known_args()
  return args


if __name__ == '__main__':
	# 这样读取后已经是灰度图
    args = get_args()
    # basepath = args.dir

    # for epoch in range(args.epochs):
    for epoch in range(args.epochs):
        # if epoch!=args.epoch:
        #     continue
        en = []
        ssim = []
        scd = []
        cc = []
        qabf = []
        for i in range(18):
            start = time.time()
            fusionpath = 'eval/'+ args.dir + '/F'+str(epoch+1) +'_'+str(i) + '.bmp'
            sourceVIpath = '../Multi-focus/Test_vi/' + str(i+1) + '.bmp'
            sourceIRpath = '../Multi-focus/Test_ir/' + str(i+1) + '.bmp'
            fusionImage = cv2.imread(fusionpath, 0)
            sourceVI_Image = cv2.imread(sourceVIpath, 0)
            sourceIR_Image = cv2.imread(sourceIRpath, 0)


            en.append(skimage.measure.shannon_entropy(fusionImage, base=2))
            # mi.append(metrics.mutual_info_score(sourceVI_Image.reshape(-1), fusionImage.reshape(-1)))
            ssim.append(structural_similarity(sourceIR_Image, fusionImage))
            ssim.append(structural_similarity(sourceVI_Image, fusionImage))
            scd.append(abs(SCD(sourceIR_Image, sourceVI_Image, fusionImage)))
            # cc.append(metrics.matthews_corrcoef(sourceVI_Image.reshape(-1), fusionImage.reshape(-1)))
            cc.append(CC(sourceVI_Image, fusionImage))
            cc.append(CC(sourceIR_Image, fusionImage))
            qabf.append(getQABF(sourceVIpath, sourceIRpath,fusionpath))
            end = time.time()
            print("Measuring [%d] success,Testing time is [%f]"%(i+1,end-start))
        cc = [n for n in cc if not(math.isnan(n))]
        en = np.array(en)
        scd = np.array(scd) 
        cc = np.array(cc) 
        ssim = np.array(ssim)
        qabf = np.array(qabf)

        print("model:   PMGI",)
        print('en:      ', str(7.5231))
        print('scd:     ', str(0.8028))
        print('cc:      ', str(0.9607))
        print('ssim:    ', str(0.8478))
        print('qabf:    ', str(0.5654))

        print('\nmodel: ', str(epoch+1))
        print('en:      {:.4f}'.format(en.mean()))
        print('scd:     {:.4f}'.format(scd.mean()))
        print('cc:      {:.4f}'.format(cc.mean()))
        print('ssim:    {:.4f}'.format(ssim.mean()))
        print('qabf:    {:.4f}'.format(qabf.mean()))

        with open(args.dir + '.txt','a') as f:
            f.write('\n' + str(epoch+1) +':\n' )
            f.write('en:  '+ str(round(en.mean(), 4)) +'\n')
            f.write('ssim:  '+ str(round(ssim.mean(), 4)) +'\n')
            f.write('scd: '+ str(round(scd.mean(), 4)) +'\n')
            f.write('cc:  '+ str(round(cc.mean(), 4)) +'\n')
            f.write('qabf:  '+ str(round(qabf.mean(), 4)) +'\n')


