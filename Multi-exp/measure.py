import cv2
import skimage.measure 
from sklearn import metrics
import numpy as np
import math
from PIL import Image
import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def SCD(ir, vi, fusion):
    r1 = np.corrcoef(fusion-vi, ir)
    r2 = np.corrcoef(fusion-ir, vi)
    return r1[0,1]+r2[0,1]

def SD(fusion):
    return np.std(fusion)

def CC(a,b):
    ret = np.corrcoef(a,b)
    return ret[0,1]


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type=str, default='auto')
  parser.add_argument('--epochs', type=int, default=8)

  args, _ = parser.parse_known_args()
  return args


if __name__ == '__main__':
	# 这样读取后已经是灰度图
    args = get_args()
    # basepath = args.dir

    # for epoch in range(args.epochs):
    for epoch in range(args.epochs):
        if epoch != 6:
            continue
        en = []
        mi = []
        scd = []
        cc = []
        sd = []
        for i in range(19):
            fusionpath = 'eval/'+ '/F'+str(epoch+1) +'_'+str(i) + '.bmp'

            # if i<10:
            #     fusionpath = '../Multi-exp/result/epoch9' +  '/F9_0'+str(i) + '.bmp'
            # else: 
            #     fusionpath = '../Multi-exp/result/epoch9'+ '/F9_'+str(i) + '.bmp'
            sourceVIpath = '../Multi-exp/Test_vi/' + str(i+1) + '.bmp'
            sourceIRpath = '../Multi-exp/Test_ir/' + str(i+1) + '.bmp'
            fusionImage = cv2.imread(fusionpath, 0)
            sourceVI_Image = cv2.imread(sourceVIpath, 0)
            sourceIR_Image = cv2.imread(sourceIRpath, 0)


            en.append(round(skimage.measure.shannon_entropy(fusionImage, base=2),4))
            # mi.append(metrics.mutual_info_score(sourceVI_Image.reshape(-1), fusionImage.reshape(-1)))
            mi.append(round(metrics.mutual_info_score(sourceIR_Image.reshape(-1), fusionImage.reshape(-1)),4))
            scd.append(round(SCD(sourceIR_Image, sourceVI_Image, fusionImage),4))
            # cc.append(metrics.matthews_corrcoef(sourceVI_Image.reshape(-1), fusionImage.reshape(-1)))
            cc.append(round(CC(sourceVI_Image, fusionImage),4))
            # cc.append(CC(sourceIR_Image, fusionImage))
            sd.append(round(SD(fusionImage)/255.0,4))

        print(en)
        print(mi)
        print(scd)
        print(cc)
        print(sd)

        cc = [n for n in cc if not(math.isnan(n))]
        en = np.array(en)
        mi = np.array(mi)
        scd = np.array(scd) 
        cc = np.array(cc) 
        sd = np.array(sd) 

        print("model: PMGI",)
        print('en:      ', str(7.0527))
        print('mi:      ', str(2.1650))
        print('scd:     ', str(1.2855))
        print('cc:      ', str(0.9003))
        print('sd:      ', str(0.1882))

        print('\nmodel: '  + str(epoch+1) )
        print('en:      ',en.mean())
        print('mi:      ',mi.mean())
        print('scd:     ',scd.mean())
        print('cc:      ',cc.mean())
        print('sd:      ',sd.mean())

        with open(args.dir + '.txt','a') as f:
            f.write('\n' + str(epoch+1) +':\n' )
            f.write('en:  '+ str(en.mean()) +'\n')
            f.write('mi:  '+ str(mi.mean()) +'\n')
            f.write('scd: '+ str(scd.mean()) +'\n')
            f.write('cc:  '+ str(cc.mean()) +'\n')
            f.write('sd:  '+ str(sd.mean()) +'\n')


