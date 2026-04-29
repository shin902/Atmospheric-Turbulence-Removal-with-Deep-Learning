#[[python]画質評価としてのPSNR #Python3 - Qiita](https://qiita.com/Daiki_P/items/94662fd340aa0381b323)


import cv2
import numpy as np
import math

#画像の読み込み
Original  = cv2.imread('Original_Image.jpg') #元画像
Distorted = cv2.imread('Distorted_10.jpg')   #圧縮した画像

#画素値の読み込み
pixel_value_Ori = Original.flatten().astype(float)
pixel_value_Dis = Distorted.flatten().astype(float)

#画素情報の取得
imageHeight, imageWidth, BPP = Original.shape

#画素数
N = imageHeight * imageWidth

#1画素あたりRGB3つの情報がある.
addr = N * BPP

#RGB画素値の差の2乗の総和
sumR=0
sumG=0
sumB=0

#差の2乗の総和を計算
for i in range(addr):
    if(i%3==0):
        sumB += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )
    elif(i%3==1):
        sumG += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )
    else:
        sumR += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )

#PSNRを求める
MSE =(sumR + sumG + sumB) / (3 * N )
PSNR = 10 * math.log(255*255/MSE,10)
print('PSNR',PSNR)
