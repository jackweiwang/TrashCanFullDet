##Automatic cloud classification of whole sky image
import cv2
from math import sqrt
import numpy as np

class Spectral_Features():

    def Cal_Mean(self,pImg_Grey):
        Height,Width = pImg_Grey.shape
        nSum=0
        for i in range(Height):
            for j in range(Width):
                nSum = nSum + pImg_Grey[i,j]
        fMean = nSum/(Height*Width)
        return fMean

    def Standard_Deviation(self,pImg_Grey,fMean):
        Height, Width = pImg_Grey.shape
        fSum = 0
        for i in range(Height):
            for j in range(Width):
                fSum = fSum + ((float)(pImg_Grey[i,j])-fMean)*((float)(pImg_Grey[i,j])-fMean)
        f_SD = sqrt(fSum/(Height*Width))
        return f_SD

    def Skewness(self,pImg_Grey,fMean,f_SD):
        Height, Width = pImg_Grey.shape
        fSum = 0
        for i in range(Height):
            for j in range(Width):
                fSub = ((float)(pImg_Grey[i,j])-fMean)/f_SD
                fSum = fSum + fSub*fSub*fSub
        f_SK = fSum / (Height*Width)
        return f_SK

    def Cloud_Cover(self,B,R,G):
        Height, Width = B.shape
        nSum_Cloud=0
        for i in range(Height):
            for j in range(Width):
                nSub = 2*(int)(B[i,j]) - (int)(R[i,j]) - (int)(G[i,j])
                if nSub<50:
                    nSum_Cloud = nSum_Cloud + 1
        fCoverRate = nSum_Cloud/(Height*Width)
        return  fCoverRate

    # def Cal_R_B(self, B, R):
    #     Height, Width = B.shape
    #     for i in range(Height):
    #         for j in range(Width):


    def Cal_SpetrelFeature(self,pImg):
        #pImg = cv2.imread(pImgName)
        B,G,R = cv2.split(pImg)
        fMean_B = self.Cal_Mean(B)
        fMean_G = self.Cal_Mean(G)
        fMean_R = self.Cal_Mean(R)
        fSD_B = self.Standard_Deviation(B,fMean_B)
        fSK_B = self.Skewness(B,fMean_B,fSD_B)
        fDiff_RG = fMean_R - fMean_G
        fDiff_RB = fMean_R-fMean_B
        fDiff_GB = fMean_G - fMean_B
        fCoverRate = self.Cloud_Cover(B,R,G)

        return (fMean_B,fMean_R,fSD_B,fSK_B,fDiff_RG,fDiff_RB,fDiff_GB,fCoverRate)

