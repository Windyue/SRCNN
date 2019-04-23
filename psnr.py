import cv2
import math
import numpy
from skimage.measure import compare_ssim


def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


if __name__ == "__main__":
    im1 = cv2.imread("./input2.jpg", cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread("./11.png", cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread("0000.png", cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im4 = cv2.imread("./pre2.png", cv2.IMREAD_COLOR)
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im5 = cv2.imread("./shft.png", cv2.IMREAD_COLOR)
    im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im6 = cv2.imread("./srgan.png", cv2.IMREAD_COLOR)
    im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]


    # print("adam:")
    # print(psnr(im2, im3))
    # print("bicubic:")
    # print(psnr(im2, im1))
    # print("SRCNN:")
    print("MSE_SRGAN  ","PSNR:              ",psnr(im1,im2),"SSIM:",compare_ssim(im1,im2))
    print("感知损失_SRGAN","PSNR:           ",psnr(im1, im5),"SSIM:",compare_ssim(im1,im5))
    print("W_GAN_SRGAN","PSNR:              ",psnr(im1, im4),"SSIM:",compare_ssim(im1,im4))
    print("DnCNN + W_GAN_SRGAN + EPF","PSNR:",psnr(im1, im6),"SSIM:",compare_ssim(im1,im6))
    print("final","PSNR:                    ",psnr(im1,im3),"SSIM:",compare_ssim(im1,im3))



