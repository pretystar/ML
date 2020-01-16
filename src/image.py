from skimage import io,data,color,exposure,transform
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import rescale, resize, downscale_local_mean
import os, fnmatch 
def pic_preprocess(self,f): 
    processed_image = []
    # Read the picture as skimage
    sk_image = io.imread(f)
    # Convert the skimage from RGB to gray scale
    img_gray = rgb2gray(sk_image)
    # Resize the picture to a fixed size of pixels
    processed_image = resize(img_gray, (25, 25))
    ###processed_image = resize(sk_image, (200, 200))
    # Return preprocessed picture
    return processed_image

def pic_normalization(self,pic):
    pic = pic-pic.min()
    pic = pic/pic.max()
    # Recover range to [0,255] and return
    return pic*255