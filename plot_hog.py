from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys',
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


img = plt.imread('./vehicles/KITTI_extracted/1.png')
ctrans_tosearch = convert_color(img, conv='RGB2YCrCb')

ch1 = ctrans_tosearch[:, :, 0]
ch2 = ctrans_tosearch[:, :, 1]
ch3 = ctrans_tosearch[:, :, 2]


test, img1 = get_hog_features(ch1, 9,8, 2, vis=True)
test, img2 = get_hog_features(ch2, 9,8, 2, vis=True)
test, img3 = get_hog_features(ch3, 9,8, 2, vis=True)

plt.imsave('./output_images/hog_feat1.jpg',img1, cmap='gray')
plt.imsave('./output_images/hog_feat2.jpg',img2, cmap='gray')
plt.imsave('./output_images/hog_feat3.jpg',img3, cmap='gray')