import cv2
import numpy as np
import stereoconfig

def getRectifyTransform(height, width, config):
    #读取矩阵参数
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    #计算校正变换
    if type(height) != "int" or type(width) != "int":
        height = int(height)
        width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q

# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2

#视差计算
def sgbm(imgL, imgR):
    #SGBM参数设置
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity = 1,
                                   numDisparities = 64,
                                   blockSize = blockSize,
                                   P1 = 8 * img_channels * blockSize * blockSize,
                                   P2 = 32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff = -1,
                                   preFilterCap = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 100,
                                   mode = cv2.STEREO_SGBM_MODE_HH)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.)#除以16得到真实视差图
    return disp
#计算三维坐标，并删除错误点
def threeD(disp, Q):
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)

    points_3d = points_3d.reshape(points_3d.shape[0] * points_3d.shape[1], 3)

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    #选择并删除错误的点
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    points_3d = np.delete(points_3d, remove_idx, 0)

    #计算目标点（这里我选择的是目标区域的中位数，可根据实际情况选取）
    if points_3d.any():
        x = np.median(points_3d[:, 0])
        y = np.median(points_3d[:, 1])
        z = np.median(points_3d[:, 2])
        targetPoint = [x, y, z]
    else:
        targetPoint = [0, 0, -1]#无法识别目标区域

    return targetPoint


imgL = cv2.imread("data/L.tif")
imgR = cv2.imread("data/R.tif")

height, width = imgL.shape[0:2]
# 读取相机内参和外参
config = stereoconfig.stereoCameral()

map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
iml_rectified, imr_rectified = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)

# wfeng: Bm只接受CV8UC_1
cv2.imwrite('./data/iml_rectified.png', iml_rectified)
cv2.imwrite('./data/imr_rectified.png', imr_rectified)
iml_rectified = cv2.cvtColor(iml_rectified, cv2.COLOR_BGR2GRAY)
imr_rectified = cv2.cvtColor(imr_rectified, cv2.COLOR_BGR2GRAY)

# disp = sgbm(iml_rectified, imr_rectified)

# wfeng: 创建GUI进行观察
# Creating an object of StereoBM algorithm
# stereo = cv2.StereoBM_create()
stereo = cv2.StereoSGBM_create()

def nothing(x):
    pass

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)

cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
# cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
# cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
# cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)



while True:
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
    # preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    # preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    # textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    # stereo.setPreFilterType(preFilterType)
    # stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    # stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    stereo.setP1(8*blockSize*blockSize)
    stereo.setP2(32 * blockSize * blockSize)

    disparity = stereo.compute(iml_rectified, imr_rectified)  # wfeng
    # Converting to float32
    disparity = disparity.astype(np.float32)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.

    # Scaling down the disparity values and normalizing them
    disparity = (disparity / 16.0 - minDisparity) / numDisparities
    cv2.imshow("disp", disparity)
    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break
# print(disp)
# cv2.imshow("disp", disp)
# cv2.imwrite('./data/disp.png', disp)
# target_point = threeD(disp, Q)#计算目标点的3D坐标（左相机坐标系下）
# print(target_point)

# cv2.waitKey(0)