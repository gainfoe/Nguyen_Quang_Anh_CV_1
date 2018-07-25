import cv2
import numpy as np
import Anh


class Object:
    def __init__(self, image, gray_image, kp_image, kp_des, good, new_good):
        self.image = image
        self.gray_image = gray_image
        self.kp_image = kp_image
        self.kp_frame = kp_frame
        self.good = good
        self.new_good = new_good


# tao SIFT
sift = cv2.xfeatures2d.SIFT_create()


cap = cv2.VideoCapture(0)

while True:
    nothing = cv2.imread("3.png")
    gray_nothing = cv2.cvtColor(nothing, cv2.COLOR_RGB2GRAY)
    kp__nothing, des_nothing = sift.detectAndCompute(gray_nothing, None)
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
    cv2.drawKeypoints(frame, kp_frame, frame)
    if len(kp_frame) > 10:
        object = Object(nothing, gray_nothing, kp__nothing, des_nothing, [], [])
        for anh in Anh.anh:
            image = cv2.imread(anh)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kp_image, des_image = sift.detectAndCompute(gray_image, None)
            # tao thuat toan match
            BF = cv2.BFMatcher_create()
            matches = BF.knnMatch(des_image, des_frame, k=2)
            good = []
            new_good = []
            for m, n in matches:
                if m.distance < 0.4 * n.distance:
                    good.append([m])
                    new_good.append(m)
            if len(object.good) < len(good):
                object = Object(image, gray_image, kp_image, des_image, good, new_good)
        if len(object.new_good) > 10:
            src_ds = np.float32([object.kp_image[m.queryIdx].pt for m in object.new_good]).reshape(-1, 1, 2)
            dst_ds = np.float32([kp_frame[m.trainIdx].pt for m in object.new_good]).reshape(-1, 1, 2)
            # find homography
            M, H = cv2.findHomography(src_ds, dst_ds, cv2.RANSAC, 5.0)
            # get h, w of pattern
            h, w = object.gray_image.shape
            pointcorner = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
            if M is None:
                print("Khong tim thay Homography")
            else:
                pts = cv2.perspectiveTransform(pointcorner, M)
                imgReg = cv2.polylines(frame, np.int32([pts]), True, [255, 0, 0], 3)

        width = object.image.shape[1] + frame.shape[1]
        height = object.image.shape[0] + frame.shape[0]
        outImg = np.zeros((width, height, 3), dtype=np.uint8)
        conan = cv2.drawMatchesKnn(object.image, object.kp_image, frame, kp_frame, object.good, outImg)

        cv2.imshow("outImage", conan)

        key = cv2.waitKey(30)

