import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Multiple View.avi')

w_aug = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_aug = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, fps, (w_aug,  h_aug))

ref_frame = cv2.imread('ReferenceFrame.png')
h_frame, w_frame = ref_frame.shape[0], ref_frame.shape[1]
object_mask = cv2.imread('ObjectMask.PNG',cv2.IMREAD_GRAYSCALE)

ref_frame_masked = ref_frame.copy()
ref_frame_masked[object_mask==0] = 0
# ref_frame_intensity = np.array(cv2.mean(ref_frame, mask=object_mask)[:3])

aug_layer = cv2.imread('AugmentedLayer.PNG')
aug_layer = aug_layer[:h_frame, :w_frame]
aug_mask = cv2.imread('AugmentedLayerMask.PNG',cv2.IMREAD_GRAYSCALE)
aug_mask = aug_mask[:h_frame, :w_frame]

H = np.eye(3)

# SIFT detector
sift = cv2.SIFT_create()
kp_rf = sift.detect(ref_frame_masked)
kp_rf, des_rf = sift.compute(ref_frame_masked, kp_rf)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    kp_frame = sift.detect(frame)
    kp_frame, des_frame = sift.compute(frame, kp_frame)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_rf,des_frame,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    src_pts = np.float32([kp_rf[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, match_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    warped = cv2.warpPerspective(aug_layer, M, (w_frame, h_frame), flags=cv2.INTER_LINEAR)
    warp_mask = cv2.warpPerspective(aug_mask, M, (w_frame, h_frame), flags=cv2.INTER_LINEAR) < 250

    '''
    # Illumination correction
    warp_object_mask = cv2.warpPerspective(object_mask, M, (w_frame, h_frame), flags=cv2.INTER_LINEAR) > 250
    frame_intensity = np.array(cv2.mean(frame, mask=warp_object_mask.astype(np.uint8))[:3])

    light_correction = frame_intensity / ref_frame_intensity
    warped = np.clip(warped * light_correction, 0, 255).astype(np.uint8)
    '''
    warped[warp_mask] = frame[warp_mask]
    
    out.write(warped)

    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    # plt.show()
