import cv2
import numpy as np

def get_mean_and_std(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean, std = cv2.meanStdDev(lab, mask=mask)
    return mean.flatten(), std.flatten()

def apply_color_transfer(src_bgr, src_mean_L, src_std_L, src_mean_ab, src_std_ab, tgt_mean, tgt_std):
    eps = 1e-5
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(src_lab)

    l_t = (l - src_mean_L[0]) * (tgt_std[0] / (src_std_L[0] + eps)) + tgt_mean[0]
    a_t = (a - src_mean_ab[1]) * (tgt_std[1] / (src_std_ab[1] + eps)) + tgt_mean[1]
    b_t = (b - src_mean_ab[2]) * (tgt_std[2] / (src_std_ab[2] + eps)) + tgt_mean[2]

    out_lab = np.clip(cv2.merge([l_t, a_t, b_t]), 0, 255).astype(np.uint8)

    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
