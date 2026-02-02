import cv2
import numpy as np

def get_mean_and_std(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean, std = cv2.meanStdDev(lab, mask=mask)
    return mean.flatten(), std.flatten()

def apply_color_transfer(src_bgr, src_ref_mean, src_ref_std, src_aug_mean, src_aug_std, tgt_mean, tgt_std,
                         treshold = 10, slope_col = 3, k = 3):
    eps = 1e-5
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(src_lab)

    # Adaptive alpha based on contrast difference
    std_ratio = tgt_std[0] / (src_ref_std[0] + eps)
    alpha = 1.0 - np.exp(-k * abs(std_ratio - 1.0))      
    # Adaptive Luminance Adjustment
    l_t_gain = l * (tgt_mean[0] / (src_ref_mean[0] + eps))
    l_t_reinhard = (l - src_ref_mean[0]) * std_ratio + tgt_mean[0]
    l_t = l_t_gain * (1 - alpha) + l_t_reinhard * alpha

    # Calculate color distance in a*b* space
    delta_a = tgt_mean[1] - src_ref_mean[1] 
    delta_b = tgt_mean[2] - src_ref_mean[2]
    color_distance = np.sqrt(delta_a**2 + delta_b**2)
    # Compute weight for color adjustment using sigmoid function
    w_color = 1.0 / (1.0 + np.exp(-slope_col * (color_distance - treshold)))

    # Adjust a* and b* channels based on augmented source and target statistics (Reinhard color transfer)
    a_t_reinhard = (a - src_aug_mean[1]) * (tgt_std[1] / (src_aug_std[1] + eps)) + tgt_mean[1]
    b_t_reinhard = (b - src_aug_mean[2]) * (tgt_std[2] / (src_aug_std[2] + eps)) + tgt_mean[2]
    
    # Adaptive blending
    a_t = a * (1 - w_color) + a_t_reinhard * w_color
    b_t = b * (1 - w_color) + b_t_reinhard * w_color

    out_lab = np.clip(cv2.merge([l_t.astype(np.float32), a_t.astype(np.float32), b_t.astype(np.float32)]), 0, 255).astype(np.uint8)

    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
