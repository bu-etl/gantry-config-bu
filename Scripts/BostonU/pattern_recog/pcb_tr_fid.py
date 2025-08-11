import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Logs/tr_search.png", cv2.IMREAD_GRAYSCALE)
tr_template = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/PCB_templates/pcb_tr_template.png", cv2.IMREAD_GRAYSCALE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

tr_offset_x = 216
tr_offset_y = 591

tr_result = cv2.matchTemplate(img, tr_template, cv2.TM_CCOEFF_NORMED)
tr_min_val, tr_max_val, tr_min_loc, tr_max_loc = cv2.minMaxLoc(tr_result)

tr_corner_x = int(tr_max_loc[0] + tr_offset_x)
tr_corner_y = int(tr_max_loc[1] + tr_offset_y)

window_w = 2448
window_h = 2048

real_x = (tr_corner_x - window_w//2)*(.83/1224)
real_y = (tr_corner_y - window_h//2)*(.7/1024)

file_name = "C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/config/tr_fid_loc.txt"
content = f"tr_x: {real_x}\ntr_y: {real_y}"

# Open the file in write mode ('w'). If the file doesn't exist, it's created.
# If it exists, its content is overwritten.
with open(file_name, 'w') as file:
    file.write(content)