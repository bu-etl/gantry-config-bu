import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Logs/tl_search.png", cv2.IMREAD_GRAYSCALE)
tl_template = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/PCB_templates/pcb_tl_template.png", cv2.IMREAD_GRAYSCALE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

tl_offset_x = 130
tl_offset_y = 130

tl_result = cv2.matchTemplate(img, tl_template, cv2.TM_CCOEFF_NORMED)
tl_min_val, tl_max_val, tl_min_loc, tl_max_loc = cv2.minMaxLoc(tl_result)

tl_corner_x = int(tl_max_loc[0] + tl_offset_x)
tl_corner_y = int(tl_max_loc[1] + tl_offset_y)

window_w = 2448
window_h = 2048

real_x = (tl_corner_x - window_w//2)*(.83/1224)
real_y = (tl_corner_y - window_h//2)*(.7/1024)

file_name = "C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/config/tl_fid_loc.txt"
content = f"tl_x: {real_x}\ntl_y: {real_y}"

# Open the file in write mode ('w'). If the file doesn't exist, it's created.
# If it exists, its content is overwritten.
with open(file_name, 'w') as file:
    file.write(content)