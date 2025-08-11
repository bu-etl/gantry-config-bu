import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Logs/bl_search.png", cv2.IMREAD_GRAYSCALE)
bl_template = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/PCB_templates/pcb_bl_template.png", cv2.IMREAD_GRAYSCALE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

bl_offset_x = 30
bl_offset_y = 1096

bl_result = cv2.matchTemplate(img, bl_template, cv2.TM_CCOEFF_NORMED)
bl_min_val, bl_max_val, bl_min_loc, bl_max_loc = cv2.minMaxLoc(bl_result)

bl_corner_x = int(bl_max_loc[0] + bl_offset_x)
bl_corner_y = int(bl_max_loc[1] + bl_offset_y)

window_w = 2448
window_h = 2048

real_x = (bl_corner_x - window_w//2)*(.83/1224)
real_y = (bl_corner_y - window_h//2)*(.7/1024)

file_name = "C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/config/bl_fid_loc.txt"
content = f"bl_x: {real_x}\nbl_y: {real_y}"

# Open the file in write mode ('w'). If the file doesn't exist, it's created.
# If it exists, its content is overwritten.
with open(file_name, 'w') as file:
    file.write(content)