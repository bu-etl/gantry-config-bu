import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Logs/br_search.png", cv2.IMREAD_GRAYSCALE)
br_template = cv2.imread("C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/PCB_templates/pcb_br_template.png", cv2.IMREAD_GRAYSCALE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

br_offset_x = 1060
br_offset_y = 693

br_result = cv2.matchTemplate(img, br_template, cv2.TM_CCOEFF_NORMED)
br_min_val, br_max_val, br_min_loc, br_max_loc = cv2.minMaxLoc(br_result)

br_corner_x = int(br_max_loc[0] + br_offset_x)
br_corner_y = int(br_max_loc[1] + br_offset_y)

window_w = 2448
window_h = 2048

real_x = (br_corner_x - window_w//2)*(.83/1224)
real_y = (br_corner_y - window_h//2)*(.7/1024)

file_name = "C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/pattern_recog/config/br_fid_loc.txt"
content = f"br_x: {real_x}\nbr_y: {real_y}"

# Open the file in write mode ('w'). If the file doesn't exist, it's created.
# If it exists, its content is overwritten.
with open(file_name, 'w') as file:
    file.write(content)