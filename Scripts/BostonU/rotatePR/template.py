import cv2
import numpy as np
import argparse

def rotate_and_pad(img, angle, pad_color=0):
    """
    1. Pad img to a square box whose side = diagonal of original (so no cropping on rotation).
    2. Rotate around that square's center.
    Returns (rotated, pad_x, pad_y) where pad_x,y are the offsets of the original image inside the pad.
    """
    h, w = img.shape
    diag = int(np.ceil(np.sqrt(h*h + w*w)))
    # centre-pad
    pad_x = (diag - w) // 2
    pad_y = (diag - h) // 2
    big = np.full((diag, diag), pad_color, dtype=img.dtype)
    big[pad_y:pad_y+h, pad_x:pad_x+w] = img
    # rotate about centre
    M = cv2.getRotationMatrix2D((diag/2, diag/2), angle, 1.0)
    rot = cv2.warpAffine(big, M, (diag, diag),
                         flags=cv2.INTER_LINEAR,
                         borderValue=pad_color)
    return rot, pad_x, pad_y

def find_best_on_rotated(target, template, angles=(-180,180,10), pad_color=0):
    """
    angles = (start, end, step)
    """
    best = {
        'score': -np.inf,
        'angle': None,
        'loc': None,
        'rotated': None
    }
    th, tw = template.shape
    orig_h, orig_w = target.shape

    for angle in range(angles[0], angles[1]+1, angles[2]):
        rotated, pad_x, pad_y = rotate_and_pad(target, angle, pad_color=pad_color)
        # match template against rotated target
        res = cv2.matchTemplate(rotated, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # compute loc of the match
        cx = max_loc[0] + tw//2
        cy = max_loc[1] + th//2
        # skip if that centre is in the padded region
        if cx < pad_x or cx >= pad_x + orig_w or cy < pad_y or cy >= pad_y + orig_h:
            continue

        if max_val > best['score']:
            best.update({
                'score': max_val,
                'angle': angle,
                'loc': max_loc,
                'rotated': rotated.copy()
            })

    return best

parser = argparse.ArgumentParser()
parser.add_argument("--corner", choices=['tl','tr','bl','br'])
parser.add_argument("--scaledown", action="store", type=float)
args = parser.parse_args()

if __name__ == "__main__":
    # === usage ===
    if args.corner == 'tl':
        template = cv2.imread('C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/rotatePR/pcb_templates/pcb_tl_template.png', cv2.IMREAD_GRAYSCALE)
        offsetx = 125
        offsety = 130
        cor = 1
    elif args.corner == 'tr':
        template = cv2.imread('C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/rotatePR/pcb_templates/pcb_tr_template.png', cv2.IMREAD_GRAYSCALE)
        offsetx = 230
        offsety = 590
        cor=2
    elif args.corner == 'br':
        template = cv2.imread('C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/rotatePR/pcb_templates/pcb_br_template.png', cv2.IMREAD_GRAYSCALE)
        offsetx = 1070
        offsety = 690
        cor=3
    elif args.corner == 'bl':
        template = cv2.imread('C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/rotatePR/pcb_templates/pcb_bl_template.png', cv2.IMREAD_GRAYSCALE)
        offsetx = 20
        offsety = 1110
        cor=4
    
    target = cv2.imread('C:/Users/User/Desktop/gantry-config-bu/Logs/search.png', cv2.IMREAD_GRAYSCALE)

    scale_down = float(args.scaledown)
    target_down = cv2.resize(target, dsize=None, fx=scale_down, fy=scale_down)
    template_down = cv2.resize(template, dsize=None, fx=scale_down, fy=scale_down)

    best = find_best_on_rotated(target_down, template_down, angles=(-180,180,5), pad_color=0)
    x, y = best['loc']

    fid_x = x+offsetx*scale_down
    fid_y = y+offsety*scale_down
    
    pich, picw = best['rotated'].shape

    gantryx = (fid_x-picw//2)*(1/scale_down)*(.83/1224)
    gantryy = (fid_y-pich//2)*(1/scale_down)*(.7/1024)

    configfile = "C:/Users/User/Desktop/gantry-config-bu/Scripts/BostonU/rotatePR/config/fid_loc.txt"
    content = f"fidx: {gantryx}\nfidy: {gantryy}\ncor: {cor}"

    with open(configfile, "w") as file:
        file.write(content)


    # out = cv2.cvtColor(best['rotated'], cv2.COLOR_GRAY2BGR)
    # cv2.circle(out, (int(fid_x), int(fid_y)), 2, (0,0,255), 2)
    # cv2.namedWindow("Matched", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Matched", 1000, 800)
    # cv2.imshow("Matched",out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
