import cv2
import numpy as np
import os
from scipy.signal import find_peaks

def detect_vertical_period(img_gray):
    # Calculate horizontal projection (mean of rows)
    projection = np.mean(img_gray, axis=1)
    
    # Remove mean to center the signal
    projection = projection - np.mean(projection)
    
    # Compute autocorrelation
    correlation = np.correlate(projection, projection, mode='full')
    correlation = correlation[correlation.size // 2:]
    
    # Find peaks
    peaks, properties = find_peaks(correlation, distance=20, prominence=np.max(correlation)*0.1)
    
    if len(peaks) > 0:
        first_peak_idx = 0
        if peaks[0] == 0:
            if len(peaks) > 1:
                first_peak_idx = 1
            else:
                return None
        return peaks[first_peak_idx]
    return None

from skimage.metrics import structural_similarity as ssim

def detect_defects(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # 1. Detect Pattern Height
    detected_h = detect_vertical_period(img_gray)
    if not detected_h:
        print("Could not detect period.")
        return
    
    template_h = int(detected_h)
    template_w = w
    
    # 2. Find Blocks
    center_y = h // 2
    template = img_gray[center_y:center_y+template_h, 0:w]
    
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)
    
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([pt[0], pt[1], template_w, template_h])
    
    rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.2)
    rects = sorted(rects, key=lambda r: r[1])
    
    print(f"Found {len(rects)} blocks.")
    
    if len(rects) == 0:
        return

    # 3. Extract Blocks and Compute Median Reference
    blocks = []
    valid_rects = []
    
    for (x, y, w_rect, h_rect) in rects:
        if y + h_rect <= h:
            roi = img_gray[y:y+h_rect, x:x+w_rect]
            blocks.append(roi)
            valid_rects.append((x, y, w_rect, h_rect))
            
    if not blocks:
        return

    blocks_stack = np.array(blocks)
    median_block = np.median(blocks_stack, axis=0).astype(np.uint8)
    
    # 4. Compare and Detect Defects using SSIM
    img_out = img.copy()
    
    for i, block in enumerate(blocks):
        x, y, w_rect, h_rect = valid_rects[i]
        
        # Refine Alignment
        pad = 5
        median_padded = cv2.copyMakeBorder(median_block, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        res_align = cv2.matchTemplate(median_padded, block, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res_align)
        dx = max_loc[0] - pad
        dy = max_loc[1] - pad
        
        if abs(dx) > 5 or abs(dy) > 5:
            dx, dy = 0, 0
            
        new_x, new_y = x - dx, y - dy
        if new_y < 0 or new_y + h_rect > h or new_x < 0 or new_x + w_rect > w:
             aligned_block = block
        else:
             aligned_block = img_gray[new_y:new_y+h_rect, new_x:new_x+w_rect]
        
        # Compute SSIM
        # SSIM returns a score and a diff map. We want the diff map.
        # full=True returns the image.
        score, diff_map = ssim(median_block, aligned_block, full=True)
        
        # diff_map is in range [-1, 1] or [0, 1] depending on input?
        # scikit-image ssim usually returns diff in same range as input if data_range is specified,
        # but here inputs are uint8, so it might infer.
        # Actually diff_map from ssim is usually float.
        # We want to find areas where ssim is LOW.
        # The diff_map represents the local SSIM value.
        # So we want to invert it: 1 - diff_map
        
        diff_map = (diff_map + 1) / 2 # Normalize to [0, 1] if it was [-1, 1]
        # Actually ssim output is [0, 1] for similarity?
        # Let's assume it's [0, 1] where 1 is identical.
        
        defect_map = 1 - diff_map
        
        # Threshold
        # SSIM is very sensitive. A value < 0.8 might be a defect.
        # Let's convert to uint8 for opencv
        defect_img = (defect_map * 255).astype(np.uint8)
        
        # Threshold: look for significant differences
        # High defect_img value means low similarity.
        # Threshold of 100 (approx 0.4 diff) might be good for "broken links".
        _, thresh = cv2.threshold(defect_img, 100, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        draw_x = new_x if 'new_x' in locals() else x
        draw_y = new_y if 'new_y' in locals() else y
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10: 
                x_d, y_d, w_d, h_d = cv2.boundingRect(cnt)
                cv2.rectangle(img_out, (draw_x + x_d, draw_y + y_d), (draw_x + x_d + w_d, draw_y + y_d + h_d), (0, 0, 255), 2)

    cv2.imwrite(output_path, img_out)
    print(f"Saved defect detection result to {output_path}")

if __name__ == "__main__":
    image_path = os.path.join("SampleImages", "Test2.png")
    output_path = "defects.png"
    detect_defects(image_path, output_path)
