import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def align_images(ref, target):
    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    h, w = ref_gray.shape
    
    # Use a central crop of the reference to find the shift
    # This avoids border artifacts
    crop_h, crop_w = h // 2, w // 2
    start_y, start_x = h // 4, w // 4
    
    ref_crop = ref_gray[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Match this crop in the target image
    res = cv2.matchTemplate(target_gray, ref_crop, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    # max_loc is the top-left of the match in target
    # The expected top-left if perfectly aligned is (start_x, start_y)
    
    dx = max_loc[0] - start_x
    dy = max_loc[1] - start_y
    
    print(f"Detected shift: dx={dx}, dy={dy}")
    
    # Shift target image to align with reference
    # We use an affine warp
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned_target = cv2.warpAffine(target, M, (w, h))
    
    return aligned_target, dx, dy

from skimage.exposure import match_histograms

def detect_defects_reference(ref_path, target_path, output_path):
    print(f"Loading Reference: {ref_path}")
    print(f"Loading Target: {target_path}")
    
    ref_img = cv2.imread(ref_path)
    target_img = cv2.imread(target_path)
    
    if ref_img is None or target_img is None:
        print("Error loading images.")
        return

    # Match histograms of target to reference (normalize lighting)
    print("Normalizing target image...")
    target_img = match_histograms(target_img, ref_img, channel_axis=-1).astype(np.uint8)

    # Align images
    aligned_target, dx, dy = align_images(ref_img, target_img)
    
    # Convert to grayscale for SSIM
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(aligned_target, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    print("Computing SSIM...")
    # win_size=7 is default, maybe increase for more structural view?
    score, diff_map = ssim(ref_gray, target_gray, full=True)
    print(f"SSIM Score: {score}")
    
    # Invert diff map
    defect_map = 1 - diff_map
    
    # Normalize to 0-255
    defect_img = (defect_map * 255).astype(np.uint8)
    
    # Threshold
    # Lowered threshold to 120 to catch subtler defects (recall)
    _, thresh = cv2.threshold(defect_img, 120, 255, cv2.THRESH_BINARY)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8) # Larger kernel
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_out = aligned_target.copy()
    
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20: # Lowered area threshold to 20
            x, y, w, h = cv2.boundingRect(cnt)
            
            img_h, img_w = ref_gray.shape
            valid_x_min = max(0, -dx)
            valid_x_max = min(img_w, img_w - dx)
            valid_y_min = max(0, -dy)
            valid_y_max = min(img_h, img_h - dy)
            
            if (x >= valid_x_min and x + w <= valid_x_max and
                y >= valid_y_min and y + h <= valid_y_max):
                
                cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
                count += 1

    print(f"Found {count} defects.")
    cv2.imwrite(output_path, img_out)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    ref_path = os.path.join("SampleImages", "Test1.png")
    target_path = os.path.join("SampleImages", "Test3.png")
    output_path = "defects_ref_1_3.png"
    detect_defects_reference(ref_path, target_path, output_path)
