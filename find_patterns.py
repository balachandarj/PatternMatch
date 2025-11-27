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
    # mode='full' returns cross-correlation of size 2*N-1
    correlation = np.correlate(projection, projection, mode='full')
    
    # Take the second half (positive lags)
    correlation = correlation[correlation.size // 2:]
    
    # Find peaks
    # distance=20 prevents finding peaks too close to each other
    # prominence=0.1 ensures we find significant peaks
    peaks, properties = find_peaks(correlation, distance=20, prominence=np.max(correlation)*0.1)
    
    if len(peaks) > 0:
        # The first peak (excluding lag 0 if it was included, but find_peaks usually handles this if we start > 0)
        # Actually lag 0 is usually the max, so we might need to skip it if it's in the list.
        # But find_peaks might pick it up.
        # Let's check the first peak that is not at index 0.
        
        # If the first peak is at 0, skip it.
        first_peak_idx = 0
        if peaks[0] == 0:
            if len(peaks) > 1:
                first_peak_idx = 1
            else:
                return None
        
        period = peaks[first_peak_idx]
        print(f"Detected vertical period: {period} pixels")
        return period
    
    return None

def find_repetitive_blocks(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # Auto-detect height
    detected_h = detect_vertical_period(img_gray)
    
    if detected_h:
        template_h = int(detected_h)
    else:
        print("Could not detect period, defaulting to 100")
        template_h = 100
        
    # User requested to use the entire image width.
    center_y = h // 2
    template_w = w # Use full width
    
    # Extract template
    # Since we use full width, x start is 0
    template = img_gray[center_y:center_y+template_h, 0:w]
    
    # Perform template matching
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(f"Max correlation: {max_val}")
    
    # Define a threshold for matches
    threshold = 0.4 # Lowered threshold
    loc = np.where(res >= threshold)
    
    # Non-Maximum Suppression
    # Convert points to a list of rectangles [x, y, w, h]
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([pt[0], pt[1], template_w, template_h])
    
    rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.2)
    
    img_rgb = img.copy()
    count = 0
    for (x, y, w, h) in rects:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
        count += 1

    print(f"Found {count} matches after NMS.")
    
    # Save result
    cv2.imwrite(output_path, img_rgb)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    image_path = os.path.join("SampleImages", "Test2.png")
    output_path = "result.png"
    find_repetitive_blocks(image_path, output_path)
