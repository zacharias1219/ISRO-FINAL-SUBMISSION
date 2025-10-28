import os
import cv2
import numpy as np
import glob
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Directory structure
INPUT_DIR = "input_images"
OUTPUT_DIR = "enhanced_outputs"
TEMP_DIR = "temp_processing"

def create_directories():
    """Create necessary directories"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Created directories:")
    print(f"  - {INPUT_DIR}/ (upload your TIFF files here)")
    print(f"  - {OUTPUT_DIR}/ (enhanced outputs will be saved here)")
    print(f"  - {TEMP_DIR}/ (temporary processing files)")

def detect_satellite_type(filename):
    """Detect if it's Sentinel or Planet based on filename"""
    filename_lower = filename.lower()
    if 'sentinel' in filename_lower or 's2' in filename_lower:
        return 'sentinel'
    elif 'planet' in filename_lower or 'ps' in filename_lower:
        return 'planet'
    else:
        # Default to sentinel if unclear
        return 'sentinel'

def percentile_norm(arr: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """Percentile normalization"""
    lo, hi = np.percentile(arr, (p_low, p_high))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max()) if arr.max() > arr.min() else (0.0, 1.0)
    x = (arr - lo) / (hi - lo + 1e-6)
    return np.clip(x, 0.0, 1.0)

def make_falsecolor_NRG(data4: np.ndarray, out_size=None) -> np.ndarray:
    """Create false-color composite (NIR-Red-Green)"""
    nir = percentile_norm(data4[3])
    red = percentile_norm(data4[2])
    grn = percentile_norm(data4[1])
    rgb = np.stack([nir, red, grn], axis=-1).astype(np.float32)
    if out_size is not None:
        h, w = out_size
        res = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(3):
            res[:, :, i] = cv2.resize(rgb[:, :, i], (w, h), interpolation=cv2.INTER_AREA)
        rgb = res
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

def load_tiff_as_rgb(tiff_path):
    """Load TIFF file and convert to RGB for processing"""
    try:
        with rasterio.open(tiff_path) as src:
            # Read all bands
            bands = []
            for i in range(1, src.count + 1):
                band = src.read(i)
                bands.append(band)
            
            # Stack bands
            if len(bands) >= 4:
                # Use first 4 bands (typically NIR, Red, Green, Blue)
                stacked = np.stack(bands[:4], axis=-1)
            else:
                # If less than 4 bands, pad with zeros
                stacked = np.stack(bands, axis=-1)
                while stacked.shape[-1] < 4:
                    stacked = np.concatenate([stacked, np.zeros_like(stacked[:, :, :1])], axis=-1)
            
            # Normalize to 0-1 range first
            stacked = stacked.astype(np.float32)
            if stacked.max() > 1.0:
                stacked = stacked / 10000.0  # Common scaling for satellite data
            stacked = np.clip(stacked, 0, 1)
            
            return stacked
    except Exception as e:
        print(f"Error loading {tiff_path}: {e}")
        return None

def super_resolution_8x(img):
    """Apply 8x super-resolution using Lanczos interpolation with memory optimization"""
    h, w = img.shape[:2]
    
    # Check if 8x upscaling would be too large (>20GB memory)
    estimated_memory_gb = (h * 8 * w * 8 * 3 * 8) / (1024**3)  # 8 bytes per float64
    
    if estimated_memory_gb > 20:
        # Use 4x instead for very large images
        print(f"    Warning: 8x upscaling would require {estimated_memory_gb:.1f}GB memory")
        print(f"    Using 4x upscaling instead for memory efficiency")
        upscaled = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
    else:
        upscaled = cv2.resize(img, (w*8, h*8), interpolation=cv2.INTER_LANCZOS4)
    
    return upscaled

def noise_reduction_enhancement(img):
    """Apply noise reduction enhancement with memory optimization"""
    # Convert to float32 to save memory
    img_f = img.astype(np.float32)
    
    # Check if image is too large for bilateral filter
    h, w = img_f.shape[:2]
    if h * w > 50_000_000:  # 50 megapixels
        print(f"    Using chunked processing for large image ({h}x{w})")
        # Process in chunks to avoid memory issues
        chunk_size = 2000
        denoised = np.zeros_like(img_f)
        
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                y_end = min(y + chunk_size, h)
                x_end = min(x + chunk_size, w)
                
                chunk = img_f[y:y_end, x:x_end]
                chunk_denoised = cv2.bilateralFilter(chunk, 5, 50, 50)  # Smaller parameters for chunks
                denoised[y:y_end, x:x_end] = chunk_denoised
    else:
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(img_f, 9, 75, 75)
    
    # Blend with original to maintain detail (using float32 arithmetic)
    result = img_f * 0.7 + denoised * 0.3
    return np.clip(result, 0, 255).astype(np.uint8)

def smart_contrast_enhancement(img):
    """Smart contrast enhancement using CLAHE (from apply_exact_visual_enhancement.py)"""
    # Convert to LAB for better contrast processing
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Blend with original
    l_final = l * 0.6 + l_enhanced * 0.4
    
    # Reconstruct LAB
    enhanced_lab = cv2.merge([l_final.astype(np.uint8), a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

def intelligent_sharpening(img):
    """Intelligent sharpening (from apply_exact_visual_enhancement.py)"""
    img_f = img.astype(np.float32)
    
    # Create sharpening kernel
    kernel = np.array([[-0.5, -0.5, -0.5], 
                       [-0.5, 5.0, -0.5], 
                       [-0.5, -0.5, -0.5]], dtype=np.float32)
    
    # Apply sharpening
    sharpened = cv2.filter2D(img_f, -1, kernel * 0.3)
    
    # Blend with original
    result = img_f * 0.7 + sharpened * 0.3
    return np.clip(result, 0, 255).astype(np.uint8)

def color_vibrancy_enhancement(img):
    """Color vibrancy enhancement (from apply_exact_visual_enhancement.py)"""
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Enhance saturation
    s = s.astype(np.float32)
    s_enhanced = s * 1.2  # 20% saturation boost
    s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
    
    # Slight brightness adjustment
    v = v.astype(np.float32)
    v_enhanced = v * 1.05  # 5% brightness boost
    v_enhanced = np.clip(v_enhanced, 0, 255).astype(np.uint8)
    
    enhanced_hsv = cv2.merge([h, s_enhanced, v_enhanced])
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

def edge_enhancement(img):
    """Edge enhancement (from apply_exact_visual_enhancement.py)"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize edges
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Create edge mask
    edge_mask = cv2.GaussianBlur(edges, (3, 3), 1.0) / 255.0
    
    # Apply edge enhancement
    img_f = img.astype(np.float32)
    enhanced = img_f + 0.1 * edge_mask[:, :, np.newaxis] * img_f
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def noise_reduction_preserving_details(img):
    """Noise reduction preserving details (from apply_exact_visual_enhancement.py) with memory optimization"""
    h, w = img.shape[:2]
    
    if h * w > 50_000_000:  # 50 megapixels
        print(f"    Using chunked processing for final noise reduction ({h}x{w})")
        # Process in chunks to avoid memory issues
        chunk_size = 2000
        denoised = np.zeros_like(img)
        
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                y_end = min(y + chunk_size, h)
                x_end = min(x + chunk_size, w)
                
                chunk = img[y:y_end, x:x_end]
                chunk_denoised = cv2.bilateralFilter(chunk, 3, 15, 15)  # Smaller parameters for chunks
                denoised[y:y_end, x:x_end] = chunk_denoised
        
        # Blend with original to preserve details
        result = img * 0.8 + denoised * 0.2
    else:
        # Apply gentle bilateral filter
        denoised = cv2.bilateralFilter(img, 5, 20, 20)
        
        # Blend with original to preserve details
        result = img * 0.8 + denoised * 0.2
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_complete_enhancement_pipeline(img):
    """Apply the complete enhancement pipeline"""
    print("    Step 1: Converting to false-color composite (NIR-Red-Green)...")
    false_color = make_falsecolor_NRG(img.transpose(2, 0, 1))  # Convert to (bands, height, width)
    
    print("    Step 2: Applying super-resolution...")
    upscaled = super_resolution_8x(false_color)
    
    # Determine actual scale factor used
    scale_factor = upscaled.shape[0] // false_color.shape[0]
    
    print("    Step 3: Noise reduction enhancement...")
    noise_reduced = noise_reduction_enhancement(upscaled)
    
    print("    Step 4: Smart contrast enhancement...")
    contrast_enhanced = smart_contrast_enhancement(noise_reduced)
    
    print("    Step 5: Color vibrancy enhancement...")
    color_enhanced = color_vibrancy_enhancement(contrast_enhanced)
    
    print("    Step 6: Edge enhancement...")
    edge_enhanced = edge_enhancement(color_enhanced)
    
    print("    Step 7: Intelligent sharpening...")
    sharpened = intelligent_sharpening(edge_enhanced)
    
    print("    Step 8: Final noise reduction...")
    final_enhanced = noise_reduction_preserving_details(sharpened)
    
    print("    Step 9: Final optimization...")
    # Final brightness and contrast adjustment
    final_enhanced = cv2.convertScaleAbs(final_enhanced, alpha=1.05, beta=3)
    final_enhanced = np.clip(final_enhanced, 0, 250)  # Prevent blowouts
    
    return final_enhanced, scale_factor

def process_tiff_file(tiff_path):
    """Process a single TIFF file through the complete pipeline"""
    filename = os.path.basename(tiff_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    print(f"\nProcessing: {filename}")
    print("-" * 50)
    
    # Detect satellite type
    satellite_type = detect_satellite_type(filename)
    print(f"Detected satellite type: {satellite_type.upper()}")
    
    # Load TIFF
    print("Loading TIFF file...")
    img_data = load_tiff_as_rgb(tiff_path)
    if img_data is None:
        print(f"Failed to load {filename}")
        return False
    
    print(f"Loaded image shape: {img_data.shape}")
    
    # Apply complete enhancement pipeline
    print("Applying enhancement pipeline...")
    enhanced_img, scale_factor = apply_complete_enhancement_pipeline(img_data)
    
    # Save enhanced image with original name
    original_name = os.path.splitext(filename)[0]  # Remove .tif extension
    output_filename = f"{original_name}_output_{scale_factor}.0x.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Convert RGB to BGR for OpenCV
    enhanced_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, enhanced_bgr)
    
    print(f"Enhanced image saved: {output_filename}")
    print(f"Output size: {enhanced_img.shape}")
    
    return True

def main():
    """Main function to run the complete pipeline"""
    print("=" * 70)
    print("SATELLITE IMAGE ENHANCEMENT PIPELINE")
    print("=" * 70)
    print()
    print("Processing steps:")
    print("1. Load TIFF files from 'input_images/' folder")
    print("2. Convert to false-color composite (NIR-Red-Green)")
    print("3. Apply 8x super-resolution upscaling")
    print("4. Apply noise reduction enhancement")
    print("5. Apply complete visual enhancement pipeline")
    print("6. Save enhanced images to 'enhanced_outputs/' folder")
    print()
    
    # Create directories
    create_directories()
    print()
    
    # Check for TIFF files
    tiff_files = glob.glob(os.path.join(INPUT_DIR, "*.tif*"))
    
    if not tiff_files:
        print(f"No TIFF files found in '{INPUT_DIR}/' folder!")
        print("Please upload your TIFF files to the 'input_images/' folder and run again.")
        print()
        print("Expected file structure:")
        print("Image_Enhancement_Pipeline/")
        print("├── input_images/          # Upload your TIFF files here")
        print("│   ├── sentinel_data.tif")
        print("│   └── planet_data.tif")
        print("├── enhanced_outputs/      # Enhanced images saved here")
        print("├── temp_processing/       # Temporary files")
        print("└── main.py               # This script")
        return
    
    print(f"Found {len(tiff_files)} TIFF file(s) to process:")
    for tiff_file in tiff_files:
        print(f"  - {os.path.basename(tiff_file)}")
    print()
    
    # Process each TIFF file
    success_count = 0
    for tiff_file in tiff_files:
        if process_tiff_file(tiff_file):
            success_count += 1
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Successfully processed {success_count}/{len(tiff_files)} files")
    print(f"Enhanced images saved to: {OUTPUT_DIR}/")
    print()
    print("Your enhanced satellite images are ready!")
    print()
    print("Output files:")
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
