# Sir's Final Satellite Image Enhancement Pipeline

## Quick Start

**Note:**  
Make sure your input `.tif` image filenames contain either `sentinel` or `planet` in their names  
(e.g., `sentinel_2024_image.tif`, `my_planet_image.tif`).  
This helps the pipeline automatically detect the correct satellite type for optimal enhancement steps.

1. **Upload your TIFF files** to the `input_images/` folder
2. **Run the script**: `python main.py`
3. **Get your enhanced images** from the `enhanced_outputs/` folder

## What This Pipeline Does

### Input Processing
- Loads TIFF files (supports 4-band satellite data: NIR, Red, Green, Blue)
- Automatically detects Sentinel vs Planet data based on filename
- Converts to false-color composite (NIR-Red-Green) for optimal visualization

### Enhancement Pipeline (Same as Your Sir's Liked Images)
1. **8x Super-Resolution**: High-quality Lanczos interpolation upscaling
2. **Noise Reduction Enhancement**: The key step your sir liked
3. **Smart Contrast Enhancement**: CLAHE for optimal contrast
4. **Color Vibrancy Enhancement**: 20% saturation boost
5. **Edge Enhancement**: Sobel-based edge detection and enhancement
6. **Intelligent Sharpening**: Kernel-based sharpening without artifacts
7. **Final Noise Reduction**: Detail-preserving noise reduction
8. **Final Optimization**: Brightness and contrast adjustments

### Output
- **High-resolution enhanced images** (8x larger than input)
- **False-color composites** optimized for satellite imagery
- **Same quality** as the noise reduction 8.0x images your sir liked

## File Structure

```
sir-final-project/
├── input_images/          # Upload your TIFF files here
│   ├── sentinel_data.tif
│   └── planet_data.tif
├── enhanced_outputs/      # Enhanced images saved here
│   ├── sentinel_noise_reduction_8.0x.png
│   └── planet_noise_reduction_8.0x.png
├── temp_processing/       # Temporary files (auto-created)
├── main.py               # Main script to run
└── README.md             # This file
```

## Supported Input Formats

- **TIFF files** with 4 bands (NIR, Red, Green, Blue)
- **Sentinel-2** data (auto-detected by filename containing 'sentinel' or 's2')
- **PlanetScope** data (auto-detected by filename containing 'planet' or 'ps')

## Output Naming

- Uses original filename + enhancement details
- Example: `1-sub-sentinel-30-april-2025_output_8.0x.png`
- Example: `4-sub-planetscope_output_4.0x.png`

## Requirements

- Python 3.7+
- OpenCV (`pip install opencv-python`)
- Rasterio (`pip install rasterio`)
- NumPy (`pip install numpy`)

## Installation

```bash
pip install opencv-python rasterio numpy
```

## Usage Example

1. Place your TIFF file in `input_images/sentinel_data.tif`
2. Run: `python main.py`
3. Get enhanced image: `enhanced_outputs/sentinel_noise_reduction_8.0x.png`

## Features

- **User-friendly**: Just upload TIFF files and run `python main.py`
- **Automatic detection**: Detects Sentinel vs Planet based on filename
- **Same quality**: Uses exact same processing as the images your sir liked
- **Clean output**: Only the final enhanced images, no intermediate files
- **Professional**: Same structure as your existing scripts

## Processing Details

This pipeline combines the best of both worlds:

- **Super-resolution generation** (like `create_comprehensive_sr_outputs.py`)
- **Visual enhancement** (like `apply_exact_visual_enhancement.py`)

The result is the exact same quality as the `noise_reduction_8.0x.png` images your sir preferred.

---

**Ready to use!** Just upload your TIFF files and run `python main.py`
