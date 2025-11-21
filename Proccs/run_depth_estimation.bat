@echo off
echo ====================================
echo Depth Anything V2 - Batch Processor
echo ====================================
echo.

echo Installing dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python tqdm accelerate

echo.
echo ====================================
echo Running depth estimation...
echo ====================================
python process_rgb_images.py

echo.
echo ====================================
echo Done!
echo ====================================
pause

