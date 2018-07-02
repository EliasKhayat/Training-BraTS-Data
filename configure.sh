#!/bin/bash
echo "Installing all necessary dependencies..."
    pip install nibabel
    pip install requests
    pip install multiprocessing
    pip install tensorflow
    pip install keras
    pip install opencv-python
    pip install matplotlib
    pip install numpy
    pip install scipy
    pip install scikit-image
    pip install IPython
echo "Done with installs!"
echo "Pulling BRATS 2018 Data down, this may take a few minutes..."
if command -v python3 &>/dev/null; then
  python3 Utils/getBraTs2018Data.py
else
  python Utils/getBraTS2018Data.py
fi
echo "Now to pull down the Mask R-CNN submodule..." 
git submodule update --init --recursive --remote
echo "Success! You should be good to go! :)" 


