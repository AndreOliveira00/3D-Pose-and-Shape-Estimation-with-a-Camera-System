# install multi-object tracker
conda activate hybrik
sudo pip install -e git+https://github.com/mkocabas/yolov3-pytorch.git#egg=yolov3
sudo pip install -e git+https://github.com/haofanwang/multi-person-tracker.git#egg=multi-person-tracker
sudo pip install numba filterpy
# conda activate glamr

# apt
sudo apt install libgl1-mesa-glx xvfb

# mesa
sudo apt update
sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -f


conda activate mot
cd src/yolo_tracking
sudo pip install -v -e .

sudo pip install boxmot

cd ../..