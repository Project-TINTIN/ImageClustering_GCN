# ImageClustering_GCN

Adapted from dgl/examples/pytorch/hilander/

Setup
conda create -n Hilander # create env
conda activate Hilander # activate env
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch # install pytorch 1.7 version
conda install -y cudatoolkit=10.2 faiss-gpu=1.6.5 -c pytorch # install faiss gpu version matching cuda 10.2
pip install dgl-cu102 # install dgl for cuda 10.2
pip install tqdm # install tqdm
git clone https://github.com/yjxiong/clustering-benchmark.git # install clustering-benchmark for evaluation
cd clustering-benchmark
python setup.py install
cd ../

I also installed pandas for reading in my annotations file and opencv for visualization

train_subgCustom.py
In this version the feature vectors are numpy arrays saved as .npy files using numpy.save
Variables to edit are root_folder and train_data
train_data is an annotation file in the format below
subpath, index
subpath, index ...
.npy files are expected to be found in root_folder + subpath

