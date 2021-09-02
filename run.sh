set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=./lib64/
python main.py