# ANN2SNN_GN

Here is the code for ICASSP 2024 "Optimal ANN-SNN Conversion with Group Neurons".

### Before using GN
You should install [SpikingJelly](https://github.com/fangwei123456/spikingjelly) first:
```
pip install spikingjelly
```

### Demo
```
# !sh
#Train ANN with QCFS.

gpus=8
bs=160
lr=0.1
epochs=120
l=8
data='cifar100'
model='resnet20'
id=${model}-${data}

python main.py  train \
    --gpus=$gpus \
    --bs=$bs \
    --lr=$lr \
    --epochs=$epochs \
    --l=$l \
    --model=$model \
    --data=$data \
    --id=$id \
```
```
# !sh
#Convert the trained ANN to SNN, and test the SNN.

gpus=8
bs=128
l=8
data='cifar100'
model='resnet20'
id='your checkpoint id'
mode='ann'
sn_type='gn'  #'gn' means group neuron; 'if' means IF neuron
tau=6
t=32
device='cuda'
seed=42

python main.py  test \
    --gpus=$gpus \
    --bs=$bs \
    --l=$l \
    --model=$model \
    --data=$data \
    --mode=$mode  \
    --id=$id \
    --sn_type=$sn_type \
    --tau=$tau \
    --t=$t \
    --device=$device \
    --seed=$seed

