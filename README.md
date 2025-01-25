# TMAE
## Introduction

This is the source code for "TMAE: Entropy-aware Masked Autoencoder for Low-cost Traffic Flow Map Inference".

The framework of TMAE is as below:
![tmae-model](https://github.com/user-attachments/assets/d5316372-fbe0-46c0-831a-6fa5a25b064b)


## Dataset
We use the public datasets [TaxiBj](https://github.com/yoshall/UrbanFM/tree/master/data),[ChengDu and XiAn](https://github.com/luimoli/RATFM/tree/master/data).
```
# Example of file construction 
XiAn
<your_root_path>/data/XiAn/train/
                                X.npy    # coarse-grained traffic flow maps
                                Y.npy    # fine-grained traffic flow maps
<your_root_path>/data/XiAn/valid/
                                X.npy   
                                Y.npy   
<your_root_path>/data/XiAn/test/
                                X.npy    
                                Y.npy     
```
## Usage
1. train
```
python train.py
```

2. test
```
python test.py
```
