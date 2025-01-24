# TMAE
## Introduction

This is the open source code for "TMAE: Entropy-aware Masked Autoencoder for Low-cost Traffic Flow Map Inference".

The framework of TMAE is as below:
![image](https://github.com/user-attachments/assets/89a63e17-1292-4482-ad34-7fb428ecf4c6)

## Dataset
We use the public dataset [TaxiBj](https://github.com/yoshall/UrbanFM/tree/master/data),[ChengDu and XiAn](https://github.com/luimoli/RATFM/tree/master/data).
```
# Example of file construction 
XiAn
<your_root_path>/data/XiAn/train/
                                X.npy/    # coarse-grained traffic flow maps
                                Y.npy/    # fine-grained traffic flow maps
                                ext.npy/  # external factor vectors
<your_root_path>/data/XiAn/valid/
                                X.npy/    
                                Y.npy/    
                                ext.npy/  
<your_root_path>/data/XiAn/test/
                                X.npy/    
                                Y.npy/    
                                ext.npy/  
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
