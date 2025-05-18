# Bidirectional Efficient Attention Parallel Network for Segmentation of 3D Medical Imaging
by Dongsheng Wang, Tiezhen Xv*, Jiehui Liu, Jianshen Li , Lijie Yang and Jinxi Guo. 


### News
```
<04.08.2024> Our paper entitled "Bidirectional Efficient Attention Parallel Network for Segmentation of 3D Medical Imaging" has been accepted by Electronics;
```
```
```
We released the codes;
```
### Introduction
This repository is for our paper: '[Bidirectional Efficient Attention Parallel Network for Segmentation of 3D Medical Imaging]([https://doi.org/10.1016/j.media.2022.102530](https://doi.org/10.3390/electronics13153086)'. Note that, the MC-Net+ model is named as mcnet3d_v2 in our repository and we also provide the mcnet2d_v1 and mcnet3d_v1 versions.

### Requirements
This repository is based on PyTorch 1.13.1, CUDA 11.7 and Python 3.9.16; All experiments in our paper were conducted on a single NVIDIA GEFORCE RTX4080 GPU.

### Usage
1. Clone the repo.;
```
```
2. Put the data in './MC-Net/data';

3. Train the model;
```
cd MC-Net
# e.g., for 20% labels on LA
python ./code/train_mcnet_3d.py --dataset_name LA --model mcnet3d_v2 --labelnum 16 --gpu 0 --temperature 0.1
```
4. Test the model;
```
cd MC-Net
# e.g., for 20% labels on LA
python ./code/test_3d.py --dataset_name LA --model mcnet3d_v2 --exp MCNet --labelnum 16 --gpu 0
```

### Citation
If our BEAP-Net model is useful for your research, please consider citing:
      @article{Sadako-X,
        title={Bidirectional Efficient Attention Parallel Network for Segmentation of 3D Medical Imaging},
        author={Dongsheng Wang, Tiezhen Xv*, Jiehui Liu, Jianshen Li , Lijie Yang and Jinxi Guo},
        journal={Electronics},
        volume={13},
        issue={15},
        pages={3086},
        year={2024},
        }

### Acknowledgements:
Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at '13940263058@163.com'
