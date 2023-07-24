# Face-image-deblurring-with-feature-correction-and-fusion

## :wrench: Dependencies and Download

- NVIDIA GPU + CUDA >=11.2
- Linux Ubuntu 20.05

## Requirements
Then install the additional requirements
```
pip install -r requirements.txt
```

## Download pre-trained weights
Download [vgg13](https://drive.google.com/file/d/144QennVnPBDlBenTWV-NVob6_sxEuHQ5/view?usp=sharing) pre-training weights and [model](https://drive.google.com/drive/folders/1dT3tMIWjPTJaqhiHePcIz-W1NhPSgMY7?usp=drive_link) weights.
Then put them in the models folder.The hierarchy is as follows：

![image](https://github.com/xuyu666/Face-image-deblurring-with-feature-correction-and-fusion/assets/49869475/1b553952-1a3f-453c-90d7-5ca3db7ce06c)

## RUN code
```
python val_train.py
```


## Acknowledgement

This code is developed based on [StyleGAN2](https://arxiv.org/abs/1912.04958). Part of the code is borrowed from [Simple StyleGan2](https://github.com/lucidrains/stylegan2-pytorch).

(cheers to the community as well)
