
<p align="center">

  <h1 align="center">Face-image-deblurring-with-feature-correction-and-fusion</h1>
  <div align="center">
    <img src="video.gif", width="600">
  </div>
</p>

## :bookmark: Our Deblurred Results
We put the full results of the test set we showed in the paper in a zip file. [Deblurred Results](https://drive.google.com/file/d/1MR2ajIjOHs2sbjkihYtLaosjgoLa-mDu/view?usp=sharing)

## :whale: Our Demo(3.72G)
We now provide a set of **user-friendly test procedures** and **test sets** used for experiments,
Include two sets of Real-world blurred image and eight sets of Synthetic blurred image.
[Demo](https://drive.google.com/file/d/11acAvX6EUvXYYZtfkCxymHGaOMOhlDvF/view?usp=sharing)

## :sparkles: Our Test Code
## Dependencies and Download

- NVIDIA GPU + CUDA >=11.2
- Linux Ubuntu 18.04

## Requirements
Then install the additional requirements
```
pip install -r requirements.txt
```

## Download pre-trained weights
Download [vgg13](https://drive.google.com/file/d/144QennVnPBDlBenTWV-NVob6_sxEuHQ5/view?usp=sharing) pre-training weights and [model](https://drive.google.com/drive/folders/1dT3tMIWjPTJaqhiHePcIz-W1NhPSgMY7?usp=drive_link) weights.
Then put them in the models folder.The hierarchy is as follows：

![image](https://github.com/xuyu666/Face-image-deblurring-with-feature-correction-and-fusion/assets/49869475/25b0b512-f29f-4749-8dbb-0e3c91be7b94)


## RUN code
```
python val_train.py
```


## Acknowledgement

Part of the code is borrowed from [Simple StyleGan2](https://github.com/lucidrains/stylegan2-pytorch). Thanks to the [vgg](https://arxiv.org/abs/1409.1556)  teamfor the pre-training weights!

(cheers to the community as well)
