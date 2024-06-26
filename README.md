# PKG-Transformer
<p align="center">
  <img src="images/PKG-Transformer.png" alt="PKG-Transformer" width="800"/>
</p>


## Installation and Dependencies
Create the `m2` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate m2
```

## Data preparation
For the evaluation metrics, Please download the [evaluation.zip](https://pan.baidu.com/s/13ZfH-CMYbW3RsW0-RX7KKQ)(BaiduPan code:wuiu) and extract it to `./evaluation`. You can also refer to the [M2 Transformer](https://github.com/aimagelab/meshed-memory-transformer).


To run the code, object features and scene features for the Sydney-Captions dataset are needed.
Please download the [features.zip](https://pan.baidu.com/s/1o4m4f5doAtAsRr1Z7tv3lw)(BaiduPan code:sdy2) and extract it to `./datasets/Sydney_Captions/features/`.

## Train
```
python train.py
```

## Evaluate
```
python test.py
```

# Citation:
```
@ARTICLE{10298250,
  author={Meng, Lingwu and Wang, Jing and Yang, Yang and Xiao, Liang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Prior Knowledge-Guided Transformer for Remote Sensing Image Captioning}, 
  year={2023},
  volume={61},
  number={4706213},
  pages={1-13},
  doi={10.1109/TGRS.2023.3328181}}
```

## Reference:
1. https://github.com/tylin/coco-caption
2. https://github.com/aimagelab/meshed-memory-transformer
3. https://github.com/kywen1119/DSRAN
