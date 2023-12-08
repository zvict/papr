# PAPR: Proximity Attention Point Rendering (NeurIPS 2023 Spotlight 🤩)
[Yanshu Zhang*](https://zvict.github.io/), [Shichong Peng*](https://sites.google.com/view/niopeng/home), [Alireza Moazeni](https://amoazeni75.github.io/), [Ke Li](https://www.sfu.ca/~keli/) (* denotes equal contribution)<br>

<img src="./images/SFU_AI.png" height=100px /><img src="images/APEX_lab.png" height=120px />

[Project Sites](https://zvict.github.io/papr)
 | [Paper](https://arxiv.org/abs/2307.11086) |
Primary contact: [Yanshu Zhang](https://zvict.github.io/)

### The code will be released soon!

Proximity Attention Point Rendering (PAPR) is a new method for joint novel view synthesis and 3D reconstruction. It simultaneously learns from scratch an accurate point cloud representation of the scene surface, and an attention-based neural network that renders the point cloud from novel views.

<!-- <img src="./images/pipeline.png" /> -->

[![NeurIPS 2023 Presentation](https://github.com/zvict/papr/blob/main/images/papr_video_cover.png)](https://youtu.be/1atBGH_pDHY)

## BibTeX
 <strong>PAPR: Proximity Attention Point Rendering</strong>.  &nbsp;&nbsp;&nbsp; 
```
@inproceedings{zhang2023papr,
    title={PAPR: Proximity Attention Point Rendering},
    author={Yanshu Zhang and Shichong Peng and Seyed Alireza Moazenipourasil and Ke Li},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=CgJJvuLjec}
}
```

## Installation
```
git clone git@github.com:zvict/papr.git   # or 'git clone https://github.com/zvict/papr'
cd papr
conda env create -f papr.yml
conda activate papr
```

## Data Preparation

Expected dataset structure in the source path location:
```
papr
├── data
│   ├── nerf_synthetic
│   │   ├── chair
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   │   ├── transforms_train.json
│   │   │   ├── transforms_val.json
│   │   │   ├── transforms_test.json
│   │   ├── ...
│   ├── tanks_temples
│   │   ├── Barn
│   │   │   ├── pose
│   │   │   ├── rgb
│   │   │   ├── intrinsics.txt
│   │   ├── ...
```
#### NeRF Synthetic
Download NeRF Synthetic Dataset from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put it under `data/nerf_synthetic/`


#### Tanks & Temples
Download [Tanks&Temples](https://www.tanksandtemples.org/) from [here](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip) and put it under:
`data/tanks_temples/`

## Overview

## Training
```
python train.py --opt configs/nerfsyn/chair.yml
```

## Evaluation
```
python test.py --opt configs/nerfsyn/chair.yml
```

## Acknowledgement
This research was enabled in part by support provided by NSERC, the BC DRI Group and the Digital Research Alliance of Canada.

## LICENSE
