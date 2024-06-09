# PAPR: Proximity Attention Point Rendering (NeurIPS 2023 Spotlight 🤩)
[Yanshu Zhang*](https://zvict.github.io/), [Shichong Peng*](https://sites.google.com/view/niopeng/home), [Alireza Moazeni](https://amoazeni75.github.io/), [Ke Li](https://www.sfu.ca/~keli/) (* denotes equal contribution)<br>

<img src="./images/SFU_AI.png" height=100px /><img src="images/APEX_lab.png" height=120px />

[Project Sites](https://zvict.github.io/papr)
 | [Paper](https://arxiv.org/abs/2307.11086) |
Primary contact: [Yanshu Zhang](https://zvict.github.io/)

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
    year={2023}
}
```

## Installation
```
git clone git@github.com:zvict/papr.git   # or 'git clone https://github.com/zvict/papr'
cd papr
conda env create -f papr.yml
conda activate papr
```
Or use virtual environment with `python=3.9`
```
python -m venv path/to/<env_name>
source path/to/<env_name>/bin/activate
pip install -r requirements.txt
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

The codebase has two main components: data loading part in `dataset/` and models in `models/`. Class `PAPR` in `models/model.py` defines our main model. All the configurations are in `configs/`, and `configs/demo.yml` is a demo configuration with comments of important arguments.

## Training
```
python train.py --opt configs/nerfsyn/chair.yml
```

## Evaluation
```
python test.py --opt configs/nerfsyn/chair.yml
```

## Pretrained Models

We provide pretrained models on NeRF Synthetic and Tanks&Temples datasets here: [Google Drive](https://drive.google.com/drive/folders/1fTWjuE-I30tBFCshbvC1W0TDdTlM-j82?usp=sharing).
To load the pretrained models, please put them under `checkpoints/`, and change the `test.load_path` in the config file.

## Exposure Control

We provide the scripts for the exposure control described in Section 4.4 and Appendix A.8. 
To finetune a pre-trained model with exposure control, run:
```
python exposure_control_train.py --opt configs/t2/Caterpillar_exposure_control.yml
```
To generate images with different exposures controlled by random latent codes, run:
```
python exposure_control_test.py --opt configs/t2/Caterpillar_exposure_control.yml --frame 0
```
To generate images by interpolating between two latent codes with different exposures, run:
```
python exposure_control_intrp.py --opt configs/t2/Caterpillar_exposure_control.yml --frame 0 --start_index 0 --end_index 1
```
We also provide a pre-trained model with exposure control on the Caterpillar scene in the Google Drive link above.

## Acknowledgement
This research was enabled in part by support provided by NSERC, the BC DRI Group and the Digital Research Alliance of Canada.
