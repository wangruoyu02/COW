# Cyclic One-Way Diffusion (COW)

This is the official implementation of Diffusion in Diffusion: Cyclic One-Way Diffusion for Text-Vision-Conditioned Generation **(ICLR 2024)**

Ruoyu Wang, Yongqi Yang, ZhiHao Qian, Ye Zhu, Yu Wu

[Paper](https://arxiv.org/abs/2306.08247) | [Project Page](https://oho43.github.io/COW/)

![level](readme.assets/level-1705478439133.png)





## Introduction

We propose COW, **a training-free pipeline for one-shot versatile customization application scenarios**. We investigate the diffusion (physics) in diffusion (machine learning) properties and propose our Cyclic One-Way Diffusion (COW) method to control the direction of diffusion phenomenon given a pre-trained frozen diffusion model for versatile customization application scenarios, where the low-level pixel information from the conditioning needs to be preserved. 

![pipeline](readme.assets/pipeline.png)

## Setup

Our codebase is built on Stable Diffusion (Implementation of [diffusers](https://github.com/huggingface/diffusers)). Thanks very much.

The codebases are built on top of [diffusers](https://github.com/huggingface/diffusers). Thanks very much.

### Creating a Conda Environment

```
git clone https://github.com/oho43/COW.git
cd COW
conda env create -f cow_env.yaml
conda activate cow_ldm
```

### Downloading Stable-Diffusion Weights

Download the StableDiffusion weights from the [Stability AI at Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) (download the `stable-diffusion-2-1-base` model), and put it under `./models` folder. You can also choose other versions of the model.



## Running COW

### Data Preparation

Several input samples are available under `./data` directory. Each sample involves one image and one user mask that denotes the desired content of visual condition. The input data structure is like this:

```
data
├── images
│  ├── 0.jpg
│  ├── 1.jpg
│  ├── 2.jpg
├── masks
│  ├── 0.jpg
│  ├── 1.jpg
│  ├── 2.jpg
│  ├── ...
```

### Image Generation

You can inference the images with following command:

```
python run_COW.py \
    --input_img ./data/imgs/0.jpg \
    --input_mask ./data/masks/0.jpg \
    --prompt "a person in the forest" \
    --model_path "./models/stable-diffusion-2-1-base" \
    --output_dir ./results 
```

 

All supported arguments are listed below (type `python run_COW.py --help`).

```
usage: run_COW.py [-h] [--input_img INPUT_IMG] [--input_mask INPUT_MASK] [--prompt PROMPT] [--output_dir OUTPUT_DIR] [--model_path MODEL_PATH]
                  [--seed_size SEED_SIZE] [--seed_x_offset SEED_X_OFFSET] [--seed_y_offset SEED_Y_OFFSET] [--seed SEED]
                  [--num_inference_steps NUM_INFERENCE_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --input_img INPUT_IMG
                        Path to input img
  --input_mask INPUT_MASK
                        Path to input mask
  --prompt PROMPT       input text condition
  --output_dir OUTPUT_DIR
                        Output path to the directory with results.
  --model_path MODEL_PATH
                        Path to pretrained model.
  --seed_size SEED_SIZE
                        The size of the seed initialization.
  --seed_x_offset SEED_X_OFFSET
                        The x coordinate of the seed initialization.
  --seed_y_offset SEED_Y_OFFSET
                        The y coordinate of the seed initialization.
  --seed SEED           random seed
  --num_inference_steps NUM_INFERENCE_STEPS
                        num_inference_steps of DDIM.
```

## Whole Image Generation and Editing

You can directly apply COW to Whole image generation and editing applications by running the command:

```
python run_COW.py \
    --input_img path/to/your/input/ \
    --input_mask path/to/your/input/ \
    --prompt "Your input prompt" \
    --model_path "./models/stable-diffusion-2-1-base" \
    --output_dir ./results \
    --seed_size 512  \
    --seed_x_offset 0 \
    --seed_y_offset 0
```

 ![whole](readme.assets/whole.png)

## More Results

### Generalized Visual Consition

![generalized](readme.assets/generalized.png)

### Cross-Domain Transforma

![cross_domian](readme.assets/cross_domian-1705478250690.png)

### Tradeoffs between Text and Visual Conditions

COW offers a flexible change in the visual condition region according to the text guidance. The level of changes that may occur within the seed image depends on the discrepancy between textual and visual conditions.

![balance](readme.assets/balance.png)

## Citing

```
@article{yang2023diffusion,
  title={Diffusion in Diffusion: Cyclic One-Way Diffusion for Text-Vision-Conditioned Generatio},
  author={Yang, Yongqi and Wang, Ruoyu and Qian, Zhihao and Zhu, Ye and Wu, Yu},
  journal={arXiv preprint arXiv:2306.08247},
  year={2023}
}
```











> datasets (CelebA-HQ, COCO, and ImageNet), and that TF-ICON surpasses prior baselines in versatile visual domains.

[![teaser](https://github.com/Shilin-LU/TF-ICON/raw/main/assets/tf-icon.png)](https://github.com/Shilin-LU/TF-ICON/blob/main/assets/tf-icon.png)

> \