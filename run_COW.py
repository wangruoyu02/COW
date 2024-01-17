import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler
import time
import argparse
from pipeline_COW import COWPipeline
from PIL import Image
import numpy as np
from torchvision import transforms
import time


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=False,
        help="Path to input img",
        default = './data/images/0.jpg'
    )
    parser.add_argument(
        "--input_mask", type=str, required=False,
        help="Path to input mask",
        default = './data/masks/0.jpg'
    )
    parser.add_argument(
        "--prompt", type=str, required=False,
        help="input text condition",
        default = 'a photo of a person attending a formal event.'
    )

    parser.add_argument(
        "--output_dir", type=str, 
        default = "./results",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--model_path", type=str, 
        default = '/home/wry/CODE/COW_diffusers/models/stable-diffusion-2-1-base',
        help="Path to pretrained model.",
    )
    parser.add_argument(
        "--seed_size", type=int, 
        default = 256,
        help="The size of the seed initialization.",
    )
    parser.add_argument(
        "--seed_x_offset", type=int, 
        default = 128,
        help="The x coordinate of the seed initialization.",
    )
    parser.add_argument(
        "--seed_y_offset", type=int, 
        default = 0,
        help="The y coordinate of the seed initialization.",
    )
    parser.add_argument(
        "--seed", type=int, 
        default = 42,
        help="random seed",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, 
        default = 10,
        help="num_inference_steps of DDIM.",
    )



if __name__ == "__main__":
    """Example usage:
    python run_COW.py \
        --input_img ./data/images/0.jpg \
        --input_mask ./data/masks/0.jpg \
        --prompt "a person in a forest" \
        --model_path "/home/wry/CODE/COW_diffusers/models/stable-diffusion-2-1-base" \
        --output_dir ./results 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])



    pipeline = COWPipeline.from_pretrained(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    generator = torch.Generator("cuda").manual_seed(args.seed)


    face_image = Image.open(args.input_img).convert("RGB")
    mask_image = Image.open(args.input_mask).convert("RGB")
    prompt = args.prompt
    seed_size = args.seed_size
    face_image = face_image.resize((seed_size, seed_size))
    mask_image = mask_image.resize((seed_size, seed_size))



    os.makedirs(args.output_dir, exist_ok=True)
       
    start_time = time.time()
    image = pipeline(prompt,
                    generator=generator,
                    image=face_image,
                    mask_image=mask_image,
                    num_inference_steps=args.num_inference_steps,
                    seed_size=args.seed_size,
                    x_offset = args.seed_x_offset,
                    y_offset = args.seed_y_offset,
                    )[0]
    end_time = time.time()
    image.save(f"{args.output_dir}/{prompt}.jpg")


