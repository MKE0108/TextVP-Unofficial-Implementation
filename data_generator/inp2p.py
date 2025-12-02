
import os
#set CUDA_VISIBLE_DEVICES=0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# `image` is an RGB PIL.Image
import glob
from PIL import Image
for path in glob.glob("/home/ksp0108/workspace/HW/video_gen/prompt-to-prompt/test_1130/*.png"):
    os.makedirs("dataset/dog/watorcolor", exist_ok=True)
    image = Image.open(path).convert("RGB")
    image.save(path.replace("test_1130", "dataset/dog/watorcolor"))
    images = pipe("turn into painting", image=image,num_inference_steps=50).images
    images[0].save(path.replace("test_1130", "dataset/dog/watorcolor").replace(".png", "_A.png"))

# for path in glob.glob("/home/ksp0108/workspace/HW/video_gen/prompt-to-prompt/dataset/dog/lego/*_B.png"):
#     os.makedirs("dataset/dog/dog2lion", exist_ok=True)
#     image = Image.open(path).convert("RGB")
#     image.save(path.replace("dog/lego", "dog/dog2lion"))
#     images = pipe("Convert the dog into a lion", image=image,num_inference_steps=50).images
#     images[0].save(path.replace("dog/lego", "dog/dog2lion").replace("B", "A"))