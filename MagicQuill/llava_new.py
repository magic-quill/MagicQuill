import torch
from transformers import TextStreamer
import webcolors
import os
import random
from collections import Counter
import numpy as np
from torchvision import transforms
from .magic_utils import get_colored_contour, find_different_colors, get_bounding_box_from_mask
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, expand2square, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import re

class LLaVAModel:
    def __init__(self):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(current_dir, "models", "llava-v1.5-7b-finetune-clean")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            load_4bit=False
        )

    def generate_description(self, images, question):
        qs = question
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        images_tensor = []
        image_sizes = []
        to_pil = transforms.ToPILImage()
        for image in images:
            image = image.clone().permute(2, 0, 1).cpu()
            image = to_pil(image)
            image_sizes.append(image.size)
            image = expand2square(image, tuple(int(x) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images_tensor.append(image.half())

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                temperature=0.2,
                do_sample=True,
                use_cache=True,
            )
        outputs = self.tokenizer.decode(output_ids[0]).strip()
        outputs = outputs.split('>')[1].split('<')[0]
        # print(outputs)
        return outputs

    def process(self, image, colored_image, add_mask):
        description = ""
        answer1 = ""
        answer2 = ""
        
        image_with_sketch = image.clone()
        if torch.sum(add_mask).item() > 0:
            x_min, y_min, x_max, y_max = get_bounding_box_from_mask(add_mask)
            # print(x_min, y_min, x_max, y_max)
            question = f"This is an 'I draw, you guess' game. I will upload an image containing some sketches. To help you locate the sketch, I will give you the normalized bounding box coordinates of the sketch where their original coordinates are divided by the image width and height. The top-left corner of the bounding box is at ({x_min}, {y_min}), and the bottom-right corner is at ({x_max}, {y_max}). Now tell me, what am I trying to draw with these sketches in the image?"
            # image_with_sketch[add_mask > 0.5] = 1.0
            bool_add_mask = add_mask > 0.5
            mean_brightness = image_with_sketch[bool_add_mask].mean()
            if mean_brightness > 0.8:
                image_with_sketch[bool_add_mask] = 0.0
            else:
                image_with_sketch[bool_add_mask] = 1.0
            answer1 = self.generate_description([image_with_sketch.squeeze() * 255], question)
            print(answer1)

        if not torch.equal(image, colored_image):
            color = find_different_colors(image.squeeze() * 255, colored_image.squeeze() * 255)
            image_with_bbox, colored_mask = get_colored_contour(colored_image.squeeze() * 255, image.squeeze() * 255)
            x_min, y_min, x_max, y_max = get_bounding_box_from_mask(colored_mask)
            question = f"The user will upload an image containing some contours in red color. To help you locate the contour, I will give you the normalized bounding box coordinates where their original coordinates are divided by the image width and height. The top-left corner of the bounding box is at ({x_min}, {y_min}), and the bottom-right corner is at ({x_max}, {y_max}). You need to identify what is inside the contours using a single word or phrase."
            answer2 = color + ', ' + self.generate_description([image_with_bbox.squeeze() * 255], question)
            print(answer2)

        return (description, answer1, answer2)
