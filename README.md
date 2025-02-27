# 🪶 MagicQuill: An Intelligent Interactive Image Editing System
<a href="https://magicquill.art/demo/"><img src="https://img.shields.io/static/v1?label=Project&message=magicquill.art&color=blue"></a>
<a href="https://arxiv.org/abs/2411.09703"><img src="https://img.shields.io/badge/arXiv-2411.09703-b31b1b.svg"></a>
<a href="https://huggingface.co/spaces/AI4Editing/MagicQuill"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)"></a>
<a href="https://creativecommons.org/licenses/by-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg"></a>



https://github.com/user-attachments/assets/8ee9663a-fef2-484a-a0b7-8427ab590424

There is an HD video on [Youtube](https://www.youtube.com/watch?v=5DiKfONMnE4).

[Zichen Liu](https://zliucz.github.io)<sup>\*,1,2</sup>, [Yue Yu](https://bruceyyu.github.io/)<sup>\*,1,2</sup>, [Hao Ouyang](https://ken-ouyang.github.io/)<sup>2</sup>, [Qiuyu Wang](https://github.com/qiuyu96)<sup>2</sup>, [Ka Leong Cheng](https://felixcheng97.github.io/)<sup>1,2</sup>, [Wen Wang](https://github.com/encounter1997)<sup>3,2</sup>, [Zhiheng Liu](https://johanan528.github.io/)<sup>4</sup>, [Qifeng Chen](https://cqf.io/)<sup>†,1</sup>, [Yujun Shen](https://shenyujun.github.io/)<sup>†,2</sup><br>
<sup>1</sup>HKUST <sup>2</sup>Ant Group <sup>3</sup>ZJU <sup>4</sup>HKU <sup>\*</sup>equal contribution <sup>†</sup>corresponding author

> TLDR: MagicQuill is an intelligent and interactive system achieving precise image editing.
>
> Key Features: 😎 User-friendly interface / 🤖 AI-powered suggestions / 🎨 Precise local editing

- [🪶 MagicQuill: An Intelligent Interactive Image Editing System](#-magicquill-an-intelligent-interactive-image-editing-system)
  - [TODO List](#todo-list)
  - [Update Log](#update-log)
  - [Hardware Requirements](#hardware-requirements)
  - [Docker Container](#docker-container)
  - [Setup](#setup)
  - [Tutorial](#tutorial)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)
  - [Note](#note)

## TODO List

- [x] Release the paper and demo page. Visit [magicquill.art](https://magicquill.art) 🪩
- [x] Release the code and checkpoints.
- [x] Release gradio demo.
- [x] Release ComfyUI MagicQuill custom node.

<img src="docs/comfyui.png" width="100%" >

## Update Log

- [2024.11.21] 📢 Update the save button; Fix path bug on Windows; Add `.bat` and `.sh` files for convenient environment install on Windows and Linux. Thanks [lior007](https://github.com/lior007) and [JamesIV4](https://github.com/JamesIV4).
- [2024.11.25] 📢 New UI Updates: Drag & Drop Images + Download Button: We've enhanced our interface with two exciting features! Now you can easily upload images with drag & drop functionality, and quickly save your work using our new download button. Try it out and let us know what you think!
- [2024.12.06] 📢 New Feature Updates: Auto-save and Resolution Adjustment are now enabled in the parameter settings. Thanks [Furkan Gözükara](https://github.com/FurkanGozukara) for his brilliant suggestions.
- [2024.12.07] 🎉 Exciting News: ComfyUI MagicQuill Node has been released! Check the repository https://github.com/magic-quill/ComfyUI_MagicQuill for more details.
- [2024.12.16] 🎉 Exciting News: MagicQuill is also available at [Modelscope](https://modelscope.cn/studios/ant-research/MagicQuill_demo). Thanks for their amazing support and infrastructure.
- [2025.01.02] 🎉 Exciting News: MagicQuill docker container is now available. You can now build & run your own image in a cleaner, isolated environment. Thanks [gbudge](https://github.com/gbudge) for his contribution.
- [2025.02.27] 🎉 Exciting News: MagicQuill has been accepted to CVPR 2025! Looking forward to meeting everyone in Tennessee!

To update the latest features, pull the latest code and re-install the gradio_magicquill:
```
pip uninstall -y gradio_magicquill-0.0.1-py3-none-any.whl
pip install gradio_magicquill-0.0.1-py3-none-any.whl
```

## Hardware Requirements

- GPU is required to run MagicQuill. **Through our testing, we have confirmed that the model can run on GPUs with 8GB VRAM (RTX4070 Laptop).**

For users with limited GPU resources, please try our [Huggingface Demo](https://huggingface.co/spaces/AI4Editing/MagicQuill) and [Modelscope Demo](https://modelscope.cn/studios/ant-research/MagicQuill_demo). Also, consider disabling the DrawNGuess automatic prompt filling by clicking the wand icon above if it takes too long time on your machine.

##  Setup
Follow the following guide to set up the environment.

1. git clone repo. **Please don't forget the `--recursive` flag.** Otherwise, you will find `LLaVA` submodule missing.
    ```
    git clone --recursive https://github.com/magic-quill/MagicQuill.git
    cd MagicQuill
    ```
2. download and unzip checkpoints
    ```
    wget -O models.zip "https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliucz_connect_ust_hk/EWlGF0WfawJIrJ1Hn85_-3gB0MtwImAnYeWXuleVQcukMg?e=Gcjugg&download=1"
    unzip models.zip
    ```
    If the .zip file is not accessible, download it via browser. All checkpoints are about 25 GB in total. It may take some time to download. Alternatively, check our checkpoints at [huggingface](https://huggingface.co/LiuZichen/MagicQuill-models).

---

If you are a Windows user, you may try to use `windows_setup.bat` to conveniently install environments, just enter `windows_setup.bat` in your Python shell. For Linux user, check `linux_setup.sh`.

Alternatively, follow the step-by-step installation guide.

3. create environment
    ```
    conda create -n MagicQuill python=3.10 -y
    conda activate MagicQuill
    ```

4. install torch with GPU support
    ```
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    ```

5. install the interface
    ```
    pip install gradio_magicquill-0.0.1-py3-none-any.whl
    ```
    
6. install llava environment
    ```
    (For Linux)
    cp -f pyproject.toml MagicQuill/LLaVA/
    pip install -e MagicQuill/LLaVA/
    ```
    or
    ```
    (For Windows)
    copy /Y pyproject.toml MagicQuill\LLaVA\
    pip install -e MagicQuill\LLaVA\
    ```
    (For Windows PowerShell, the first line should be `Copy-Item -Path pyproject.toml -Destination "MagicQuill\LLaVA" -Force`)
    

7. install the remaining environment
    ```
    pip install -r requirements.txt
    ```

8. run magicquill
    ```
    python gradio_run.py
    ```
    If you are mainland user, you may try `export HF_ENDPOINT=https://hf-mirror.com` to use huggingface mirror to facilitate the download of some necessary checkpoints to run our system.


## Docker Container

You can build a docker container with MagicQuill as follows:

1. git clone repo. **Please don't forget the `--recursive` flag.** Otherwise, you will find `LLaVA` submodule missing.
    ```
    git clone --recursive https://github.com/magic-quill/MagicQuill.git
    cd MagicQuill
    ```

2. download and unzip checkpoints
    ```
    wget -O models.zip "https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliucz_connect_ust_hk/EWlGF0WfawJIrJ1Hn85_-3gB0MtwImAnYeWXuleVQcukMg?e=Gcjugg&download=1"
    unzip models.zip
    ```
    If the .zip file is not accessible, download it via browser. All checkpoints are about 25 GB in total. It may take some time to download. Alternatively, check our checkpoints at [huggingface](https://huggingface.co/LiuZichen/MagicQuill-models).
    
    Note: these can be located anywhere on the host computer, but Docker Compose expects them to be in `/data/magicquill/models` by default. Update `docker-compose.yaml` if you unzip them to another location.

3. build the image
   ```
   docker compose build
   ```

4. run the image
   ```
   docker compose up -d
   ```

## Tutorial

Please read before you try!

<!DOCTYPE html>
<html>
<body>
<div class="tutorial"><div><h3 align="center" class="heading">I. Three type of magic quills</h3><div align="center"><img fill="white" src="docs/icons/brush_edge_add.svg" alt="SVG image" class="icon" width="100"></div><div align="center"><br>Use the <b>add brush</b> to add details and elements guided by prompts - express your ideas with your own lively strokes!<br></div><div class="ant-row css-1kuana8"><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/deer.gif" alt="gif description" class="gif"></div><div align="center"><small>"With just a few strokes, a vivid little deer comes to life"</small><br></div></div><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/necklace.gif" alt="gif description" class="gif"></div><div align="center"><small> "Adorn the beautiful lady with a necklace"</small><br></div></div></div><div align="center"><br><img src="docs/icons/brush_edge_remove.svg" alt="SVG image" class="icon" width="100"></div><div align="center"><br>The <b>subtract brush</b> can remove excess details or redraw areas based on prompts. If there's anything you're not satisfied with, just subtract it away!<br></div><div class="ant-row css-1kuana8"><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/dolphin.gif" alt="gif description" class="gif"></div><div align="center"><small> "A dolphin with two tail fins? Let's give it a quick 'treatment'!"</small><br></div></div><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/skeleton cowboy.gif" alt="gif description" class="gif"></div><div align="center"><small> "Let's take off Mr. Skeleton's hat and help him cool down."</small><br><br></div></div></div><div align="center" style="display: flex; justify-content: center; align-items: center; gap: 10px;"><br><img src="docs/icons/brush_edge_add.svg" alt="add brush" class="icon" width="100"><span>&amp;</span><img src="docs/icons/brush_edge_remove.svg" alt="minus brush" class="icon" width="100"></div><div align="center"><br>Combine the <b>add and subtract brushes</b> to create amazing combo effects!<br></div><div class="ant-row css-1kuana8"><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/mona lisa cat.gif" alt="gif description" class="gif"></div><div align="center"><small> "Let's give Mona Lisa a pet cat~"</small><br></div></div><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/handsome bowtie.gif" alt="gif description" class="gif"></div><div align="center"><small> "Let's give this handsome fellow a new tie!"</small><br><br></div></div></div><div align="center"><img src="docs/icons/brush.svg" alt="SVG image" class="icon" width="100"></div><div align="center"><br>The <b>color brush</b> can precisely color the image, matching the color of your brush~<br></div><div class="ant-row css-1kuana8"><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/beautiful hair.gif" alt="gif description" class="gif"></div><div align="center"><small>"Precise color highlighting - paint exactly where you want to color"</small><br></div></div><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/cake flowers.gif" alt="gif description" class="gif"></div><div align="center"><small> "Don't you think the blue flowers look more dreamy than the pink ones?"</small><br><br></div></div></div><div align="center">*Please note the color brush and add&amp;subtract brush are mutually exclusive - you can only use one at a time!<br><br></div><hr><h3 align="center" class="heading">II. Draw and Guess</h3><div align="center"><img src="docs/icons/wand.svg" alt="SVG image" class="icon" width="100"></div><div align="center">Our brush is super smart! Look at the examples above - as soon as you finish drawing, it quickly guesses what you want to create and fills in the prompts for you~ Sometimes it might guess wrong though, so feel free to tell it what you actually want to draw~<br></div><div class="ant-row ant-row-center css-1kuana8"><div class="ant-col ant-col-xs-24 ant-col-md-12 css-1kuana8"><div align="center"><br><img width="300" src="docs/gifs/path.gif" alt="gif description" class="gif"></div><div align="center"><small> "Oops! I don't want to draw a vine, I want to create a path!"</small><br><br></div></div></div><hr><h3 align="center" class="heading">III. Super useful canvas tools!</h3><div align="center"><br><img src="docs/icons/upload.svg" alt="SVG image" class="icon" width="100"></div><div align="center">Click this button to upload the photo you want to edit~<br></div><div align="center"><br><img src="docs/icons/eraser.svg" alt="SVG image" class="icon" width="100"></div><div align="center">Made a mistake with the brush? Just erase it with the rubber tool!<br></div><div align="center"><br><img src="docs/icons/cursor.svg" alt="SVG image" class="icon" width="100"></div><div align="center">Drag, rotate, and resize your strokes with the cursor - just like when you're working in PowerPoint!<br></div><br><div align="center" style="display: flex; justify-content: center; align-items: center; gap: 10px;"><img src="docs/icons/undo.svg" alt="add brush" class="icon" width="100"><span>&amp;</span><img src="docs/icons/redo.svg" alt="minus brush" class="icon" width="100"></div><div align="center">Left is ctrl+z, right is ctrl+y - you know what that means! 😊<br>And for Mac users, left is command+z, right is command+shift+z! 😝<br></div><div align="center"><br><img src="docs/icons/delete.svg" alt="SVG image" class="icon" width="100"></div><div align="center">Oops! That doesn't look right 😵 - click this trash bin to delete the stroke<br></div><div align="center"><br><img src="docs/icons/eye.svg" alt="SVG image" class="icon" width="100"></div><div align="center">The brush strokes are in my way, how can I see the image😡?! Try clicking this button to temporarily hide your strokes<br></div><br><div align="center" style="display: flex; justify-content: center; align-items: center; gap: 10px;"><img src="docs/icons/accept.svg" alt="add brush" class="icon" width="100"><span>&amp;</span><img src="docs/icons/discard.svg" alt="minus brush" class="icon" width="100"></div><div align="center">These two icons will appear after the image is generated...<br>I love this generated image 😍, I want to keep editing! ➡️ Click ✅ to continue editing<br>What is this thing 😡, I don't want to see it! ➡️ Click ❎ to discard the result<br><br></div><hr><h3 align="center" class="heading">IV. Notes</h3><div align="center"><br><img src="docs/icons/loading.svg" alt="SVG image" class="icon" width="100"></div><div align="center">When you see the spinning icon in the bottom left corner, it means the magicquill is still charging up 💪 Wait for it to disappear before clicking the Run button!<br></div><div align="center"><br><img src="docs/icons/wand.svg" alt="SVG image" class="icon" width="100"></div><div align="center">When the magic wand is flashing, our brush is working hard to guess what you're trying to draw 🤔 Please be patient! 🙏<br><br></div><hr><h3 align="center" class="heading">V. Parameters</h3><div align="center">If you've made it here, you must really love our work! 😍<br>If you want to learn how to better control the generation results, don't miss this section! 😘<br>Next to the Run button, you can select parameters to modify advanced settings 🧐<br><br></div><div><ul><li><u>Base Model Name</u>: Users can adjust this to select appropriate base models for different editing styles<ul><li><a href="https://civitai.com/models/4201/realistic-vision-v60-b1">SD1.5/realisticVisionV60B1_v51VAE.safetensors</a>: This generates realistic style images! Use this most of the time.</li><li><a href="https://civitai.com/models/4384?modelVersionId=128713">SD1.5/DreamShaper.safetensors</a>: This one is for generating fantasy style images</li><li><a href="https://civitai.com/models/43331/majicmix-realistic">SD1.5/majicMIX_realistic</a>: This one is good at generating portraits 👩</li><li><a href="https://civitai.com/models/7240?modelVersionId=948574">SD1.5/MeinaMix.safetensors</a>: This one is good at generating anime images.</li><li><a href="https://civitai.com/models/36520/ghostmix">SD1.5/ghostmix_v20Bakedvae.safetensors</a>: Another model for anime image generation.</li><li>If there are any models you'd like to add, contact us!</li></ul></li><li><u>Negative Prompt</u>: Users can input content they want the model to avoid generating. Whatever you don't want to generate, put it here.</li><li><u>Fine Edge</u>: Users can enable this option to activate fine edge control.</li><li><u>Grow Size</u>: Adjust this parameter to set the pixel range affected around brush strokes when editing images, to expand/reduce the brush stroke influence area.</li><li><u>Edge Strength</u>: Parameter for adjusting the add/subtract brush control strength. Simply put, if you're confident in your drawing skills, you can increase strength. If you're a bad drawer like us 🤦, please keep the parameter as is, or reduce the strength a bit.</li><li><u>Color Strength</u>: Parameter for adjusting the color brush control strength, can adjust the image's coloring effects.</li><li>The remaining parameters are just some common parameters for diffusion models! You basically don't need to manage these, but if you're in the industry/AI art expert, feel free to try adjusting them.</li></ul></div>
</body>
</html>

## Citation
Don't forget to cite this source if it proves useful in your research!
```bibtex
@article{liu2024magicquill, 
	title={MagicQuill: An Intelligent Interactive Image Editing System}, 
	author={Zichen Liu and Yue Yu and Hao Ouyang and Qiuyu Wang and Ka Leong Cheng and Wen Wang and Zhiheng Liu and Qifeng Chen and Yujun Shen}, 
	year={2024}, 
	eprint={2411.09703}, 
	archivePrefix={arXiv}, 
	primaryClass={cs.CV}}
```

## Acknowledgement
Our implementation is based on 
- [ComfyUI-BrushNet](https://github.com/nullquant/ComfyUI-BrushNet)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
- [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet)
- [fabric.js](https://github.com/fabricjs/fabric.js)

Thanks for their remarkable contribution and released code!

## Note
Note: This repo is governed by the license of CC BY-NC 4.0 We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc. 

(注：本仓库受CC BY-NC的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
