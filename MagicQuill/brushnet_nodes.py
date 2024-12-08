import os
import types
from typing import Tuple

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import sys

import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.sd1_clip
import comfy.ldm.models.autoencoder
import comfy.supported_models

import folder_paths

from .model_patch import add_model_patch_option, patch_model_function_wrapper
from .brushnet.brushnet import BrushNetModel
from .brushnet.brushnet_ca import BrushNetModel as PowerPaintModel
from .brushnet.powerpaint_utils import TokenizerWrapper, add_tokens

current_directory = os.path.dirname(os.path.abspath(__file__))
brushnet_config_file = os.path.join(current_directory, 'brushnet', 'brushnet.json')
brushnet_xl_config_file = os.path.join(current_directory, 'brushnet', 'brushnet_xl.json')
powerpaint_config_file = os.path.join(current_directory,'brushnet', 'powerpaint.json')

sd15_scaling_factor = 0.18215
sdxl_scaling_factor = 0.13025

print(sys.path)
ModelsToUnload = [comfy.sd1_clip.SD1ClipModel, 
                  comfy.ldm.models.autoencoder.AutoencoderKL
                 ]


class BrushNetLoader:
    @classmethod
    def INPUT_TYPES(self):
        self.inpaint_files = get_files_with_extension('inpaint')
        return {"required":
                    {    
                        "brushnet": ([file for file in self.inpaint_files], ),
                        "dtype": (['float16', 'bfloat16', 'float32', 'float64'], ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("BRMODEL",)
    RETURN_NAMES = ("brushnet",)

    FUNCTION = "brushnet_loading"

    def brushnet_loading(self, brushnet, dtype):
        brushnet_file = os.path.join(self.inpaint_files[brushnet], brushnet)
        print('BrushNet model file:', brushnet_file)
        is_SDXL = False
        is_PP = False
        sd = comfy.utils.load_torch_file(brushnet_file)
        brushnet_down_block, brushnet_mid_block, brushnet_up_block, keys = brushnet_blocks(sd)
        del sd
        if brushnet_down_block == 24 and brushnet_mid_block == 2 and brushnet_up_block == 30:
            is_SDXL = False
            if keys == 322:
                is_PP = False
                print('BrushNet model type: SD1.5')
            else:
                is_PP = True
                print('PowerPaint model type: SD1.5')
        elif brushnet_down_block == 18 and brushnet_mid_block == 2 and brushnet_up_block == 22:
            print('BrushNet model type: Loading SDXL')
            is_SDXL = True
            is_PP = False
        else:
            raise Exception("Unknown BrushNet model")

        with init_empty_weights():
            if is_SDXL:
                brushnet_config = BrushNetModel.load_config(brushnet_xl_config_file)
                brushnet_model = BrushNetModel.from_config(brushnet_config)
            elif is_PP:
                brushnet_config = PowerPaintModel.load_config(powerpaint_config_file)
                brushnet_model = PowerPaintModel.from_config(brushnet_config)
            else:
                brushnet_config = BrushNetModel.load_config(brushnet_config_file)
                brushnet_model = BrushNetModel.from_config(brushnet_config)

        if is_PP:
            print("PowerPaint model file:", brushnet_file)
        else:
            print("BrushNet model file:", brushnet_file)

        if dtype == 'float16':
            torch_dtype = torch.float16
        elif dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif dtype == 'float32':
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float64

        brushnet_model = load_checkpoint_and_dispatch(
            brushnet_model,
            brushnet_file,
            device_map="sequential",
            max_memory=None,
            offload_folder=None,
            offload_state_dict=False,
            dtype=torch_dtype,
            force_hooks=False,
        )

        if is_PP: 
            print("PowerPaint model is loaded")
        elif is_SDXL:
            print("BrushNet SDXL model is loaded")
        else:
            print("BrushNet SD1.5 model is loaded")

        return ({"brushnet": brushnet_model, "SDXL": is_SDXL, "PP": is_PP, "dtype": torch_dtype}, )


class PowerPaintCLIPLoader:

    @classmethod
    def INPUT_TYPES(self):
        self.inpaint_files = get_files_with_extension('inpaint', ['.bin'])
        self.clip_files = get_files_with_extension('clip')
        return {"required":
                    {    
                        "base": ([file for file in self.clip_files], ),
                        "powerpaint": ([file for file in self.inpaint_files], ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)

    FUNCTION = "ppclip_loading"

    def ppclip_loading(self, base, powerpaint):
        base_CLIP_file = os.path.join(self.clip_files[base], base)
        pp_CLIP_file = os.path.join(self.inpaint_files[powerpaint], powerpaint)

        pp_clip = comfy.sd.load_clip(ckpt_paths=[base_CLIP_file])

        print('PowerPaint base CLIP file: ', base_CLIP_file)

        pp_tokenizer = TokenizerWrapper(pp_clip.tokenizer.clip_l.tokenizer)
        pp_text_encoder = pp_clip.patcher.model.clip_l.transformer

        add_tokens(
            tokenizer = pp_tokenizer,
            text_encoder = pp_text_encoder,
            placeholder_tokens = ["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens = ["a", "a", "a"],
            num_vectors_per_token = 10,
        )

        pp_text_encoder.load_state_dict(comfy.utils.load_torch_file(pp_CLIP_file), strict=False)

        print('PowerPaint CLIP file: ', pp_CLIP_file)

        pp_clip.tokenizer.clip_l.tokenizer = pp_tokenizer
        pp_clip.patcher.model.clip_l.transformer = pp_text_encoder

        return (pp_clip,)
    

class PowerPaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "model": ("MODEL",),
                        "vae": ("VAE", ),
                        "image": ("IMAGE",),
                        "mask": ("MASK",),
                        "powerpaint": ("BRMODEL", ),
                        "clip": ("CLIP", ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "fitting" : ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                        "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'], ),
                        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                        "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                        "save_memory": (['none', 'auto', 'max'], ),
                     },
        }
    
    CATEGORY = "inpaint"
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("model","positive","negative","latent",)

    FUNCTION = "model_update"

    def model_update(self, model, vae, image, mask, powerpaint, clip, positive, negative, fitting, function, scale, start_at, end_at, save_memory):

        is_SDXL, is_PP = check_compatibility(model, powerpaint)
        if not is_PP:
            raise Exception("BrushNet model was loaded, please use BrushNet node")  

        # Make a copy of the model so that we're not patching it everywhere in the workflow.
        model = model.clone()

        # prepare image and mask
        # no batches for original image and mask
        masked_image, mask = prepare_image(image, mask)

        batch = masked_image.shape[0]
        #width = masked_image.shape[2]
        #height = masked_image.shape[1]

        if hasattr(model.model.model_config, 'latent_format') and hasattr(model.model.model_config.latent_format, 'scale_factor'):
            scaling_factor = model.model.model_config.latent_format.scale_factor
        else:
            scaling_factor = sd15_scaling_factor

        torch_dtype = powerpaint['dtype']

        # prepare conditioning latents
        conditioning_latents = get_image_latents(masked_image, mask, vae, scaling_factor)
        conditioning_latents[0] = conditioning_latents[0].to(dtype=torch_dtype).to(powerpaint['brushnet'].device)
        conditioning_latents[1] = conditioning_latents[1].to(dtype=torch_dtype).to(powerpaint['brushnet'].device)

        # prepare embeddings

        if function == "object removal":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"
            print('You should add to positive prompt: "empty scene blur"')
            #positive = positive + " empty scene blur"
        elif function == "context aware":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = ""
            negative_promptB = ""
            #positive = positive + " empty scene"
            print('You should add to positive prompt: "empty scene"')
        elif function == "shape guided":
            promptA = "P_shape"
            promptB = "P_ctxt"
            negative_promptA = "P_shape"
            negative_promptB = "P_ctxt"
        elif function == "image outpainting":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"
            #positive = positive + " empty scene"
            print('You should add to positive prompt: "empty scene"')
        else:
            promptA = "P_obj"
            promptB = "P_obj"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"

        tokens = clip.tokenize(promptA)
        prompt_embedsA = clip.encode_from_tokens(tokens, return_pooled=False)

        tokens = clip.tokenize(negative_promptA)
        negative_prompt_embedsA = clip.encode_from_tokens(tokens, return_pooled=False)

        tokens = clip.tokenize(promptB)
        prompt_embedsB = clip.encode_from_tokens(tokens, return_pooled=False)

        tokens = clip.tokenize(negative_promptB)
        negative_prompt_embedsB = clip.encode_from_tokens(tokens, return_pooled=False)

        prompt_embeds_pp = (prompt_embedsA * fitting + (1.0 - fitting) * prompt_embedsB).to(dtype=torch_dtype).to(powerpaint['brushnet'].device)
        negative_prompt_embeds_pp = (negative_prompt_embedsA * fitting + (1.0 - fitting) * negative_prompt_embedsB).to(dtype=torch_dtype).to(powerpaint['brushnet'].device)

        # unload vae and CLIPs
        del vae
        del clip
        for loaded_model in comfy.model_management.current_loaded_models:
            if type(loaded_model.model.model) in ModelsToUnload:
                comfy.model_management.current_loaded_models.remove(loaded_model)
                loaded_model.model_unload()
                del loaded_model

        # apply patch to model

        brushnet_conditioning_scale = scale
        control_guidance_start = start_at
        control_guidance_end = end_at

        if save_memory != 'none':
            powerpaint['brushnet'].set_attention_slice(save_memory)

        add_brushnet_patch(model, 
                           powerpaint['brushnet'],
                           torch_dtype,
                           conditioning_latents, 
                           (brushnet_conditioning_scale, control_guidance_start, control_guidance_end), 
                           negative_prompt_embeds_pp, prompt_embeds_pp, 
                           None, None, None,
                           False)

        latent = torch.zeros([batch, 4, conditioning_latents[0].shape[2], conditioning_latents[0].shape[3]], device=powerpaint['brushnet'].device)

        return (model, positive, negative, {"samples":latent},)

    
class BrushNet:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "model": ("MODEL",),
                        "vae": ("VAE", ),
                        "image": ("IMAGE",),
                        "mask": ("MASK",),
                        "brushnet": ("BRMODEL", ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                        "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     },
        }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("model","positive","negative","latent",)

    FUNCTION = "model_update"

    def model_update(self, model, vae, image, mask, brushnet, positive, negative, scale, start_at, end_at):

        is_SDXL, is_PP = check_compatibility(model, brushnet)

        if is_PP:
            raise Exception("PowerPaint model was loaded, please use PowerPaint node")  

        # Make a copy of the model so that we're not patching it everywhere in the workflow.
        model = model.clone()

        # prepare image and mask
        # no batches for original image and mask
        masked_image, mask = prepare_image(image, mask)

        batch = masked_image.shape[0]
        width = masked_image.shape[2]
        height = masked_image.shape[1]

        if hasattr(model.model.model_config, 'latent_format') and hasattr(model.model.model_config.latent_format, 'scale_factor'):
            scaling_factor = model.model.model_config.latent_format.scale_factor
        elif is_SDXL:
            scaling_factor = sdxl_scaling_factor
        else:
            scaling_factor = sd15_scaling_factor

        torch_dtype = brushnet['dtype']

        # prepare conditioning latents
        conditioning_latents = get_image_latents(masked_image, mask, vae, scaling_factor)
        conditioning_latents[0] = conditioning_latents[0].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        conditioning_latents[1] = conditioning_latents[1].to(dtype=torch_dtype).to(brushnet['brushnet'].device)

        # unload vae
        del vae
        for loaded_model in comfy.model_management.current_loaded_models:
            if type(loaded_model.model.model) in ModelsToUnload:
                comfy.model_management.current_loaded_models.remove(loaded_model)
                loaded_model.model_unload()
                del loaded_model

        # prepare embeddings

        prompt_embeds = positive[0][0].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        negative_prompt_embeds = negative[0][0].to(dtype=torch_dtype).to(brushnet['brushnet'].device)

        max_tokens = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
        if prompt_embeds.shape[1] < max_tokens:
            multiplier = max_tokens // 77 - prompt_embeds.shape[1] // 77
            prompt_embeds = torch.concat([prompt_embeds] + [prompt_embeds[:,-77:,:]] * multiplier, dim=1)
            print('BrushNet: negative prompt more than 75 tokens:', negative_prompt_embeds.shape, 'multiplying prompt_embeds')
        if negative_prompt_embeds.shape[1] < max_tokens:
            multiplier = max_tokens // 77 - negative_prompt_embeds.shape[1] // 77
            negative_prompt_embeds = torch.concat([negative_prompt_embeds] + [negative_prompt_embeds[:,-77:,:]] * multiplier, dim=1)
            print('BrushNet: positive prompt more than 75 tokens:', prompt_embeds.shape, 'multiplying negative_prompt_embeds')

        if len(positive[0]) > 1 and 'pooled_output' in positive[0][1] and positive[0][1]['pooled_output'] is not None:
            pooled_prompt_embeds = positive[0][1]['pooled_output'].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        else:
            print('BrushNet: positive conditioning has not pooled_output')
            if is_SDXL:
                print('BrushNet will not produce correct results')
            pooled_prompt_embeds = torch.empty([2, 1280], device=brushnet['brushnet'].device).to(dtype=torch_dtype)

        if len(negative[0]) > 1 and 'pooled_output' in negative[0][1] and negative[0][1]['pooled_output'] is not None:
            negative_pooled_prompt_embeds = negative[0][1]['pooled_output'].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        else:
            print('BrushNet: negative conditioning has not pooled_output')
            if is_SDXL:
                print('BrushNet will not produce correct results')
            negative_pooled_prompt_embeds = torch.empty([1, pooled_prompt_embeds.shape[1]], device=brushnet['brushnet'].device).to(dtype=torch_dtype)

        time_ids = torch.FloatTensor([[height, width, 0., 0., height, width]]).to(dtype=torch_dtype).to(brushnet['brushnet'].device)

        if not is_SDXL:
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            time_ids = None

        # apply patch to model

        brushnet_conditioning_scale = scale
        control_guidance_start = start_at
        control_guidance_end = end_at

        add_brushnet_patch(model, 
                           brushnet['brushnet'],
                           torch_dtype,
                           conditioning_latents, 
                           (brushnet_conditioning_scale, control_guidance_start, control_guidance_end), 
                           prompt_embeds, negative_prompt_embeds,
                           pooled_prompt_embeds, negative_pooled_prompt_embeds, time_ids,
                           False)

        latent = torch.zeros([batch, 4, conditioning_latents[0].shape[2], conditioning_latents[0].shape[3]], device=brushnet['brushnet'].device)

        return (model, positive, negative, {"samples":latent},)


class BlendInpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "inpaint": ("IMAGE",),
                        "original": ("IMAGE",),
                        "mask": ("MASK",),
                        "kernel": ("INT", {"default": 10, "min": 1, "max": 1000}),
                        "sigma": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 1000}),
                     },
                "optional":
                    {
                        "origin": ("VECTOR",),
                    },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("image","MASK",)

    FUNCTION = "blend_inpaint"

    def blend_inpaint(self, inpaint: torch.Tensor, original: torch.Tensor, mask, kernel: int, sigma:int, origin=None) -> Tuple[torch.Tensor]:

        original, mask = check_image_mask(original, mask, 'Blend Inpaint')

        if len(inpaint.shape) < 4:
            # image tensor shape should be [B, H, W, C], but batch somehow is missing
            inpaint = inpaint[None,:,:,:]

        if inpaint.shape[0] < original.shape[0]:
            print("Blend Inpaint gets batch of original images (%d) but only (%d) inpaint images" % (original.shape[0], inpaint.shape[0]))
            original= original[:inpaint.shape[0],:,:]
            mask = mask[:inpaint.shape[0],:,:]

        if inpaint.shape[0] > original.shape[0]:
            # batch over inpaint
            count = 0
            original_list = []
            mask_list = []
            origin_list = []
            while (count < inpaint.shape[0]):
                for i in range(original.shape[0]):
                    original_list.append(original[i][None,:,:,:])
                    mask_list.append(mask[i][None,:,:])
                    if origin is not None:
                        origin_list.append(origin[i][None,:])
                    count += 1
                    if count >= inpaint.shape[0]:
                        break
            original = torch.concat(original_list, dim=0)
            mask = torch.concat(mask_list, dim=0)
            if origin is not None:
                origin = torch.concat(origin_list, dim=0)

        if kernel % 2 == 0:
            kernel += 1
        transform = T.GaussianBlur(kernel_size=(kernel, kernel), sigma=(sigma, sigma))

        ret = []
        blurred = []
        for i in range(inpaint.shape[0]):
            if origin is None:
                blurred_mask = transform(mask[i][None,None,:,:]).to(original.device).to(original.dtype)
                blurred.append(blurred_mask[0])

                result = torch.nn.functional.interpolate(
                    inpaint[i][None,:,:,:].permute(0, 3, 1, 2), 
                    size=(
                        original[i].shape[0], 
                        original[i].shape[1],
                    )
                ).permute(0, 2, 3, 1).to(original.device).to(original.dtype)
            else:
                # got mask from CutForInpaint
                height, width, _ = original[i].shape
                x0 = origin[i][0].item()
                y0 = origin[i][1].item()

                if mask[i].shape[0] < height or mask[i].shape[1] < width:
                    padded_mask = F.pad(input=mask[i], pad=(x0, width-x0-mask[i].shape[1], 
                                                            y0, height-y0-mask[i].shape[0]), mode='constant', value=0)
                else:
                    padded_mask = mask[i]
                blurred_mask = transform(padded_mask[None,None,:,:]).to(original.device).to(original.dtype)
                blurred.append(blurred_mask[0][0])

                result = F.pad(input=inpaint[i], pad=(0, 0, x0, width-x0-inpaint[i].shape[1], 
                                                      y0, height-y0-inpaint[i].shape[0]), mode='constant', value=0)
                result = result[None,:,:,:].to(original.device).to(original.dtype)

            ret.append(original[i] * (1.0 - blurred_mask[0][0][:,:,None]) + result[0] * blurred_mask[0][0][:,:,None])

        return (torch.stack(ret), torch.stack(blurred), )


class CutForInpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "image": ("IMAGE",),
                        "mask": ("MASK",),
                        "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                        "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("IMAGE","MASK","VECTOR",)
    RETURN_NAMES = ("image","mask","origin",)

    FUNCTION = "cut_for_inpaint"

    def cut_for_inpaint(self, image: torch.Tensor, mask: torch.Tensor, width: int, height: int):

        image, mask = check_image_mask(image, mask, 'BrushNet')

        ret = []
        msk = []
        org = []
        for i in range(image.shape[0]):
            x0, y0, w, h = cut_with_mask(mask[i], width, height)
            ret.append((image[i][y0:y0+h,x0:x0+w,:]))
            msk.append((mask[i][y0:y0+h,x0:x0+w]))
            org.append(torch.IntTensor([x0,y0]))

        return (torch.stack(ret), torch.stack(msk), torch.stack(org), )


#### Utility function

def get_files_with_extension(folder_name, extension=['.safetensors']):

    try:
        folders = folder_paths.get_folder_paths(folder_name)
    except:
        folders = []

    if not folders:
        folders = [os.path.join(folder_paths.models_dir, folder_name)]
    if not os.path.isdir(folders[0]):
        folders = [os.path.join(folder_paths.base_path, folder_name)]
    if not os.path.isdir(folders[0]):
        return {}
    
    filtered_folders = []
    for x in folders:
        if not os.path.isdir(x):
            continue
        the_same = False
        for y in filtered_folders:
            if os.path.samefile(x, y):
                the_same = True
                break
        if not the_same:
            filtered_folders.append(x)

    if not filtered_folders:
        return {}

    output = {}
    for x in filtered_folders:
        files, folders_all = folder_paths.recursive_search(x, excluded_dir_names=[".git"])
        filtered_files = folder_paths.filter_files_extensions(files, extension)

        for f in filtered_files:
            output[f] = x

    return output


# get blocks from state_dict so we could know which model it is
def brushnet_blocks(sd):
    brushnet_down_block = 0
    brushnet_mid_block = 0
    brushnet_up_block = 0
    for key in sd:
        if 'brushnet_down_block' in key:
            brushnet_down_block += 1
        if 'brushnet_mid_block' in key:
            brushnet_mid_block += 1        
        if 'brushnet_up_block' in key:
            brushnet_up_block += 1
    return (brushnet_down_block, brushnet_mid_block, brushnet_up_block, len(sd))


# Check models compatibility
def check_compatibility(model, brushnet):
    is_SDXL = False
    is_PP = False
    if isinstance(model.model.model_config, comfy.supported_models.SD15):
        print('Base model type: SD1.5')
        is_SDXL = False
        if brushnet["SDXL"]:
            raise Exception("Base model is SD15, but BrushNet is SDXL type")  
        if brushnet["PP"]:
            is_PP = True
    elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
        print('Base model type: SDXL')
        is_SDXL = True
        if not brushnet["SDXL"]:
            raise Exception("Base model is SDXL, but BrushNet is SD15 type")    
    else:
        print('Base model type: ', type(model.model.model_config))
        raise Exception("Unsupported model type: " + str(type(model.model.model_config)))

    return (is_SDXL, is_PP)


def check_image_mask(image, mask, name):
    if len(image.shape) < 4:
        # image tensor shape should be [B, H, W, C], but batch somehow is missing
        image = image[None,:,:,:]
    
    if len(mask.shape) > 3:
        # mask tensor shape should be [B, H, W] but we get [B, H, W, C], image may be?
        # take first mask, red channel
        mask = (mask[:,:,:,0])[:,:,:]
    elif len(mask.shape) < 3:
        # mask tensor shape should be [B, H, W] but batch somehow is missing
        mask = mask[None,:,:]

    if image.shape[0] > mask.shape[0]:
        print(name, "gets batch of images (%d) but only %d masks" % (image.shape[0], mask.shape[0]))
        if mask.shape[0] == 1: 
            print(name, "will copy the mask to fill batch")
            mask = torch.cat([mask] * image.shape[0], dim=0)
        else:
            print(name, "will add empty masks to fill batch")
            empty_mask = torch.zeros([image.shape[0] - mask.shape[0], mask.shape[1], mask.shape[2]])
            mask = torch.cat([mask, empty_mask], dim=0)
    elif image.shape[0] < mask.shape[0]:
        print(name, "gets batch of images (%d) but too many (%d) masks" % (image.shape[0], mask.shape[0]))
        mask = mask[:image.shape[0],:,:]

    return (image, mask)


# Prepare image and mask
def prepare_image(image, mask):

    image, mask = check_image_mask(image, mask, 'BrushNet')

    print("BrushNet image.shape =", image.shape, "mask.shape =", mask.shape)

    if mask.shape[2] != image.shape[2] or mask.shape[1] != image.shape[1]:
        raise Exception("Image and mask should be the same size")
    
    # As a suggestion of inferno46n2 (https://github.com/nullquant/ComfyUI-BrushNet/issues/64)
    mask = mask.round()

    masked_image = image * (1.0 - mask[:,:,:,None])

    return (masked_image, mask)


# Get origin of the mask
def cut_with_mask(mask, width, height):
    iy, ix = (mask == 1).nonzero(as_tuple=True)

    h0, w0 = mask.shape
    
    if iy.numel() == 0:
        x_c = w0 / 2.0
        y_c = h0 / 2.0
    else:
        x_min = ix.min().item()
        x_max = ix.max().item()
        y_min = iy.min().item()
        y_max = iy.max().item()

        if x_max - x_min > width or y_max - y_min > height:
            raise Exception("Masked area is bigger than provided dimensions")

        x_c = (x_min + x_max) / 2.0
        y_c = (y_min + y_max) / 2.0
    
    width2 = width / 2.0
    height2 = height / 2.0

    if w0 <= width:
        x0 = 0
        w = w0
    else:
        x0 = max(0, x_c - width2)
        w = width
        if x0 + width > w0:
            x0 = w0 - width

    if h0 <= height:
        y0 = 0
        h = h0
    else:
        y0 = max(0, y_c - height2)
        h = height
        if y0 + height > h0:
            y0 = h0 - height

    return (int(x0), int(y0), int(w), int(h))


# Prepare conditioning_latents
@torch.inference_mode()
def get_image_latents(masked_image, mask, vae, scaling_factor):
    processed_image = masked_image.to(vae.device)
    image_latents = vae.encode(processed_image[:,:,:,:3]) * scaling_factor
    processed_mask = 1. - mask[:,None,:,:]
    interpolated_mask = torch.nn.functional.interpolate(
                processed_mask, 
                size=(
                    image_latents.shape[-2], 
                    image_latents.shape[-1]
                )
            )
    interpolated_mask = interpolated_mask.to(image_latents.device)

    conditioning_latents = [image_latents, interpolated_mask]

    print('BrushNet CL: image_latents shape =', image_latents.shape, 'interpolated_mask shape =', interpolated_mask.shape)

    return conditioning_latents


# Main function where magic happens
@torch.inference_mode()
def brushnet_inference(x, timesteps, transformer_options, debug):
    if 'model_patch' not in transformer_options:
        print('BrushNet inference: there is no model_patch key in transformer_options')
        return ([], 0, [])
    mp = transformer_options['model_patch']
    if 'brushnet' not in mp:
        print('BrushNet inference: there is no brushnet key in mdel_patch')
        return ([], 0, [])
    bo = mp['brushnet']
    if 'model' not in bo:
        print('BrushNet inference: there is no model key in brushnet')
        return ([], 0, [])
    brushnet = bo['model']
    if not (isinstance(brushnet, BrushNetModel) or isinstance(brushnet, PowerPaintModel)):
        print('BrushNet model is not a BrushNetModel class')
        return ([], 0, [])

    torch_dtype = bo['dtype']
    cl_list = bo['latents']
    brushnet_conditioning_scale, control_guidance_start, control_guidance_end = bo['controls']
    pe = bo['prompt_embeds']
    npe = bo['negative_prompt_embeds']
    ppe, nppe, time_ids = bo['add_embeds']

    #do_classifier_free_guidance = mp['free_guidance']
    do_classifier_free_guidance = len(transformer_options['cond_or_uncond']) > 1

    x = x.detach().clone()
    x = x.to(torch_dtype).to(brushnet.device)

    timesteps = timesteps.detach().clone()
    timesteps = timesteps.to(torch_dtype).to(brushnet.device)

    total_steps = mp['total_steps']
    step = mp['step']

    added_cond_kwargs = {}

    if do_classifier_free_guidance and step == 0:
        print('BrushNet inference: do_classifier_free_guidance is True')

    sub_idx = None
    if 'ad_params' in transformer_options and 'sub_idxs' in transformer_options['ad_params']:
        sub_idx = transformer_options['ad_params']['sub_idxs']

    # we have batch input images
    batch = cl_list[0].shape[0]
    # we have incoming latents
    latents_incoming = x.shape[0]
    # and we already got some
    latents_got = bo['latent_id']
    if step == 0 or batch > 1:
        print('BrushNet inference, step = %d: image batch = %d, got %d latents, starting from %d' \
                % (step, batch, latents_incoming, latents_got))

    image_latents = []
    masks = []
    prompt_embeds = []
    negative_prompt_embeds = []
    pooled_prompt_embeds = []
    negative_pooled_prompt_embeds = []
    if sub_idx:
        # AnimateDiff indexes detected
        if step == 0:
            print('BrushNet inference: AnimateDiff indexes detected and applied')

        batch = len(sub_idx)

        if do_classifier_free_guidance:
            for i in sub_idx:
                image_latents.append(cl_list[0][i][None,:,:,:])
                masks.append(cl_list[1][i][None,:,:,:])
                prompt_embeds.append(pe)
                negative_prompt_embeds.append(npe)
                pooled_prompt_embeds.append(ppe)
                negative_pooled_prompt_embeds.append(nppe)
            for i in sub_idx:
                image_latents.append(cl_list[0][i][None,:,:,:])
                masks.append(cl_list[1][i][None,:,:,:])
        else:
            for i in sub_idx:
                image_latents.append(cl_list[0][i][None,:,:,:])
                masks.append(cl_list[1][i][None,:,:,:])
                prompt_embeds.append(pe)
                pooled_prompt_embeds.append(ppe)
    else:
        # do_classifier_free_guidance = 2 passes, 1st pass is cond, 2nd is uncond
        continue_batch = True
        for i in range(latents_incoming):
            number = latents_got + i
            if number < batch:
                # 1st pass, cond
                image_latents.append(cl_list[0][number][None,:,:,:])
                masks.append(cl_list[1][number][None,:,:,:])
                prompt_embeds.append(pe)
                pooled_prompt_embeds.append(ppe)
            elif do_classifier_free_guidance and number < batch * 2:
                # 2nd pass, uncond
                image_latents.append(cl_list[0][number-batch][None,:,:,:])
                masks.append(cl_list[1][number-batch][None,:,:,:])
                negative_prompt_embeds.append(npe)
                negative_pooled_prompt_embeds.append(nppe)
            else:
                # latent batch
                image_latents.append(cl_list[0][0][None,:,:,:])
                masks.append(cl_list[1][0][None,:,:,:])
                prompt_embeds.append(pe)
                pooled_prompt_embeds.append(ppe)
                latents_got = -i
                continue_batch = False

        if continue_batch:
            # we don't have full batch yet
            if do_classifier_free_guidance:
                if number < batch * 2 - 1:
                    bo['latent_id'] = number + 1
                else:
                    bo['latent_id'] = 0
            else:
                if number < batch - 1:
                    bo['latent_id'] = number + 1
                else:
                    bo['latent_id'] = 0
        else:
            bo['latent_id'] = 0

    cl = []
    for il, m in zip(image_latents, masks):
        cl.append(torch.concat([il, m], dim=1))
    cl2apply = torch.concat(cl, dim=0)

    conditioning_latents = cl2apply.to(torch_dtype).to(brushnet.device)

    # print("BrushNet CL: conditioning_latents shape =", conditioning_latents.shape)
    # print("BrushNet CL: x shape =", x.shape)

    prompt_embeds.extend(negative_prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds, dim=0).to(torch_dtype).to(brushnet.device)

    if ppe is not None:
        added_cond_kwargs = {}
        added_cond_kwargs['time_ids'] = torch.concat([time_ids] * latents_incoming, dim = 0).to(torch_dtype).to(brushnet.device)

        pooled_prompt_embeds.extend(negative_pooled_prompt_embeds)
        pooled_prompt_embeds = torch.concat(pooled_prompt_embeds, dim=0).to(torch_dtype).to(brushnet.device)
        added_cond_kwargs['text_embeds'] = pooled_prompt_embeds
    else:
        added_cond_kwargs = None

    if x.shape[2] != conditioning_latents.shape[2] or x.shape[3] != conditioning_latents.shape[3]:
        if step == 0:
            print('BrushNet inference: image', conditioning_latents.shape, 'and latent', x.shape, 'have different size, resizing image')
        conditioning_latents = torch.nn.functional.interpolate(
            conditioning_latents, size=(
                x.shape[2], 
                x.shape[3],
            ), mode='bicubic',
        ).to(torch_dtype).to(brushnet.device)

    if step == 0:
        print('BrushNet inference: sample', x.shape, ', CL', conditioning_latents.shape, 'dtype', torch_dtype)

    if debug: print('BrushNet: step =', step)

    if step < control_guidance_start or step > control_guidance_end:
        cond_scale = 0.0
    else:
        cond_scale = brushnet_conditioning_scale

    return brushnet(x,
                    encoder_hidden_states=prompt_embeds,
                    brushnet_cond=conditioning_latents,
                    timestep = timesteps,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    debug=debug,
                )


# This is main patch function
def add_brushnet_patch(model, brushnet, torch_dtype, conditioning_latents, 
                       controls, 
                       prompt_embeds, negative_prompt_embeds,
                       pooled_prompt_embeds, negative_pooled_prompt_embeds, time_ids,
                       debug):
    
    is_SDXL = isinstance(model.model.model_config, comfy.supported_models.SDXL)

    if is_SDXL:
        input_blocks = [[0, comfy.ops.disable_weight_init.Conv2d],
                        [1, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [2, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [3, comfy.ldm.modules.diffusionmodules.openaimodel.Downsample],
                        [4, comfy.ldm.modules.attention.SpatialTransformer],
                        [5, comfy.ldm.modules.attention.SpatialTransformer],
                        [6, comfy.ldm.modules.diffusionmodules.openaimodel.Downsample],
                        [7, comfy.ldm.modules.attention.SpatialTransformer],
                        [8, comfy.ldm.modules.attention.SpatialTransformer]]
        middle_block  = [0, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock]
        output_blocks = [[0, comfy.ldm.modules.attention.SpatialTransformer],
                        [1, comfy.ldm.modules.attention.SpatialTransformer],
                        [2, comfy.ldm.modules.attention.SpatialTransformer],
                        [2, comfy.ldm.modules.diffusionmodules.openaimodel.Upsample],
                        [3, comfy.ldm.modules.attention.SpatialTransformer],
                        [4, comfy.ldm.modules.attention.SpatialTransformer],
                        [5, comfy.ldm.modules.attention.SpatialTransformer],
                        [5, comfy.ldm.modules.diffusionmodules.openaimodel.Upsample],
                        [6, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [7, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [8, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock]]
    else:
        input_blocks = [[0, comfy.ops.disable_weight_init.Conv2d],
                        [1, comfy.ldm.modules.attention.SpatialTransformer],
                        [2, comfy.ldm.modules.attention.SpatialTransformer],
                        [3, comfy.ldm.modules.diffusionmodules.openaimodel.Downsample],
                        [4, comfy.ldm.modules.attention.SpatialTransformer],
                        [5, comfy.ldm.modules.attention.SpatialTransformer],
                        [6, comfy.ldm.modules.diffusionmodules.openaimodel.Downsample],
                        [7, comfy.ldm.modules.attention.SpatialTransformer],
                        [8, comfy.ldm.modules.attention.SpatialTransformer],
                        [9, comfy.ldm.modules.diffusionmodules.openaimodel.Downsample],
                        [10, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [11, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock]]
        middle_block  = [0, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock]
        output_blocks = [[0, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [1, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [2, comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock],
                        [2, comfy.ldm.modules.diffusionmodules.openaimodel.Upsample],
                        [3, comfy.ldm.modules.attention.SpatialTransformer],
                        [4, comfy.ldm.modules.attention.SpatialTransformer],
                        [5, comfy.ldm.modules.attention.SpatialTransformer],
                        [5, comfy.ldm.modules.diffusionmodules.openaimodel.Upsample],
                        [6, comfy.ldm.modules.attention.SpatialTransformer],
                        [7, comfy.ldm.modules.attention.SpatialTransformer],
                        [8, comfy.ldm.modules.attention.SpatialTransformer],
                        [8, comfy.ldm.modules.diffusionmodules.openaimodel.Upsample],
                        [9, comfy.ldm.modules.attention.SpatialTransformer],
                        [10, comfy.ldm.modules.attention.SpatialTransformer],
                        [11, comfy.ldm.modules.attention.SpatialTransformer]]

    def last_layer_index(block, tp):
        layer_list = []
        for layer in block:
            layer_list.append(type(layer))
        layer_list.reverse()
        if tp not in layer_list:
            return -1, layer_list.reverse()
        return len(layer_list) - 1 - layer_list.index(tp), layer_list

    def brushnet_forward(model, x, timesteps, transformer_options, control):
        if 'brushnet' not in transformer_options['model_patch']:
            input_samples = []
            mid_sample = 0
            output_samples = []
        else:    
            # brushnet inference
            input_samples, mid_sample, output_samples = brushnet_inference(x, timesteps, transformer_options, debug)

        # give additional samples to blocks
        for i, tp in input_blocks:
            idx, layer_list = last_layer_index(model.input_blocks[i], tp)
            if idx < 0:
                print("BrushNet can't find", tp, "layer in", i,"input block:", layer_list)
                continue
            model.input_blocks[i][idx].add_sample_after = input_samples.pop(0) if input_samples else 0

        idx, layer_list = last_layer_index(model.middle_block, middle_block[1])
        if idx < 0:
            print("BrushNet can't find", middle_block[1], "layer in middle block", layer_list)
        model.middle_block[idx].add_sample_after = mid_sample

        for i, tp in output_blocks:
            idx, layer_list = last_layer_index(model.output_blocks[i], tp)
            if idx < 0:
                print("BrushNet can't find", tp, "layer in", i,"outnput block:", layer_list)
                continue
            model.output_blocks[i][idx].add_sample_after = output_samples.pop(0) if output_samples else 0

    patch_model_function_wrapper(model, brushnet_forward)

    to = add_model_patch_option(model)
    mp = to['model_patch']
    if 'brushnet' not in mp:
        mp['brushnet'] = {}
    bo = mp['brushnet']

    bo['model'] = brushnet
    bo['dtype'] = torch_dtype
    bo['latents'] = conditioning_latents
    bo['controls'] = controls
    bo['prompt_embeds'] = prompt_embeds
    bo['negative_prompt_embeds'] = negative_prompt_embeds
    bo['add_embeds'] = (pooled_prompt_embeds, negative_pooled_prompt_embeds, time_ids)
    bo['latent_id'] = 0

    # patch layers `forward` so we can apply brushnet
    def forward_patched_by_brushnet(self, x, *args, **kwargs):
        h = self.original_forward(x, *args, **kwargs)
        if hasattr(self, 'add_sample_after') and type(self):
            to_add = self.add_sample_after
            if torch.is_tensor(to_add):
                # interpolate due to RAUNet
                if h.shape[2] != to_add.shape[2] or h.shape[3] != to_add.shape[3]:
                    to_add = torch.nn.functional.interpolate(to_add, size=(h.shape[2], h.shape[3]), mode='bicubic')                  
                h += to_add.to(h.dtype).to(h.device)
            else:
                h += self.add_sample_after
            self.add_sample_after = 0
        return h

    for i, block in enumerate(model.model.diffusion_model.input_blocks):
        for j, layer in enumerate(block):
            if not hasattr(layer, 'original_forward'):
                layer.original_forward = layer.forward
            layer.forward = types.MethodType(forward_patched_by_brushnet, layer)
            layer.add_sample_after = 0

    for j, layer in enumerate(model.model.diffusion_model.middle_block):
        if not hasattr(layer, 'original_forward'):
            layer.original_forward = layer.forward
        layer.forward = types.MethodType(forward_patched_by_brushnet, layer)
        layer.add_sample_after = 0

    for i, block in enumerate(model.model.diffusion_model.output_blocks):
        for j, layer in enumerate(block):
            if not hasattr(layer, 'original_forward'):
                layer.original_forward = layer.forward
            layer.forward = types.MethodType(forward_patched_by_brushnet, layer)
            layer.add_sample_after = 0
