from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import copy
from torch.nn import functional as F
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import PIL
from diffusers import DDIMScheduler, DDPMScheduler, DDIMInverseScheduler
from diffusers import image_processor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
logger = logging.get_logger(__name__)
from PIL import ImageChops
import time

class COWPipeline(StableDiffusionPipeline):

    _optional_components = ["safety_checker", "feature_extractor", "inverse_scheduler"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: DDIMScheduler,
            inverse_scheduler: DDIMInverseScheduler,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
    ):

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            inverse_scheduler=inverse_scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = image_processor.VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            image: Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ] = None,
            mask_image: Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ] = None,
            seed_size: int =256,
            x_offset: int =128,
            y_offset: int =0,
            
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
   

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        self.inverse_scheduler_copy = copy.deepcopy(self.inverse_scheduler)

        # 3. Preprocess image
        mask_image = mask_image.convert('1')
        inverted_mask_image = ImageChops.invert(mask_image)
        image = ImageChops.composite(image, Image.new('RGB', image.size, (128, 128, 128)), inverted_mask_image)
        self.x_offset = x_offset
        self.y_offset = y_offset
        background_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        background_image.paste(image, (x_offset, y_offset))
        mask = transforms.ToTensor()(inverted_mask_image)
        latent_mask = mask.unsqueeze(0)[:,:1,:,:].repeat(batch_size,4,1,1)
        latent_mask = torch.nn.functional.interpolate(latent_mask, size=(seed_size//8, seed_size//8), mode='nearest')
        latent_mask[latent_mask != 1] = 0
        has_other_values = ((latent_mask != 0) & (latent_mask != 1)).any()
        mask_numpy = latent_mask[:, 0, :, :].squeeze().numpy()
        mask_numpy = (mask_numpy * 255).astype('uint8')
        mask_image = Image.fromarray(mask_numpy, mode='L')
        latent_mask = latent_mask.to(device)
        sample_mask = torch.zeros((1,4,64,64))
        sample_mask[:,:, y_offset//8:y_offset//8+seed_size//8, x_offset//8:x_offset//8+seed_size//8] = latent_mask

        face_image = self.image_processor.preprocess(image)
        BG_image = self.image_processor.preprocess(background_image)


        # 4. Prepare latent variables
        face_latents = self.prepare_image_latents(face_image, batch_size, self.vae.dtype, device, generator)
        bg_latents = self.prepare_image_latents(BG_image, batch_size, self.vae.dtype, device, generator)
        replace_step = int(0.5 * num_inference_steps)
        back_step = int(0.7 * num_inference_steps)

        #face inverse: x0 -> replace_step, and save
        self.inverse_scheduler = copy.deepcopy(self.inverse_scheduler_copy)

        face_in_replace_step, face_x0_to_replace = self.invert(prompt=prompt,guidance_scale=7.5,latents=face_latents,start_step=0,end_step=replace_step,save_latens=True, num_inference_steps=num_inference_steps)
        #face inverse: replace_step -> back_step, and save
        face_in_back_step, face_replace_to_back = self.invert(prompt=prompt,guidance_scale=7.5,latents=face_in_replace_step,start_step=replace_step,end_step=back_step,save_latens=True, num_inference_steps=num_inference_steps)
        # reverse list
        face_replace_to_back.reverse()
        face_x0_to_replace.reverse()
        face_back_to_replace = face_replace_to_back
        face_replace_to_x0 = face_x0_to_replace

        # bg_xt inverse: x0 -> xT(end step)
        self.inverse_scheduler = copy.deepcopy(self.inverse_scheduler_copy)
        bg_in_replace_step = self.invert(prompt=prompt, guidance_scale=7.5,latents=bg_latents,start_step=0,end_step=replace_step,num_inference_steps=num_inference_steps)

        
        #begin cycle
        bg_tmp = bg_in_replace_step
        back_to_replace = [i for i in range(back_step, replace_step-1, -1)]
    
        replace_to_x0 = [int(0.4 * num_inference_steps)]

        total_time = 0
        for cycle_id in range(10):
            start_time = time.time()
          
            bg_tmp = self.inject_noise(x_t=bg_tmp, start_step=replace_step,end_step=back_step)
            # back -> replace, and replace in between run
            bg_tmp = self.sample_and_replace_with_mask(x=bg_tmp, last_step=back_step, end_step=replace_step, back_step=back_step,
                                         replace_step_list=back_to_replace, eta=0., prompt=prompt, generator=generator,
                                         face_back_to_front=face_back_to_replace, latent_mask=latent_mask,sample_mask=sample_mask, num_inference_steps=num_inference_steps)
            end_time = time.time()
            cyc_time = end_time-start_time
            total_time += cyc_time
        average_time_per_cycle = total_time / 10

            

        # save last cycle, replace -> x0 and replace in between
        x0 = bg_tmp.clone()
        x0 = self.sample_and_replace(x=x0, last_step=replace_step, end_step=0, back_step=replace_step,
                                               replace_step_list=replace_to_x0, eta=0.1, prompt=prompt,
                                               generator=generator,
                                               face_back_to_front=face_replace_to_x0, latent_mask=latent_mask, num_inference_steps=num_inference_steps)

        if not output_type == "latent":
            image = self.vae.decode(x0 / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        return image

 

    @torch.no_grad()
    def invert(
            self,
            prompt: Optional[str] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 0.,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "latent",
            prompt_embeds: Optional[torch.FloatTensor] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            start_step:int = 0,
            end_step:int = 50,
            save_latens : bool =False,
    ):

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        latents = latents
        

        replace_to_back = []
        if save_latens:
            replace_to_back.append(latents.to('cpu'))

        num_images_per_prompt = 1
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )

        # Prepare timesteps
        timesteps = self.inverse_scheduler.timesteps
        # Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                index = i
                if index < start_step:
                    continue
                if index >= end_step:
                    break
       
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

                if save_latens:
                    replace_to_back.append(latents.to('cpu'))

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        inverted_latents = latents.detach().clone()


        if save_latens:
            print(f"save intermediate inversion from_{start_step}_to_{end_step},total:{len(replace_to_back)}")
            return inverted_latents,replace_to_back
        else:
            return inverted_latents


    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents

    def replace_with_seed(self, bg_noise, face_noise, latent_mask, x_offset, y_offset):
        seed_size_latent = face_noise.shape[-1]
        face_noise = face_noise.to(self.device)
        bg_noise[:,:,y_offset:y_offset+seed_size_latent,x_offset:x_offset+seed_size_latent][latent_mask==1] = face_noise[latent_mask==1]
        return bg_noise

    def sample_and_replace(self, x, last_step, end_step, back_step, replace_step_list, eta, latent_mask, prompt,
                           generator, face_back_to_front,num_inference_steps):
        for k in replace_step_list:  
            x = self.sample(prompt=prompt, eta=eta, generator=generator, latents=x, start_step=last_step, end_step=k, num_inference_steps=num_inference_steps)
            last_step = k
            replace_index = back_step - k  
            x = self.replace_with_seed(bg_noise=x, face_noise=face_back_to_front[replace_index],
                                       latent_mask=latent_mask, x_offset=self.x_offset // 8, y_offset=self.y_offset//8)
        x = self.sample(prompt=prompt, eta=eta, generator=generator, latents=x, start_step=last_step, end_step=end_step, num_inference_steps=num_inference_steps)
        return x

    def sample_and_replace_with_mask(self, x, last_step, end_step, back_step, replace_step_list, eta, latent_mask, prompt,
                           generator, face_back_to_front, sample_mask,num_inference_steps):
        for k in replace_step_list:  
            x = self.sample_with_mask(prompt=prompt, eta=eta, generator=generator, latents=x, start_step=last_step, end_step=k, sample_mask=sample_mask, num_inference_steps=num_inference_steps)
            last_step = k
            replace_index = back_step - k  
            x = self.replace_with_seed(bg_noise=x, face_noise=face_back_to_front[replace_index],
                                       latent_mask=latent_mask, x_offset= self.x_offset // 8, y_offset=self.y_offset//8)
        x = self.sample_with_mask(prompt=prompt, eta=eta, generator=generator, latents=x, start_step=last_step, end_step=end_step,sample_mask=sample_mask, num_inference_steps=num_inference_steps)
        return x

    @torch.no_grad()
    def inject_noise(self,
                     x_t: torch.Tensor,
                     end_step: int,
                     start_step: int,
                     ):  

        a_t = self.scheduler.alphas_cumprod[self.inverse_scheduler.timesteps[start_step]]
        a_tm = self.scheduler.alphas_cumprod[self.inverse_scheduler.timesteps[end_step]]
        noise = torch.randn_like(x_t)
        x_tm = ((a_tm / a_t) ** 0.5) * x_t + ((1 - (a_tm / a_t)) ** 0.5) * noise
        return x_tm

    @torch.no_grad()
    def sample(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "latent",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            start_step: int = 50,
            end_step: int = 0,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                index = len(timesteps) - i - 1  
                if index >= start_step:
                    continue
                if index < end_step:  
                    break

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        return image



    @torch.no_grad()
    def sample_with_mask(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "latent",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            start_step: int = 50,
            end_step: int = 0,
            sample_mask : torch.Tensor = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt

        neg_prompt = "a bad quality and low resolution image,extra fingers,deformed hands"
        diverse_prompt = "a high quality and colorful image"
        suppress_prompt = "human face"

        prompt_neg_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=neg_prompt,
        )
        div_sup_embeds = self._encode_prompt(
            diverse_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=suppress_prompt,
        )


        # 4. Prepare timesteps

        timesteps = self.scheduler.timesteps


        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_neg_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
   
        for i, t in enumerate(timesteps):

            index = len(timesteps) - i - 1  # 49
            if index >= start_step:
                continue
            if index < end_step:
                break
            latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents

            prompt_embeds = torch.cat([prompt_neg_embeds, div_sup_embeds])

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_neg, noise_pred_text, noise_pred_sup, noise_pred_div = noise_pred.chunk(4)

                sample_mask = sample_mask.to(device)

                all_one_mask = torch.ones_like(sample_mask).to(device)
                background_mask = all_one_mask - sample_mask

                noise_pred_total_neg = sample_mask * noise_pred_neg + background_mask * (noise_pred_neg + noise_pred_sup - noise_pred_div)
                noise_pred = noise_pred_total_neg + guidance_scale * (noise_pred_text - noise_pred_total_neg)


            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)



        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        return image
