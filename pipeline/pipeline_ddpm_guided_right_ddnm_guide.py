# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union, Callable

import torch
import sys

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# Code form RePaint   
def get_repeat_timesteps(t_T, stride_num, sample_num):
    strides = {}
    for j in range(0, t_T - stride_num, stride_num):
        strides[j] = sample_num - 1
    t = t_T
    ts = []
    while t >= 1:
        t = t - 1
        ts.append(t)
        if (strides.get(t, 0) > 0):
            strides[t] = strides[t] - 1
            for _ in range(stride_num):
                t = t + 1
                ts.append(t)
    ts.append(-1)
    print(f"{len(ts)=}")
    # print(f"{ts=}")

    _check_times(ts, -1, t_T)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0).cuda()
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a



class DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    #@torch.no_grad()
    def __call__(
        self,
        measure_func: Callable[[torch.FloatTensor, list], torch.FloatTensor],
        observation: list = [],
        constraint: list = [],
        cloth_related: list = [],
        guide_scale: float = 1.0,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        start_t: int = 0,
        use_temporal: bool = False,
        start_image: Optional[torch.FloatTensor] = None,
        save_middle_result: bool = False,
        is_ddim: bool = False,
        use_guidance: bool = False,
        use_constraint: bool = False,
        repeat_t: int = 100, 
        stride_num: int = 1, 
        sample_num: int = 1,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if start_image is not None:
            image = start_image # [-1, 1]
            # image = (start_image - 0.5) * 2
            # print(f"{image.max()=} {image.min()=} {image.shape=}")
            
            # alpha_bar_t = self.scheduler.alphas_cumprod[start_t]  # alpha_bar_t 是噪声调度器中的参数
            # beta_bar_t = 1 - alpha_bar_t # self.scheduler.betas[start_t]    # beta_bar_t 是噪声调度器中的参数
            t = torch.full((image.shape[0],), start_t, device=image.device).cuda()
            alpha_bar_t = compute_alpha(self.scheduler.betas, t.long())
            beta_bar_t = 1 - alpha_bar_t
            print(f"{alpha_bar_t=} {beta_bar_t=}")

            # 生成与图像形状相同的随机噪声(gaussian)
            noise = torch.randn_like(image)

            # 根据前向过程公式添加噪声
            noisy_image = torch.sqrt(alpha_bar_t) * image + torch.sqrt(beta_bar_t) * noise

            # 将 noisy_image 作为初始值
            image = noisy_image
            
        elif self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        if use_temporal:
            self.scheduler.timesteps = self.scheduler.timesteps[-start_t-1:]
        # print(f"{self.scheduler.timesteps=} {num_inference_steps}")
        # sys.exit()

        y, A, Ap, sigma_y = constraint
        # skip = len(self.scheduler.timesteps) // T_sampling  # 计算步长
        n = image.size(0)
        x0_preds = []
        # xs = [image]
        xt = image

        repeat_timesteps = get_repeat_timesteps(repeat_t, stride_num, sample_num)
        total_timesteps = torch.cat([
            self.scheduler.timesteps[:-repeat_t], 
            torch.tensor(repeat_timesteps, device=self.scheduler.timesteps.device)
        ])
            
        time_pairs = list(zip(total_timesteps[:-1], total_timesteps[1:]))
        # print(f"{len(self.scheduler.timesteps)=}")
        # print(f"{len(total_timesteps)=}")
        # print(f"{len(self.scheduler.betas)=}")
        # print(f"{self.scheduler.timesteps[-105:-90]=}  ")
        # sys.exit()
        for i, j in self.progress_bar(time_pairs):
            # i, j = i * skip, j * skip
            # if j < 0: j = -1 

            if j < i:  # Normal sampling
                
                t = torch.full((n,), i, device=image.device).cuda()
                next_t = torch.full((n,), j, device=image.device).cuda()
                at = compute_alpha(self.scheduler.betas, t.long())
                at_next = compute_alpha(self.scheduler.betas, next_t.long())
                # t = i
                # at = self.scheduler.alphas_cumprod[i]
                # at_next = self.scheduler.alphas_cumprod[j]
                
                # print(f"{at=} {at_next=}")
                sigma_t = (1 - at_next**2).sqrt()
                # xt = xs[-1].to('cuda')

                model_output = self.unet(xt, t).sample
                # print(f"{model_output.shape=}") # model_output.shape=torch.Size([1, 4, 128, 256])
                # sys.exit()
                if use_guidance:
                    xt.requires_grad = True
                    # 2. compute previous image: x_t -> x_t-1
                    # schedulerOutput = self.scheduler.step(model_output, t, xt, generator=generator)
                    # prev_sample = schedulerOutput.prev_sample
                    # pred_original_sample = schedulerOutput.pred_original_sample

                    # guidance_loss = measure_func(pred_original_sample, observation, t=t)
                    # guide_grad = torch.autograd.grad(outputs=guidance_loss, inputs=image)[0]
                    # image = prev_sample - guide_grad * guide_scale

                    # Eq. 12
                    x0_t = (xt - model_output * (1 - at).sqrt()) / at.sqrt()
                    
                    guidance_loss = measure_func(x0_t, observation, t=t)
                    if guidance_loss != 0:
                        guide_grad = torch.autograd.grad(outputs=guidance_loss, inputs=xt)[0]
                    
                    # Eq. 19
                    if sigma_t >= at_next * sigma_y:
                        lambda_t = 1.
                        gamma_t = (sigma_t**2 - (at_next * sigma_y)**2).sqrt()
                    else:
                        lambda_t = sigma_t / (at_next * sigma_y)
                        gamma_t = 0.

                    x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)

                    eta = 0.85 # self.args.eta
                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                    # Different from the paper, we use DDIM here instead of DDPM
                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * model_output)
                    
                    if guidance_loss != 0:
                        xt_next = xt_next - guide_grad * guide_scale
                        
                    xt = xt_next
                    
                else:
                    # Eq. 12
                    x0_t = (xt - model_output * (1 - at).sqrt()) / at.sqrt()

                    # Eq. 19
                    if sigma_t >= at_next * sigma_y:
                        lambda_t = 1.
                        gamma_t = (sigma_t**2 - (at_next * sigma_y)**2).sqrt()
                    else:
                        lambda_t = sigma_t / (at_next * sigma_y)
                        gamma_t = 0.

                    x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)

                    eta = 0.85 # self.args.eta
                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                    # Different from the paper, we use DDIM here instead of DDPM
                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * model_output)
                    
                    xt = xt_next

                x0_preds = x0_t #.append(x0_t.to('cpu'))
                # xs.append(xt_next.to('cpu'))
                xt = xt.detach()

            else:  # Time-travel back
                next_t = torch.full((n,), j, device=image.device)
                at_next = compute_alpha(self.scheduler.betas, next_t.long())
                # at_next = self.scheduler.alphas_cumprod[j]
                x0_t = x0_preds.to(image.device)

                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
                # xs.append(xt_next.to('cpu'))

        image = xt.detach() # xs[-1].detach()

            # image = image.detach()

        # image = schedulerOutput.pred_original_sample.detach()
        #image = (image).clamp(-1, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
