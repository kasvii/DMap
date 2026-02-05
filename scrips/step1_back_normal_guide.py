import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import diffusers
from diffusers import DDPMScheduler, UNet2DModel
import argparse

sys.path.append('../temporal_diffusion')
from pipeline_ddpm_condition_seq_guide import DDPMPipeline
from models.unet import UNet3DConditionModel

sys.path.append('..')
from utils.rasterize import get_pix_to_face
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    OrthographicCameras,
    FoVOrthographicCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    blending,
)

np.random.seed(62)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


from typing import NamedTuple, Sequence
class BlendParams_blackBG(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (0.0, 0.0, 0.0)


class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images

def get_render(is_back=False):
    render_res = 256
    dis = 100.0
    scale = 100
    mesh_y_center = 0.0
    cam_pos = torch.tensor([
                    (0, mesh_y_center, dis),
                    (0, mesh_y_center, -dis),
                ])
    R, T = look_at_view_transform(
        eye=cam_pos[[0]] if not is_back else cam_pos[[1]],
        at=((0, mesh_y_center, 0), ),
        up=((0, 1, 0), ),
    )

    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=100.0,
        zfar=-100.0,
        max_y=100.0,
        min_y=-100.0,
        max_x=100.0,
        min_x=-100.0,
        scale_xyz=(scale * np.ones(3), ) * len(R),
    )

    sigma = 1e-7

    raster_settings_hard = RasterizationSettings(
        image_size=render_res, 
        blur_radius=np.log(1. / 1e-4)*sigma, 
        faces_per_pixel=1, 
        max_faces_per_bin=500000,
        perspective_correct=False,
    )

    meshRas_hard = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_hard)

    renderer_textured_hard = MeshRenderer(
        rasterizer=meshRas_hard,
        shader=cleanShader(blend_params=BlendParams_blackBG())
    )

    return meshRas_hard, renderer_textured_hard

def face_to_uv_coord(pix_to_face):
    x, y = np.where(pix_to_face != -1)
    coord_img = np.stack((x,y), axis=-1)

    faces = pix_to_face[x, y]

    return coord_img, faces

def render_segmentation(body, renderer_textured_hard, raster):

    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()

    pix_to_face = get_pix_to_face(verts, faces, raster)
    pix_to_face = pix_to_face.detach().cpu().numpy()
    coord_img, coord_faces = face_to_uv_coord(pix_to_face)

    body_seg = np.zeros((len(pix_to_face), len(pix_to_face), 3))
    body_seg[coord_img[:,0], coord_img[:,1]] = color_smpl[coord_faces]
    body_seg = np.round(body_seg*255).astype(np.uint8)

    return body_seg

def render_depth(body, renderer_textured_hard):
    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()

    depth = body.vertices[:, [-1]]
    depth = np.concatenate((depth,depth,depth), axis=-1)
    depth = torch.FloatTensor(depth).cuda()

    textures_depth = TexturesVertex(verts_features=depth[None])
    mesh_depth = Meshes(
        verts=[verts],   
        faces=[faces],
        textures=textures_depth
    )
    
    images_depth = renderer_textured_hard(mesh_depth)
    mask_depth = images_depth[0, :, :, -1].detach().cpu().numpy() <= 0
    images_depth = images_depth[0, :, :, :1].detach().cpu().numpy()
    images_depth[mask_depth] = -1
    return images_depth

def _process_depth(image, resize=True):
    if resize:
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_NEAREST)
    if len(image.shape) != 3:
        image = np.expand_dims(image, axis=-1)
    return image

def _process_seg(image, resize=True):
    if resize:
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_NEAREST)
    image = image.astype(np.float32)/255
    image = (image - 0.5)/0.5
    return image

def _process_image(image, resize=True):
    if resize:
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_NEAREST)
    image_raw = image.copy()
    image = image.astype(np.float32)/255
    image = (image - 0.5)/0.5
    return image, image_raw

def _process_prev_image(image):
    image = (image - 0.5)/0.5
    return image


parser = argparse.ArgumentParser(description="Generate the back normal maps")
parser.add_argument("--garment", type=str, default='Skirt', help="The type of garment")
parser.add_argument("--scale", type=float, default=0.8, help="The scale of the garment")
parser.add_argument("--vid_name", type=str, default='vid_demo', help="The name of the video")
args = parser.parse_args()

garment = args.garment

scale = args.scale
vid_name = args.vid_name

target_label = 240
pretrained_unet_path = '../checkpoints/back_normal_guide/unet_ema'

unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_unet_path, 
        subfolder="unet", 
)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    prediction_type="epsilon", 
    beta_schedule="linear"
)
pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler).to("cuda")

normal_dir = f'../data/{vid_name}/results-{scale}'
body_dir = f"../data/{vid_name}/cropped_body"
save_folder = f'../fitting-results/{vid_name}/uv-mapping-back-{scale}-{garment}'

images_list = sorted(list(set([i.split('_')[0] for i in sorted(os.listdir(normal_dir))])))
    
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def _measure_func(model_output, observation, t, eps=0, log_file=os.path.join(save_folder, "a_loss_back_normal.txt")):
        
    prev_output = observation[0]
    if prev_output is not None:
        prev_output = torch.from_numpy(prev_output).cuda()
        cat_seq1 = prev_output[:, -5:, :, :, :3].permute(0, 4, 1, 2, 3) # [B, C, F, H, W]
        mask_seq1 = prev_output[:, -5:, :, :, -1:].permute(0, 4, 1, 2, 3) # [B, C, F, H, W]
        mask_seq1 = mask_seq1 > 0
        prev_output = prev_output[:, -5:].permute(0, 4, 1, 2, 3)
        
    cur_output = model_output[:, :, :5]
    cat_seq2 = model_output[:, :3, -5:] # [B, C, F, H, W]
    mask_seq2 = model_output[:, -1:, -5:] # [B, C, F, H, W]
    mask_seq2 = mask_seq2 > 0
    
    if t < 200:
        if prev_output is not None:
            loss = ((prev_output - cur_output) ** 2).mean() * 20
            
            full_seq = torch.cat([cat_seq1, cat_seq2], dim=2)    # [B, 3, F_total, H, W]
            mask_seq = torch.cat([mask_seq1, mask_seq2], dim=2)  # [B, 1, F_total, H, W]
            vel = full_seq[:, :, 1:] - full_seq[:, :, :-1]       # [B, C, F_total-1, H, W]
            mask_vel = mask_seq[:, :, 1:] * mask_seq[:, :, :-1]  # [B, 1, F_total-1, H, W]
            eps = 1e-8
            valid_vel_elements = mask_vel.sum() + eps

            acc = vel[:, :, 1:] - vel[:, :, :-1]  # [B, C, F_total-2, H, W]
            mask_acc = mask_seq[:, :, 2:] * mask_seq[:, :, 1:-1] * mask_seq[:, :, :-2]  # [B, 1, F_total-2, H, W]
            valid_acc_elements = mask_acc.sum() + eps

            loss_vel = (vel.pow(2) * mask_vel).sum() / valid_vel_elements
            loss_acc = (acc.pow(2) * mask_acc).sum() / valid_acc_elements
            
        else:
            loss = torch.tensor(0).cuda()
            loss_vel = torch.tensor(0).cuda()
            loss_acc = torch.tensor(0).cuda()
    else:
        loss = torch.tensor(0).cuda()
        loss_vel = torch.tensor(0).cuda()
        loss_acc = torch.tensor(0).cuda()
        
    if t < 200:
        log_msg = f"t: {t}, loss_overlap: {loss.item():.4f}, loss_vel: {loss_vel.item():.4f}, loss_acc: {loss_acc.item():.4f}\n"
        print(log_msg.strip())
        
        with open(log_file, "a") as f:
            f.write(log_msg)
    return loss + loss_vel + loss_acc

def sliding_window_with_padding(images_list, seq_len):
    """
    Divide indices from 0 to len(images_list)-1 into len(images_list) sliding windows 
    of length seq_len, padding with start/end elements if needed.

    Args:
        images_list (list): List of images or any elements to process.
        seq_len (int): Length of each sliding window.

    Returns:
        list: A list of sliding windows with padding.
    """
    n = len(images_list)
    indices = list(range(n))
    windows = []

    for i in range(n):
        start = max(0, i - seq_len // 2)
        end = min(n, i + seq_len // 2 + 1)

        window = indices[start:end]

        while len(window) < seq_len and start == 0:
            window.insert(0, indices[0])

        while len(window) < seq_len and end == n:
            window.append(indices[-1])

        window = window[:seq_len]
        windows.append(window)
    window = indices[-(seq_len // 2):-1]
    while len(window) < seq_len:
        window.append(indices[-1])
    windows.append(window)
    unique_windows = []
    for w in windows:
        if len(unique_windows) == 0 or unique_windows[-1] != w:
            unique_windows.append(w)

    return unique_windows

is_depth = True
use_guidance = True
measure_func = _measure_func if use_guidance else None
resolution = 128
ddpm_num_steps = 1000
ddpm_beta_schedule = 'linear'
eval_batch_size = 1
ddpm_num_inference_steps = 1000
weight_dtype = torch.float32
seq_len = 10
num_stride = 1

raster, renderer_textured_hard = get_render()
raster_back, renderer_textured_hard_back = get_render(is_back=True)
color_smpl = np.load('../extra-data/color_smpl_faces.npy')/15

sliding_windows = sliding_window_with_padding(images_list, seq_len)[4:-5:5]
    
prev_images_uv = None
start_t=1000-1
denoise_t=start_t
last_images = None

with open(os.path.join(save_folder, "a_loss_back_normal.txt"), 'w') as f:
    f.write("")
    
for index, window in enumerate(sliding_windows):
    cond_imgs = []

    if True:
        for i in window:
            img_name = images_list[i]

            normal = cv2.imread(os.path.join(normal_dir, '%s_normal_align.png'%img_name))
            if garment == 'Skirt':
                seg = cv2.imread(os.path.join(normal_dir, '%s_seg_align.png'%img_name))[:,:,0]
            else:
                seg = cv2.imread(os.path.join(normal_dir, f'{img_name}_seg_{garment}_align.png'))[:,:,0]
            mask = ((seg == target_label).astype(np.uint8))*255
            normal[seg != target_label] = 0
            
            body_smpl = trimesh.load(os.path.join(body_dir, '%s_body.ply'%img_name))
            body_seg = render_segmentation(body_smpl, renderer_textured_hard, raster)
            body_seg_back = render_segmentation(body_smpl, renderer_textured_hard_back, raster_back)
            body_seg_back = np.fliplr(body_seg_back)
            body_depth = render_depth(body_smpl, renderer_textured_hard)
            body_depth_back = render_depth(body_smpl, renderer_textured_hard_back)
            body_depth_back = np.fliplr(body_depth_back)
            cv2.imwrite(os.path.join(save_folder, 'body_seg_%s.png'%img_name), body_seg)
            cv2.imwrite(os.path.join(save_folder, 'body_seg_back_%s.png'%img_name), body_seg_back)
            cv2.imwrite(os.path.join(save_folder, 'body_depth_%s.png'%img_name), ((body_depth+1)/2*255).astype(np.uint8))
            cv2.imwrite(os.path.join(save_folder, 'body_depth_back_%s.png'%img_name), ((body_depth_back+1)/2*255).astype(np.uint8))
            
            body_seg = _process_seg(body_seg, resize=True)
            body_seg_back = _process_seg(body_seg_back, resize=True)
            body_depth = _process_depth(body_depth, resize=True)
            body_depth_back = _process_depth(body_depth_back, resize=True)

            cond_img, normal_resize = _process_image(normal, resize=True)
            cv2.imwrite(os.path.join(save_folder, 'normal_resize_%s.png'%img_name), normal_resize)
            
            cond_img = np.concatenate((cond_img, body_seg[:,:, [0]], body_depth, body_seg_back[:,:, [0]], body_depth_back), axis=-1)
            
            cond_imgs.append(cond_img)
            
        cond_imgs = np.stack(cond_imgs, axis=0)
        conditions = torch.FloatTensor(cond_imgs).cuda().permute(3, 0, 1, 2).unsqueeze(0)
            
        observation = [prev_images_uv] if use_guidance else []

        for j in range(0, 1):
            images_uv = pipeline(
                conditions=conditions,
                generator=torch.Generator(device=pipeline.device).manual_seed(j),
                batch_size=1,
                num_inference_steps=ddpm_num_inference_steps,
                output_type="numpy",
                start_image=None,
                start_t=start_t,
                denoise_t=denoise_t,
                use_guidance=use_guidance,
                measure_func=measure_func,
                observation=observation,
                guide_scale=50.0,
                is_3D=len(conditions.shape)==5
            ).images 
            
            if index == 0:
                begin = 0
            else:
                images_uv = images_uv[:, -seq_len//2:]
                begin = seq_len//2
            
            prev_images_uv = _process_prev_image(images_uv[:, -seq_len//2:])
            
            for k in range(images_uv.shape[1]):
                img_name = images_list[window[begin+k]]
                images_normal = images_uv[:, k, :,:,:3]
                images_mask = images_uv[:, k, :,:,-1:]

                images_mask = images_mask > 0.5
                images_mask = images_mask.astype(np.uint8)*255
                images_normal = (images_normal*255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_folder, 'images_normal_back_%s.png'%(img_name)), images_normal[0])
                cv2.imwrite(os.path.join(save_folder, 'images_mask_back_%s.png'%(img_name)), images_mask[0])