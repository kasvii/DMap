import os, sys
import cv2
import numpy as np
import trimesh
import torch
import diffusers
from diffusers import DDPMScheduler, UNet2DModel
import torch.nn.functional as F
from einops import rearrange
import argparse

sys.path.append('../temporal_diffusion')
from pipeline_ddpm_condition_seq_guide import DDPMPipeline
from models.unet import UNet3DConditionModel

sys.path.append('..')
from utils.mesh import apply_rotation
from utils.readfile import load_pkl
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

def process_image(image, size=(256,256)):
    image_new = (image_new.astype(np.float32)/255 - 0.5)/0.5
    return image_new

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

def render_torsor(body, renderer_textured_hard, raster):

    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()

    pix_to_face = get_pix_to_face(verts, faces, raster)
    pix_to_face = pix_to_face.detach().cpu().numpy()
    coord_img, coord_faces = face_to_uv_coord(pix_to_face)

    body_torsor = np.zeros((len(pix_to_face), len(pix_to_face), 3))
    body_torsor[coord_img[:,0], coord_img[:,1]] = color_smpl_raw[coord_faces]
    body_torsor = np.logical_or(body_torsor==1, body_torsor==2)

    return body_torsor

def render_depth_discrete(body, renderer_textured_hard, raster, flip_bg=False):
    
    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()
    tri_depth = torch.FloatTensor(body.triangles_center).cuda()[:,-1]

    pix_to_face = get_pix_to_face(verts, faces, raster)
    pix_to_face = pix_to_face.detach().cpu().numpy()
    coord_img, coord_faces = face_to_uv_coord(pix_to_face)

    body_depth = np.zeros((len(pix_to_face), len(pix_to_face))) - 1
    body_depth[coord_img[:,0], coord_img[:,1]] = tri_depth[coord_faces].detach().cpu().numpy()

    mask_depth = body_depth == -1
    if flip_bg:
        body_depth[mask_depth] = 1
        
    return body_depth

def render_depth(body, renderer_textured_hard, flip_bg=False):
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
    if flip_bg:
        images_depth[mask_depth] = 1
    else:
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

def draw_uv_mapping(images, coor, size_uv=256):
    offset = (size_uv - 1.0)/2
    uv = images[:,:,:,:2]*2 -1
    fb = images[:,:,:,-1]*2 -1
    uv = -uv*offset + offset
    
    uv_mapping = []
    uv_mapping_mask = []
    for i in range(len(uv)):
        uv_i = uv[i]
        fb_i = fb[i]
        coor_i = coor[i]

        uv_i = uv_i[coor_i[:, 0], coor_i[:, 1]]
        fb_i = fb_i[coor_i[:, 0], coor_i[:, 1]]

        idx_front = fb_i > 0
        idx_back = fb_i < 0

        img_front = np.zeros((size_uv, size_uv, 3))
        img_back = np.zeros((size_uv, size_uv, 3))
        img_front_mask = np.zeros((size_uv, size_uv, 3))
        img_back_mask = np.zeros((size_uv, size_uv, 3))

        uv_i = np.round(uv_i).astype(int)
        uv_i_f = uv_i[idx_front]
        uv_i_b = uv_i[idx_back]
        coor_i_f = coor_i[idx_front]
        coor_i_b = coor_i[idx_back]

        y_f, x_f = uv_i_f[:, 0], uv_i_f[:, 1]
        y_b, x_b = uv_i_b[:, 0], uv_i_b[:, 1]
        img_front[x_f, y_f, :2] = coor_i_f
        img_back[x_b, y_b, :2] = coor_i_b

        img_front_mask[x_f, y_f] = 255
        img_back_mask[x_b, y_b] = 255

        img = np.concatenate((img_front, img_back), axis=1).astype(np.uint8)
        img_mask = np.concatenate((img_front_mask, img_back_mask), axis=1).astype(np.uint8)
        uv_mapping.append(img)
        uv_mapping_mask.append(img_mask)

    uv_mapping = np.stack(uv_mapping, axis=0)
    uv_mapping_mask = np.stack(uv_mapping_mask, axis=0)
    return uv_mapping, uv_mapping_mask

def mask_to_coord(mask):
    coords = []

    for i in range(len(mask)):
        x, y = np.where(mask[i])
        coord = np.stack((x,y), axis=-1)
        coords.append(coord)

    return coords

parser = argparse.ArgumentParser(description="Generate the back normal maps")
parser.add_argument("--garment", type=str, default='Skirt', help="The type of garment")
parser.add_argument("--scale", type=float, default=0.8, help="The scale of the garment")
parser.add_argument("--vid_name", type=str, default='vid_demo', help="The name of the video")
args = parser.parse_args()

vid_name = args.vid_name
garment = args.garment
scale = args.scale

load_folder = f'../fitting-results/{vid_name}/uv-mapping-back-{scale}-{garment}'
save_folder = f'../fitting-results/{vid_name}/uv-mapping-back-{scale}-{garment}'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def masked_cosine_similarity_loss(pred_normals, gt_normals, mask):
    mask = mask.unsqueeze(1)
    
    pred_normals = F.normalize(pred_normals, dim=1)
    gt_normals = F.normalize(gt_normals, dim=1)

    cos_sim = torch.sum(pred_normals * gt_normals, dim=1, keepdim=True)  # [B, 1, T, H, W]

    loss = (1 - cos_sim) * mask
    loss = loss.sum() / (mask.sum() + 1e-8)

    return loss

def depth_to_normal(depth_map, mask, img_size=192, is_back=False):
    scale = 1
    focal = 1
    depth_map = depth_map.unsqueeze(1)
    mask = mask.unsqueeze(1)
    B, C, T, H, W = depth_map.shape

    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float().to(depth_map.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float().to(depth_map.device)
    depth_reshaped = rearrange(depth_map, 'b c t h w -> (b t) c h w')
    mask_reshaped = rearrange(mask, 'b c t h w -> (b t) c h w')
    
    dzdx = F.conv2d(depth_reshaped, sobel_x, padding=1)  / 8 * img_size / 2 
    dzdy = F.conv2d(depth_reshaped, sobel_y, padding=1)  / 8 * img_size / 2 
    
    z_component = torch.ones_like(depth_reshaped)
    dzdx *= mask_reshaped
    dzdy *= mask_reshaped
    z_component *= mask_reshaped
    
    dzdx = dzdx * scale * focal
    dzdy = dzdy * scale * focal

    normal = torch.cat([-dzdx, dzdy, z_component], dim=1)  # [B*T, 3, H, W]
    if is_back:
        normal = -normal
        
    normal = normal / (torch.norm(normal, dim=1, keepdim=True) + 1e-8)
    normal = normal[:, [2, 1, 0], ...] 

    normal = rearrange(normal, '(b t) c h w -> b c t h w', b=B, t=T)

    return normal

def _measure_func(model_output, observation, t, eps=0, log_file=os.path.join(save_folder, "loss_log.txt")):
    prev_output = observation[0]
    if prev_output is not None:
        prev_output = torch.from_numpy(prev_output).cuda()
        prev_output = prev_output[:, -5:].permute(0, 4, 1, 2, 3)
    cur_output = model_output[:, :, :5]
    
    mask_bool_f, mask_bool_b, depth_f, depth_b, mask_full_f, mask_full_b, normal_f, normal_b, mask_close_f, mask_close_b = observation[1:]
    output_depth_f = model_output[:,3] # garment front depth
    output_depth_b = model_output[:,7] # garment back depth
    
    if t < 200:
        if prev_output is not None:
            loss_consi = ((prev_output - cur_output) ** 2).mean() * 100.0
            
        else:
            loss_consi = torch.tensor(0).cuda()
            
        loss_f = F.relu(depth_f[mask_bool_f] - output_depth_f[mask_bool_f] + eps).mean()
        loss_b = F.relu(output_depth_b[mask_bool_b] - depth_b[mask_bool_b] + eps).mean()
        loss_depth = (loss_f + loss_b) * 100
        
        if garment == 'Skirt':
            dpt2normal_f = depth_to_normal(output_depth_f, mask_full_f)
            dpt2normal_b = depth_to_normal(output_depth_b, mask_full_b, is_back=True)
            loss_normal_f = masked_cosine_similarity_loss(dpt2normal_f, normal_f, mask_full_f)
            loss_normal_b = masked_cosine_similarity_loss(dpt2normal_b, normal_b, mask_full_b)
            loss_normal = (loss_normal_f + loss_normal_b) * 0.1
            if t == 0:    
                for idx in range(dpt2normal_f.shape[2]):
                    cv2.imwrite(os.path.join('../tmp', f'dpt2normal_f_{t}_{idx}.png'), ((dpt2normal_f[0, :, idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join('../tmp', f'dpt2normal_f_gt_{t}_{idx}.png'), ((normal_f[0, :, idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join('../tmp', f'dpt2normal_b_{t}_{idx}.png'), ((dpt2normal_b[0, :, idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join('../tmp', f'dpt2normal_b_gt_{t}_{idx}.png'), ((normal_b[0, :, idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8))
        else:
            loss_normal_f = torch.tensor(0).cuda()
            loss_normal_b = torch.tensor(0).cuda()
            loss_normal = torch.tensor(0).cuda()
        
        loss_body_f = ((depth_f[mask_close_f] - output_depth_f[mask_close_f])**2).mean()
        loss_body_b = ((depth_b[mask_close_b] - output_depth_b[mask_close_b])**2).mean()
        loss_body = (loss_body_f + loss_body_b)
    else:
        loss_consi = torch.tensor(0).cuda()
        loss_f = F.relu(depth_f[mask_bool_f] - output_depth_f[mask_bool_f] + eps).mean()
        loss_b = F.relu(output_depth_b[mask_bool_b] - depth_b[mask_bool_b] + eps).mean()
        loss_depth = (loss_f + loss_b)*0
        
        if garment == 'Skirt':
            dpt2normal_f = depth_to_normal(output_depth_f, mask_full_f)
            dpt2normal_b = depth_to_normal(output_depth_b, mask_full_b, is_back=True)
            loss_normal_f = masked_cosine_similarity_loss(dpt2normal_f, normal_f, mask_full_f)
            loss_normal_b = masked_cosine_similarity_loss(dpt2normal_b, normal_b, mask_full_b)
            loss_normal = (loss_normal_f + loss_normal_b) * 0
        else:
            loss_normal_f = torch.tensor(0).cuda()
            loss_normal_b = torch.tensor(0).cuda()
            loss_normal = torch.tensor(0).cuda()
        
        loss_body_f = (depth_f[mask_close_f] - output_depth_f[mask_close_f]).abs().mean()
        loss_body_b = (depth_b[mask_close_b] - output_depth_b[mask_close_b]).abs().mean()
        loss_body = (loss_body_f + loss_body_b)*0
        
    loss = loss_consi + loss_depth + loss_normal + loss_body
    
    if t < 200:
        log_msg = f"t: {t}, loss: {loss.item():.4f}, loss_consi: {loss_consi.item():.4f}, loss_depth: {loss_depth.item():.4f}, loss_body: {loss_body.item():.4f}, loss_normal: {loss_normal.item():.4f}, loss_depth_f: {loss_f.item():.4f}, loss_depth_b: {loss_b.item():.4f}, loss_normal_f: {loss_normal_f.item():.4f}, loss_normal_b: {loss_normal_b.item():.4f}, loss_body_f: {loss_body_f.item():.4f}, loss_body_b: {loss_body_b.item():.4f}\n"
        with open(log_file, "a") as f:
            f.write(log_msg)
    return loss

def _to_xyz(coord_img, z, img_size=191.):
    scale = img_size/2
    yx = (coord_img - scale)/scale
    y, x = -yx[:,0], yx[:,1]

    xyz = np.stack((x,y,z), axis=-1)
    return xyz

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
    indices = list(range(n))  # Create a list of indices [0, 1, ..., n-1]
    windows = []

    for i in range(n):
        # Start and end indices for the window
        start = max(0, i - seq_len // 2)
        end = min(n, i + seq_len // 2 + 1)

        # Get the current window and pad if necessary
        window = indices[start:end]

        # Pad at the beginning if the window is too short
        while len(window) < seq_len and start == 0:
            window.insert(0, indices[0])

        # Pad at the end if the window is too short
        while len(window) < seq_len and end == n:
            window.append(indices[-1])

        # Ensure the window length is strictly seq_len
        window = window[:seq_len]  # Trim extra elements if needed
        windows.append(window)
    window = indices[-(seq_len // 2):-1]
    while len(window) < seq_len:
        window.append(indices[-1])
    windows.append(window)
    # Remove consecutive duplicate windows
    unique_windows = []
    for w in windows:
        if len(unique_windows) == 0 or unique_windows[-1] != w:
            unique_windows.append(w)

    return unique_windows

def overlay_images(body_seg, mask_full, alpha=0.5):
    if body_seg.shape[:2] != mask_full.shape[:2]:
        mask_full = cv2.resize(mask_full, (body_seg.shape[1], body_seg.shape[0]))

    if len(mask_full.shape) == 2:
        mask_full = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(body_seg, 1 - alpha, mask_full, alpha, 0)
    return blended

def remove_arm(color_smpl_faces):
    new_faces_id = []
    for i in range(len(color_smpl_faces)):
        if color_smpl_faces[i,0] in [3, 4, 11, 12, 13, 14]:
            continue
        else:
            new_faces_id.append(i)

    return new_faces_id


pretrained_unet_path = '../checkpoints/uv_mapping_FB/unet_ema'

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
target_label = 240

use_guidance = True
measure_func = _measure_func if use_guidance else None
is_depth = True
use_uv = True and is_depth
resolution = 128
ddpm_num_steps = 1000
ddpm_beta_schedule = 'linear'
eval_batch_size = 1
repeat_size = 1
ddpm_num_inference_steps = 1000
weight_dtype = torch.float32
seq_len = 10
num_stride = 1

generator = torch.Generator(device=pipeline.device).manual_seed(0)

image_dir = f'../data/{vid_name}/images'
seg_body_dir = f'../data/{vid_name}/processed/segmentation_all/'
seg_gar_dir = f'../data/{vid_name}/processed/segmentation_{garment}/'
normal_dir = f'../data/{vid_name}/results-{scale}'
body_dir = f"../data/{vid_name}/cropped_body"

images_list = sorted([img.split('.')[0] for img in os.listdir(image_dir)])

sliding_windows = sliding_window_with_padding(images_list, seq_len)[4:-5:5]

raster, renderer_textured_hard = get_render()
raster_back, renderer_textured_hard_back = get_render(is_back=True)
color_smpl_raw = np.load('../extra-data/color_smpl_faces.npy')
color_smpl = np.load('../extra-data/color_smpl_faces.npy')/15
faces_id_no_arm = remove_arm((color_smpl_raw).astype(int))

prev_images_uv = None
start_t=900-1
denoise_t=start_t
last_images = None
repeat_t, stride_num, sample_num = 100, 10, 5
log_file=os.path.join(save_folder, "loss_log.txt")
with open(log_file, "w") as f:
    f.write("")
for index, window in enumerate(sliding_windows):
    with open(log_file, "a") as f:
        f.write(f'{window=}\n')
    cond_img_fbs, body_depths, mask_fronts, mask_backs = [], [], [], []
    body_depth_backs, body_depth_raws, body_depth_back_raws = [], [], []
    mask_torsors, mask_torsor_backs = [], []
    normal_imgs, normal_img_backs = [], []
    last_name = images_list[window[-1]].split('_')[0]
    if os.path.exists(os.path.join(save_folder, 'n_%s.npy'%(last_name))):
        continue
    
    if True:
        for i in window:
            img_name = images_list[i].split('_')[0]
            
            normal = cv2.imread(os.path.join(normal_dir, '%s_normal_align.png'%img_name))
            if garment == 'Skirt':
                seg = cv2.imread(os.path.join(normal_dir, '%s_seg_align.png'%img_name))[:,:,0]
            else:
                seg = cv2.imread(os.path.join(normal_dir, f'{img_name}_seg_{garment}_align.png'))[:,:,0]
            mask = ((seg == target_label).astype(np.uint8))*255
            normal[seg != target_label] = 0
            
            mask_back = cv2.imread(os.path.join(load_folder, 'images_mask_back_%s.png'%(img_name)))[:,:,0]/255
            normal_back = cv2.imread(os.path.join(load_folder, 'images_normal_back_%s.png'%(img_name)))
            normal_back = normal_back.astype(np.float32)/255
            normal_back = (normal_back - 0.5)/0.5
            normal_back[mask_back==0] = -1
            
            body_smpl = trimesh.load(os.path.join(body_dir, '%s_body.ply'%img_name))
            body_smpl.export(os.path.join(save_folder, 'body_%s.ply'%img_name))
            
            body_mesh_no_arm = trimesh.Trimesh(body_smpl.vertices, body_smpl.faces[faces_id_no_arm])
            body_mesh_no_arm.export(os.path.join(save_folder, 'body_smpl_no_arm_%s.ply'%img_name))
            
            body_seg = render_segmentation(body_smpl, renderer_textured_hard, raster)
            body_seg_back = render_segmentation(body_smpl, renderer_textured_hard_back, raster_back)
            body_seg_back = np.fliplr(body_seg_back)
            body_depth = render_depth(body_smpl, renderer_textured_hard)
            body_depth_back = render_depth(body_smpl, renderer_textured_hard_back, flip_bg=False)
            body_depth_back = np.fliplr(body_depth_back)
            
            cv2.imwrite(os.path.join(save_folder, 'body_seg_%s.png'%img_name), body_seg)
            
            body_depth_raw = render_depth_discrete(body_mesh_no_arm, renderer_textured_hard, raster)
            body_depth_back_raw = render_depth_discrete(body_mesh_no_arm, renderer_textured_hard_back, raster_back)
            body_depth_back_raw = np.fliplr(body_depth_back_raw)
            
            mask_torsor = render_torsor(body_smpl, renderer_textured_hard, raster).astype(np.uint8)[:,:,0]
            mask_torsor_back = render_torsor(body_smpl, renderer_textured_hard_back, raster_back).astype(np.uint8)[:,:,0]
            mask_torsor_back = np.fliplr(mask_torsor_back)
            mask_torsor = cv2.resize(mask_torsor, (192, 192), interpolation=cv2.INTER_NEAREST)
            mask_torsor_back = cv2.resize(mask_torsor_back, (192, 192), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(save_folder, 'mask_torsor_%s.png'%img_name), mask_torsor*255)
            cv2.imwrite(os.path.join(save_folder, 'mask_torsor_back_%s.png'%img_name), mask_torsor_back*255)
            
            body_seg = _process_seg(body_seg, resize=True)
            body_seg_back = _process_seg(body_seg_back, resize=True)
            body_depth = _process_depth(body_depth, resize=True)
            body_depth_back = _process_depth(body_depth_back, resize=True)
            
            body_depth_raw = _process_depth(body_depth_raw, resize=True)
            body_depth_back_raw = _process_depth(body_depth_back_raw, resize=True)
            
            cond_img, normal_front = _process_image(normal, resize=True)
            cv2.imwrite(os.path.join(save_folder, 'normal_front_%s.png'%img_name), normal_front)

            _, mask_front = _process_image(mask, resize=True)
            mask_front = mask_front == 255
            cv2.imwrite(os.path.join(save_folder, 'mask_%s.png'%img_name), mask.astype(np.uint8))
            cv2.imwrite(os.path.join(save_folder, 'mask_front_%s.png'%img_name), mask_front.astype(np.uint8)*255)
            
            normal_imgs.append(cond_img)
            normal_img_backs.append(normal_back)
            
            cond_img = np.concatenate((cond_img, body_seg[:,:, [0]], body_depth), axis=-1)
            cond_img_back = np.concatenate((normal_back, body_seg_back[:,:, [0]], body_depth_back), axis=-1)
            cond_img_fb = np.concatenate((cond_img, cond_img_back), axis=-1)
            
            cond_img_fbs.append(cond_img_fb)
            body_depths.append(body_depth)
            body_depth_backs.append(body_depth_back)
            body_depth_raws.append(body_depth_raw)
            body_depth_back_raws.append(body_depth_back_raw)
            mask_fronts.append(mask_front)
            mask_backs.append(mask_back)
            mask_torsors.append(mask_torsor)
            mask_torsor_backs.append(mask_torsor_back)
            
        cond_img_fbs = np.stack(cond_img_fbs, axis=0)
        body_depths = np.stack(body_depths, axis=0)
        body_depth_backs = np.stack(body_depth_backs, axis=0)
        body_depth_raws = np.stack(body_depth_raws, axis=0)
        body_depth_back_raws = np.stack(body_depth_back_raws, axis=0)
        mask_fronts = np.stack(mask_fronts, axis=0)
        mask_backs = np.stack(mask_backs, axis=0)
        mask_torsors = np.stack(mask_torsors, axis=0)
        mask_torsor_backs = np.stack(mask_torsor_backs, axis=0)
        
        normal_imgs = np.stack(normal_imgs, axis=0)
        normal_img_backs = np.stack(normal_img_backs, axis=0)
        
        conditions = torch.FloatTensor(cond_img_fbs).cuda().permute(3, 0, 1, 2).unsqueeze(0)
        mask_bools = torch.BoolTensor(mask_fronts).unsqueeze(0).cuda()
        mask_bool_backs = torch.BoolTensor(mask_backs).unsqueeze(0).cuda()
        body_depths = torch.FloatTensor(body_depths[...,0]).unsqueeze(0).cuda()
        body_depth_backs = torch.FloatTensor(body_depth_backs[...,0]).unsqueeze(0).cuda()
        body_depth_raws = torch.FloatTensor(body_depth_raws[...,0]).unsqueeze(0).cuda()
        body_depth_back_raws = torch.FloatTensor(body_depth_back_raws[...,0]).unsqueeze(0).cuda()
        mask_torsors = torch.BoolTensor(mask_torsors).unsqueeze(0).cuda()
        mask_torsor_backs = torch.BoolTensor(mask_torsor_backs).unsqueeze(0).cuda()
        
        normal_imgs = torch.FloatTensor(normal_imgs).cuda().permute(3, 0, 1, 2).unsqueeze(0) # [B, C, T, H, W]
        normal_img_backs = torch.FloatTensor(normal_img_backs).cuda().permute(3, 0, 1, 2).unsqueeze(0) # [B, C, T, H, W]
        
        mask_body_fs = torch.logical_and(body_depth_raws != 1, body_depth_raws != -1)
        mask_body_bs = torch.logical_and(body_depth_back_raws != 1, body_depth_back_raws != -1)
        mask_body_fs = torch.logical_and(mask_body_fs, mask_bools)
        mask_body_bs = torch.logical_and(mask_body_bs, mask_bool_backs)
        cv2.imwrite(os.path.join('../tmp/mask_body_fs_%s.png'%(img_name)), ((mask_body_fs[0, -1]).detach().cpu().numpy()*255).astype(np.uint8))
        cv2.imwrite(os.path.join('../tmp/mask_body_bs_%s.png'%(img_name)), ((mask_body_bs[0, -1]).detach().cpu().numpy()*255).astype(np.uint8))
        
        body_depth_backs[body_depth_backs==-1] = 1
        body_depth_back_raws[body_depth_back_raws==-1] = 1
        
        mask_torsors = torch.logical_and(mask_bools, mask_torsors)
        mask_torsors = torch.logical_and(mask_body_fs, mask_torsors)
        mask_torsor_backs = torch.logical_and(mask_bool_backs, mask_torsor_backs)
        mask_torsor_backs = torch.logical_and(mask_body_bs, mask_torsor_backs)
        
        if garment == 'Skirt':
            observation = [prev_images_uv, mask_body_fs, mask_body_bs, body_depth_raws, body_depth_back_raws, mask_bools, mask_bool_backs, normal_imgs, normal_img_backs, mask_torsors, mask_torsor_backs] if use_guidance else []
        
        
        if last_images is not None:
            noisy_images = torch.from_numpy(last_images).float().permute(0, 4, 1, 2, 3).cuda() # [B, C, F, H, W]
        else:
            noisy_images = None
        
        images_uv = pipeline(
                conditions=conditions,
                generator=generator,
                batch_size=repeat_size,
                num_inference_steps=ddpm_num_inference_steps,
                output_type="numpy",
                start_image=noisy_images,
                start_t=start_t,
                denoise_t=denoise_t,
                use_guidance=use_guidance,
                measure_func=measure_func,
                observation=observation,
                guide_scale=50.0, # 20.0
                is_3D=len(conditions.shape)==5
            ).images
        
        last_images = images_uv.copy()
        
        if index == 0:
            begin = 0
        else:
            images_uv = images_uv[:, -seq_len//2:]
            begin = seq_len//2
            
        prev_images_uv = _process_prev_image(images_uv[:, -seq_len//2:])
        
        for i in range(images_uv.shape[1]):
            img_id = window[begin+i]
            output_img_name = str(img_id).zfill(6)
            
            images_uv_f = images_uv[:,i,:,:,:4]
            images_uv_b = images_uv[:,i,:,:,4:]
            coord_img_f = mask_to_coord(mask_fronts[begin+i].reshape(1, mask_front.shape[0], mask_front.shape[1]))
            coord_img_b = mask_to_coord(mask_backs[begin+i].reshape(1, mask_back.shape[0], mask_back.shape[1]))
            
            np.savez(
                os.path.join(save_folder, f'uv_transfer_{output_img_name}'),
                uv_transfer=images_uv[0, i] * 2 - 1
            )
            
            images_depth_f = (images_uv_f[:,:,:,[-1]] * 255).round().astype("uint8")
            images_depth_b = (images_uv_b[:,:,:,[-1]] * 255).round().astype("uint8")

            uv_mapping_pred_f, uv_mapping_pred_mask_f = draw_uv_mapping(images_uv_f[:,:,:,:3], coord_img_f)
            uv_mapping_pred_b, uv_mapping_pred_mask_b = draw_uv_mapping(images_uv_b[:,:,:,:3], coord_img_b)

            cv2.imwrite(os.path.join(save_folder, 'img_pred_depth_transfer_f_%s.png'%(output_img_name)), images_depth_f[0])
            cv2.imwrite(os.path.join(save_folder, 'img_pred_depth_transfer_b_%s.png'%(output_img_name)), images_depth_b[0])

            cv2.imwrite(os.path.join(save_folder, 'img_pred_uv_transfer_f_%s.png'%(output_img_name)), (images_uv_f[0,:,:,:3] * 255).round().astype("uint8"))
            cv2.imwrite(os.path.join(save_folder, 'img_pred_uv_transfer_b_%s.png'%(output_img_name)), (images_uv_b[0,:,:,:3] * 255).round().astype("uint8"))
            cv2.imwrite(os.path.join(save_folder, 'img_pred_uv_f_%s.png'%(output_img_name)), uv_mapping_pred_f[0])
            cv2.imwrite(os.path.join(save_folder, 'img_pred_uv_b_%s.png'%(output_img_name)), uv_mapping_pred_b[0])
            cv2.imwrite(os.path.join(save_folder, 'img_pred_uv_mask_f_%s.png'%(output_img_name)), uv_mapping_pred_mask_f[0])
            cv2.imwrite(os.path.join(save_folder, 'img_pred_uv_mask_b_%s.png'%(output_img_name)), uv_mapping_pred_mask_b[0])

            coord_img_f = coord_img_f[0]
            coord_img_b = coord_img_b[0]
            depth_img_f = images_uv_f[0][:,:,[-1]]*2-1
            depth_img_b = images_uv_b[0][:,:,[-1]]*2-1
            z_f = depth_img_f[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
            z_b = depth_img_b[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
            xyz_f = _to_xyz(coord_img_f, z_f).astype(np.float32)
            xyz_b = _to_xyz(coord_img_b, z_b).astype(np.float32)
            xyz = np.concatenate((xyz_f, xyz_b), axis=0)
            pt = trimesh.PointCloud(xyz)
            pt.export(os.path.join(save_folder, 'xyz_%s.ply'%(output_img_name)))

            n_f = (normal_front[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1, 3).astype(np.float32)/255 - 0.5)/0.5
            n_b = normal_back[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1, 3)
            n_f = n_f/np.linalg.norm(n_f, keepdims=True, axis=-1)
            n_b = n_b/np.linalg.norm(n_b, keepdims=True, axis=-1)
            n = np.concatenate((n_f, n_b), axis=0)
            np.save(os.path.join(save_folder, 'n_%s.npy'%(output_img_name)), n)
            