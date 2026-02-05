import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import diffusers
from diffusers import DDPMScheduler
sys.path.append('..')
from pipeline.pipeline_ddpm_guided_right_ddnm_guide import DDPMPipeline
from utils.chamfer import chamfer_distance_single, chamfer_distance
from utils.isp import create_uv_mesh, uv_to_3D, barycentric_faces, get_barycentric
from utils.readfile import load_pkl
from utils.rasterize import get_pix_to_face_with_body, get_pix_to_face_v2, get_pix_to_face_index, get_pix_to_face_with_body_index, get_raster, get_pix_to_face, get_pix_to_face_v2_cam
from snug.snug_helper import stretching_energy, bending_energy, gravitational_energy, collision_penalty, collision_penalty_lite, spring_energy, inertial_term_sequence
from snug.snug_class import Cloth_from_NP, Material
from utils.cutting import get_connected_paths_skirt, select_boundary, get_connected_paths_sleeves
from utils.mesh import apply_rotation
from networks.SDF import SDF

import open3d as o3d
import argparse

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

import time
from einops import rearrange

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

def get_render(render_res=256, direction=0):
    render_res = render_res
    dis = 100.0
    scale = 100
    mesh_y_center = 0.0
    cam_pos = torch.tensor([
                    (0, mesh_y_center, dis), # front
                    (0, mesh_y_center, -dis), # back
                    (dis, mesh_y_center, 0), # left
                    (-dis, mesh_y_center, 0), # right
                ])
    R, T = look_at_view_transform(
        eye=cam_pos[[direction]],
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
    transform = cameras.get_full_projection_transform()

    sigma = 1e-7

    raster_settings_hard = RasterizationSettings(
        image_size=render_res, 
        blur_radius=np.log(1. / 1e-4)*sigma, 
        faces_per_pixel=1, 
        max_faces_per_bin=500000,
        perspective_correct=False,
    )
    
    raster_settings_soft = RasterizationSettings(
        image_size=render_res, 
        blur_radius=np.log(1. / 1e-4 - 1.)*1e-5,
        faces_per_pixel=50, 
        max_faces_per_bin=500000,
        perspective_correct=False,
    )

    meshRas_hard = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_hard)
    meshRas_soft = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_soft)

    renderer_textured_hard = MeshRenderer(
        rasterizer=meshRas_hard,
        shader=cleanShader(blend_params=BlendParams_blackBG())
    )
    
    renderer_textured_soft = MeshRenderer(
        rasterizer=meshRas_soft,
        shader=cleanShader(blend_params=BlendParams_blackBG())
    )

    return meshRas_hard, renderer_textured_hard, renderer_textured_soft, transform, cameras

def align_observation_pt_verts(img_init, cloth_pose, cloth_state, body_mesh, normals, vertices_waist, waist_v_id, depths, renderer_soft, renderer_soft_back, consis_info, mapping_related, log_file_path):

    prev_uv, prev_prev_uv, prev_uv_mask = consis_info

    depth_img, mask_depth = depths
    vertices_waist = torch.FloatTensor(vertices_waist).cuda()
    
    faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz, xyz_f, xyz_b = mapping_related
    
    faces_f = torch.LongTensor(faces_f).cuda()
    faces_b = torch.LongTensor(faces_b).cuda()
    closest_face_idx_f = torch.LongTensor(closest_face_idx_f).cuda()
    closest_face_idx_b = torch.LongTensor(closest_face_idx_b).cuda()
    v_barycentric_f = torch.FloatTensor(v_barycentric_f).cuda()
    v_barycentric_b = torch.FloatTensor(v_barycentric_b).cuda()

    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    vb_flip = vb.clone()
    vb_flip[:, -1] *= -1
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()
    fb = torch.LongTensor(body_mesh.faces).cuda()

    vb_clothed = torch.FloatTensor(clothed_mesh.vertices).cuda()
    vb_clothed_flip = vb_clothed.clone()
    vb_clothed_flip[:, -1] *= -1
    fb_clothed = torch.LongTensor(clothed_mesh.faces).cuda()

    normal_front, normal_back, mask_front, mask_back, mask_top, n_xyz = normals
    mask_front = torch.FloatTensor(mask_front).cuda()
    mask_back = torch.FloatTensor(mask_back).cuda()
    mask_top = torch.FloatTensor(mask_top).cuda()
    n_xyz = torch.FloatTensor(n_xyz).cuda().unsqueeze(0)
    

    with torch.no_grad():
        faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
        verts_cloth_zero = torch.FloatTensor(cloth_pose.vertices).cuda()*0
        cloth_rgb = torch.zeros(len(verts_cloth_zero), 3) + 255 # (1, V, 3)
        verts_rgb = cloth_rgb[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        
        idx_x, idx_y = np.where(mask_front.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_f = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

        normal = (normal_front[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_f = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_f = normal_img_f.unsqueeze(0)
        
        idx_x, idx_y = np.where(mask_back.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_b = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
        normal = (normal_back[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_b = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_b = normal_img_b.unsqueeze(0)
        
        idx_x, idx_y = np.where(((mask_front+mask_top).cpu().numpy())>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_full_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
    if prev_uv is not None:
        prev_uv_f = prev_uv[:, :, :128, :].reshape(-1, 3)  # [1, 128, 128, 3]
        prev_uv_b = prev_uv[:, :, 128:, :].reshape(-1, 3)  # [1, 128, 128, 3]
    if prev_prev_uv is not None:
        prev_prev_uv_f = prev_prev_uv[:, :, :128, :].reshape(-1, 3)  # [1, 128, 128, 3]
        prev_prev_uv_b = prev_prev_uv[:, :, 128:, :].reshape(-1, 3)  # [1, 128, 128, 3]

    if prev_uv_mask is not None:
        mask_f = prev_uv_mask[:, :, :128, :].reshape(-1, 3)   # [1, 128, 128]
        mask_b = prev_uv_mask[:, :, 128:, :].reshape(-1, 3)   # [1, 128, 128]



    verts_zero_clothed = torch.zeros(len(body_mesh.vertices)+len(cloth_pose.vertices), 3).cuda()
    faces_clothed = torch.LongTensor(np.concatenate((body_mesh.faces, cloth_pose.faces + len(body_mesh.vertices)))).cuda()
    smpl_rgb = torch.zeros(len(body_mesh.vertices), 3)
    smpl_rgb[:,0] += 255
    gar_rgb = torch.zeros(len(cloth_pose.vertices), 3)
    gar_rgb[:,1] += 255
    verts_rgb = torch.cat((smpl_rgb, gar_rgb))[None]
    textures_clothed = TexturesVertex(verts_features=verts_rgb.to(device))

    
    offset = torch.zeros_like(img_init[:,:3])
    offset.requires_grad = True
    lr = 1e-4
    eps = 2e-3
    optimizer = torch.optim.Adam([{'params': offset, 'lr': lr},])

    iters = 200
    
    for i in range(iters):
        image_est = img_init[:,:3] + offset
        uv_est = image_est[:,:3].squeeze().permute(1,2,0)
        uv_f = uv_est[:,:128].reshape(-1,3)
        uv_b = uv_est[:,128:].reshape(-1,3)

        verts_f = uv_to_3D_torch(uv_f, faces_f, v_barycentric_f, closest_face_idx_f)
        verts_b = uv_to_3D_torch(uv_b, faces_b, v_barycentric_b, closest_face_idx_b)
        verts_cloth_new = torch.cat((verts_f, verts_b), axis=0)  
        
        if prev_uv is not None:
            loss_vel_front = torch.linalg.norm(uv_f[mask_f] - prev_uv_f[mask_f], dim=-1).mean() * 0.25
            loss_vel_back = torch.linalg.norm(uv_b[mask_b] - prev_uv_b[mask_b], dim=-1).mean() * 0.5
            loss_vel = loss_vel_front + loss_vel_back
        else:
            loss_vel = torch.tensor(0.0).cuda()
            
        if prev_prev_uv is not None:
            loss_acc_front = torch.linalg.norm(uv_f[mask_f] - 2*prev_uv_f[mask_f] + prev_prev_uv_f[mask_f], dim=-1).mean() * 0.25
            loss_acc_back = torch.linalg.norm(uv_b[mask_b] - 2*prev_uv_b[mask_b] + prev_prev_uv_b[mask_b], dim=-1).mean() * 0.5
            loss_acc = loss_acc_front + loss_acc_back
        else:
            loss_acc = torch.tensor(0.0).cuda()
        

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)

        if i == iters-1:
            mesh = Meshes(
                verts=[verts_cloth_zero],   
                faces=[faces_cloth],
                textures=textures
            )
            new_src_mesh = mesh.offset_verts(verts_cloth_new)
            images_predicted = renderer_soft(new_src_mesh)
            images_pred = images_predicted[0, :, :, 3]

            img_mask = (images_pred.detach().cpu().numpy()*255).astype(np.uint8)
        

        with torch.no_grad():
            idx_faces_f, _ = get_pix_to_face_with_body(verts_cloth_new, faces_cloth, vb_clothed, fb_clothed, raster)
            verts_cloth_new_flip = verts_cloth_new.clone()
            verts_cloth_new_flip[:,-1] *=-1
            idx_faces_b, _ = get_pix_to_face_with_body(verts_cloth_new_flip, faces_cloth, vb_clothed_flip, fb_clothed, raster)
            faces_cloth_f = faces_cloth[idx_faces_f]
            faces_cloth_b = faces_cloth[idx_faces_b]
        tri_f = verts_cloth_new[faces_cloth_f.reshape(-1)].reshape(-1,3,3)
        tri_b = verts_cloth_new[faces_cloth_b.reshape(-1)].reshape(-1,3,3)
        tri_center_f = tri_f.mean(dim=1)
        tri_center_b = tri_b.mean(dim=1)
        vectors_f = tri_f[:,1:] - tri_f[:,:2]
        vectors_b = tri_b[:,1:] - tri_b[:,:2]
        normal_f = torch.cross(vectors_f[:, 0], vectors_f[:, 1], dim=-1)
        normal_b = torch.cross(vectors_b[:, 0], vectors_b[:, 1], dim=-1)
        normal_f = normal_f/normal_f.norm(p=2, dim=-1, keepdim=True)
        normal_b = normal_b/normal_b.norm(p=2, dim=-1, keepdim=True)
        normal_f = normal_f.unsqueeze(0)
        normal_b = normal_b.unsqueeze(0)


        verts_cloth_2D_f = (transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_b = (transform.transform_points(tri_center_b.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_pend_f = torch.cat((verts_cloth_2D_f, torch.zeros(verts_cloth_2D_f.shape[0], verts_cloth_2D_f.shape[1], 1).cuda()), dim=-1)
        verts_cloth_2D_pend_b = torch.cat((verts_cloth_2D_b, torch.zeros(verts_cloth_2D_b.shape[0], verts_cloth_2D_b.shape[1], 1).cuda()), dim=-1)
        
        _, loss_normal_f = chamfer_distance(verts_cloth_2D_pend_f, idx_mask_pend_f, x_normals=normal_f, y_normals=normal_img_f)
        _, loss_normal_b = chamfer_distance(idx_mask_pend_b, verts_cloth_2D_pend_b, x_normals=normal_img_b, y_normals=normal_b)
        loss_cd_2d_f_0, _ = chamfer_distance_single(idx_mask_pend_f, verts_cloth_2D_pend_f)
        loss_cd_2d_f_1, _ = chamfer_distance_single(verts_cloth_2D_pend_f, idx_mask_full_pend)
        loss_cd_2d_b_0, _ = chamfer_distance_single(idx_mask_pend_b, verts_cloth_2D_pend_b)
        loss_cd_2d_b_1, _ = chamfer_distance_single(verts_cloth_2D_pend_b, idx_mask_full_pend)
        loss_mask = (loss_cd_2d_f_0 + loss_cd_2d_f_1 + loss_cd_2d_b_0 + loss_cd_2d_b_1)/5/10
        loss_normal = (loss_normal_f + loss_normal_b)*2

        verts_clothed = torch.cat((vb, verts_cloth_new), dim=0)
        mesh_f = Meshes(
                verts=[verts_zero_clothed],   
                faces=[faces_clothed],
                textures=textures_clothed
        )
        new_src_mesh_f = mesh_f.offset_verts(verts_clothed)
        images_predicted_f = renderer_soft(new_src_mesh_f)
        images_predicted_b = renderer_soft_back(new_src_mesh_f)
        images_pred_f = images_predicted_f[0, :, :, 1]/255
        images_pred_b = images_predicted_b[0, :, :, 1]/255

        intersection_f = (images_pred_f*mask_front).sum()
        intersection_b = (images_pred_b*mask_back).sum()
        union_f = images_pred_f.sum() + mask_front.sum() - intersection_f
        union_b = images_pred_b.sum() + mask_back.sum() - intersection_b
        loss_mask = ((1 - intersection_f/union_f) + (1 - intersection_b/union_b)) #*224
        
        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_3d, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_3d*2
        loss_cd_3d *= 100*50*2


        loss_collision = collision_penalty(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps)

        loss_waist = torch.sqrt(((verts_cloth_new[waist_v_id] - vertices_waist)**2).sum(dim=-1)).mean()*10

        if use_depth:
            verts_cloth_2D_sample = transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[0,1]]*(-1)
            verts_cloth_2D_sample = rearrange(verts_cloth_2D_sample.detach(), 'b n t -> b n 1 t')

            mask_img_sample = torch.nn.functional.grid_sample(mask_depth.unsqueeze(0).unsqueeze(0), verts_cloth_2D_sample, align_corners=True)
            mask_img_sample = rearrange(mask_img_sample, 'b c n 1 -> b c n').squeeze()
            depth_img_sample = torch.nn.functional.grid_sample(depth_img.unsqueeze(0).unsqueeze(0), verts_cloth_2D_sample, align_corners=True)
            depth_img_sample = rearrange(depth_img_sample, 'b c n 1 -> b c n').squeeze()
            loss_depth = ((tri_center_f[mask_img_sample > 0.99, -1] - depth_img_sample[mask_img_sample > 0.99])**2).mean()*100
        else:
            loss_depth = torch.zeros(1).cuda()

        loss = (loss_bending + loss_strain/2 + loss_gravity) + loss_cd_3d + loss_mask + loss_normal + loss_collision + loss_waist + loss_depth + loss_vel + loss_acc
        log_str = 'opt verts iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_mask: %0.4f , loss_cd_3d: %0.4f , loss_normal: %0.4f , loss_collision: %0.4f , loss_waist: %0.4f , loss_depth: %0.4f, loss_vel: %0.4f, loss_acc: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_mask.item(), loss_cd_3d.item(), loss_normal.item(), loss_collision.item(), loss_waist.item(), loss_depth.item(), 
            loss_vel.item(), loss_acc.item())
        
        print(log_str)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with open(log_file_path, 'a') as f:
            f.write(log_str + '\n')


    cloth_pose_new = cloth_pose.copy()
    cloth_pose_new.vertices = verts_cloth_new.detach().cpu().numpy()
    image_est_remesh = image_est.detach().clone()

    return cloth_pose_new, image_est_remesh, img_mask

def align_observation_uv(img_init, cloth_pose, cloth_state, mapping_related, image_rest, masks, normals, body_mesh, clothed_mesh, waist_v_id, depths, renderer_soft, renderer_soft_back, mask_fulls, consis_info, log_file_path, vertices_waist):
    
    prev_uv, prev_prev_uv, prev_uv_mask = consis_info

    vertices_waist = torch.FloatTensor(vertices_waist).cuda()
    
    verts_cloth = torch.FloatTensor(cloth_pose.vertices).cuda()

    depth_img, mask_depth = depths
    depth_img = torch.FloatTensor(depth_img).cuda()
    mask_depth = torch.FloatTensor(mask_depth).cuda()
    
    mask_full, mask_full_flip = mask_fulls

    faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz, xyz_f, xyz_b = mapping_related

    waist_v_id_loop = waist_v_id + waist_v_id[:1]
    waist_edges = np.array([waist_v_id_loop[:-1], waist_v_id_loop[1:]]).astype(int)
    edges = cloth_pose.vertices[waist_edges.reshape(-1)].reshape(-1, 2, 3)
    edges_oritation_gt = edges[:, 0] - edges[:, 1]
    edges_oritation_gt = torch.FloatTensor(edges_oritation_gt).cuda()
    waist_edges = torch.from_numpy(waist_edges).cuda()

    faces_f = torch.LongTensor(faces_f).cuda()
    faces_b = torch.LongTensor(faces_b).cuda()
    closest_face_idx_f = torch.LongTensor(closest_face_idx_f).cuda()
    closest_face_idx_b = torch.LongTensor(closest_face_idx_b).cuda()
    v_barycentric_f = torch.FloatTensor(v_barycentric_f).cuda()
    v_barycentric_b = torch.FloatTensor(v_barycentric_b).cuda()


    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    vb_flip = vb.clone()
    vb_flip[:, -1] *= -1
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()
    fb = torch.LongTensor(body_mesh.faces).cuda()

    vb_clothed = torch.FloatTensor(clothed_mesh.vertices).cuda()
    vb_clothed_flip = vb_clothed.clone()
    vb_clothed_flip[:, -1] *= -1
    fb_clothed = torch.LongTensor(clothed_mesh.faces).cuda()


    normal_front, normal_back, mask_front, mask_back, mask_top, n_xyz = normals
    mask_front = torch.FloatTensor(mask_front).cuda()
    mask_back = torch.FloatTensor(mask_back).cuda()
    mask_top = torch.FloatTensor(mask_top).cuda()
    n_xyz = torch.FloatTensor(n_xyz).cuda().unsqueeze(0)

    with torch.no_grad():
        faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
        verts_cloth_zero = torch.FloatTensor(cloth_pose.vertices).cuda()*0
        cloth_rgb = torch.zeros(len(verts_cloth_zero), 3) + 255 # (1, V, 3)
        verts_rgb = cloth_rgb[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        
        idx_x, idx_y = np.where(mask_front.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_f = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

        normal = (normal_front[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_f = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_f = normal_img_f.unsqueeze(0)
        
        idx_x, idx_y = np.where(mask_back.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_b = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
        normal = (normal_back[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_b = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_b = normal_img_b.unsqueeze(0)

        idx_x, idx_y = np.where(((mask_front+mask_top).cpu().numpy())>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_full_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
    
    if prev_uv is not None:
        prev_uv_f = prev_uv[:, :, :128, :].reshape(-1, 3)  # [1, 128, 128, 3]
        prev_uv_b = prev_uv[:, :, 128:, :].reshape(-1, 3)  # [1, 128, 128, 3]
    if prev_prev_uv is not None:
        prev_prev_uv_f = prev_prev_uv[:, :, :128, :].reshape(-1, 3)  # [1, 128, 128, 3]
        prev_prev_uv_b = prev_prev_uv[:, :, 128:, :].reshape(-1, 3)  # [1, 128, 128, 3]

    if prev_uv_mask is not None:
        mask_f = prev_uv_mask[:, :, :128, :].reshape(-1, 3)   # [1, 128, 128]
        mask_b = prev_uv_mask[:, :, 128:, :].reshape(-1, 3)   # [1, 128, 128]

    verts_zero_clothed = torch.zeros(len(body_mesh.vertices)+len(cloth_pose.vertices), 3).cuda()
    faces_clothed = torch.LongTensor(np.concatenate((body_mesh.faces, cloth_pose.faces + len(body_mesh.vertices)))).cuda()
    smpl_rgb = torch.zeros(len(body_mesh.vertices), 3)
    smpl_rgb[:,0] += 255
    gar_rgb = torch.zeros(len(cloth_pose.vertices), 3)
    gar_rgb[:,1] += 255
    verts_rgb = torch.cat((smpl_rgb, gar_rgb))[None]
    textures_clothed = TexturesVertex(verts_features=verts_rgb.to(device))
    
    nn = SDF(d_in=6, d_out=3, dims=[256, 256, 256], skip_in=[]).cuda()
    lr = 1e-3
    eps = 2e-3
    optimizer = torch.optim.Adam(list(nn.parameters()), lr=lr)
    
    condition = torch.cat((image_rest, img_init[:,:3]), dim=1)*10

    iters = 1000
    for i in range(iters):
        condition_reshape = condition.permute(0,2,3,1).reshape(-1, 6)
        offset = nn(condition_reshape, None)/100
        offset = offset.reshape(1, 128, 256, 3).permute(0,3,1,2)

        image_est = img_init[:,:3] + offset#*0

        uv_est = image_est[:,:3].squeeze().permute(1,2,0)
        uv_f = uv_est[:,:128].reshape(-1,3)
        uv_b = uv_est[:,128:].reshape(-1,3)

        output_uv = image_est[:,:3].permute(0,2,3,1)
        loss_sparse_uv = torch.linalg.norm((output_uv[sparse_mask] - sparse_uv[sparse_mask]), dim=-1).mean()*100*5*2#/2
        
        if i >= 500:
            loss_sparse_uv /= 10
        
        if prev_uv is not None:
            loss_vel_front = torch.linalg.norm(uv_f[mask_f] - prev_uv_f[mask_f], dim=-1).mean() * 0.25
            loss_vel_back = torch.linalg.norm(uv_b[mask_b] - prev_uv_b[mask_b], dim=-1).mean() * 0.5
            loss_vel = loss_vel_front + loss_vel_back
        else:
            loss_vel = torch.tensor(0.0).cuda()
            
        if prev_prev_uv is not None:
            loss_acc_front = torch.linalg.norm(uv_f[mask_f] - 2*prev_uv_f[mask_f] + prev_prev_uv_f[mask_f], dim=-1).mean() * 0.25
            loss_acc_back = torch.linalg.norm(uv_b[mask_b] - 2*prev_uv_b[mask_b] + prev_prev_uv_b[mask_b], dim=-1).mean() * 0.5
            loss_acc = loss_acc_front + loss_acc_back
        else:
            loss_acc = torch.tensor(0.0).cuda()
        

        verts_f = uv_to_3D_torch(uv_f, faces_f, v_barycentric_f, closest_face_idx_f)
        verts_b = uv_to_3D_torch(uv_b, faces_b, v_barycentric_b, closest_face_idx_b)
        verts_cloth_new = torch.cat((verts_f, verts_b), axis=0)      
    

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)


        if i == iters-1:
            mesh = Meshes(
                verts=[verts_cloth_zero],   
                faces=[faces_cloth],
                textures=textures
            )
            new_src_mesh = mesh.offset_verts(verts_cloth_new)
            images_predicted = renderer_soft(new_src_mesh)
            images_pred = images_predicted[0, :, :, 3]

            img_mask = (images_pred.detach().cpu().numpy()*255).astype(np.uint8)
        

        with torch.no_grad():
            idx_faces_f, _ = get_pix_to_face_with_body(verts_cloth_new, faces_cloth, vb_clothed, fb_clothed, raster)
            verts_cloth_new_flip = verts_cloth_new.clone()
            verts_cloth_new_flip[:,-1] *=-1
            idx_faces_b, _ = get_pix_to_face_with_body(verts_cloth_new_flip, faces_cloth, vb_clothed_flip, fb_clothed, raster)
            faces_cloth_f = faces_cloth[idx_faces_f]
            faces_cloth_b = faces_cloth[idx_faces_b]
        tri_f = verts_cloth_new[faces_cloth_f.reshape(-1)].reshape(-1,3,3)
        tri_b = verts_cloth_new[faces_cloth_b.reshape(-1)].reshape(-1,3,3)
        tri_center_f = tri_f.mean(dim=1)
        tri_center_b = tri_b.mean(dim=1)
        vectors_f = tri_f[:,1:] - tri_f[:,:2]
        vectors_b = tri_b[:,1:] - tri_b[:,:2]
        normal_f = torch.cross(vectors_f[:, 0], vectors_f[:, 1], dim=-1)
        normal_b = torch.cross(vectors_b[:, 0], vectors_b[:, 1], dim=-1)
        normal_f = normal_f/normal_f.norm(p=2, dim=-1, keepdim=True)
        normal_b = normal_b/normal_b.norm(p=2, dim=-1, keepdim=True)
        normal_f = normal_f.unsqueeze(0)
        normal_b = normal_b.unsqueeze(0)

        
        verts_cloth_2D_f = (transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5
        verts_cloth_2D_b = (transform.transform_points(tri_center_b.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5
        verts_cloth_2D_pend_f = torch.cat((verts_cloth_2D_f, torch.zeros(verts_cloth_2D_f.shape[0], verts_cloth_2D_f.shape[1], 1).cuda()), dim=-1)
        verts_cloth_2D_pend_b = torch.cat((verts_cloth_2D_b, torch.zeros(verts_cloth_2D_b.shape[0], verts_cloth_2D_b.shape[1], 1).cuda()), dim=-1)
        
        _, loss_normal_f = chamfer_distance(verts_cloth_2D_pend_f, idx_mask_pend_f, x_normals=normal_f, y_normals=normal_img_f)
        _, loss_normal_b = chamfer_distance(idx_mask_pend_b, verts_cloth_2D_pend_b, x_normals=normal_img_b, y_normals=normal_b)
        loss_cd_2d_f_0, _ = chamfer_distance_single(idx_mask_pend_f, verts_cloth_2D_pend_f)
        loss_cd_2d_f_1, _ = chamfer_distance_single(verts_cloth_2D_pend_f, idx_mask_full_pend)
        loss_cd_2d_b_0, _ = chamfer_distance_single(idx_mask_pend_b, verts_cloth_2D_pend_b)
        loss_cd_2d_b_1, _ = chamfer_distance_single(verts_cloth_2D_pend_b, idx_mask_full_pend)
        loss_mask = (loss_cd_2d_f_0 + loss_cd_2d_f_1 + loss_cd_2d_b_0 + loss_cd_2d_b_1)/5
        loss_normal = (loss_normal_f + loss_normal_b)*2

        verts_clothed = torch.cat((vb, verts_cloth_new), dim=0)
        mesh_f = Meshes(
                verts=[verts_zero_clothed],   
                faces=[faces_clothed],
                textures=textures_clothed
        )
        new_src_mesh_f = mesh_f.offset_verts(verts_clothed)
        images_predicted_f = renderer_soft(new_src_mesh_f)
        images_predicted_b = renderer_soft_back(new_src_mesh_f)
        images_pred_f = images_predicted_f[0, :, :, 1]/255
        images_pred_b = images_predicted_b[0, :, :, 1]/255

        intersection_f = (images_pred_f*mask_full).sum()
        intersection_b = (images_pred_b*mask_full_flip).sum()
        union_f = images_pred_f.sum() + mask_full.sum() - intersection_f
        union_b = images_pred_b.sum() + mask_full_flip.sum() - intersection_b
        loss_full_mask_f = (1 - intersection_f/union_f)
        loss_full_mask_b = (1 - intersection_b/union_b)
        loss_mask = (loss_full_mask_f + loss_full_mask_b)

        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_3d, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_3d*2
        if i < 500:
            loss_cd_3d *= 100*5*0
        else:
            loss_cd_3d *= 100*50*2
            
        loss_cd_3d_ori, _ = chamfer_distance(verts_cloth.unsqueeze(0), verts_cloth_new.unsqueeze(0), )
        if i < 300:
            loss_cd_3d_ori *= 100 / 5
        else:
            loss_cd_3d_ori *= 10000 / 5

        loss_collision = collision_penalty(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps)
        loss_edge = torch.sqrt(((verts_cloth_new[waist_v_id] - vertices_waist)**2).sum(dim=-1)).mean()*10

        if use_depth:
            verts_cloth_2D_sample = transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[0,1]]*(-1)
            verts_cloth_2D_sample = rearrange(verts_cloth_2D_sample.detach(), 'b n t -> b n 1 t')

            mask_img_sample = torch.nn.functional.grid_sample(mask_depth.unsqueeze(0).unsqueeze(0), verts_cloth_2D_sample, align_corners=True)
            mask_img_sample = rearrange(mask_img_sample, 'b c n 1 -> b c n').squeeze()
            depth_img_sample = torch.nn.functional.grid_sample(depth_img.unsqueeze(0).unsqueeze(0), verts_cloth_2D_sample, align_corners=True)
            depth_img_sample = rearrange(depth_img_sample, 'b c n 1 -> b c n').squeeze()
            loss_depth = ((tri_center_f[mask_img_sample > 0.99, -1] - depth_img_sample[mask_img_sample > 0.99])**2).mean()*100*(i/1500)
        else:
            loss_depth = torch.zeros(1).cuda()

        loss = (loss_bending + loss_strain/4 + loss_gravity) + loss_sparse_uv + loss_cd_3d + loss_mask + loss_normal + loss_collision + loss_edge + loss_depth + loss_cd_3d_ori + loss_vel + loss_acc
        log_str = 'uv iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_sparse_uv: %0.4f , loss_mask: %0.4f , loss_cd_3d: %0.4f , loss_normal: %0.4f , loss_collision: %0.4f , loss_edge: %0.4f , loss_depth: %0.4f , loss_cd_3d_ori: %0.4f, loss_vel: %0.4f, loss_acc: %0.4f '%(
            i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_sparse_uv.item(), loss_mask.item(), 
            loss_cd_3d.item(), loss_normal.item(), loss_collision.item(), loss_edge.item(), loss_depth.item(), loss_cd_3d_ori.item(), 
            loss_vel.item(), loss_acc.item())

        print(log_str)
        
        with open(log_file_path, 'a') as f:
            f.write(log_str + '\n')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 499:
            verts_cloth_new = verts_cloth_new.detach().cpu().numpy()
            cloth_pose_uv = cloth_pose.copy()
            cloth_pose_uv.vertices = verts_cloth_new
            image_est_uv = image_est.detach().clone()


    verts_cloth_new = verts_cloth_new.detach().cpu().numpy()
    cloth_pose_remesh = cloth_pose.copy()
    cloth_pose_remesh.vertices = verts_cloth_new
    image_est_remesh = image_est.detach().clone()

    return cloth_pose_uv, image_est_uv, cloth_pose_remesh, image_est_remesh, img_mask

def remesh_uv(uv_f, uv_b, cloth_pose, mapping_related, cloth_state, body_mesh, consis_info, log_file_path):
    
    faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, cloth_faces = mapping_related
    
    prev_uv, prev_prev_uv, prev_uv_mask = consis_info
    
    if prev_uv is not None:
        prev_uv_f = prev_uv[:, :, :128, :].reshape(-1, 3)  # [1, 128, 128, 3]
        prev_uv_b = prev_uv[:, :, 128:, :].reshape(-1, 3)  # [1, 128, 128, 3]
    if prev_prev_uv is not None:
        prev_prev_uv_f = prev_prev_uv[:, :, :128, :].reshape(-1, 3)  # [1, 128, 128, 3]
        prev_prev_uv_b = prev_prev_uv[:, :, 128:, :].reshape(-1, 3)  # [1, 128, 128, 3]

    if prev_uv_mask is not None:
        mask_f = prev_uv_mask[:, :, :128, :].reshape(-1, 3)   # [1, 128, 128]
        mask_b = prev_uv_mask[:, :, 128:, :].reshape(-1, 3)   # [1, 128, 128]
    
    idx_boundary_v, _ = select_boundary(cloth_pose)
    
    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()

    verts_cloth = torch.FloatTensor(cloth_pose.vertices).cuda()
    tri_center_cloth = torch.FloatTensor(cloth_pose.triangles_center).cuda()
    faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
    normals_cloth = torch.FloatTensor(cloth_pose.face_normals).cuda()
    valid_fn = torch.isnan(normals_cloth).sum(dim=-1) == 0
    
    verts_boundary = torch.FloatTensor(cloth_pose.vertices[idx_boundary_v]).cuda()
    idx_boundary = torch.LongTensor(idx_boundary_v).cuda()
    
    offset_f = torch.randn(uv_f.shape).cuda()*0.001*0
    offset_b = torch.randn(uv_b.shape).cuda()*0.001*0
    offset_f.requires_grad = True
    offset_b.requires_grad = True
    lr = 1e-3
    eps = 1e-3
    optimizer = torch.optim.Adam([
        {'params': offset_f, 'lr': lr},
        {'params': offset_b, 'lr': lr},
    ])

    iters = 1000
    for i in range(iters):
        uv_f_new = uv_f + offset_f
        uv_b_new = uv_b + offset_b
        verts_f_new = uv_to_3D_torch(uv_f_new, faces_f, v_barycentric_f, closest_face_idx_f)
        verts_b_new = uv_to_3D_torch(uv_b_new, faces_b, v_barycentric_b, closest_face_idx_b)
        verts_cloth_new = torch.cat((verts_f_new, verts_b_new), axis=0)
        uv_full_new = torch.cat((uv_f_new.reshape(1, 128, 128, 3), uv_b_new.reshape(1, 128, 128, 3)), axis=1).reshape(1, 128, 256, 3) # [1, 128, 256, 3]
        loss_waist, _ = chamfer_distance(verts_boundary.unsqueeze(0), verts_cloth_new[idx_boundary].unsqueeze(0), )
        if i < 300:
            loss_waist *= 100
        else:
            loss_waist *= 10000

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)#*50

        if prev_uv is not None:
            loss_vel = torch.linalg.norm(uv_full_new[prev_uv_mask] - prev_uv[prev_uv_mask], dim=-1).mean() * 0.005 # 0.5 # 
        else:
            loss_vel = torch.tensor(0.0).cuda()
            
        if prev_prev_uv is not None:
            loss_acc = torch.linalg.norm(uv_full_new[prev_uv_mask] - 2*prev_uv[prev_uv_mask] + prev_prev_uv[prev_uv_mask], dim=-1).mean() *  0.005
        else:
            loss_acc = torch.tensor(0.0).cuda()
        
        tri_full = verts_cloth_new[faces_cloth.reshape(-1)].reshape(-1,3,3)
        tri_center = tri_full.mean(dim=1)
        vec1 = tri_full[:,1] - tri_full[:,0]
        vec2 = tri_full[:,2] - tri_full[:,0]
        normal_full = torch.cross(vec1, vec2, dim=-1)
        normal_full = normal_full/normal_full.norm(p=2, dim=-1, keepdim=True)

        valid_fn_pred = torch.isnan(normal_full).sum(dim=-1) == 0
        _valid_fn = torch.logical_and(valid_fn_pred, valid_fn)

        
        loss_cd_3d, _ = chamfer_distance(verts_cloth.unsqueeze(0), verts_cloth_new.unsqueeze(0), )
        if i < 300:
            loss_cd_3d *= 100
        else:
            loss_cd_3d *= 10000

        _, loss_normal_3d = chamfer_distance(tri_center[_valid_fn].unsqueeze(0), tri_center_cloth[_valid_fn].unsqueeze(0), x_normals=normal_full[_valid_fn].unsqueeze(0), y_normals=normals_cloth[_valid_fn].unsqueeze(0), abs_normal=True)
        if i < 300:
            loss_normal_3d /= 100

        loss_collision = collision_penalty(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps)

        loss = (loss_bending + loss_strain/2 + loss_gravity) + loss_cd_3d + loss_waist + loss_normal_3d*10 + loss_collision
        log_str = 'remesh-uv iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_cd_3d: %0.4f , loss_waist: %0.4f  , loss_normal_3d: %0.4f , loss_collision: %0.4f, loss_vel: %0.4f, loss_acc: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_cd_3d.item(), loss_waist.item(), loss_normal_3d.item(), loss_collision.item(), loss_vel.item(), loss_acc.item())
        print(log_str)
        
        with open(log_file_path, 'a') as f:
            f.write(log_str + '\n')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cloth_pose_remesh = cloth_pose.copy()
    cloth_pose_remesh.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_remesh, uv_f_new.detach().cpu().numpy(), uv_b_new.detach().cpu().numpy()

def face_to_uv_coord(pix_to_face):
    x, y = np.where(pix_to_face != -1)
    coord_img = np.stack((x,y), axis=-1)

    faces = pix_to_face[x, y]

    return coord_img, faces

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

def render_torsor(body, renderer_textured_hard, raster):

    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()

    pix_to_face = get_pix_to_face(verts, faces, raster)
    pix_to_face = pix_to_face.detach().cpu().numpy()
    coord_img, coord_faces = face_to_uv_coord(pix_to_face)

    body_torsor = np.zeros((len(pix_to_face), len(pix_to_face), 3))
    body_torsor[coord_img[:,0], coord_img[:,1]] = color_smpl[coord_faces]
    body_torsor = np.logical_or(body_torsor==1, body_torsor==2)

    return body_torsor

def _to_xyz(coord_img, z, img_size=255.):
    scale = img_size/2
    yx = (coord_img - scale)/scale
    y, x = -yx[:,0], yx[:,1]

    xyz = np.stack((x,y,z), axis=-1)
    return xyz

def _to_uv(image, coord_img, xyz, size_uv=128):
    offset = (size_uv - 1.0)/2
    uv = image[:,:,:2]
    fb = image[:,:,-1]
    uv = -uv*offset + offset

    uv = uv[coord_img[:, 0], coord_img[:, 1]]
    fb = fb[coord_img[:, 0], coord_img[:, 1]]
    idx_front = fb > 0
    idx_back = fb < 0

    img_front = np.zeros((size_uv, size_uv, 3)) - 1
    img_back = np.zeros((size_uv, size_uv, 3)) - 1
    img_front_mask = np.zeros((size_uv, size_uv))
    img_back_mask = np.zeros((size_uv, size_uv))

    uv = np.round(uv).astype(int)
    uv_f = uv[idx_front]
    uv_b = uv[idx_back]
    xyz_f = xyz[idx_front]
    xyz_b = xyz[idx_back]

    y_f, x_f = uv_f[:, 0], uv_f[:, 1]
    y_b, x_b = uv_b[:, 0], uv_b[:, 1]
    
    img_front[x_f, y_f] = xyz_f
    img_back[x_b, y_b] = xyz_b

    img_front_mask[x_f, y_f] = 1
    img_back_mask[x_b, y_b] = 1

    sparse_uv = np.concatenate((img_front, img_back), axis=1)
    sparse_mask = np.concatenate((img_front_mask, img_back_mask), axis=1)

    return sparse_uv, sparse_mask

def _to_uv_FB(image_F, image_B, coord_img_F, coord_img_B, xyz_F, xyz_B, size_uv=128):
    offset = (size_uv - 1.0)/2
    uv_F = image_F[:,:,:2]
    uv_B = image_B[:,:,:2]
    fb_F = image_F[:,:,-1]
    fb_B = image_B[:,:,-1]
    uv_F = -uv_F*offset + offset
    uv_B = -uv_B*offset + offset

    uv_F = uv_F[coord_img_F[:, 0], coord_img_F[:, 1]]
    uv_B = uv_B[coord_img_B[:, 0], coord_img_B[:, 1]]
    fb_F = fb_F[coord_img_F[:, 0], coord_img_F[:, 1]]
    fb_B = fb_B[coord_img_B[:, 0], coord_img_B[:, 1]]
    idx_front_F = fb_F > 0
    idx_front_B = fb_B > 0
    idx_back_F = fb_F < 0
    idx_back_B = fb_B < 0

    img_front = np.zeros((size_uv, size_uv, 3)) - 1
    img_back = np.zeros((size_uv, size_uv, 3)) - 1
    img_front_mask = np.zeros((size_uv, size_uv))
    img_back_mask = np.zeros((size_uv, size_uv))

    uv_F = np.round(uv_F).astype(int)
    uv_f_F = uv_F[idx_front_F]
    uv_b_F = uv_F[idx_back_F]
    xyz_f_F = xyz_F[idx_front_F]
    xyz_b_F = xyz_F[idx_back_F]

    uv_B = np.round(uv_B).astype(int)
    uv_f_B = uv_B[idx_front_B]
    uv_b_B = uv_B[idx_back_B]
    xyz_f_B = xyz_B[idx_front_B]
    xyz_b_B = xyz_B[idx_back_B]

    y_f_F, x_f_F = uv_f_F[:, 0], uv_f_F[:, 1]
    y_b_F, x_b_F = uv_b_F[:, 0], uv_b_F[:, 1]

    y_f_B, x_f_B = uv_f_B[:, 0], uv_f_B[:, 1]
    y_b_B, x_b_B = uv_b_B[:, 0], uv_b_B[:, 1]
    
    img_front[x_f_B, y_f_B] = xyz_f_B
    img_back[x_b_B, y_b_B] = xyz_b_B
    img_front[x_f_F, y_f_F] = xyz_f_F
    img_back[x_b_F, y_b_F] = xyz_b_F

    img_front_mask[x_f_F, y_f_F] = 1
    img_back_mask[x_b_F, y_b_F] = 1
    img_front_mask[x_f_B, y_f_B] = 1
    img_back_mask[x_b_B, y_b_B] = 1

    sparse_uv = np.concatenate((img_front, img_back), axis=1)
    sparse_mask = np.concatenate((img_front_mask, img_back_mask), axis=1)

    return sparse_uv, sparse_mask

def uv_to_3D_torch(pattern_deform, uv_faces, barycentric_uv, closest_face_idx_uv):
    uv_faces_id = uv_faces[closest_face_idx_uv]
    uv_faces_id = uv_faces_id.reshape(-1)

    pattern_deform_triangles = pattern_deform[uv_faces_id].reshape(-1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv[:, :, None]).sum(dim=-2)
    return pattern_deform_bary

def clean_uv(uv_mask):
    invalid_uv = np.where(uv_mask == 0)[0]
    valid_uv = np.where(uv_mask != 0)[0]
    return set(valid_uv.flatten().tolist())

def filter_faces(faces, valid_v):
    faces_new = []
    for f in faces:
        if f[0] in valid_v and f[1] in valid_v and f[2] in valid_v:
            faces_new.append(f)

    faces_new = np.array(faces_new).astype(int)
    return faces_new

def dilate_indicator(mask, size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    mask = cv2.dilate(mask, kernel)
    return mask

def erode_indicator(mask, size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    mask = cv2.erode(mask, kernel)
    return mask

def save_point_cloud(filename, points):
    """
    Save 2D point cloud as a PLY file (with dummy z-values set to 0).
    
    Args:
        filename (str): Path to save the PLY file.
        points (torch.Tensor): Tensor of shape (B, N, 1, 2) containing 2D points.
    """
    points = points.squeeze(0).squeeze(1).detach().cpu().numpy()  # Shape (N, 2)
    
    # Convert to structured array with x, y, and dummy z=0
    if points.shape[1] == 2:
        vertex = np.array([(x, y, 0.0) for x, y in points])
    else:
        vertex = np.array([(x, y, z) for x, y, z in points])
    pt = trimesh.PointCloud(vertex)
    pt.export(filename)
    print(f"Saved {filename}")
    
import torch
import torch.nn.functional as F

def uv_smoothness_conv_loss(uv_map, kernel_size=5):
    """
    uv_map: [B, 2, H, W]
    kernel_size: 感受野大小（建议 3, 5, 7）
    """
    uv_map = uv_map.permute(0, 3, 1, 2)  # [B, C, H, W]
    B, C, H, W = uv_map.shape

    device = uv_map.device
    padding = kernel_size // 2

    kernel = -1 * torch.ones((1, 1, kernel_size, kernel_size), device=device)
    kernel[0, 0, padding, padding] = kernel_size * kernel_size - 1

    loss = 0.0
    for c in range(C):
        uv_c = uv_map[:, c:c+1, :, :]  # [B, 1, H, W]
        diff = F.conv2d(uv_c, kernel, padding=padding)
        loss = loss + (diff**2).mean()

    return loss

def measure_func(model_output, observation, t=None, eps=2e-3):
    isp_mask_bool, isp_mask, sparse_mask, sparse_uv, body_info, mapping_related, consis_info = observation
    
    body_depth_raw, body_depth_back_raw, mask_body_f,  mask_body_b, mask_torsor, mask_torsor_back, vb, nb, fb, mask_full, mask_full_flip, mask_full_512, mask_full_flip_512, mask_cloth_f_512, mask_cloth_b_512, waist_vertex_ids, waist_face_ids, body_mesh_no_arm_vertices = body_info
    faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, cloth_faces, waist_v_id = mapping_related
    
    prev_uv, prev_prev_uv, prev_uv_mask = consis_info

    output_uv = model_output[:,:3].permute(0,2,3,1) # (B, 3, H, W) -> (B, H, W, 3)
    output_mask = model_output[:,3]
    
    # sparce uv loss
    loss_sparse_uv = torch.linalg.norm((output_uv[sparse_mask] - sparse_uv[sparse_mask]), dim=-1).mean() * 0
    
    if prev_uv is not None:
        loss_vel = torch.linalg.norm(output_uv[prev_uv_mask] - prev_uv[prev_uv_mask], dim=-1).mean() * 0.005
    else:
        loss_vel = torch.tensor(0.0).cuda()
        
    if prev_prev_uv is not None:
        loss_acc = torch.linalg.norm(output_uv[prev_uv_mask] - 2*prev_uv[prev_uv_mask] + prev_prev_uv[prev_uv_mask], dim=-1).mean() * 0.005
    else:
        loss_acc = torch.tensor(0.0).cuda()

    # mask loss
    loss_mask = ((output_mask - isp_mask).abs()).mean() * 10
    
    # body depth loss
    if t < 100:
        uv_f = output_uv[0,:,:128].reshape(-1,3)
        uv_b = output_uv[0,:,128:].reshape(-1,3)
    
        verts_f = uv_to_3D_torch(uv_f, faces_f, v_barycentric_f, closest_face_idx_f)
        verts_b = uv_to_3D_torch(uv_b, faces_b, v_barycentric_b, closest_face_idx_b)
        verts_cloth_new = torch.cat((verts_f, verts_b), axis=0)
        
        loss_smooth = torch.tensor(0.0).cuda()
        
        
        with torch.no_grad():
            full_idx_faces_f, _ = get_pix_to_face_v2(verts_cloth_new, cloth_faces, raster)
            verts_cloth_new_flip = verts_cloth_new.clone()
            full_idx_faces_b, _ = get_pix_to_face_v2(verts_cloth_new_flip, cloth_faces[:, [2,1,0]], raster_back)
            full_faces_cloth_f = cloth_faces[full_idx_faces_f]
            full_faces_cloth_b = cloth_faces[full_idx_faces_b]
        full_tri_f = verts_cloth_new[full_faces_cloth_f.reshape(-1)].reshape(-1,3,3)
        full_tri_b = verts_cloth_new[full_faces_cloth_b.reshape(-1)].reshape(-1,3,3)
        full_tri_center_f = full_tri_f.mean(dim=1)
        full_tri_center_b = full_tri_b.mean(dim=1)
        
        idx_x, idx_y = np.where(mask_cloth_f_512>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
        idx_x, idx_y = np.where(mask_cloth_b_512>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_back_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

        idx_x, idx_y = np.where(mask_full_512>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_full_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
        idx_x, idx_y = np.where(mask_full_flip_512>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_full_back_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
        verts_cloth_2D = (transform.transform_points(full_tri_center_f.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_pend = torch.cat((verts_cloth_2D, torch.zeros(verts_cloth_2D.shape[0], verts_cloth_2D.shape[1], 1).cuda()), dim=-1)
        verts_cloth_2D_back = (transform_back.transform_points(full_tri_center_b.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_back_pend = torch.cat((verts_cloth_2D_back, torch.zeros(verts_cloth_2D_back.shape[0], verts_cloth_2D_back.shape[1], 1).cuda()), dim=-1)

        if double_mask:
            loss_cd_2d_0, _ = chamfer_distance_single(idx_mask_pend, verts_cloth_2D_pend)
            loss_cd_2d_1, _ = chamfer_distance_single(verts_cloth_2D_pend, idx_mask_full_pend)
            loss_cd_2d_2, _ = chamfer_distance_single(idx_mask_back_pend, verts_cloth_2D_back_pend)
            loss_cd_2d_3, _ = chamfer_distance_single(verts_cloth_2D_back_pend, idx_mask_full_back_pend)
            loss_cd_2d = loss_cd_2d_0 + loss_cd_2d_1 + loss_cd_2d_2 + loss_cd_2d_3
        else:
            loss_cd_2d_f, _ = chamfer_distance(verts_cloth_2D_pend, idx_mask_pend)
            loss_cd_2d_b, _ = chamfer_distance(verts_cloth_2D_back_pend, idx_mask_back_pend)
            loss_cd_2d = loss_cd_2d_f + loss_cd_2d_b
            
        step = 5./200
        scale = 1 + (200 - t)*step
        loss_cd_2d = loss_cd_2d / 50000*scale #50000*scale
        
        with torch.no_grad():
            idx_faces_f, _ = get_pix_to_face_v2_cam(verts_cloth_new, cloth_faces, raster, camera)
            verts_cloth_new_flip = verts_cloth_new.clone()
            idx_faces_b, _ = get_pix_to_face_v2_cam(verts_cloth_new_flip, cloth_faces[:, [2,1,0]], raster_back, camera_back)
            faces_cloth_f = cloth_faces[idx_faces_f]
            faces_cloth_b = cloth_faces[idx_faces_b]
        tri_f = verts_cloth_new[faces_cloth_f.reshape(-1)].reshape(-1,3,3)
        tri_b = verts_cloth_new[faces_cloth_b.reshape(-1)].reshape(-1,3,3)
        tri_center_f = tri_f.mean(dim=1)
        tri_center_b = tri_b.mean(dim=1)
        
        verts_cloth_2D_sample = transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[0,1]]*(-1)
        verts_cloth_2D_sample = rearrange(verts_cloth_2D_sample.detach(), 'b n t -> b n 1 t')
        
        verts_cloth_2D_back_sample = transform_back.transform_points(tri_center_b.unsqueeze(0))[:,:,[0,1]]*(-1)
        verts_cloth_2D_back_sample = rearrange(verts_cloth_2D_back_sample.detach(), 'b n t -> b n 1 t')
        
        if t==0 and vis:
            tmp_deform_f = trimesh.Trimesh(verts_f.detach().cpu().numpy(), pattern_f.faces, validate=False, process=False)
            tmp_deform_f.export('tmp_deform_f.ply')
            tmp_deform_b = trimesh.Trimesh(verts_b.detach().cpu().numpy(), pattern_b.faces, validate=False, process=False)
            tmp_deform_b.export('tmp_deform_b.ply')
            tmp_mesh = trimesh.Trimesh(verts_cloth_new.detach().cpu().numpy(), cloth_faces.detach().cpu().numpy(), validate=False, process=False) # 
            tmp_mesh.export('tmp_mesh.ply')
        
            save_point_cloud("verts_cloth_2D_sample.ply", verts_cloth_2D_sample.detach())
            save_point_cloud("verts_cloth_2D_back_sample.ply", verts_cloth_2D_back_sample.detach())
            
            z_values = tri_center_f[:, -1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, N, 1, 1)
            verts_cloth_3D = torch.cat([verts_cloth_2D_sample, z_values], dim=-1)  # (1, N, 1, 3)
            save_point_cloud("verts_cloth_3D_f.ply", verts_cloth_3D.detach())
            
            z_values_b = tri_center_b[:, -1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, N, 1)
            verts_cloth_3D_b = torch.cat([verts_cloth_2D_back_sample, z_values_b], dim=-1)  # (1, N, 3)
            save_point_cloud("verts_cloth_3D_b.ply", verts_cloth_3D_b.detach())
        
        depth_body_sample = torch.nn.functional.grid_sample(body_depth_raw.unsqueeze(0), verts_cloth_2D_sample, align_corners=True).squeeze()
        depth_body_back_sample = torch.nn.functional.grid_sample(body_depth_back_raw.unsqueeze(0), verts_cloth_2D_back_sample, align_corners=True).squeeze()
        
        mask_body_sample = torch.nn.functional.grid_sample(mask_body_f.unsqueeze(0).float(), verts_cloth_2D_sample, align_corners=True)
        mask_body_sample = rearrange(mask_body_sample, 'b c n 1 -> b c n').squeeze()
        mask_img_back_sample = torch.nn.functional.grid_sample(mask_body_b.unsqueeze(0).float(), verts_cloth_2D_back_sample, align_corners=True)
        mask_img_back_sample = rearrange(mask_img_back_sample, 'b c n 1 -> b c n').squeeze()
        
        if t == 0 and vis:
            z_values_body = depth_body_sample.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, N, 1, 1)
            print(f"{verts_cloth_2D_sample[mask_body_sample.unsqueeze(0) > 0.99].shape=}")
            print(f"{z_values_body[mask_body_sample.unsqueeze(0) > 0.99].shape=}")
            verts_body_3D = torch.cat([verts_cloth_2D_sample[mask_body_sample.unsqueeze(0) > 0.99], z_values_body[mask_body_sample.unsqueeze(0) > 0.99]], dim=-1)  # (1, N, 1, 3)
            save_point_cloud("verts_body_3D_f.ply", verts_body_3D.detach())
            
            z_values_body_b = depth_body_back_sample.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, N, 1, 1)
            verts_body_3D_b = torch.cat([verts_cloth_2D_back_sample[mask_img_back_sample.unsqueeze(0) > 0.99], z_values_body_b[mask_img_back_sample.unsqueeze(0) > 0.99]], dim=-1)  # (1, N, 1, 3)
            save_point_cloud("verts_body_3D_b.ply", verts_body_3D_b.detach())
        
        loss_depth = torch.tensor(0.0).cuda()
        
        #####################
        # torsor close loss #
        #####################
        faces_cloth_waist = cloth_faces[waist_face_ids]
        tri_waist = verts_cloth_new[faces_cloth_waist.reshape(-1)].reshape(-1,3,3)
        tri_center_waist = tri_waist.mean(dim=1)
        loss_close_torsor, _ = chamfer_distance_single(tri_center_waist.unsqueeze(0), body_mesh_no_arm_vertices.unsqueeze(0))
        loss_close_torsor = loss_close_torsor * 2 # 1 - 2

        loss_full_mask = torch.tensor(0).cuda()
        loss_collision = torch.tensor(0).cuda()
        
    else:
        loss_depth = torch.tensor(0).cuda()
        loss_close_f = torch.tensor(0).cuda()
        loss_close_b = torch.tensor(0).cuda()
        loss_close_torsor = torch.tensor(0).cuda()
        loss_full_mask = torch.tensor(0).cuda()
        loss_cd_2d = torch.tensor(0).cuda()
        loss_smooth = torch.tensor(0).cuda()
        loss_collision = torch.tensor(0).cuda()
        
    
    msg = (
        f"t={t.item()} | "
        f"vel={loss_vel.item():.4f}, acc={loss_acc.item():.4f}, "
        f"mask={loss_mask.item():.4f}, close={loss_close_torsor.item():.4f}, cd2d={loss_cd_2d.item():.4f}"
    )

    print(msg)

    with open(os.path.join(save_folder, "losses.txt"), "a") as f:
        f.write(msg + "\n")

    loss = loss_vel + loss_acc + loss_sparse_uv + loss_mask + loss_depth + loss_close_torsor + loss_full_mask + loss_cd_2d + loss_smooth + loss_collision
    return loss

def get_back_in_body_face(cloth_mesh, body_mesh):
    fn = cloth_mesh.face_normals
    indicator_f_back = fn[:, -1] < -0.5

    vb = torch.FloatTensor(body_pose.vertices).cuda().unsqueeze(0)
    nb = torch.FloatTensor(body_pose.vertex_normals).cuda().unsqueeze(0)

    vc = torch.FloatTensor(cloth_mesh.triangles_center).cuda().unsqueeze(0)

    vec = vc[:, :, None] - vb[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1,1,3)
    vb = torch.gather(vb, 1, closest_vertices)
    nb = torch.gather(nb, 1, closest_vertices)

    distance = (nb*(vc - vb)).sum(dim=-1) 
    idx_coll = (distance < 0).reshape(-1)

    idx_coll = torch.logical_and(torch.BoolTensor(indicator_f_back).cuda(), idx_coll)
    return idx_coll.reshape(-1)


def mask_to_coord(mask):
    x, y = np.where(mask)
    coord = np.stack((x,y), axis=-1)
    return coord

def remove_arm(color_smpl_faces):
    new_faces_id = []
    for i in range(len(color_smpl_faces)):
        if color_smpl_faces[i,0] in [3, 4, 11, 12, 13, 14]:
            continue
        else:
            new_faces_id.append(i)

    return new_faces_id


def clean_sparse_uv(sparse_uv, sparse_mask, nb_neighbors=20, std_ratio=1.0):
    H, W, _ = sparse_uv.shape

    valid_coords = np.argwhere(sparse_mask > 0)  # [N, 2]
    xyz = sparse_uv[valid_coords[:, 0], valid_coords[:, 1]]  # [N, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    ind = np.array(ind)

    valid_coords_clean = valid_coords[ind]  # [M, 2]
    xyz_clean = np.asarray(pcd_clean.points)  # [M, 3]

    clean_uv = np.zeros_like(sparse_uv) - 1
    clean_mask = np.zeros((H, W), dtype=np.uint8)

    clean_uv[valid_coords_clean[:, 0], valid_coords_clean[:, 1]] = xyz_clean
    clean_mask[valid_coords_clean[:, 0], valid_coords_clean[:, 1]] = 1

    return clean_uv, clean_mask

def rescale(cloth_pose_f, cloth_pose_b, altas_f, altas_b):

    ave_area_pose_f = cloth_pose_f.area_faces.mean() 
    ave_area_pose_b = cloth_pose_b.area_faces.mean() 
    ave_area_rest_f = altas_f.area_faces.mean() 
    ave_area_rest_b = altas_b.area_faces.mean() 

    scale = (ave_area_pose_f/ave_area_rest_f + ave_area_pose_b/ave_area_rest_b)/2
    print(ave_area_pose_f, ave_area_rest_f, ave_area_pose_f/ave_area_rest_f)
    print(ave_area_pose_b, ave_area_rest_b, ave_area_pose_b/ave_area_rest_b)
    scale = np.sqrt(scale)
    print(scale)
    return scale

def clean_pt(path, nb_neighbors=10, std_ratio=0.01):
    pcd = o3d.io.read_point_cloud(path)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    vertices = np.asarray(inlier_cloud.points)
    pt = trimesh.Trimesh(vertices)
    return pt, ind

def _process_depth(image, resize=0):
    if resize > 0:
        image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_NEAREST)
    if len(image.shape) != 3:
        image = np.expand_dims(image, axis=-1)
    return image

from scipy.spatial import cKDTree
def fill_background_with_nearest_foreground(background_img, indicator_img):
    # Find the indices of background pixels in the indicator image
    background_indices = np.argwhere(indicator_img == 0)
    
    # Find the indices of foreground pixels in the indicator image
    foreground_indices = np.argwhere(indicator_img != 0)
    
    # Build a KDTree using the foreground pixel indices
    tree = cKDTree(foreground_indices)
    
    # For each background pixel, find the nearest foreground pixel and update its value
    for bg_index in background_indices:
        _, nearest_fg_index = tree.query(bg_index)
        background_img[tuple(bg_index)] = background_img[tuple(foreground_indices[nearest_fg_index])]
    
    return background_img

def move_mask_and_trim(mask, shift_pixels):
    mask1 = mask.copy()

    mask2 = np.zeros_like(mask1)

    mask2[:, 0:-shift_pixels, :] = mask1[:, shift_pixels:, :]

    to_remove = (mask1 > 0) & (mask2 == 0)

    mask1[to_remove] = 0

    return mask1

def get_waist_vertex_ids_by_y(skirt_vertices, percentile=5):
    y_coords = skirt_vertices[:, 1]
    threshold = np.percentile(y_coords, 100 - percentile)
    waist_vertex_ids = np.where(y_coords >= threshold)[0]
    return waist_vertex_ids

def get_waist_face_ids(faces, waist_vertex_ids):
    waist_vertex_ids = set(waist_vertex_ids)
    return [i for i, face in enumerate(faces) if any(v in waist_vertex_ids for v in face)]

def project_waist(body, barycentric, idx_f, eps=1e-3):

    faces_waist = body.faces[idx_f].reshape(-1)
    fn_waist = body.face_normals[idx_f]

    triangles = body.vertices[faces_waist].reshape(-1, 3, 3)
    v_waist = trimesh.triangles.barycentric_to_points(triangles, barycentric)
    v_waist += fn_waist*eps

    return v_waist

def compute_mesh_vertex_distance(mesh1_verts, mesh2_verts):
    diff = mesh1_verts - mesh2_verts
    dists = torch.norm(diff, dim=1)

    avg_dist = dists.mean()
    max_dist = dists.max()

    return avg_dist.item(), max_dist.item()


parser = argparse.ArgumentParser(description="Generate the back normal maps")
parser.add_argument("--garment", type=str, default='Skirt', help="The type of garment")
parser.add_argument("--scale", type=float, default=0.8, help="The scale of the garment")
parser.add_argument("--vid_name", type=str, default='vid_demo', help="The name of the video")
parser.add_argument("--sigma_y", type=float, default=0.02, help="The sigma of the y direction")
args = parser.parse_args()

garment = args.garment
vid_name = args.vid_name
scale = args.scale
sigma_y = args.sigma_y
use_guidance = True
vis = False

normal_dir = f'../data//{vid_name}/results-{scale}'
step1_load_folder = f'../fitting-results/{vid_name}/uv-mapping-back-{scale}-{garment}'
step2_load_folder = f'../fitting-results/{vid_name}/uv-mapping-back-{scale}-{garment}'
load_isp_mask_folder = os.path.join(step2_load_folder, 'mask-all0.1')
save_folder = f'../fitting-results/{vid_name}/uv-mapping-back-inpaint-{scale}-{garment}'
align_dir = f'../data/{vid_name}/results-{scale}'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

pipeline = DDPMPipeline.from_pretrained('../checkpoints/uv-128-full-rotate-shift', use_safetensors=True).to("cuda")
target_label = 240

double_mask = True
resolution = 128
ddpm_num_steps = 1000
ddpm_beta_schedule = 'linear'
eval_batch_size = 1
ddpm_num_inference_steps = 1000
prediction_type = 'epsilon'
weight_dtype = torch.float32

noise_scheduler = DDPMScheduler(num_train_timesteps=ddpm_num_steps, beta_schedule='linear', prediction_type='epsilon')
pipeline.scheduler = noise_scheduler
generator = torch.Generator(device=pipeline.device).manual_seed(0)

img_pred_uvs = sorted([img for img in os.listdir(step1_load_folder) if img.startswith('images_normal_back_') and img.endswith('.png')])

raster, renderer_textured_hard, renderer_soft, transform, camera = get_render(render_res=512)
raster_back, renderer_textured_hard_back, renderer_soft_back, transform_back, camera_back = get_render(render_res=512, direction=1)

color_smpl = np.load('../extra-data/color_smpl_faces.npy')
last_images = None
last_frame = None
prev_frame_ply = None
noisy_images = None
repeat_t = 100 
stride_num = 1
sample_num = 1
use_double_cd = False
use_depth = False

bottom_label = 240
top_label = 60

vertices, faces = create_uv_mesh(128, 128)
faces_f = faces
faces_b = faces

pattern_f_128 = trimesh.Trimesh(vertices, faces_f, valid=False, process=False)
pattern_b_128 = trimesh.Trimesh(vertices, faces_b, valid=False, process=False)
sewing = trimesh.load(os.path.join(load_isp_mask_folder, f'sewing.ply'), validate=False, process=False)

waist_vertex_ids = get_waist_vertex_ids_by_y(sewing.vertices, percentile=10)
waist_face_ids = get_waist_face_ids(sewing.faces, waist_vertex_ids)

with open(os.path.join(save_folder, "losses.txt"), "w") as f:
    f.write("")
    
uv_log_path = os.path.join(save_folder, 'a_uv_log.txt')
with open(uv_log_path, "w") as f:
    f.write("")
idx_f_waist = None

cloth_rest_z_up = apply_rotation(np.pi/2, sewing.copy(), 'x')
cloth_rest_z_up.export(os.path.join(load_isp_mask_folder, 'cloth_rest_z_up.ply'))
waist_v_id = get_connected_paths_skirt(cloth_rest_z_up)[0]

for index, img_pred_uv_name in enumerate(img_pred_uvs):
    frame = img_pred_uv_name.split('_')[-1].split('.')[0] 
    print(frame)       
    if os.path.exists(os.path.join(save_folder, 'mesh_remesh_cut-%s.ply'%frame)):
        print(f"{frame} exists, skip")
        continue
        
    if True:
        if index == 0:
            prev_uv = None
            prev_prev_uv = None
            prev_uv_mask = None
            prev_frame_ply = None
            prev_prev_frame_ply = None
            
            last_diff = None
        elif index == 1:
            prev_frame = img_pred_uvs[index-1].split('_')[-1].split('.')[0]
            if os.path.exists(os.path.join(save_folder, 'uv-updated_%s.png'%prev_frame)):
                prev_uv = cv2.imread(os.path.join(save_folder, 'uv-updated_%s.png'%prev_frame))
            else:
                prev_uv = cv2.imread(os.path.join(save_folder, 'uv-inpaint_%s.png'%prev_frame))
            prev_uv_mask = cv2.imread(os.path.join(save_folder, 'uv-inpaint-mask_%s.png'%prev_frame))
            prev_prev_uv = None
            
            prev_uv = prev_uv / 255.0 * 2 - 1 # [-1, 1] # [H, W, 3]
            prev_uv = torch.FloatTensor(prev_uv).unsqueeze(0).cuda() # [1, H, W, 3]
            prev_uv_mask = prev_uv_mask / 255.0 # [H, W, 3] # [0, 1]
            prev_uv_mask = torch.BoolTensor(prev_uv_mask).unsqueeze(0).cuda() # [1, H, W, 3]
            
            prev_frame_ply = trimesh.load(os.path.join(save_folder, 'deform-inpaint-%s.ply'%prev_frame), validate=False, process=False)
            
            prev_prev_frame_ply = None
            
            last_diff = None
            
        else:
            prev_frame = img_pred_uvs[index-1].split('_')[-1].split('.')[0]
            prev_prev_frame = img_pred_uvs[index-2].split('_')[-1].split('.')[0]
            if os.path.exists(os.path.join(save_folder, 'uv-updated_%s.png'%prev_frame)):
                prev_uv = cv2.imread(os.path.join(save_folder, 'uv-updated_%s.png'%prev_frame))
            else:
                prev_uv = cv2.imread(os.path.join(save_folder, 'uv-inpaint_%s.png'%prev_frame))
            if os.path.exists(os.path.join(save_folder, 'uv-updated_%s.png'%prev_prev_frame)):
                prev_prev_uv = cv2.imread(os.path.join(save_folder, 'uv-updated_%s.png'%prev_prev_frame))
            else:
                prev_prev_uv = cv2.imread(os.path.join(save_folder, 'uv-inpaint_%s.png'%prev_prev_frame))
            prev_uv_mask = cv2.imread(os.path.join(save_folder, 'uv-inpaint-mask_%s.png'%prev_frame))
            
            prev_uv = prev_uv / 255.0 * 2 - 1 # [-1, 1] # [H, W, 3]
            prev_uv = torch.FloatTensor(prev_uv).unsqueeze(0).cuda() # [1, H, W, 3]
            prev_uv_mask = prev_uv_mask / 255.0 # [H, W, 3]
            prev_uv_mask = torch.BoolTensor(prev_uv_mask).unsqueeze(0).cuda() # [1, H, W, 3]
            prev_prev_uv = prev_prev_uv / 255.0 * 2 - 1 # [-1, 1] # [H, W, 3]
            prev_prev_uv = torch.FloatTensor(prev_prev_uv).unsqueeze(0).cuda() # [1, H, W, 3]
            
            prev_frame_ply = trimesh.load(os.path.join(save_folder, 'deform-inpaint-%s.ply'%prev_frame), validate=False, process=False)
            prev_prev_frame_ply = trimesh.load(os.path.join(save_folder, 'deform-inpaint-%s.ply'%prev_prev_frame), validate=False, process=False)
            verts1 = torch.from_numpy(prev_frame_ply.vertices).float()
            verts2 = torch.from_numpy(prev_prev_frame_ply.vertices).float()
            last_diff, _= compute_mesh_vertex_distance(verts1[waist_v_id], verts2[waist_v_id])
        
        load_mask_path = load_isp_mask_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        isp_mask = cv2.imread(os.path.join(load_mask_path, f'isp-fitting.png'))[:, :, 0]/255
        isp_mask = cv2.resize(isp_mask.astype(float), (256,128), interpolation = cv2.INTER_AREA) >= 0.5
        isp_mask = dilate_indicator(isp_mask.astype(np.uint8))
        cv2.imwrite(os.path.join(save_folder, 'isp_mask_resize_%s.png'%frame), isp_mask*255)
        isp_mask_bool = torch.BoolTensor(isp_mask).cuda().unsqueeze(0)
        isp_mask = torch.FloatTensor((isp_mask.astype(float) - 0.5)/0.5).cuda().unsqueeze(0)

        prediction = np.load(os.path.join(step2_load_folder, 'uv_transfer_%s.npz'%frame))
        data_transfer = prediction['uv_transfer'] # [-1, 1]
        uv_transfer_f = data_transfer[:,:,:3]
        uv_transfer_b = data_transfer[:,:,4:4+3]
        depth_transfer_f = data_transfer[:,:,3]
        depth_transfer_b = data_transfer[:,:,7]
        
        mask_cloth_f = cv2.imread(os.path.join(step2_load_folder, 'mask_front_%s.png'%frame))[:,:,0]
        mask_cloth_b = cv2.imread(os.path.join(step1_load_folder, 'images_mask_back_%s.png'%(frame)))[:,:,0]
        
        coord_img_f = mask_to_coord(mask_cloth_f)
        coord_img_b = mask_to_coord(mask_cloth_b)
        mask_cloth_f_512 = cv2.resize(mask_cloth_f, (512,512))
        mask_cloth_b_512 = cv2.resize(mask_cloth_b, (512,512))
        mask_cloth_b_512 = np.fliplr(mask_cloth_b_512)
        mask_cloth_f_512 = (mask_cloth_f_512).astype(np.uint8)
        mask_cloth_b_512 = (mask_cloth_b_512).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)

        mask_cloth_f_eroded = mask_cloth_f_512
        mask_cloth_b_eroded = mask_cloth_b_512
        
        mask_cloth_f_512 = mask_cloth_f_eroded.astype(float)/255
        mask_cloth_b_512 = mask_cloth_b_eroded.astype(float)/255
        
        
        z_f = depth_transfer_f[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
        z_b = depth_transfer_b[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
        xyz_f = _to_xyz(coord_img_f, z_f, img_size=191.).astype(np.float32)
        xyz_b = _to_xyz(coord_img_b, z_b, img_size=191.).astype(np.float32)
        
        xyz = trimesh.load(os.path.join(step2_load_folder, 'xyz_%s.ply'%frame), validate=False, process=False)
        xyz.export(os.path.join(save_folder, 'xyz_%s.ply'%frame))
        
        pattern_f = trimesh.load(os.path.join(load_isp_mask_folder, f'pattern-f.ply'), validate=False, process=False)
        pattern_b = trimesh.load(os.path.join(load_isp_mask_folder, f'pattern-b.ply'), validate=False, process=False)
        sewing = trimesh.load(os.path.join(load_isp_mask_folder, f'sewing.ply'), validate=False, process=False)

        v_barycentric_f, closest_face_idx_f = barycentric_faces(pattern_f, pattern_f_128)
        v_barycentric_b, closest_face_idx_b = barycentric_faces(pattern_b, pattern_b_128)

        sparse_uv, sparse_mask = _to_uv_FB(uv_transfer_f, uv_transfer_b, coord_img_f, coord_img_b, xyz_f, xyz_b, size_uv=128)
        cv2.imwrite(os.path.join(save_folder, 'sparse_uv_%s.png'%(frame)), ((sparse_uv+1)/2*255).astype(np.uint8))
        cv2.imwrite(os.path.join(save_folder, 'sparse_mask_%s.png'%(frame)), ((sparse_mask)*255).astype(np.uint8))
        sparse_uv, sparse_mask = clean_sparse_uv(sparse_uv, sparse_mask, nb_neighbors=10, std_ratio=2.0)
        cv2.imwrite(os.path.join(save_folder, 'clean_sparse_uv_%s.png'%(frame)), ((sparse_uv+1)/2*255).astype(np.uint8))
        cv2.imwrite(os.path.join(save_folder, 'clean_sparse_mask_%s.png'%(frame)), ((sparse_mask)*255).astype(np.uint8))
        
        sparse_mask = torch.BoolTensor(sparse_mask).cuda().unsqueeze(0)
        sparse_uv = torch.FloatTensor(sparse_uv).cuda().unsqueeze(0)
        sparse_mask = torch.logical_and(sparse_mask, isp_mask_bool)
        sparse_uv = sparse_uv * isp_mask_bool.unsqueeze(-1)
        
        body_mesh_no_arm = trimesh.load(os.path.join(step2_load_folder, 'body_smpl_no_arm_%s.ply'%frame))
        body_depth_raw = render_depth_discrete(body_mesh_no_arm, renderer_textured_hard, raster)
        body_depth_back_raw = render_depth_discrete(body_mesh_no_arm, renderer_textured_hard_back, raster_back)
        body_depth_raw = _process_depth(body_depth_raw, resize=512)
        body_depth_back_raw = _process_depth(body_depth_back_raw, resize=512)
        body_depth_raw = torch.FloatTensor(body_depth_raw[...,0]).unsqueeze(0).cuda()
        body_depth_back_raw = torch.FloatTensor(body_depth_back_raw[...,0]).unsqueeze(0).cuda()
        
        # body
        mask_body_f = torch.logical_and(body_depth_raw != 1, body_depth_raw != -1)
        mask_body_b = torch.logical_and(body_depth_back_raw != 1, body_depth_back_raw != -1)
        
        # body-torsor depth
        body_smpl = trimesh.load(os.path.join(step2_load_folder, 'body_%s.ply'%frame))
        mask_torsor = render_torsor(body_smpl, renderer_textured_hard, raster).astype(np.uint8)[:,:,0]
        mask_torsor_back = render_torsor(body_smpl, renderer_textured_hard_back, raster_back).astype(np.uint8)[:,:,0]
        mask_torsor = _process_depth(mask_torsor, resize=512)
        mask_torsor_back = _process_depth(mask_torsor_back, resize=512)
        
        mask_torsor = torch.BoolTensor(mask_torsor[...,0]).unsqueeze(0).cuda()
        mask_torsor_back = torch.BoolTensor(mask_torsor_back[...,0]).unsqueeze(0).cuda()
        
        # garment mask
        mask_front = cv2.imread(os.path.join(step2_load_folder, 'mask_front_%s.png'%frame))[:,:,0]/255
        mask_back = cv2.imread(os.path.join(step1_load_folder, 'images_mask_back_%s.png'%(frame)))[:,:,0]/255
        mask_front = torch.BoolTensor(mask_front).unsqueeze(0).cuda()
        mask_back = torch.BoolTensor(mask_back).unsqueeze(0).cuda()
        
        mask_full = cv2.imread(os.path.join(normal_dir, '%s_mask_full_align.png'%frame))[:,:,0]/255
        mask_full_flip = np.fliplr(mask_full)
        
        mask_full_512 = cv2.resize(mask_full, (512, 512))
        mask_full_flip_512 = cv2.resize(mask_full_flip, (512, 512))
        mask_full_512 = (mask_full_512 * 255).astype(np.uint8)
        mask_full_flip_512 = (mask_full_flip_512 * 255).astype(np.uint8)
        mask_full_eroded = mask_full_512
        mask_full_flip_eroded = mask_full_flip_512
        mask_full_512 = mask_full_eroded.astype(float) / 255.0
        mask_full_flip_512 = mask_full_flip_eroded.astype(float) / 255.0
        
        body_info = [body_depth_raw, body_depth_back_raw, mask_body_f,  mask_body_b, mask_torsor, mask_torsor_back, torch.FloatTensor(body_smpl.vertices).cuda(), torch.FloatTensor(body_smpl.vertex_normals).cuda(), torch.LongTensor(body_smpl.faces).cuda(), torch.FloatTensor((mask_full) > 0).cuda(), torch.FloatTensor((mask_full_flip) > 0).cuda(), mask_full_512, mask_full_flip_512, mask_cloth_f_512, mask_cloth_b_512, waist_vertex_ids, waist_face_ids, torch.FloatTensor(body_mesh_no_arm.vertices).cuda()]
        
        mapping_related = [torch.LongTensor(faces_f).cuda(), torch.LongTensor(faces_b).cuda(), torch.FloatTensor(v_barycentric_f).cuda(), torch.FloatTensor(v_barycentric_b).cuda(), torch.LongTensor(closest_face_idx_f).cuda(), torch.LongTensor(closest_face_idx_b).cuda(), torch.LongTensor(sewing.faces).cuda(), waist_v_id]
        
        consis_info = [prev_uv, prev_prev_uv, prev_uv_mask]

        current_frame = int(frame)
        if last_frame is not None:
            print("use last frame to initialize current noise")
            noisy_images = torch.from_numpy(last_images).float().permute(0, 3, 1, 2).cuda() # [B, C, H, W]
        else:
            noisy_images = None
            
        start_t = ddpm_num_steps-1
        observation = [isp_mask_bool, isp_mask, sparse_mask, sparse_uv, body_info, mapping_related, consis_info]
        
        y = torch.cat([sparse_uv.permute(0, 3, 1, 2), isp_mask_bool.unsqueeze(0)], dim=1).cuda()
        A_matrix = torch.cat([sparse_mask.unsqueeze(0), sparse_mask.unsqueeze(0), sparse_mask.unsqueeze(0), isp_mask_bool.unsqueeze(0)], dim=1).cuda()
        constraint = [y, lambda z: z*A_matrix, lambda z: z*A_matrix, sigma_y] # (y, A, Ap)
        
        cur_diff = None
        cnt = 0
        
        while (cnt == 0) or (last_diff is not None and cur_diff is not None and cur_diff>last_diff*1.5):
            if cnt > 0:
                cur_str = f"{cur_diff:.6f}" if cur_diff is not None else "None"
                last_str = f"{last_diff:.6f}" if last_diff is not None else "None"
                msg = f"repeat generation: {frame} {cnt} {cur_str} {last_str}\n"
                with open(uv_log_path, "a") as f:
                    f.write(msg)
                print(msg)

            if cnt >= 5:
                cur_str = f"{cur_diff:.6f}" if cur_diff is not None else "None"
                last_str = f"{last_diff:.6f}" if last_diff is not None else "None"
                msg = f"break generation: {frame} {cnt} {cur_str} {last_str}\n"
                with open(uv_log_path, "a") as f:
                    f.write(msg)
                print(msg)
                break
                    
            images = pipeline(
                measure_func=measure_func,
                observation=observation,
                constraint=constraint,
                guide_scale=20.0,#20.0,#40.0,
                generator=torch.Generator(device=pipeline.device).manual_seed(cnt),
                batch_size=eval_batch_size,
                num_inference_steps=ddpm_num_inference_steps,
                output_type="numpy",
                start_image=noisy_images,
                use_guidance=use_guidance,
                use_constraint=True,
                start_t=start_t,
                use_temporal=True,
                repeat_t = repeat_t, 
                stride_num = stride_num, 
                sample_num = sample_num,
            ).images
            cnt += 1
        
            last_images = images.copy()
            last_frame = current_frame

            images_processed_uv = images[:,:,:,:3].copy()
            gen_images_processed_mask = images[:,:,:,-1].copy()
            gen_images_processed_mask[gen_images_processed_mask>0.5] = 1
            gen_images_processed_mask[gen_images_processed_mask<0.5] = 0
            cv2.imwrite(os.path.join(save_folder, 'gen-uv-inpaint_%s.png'%frame), ((images_processed_uv / 2 + 0.5)[0] * 255).astype("uint8"))
            cv2.imwrite(os.path.join(save_folder, 'gen-uv-inpaint-mask_%s.png'%frame), (gen_images_processed_mask[0]* 255).astype("uint8"))
            
            
            images_processed_mask = (isp_mask.cpu().numpy()+1)/2
            images_processed_mask[images_processed_mask>0.5] = 1
            images_processed_mask[images_processed_mask<0.5] = 0

            vertices, faces = create_uv_mesh(128, 128)
            
            uv_f = images_processed_uv[0,:,:128]
            uv_b = images_processed_uv[0,:,128:]
            mask_f = images_processed_mask[0,:,:128]
            mask_b = images_processed_mask[0,:,128:]

            uv_f = fill_background_with_nearest_foreground(uv_f, mask_f)
            uv_b = fill_background_with_nearest_foreground(uv_b, mask_b)
            images_processed_uv = np.concatenate((uv_f, uv_b), axis=1).reshape(1, 128,256, 3)
            images[:,:,:,:3] = images_processed_uv
            faces_f = faces
            faces_b = faces

            uv_f = images_processed_uv[0, :,:128].reshape(-1, 3)
            uv_b = images_processed_uv[0, :,128:].reshape(-1, 3)

            np.save(os.path.join(save_folder, 'uv-inpaint_%s.npy'%frame), images_processed_uv[0])
            images_processed_uv = (images_processed_uv / 2 + 0.5)
            images_processed_uv = (images_processed_uv[0] * 255).round().astype("uint8")
            cv2.imwrite(os.path.join(save_folder, 'uv-inpaint_%s.png'%frame), images_processed_uv)
            cv2.imwrite(os.path.join(save_folder, 'uv-inpaint-mask_%s.png'%frame), (images_processed_mask[0]* 255).astype("uint8"))

            verts_f = uv_to_3D(uv_f, faces_f, v_barycentric_f, closest_face_idx_f)
            verts_b = uv_to_3D(uv_b, faces_b, v_barycentric_b, closest_face_idx_b)
            verts = np.concatenate((verts_f, verts_b), axis=0)

            deform = trimesh.Trimesh(verts, sewing.faces, validate=False, process=False)
            deform_f = trimesh.Trimesh(verts_f, pattern_f.faces, validate=False, process=False)
            deform_b = trimesh.Trimesh(verts_b, pattern_b.faces, validate=False, process=False)
            
            if prev_frame_ply is not None:
                verts_last = torch.from_numpy(prev_frame_ply.vertices).float()
                verts_curr = torch.from_numpy(deform.vertices).float()
                cur_diff, _= compute_mesh_vertex_distance(verts_last[waist_v_id], verts_curr[waist_v_id])
            
        deform.export(os.path.join(save_folder, 'deform-inpaint-%s.ply'%frame))
        deform_f.export(os.path.join(save_folder, 'deform-f-inpaint-%s.ply'%frame))
        deform_b.export(os.path.join(save_folder, 'deform-b-inpaint-%s.ply'%frame))
        
        np.savez(os.path.join(save_folder, 'barycentric-%s'%frame), v_barycentric_f=v_barycentric_f, v_barycentric_b=v_barycentric_b, closest_face_idx_f=closest_face_idx_f, closest_face_idx_b=closest_face_idx_b, faces_f=faces_f, faces_b=faces_b)
        
        cloth_rest = trimesh.load(os.path.join(load_isp_mask_folder, 'sewing.ply'), validate=False, process=False)
        altas_f = trimesh.load(os.path.join(load_isp_mask_folder, 'atlas-f.ply'), validate=False, process=False)
        altas_b = trimesh.load(os.path.join(load_isp_mask_folder, 'atlas-b.ply'), validate=False, process=False)
        cloth_pose = trimesh.load(os.path.join(save_folder, 'deform-inpaint-%s.ply'%frame), validate=False, process=False)
        cloth_pose_f = trimesh.load(os.path.join(save_folder, 'deform-f-inpaint-%s.ply'%(frame)), validate=False, process=False)
        cloth_pose_b = trimesh.load(os.path.join(save_folder, 'deform-b-inpaint-%s.ply'%(frame)), validate=False, process=False)
        mesh_scale = rescale(cloth_pose_f, cloth_pose_b, altas_f, altas_b)
        material = Material()
        cloth_state = Cloth_from_NP(sewing.vertices*mesh_scale, sewing.faces, material)
        cloth_pose_pt = trimesh.load(os.path.join(save_folder, 'deform-inpaint-%s.ply'%frame), process=False, validate=False)
        
        
        uv_inpaint = np.load(os.path.join(save_folder, 'uv-inpaint_%s.npy'%(frame))) # (128, 256, 3)
        image = np.transpose(uv_inpaint, (2,0,1))
        image = torch.FloatTensor(image).cuda().unsqueeze(0) # [1, 3, 128, 256]
        
        uv_f = torch.FloatTensor(uv_inpaint[:,:128].reshape(-1, 3)).cuda()
        uv_b = torch.FloatTensor(uv_inpaint[:,128:].reshape(-1, 3)).cuda()
        
        mapping_related_remesh = [torch.LongTensor(faces_f).cuda(), torch.LongTensor(faces_b).cuda(), torch.FloatTensor(v_barycentric_f).cuda(), torch.FloatTensor(v_barycentric_b).cuda(), torch.LongTensor(closest_face_idx_f).cuda(), torch.LongTensor(closest_face_idx_b).cuda(), torch.LongTensor(sewing.faces).cuda()]
        
        
        x_res = y_res = 128
        num_v_f = len(pattern_f.vertices)
        uv_vertices, uv_faces = create_uv_mesh(x_res, y_res, debug=False)
        barycentric_front, idx_faces_front = get_barycentric(pattern_f, uv_vertices)
        barycentric_back, idx_faces_back = get_barycentric(pattern_b, uv_vertices)
        faces_front = pattern_f.faces
        faces_back = pattern_b.faces + num_v_f
        
        triangles_front = cloth_rest.vertices[faces_front[idx_faces_front]]
        triangles_back = cloth_rest.vertices[faces_back[idx_faces_back]]
        bary_f = (triangles_front * barycentric_front[:, :, None]).sum(axis=-2)
        bary_b = (triangles_back * barycentric_back[:, :, None]).sum(axis=-2)
        uv_size = 128
        bary_f = bary_f.reshape(uv_size, uv_size, 3)
        bary_b = bary_b.reshape(uv_size, uv_size, 3)
        image_rest = np.concatenate((bary_f, bary_b), axis=1)
        image_rest = torch.FloatTensor(image_rest).unsqueeze(0).permute(0,3,1,2).cuda()
        
        std_ratio = 2
        xyz, idx = clean_pt(os.path.join(step2_load_folder, 'xyz_%s.ply'%(frame)), nb_neighbors=10, std_ratio=std_ratio)
        xyz = trimesh.load(os.path.join(step2_load_folder, 'xyz_%s.ply'%(frame)))
        n_xyz = np.load(os.path.join(step2_load_folder, 'n_%s.npy'%(frame)))
        n_xyz = n_xyz[idx]
        normal_front = cv2.imread(os.path.join(normal_dir, '%s_normal_align.png'%frame))[:,:,::-1].copy()
        normal_back = cv2.imread(os.path.join(step1_load_folder, 'images_normal_back_%s.png'%(frame)))[:,:,::-1].copy()
        normal_back = cv2.resize(normal_back, (512, 512))
        seg = cv2.imread(os.path.join(normal_dir, '%s_seg_align.png'%frame))[:,:,0]
        mask_bottom = ((seg == bottom_label).astype(np.uint8))
        mask_top = ((seg == top_label).astype(np.uint8))
        masks = [mask_bottom, mask_top]
        mask_back = cv2.imread(os.path.join(step1_load_folder, 'images_mask_back_%s.png'%(frame)))[:,:,0]
        mask_back = (cv2.resize(mask_back, (512, 512)) > 122).astype(np.uint8)
        
        normals = [normal_front, normal_back, mask_bottom, mask_back, mask_top, n_xyz]
        clothed_mesh = body_smpl
        
        cloth_rest_z_up = apply_rotation(np.pi/2, cloth_rest.copy(), 'x')
        cloth_rest_z_up.export(os.path.join(save_folder, 'cloth_rest_z_up_%s.ply'%(frame)))
        waist_v_id = get_connected_paths_skirt(cloth_rest_z_up)[0]
        
        prediction = np.load(os.path.join(step2_load_folder, 'uv_transfer_%s.npz'%(frame)))
        data_transfer = prediction['uv_transfer'] # [-1, 1]
        depth_front = data_transfer[:,:,3]
        depth_back = data_transfer[:,:,7]
        mask_depth_front = cv2.imread(os.path.join(step2_load_folder, 'mask_front_%s.png'%frame))[:,:,0]/255
        mask_depth_back = cv2.imread(os.path.join(step1_load_folder, 'images_mask_back_%s.png'%(frame)))[:,:,0]/255
        depth_front = cv2.resize(depth_front, (512, 512))
        depth_back = cv2.resize(depth_back, (512, 512))
        mask_depth_front = (cv2.resize(mask_depth_front*255, (512, 512)) == 255).astype(np.uint8)
        mask_depth_back = (cv2.resize(mask_depth_back*255, (512, 512)) == 255).astype(np.uint8)
        coord_img_f = mask_to_coord(mask_depth_front)
        coord_img_b = mask_to_coord(mask_depth_back)
        
        depths = [depth_front, mask_depth_front]
        
        mask_fulls = [torch.FloatTensor((mask_full) > 0).cuda(), torch.FloatTensor((mask_full_flip) > 0).cuda()]
        
        z_f = depth_front[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
        z_b = depth_back[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
        xyz_f = _to_xyz(coord_img_f, z_f, img_size=511.).astype(np.float32)
        xyz_b = _to_xyz(coord_img_b, z_b, img_size=511.).astype(np.float32)

        xyz = np.concatenate((xyz_f, xyz.vertices), axis=0)


        xyz = trimesh.PointCloud(xyz)
        xyz.export('../tmp/xyz.ply')
        
        xyz = xyz.vertices
        xyz = torch.FloatTensor(xyz).cuda().unsqueeze(0)
        xyz_f = torch.FloatTensor(xyz_f).cuda().unsqueeze(0)
        xyz_b = torch.FloatTensor(xyz_b).cuda().unsqueeze(0)
        
        barycentric = np.load(os.path.join(save_folder, 'barycentric-%s.npz'%(frame)))
        v_barycentric_f = barycentric['v_barycentric_f']
        v_barycentric_b = barycentric['v_barycentric_b']
        closest_face_idx_f = barycentric['closest_face_idx_f']
        closest_face_idx_b = barycentric['closest_face_idx_b']
        faces_f = barycentric['faces_f']
        faces_b = barycentric['faces_b']
        
        
        mapping_related_uv = [faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz, xyz_f, xyz_b]
        
        barycentric_waist, idx_f_waist = get_barycentric(body_mesh_no_arm, cloth_pose.vertices[waist_v_id])
        
        vertices_waist = project_waist(body_mesh_no_arm, barycentric_waist, idx_f_waist)
        
        cloth_pose_uv, uv_new, cloth_pose_pt, uv_pt_new, img_mask = align_observation_uv(image, cloth_pose, cloth_state, mapping_related_uv, image_rest, masks, normals, body_smpl, clothed_mesh, waist_v_id, depths, renderer_soft, renderer_soft_back, mask_fulls, consis_info, uv_log_path, vertices_waist)
        
        uv_pt_new = uv_pt_new[:,:3].squeeze().permute(1,2,0).detach().cpu().numpy()
        np.save(os.path.join(save_folder, 'uv-align-uv_%s.npy'%frame), uv_pt_new)
        cloth_pose_uv.export(os.path.join(save_folder, 'mesh-uv-%s.ply'%(frame)))
        cloth_pose_pt.export(os.path.join(save_folder, 'mesh-uv-cd-%s.ply'%(frame)))
        img_mask = seg/2 + img_mask/2
        cv2.imwrite(os.path.join(save_folder, 'mask-overlay-uv-%s.png'%(frame)), img_mask)
        
        cloth_pose_pt = trimesh.load(os.path.join(save_folder, 'mesh-uv-cd-%s.ply'%(frame)), process=False, validate=False)
        uv_pt = np.load(os.path.join(save_folder, 'uv-align-uv_%s.npy'%frame))
        uv_pt = np.transpose(uv_pt, (2,0,1))
        uv_pt = torch.FloatTensor(uv_pt).cuda().unsqueeze(0)
        
        cloth_pose_pt_new, uv_pt_new, img_mask = align_observation_pt_verts(uv_pt, cloth_pose_pt, cloth_state, body_smpl, normals, vertices_waist, waist_v_id, depths, renderer_soft, renderer_soft_back, consis_info, mapping_related_uv, uv_log_path)
        img_mask = seg/2 + img_mask/2
        cloth_pose_pt_new.export(os.path.join(save_folder, 'mesh_pt_verts_%s.ply'%(frame)))
        cv2.imwrite(os.path.join(save_folder, 'mask_overlay_pt_verts_%s.png'%(frame)), img_mask)
        uv_pt_new = uv_pt_new[:,:3].squeeze().permute(1,2,0).detach().cpu().numpy() # [128, 256, 3]
        np.save(os.path.join(save_folder, 'uv-align-pt-verts_%s.npy'%frame), uv_pt_new)
        
        
        cloth_pose_pt = trimesh.load(os.path.join(save_folder, 'mesh_pt_verts_%s.ply'%(frame)), process=False, validate=False)
        
        uv_est = np.load(os.path.join(save_folder, 'uv-align-pt-verts_%s.npy'%frame)) # (128, 256, 3)
        uv_est = torch.FloatTensor(uv_est).cuda()
        uv_f = uv_est[:,:128].reshape(-1,3)
        uv_b = uv_est[:,128:].reshape(-1,3)
        
        
        cloth_pose_remseh, uv_f_new, uv_b_new = remesh_uv(uv_f, uv_b, cloth_pose_pt, mapping_related_remesh, cloth_state, body_mesh_no_arm, consis_info, uv_log_path)
        
        cloth_pose_remseh.export(os.path.join(save_folder, 'mesh_remesh_%s.ply'%(frame)))
        
        H = uv_inpaint.shape[0]
        W = uv_inpaint.shape[1]

        uv_f_img = uv_f_new.reshape(H, W // 2, 3)
        uv_b_img = uv_b_new.reshape(H, W // 2, 3)

        combined_uv_img = np.concatenate([uv_f_img, uv_b_img], axis=1)  # shape: [H, W, 3]
        combined_uv_img = combined_uv_img * 0.5 + 0.5
        print(f"combined_uv_img.shape: {combined_uv_img.shape}")
        print(f"{combined_uv_img.max()=} {combined_uv_img.min()=}")

        combined_uv_img = np.clip(combined_uv_img, 0.0, 1.0)
        combined_uv_img = (combined_uv_img * 255).round().astype(np.uint8)
        cv2.imwrite(os.path.join(save_folder, f'uv-updated_{frame}.png'), combined_uv_img)
        
        isp_mask = (isp_mask.cpu().numpy()+1)/2
        
        images_processed_mask_new = isp_mask
        images_processed_mask_new[images_processed_mask_new>0.5] = 1
        images_processed_mask_new[images_processed_mask_new<0.5] = 0
        mask_f_new = images_processed_mask_new[0,:,:128]
        mask_b_new = images_processed_mask_new[0,:,128:]
        
        uv_f_img = fill_background_with_nearest_foreground(uv_f_img, mask_f_new)
        uv_b_img = fill_background_with_nearest_foreground(uv_b_img, mask_b_new)
        images_processed_uv_new = np.concatenate((uv_f_img, uv_b_img), axis=1).reshape(1, 128,256, 3)

        uv_f_img = images_processed_uv_new[0, :,:128].reshape(-1, 3)
        uv_b_img = images_processed_uv_new[0, :,128:].reshape(-1, 3)
        
        np.save(os.path.join(save_folder, 'uv-updated-cut_%s.npy'%frame), images_processed_uv_new[0])
        images_processed_uv_new = (images_processed_uv_new / 2 + 0.5)
        images_processed_uv_new = (images_processed_uv_new[0] * 255).round().astype("uint8")
        cv2.imwrite(os.path.join(save_folder, 'uv-updated-cut_%s.png'%frame), images_processed_uv_new)
        cv2.imwrite(os.path.join(save_folder, 'uv-updated-mask-cut_%s.png'%frame), (images_processed_mask_new[0]* 255).astype("uint8"))

        verts_f = uv_to_3D(uv_f_img, faces_f, v_barycentric_f, closest_face_idx_f)
        verts_b = uv_to_3D(uv_b_img, faces_b, v_barycentric_b, closest_face_idx_b)
        verts = np.concatenate((verts_f, verts_b), axis=0)

        deform = trimesh.Trimesh(verts, sewing.faces, validate=False, process=False)
        deform_f = trimesh.Trimesh(verts_f, pattern_f.faces, validate=False, process=False)
        deform_b = trimesh.Trimesh(verts_b, pattern_b.faces, validate=False, process=False)
        deform.export(os.path.join(save_folder, 'mesh_remesh_cut-%s.ply'%frame))
        deform_f.export(os.path.join(save_folder, 'mesh_remesh_f_cut-%s.ply'%frame))
        deform_b.export(os.path.join(save_folder, 'mesh_remesh_b_cut-%s.ply'%frame))