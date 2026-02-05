import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay
import diffusers
from diffusers import DDPMScheduler, UNet2DModel
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss import mesh_laplacian_smoothing
from einops import rearrange
from diffusers import UNet2DModel
import open3d as o3d
import time
import shutil
import argparse

sys.path.append('..')
from utils.isp import create_uv_mesh, get_barycentric
from utils.cutting import get_connected_paths_skirt, select_boundary, get_connected_paths_sleeves
from utils.mesh import apply_rotation
from utils.readfile import load_pkl
from snug.snug_helper import stretching_energy, bending_energy, gravitational_energy, collision_penalty_lite, spring_energy, inertial_term_sequence
from snug.snug_class import Cloth_from_NP, Material
from utils.rasterize import get_pix_to_face_with_body, get_pix_to_face_v2, get_raster
from utils.chamfer import chamfer_distance_single, chamfer_distance
from networks.unet import UNet
from networks.SDF import SDF

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

def get_render(render_res=256, faces_per_pixel=1, is_back=False):
    render_res = render_res
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
    transform = cameras.get_full_projection_transform()

    sigma = 1e-7

    raster_settings_hard = RasterizationSettings(
        image_size=render_res, 
        blur_radius=np.log(1. / 1e-4)*sigma, 
        faces_per_pixel=faces_per_pixel, 
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

    return meshRas_hard, transform, renderer_textured_hard, renderer_textured_soft

def dilate_indicator(mask, size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    mask = cv2.dilate(mask, kernel)
    return mask

def _to_xyz(coord_img, z, img_size=255.):
    scale = img_size/2
    yx = (coord_img - scale)/scale
    y, x = -yx[:,0], yx[:,1]

    xyz = np.stack((x,y,z), axis=-1)
    return xyz

def erode_indicator(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel)
    return mask
def mask_to_coord(mask):
    x, y = np.where(mask)
    coord = np.stack((x,y), axis=-1)
    return coord

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

def barycentric_faces(mesh_query, mesh_base):
    v_query = mesh_query.vertices
    base = trimesh.proximity.ProximityQuery(mesh_base)
    closest_pt, _, closest_face_idx = base.on_surface(v_query)
    triangles = mesh_base.triangles[closest_face_idx]
    v_barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_pt)
    return v_barycentric, closest_face_idx

def uv_to_3D(pattern_deform, uv_faces, barycentric_uv, closest_face_idx_uv):
    uv_faces_id = uv_faces[closest_face_idx_uv]
    uv_faces_id = uv_faces_id.reshape(-1)

    pattern_deform_triangles = pattern_deform[uv_faces_id].reshape(-1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv[:, :, None]).sum(axis=-2)
    return pattern_deform_bary

def uv_to_3D_torch(pattern_deform, uv_faces, barycentric_uv, closest_face_idx_uv):
    uv_faces_id = uv_faces[closest_face_idx_uv]
    uv_faces_id = uv_faces_id.reshape(-1)

    pattern_deform_triangles = pattern_deform[uv_faces_id].reshape(-1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv[:, :, None]).sum(dim=-2)
    return pattern_deform_bary

def get_mapping_to_ori(mapping_to_reorder_front, mapping_to_reorder_back, num_v):
    
    idx_front = np.zeros((num_v)) - 1
    idx_back = np.zeros((num_v)) - 1
    for i in range(num_v):
        if i in mapping_to_reorder_front:
            idx_front[i] = mapping_to_reorder_front[i]
        if i in mapping_to_reorder_back:
            idx_back[i] = mapping_to_reorder_back[i]

    idx_front = idx_front.astype(int)
    idx_back = idx_back.astype(int)
    weight_front = np.ones((num_v))
    weight_back = np.ones((num_v))

    weight_front[idx_front==-1] = 0
    weight_back[idx_back==-1] = 0

    weight = weight_front + weight_back
    overlap = weight == 2
    weight_front[overlap] = 0.5
    weight_back[overlap] = 0.5

    return idx_front, idx_back, weight_front, weight_back

def normalize(img):
    img = torch.FloatTensor(img).cuda()
    img = img/img.norm(p=2, dim=-1, keepdim=True)
    img = img.detach().unsqueeze(0).permute(0,3,1,2)
    img[torch.isnan(img)] = 0
    return img

def normal_loss(verts_cloth, faces_cloth, raster, transform, normal_img, mask, is_back=False):
    with torch.no_grad():
        verts_cloth_tmp = verts_cloth.clone()
        if is_back:
            verts_cloth_tmp[:,-1] *= -1
        idx_faces, idx_vertices = get_pix_to_face_v2(verts_cloth_tmp, faces_cloth, raster)
        faces = faces_cloth[idx_faces]
    tri = verts_cloth[faces.reshape(-1)].reshape(-1,3,3)
    tri_center = tri.mean(dim=1)
    vectors = tri[:,1:] - tri[:,:2]
    normal = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
    normal = normal/normal.norm(p=2, dim=-1, keepdim=True)

    verts_cloth_2D = (transform.transform_points(tri_center.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
    verts_cloth_2D_pend = torch.cat((verts_cloth_2D, torch.zeros(verts_cloth_2D.shape[0], verts_cloth_2D.shape[1], 1).cuda()), dim=-1)

    verts_cloth_2D_sample = transform.transform_points(tri_center.unsqueeze(0))[:,:,[0,1]]*(-1)
    verts_cloth_2D_sample = rearrange(verts_cloth_2D_sample.detach(), 'b n t -> b n 1 t')

    mask_normal_sample = torch.nn.functional.grid_sample(mask.unsqueeze(0).unsqueeze(0), verts_cloth_2D_sample, align_corners=True)
    mask_normal_sample = rearrange(mask_normal_sample, 'b c n 1 -> b c n').squeeze()

    normal_img_sample = torch.nn.functional.grid_sample(normal_img, verts_cloth_2D_sample, align_corners=True)
    normal_img_sample = rearrange(normal_img_sample, 'b c n 1 -> b n c').squeeze()

    loss = (1 - F.cosine_similarity(normal[mask_normal_sample > 0.99], normal_img_sample[mask_normal_sample > 0.99], dim=-1).abs()).mean()

    return loss


def align_observation_uv(img_init, cloth_pose, cloth_state, mapping_related, image_rest, masks, normals, body_mesh, clothed_mesh, waist_v_id, depths):

    vertices_waist = torch.FloatTensor(cloth_pose.vertices[waist_v_id]).cuda()

    depth_img, mask_depth = depths
    depth_img = torch.FloatTensor(depth_img).cuda()
    mask_depth = torch.FloatTensor(mask_depth).cuda()

    faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz = mapping_related

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


    normal_front, normal_back, mask_front, mask_back, mask_top, n_xyz, mask_full = normals
    mask_front = torch.FloatTensor(mask_front).cuda()
    mask_back = torch.FloatTensor(mask_back).cuda()
    mask_top = torch.FloatTensor(mask_top).cuda()
    n_xyz = torch.FloatTensor(n_xyz).cuda().unsqueeze(0)
    mask_full = torch.FloatTensor(mask_full).cuda()
    

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

        idx_x, idx_y = np.where(mask_full.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_full = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

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

    iters = 5 if debug else 1500
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
            images_predicted = renderer_textured_soft(new_src_mesh)
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
        
        loss_cd_2d_f_0, loss_normal_f = chamfer_distance_single(idx_mask_pend_f, verts_cloth_2D_pend_f, x_normals=normal_img_f, y_normals=normal_f)
        loss_cd_2d_b_0, loss_normal_b = chamfer_distance_single(idx_mask_pend_b, verts_cloth_2D_pend_b, x_normals=normal_img_b, y_normals=normal_b)
        loss_cd_2d_f_1, _ = chamfer_distance_single(verts_cloth_2D_pend_f, idx_mask_pend_full)
        loss_cd_2d_b_1, _ = chamfer_distance_single(verts_cloth_2D_pend_b, idx_mask_pend_full)
        loss_mask = (loss_cd_2d_f_0 + loss_cd_2d_f_1 + loss_cd_2d_b_0 + loss_cd_2d_b_1)/5
        loss_normal = (loss_normal_f + loss_normal_b)*2

        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_3d, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_3d*2
        if i < 500:
            loss_cd_3d *= 100*5*0
        else:
            loss_cd_3d *= 100*50*2#10

        loss_collision = collision_penalty_lite(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps) * colli_weight
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

        loss = (loss_bending + loss_strain/4 + loss_gravity) + loss_sparse_uv + loss_cd_3d + loss_mask + loss_normal + loss_collision + loss_edge + loss_depth
        print('align-uv iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_sparse_uv: %0.4f , loss_mask: %0.4f , loss_cd_3d: %0.4f , loss_normal: %0.4f , loss_collision: %0.4f , loss_edge: %0.4f , loss_depth: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_sparse_uv.item(), loss_mask.item(), loss_cd_3d.item(), loss_normal.item(), loss_collision.item(), loss_edge.item(), loss_depth.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target_i = 3 if debug else 499
        if i == target_i:
            verts_cloth_new = verts_cloth_new.detach().cpu().numpy()
            cloth_pose_uv = cloth_pose.copy()
            cloth_pose_uv.vertices = verts_cloth_new
            image_est_uv = image_est.detach().clone()


    verts_cloth_new = verts_cloth_new.detach().cpu().numpy()
    cloth_pose_remesh = cloth_pose.copy()
    cloth_pose_remesh.vertices = verts_cloth_new
    image_est_remesh = image_est.detach().clone()

    return cloth_pose_uv, image_est_uv, cloth_pose_remesh, image_est_remesh, img_mask


def align_observation_pt_ori(cloth_pose, cloth_state, mapping_related, masks, normals, vertices_waist, waist_v_id):

    faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz = mapping_related
    
    idx_boundary_v, _ = select_boundary(cloth_pose)
    

    with torch.no_grad():
        faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
        verts_cloth_zero = torch.FloatTensor(cloth_pose.vertices).cuda()*0
        cloth_rgb = torch.zeros(len(verts_cloth_zero), 3) + 255 # (1, V, 3)
        verts_rgb = cloth_rgb[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        mask_bottom, mask_top = masks
        mask = torch.FloatTensor((mask_bottom+mask_top) > 0).cuda()
        mask_top = torch.FloatTensor(mask_top > 0).cuda()

        
        idx_x, idx_y = np.where(mask_bottom>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

        idx_x, idx_y = np.where((mask_bottom+mask_top.cpu().numpy())>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_full_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

    verts_pose = torch.FloatTensor(cloth_pose.vertices).cuda()
    verts_rest = cloth_state.v_template.clone()
    
    
    nn = SDF(d_in=6, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    lr = 1e-3
    optimizer = torch.optim.Adam(list(nn.parameters()), lr=lr)

    
    condition = torch.cat((verts_pose, verts_rest), dim=1)*10

    iters = 500
    for i in range(iters):
        offset = nn(condition, None)/100

        verts_cloth_new = verts_pose + offset

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)

        if use_mask:
            if i == iters-1:
                mesh = Meshes(
                    verts=[verts_cloth_zero],   
                    faces=[faces_cloth],
                    textures=textures
                )
                new_src_mesh = mesh.offset_verts(verts_cloth_new)
                images_predicted = renderer_textured_soft(new_src_mesh)
                images_pred = images_predicted[0, :, :, 3]
                images_pred = torch.clamp(images_pred + mask_top, 0, 1)

                img_mask = (images_pred.detach().cpu().numpy()*255).astype(np.uint8)
            

            with torch.no_grad():
                idx_faces, idx_vertices = get_pix_to_face_v2(verts_cloth_new, faces_cloth, raster)
                faces = faces_cloth[idx_faces]
            tri = verts_cloth_new[faces.reshape(-1)].reshape(-1,3,3)
            tri_center = tri.mean(dim=1)

            verts_cloth_2D = (transform.transform_points(tri_center.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
            verts_cloth_2D_pend = torch.cat((verts_cloth_2D, torch.zeros(verts_cloth_2D.shape[0], verts_cloth_2D.shape[1], 1).cuda()), dim=-1)
            
            loss_cd_2d_0, _ = chamfer_distance_single(idx_mask_pend, verts_cloth_2D_pend)
            loss_cd_2d_1, _ = chamfer_distance_single(verts_cloth_2D_pend, idx_mask_full_pend)
            loss_mask = loss_cd_2d_0 + loss_cd_2d_1
            loss_mask /= 5

        else:
            loss_mask = torch.zeros(1).cuda()
            img_mask = np.zeros((512, 512)).astype(np.uint8)


        if use_normal:
            tri_full = verts_cloth_new[faces_cloth.reshape(-1)].reshape(-1,3,3)
            tri_full_center = tri_full.mean(dim=1)
            vectors = tri_full[:,1:] - tri_full[:,:2]
            normal = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
            normal = normal/normal.norm(p=2, dim=-1, keepdim=True)
            _, loss_normal = chamfer_distance_single(xyz, tri_full_center.unsqueeze(0), x_normals=n_xyz, y_normals=normal.unsqueeze(0), abs_normal=True)
        else:
            loss_normal = torch.zeros(1).cuda()

        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_1, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_1*2
        loss_cd_3d *= 100*50*2


        loss = (loss_bending + loss_strain/4 + loss_gravity) + loss_cd_3d + loss_mask + loss_normal
        print('iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_mask: %0.4f , loss_cd_3d: %0.4f , loss_normal: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_mask.item(), loss_cd_3d.item(), loss_normal.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    cloth_pose_new = cloth_pose.copy()
    cloth_pose_new.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_new, img_mask

def align_observation_pt(cloth_pose, cloth_state, body_mesh, normals, vertices_waist, waist_v_id, depths):

    depth_img, mask_depth = depths
    vertices_waist = torch.FloatTensor(vertices_waist).cuda()

    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    vb_flip = vb.clone()
    vb_flip[:, -1] *= -1
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()
    fb = torch.LongTensor(body_mesh.faces).cuda()

    vb_clothed = torch.FloatTensor(clothed_mesh.vertices).cuda()
    vb_clothed_flip = vb_clothed.clone()
    vb_clothed_flip[:, -1] *= -1
    fb_clothed = torch.LongTensor(clothed_mesh.faces).cuda()

    normal_front, normal_back, mask_front, mask_back, mask_top, n_xyz, mask_full = normals
    mask_front = torch.FloatTensor(mask_front).cuda()
    mask_back = torch.FloatTensor(mask_back).cuda()
    mask_top = torch.FloatTensor(mask_top).cuda()
    n_xyz = torch.FloatTensor(n_xyz).cuda().unsqueeze(0)
    mask_full = torch.FloatTensor(mask_full).cuda()
    

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
        
        idx_x, idx_y = np.where(mask_full.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_full = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)


    verts_zero_clothed = torch.zeros(len(body_mesh.vertices)+len(cloth_pose.vertices), 3).cuda()
    faces_clothed = torch.LongTensor(np.concatenate((body_mesh.faces, cloth_pose.faces + len(body_mesh.vertices)))).cuda()
    smpl_rgb = torch.zeros(len(body_mesh.vertices), 3)
    smpl_rgb[:,0] += 255
    gar_rgb = torch.zeros(len(cloth_pose.vertices), 3)
    gar_rgb[:,1] += 255
    verts_rgb = torch.cat((smpl_rgb, gar_rgb))[None]
    textures_clothed = TexturesVertex(verts_features=verts_rgb.to(device))

    verts_pose = torch.FloatTensor(cloth_pose.vertices).cuda()
    verts_rest = cloth_state.v_template.clone()
    
    nn = SDF(d_in=6, d_out=3, dims=[256, 256, 256], skip_in=[]).cuda()
    lr = 1e-3
    eps = 2e-3
    optimizer = torch.optim.Adam(list(nn.parameters()), lr=lr)

    condition = torch.cat((verts_pose, verts_rest), dim=1)*10

    iters = 5 if debug else 500
    for i in range(iters):
        offset = nn(condition, None)/100

        verts_cloth_new = verts_pose + offset

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
            images_predicted = renderer_textured_soft(new_src_mesh)
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
        loss_cd_2d_f_0, loss_normal_f = chamfer_distance_single(idx_mask_pend_f, verts_cloth_2D_pend_f, x_normals=normal_img_f, y_normals=normal_f)
        loss_cd_2d_b_0, loss_normal_b = chamfer_distance_single(idx_mask_pend_b, verts_cloth_2D_pend_b, x_normals=normal_img_b, y_normals=normal_b)
        loss_cd_2d_f_1, _ = chamfer_distance_single(verts_cloth_2D_pend_f, idx_mask_pend_full)
        loss_cd_2d_b_1, _ = chamfer_distance_single(verts_cloth_2D_pend_b, idx_mask_pend_full)
        loss_mask = (loss_cd_2d_f_0 + loss_cd_2d_f_1 + loss_cd_2d_b_0 + loss_cd_2d_b_1)/5
        loss_normal = (loss_normal_f + loss_normal_b)*2
        
        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_3d, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_3d*2
        loss_cd_3d *= 100*50*2#/10#10


        loss_collision = collision_penalty_lite(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps) * colli_weight

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

        loss = (loss_bending + loss_strain/4 + loss_gravity) + loss_cd_3d + loss_mask + loss_normal + loss_collision + loss_waist + loss_depth
        print('align-pt iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_mask: %0.4f, loss_cd_3d: %0.4f, loss_normal: %0.4f, loss_collision: %0.4f, loss_waist: %0.4f, loss_depth: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_mask.item(), loss_cd_3d.item(), loss_normal.item(), loss_collision.item(), loss_waist.item(), loss_depth.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    cloth_pose_new = cloth_pose.copy()
    cloth_pose_new.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_new, img_mask

def align_observation_pt_verts(cloth_pose, cloth_state, body_mesh, normals, vertices_waist, waist_v_id, depths, prev_mesh, prev_prev_mesh):

    depth_img, mask_depth = depths
    vertices_waist = torch.FloatTensor(vertices_waist).cuda()

    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    vb_flip = vb.clone()
    vb_flip[:, -1] *= -1
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()
    fb = torch.LongTensor(body_mesh.faces).cuda()

    vb_clothed = torch.FloatTensor(clothed_mesh.vertices).cuda()
    vb_clothed_flip = vb_clothed.clone()
    vb_clothed_flip[:, -1] *= -1
    fb_clothed = torch.LongTensor(clothed_mesh.faces).cuda()

    normal_front, normal_back, mask_front, mask_back, mask_top, n_xyz, mask_full = normals
    mask_front = torch.FloatTensor(mask_front).cuda()
    mask_back = torch.FloatTensor(mask_back).cuda()
    mask_top = torch.FloatTensor(mask_top).cuda()
    n_xyz = torch.FloatTensor(n_xyz).cuda().unsqueeze(0)
    mask_full = torch.FloatTensor(mask_full).cuda()
    

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
        
        idx_x, idx_y = np.where(mask_full.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_full = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)



    verts_zero_clothed = torch.zeros(len(body_mesh.vertices)+len(cloth_pose.vertices), 3).cuda()
    faces_clothed = torch.LongTensor(np.concatenate((body_mesh.faces, cloth_pose.faces + len(body_mesh.vertices)))).cuda()
    smpl_rgb = torch.zeros(len(body_mesh.vertices), 3)
    smpl_rgb[:,0] += 255
    gar_rgb = torch.zeros(len(cloth_pose.vertices), 3)
    gar_rgb[:,1] += 255
    verts_rgb = torch.cat((smpl_rgb, gar_rgb))[None]
    textures_clothed = TexturesVertex(verts_features=verts_rgb.to(device))

    verts_pose = torch.FloatTensor(cloth_pose.vertices).cuda()
    
    offset = torch.zeros_like(verts_pose)
    offset.requires_grad = True
    lr = 1e-4
    eps = 2e-3
    optimizer = torch.optim.Adam([{'params': offset, 'lr': lr},])

    iters = 5 if debug else 200
    for i in range(iters):
        verts_cloth_new = verts_pose + offset

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)
        
        if prev_mesh is not None:
            loss_vel = ((verts_cloth_new - prev_mesh)**2).sum(dim=-1).mean() * consis_weight
        else:
            loss_vel = torch.zeros(1).cuda()
        if prev_prev_mesh is not None:
            loss_acc = ((verts_cloth_new - 2*prev_mesh + prev_prev_mesh)**2).sum(dim=-1).mean() * consis_weight
        else:
            loss_acc = torch.zeros(1).cuda()

        if i == iters-1:
            mesh = Meshes(
                verts=[verts_cloth_zero],   
                faces=[faces_cloth],
                textures=textures
            )
            new_src_mesh = mesh.offset_verts(verts_cloth_new)
            images_predicted = renderer_textured_soft(new_src_mesh)
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
        
        loss_cd_2d_f_0, loss_normal_f = chamfer_distance_single(idx_mask_pend_f, verts_cloth_2D_pend_f, x_normals=normal_img_f, y_normals=normal_f)
        loss_cd_2d_b_0, loss_normal_b = chamfer_distance_single(idx_mask_pend_b, verts_cloth_2D_pend_b, x_normals=normal_img_b, y_normals=normal_b)
        loss_cd_2d_f_1, _ = chamfer_distance_single(verts_cloth_2D_pend_f, idx_mask_pend_full)
        loss_cd_2d_b_1, _ = chamfer_distance_single(verts_cloth_2D_pend_b, idx_mask_pend_full)
        loss_mask = (loss_cd_2d_f_0 + loss_cd_2d_f_1 + loss_cd_2d_b_0 + loss_cd_2d_b_1)/5/10
        loss_normal = (loss_normal_f + loss_normal_b)*2
        
        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_3d, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_3d*2
        loss_cd_3d *= 100*50*2#10


        loss_collision = collision_penalty_lite(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps) * colli_weight

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
        print('align-pt-verts iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_mask: %0.4f , loss_cd_3d: %0.4f , loss_normal: %0.4f , loss_collision: %0.4f , loss_waist: %0.4f , loss_depth: %0.4f, loss_vel: %0.4f, loss_acc: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_mask.item(), loss_cd_3d.item(), loss_normal.item(), loss_collision.item(), loss_waist.item(), loss_depth.item(), loss_vel.item(), loss_acc.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    cloth_pose_new = cloth_pose.copy()
    cloth_pose_new.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_new, img_mask

def remesh(cloth_pose, cloth_state, body_mesh, prev_mesh, prev_prev_mesh):
    
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
    
    offset = torch.randn(verts_cloth.shape).cuda()*0.001*0
    offset.requires_grad = True
    lr = 1e-3
    eps = 1e-3
    optimizer = torch.optim.Adam([{'params': offset, 'lr': lr},])
    

    iters = 2000
    iters = 10 if debug else 1000
    for i in range(iters):
        
        verts_cloth_new = verts_cloth + offset
        loss_waist, _ = chamfer_distance(verts_boundary.unsqueeze(0), verts_cloth_new[idx_boundary].unsqueeze(0), )
        if i < 300:
            loss_waist *= 100
        else:
            loss_waist *= 10000

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)
        
        if prev_mesh is not None:
            loss_vel = ((verts_cloth_new - prev_mesh)**2).sum(dim=-1).mean() *consis_weight
        else:
            loss_vel = torch.zeros(1).cuda()
        if prev_prev_mesh is not None:
            loss_acc = ((verts_cloth_new - 2*prev_mesh + prev_prev_mesh)**2).sum(dim=-1).mean() *consis_weight
        else:
            loss_acc = torch.zeros(1).cuda()

        
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

        loss_collision = collision_penalty_lite(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps) * colli_weight

        loss = (loss_bending + loss_strain/2 + loss_gravity) + loss_cd_3d + loss_waist + loss_normal_3d*10 + loss_collision + loss_vel + loss_acc
        print('remesh iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_cd_3d: %0.4f , loss_waist: %0.4f  , loss_normal_3d: %0.4f , loss_collision: %0.4f, loss_vel: %0.4f, loss_acc: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_cd_3d.item(), loss_waist.item(), loss_normal_3d.item(), loss_collision.item(), loss_vel.item(), loss_acc.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cloth_pose_remesh = cloth_pose.copy()
    cloth_pose_remesh.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_remesh

def remesh_fix_waist(cloth_pose, cloth_state, body_mesh, waist_v_id, prev_mesh, prev_prev_mesh):
    
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

    waist_mask = torch.zeros_like(verts_cloth)
    waist_mask[waist_v_id] = 1
    vertices_waist = verts_cloth[waist_v_id]
    
    waist_v_id_loop = waist_v_id + waist_v_id[:1]
    waist_edges = np.array([waist_v_id_loop[:-1], waist_v_id_loop[1:]]).astype(int)
    edges = cloth_pose.vertices[waist_edges.reshape(-1)].reshape(-1, 2, 3)
    edges_oritation_gt = edges[:, 0] - edges[:, 1]
    edges_oritation_gt = torch.FloatTensor(edges_oritation_gt).cuda()
    waist_edges = torch.from_numpy(waist_edges).cuda()
    
    offset = torch.randn(verts_cloth.shape).cuda()*0.001
    offset.requires_grad = True
    lr = 1e-4
    eps = 1e-3
    optimizer = torch.optim.Adam([{'params': offset, 'lr': lr},])
    

    iters = 2000
    iters = 10 if debug else 1000
    for i in range(iters):
        
        verts_cloth_new = verts_cloth + offset
        loss_waist, _ = chamfer_distance(verts_boundary.unsqueeze(0), verts_cloth_new[idx_boundary].unsqueeze(0), )
        if i < 300:
            loss_waist *= 100
        else:
            loss_waist *= 10000

        edges_update = verts_cloth_new[waist_edges.reshape(-1)].reshape(-1, 2, 3)
        edges_oritation_update = edges_update[:, 0] - edges_update[:, 1]
        loss_edge = (1 - F.cosine_similarity(edges_oritation_update, edges_oritation_gt, dim=-1)).mean()*10
        loss_waist += loss_edge

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)
        
        if prev_mesh is not None:
            loss_vel = ((verts_cloth_new - prev_mesh)**2).sum(dim=-1).mean() *consis_weight
        else:
            loss_vel = torch.zeros(1).cuda()
        if prev_prev_mesh is not None:
            loss_acc = ((verts_cloth_new - 2*prev_mesh + prev_prev_mesh)**2).sum(dim=-1).mean() *consis_weight
        else:
            loss_acc = torch.zeros(1).cuda()

        
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

        loss_collision = collision_penalty_lite(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps) * colli_weight

        loss = (loss_bending + loss_strain/2 + loss_gravity) + loss_cd_3d + loss_waist + loss_normal_3d + loss_collision + loss_vel + loss_acc
        print('fix waist iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_cd_3d: %0.4f , loss_waist: %0.4f  , loss_normal_3d: %0.4f , loss_collision: %0.4f, loss_vel: %0.4f, loss_acc: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_cd_3d.item(), loss_waist.item(), loss_normal_3d.item(), loss_collision.item(), loss_vel.item(), loss_acc.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cloth_pose_remesh = cloth_pose.copy()
    cloth_pose_remesh.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_remesh

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

def project_waist(body, barycentric, idx_f, eps=1e-3):

    faces_waist = body.faces[idx_f].reshape(-1)
    fn_waist = body.face_normals[idx_f]

    triangles = body.vertices[faces_waist].reshape(-1, 3, 3)
    v_waist = trimesh.triangles.barycentric_to_points(triangles, barycentric)
    v_waist += fn_waist*eps

    return v_waist


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


def clean_pt(path, nb_neighbors=10, std_ratio=0.01):
    pcd = o3d.io.read_point_cloud(path)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    vertices = np.asarray(inlier_cloud.points)
    pt = trimesh.Trimesh(vertices)
    return pt, ind


def drag_waist(bottom_mesh, top_mesh, waist_v_id):
    waist = bottom_mesh.vertices[waist_v_id]
    y_max = waist[:,1].max()
    y_max = y_max + 0.05
    bottom_mesh.vertices[waist_v_id,1] += 0.05
    return 

def remove_arm(color_smpl_faces):
    new_faces_id = []
    for i in range(len(color_smpl_faces)):
        if color_smpl_faces[i,0] in [3, 4, 11, 12, 13, 14]:
            continue
        else:
            new_faces_id.append(i)

    return new_faces_id

parser = argparse.ArgumentParser(description="Generate the back normal maps")
parser.add_argument("--garment", type=str, default='Skirt', help="The type of garment")
parser.add_argument("--scale", type=float, default=0.8, help="The scale of the garment")
parser.add_argument("--vid_name", type=str, default='vid_demo', help="The name of the video")
parser.add_argument("--consis_weight", type=float, default=1000.0, help="consis_weight")
parser.add_argument("--colli_weight", type=int, default=5000, help="consis_weight")
parser.add_argument("--body_level", type=int, default=1, help="body level")

args = parser.parse_args()

garment = args.garment
scale = args.scale
vid_name = args.vid_name
consis_weight = args.consis_weight
colli_weight = args.colli_weight

normal_dir = body_dir = f'../data/{vid_name}/results-{scale}'

load_folder = f'../fitting-results/{vid_name}/uv-mapping-back-{scale}-{garment}'
load_isp_mask_folder = os.path.join(load_folder, 'mask-all0.1')
save_folder = f'../fitting-results/{vid_name}/uv-mapping-back-inpaint-{scale}-{garment}' # _debug_consis{consis_weight}
body_load_folder = f'../data/{vid_name}/cropped_body' 
top_folder = f'../fitting-results/{vid_name}/uv-mapping-back-inpaint-1.0-Tshirt'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

bottom_label = 240
top_label = 60

use_double_cd = False
use_depth = False
debug = False

raster, transform, _, renderer_textured_soft = get_render(render_res=512)

color_smpl = np.load('../extra-data/color_smpl_faces.npy')/15
faces_id_no_arm = remove_arm((color_smpl*15).astype(int))

images_list = sorted([img.split('_')[-1].split('.')[0] for img in os.listdir(load_folder) if img.startswith('images_normal_back_') and img.endswith('.png')])

for i in range(0, len(images_list)):
    
    img_name = images_list[i]

    step1_load_path = load_folder
    step2_load_path = load_folder
    save_path = save_folder
    if os.path.exists(os.path.join(save_path, 'final_mesh_%s.ply'%(img_name))):
        print(f"{img_name} exists")
        continue

    normal_front = cv2.imread(os.path.join(normal_dir, '%s_normal_align.png'%img_name))[:,:,::-1].copy()
    
    seg_bottom = cv2.imread(os.path.join(normal_dir, '%s_seg_align.png'%img_name))[:,:,0]

    if os.path.exists(os.path.join(normal_dir, f'{img_name}_seg_Tshirt_align.png')):
        seg_top = cv2.imread(os.path.join(normal_dir, f'{img_name}_seg_Tshirt_align.png'))[:,:,0]
    else:
        seg_top = np.zeros_like(seg_bottom)
    mask_bottom = ((seg_bottom == bottom_label).astype(np.uint8))
    mask_top = ((seg_top == top_label).astype(np.uint8))
    masks = [mask_bottom, mask_top]
    mask_full = cv2.imread(os.path.join(normal_dir, '%s_mask_full_align.png'%img_name))[:,:,0]
    mask_full = ((mask_full == 255).astype(np.uint8))
    
    for j in range(0,1):
        top_j = j
        body_mesh = trimesh.load(os.path.join(body_load_folder, '%s_body.ply'%(img_name,)))
        body_mesh.vertices *= 0.8
        if os.path.exists(os.path.join(top_folder, 'mesh-verts-%s.ply'%(img_name))):
            top_mesh = trimesh.load(os.path.join(top_folder, 'mesh-verts-%s.ply'%(img_name)))
            top_mesh.vertices *= 0.8
            top_mesh.export(os.path.join(save_path, 'top_%s.ply'%(img_name)))
            clothed_mesh = body_mesh + top_mesh
        else:
            clothed_mesh = body_mesh

        mask_back = cv2.imread(os.path.join(step1_load_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0]
        normal_back = cv2.imread(os.path.join(step1_load_path, 'images_normal_back_%s.png'%(img_name)))[:,:,::-1].copy()
        normal_back = cv2.resize(normal_back, (512, 512))
        mask_back = (cv2.resize(mask_back, (512, 512)) > 122).astype(np.uint8)
        n_xyz = np.load(os.path.join(step2_load_path, 'n_%s.npy'%(img_name)))

        std_ratio = 2
        xyz, idx = clean_pt(os.path.join(step2_load_path, 'xyz_%s.ply'%(img_name)), nb_neighbors=10, std_ratio=std_ratio)
        xyz.export(os.path.join(save_path, 'xyz_clean_%s.ply'%(img_name)))
        xyz = trimesh.load(os.path.join(step2_load_path, 'xyz_%s.ply'%(img_name)))
        n_xyz = n_xyz[idx]
        normals = [normal_front, normal_back, mask_bottom, mask_back, mask_top, n_xyz, mask_full]

        
        #####################################################################################
        prediction = np.load(os.path.join(step2_load_path, 'uv_transfer_%s.npz'%(img_name)))
        data_transfer = prediction['uv_transfer'] # [-1, 1]
        depth_front = data_transfer[:,:,3]
        depth_back = data_transfer[:,:,7]
        mask_depth_front = cv2.imread(os.path.join(step2_load_path, 'mask_front_%s.png'%img_name))[:,:,0]/255
        mask_depth_back = cv2.imread(os.path.join(step1_load_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0]/255
        depth_front = cv2.resize(depth_front, (512, 512))
        depth_back = cv2.resize(depth_back, (512, 512))
        mask_depth_front = (cv2.resize(mask_depth_front*255, (512, 512)) == 255).astype(np.uint8)
        mask_depth_back = (cv2.resize(mask_depth_back*255, (512, 512)) == 255).astype(np.uint8)
        coord_img_f = mask_to_coord(mask_depth_front)
        coord_img_b = mask_to_coord(mask_depth_back)
        
        depths = [depth_front, mask_depth_front]
        

        z_f = depth_front[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
        z_b = depth_back[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
        xyz_f = _to_xyz(coord_img_f, z_f, img_size=511.).astype(np.float32)
        xyz_b = _to_xyz(coord_img_b, z_b, img_size=511.).astype(np.float32)

        xyz = np.concatenate((xyz_f, xyz.vertices), axis=0)


        xyz = trimesh.PointCloud(xyz)
        xyz.export('../tmp/xyz.ply')
        
        xyz = xyz.vertices
        xyz = torch.FloatTensor(xyz).cuda().unsqueeze(0)
        

        barycentric = np.load(os.path.join(save_path, 'barycentric-%s.npz'%(img_name)))
        v_barycentric_f = barycentric['v_barycentric_f']
        v_barycentric_b = barycentric['v_barycentric_b']
        closest_face_idx_f = barycentric['closest_face_idx_f']
        closest_face_idx_b = barycentric['closest_face_idx_b']
        faces_f = barycentric['faces_f']
        faces_b = barycentric['faces_b']


        cloth_rest = trimesh.load(os.path.join(load_isp_mask_folder, 'sewing.ply'), validate=False, process=False)
        altas_f = trimesh.load(os.path.join(load_isp_mask_folder, 'atlas-f.ply'), validate=False, process=False)
        altas_b = trimesh.load(os.path.join(load_isp_mask_folder, 'atlas-b.ply'), validate=False, process=False)
        pattern_f = trimesh.load(os.path.join(load_isp_mask_folder, 'pattern-f.ply'), validate=False, process=False)
        pattern_b = trimesh.load(os.path.join(load_isp_mask_folder, 'pattern-b.ply'), validate=False, process=False)
        num_v_f = len(pattern_f.vertices)

        cloth_rest_z_up = apply_rotation(np.pi/2, cloth_rest.copy(), 'x')
        cloth_rest_z_up.export(os.path.join(save_path, 'cloth_rest_z_up_%s.ply'%(img_name)))
        waist_v_id = get_connected_paths_skirt(cloth_rest_z_up)[0]

        x_res = y_res = 128
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

        
        ############### inpaint uv ###############
        uv_inpaint = np.load(os.path.join(save_path, 'uv-inpaint_%s.npy'%(img_name)))
        image = np.transpose(uv_inpaint, (2,0,1))
        image = torch.FloatTensor(image).cuda().unsqueeze(0)

        ############### sparse uv ###############
        prediction = np.load(os.path.join(step2_load_path, 'uv_transfer_%s.npz'%(img_name)))
        data_transfer = prediction['uv_transfer'] # [-1, 1]
        uv_transfer_f = data_transfer[:,:,:3]
        uv_transfer_b = data_transfer[:,:,4:4+3]
        depth_transfer_f = data_transfer[:,:,3]
        depth_transfer_b = data_transfer[:,:,7]

        mask_cloth_f = cv2.imread(os.path.join(step2_load_path, 'mask_front_%s.png'%img_name))[:,:,0]
        mask_cloth_b = cv2.imread(os.path.join(step1_load_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0]
        coord_img_f = mask_to_coord(mask_cloth_f)
        coord_img_b = mask_to_coord(mask_cloth_b)

        z_f = depth_transfer_f[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
        z_b = depth_transfer_b[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
        xyz_f = _to_xyz(coord_img_f, z_f, img_size=191.).astype(np.float32)
        xyz_b = _to_xyz(coord_img_b, z_b, img_size=191.).astype(np.float32)

        sparse_uv, sparse_mask = _to_uv_FB(uv_transfer_f, uv_transfer_b, coord_img_f, coord_img_b, xyz_f, xyz_b, size_uv=128)
        sparse_uv = torch.FloatTensor(sparse_uv).cuda().unsqueeze(0)
        sparse_mask = torch.BoolTensor(sparse_mask).cuda().unsqueeze(0)


        cloth_pose = trimesh.load(os.path.join(save_path, 'deform-inpaint-%s.ply'%(img_name)), validate=False, process=False)
        cloth_pose_f = trimesh.load(os.path.join(save_path, 'deform-f-inpaint-%s.ply'%(img_name)), validate=False, process=False)
        cloth_pose_b = trimesh.load(os.path.join(save_path, 'deform-b-inpaint-%s.ply'%(img_name)), validate=False, process=False)

        scale = rescale(cloth_pose_f, cloth_pose_b, altas_f, altas_b)
        
        material = Material()
        cloth_state = Cloth_from_NP(cloth_rest.vertices*scale, cloth_rest.faces, material)
        np.savez(os.path.join(save_path, 'cloth_state_%s'%(img_name)), vertices=cloth_rest.vertices*scale, faces=cloth_rest.faces)

        mapping_related = [faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz]
        
        cloth_pose_pt = trimesh.load(os.path.join(save_path, 'deform-inpaint-%s.ply'%(img_name)), validate=False, process=False)
        

        body_mesh_no_arm = body_mesh.copy()
        body_mesh_no_arm.faces = body_mesh_no_arm.faces[faces_id_no_arm]
        barycentric_waist, idx_f_waist = get_barycentric(body_mesh_no_arm, cloth_pose_pt.vertices[waist_v_id])
        vertices_waist = project_waist(body_mesh_no_arm, barycentric_waist, idx_f_waist)

        
        cloth_pose_pt_new, img_mask = align_observation_pt(cloth_pose_pt, cloth_state, body_mesh, normals, vertices_waist, waist_v_id, depths)
        img_mask = seg_bottom/2 + seg_top/2 + img_mask/2
        cloth_pose_pt_new.export(os.path.join(save_path, 'mesh_pt_nn_%s.ply'%(img_name)))
        cv2.imwrite(os.path.join(save_path, 'mask_overlay_pt_nn_%s.png'%(img_name)), img_mask)
        
        prev_mesh = None
        prev_prev_mesh = None
        if os.path.exists(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(str(int(img_name)-1).zfill(6)))):
            prev_mesh = trimesh.load(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(str(int(img_name)-1).zfill(6))), validate=False, process=False).vertices
            prev_mesh = torch.FloatTensor(prev_mesh).cuda()
        if os.path.exists(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(str(int(img_name)-2).zfill(6)))):
            prev_prev_mesh = trimesh.load(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(str(int(img_name)-2).zfill(6))), validate=False, process=False).vertices
            prev_prev_mesh = torch.FloatTensor(prev_prev_mesh).cuda()
            
        cloth_pose_pt = trimesh.load(os.path.join(save_path, 'mesh_pt_nn_%s.ply'%(img_name)), process=False, validate=False)
        cloth_pose_pt_new, img_mask = align_observation_pt_verts(cloth_pose_pt, cloth_state, body_mesh, normals, vertices_waist, waist_v_id, depths, prev_mesh, prev_prev_mesh)
        img_mask = seg_bottom/2 + seg_top/2 + img_mask/2
        cloth_pose_pt_new.export(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(img_name)))
        cv2.imwrite(os.path.join(save_path, 'mask_overlay_pt_verts_%s.png'%(img_name)), img_mask)
        
        prev_mesh = None
        prev_prev_mesh = None
        if os.path.exists(os.path.join(save_path, 'mesh_remesh_%s.ply'%(str(int(img_name)-1).zfill(6)))):
            prev_mesh = trimesh.load(os.path.join(save_path, 'mesh_remesh_%s.ply'%(str(int(img_name)-1).zfill(6))), validate=False, process=False).vertices
            prev_mesh = torch.FloatTensor(prev_mesh).cuda()
        if os.path.exists(os.path.join(save_path, 'mesh_remesh_%s.ply'%(str(int(img_name)-2).zfill(6)))):
            prev_prev_mesh = trimesh.load(os.path.join(save_path, 'mesh_remesh_%s.ply'%(str(int(img_name)-2).zfill(6))), validate=False, process=False).vertices
            prev_prev_mesh = torch.FloatTensor(prev_prev_mesh).cuda()
        
        cloth_pose_pt = trimesh.load(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(img_name)), process=False, validate=False)
        cloth_pose_remseh = remesh(cloth_pose_pt, cloth_state, body_mesh, prev_mesh, prev_prev_mesh)
        cloth_pose_remseh.export(os.path.join(save_path, 'mesh_remesh_%s.ply'%(img_name)))
        
        
        cloth_pose_pt = trimesh.load(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(img_name)), process=False, validate=False)
        cloth_pose_pt.vertices[waist_v_id, 1] += 0.03
        body_mesh_no_arm = body_mesh.copy()
        body_mesh_no_arm.faces = body_mesh_no_arm.faces[faces_id_no_arm]
        barycentric_waist, idx_f_waist = get_barycentric(body_mesh_no_arm, cloth_pose_pt.vertices[waist_v_id])
        vertices_waist = project_waist(body_mesh_no_arm, barycentric_waist, idx_f_waist)
        cloth_pose_pt.vertices[waist_v_id] = vertices_waist
        
        prev_mesh = None
        prev_prev_mesh = None
        if os.path.exists(os.path.join(save_path, 'final_mesh_%s.ply'%(str(int(img_name)-1).zfill(6)))):
            prev_mesh = trimesh.load(os.path.join(save_path, 'final_mesh_%s.ply'%(str(int(img_name)-1).zfill(6))), validate=False, process=False).vertices
            prev_mesh = torch.FloatTensor(prev_mesh).cuda()
        if os.path.exists(os.path.join(save_path, 'final_mesh_%s.ply'%(str(int(img_name)-2).zfill(6)))):
            prev_prev_mesh = trimesh.load(os.path.join(save_path, 'final_mesh_%s.ply'%(str(int(img_name)-2).zfill(6))), validate=False, process=False).vertices
            prev_prev_mesh = torch.FloatTensor(prev_prev_mesh).cuda()

        cloth_pose_remseh = remesh_fix_waist(cloth_pose_pt, cloth_state, body_mesh, waist_v_id, prev_mesh, prev_prev_mesh)
        cloth_pose_remseh.export(os.path.join(save_path, 'final_mesh_%s.ply'%(img_name)))