import os,sys
import numpy as np 
import trimesh
import random
import cv2
import torch

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

def get_raster(render_res, scale, faces_per_pixel=2, sigma=1e-7):
    #render_res = 256
    dis = 100.0
    #scale = 80.0 #100.0
    mesh_y_center = 0.0
    cam_pos = torch.tensor([
                    (0, mesh_y_center, dis),
                    (0, mesh_y_center, -dis),
                ])
    R, T = look_at_view_transform(
        eye=cam_pos[[0]],
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

    raster_settings_hard = RasterizationSettings(
        image_size=render_res, 
        blur_radius=np.log(1.0 / 1e-4)*sigma,#1e-5, 
        faces_per_pixel=faces_per_pixel,#1, 
        max_faces_per_bin=500000,
        perspective_correct=False,
    )

    meshRas_hard = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_hard)

    return meshRas_hard

def get_pix_to_face(verts, faces, raster):
    mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
        )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()
    return pix_to_face

def get_pix_to_face_index(verts, faces, raster, indicator=None):
    mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
        )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()
    x, y = torch.nonzero(pix_to_face!=-1, as_tuple=True)
    idx_face = pix_to_face[x, y]
    #print(indicator.shape, idx_face.shape, idx_face.max())
    #ys.exit()

    if indicator is not None:
        valid = torch.nonzero(indicator[idx_face])
        x = x[valid]
        y = y[valid]
        idx_face = idx_face[valid]

    return x, y, idx_face

def get_pix_to_face_v2(verts, faces, raster):
    mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
        )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()
    valid_pixel = pix_to_face > -1
    valid_faces = pix_to_face[valid_pixel]
    valid_vertices = torch.unique(faces[valid_faces].flatten())
    return valid_faces, valid_vertices

def compute_face_normals(verts, faces):
    """
    计算每个面的法向量，并归一化
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)  # 计算叉积
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # 归一化
    return face_normals

def get_pix_to_face_v2_cam(verts, faces, raster, cameras):
    """
    获取像素到面索引，并筛选符合法向量朝向相机的有效面
    """
    # 计算面法向量（世界坐标系下）
    world_normals = compute_face_normals(verts, faces)

    # 获取相机旋转矩阵 R（形状为 (1, 3, 3)）
    R = cameras.R[0]  # 仅取一个相机（默认只有一个）
    
    # 变换法向量到相机坐标系
    camera_normals = torch.einsum('ij,kj->ki', R, world_normals)  # (num_faces, 3)

    # 计算法向量与相机朝向的夹角（正交相机：视线方向始终是 [0, 0, -1]）
    view_dir = R[2, :] # torch.tensor([0, 0, -1], device=verts.device)
    # print(f"view_dir: {view_dir}")
    dot_product = torch.einsum('ij,j->i', camera_normals, view_dir)  # (num_faces,)

    # 仅保留朝向相机的面（dot_product > 0）
    visibility_mask = dot_product > 0

    # 创建 Pytorch3D 网格
    mesh_py3d = Meshes(
        verts=[verts],   
        faces=[faces],
        textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
    )

    # 进行光栅化
    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()  # (H, W, K)

    # 获取有效像素（非 -1）
    valid_pixel = pix_to_face > -1
    valid_faces = pix_to_face[valid_pixel]  # (num_valid_pixels,)

    # 仅保留朝向相机的面
    valid_faces = valid_faces[visibility_mask[valid_faces]]

    # 获取有效面涉及的顶点
    valid_vertices = torch.unique(faces[valid_faces].flatten())

    return valid_faces, valid_vertices

def get_pix_to_face_with_body(verts_gar, faces_gar, verts_body, faces_body, raster):
    len_faces_gar = len(faces_gar)
    verts = torch.cat((verts_gar, verts_body), dim=0)
    faces = torch.cat((faces_gar, faces_body+len(verts_gar)), dim=0)
    mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
        )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()
    valid_pixel = torch.logical_and(pix_to_face > -1, pix_to_face < len_faces_gar)
    #indices = valid_pixel.nonzero()
    #print(indices.shape, pix_to_face.min(), pix_to_face.max())
    #sys.exit()

    valid_faces = pix_to_face[valid_pixel]

    valid_faces = torch.unique(valid_faces.flatten())
    #print(type(valid_faces), type(faces_gar))
    valid_vertices = torch.unique(faces_gar[valid_faces].flatten())
    return valid_faces, valid_vertices

def get_pix_to_face_with_body_output(verts_gar, faces_gar, verts_body, faces_body, raster):
    """
    Computes visible faces and vertices for both garment and body meshes.

    Args:
        verts_gar (torch.Tensor): Garment vertices (N_g, 3)
        faces_gar (torch.Tensor): Garment faces (F_g, 3)
        verts_body (torch.Tensor): Body vertices (N_b, 3)
        faces_body (torch.Tensor): Body faces (F_b, 3)
        raster (function): Rasterization function

    Returns:
        valid_faces_gar (torch.Tensor): Valid garment faces
        valid_vertices_gar (torch.Tensor): Valid garment vertices
        valid_faces_body (torch.Tensor): Valid body faces
        valid_vertices_body (torch.Tensor): Valid body vertices
    """

    len_faces_gar = len(faces_gar)
    len_verts_gar = len(verts_gar)

    # **Step 1: Merge garment and body meshes**
    verts = torch.cat((verts_gar, verts_body), dim=0)
    faces = torch.cat((faces_gar, faces_body + len_verts_gar), dim=0)  # Offset body face indices

    # Create PyTorch3D Mesh
    mesh_py3d = Meshes(
        verts=[verts],   
        faces=[faces],
        textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
    )

    # **Step 2: Rasterization**
    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()

    # **Step 3: Identify valid garment faces and vertices**
    valid_pixel_gar = torch.logical_and(pix_to_face > -1, pix_to_face < len_faces_gar)
    valid_faces_gar = torch.unique(pix_to_face[valid_pixel_gar].flatten())
    valid_vertices_gar = torch.unique(faces_gar[valid_faces_gar].flatten())

    # **Step 4: Identify valid body faces and vertices**
    valid_pixel_body = pix_to_face >= len_faces_gar  # Body faces are indexed after garment faces
    valid_faces_body = torch.unique((pix_to_face[valid_pixel_body] - len_faces_gar).flatten())  # Adjust index back
    valid_vertices_body = torch.unique(faces_body[valid_faces_body].flatten())

    return valid_faces_gar, valid_vertices_gar, valid_faces_body, valid_vertices_body


def get_pix_to_face_with_body_index(verts_gar, faces_gar, verts_body, faces_body, raster):
    len_faces_gar = len(faces_gar)
    verts = torch.cat((verts_gar, verts_body), dim=0)
    faces = torch.cat((faces_gar, faces_body+len(verts_gar)), dim=0)
    mesh_py3d = Meshes(
            verts=[verts],   
            faces=[faces],
            textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
        )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()
    valid_pixel = torch.logical_and(pix_to_face > -1, pix_to_face < len_faces_gar)
    #indices = valid_pixel.nonzero()
    #print(indices.shape, pix_to_face.min(), pix_to_face.max())
    #sys.exit()

    pix_to_face[~valid_pixel] = -1
    x, y = torch.nonzero(pix_to_face!=-1, as_tuple=True)
    idx_face = pix_to_face[x, y]

    return x, y, idx_face