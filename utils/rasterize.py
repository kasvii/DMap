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
    dis = 100.0
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
        blur_radius=np.log(1.0 / 1e-4)*sigma,
        faces_per_pixel=faces_per_pixel,
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
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)
    return face_normals

def get_pix_to_face_v2_cam(verts, faces, raster, cameras):
    world_normals = compute_face_normals(verts, faces)

    R = cameras.R[0]
    
    camera_normals = torch.einsum('ij,kj->ki', R, world_normals)

    view_dir = R[2, :]
    dot_product = torch.einsum('ij,j->i', camera_normals, view_dir)

    visibility_mask = dot_product > 0

    mesh_py3d = Meshes(
        verts=[verts],   
        faces=[faces],
        textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
    )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()

    valid_pixel = pix_to_face > -1
    valid_faces = pix_to_face[valid_pixel]

    valid_faces = valid_faces[visibility_mask[valid_faces]]

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

    valid_faces = pix_to_face[valid_pixel]

    valid_faces = torch.unique(valid_faces.flatten())
    valid_vertices = torch.unique(faces_gar[valid_faces].flatten())
    return valid_faces, valid_vertices

def get_pix_to_face_with_body_output(verts_gar, faces_gar, verts_body, faces_body, raster):
    len_faces_gar = len(faces_gar)
    len_verts_gar = len(verts_gar)

    verts = torch.cat((verts_gar, verts_body), dim=0)
    faces = torch.cat((faces_gar, faces_body + len_verts_gar), dim=0)

    mesh_py3d = Meshes(
        verts=[verts],   
        faces=[faces],
        textures=TexturesVertex(verts_features=torch.ones_like(verts[None]))
    )

    Fragments = raster(mesh_py3d)
    pix_to_face = Fragments.pix_to_face.squeeze()

    valid_pixel_gar = torch.logical_and(pix_to_face > -1, pix_to_face < len_faces_gar)
    valid_faces_gar = torch.unique(pix_to_face[valid_pixel_gar].flatten())
    valid_vertices_gar = torch.unique(faces_gar[valid_faces_gar].flatten())

    valid_pixel_body = pix_to_face >= len_faces_gar
    valid_faces_body = torch.unique((pix_to_face[valid_pixel_body] - len_faces_gar).flatten())
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

    pix_to_face[~valid_pixel] = -1
    x, y = torch.nonzero(pix_to_face!=-1, as_tuple=True)
    idx_face = pix_to_face[x, y]

    return x, y, idx_face