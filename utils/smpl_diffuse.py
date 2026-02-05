import os
import sys
import tqdm
import numpy as np
import trimesh
import torch
import random
random.seed(0)
np.random.seed(0)

sys.path.append('../')
from utils.readfile import dump_pkl
from utils.flatten import reorder_vertices

color_smplx = np.load('../extra-data/color_smplx_faces.npy')

def filter_faces(mesh_smplx):
    lower_body = set([1,2, 5,6,7,8,9,10])
    upper_body = set([1,2, 3, 4, 7,8, 9,10, 11,12,13,14,15])
    idx_face_upper = []
    idx_face_lower = []
    for i in range(len(color_smplx)):
        if color_smplx[i, 0] in lower_body:
            idx_face_lower.append(i)
        if color_smplx[i, 0] in upper_body:
            idx_face_upper.append(i)

    mesh_smplx_upper = trimesh.Trimesh(mesh_smplx.vertices, mesh_smplx.faces[idx_face_upper], validate=False, process=False)
    mesh_smplx_lower = trimesh.Trimesh(mesh_smplx.vertices, mesh_smplx.faces[idx_face_lower], validate=False, process=False)

    mesh_smplx_upper_new, _, _, mapping_idx_upper = reorder_vertices(mesh_smplx_upper)
    mesh_smplx_lower_new, _, _, mapping_idx_lower = reorder_vertices(mesh_smplx_lower)

    return mesh_smplx_upper_new, mapping_idx_upper, mesh_smplx_lower_new, mapping_idx_lower

def random_ball(N, d, debug=False):
    '''
    u = np.random.normal(0, 1, N)
    v = np.random.normal(0, 1, N)
    w = np.random.normal(0, 1, N)
    r = random()**(1./3)
    norm= (u*u + v*v + w*w)**(0.5)
    (x,y,z) = r*(u,v,w)/norm*d
    '''

    # sample within a ball (Gaussian) ~ N([0,0,0], d)
    theta_rad = np.random.rand(N)*np.pi*2
    phi_rad = np.random.rand(N)*np.pi
    r = np.random.normal(0, 1, N)*d

    x = r*np.sin(theta_rad)*np.cos(phi_rad)
    y = r*np.sin(theta_rad)*np.sin(phi_rad)
    z = r*np.cos(theta_rad)

    random_samples = np.stack([x,y,z], axis=1)

    return random_samples

def smpl_diffuse_weights(mesh_body_part, points):
    Num_gaussian = 1000#10#1000
    query_body = trimesh.proximity.ProximityQuery(mesh_body_part)

    gaussian_samples_unit = random_ball(Num_gaussian, 1) 
    closest_pt, distance, closest_face_idx = query_body.on_surface(points)
    weight = np.zeros((len(points), len(mesh_body_part.vertices)))

    for i in tqdm.tqdm(range(len(points))):
        gaussian_samples = gaussian_samples_unit*distance[i]
        samples_ball = gaussian_samples + points[[i]]
        samples_ball = np.concatenate((samples_ball, points[[i]]), axis=0)

        _, closest_idx_gau = query_body.vertex(samples_ball)

        for idx in closest_idx_gau:
            weight[i, idx] += 1

        weight[i] /= Num_gaussian+1

    return weight

def offset_points(smpl_verts_offset, weight):
    return weight@smpl_verts_offset


if __name__ == "__main__":
    mesh_smplx = trimesh.load('../extra-data/smplx.obj')
    mesh_smplx_upper_new, mapping_idx_upper, mesh_smplx_lower_new, mapping_idx_lower = filter_faces(mesh_smplx)
    dump_pkl({'mapping_idx_upper':mapping_idx_upper, 'mapping_idx_lower':mapping_idx_lower, 'faces_upper':mesh_smplx_upper_new.faces, 'faces_lower':mesh_smplx_lower_new.faces},'../extra-data/mapping_upper_lower_idx.pkl')
    mesh_smplx_upper_new.export('../extra-data/smplx_upper.ply')
    mesh_smplx_lower_new.export('../extra-data/smplx_lower.ply')