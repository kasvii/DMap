import os,sys
import numpy as np 
import torch
import trimesh
import cv2
from tqdm import tqdm
sys.path.append('..')
from networks import SDF
from utils.isp import create_uv_mesh
from utils.mesh_reader import read_mesh_from_sdf_test_batch_v2_with_label, triangulation_seam_v2
from utils.cutting import select_boundary, connect_2_way, one_ring_neighour
from utils.optimization import optimize_lat_code_anchors, vis_diff, cat_images

import pymesh
import time
import argparse

def color_boundary(label):
    color_map = [[255, 51, 51],
                [255, 153, 51],
                [255, 255, 51],
                [153, 255, 51],
                [51, 255, 51],
                [51, 255, 153],
                [51, 255, 255],
                [51, 153, 255],
                [51, 51, 255],
                [153, 51, 255],
                [255, 51, 255],
                [255, 51, 153],
                [160, 160, 160]]
    color_map = np.array(color_map).astype(int)

    return color_map[label]

def repair_pattern(mesh_trimesh, res=128):

    mesh = pymesh.form_mesh(mesh_trimesh.vertices, mesh_trimesh.faces)
    count = 0
    target_len_long = 2/res*np.sqrt(2)*1.2
    target_len_short = 2/res*0.4
    print(mesh.num_vertices)
    mesh, __ = pymesh.split_long_edges(mesh, target_len_long)

    num_vertices = mesh.num_vertices
    print(num_vertices)
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len_short, preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 120.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh_trimesh_new  = trimesh.Trimesh(mesh.vertices, mesh.faces, validate=False, process=False)

    return mesh_trimesh_new

def reconstruct_pattern_with_label(model_isp, latent_code, uv_vertices, uv_faces, edges, resolution=256):
    model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b = model_isp
    with torch.no_grad():
        uv_faces_torch_f = torch.LongTensor(uv_faces).cuda()
        uv_faces_torch_b = torch.LongTensor(uv_faces[:,[0,2,1]]).cuda()
        vertices_new_f = uv_vertices[:,:2].clone()
        vertices_new_b = uv_vertices[:,:2].clone()

        uv_input = uv_vertices[:,:2]*10
        num_points = len(uv_vertices)
        latent_code = latent_code.repeat(num_points, 1)
        pred_f = model_sdf_f(uv_input, latent_code)
        pred_b = model_sdf_b(uv_input, latent_code)
        sdf_pred_f = pred_f[:, 0]
        sdf_pred_b = pred_b[:, 0]
        label_f = pred_f[:, 1:]
        label_b = pred_b[:, 1:]
        label_f = torch.argmax(label_f, dim=-1)
        label_b = torch.argmax(label_b, dim=-1)

        sdf_pred = torch.stack((sdf_pred_f, sdf_pred_b), dim=0)
        uv_vertices_batch = torch.stack((uv_vertices[:,:2], uv_vertices[:,:2]), dim=0)
        label_pred = torch.stack((label_f, label_b), dim=0)
        vertices_new, faces_list, labels_list = read_mesh_from_sdf_test_batch_v2_with_label(uv_vertices_batch, uv_faces_torch_f, sdf_pred, label_pred, edges, reorder=True, thresh=-1e-3)
        vertices_new_f = vertices_new[0]
        vertices_new_b = vertices_new[1]
        faces_new_f = faces_list[0]
        faces_new_b = faces_list[1][:,[0,2,1]]
        label_new_f = labels_list[0]
        label_new_b = labels_list[1]

        v_f = np.zeros((len(vertices_new_f), 3))
        v_b = np.zeros((len(vertices_new_b), 3))
        v_f[:, :2] = vertices_new_f
        v_b[:, :2] = vertices_new_b
        mesh_pattern_f = trimesh.Trimesh(v_f, faces_new_f, validate=False, process=False)
        mesh_pattern_b = trimesh.Trimesh(v_b, faces_new_b, validate=False, process=False)
        if using_repair:
            print('repair mesh_pattern_f')
            mesh_pattern_f = repair_pattern(mesh_pattern_f, res=resolution)
            print('repair mesh_pattern_b')
            mesh_pattern_b = repair_pattern(mesh_pattern_b, res=resolution)
            
        
        pattern_vertices_f = torch.FloatTensor(mesh_pattern_f.vertices).cuda()[:,:2]
        pattern_vertices_b = torch.FloatTensor(mesh_pattern_b.vertices).cuda()[:,:2]

        pred_f = model_sdf_f(pattern_vertices_f*10, latent_code[:len(pattern_vertices_f)])
        pred_b = model_sdf_b(pattern_vertices_b*10, latent_code[:len(pattern_vertices_b)])
        label_new_f = pred_f[:, 1:]
        label_new_b = pred_b[:, 1:]
        label_new_f = torch.argmax(label_new_f, dim=-1).cpu().numpy()
        label_new_b = torch.argmax(label_new_b, dim=-1).cpu().numpy()

        pred_atlas_f = model_atlas_f(pattern_vertices_f*10, latent_code[:len(pattern_vertices_f)])/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b*10, latent_code[:len(pattern_vertices_b)])/10

        mesh_atlas_f = trimesh.Trimesh(pred_atlas_f.cpu().numpy(), mesh_pattern_f.faces, process=False, valid=False)
        mesh_atlas_b = trimesh.Trimesh(pred_atlas_b.cpu().numpy(), mesh_pattern_b.faces, process=False, valid=False)

        idx_boundary_v_f, boundary_edges_f = select_boundary(mesh_pattern_f)
        idx_boundary_v_b, boundary_edges_b = select_boundary(mesh_pattern_b)
        boundary_edges_f = set([tuple(sorted(e)) for e in boundary_edges_f.tolist()])
        boundary_edges_b = set([tuple(sorted(e)) for e in boundary_edges_b.tolist()])
        label_boundary_v_f = label_new_f[idx_boundary_v_f]
        label_boundary_v_b = label_new_b[idx_boundary_v_b]

    return mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b, label_new_f, label_new_b

def sewing_vertical(idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, return_seam_top=False):
    idx_boundary_v_f, idx_boundary_v_b = idx_boundary_v
    boundary_edges_f, boundary_edges_b = boundary_edges
    label_boundary_v_f, label_boundary_v_b = label_boundary_v
    mesh_pattern_f, mesh_pattern_b = mesh_pattern
    labels_f, labels_b = labels

    indicator_seam_f = label_boundary_v_f == seam_i
    indicator_seam_b = label_boundary_v_b == seam_i
    
    idx_seam_v_f = idx_boundary_v_f[indicator_seam_f]
    idx_seam_v_b = idx_boundary_v_b[indicator_seam_b]
    
    one_rings_seam_f = one_ring_neighour(idx_seam_v_f, mesh_pattern_f, is_dic=True, mask_set=set(idx_seam_v_f))
    one_rings_seam_b = one_ring_neighour(idx_seam_v_b, mesh_pattern_b, is_dic=True, mask_set=set(idx_seam_v_b))
    
    path_seam_f, _ = connect_2_way(set(idx_seam_v_f), one_rings_seam_f, boundary_edges_f)
    path_seam_b, _ = connect_2_way(set(idx_seam_v_b), one_rings_seam_b, boundary_edges_b)

    if mesh_pattern_f.vertices[path_seam_f[0], 1] < mesh_pattern_f.vertices[path_seam_f[-1], 1]: # high to low
        path_seam_f = path_seam_f[::-1]
    if mesh_pattern_b.vertices[path_seam_b[0], 1] < mesh_pattern_b.vertices[path_seam_b[-1], 1]:
        path_seam_b = path_seam_b[::-1]

    idx_offset = len(mesh_pattern_f.vertices)

    faces_seam = triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam_f, path_seam_b, idx_offset, reverse=False)

    if return_seam_top:
        return faces_seam, [path_seam_f[0], path_seam_b[0]+idx_offset]
    else:
        return faces_seam

def sewing_vertical_tshirt(idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, return_seam_top=False, horizontal=False):
    idx_boundary_v_f, idx_boundary_v_b = idx_boundary_v
    boundary_edges_f, boundary_edges_b = boundary_edges
    label_boundary_v_f, label_boundary_v_b = label_boundary_v
    mesh_pattern_f, mesh_pattern_b = mesh_pattern
    labels_f, labels_b = labels

    indicator_seam_f = label_boundary_v_f == seam_i
    indicator_seam_b = label_boundary_v_b == seam_i
    
    idx_seam_v_f = idx_boundary_v_f[indicator_seam_f]
    idx_seam_v_b = idx_boundary_v_b[indicator_seam_b]
    
    one_rings_seam_f = one_ring_neighour(idx_seam_v_f, mesh_pattern_f, is_dic=True, mask_set=set(idx_seam_v_f))
    one_rings_seam_b = one_ring_neighour(idx_seam_v_b, mesh_pattern_b, is_dic=True, mask_set=set(idx_seam_v_b))
    
    path_seam_f, _ = connect_2_way(set(idx_seam_v_f), one_rings_seam_f, boundary_edges_f)
    path_seam_b, _ = connect_2_way(set(idx_seam_v_b), one_rings_seam_b, boundary_edges_b)

    if horizontal:
        if mesh_pattern_f.vertices[path_seam_f[0], 0] > mesh_pattern_f.vertices[path_seam_f[-1], 0]: # left to right
            path_seam_f = path_seam_f[::-1]
        if mesh_pattern_b.vertices[path_seam_b[0], 0] > mesh_pattern_b.vertices[path_seam_b[-1], 0]:
            path_seam_b = path_seam_b[::-1]
    
    else:
        if mesh_pattern_f.vertices[path_seam_f[0], 1] < mesh_pattern_f.vertices[path_seam_f[-1], 1]: # high to low
            path_seam_f = path_seam_f[::-1]
        if mesh_pattern_b.vertices[path_seam_b[0], 1] < mesh_pattern_b.vertices[path_seam_b[-1], 1]:
            path_seam_b = path_seam_b[::-1]

    idx_offset = len(mesh_pattern_f.vertices)

    faces_seam = triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam_f, path_seam_b, idx_offset, reverse=False)

    if return_seam_top:
        return faces_seam, [path_seam_f[0], path_seam_b[0]+idx_offset]
    else:
        return faces_seam

def compute_offset(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.001):
    idx_seam_v_f_1 = idx_boundary_v_f[label_boundary_v_f == 1]
    idx_seam_v_f_2 = idx_boundary_v_f[label_boundary_v_f == 2]
    seam_v_f_1 = mesh_atlas_f.vertices[idx_seam_v_f_1]
    seam_v_f_2 = mesh_atlas_f.vertices[idx_seam_v_f_2]

    highest_1_i = np.argmax(seam_v_f_1[:, 1].flatten())
    highest_2_i = np.argmax(seam_v_f_2[:, 1].flatten())

    offset = ((seam_v_f_1[highest_1_i] - seam_v_f_2[highest_2_i])**2).sum()*ratio
    return offset

def compute_offset_tshirt(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.001):
    idx_seam_v_f_0 = idx_boundary_v_f[label_boundary_v_f == 0]
    seam_v_f_0 = mesh_atlas_f.vertices[idx_seam_v_f_0]

    highest_0_i = np.argmax(seam_v_f_0[:, -1].flatten())
    lowest_0_i = np.argmin(seam_v_f_0[:, -1].flatten())

    offset = ((seam_v_f_0[highest_0_i] - seam_v_f_0[lowest_0_i])**2).sum()*ratio
    return offset

def compute_offset_trousers(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.001):
    idx_seam_v_f_1 = idx_boundary_v_f[label_boundary_v_f == 1]
    idx_seam_v_f_2 = idx_boundary_v_f[label_boundary_v_f == 4]
    seam_v_f_1 = mesh_atlas_f.vertices[idx_seam_v_f_1]
    seam_v_f_2 = mesh_atlas_f.vertices[idx_seam_v_f_2]

    highest_1_i = np.argmax(seam_v_f_1[:, 1].flatten())
    highest_2_i = np.argmax(seam_v_f_2[:, 1].flatten())

    offset = ((seam_v_f_1[highest_1_i] - seam_v_f_2[highest_2_i])**2).sum()*ratio
    return offset

def sewing_front_back(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, labels_f, labels_b, num_seams=2):

    idx_boundary_v_f, boundary_edges_f = select_boundary(mesh_pattern_f)
    idx_boundary_v_b, boundary_edges_b = select_boundary(mesh_pattern_b)
    boundary_edges_f = set([tuple(sorted(e)) for e in boundary_edges_f.tolist()])
    boundary_edges_b = set([tuple(sorted(e)) for e in boundary_edges_b.tolist()])
    label_boundary_v_f = labels_f[idx_boundary_v_f]
    label_boundary_v_b = labels_b[idx_boundary_v_b]

    idx_boundary_v = [idx_boundary_v_f, idx_boundary_v_b]
    boundary_edges = [boundary_edges_f, boundary_edges_b]
    label_boundary_v = [label_boundary_v_f, label_boundary_v_b]
    mesh_pattern = [mesh_pattern_f, mesh_pattern_b]
    labels = [labels_f, labels_b]

    idx_offset = len(mesh_pattern_f.vertices)
    faces_sewing = [mesh_atlas_f.faces, mesh_atlas_b.faces + idx_offset]
    seam_tops = []

    if garment == 'Skirt':
        faces_flip = [True, False]
    elif garment == 'Tshirt' or garment == 'Jacket':
        faces_flip = [True, False, False, False]
        horizontal = [False, False, True, True]
    elif garment == 'Trousers':
        faces_flip = [True, False, True, False]


    for seam_i in range(1, num_seams+1):
        if garment == 'Skirt':
            faces_seam_i = sewing_vertical(idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i)
        elif garment == 'Tshirt' or garment == 'Jacket':
            faces_seam_i = sewing_vertical_tshirt(idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, horizontal=horizontal[seam_i-1])
        elif garment == 'Trousers':
            if seam_i == 2 or seam_i == 3:
                faces_seam_i, seam_i_top = sewing_vertical(idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, return_seam_top=True)
                seam_tops.append(seam_i_top)
            else:
                faces_seam_i = sewing_vertical(idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i)

        if faces_flip[seam_i-1]:
            faces_seam_i = faces_seam_i[:,::-1]
        faces_sewing.append(faces_seam_i)

    if len(seam_tops) > 0:
        faces_extra = np.array([seam_tops[0]+[seam_tops[1][0]], [seam_tops[0][1]]+seam_tops[1][::-1]])
        faces_sewing.append(faces_extra)

    if garment == 'Skirt':
        z_offset = compute_offset(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.1)
        print(z_offset)
    elif garment == 'Tshirt' or garment == 'Jacket':
        z_offset = compute_offset_tshirt(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.01)
        print(z_offset)
    elif garment == 'Trousers':
        z_offset = compute_offset_trousers(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.1)
        print(z_offset)


    mesh_atlas_f.vertices[:, -1] += z_offset
    verts_sewing = np.concatenate((mesh_atlas_f.vertices, mesh_atlas_b.vertices), axis=0)
    faces_sewing = np.concatenate(faces_sewing, axis=0)
    mesh_sewing = trimesh.Trimesh(verts_sewing, faces_sewing, validate=False, process=False)

    labels_sewing = np.concatenate((labels_f, labels_b), axis=0)

    return mesh_sewing, labels_sewing

def concatenate_mesh(mesh_left, mesh_right):

    verts = np.concatenate((mesh_left.vertices, mesh_right.vertices), axis=0)
    faces = np.concatenate((mesh_left.faces, len(mesh_left.vertices) + mesh_right.faces), axis=0)

    mesh_new = trimesh.Trimesh(verts, faces, validate=False, process=False)
    return mesh_new

def erode_indicator(mask, size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    mask = cv2.erode(mask, kernel)
    return mask

def erode_bottom(mask, size):
    kernel = np.ones((size, 1), np.uint8)
    mask_copy = mask.copy()

    height, width = mask.shape
    mask_copy[height // 2:, :] = cv2.erode(mask[height // 2:, :], kernel)

    return mask_copy

def erode_up(mask, size):
    kernel = np.ones((size, 1), np.uint8)
    mask_copy = mask.copy()

    height, width = mask.shape

    mask_copy[:height // 2, :] = cv2.erode(mask[:height // 2, :], kernel)

    return mask_copy

def move_mask_and_trim(mask, last_mask, shift_pixels):
    mask1 = mask.copy()

    mask2 = np.zeros_like(mask1)

    mask2[0:-shift_pixels, :] = last_mask[shift_pixels:, :]

    to_remove = (mask1 > 0) & (mask2 == 0)

    mask1[to_remove] = 0

    return mask1

parser = argparse.ArgumentParser(description="Generate the back normal maps")
parser.add_argument("--garment", type=str, default='Skirt', help="The type of garment")
parser.add_argument("--scale", type=float, default=0.8, help="The scale of the garment")
parser.add_argument("--vid_name", type=str, default='vid_demo', help="The name of the video")

args = parser.parse_args()

vid_name = args.vid_name
using_repair = True
garment = args.garment
if garment == 'Skirt' or garment == 'Tshirt':
    numG = 100 
elif  garment == 'Trousers':
    numG = 239 
elif  garment == 'Jacket':
    numG = 146 
num_edges = 3 if garment == 'Skirt' else 5

scale = args.scale
rep_size = 32
model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+num_edges, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+num_edges, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
model_rep = SDF.learnt_representations(rep_size=rep_size, samples=numG).cuda()
model_atlas_f = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3], geometric_init=False).cuda()
model_atlas_b = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3], geometric_init=False).cuda()

if garment == 'Skirt':
    model_sdf_f.load_state_dict(torch.load('../checkpoints/isp-skirt-fix/net_epoch_8999_id_sdf_f.pth'))
    model_sdf_b.load_state_dict(torch.load('../checkpoints/isp-skirt-fix/net_epoch_8999_id_sdf_b.pth'))
    model_rep.load_state_dict(torch.load('../checkpoints/isp-skirt-fix/net_epoch_8999_id_rep.pth'))
    model_atlas_f.load_state_dict(torch.load('../checkpoints/isp-skirt-fix/net_epoch_8999_id_atlas_f.pth'))
    model_atlas_b.load_state_dict(torch.load('../checkpoints/isp-skirt-fix/net_epoch_8999_id_atlas_b.pth'))
    num_seams = 2

x_res = y_res = 256
uv_vertices, uv_faces = create_uv_mesh(x_res, y_res, debug=False)
mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, process=False, validate=False)
edges = torch.LongTensor(mesh_uv.edges).cuda()
uv_vertices = torch.FloatTensor(uv_vertices).cuda()

latent_codes = model_rep.weights.detach()

load_folder = save_folder = f'../fitting-results/{vid_name}/uv-mapping-back-{scale}-{garment}'

eval_batch_size = 1
seq_len = 10
num_stride = 10

img_pred_uvs = sorted([img for img in os.listdir(load_folder) if img.startswith('img_pred_uv_f_0') and img.endswith('.png')])

mask = None
for index, img_pred_uv_name in enumerate(img_pred_uvs):
    
    frame = img_pred_uv_name.split('_')[-1].split('.')[0]
    
    mask_f = cv2.imread(os.path.join(save_folder, 'img_pred_uv_mask_f_%s.png'%(frame)))[:,:,0]
    mask_b = cv2.imread(os.path.join(save_folder, 'img_pred_uv_mask_b_%s.png'%(frame)))[:,:,0]
    
    if garment == "Tshirt":
        single_mask = (mask_f == 255).astype(int)
    else:
        single_mask = ((mask_f == 255) + (mask_b == 255)).astype(int)
    if mask is None:
        mask = single_mask
    else:
        mask = mask + single_mask
    
mask = (mask > len(img_pred_uvs)/10).astype(int)

print(f'{save_folder}: calculate isp')

save_path = os.path.join(save_folder, f'mask-all0.1')
os.makedirs(save_path, exist_ok=True)

mask *= 255 

W = mask.shape[1]
img_f = mask[:,:W//2]
img_b = mask[:,W//2:]

anchor_codes = latent_codes
weight_rep = 0.1/2 if garment == 'Skirt' else 0.02
weight_area = 0.5
latent_code, img_f_new, img_b_new, label_f, label_b = optimize_lat_code_anchors([model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b], anchor_codes, [img_f, img_b], uv_vertices, iters=500, weight_rep=weight_rep, weight_area=weight_area)

img_f_diff = vis_diff(img_f_new.copy(), img_f.copy())
img_b_diff = vis_diff(img_b_new.copy(), img_b.copy())

img_new = np.concatenate((img_f_new, img_b_new), axis=1)
label_new = np.concatenate((label_f, label_b), axis=1)
img_diff = np.concatenate((img_f_diff, img_b_diff), axis=1)

img_cat = cat_images(mask, img_new, img_diff)
cv2.imwrite(os.path.join(save_path, 'isp-fitting-difference.png'), img_cat)
cv2.imwrite(os.path.join(save_path, 'isp-fitting.png'), img_new)
if garment == 'Skirt':
    cv2.imwrite(os.path.join(save_path, 'label-fitting.png'), label_new*100)
elif garment == 'Tshirt' or garment == 'Jacket':
    cv2.imwrite(os.path.join(save_path, 'label-fitting.png'), label_new*50)

mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b, label_f, label_b = reconstruct_pattern_with_label([model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b], latent_code, uv_vertices, uv_faces, edges, resolution=x_res)

mesh_atlas_sewing, labels = sewing_front_back(mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, label_f, label_b, num_seams=num_seams)

mesh_pattern_f.export(os.path.join(save_path, 'pattern-f.ply'))
mesh_pattern_b.export(os.path.join(save_path, 'pattern-b.ply'))
mesh_atlas_f.export(os.path.join(save_path, 'atlas-f.ply'))
mesh_atlas_b.export(os.path.join(save_path, 'atlas-b.ply'))
mesh_atlas_sewing.export(os.path.join(save_path, 'sewing.ply'))
cv2.imwrite(os.path.join(save_path, 'combined_mask.png'), mask)

