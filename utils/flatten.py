import os
import trimesh
import numpy as np
import math
from trimesh import transformations


def equal_values(idx1, idx2, name='constraint_symmetry', energy_coeff=100.0):

    dic_x = {name+'_x':{
                'type': 'equal_values',
                'energy_coeff': energy_coeff,
                'axis': 0,
                'opposite_sign': 1,
                'ordered_vertex_ids1': idx1,
                'ordered_vertex_ids2': idx2
                }
            }

    dic_y = {name+'_y':{
                'type': 'equal_values',
                'energy_coeff': energy_coeff,
                'axis': 1,
                'opposite_sign': 0,
                'ordered_vertex_ids1': idx1,
                'ordered_vertex_ids2': idx2
                }
            }

    dic = {**dic_x, **dic_y}
    return dic
    
    
def shared_value(idx, axis=0, energy_coeff=100.0, name='constraint'):  

    dic = {name: {
                'type': 'shared_value',
                'energy_coeff': energy_coeff,
                'axis': axis,
                'vertex_ids': idx
                }
          }

    return dic
    
def predefined_value(idx, value=0, axis=0, energy_coeff=100.0, name='constraint'): 

    dic = {name: {
                'type': 'predefined_value',
                'energy_coeff': energy_coeff,
                'axis': axis,
                'vertex_ids': idx,
                'values': [value]*len(idx)
                }
          }

    return dic

def predefined_values(idx, values=[], axis=0, energy_coeff=100.0, name='constraint'): 

    dic = {name: {
                'type': 'predefined_value',
                'energy_coeff': energy_coeff,
                'axis': axis,
                'vertex_ids': idx,
                'values': values
                }
          }

    return dic


def reorder_vertices(mesh):
    vertices = mesh.vertices
    faces = mesh.faces

    faces_flatten = faces.reshape(-1)
    mask = np.zeros((len(vertices))).astype(bool)
    mask[faces_flatten] = True
    vertices_reorder = vertices[mask]
    
    re_id = np.zeros((len(vertices))).astype(int) - 1
    re_id[mask] = np.arange(len(vertices_reorder))
    faces_reorder = re_id[faces_flatten].reshape(-1, 3)

    mapping_idx = sorted(np.where(re_id!=-1)[0].tolist())

    mapping_to_reorder = {}
    mapping_to_ori = {}
    for i in range(len(mapping_idx)):
        mapping_to_reorder[mapping_idx[i]] = i
        mapping_to_ori[i] = mapping_idx[i]

    mesh_new = trimesh.Trimesh(vertices_reorder, faces_reorder, process=False, validate=False)

    return mesh_new, mapping_to_reorder, mapping_to_ori, mapping_idx

def reorder_json(_json, mapping):
    
    for k, constrains in _json.items():
        for k_c, v in constrains.items():
            if '_ids' not in k_c:
                continue
            #print(k, k_c, v)

            v = [mapping[i] for i in v]
            constrains[k_c] = v
            _json[k] = constrains
    
    return _json

def reorder_path(path, mapping):
    path_reorder = [mapping[p] for p in path]
    return path_reorder


def map_back_mesh(mesh, mapping, num_v):
    vertices = np.zeros((num_v, 3))
    faces = np.zeros((len(mesh.faces), 3)).astype(int)

    for i in range(len(mesh.vertices)):
        vertices[mapping[i]] = mesh.vertices[i]

    for i in range(len(mesh.faces)):
        for j in range(3):
            faces[i, j] = mapping[mesh.faces[i, j]]
            
    mesh_new = trimesh.Trimesh(vertices, faces, process=False, validate=False)

    return mesh_new

def rotate_mesh(mesh, angle=math.pi/2, center=[0, 0, 0], direction=[1, 0, 0]):
    rot_matrix = transformations.rotation_matrix(angle, direction, center)
    mesh.apply_transform(rot_matrix)
    return mesh

def split_front(mesh):
    
    right = set(np.where(mesh.vertices[:,0] > 0)[0].flatten().tolist())
    left = set(np.where(mesh.vertices[:,0] < 0)[0].flatten().tolist())

    faces_left = []
    faces_right = []
    for i in range(len(mesh.faces)):
        f = mesh.faces[i]
        if f[0] in left:
            faces_left.append(i)
        else:
            faces_right.append(i)

    mesh_left = trimesh.Trimesh(mesh.vertices, mesh.faces[faces_left], validate=False, process=False)
    mesh_right = trimesh.Trimesh(mesh.vertices, mesh.faces[faces_right], validate=False, process=False)

    idx_v_left = np.unique(mesh.faces[faces_left].flatten())
    idx_v_right = np.unique(mesh.faces[faces_right].flatten())
    v_left = mesh.vertices[idx_v_left, 0].mean()
    v_right = mesh.vertices[idx_v_right, 0].mean()

    if v_left > v_right:
        tmp = mesh_left.copy()
        mesh_left = mesh_right.copy()
        mesh_right = tmp

    return mesh_left, mesh_right


def clean_path(path, mesh):
    path_new = path.copy()
    while 1:
        path_new = path_new + [path_new[0]]
        points = mesh.vertices[path_new, :2]
        results = find_intersection_point(points)

        if results is not None:
            i, j = results
            print('i, j: ', i, j, len(path))
            path_new_cand1 = path_new[:i+1] + path_new[j+1:-1]
            path_new_cand2 = path_new[i+1:j+1]
            path_new = path_new_cand1 if len(path_new_cand1) > len(path_new_cand2) else path_new_cand2
        else:
            path_new = path_new[:-1]
            break

    return path_new


def find_intersection_point(points):
    num_points = len(points)

    #print(calculate_intersection(points[0], points[1], points[num_points - 2], points[num_points - 1]))
    #print(points[0], points[1], points[num_points - 2], points[num_points - 1])
    #sys.exit()
    
    # Iterate over each segment of the line
    for i in range(num_points - 1):
        seg1_start = points[i]
        seg1_end = points[i + 1]
        
        # Check for intersection with other segments
        for j in range(i + 2, num_points - 1):
            if i==0 and j == (num_points - 2):
                continue

            seg2_start = points[j]
            seg2_end = points[j + 1]
            
            # Calculate the intersection point
            intersection = calculate_intersection(seg1_start, seg1_end, seg2_start, seg2_end)
            
            # Check if an intersection point exists
            if intersection is not None:
                return [i, j]
    
    # No intersection point found
    return None

def calculate_intersection(p1, p2, p3, p4):
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det((p1[0], p1[1]), (p2[0], p2[1])), det((p3[0], p3[1]), (p4[0], p4[1])))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    if not is_point_on_segment(x, y, p1, p2) or not is_point_on_segment(x, y, p3, p4):
        return None

    return x, y

def is_point_on_segment(x, y, p1, p2):
    min_x = min(p1[0], p2[0])
    max_x = max(p1[0], p2[0])
    min_y = min(p1[1], p2[1])
    max_y = max(p1[1], p2[1])

    return min_x <= x <= max_x and min_y <= y <= max_y