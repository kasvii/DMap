import os, sys
import numpy as np
import torch

from pytorch3d.ops.knn import knn_gather, knn_points
from scipy.spatial.transform import Rotation as R

from utils.readfile import load_pkl

def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
    num_joints = poses.shape[-1] //3

    poses = poses.reshape((-1, num_joints, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

    poses[:, 23] *= 0.1
    poses[:, 22] *= 0.1

    return poses.reshape((poses.shape[0], -1))

def finite_diff(x, h, diff=1):
    if diff == 0:
        return x

    v = np.zeros(x.shape, dtype=x.dtype)
    v[1:] = (x[1:] - x[0:-1]) / h

    return finite_diff(v, h, diff-1)

def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2*np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances

def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)


def get_shape_matrix(x):
    if x.ndim == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif x.ndim == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError
    

def gather_triangles(vertices, indices):
    #if vertices.ndim == (indices.ndim + 1):
    #    indices = indices.unsqueeze(0).repeat(len(vertices), 1, 1)

    #triangles = tf.gather(vertices, indices,
    #                      axis=-2,
    #                      batch_dims=vertices.shape.ndims - 2)

    # indices: [num_faces, 3]
    # vertices: [batch_size, num_points, 3]
    num_faces = len(indices)
    triangles = vertices[:, indices.reshape(-1)]
    triangles = triangles.reshape(-1, num_faces, 3, 3)
    return triangles


def gather_triangles_batch(vertices, indices):
    #if vertices.ndim == (indices.ndim + 1):
    #    indices = indices.unsqueeze(0).repeat(len(vertices), 1, 1)

    #triangles = tf.gather(vertices, indices,
    #                      axis=-2,
    #                      batch_dims=vertices.shape.ndims - 2)

    # indices: [num_faces, 3]
    # vertices: [batch_size, num_points, 3]
    batch_size = indices.shape[0]
    num_faces = indices.shape[1]
    #print(vertices[0,[81,1235,3664]])
    #print(vertices.shape, indices.shape)
    #print(indices[0,0])
    indices = indices.reshape(batch_size, -1)
    indices = indices.unsqueeze(-1).repeat(1, 1, 3)
    #print(indices.shape)
    #print(indices[0,0])
    triangles = torch.gather(vertices, 1, indices)
    triangles = triangles.reshape(batch_size, num_faces, 3, 3)
    #print(triangles[0,0])
    #print(vertices[0,[81,1235,3664]])
    #sys.exit()
    return triangles



################################# Loss Computation #################################
#                                Modified from SNUG                                #
################################# Loss Computation #################################
from snug.snug_class import FaceNormals

def deformation_gradient(triangles, Dm_inv):
    # Dm_inv: [num_faces, 2, 2]
    Ds = get_shape_matrix(triangles)
    if Ds.ndim == 3:
        return torch.einsum('nij,njk->nik', Ds, Dm_inv)
    elif Ds.ndim == 4 and Dm_inv.ndim == 3:
        return torch.einsum('bnij,njk->bnik', Ds, Dm_inv)
    elif Ds.ndim == 4 and Dm_inv.ndim == 4:
        return torch.einsum('bnij,bnjk->bnik', Ds, Dm_inv)
    raise NotImplementedError
    #return Ds @ Dm_inv 


def green_strain_tensor(F):
    I = torch.eye(2, dtype=F.dtype, device=F.device)
    Ft = torch.permute(F, [0, 1, 3, 2])
    #Ft = tf.transpose(F, perm=[0, 1, 3, 2])
    #return 0.5*(Ft @ F - I)
    # print(F.shape, Ft.shape)
    # print(F.min(), F.max(), Ft.min(), Ft.max())
    # print(torch.isnan(F).any(), torch.isinf(F).any())
    # sys.exit()
    return (torch.einsum('bnij,bnjk->bnik', Ft, F) - I[None, None, :, :])*0.5


def stretching_energy(v, cloth, return_average=True, weight_f=None): 
    '''
    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant-Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]
    triangles = gather_triangles(v, cloth.f)

    #Dm_inv = tf.repeat([cloth.Dm_inv], tf.shape(v)[0], axis=0)
    Dm_inv = cloth.Dm_inv.clone()

    F = deformation_gradient(triangles, Dm_inv)
    G = green_strain_tensor(F)

    # Energy
    '''
    mat = cloth.material
    I = tf.eye(2, batch_shape=tf.shape(G)[:2], dtype=G.dtype)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * tf.linalg.trace(G)[:, :, tf.newaxis, tf.newaxis] * I    ### * element-wise mul
    energy_density = tf.linalg.trace(tf.transpose(S, [0, 1, 3, 2]) @ G)                                   ### @ matrix mul, output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])
    energy = cloth.f_area[tf.newaxis] * mat.thickness * energy_density
    '''
    mat = cloth.material
    I = torch.eye(2, dtype=G.dtype, device=G.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, G.shape[1], 1, 1)
    trace_G = torch.einsum('bnii->bn', G)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * trace_G[:, :, None, None] * I
    energy_density = torch.einsum('bnij,bnjk->bnik', torch.permute(S, [0, 1, 3, 2]), G)
    energy_density = torch.einsum('bnii->bn', energy_density)
    energy = cloth.f_area[None] * mat.thickness * energy_density
    '''
    print(len(energy[energy>10]), len(energy[0]))
    print(energy[energy>10])
    print(energy_density[energy>10]) 
    print(cloth.f_area.unsqueeze(0)[energy>10])
    print(energy[0,0])
    print(energy_density[0,0])
    print(cloth.f_area.unsqueeze(0)[0,0])
    print(energy.max())
    '''

    if not (weight_f is None):
        energy = energy*weight_f

    if return_average:
        return energy.sum() / batch_size
    
    return energy.sum(dim=-1)

def spring_energy(v, cloth, return_average=True): 
    '''
    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant-Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''

    edges = v[cloth.e.reshape(-1)].reshape(-1,2,3)
    edges_length = (edges[:,0] - edges[:,1]).norm(p=2,dim=-1)

    energy = (edges_length - cloth.e_rest).abs()
    
    return energy.sum(dim=-1)


def bending_energy(v, cloth, return_average=True): 
    '''
    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]

    # Compute face normals
    fn = FaceNormals().call(v, cloth.f)
    #print(fn.shape)
    #print(fn[0,0])
    #sys.exit()
    
    #n0 = tf.gather(fn, cloth.f_connectivity[:, 0], axis=1)
    #n1 = tf.gather(fn, cloth.f_connectivity[:, 1], axis=1)
    face0_idx = cloth.f_connectivity[:, 0]
    face1_idx = cloth.f_connectivity[:, 1]
    n0 = fn[:, face0_idx]
    n1 = fn[:, face1_idx]

    # Compute edge length
    #v0 = tf.gather(v, cloth.f_connectivity_edges[:, 0], axis=1)
    #v1 = tf.gather(v, cloth.f_connectivity_edges[:, 1], axis=1)
    #e = v1 - v0
    #e_norm, l = tf.linalg.normalize(e, axis=-1)
    v0_idx = cloth.f_connectivity_edges[:, 0]
    v1_idx = cloth.f_connectivity_edges[:, 1]
    e = v[:, v1_idx] - v[:, v0_idx]
    l = torch.norm(e, p=2, dim=-1, keepdim=True)
    e_norm = e/l

    # Compute area
    #f_area = tf.repeat([cloth.f_area], tf.shape(v)[0], axis=0)
    #a0 = tf.gather(f_area, cloth.f_connectivity[:, 0], axis=1)
    #a1 = tf.gather(f_area, cloth.f_connectivity[:, 1], axis=1)
    #a = a0 + a1
    f_area = cloth.f_area.unsqueeze(0).repeat(batch_size, 1)
    a0 = f_area[:, face0_idx]
    a1 = f_area[:, face1_idx]
    a = a0 + a1

    # Compute dihedral angle between faces
    #cos = tf.reduce_sum(tf.multiply(n0, n1), axis=-1)
    #sin = tf.reduce_sum(tf.multiply(e_norm, tf.linalg.cross(n0, n1)), axis=-1)
    #theta = tf.math.atan2(sin, cos)
    ## theta = tf.math.acos(cos)
    cos = (n0 * n1).sum(dim=-1)
    sin = (e_norm * torch.cross(n0, n1, dim=-1)).sum(dim=-1)
    theta = torch.atan2(sin, cos)
    #theta = torch.acos(cos)
    
    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = cloth.material
    #scale = l[..., 0]**2 / (4*a)
    scale = l[:, :, 0]**2 / (4*a)

    '''
    print(torch.isnan(fn).sum())
    print(torch.isnan(e).sum())
    print(torch.isnan(e_norm).sum())
    print((l==0).sum())
    print(torch.isnan(f_area).sum())
    print(torch.isnan(cos).sum())
    print(torch.isnan(sin).sum())
    print(torch.isnan(theta).sum())
    print(torch.isnan(scale).sum())
    sys.exit()
    '''
    
    
    #print(e_norm.shape)
    #sys.exit()
    valid = torch.logical_and(~torch.isnan(theta), ~torch.isnan(e_norm[:,:,-1]))
    #valid = ~torch.isnan(theta)

    # Bending energy
    energy = mat.bending_coeff * scale * (theta ** 2) / 2
    energy = energy[valid]
    #print(mat.bending_coeff, scale.shape)
    #print(scale[0][:10].detach())
    #print(theta[0][:10].detach())
    #print(cos[0][:10].detach())
    #print(sin[0][:10].detach())
    #print(n0[0][:2].detach())
    #print(n1[0][:2].detach())
    #sys.exit()

    if return_average:
        return energy.sum() / batch_size

    return energy.sum(dim=-1)

def dihedral_angle(v, cloth):
    # Compute face normals
    fn = FaceNormals().call(v, cloth.f)

    face0_idx = cloth.f_connectivity[:, 0]
    face1_idx = cloth.f_connectivity[:, 1]
    n0 = fn[:, face0_idx]
    n1 = fn[:, face1_idx]
    
    v0_idx = cloth.f_connectivity_edges[:, 0]
    v1_idx = cloth.f_connectivity_edges[:, 1]
    e = v[:, v1_idx] - v[:, v0_idx]
    l = torch.norm(e, p=2, dim=-1, keepdim=True)
    e_norm = e/l

    cos = (n0 * n1).sum(dim=-1)
    sin = (e_norm * torch.cross(n0, n1, dim=-1)).sum(dim=-1)
    theta = torch.atan2(sin, cos)

    return theta, l

def bending_energy_supervised(v, theta_rest, cloth, return_average=True): 
    '''
    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]

    
    face0_idx = cloth.f_connectivity[:, 0]
    face1_idx = cloth.f_connectivity[:, 1]
    f_area = cloth.f_area.unsqueeze(0).repeat(batch_size, 1)
    a0 = f_area[:, face0_idx]
    a1 = f_area[:, face1_idx]
    a = a0 + a1

    theta, l = dihedral_angle(v, cloth)
    
    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = cloth.material
    #scale = l[..., 0]**2 / (4*a)
    scale = l[:, :, 0]**2 / (4*a)

    # Bending energy
    energy = mat.bending_coeff * scale * ((theta-theta_rest) ** 2) / 2
    #print(energy.shape)
    #print(energy[0,:10])
    #print(theta[0,:10])
    #print(theta_rest[0,:10])
    #sys.exit()

    if return_average:
        return energy.sum() / batch_size

    return energy.sum(dim=-1)

def bending_energy_collar(v, theta_rest, cloth, idx_adjacent_faces_collar, return_average=True): 
    '''
    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]

    
    face0_idx = cloth.f_connectivity[:, 0]
    face1_idx = cloth.f_connectivity[:, 1]
    f_area = cloth.f_area.unsqueeze(0).repeat(batch_size, 1)
    a0 = f_area[:, face0_idx]
    a1 = f_area[:, face1_idx]
    a = a0 + a1

    theta, l = dihedral_angle(v, cloth)
    
    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = cloth.material
    #scale = l[..., 0]**2 / (4*a)
    scale = l[:, :, 0]**2 / (4*a)

    # Bending energy
    energy = mat.bending_coeff * scale * ((theta-theta_rest) ** 2) / 2
    energy = energy[idx_adjacent_faces_collar]
    #print(energy.shape)
    #print(energy[0,:10])
    #print(theta[0,:10])
    #print(theta_rest[0,:10])
    #sys.exit()
    #print(energy.sum())
    #sys.exit()

    if return_average:
        return energy.sum() / batch_size

    return energy.sum(dim=-1)

def gravitational_energy(x, mass, g=9.81, return_average=True, shift_ground=False, offset=0, z=False):
    batch_size = x.shape[0]
    #U = g * mass[tf.newaxis, tf.newaxis] * x[:, :, 1]
    #U = g * mass[None, :] * x[:, :, 1]
    if shift_ground:
        x[:, :, 1] += offset
    U = g * mass[None, None, :] * x[:, :, 1]

    if z:
        U = g * mass[None, None, :] * x[:, :, -1]

    if return_average:
        return U.sum() / batch_size

    return U.sum(dim=-1)


def inertial_term(x, x_prev, v_prev, mass, time_step, return_average=True):
    batch_size = x.shape[0]
    
    x_hat = x_prev + time_step * v_prev
    x_diff = x - x_hat

    #print(mass.shape, x_diff.shape)
    #sys.exit()
    #num = tf.einsum('bvi,bvi->bv', x_diff, mass[:, tf.newaxis] * x_diff)
    num = torch.einsum('bvi,bvi->bv', x_diff, mass[None, :, None] * x_diff)
    den = 2 * time_step ** 2

    if return_average:
        return (num / den).sum() / batch_size

    return (num / den).sum(dim=-1)


def inertial_term_sequence(x, mass, time_step, return_average=True):
    """
    x: tf.Tensor of shape [batch_size, num_frames, num_vertices, 3]
    """
    #print(x.shape)
    batch_size = x.shape[0]
    num_vertices = x.shape[-2]

    # Compute velocities
    x_current = x[:, 1:]
    x_prev = x[:, :-1] 
    v = (x_current - x_prev) / time_step
    #zeros = tf.zeros([batch_size, 1, num_vertices, 3], x.dtype)
    #v_prev = tf.concat([zeros, v[:, :-1]], axis=1)   
    zeros = torch.zeros([batch_size, 1, num_vertices, 3], device=x.device)
    #print(v.shape, zeros.shape)
    #sys.exit()
    v_prev = torch.cat([zeros, v[:, :-1]], dim=1)   

    # Flatten
    #x_current = tf.reshape(x_current, [-1, num_vertices, 3])   
    #x_prev = tf.reshape(x_prev, [-1, num_vertices, 3])   
    #v_prev = tf.reshape(v_prev, [-1, num_vertices, 3])   
    x_current = torch.reshape(x_current, [-1, num_vertices, 3])   
    x_prev = torch.reshape(x_prev, [-1, num_vertices, 3])   
    v_prev = torch.reshape(v_prev, [-1, num_vertices, 3])   

    return inertial_term(x_current, x_prev, v_prev, mass, time_step, return_average)


#def collision_penalty(va, vb, nb, eps=2e-3, kcollision=250):#250): # eps=2e-3 ????
def collision_penalty(va, vb, nb, eps=2e-3, kcollision=2500):#250): # eps=2e-3 ????
    batch_size = va.shape[0]
    '''
    closest_vertices = NearestNeighbour(dtype=va.dtype)(va, vb)
    vb = tf.gather(vb, closest_vertices, batch_dims=1)
    nb = tf.gather(nb, closest_vertices, batch_dims=1)

    distance = tf.reduce_sum(nb*(va - vb), axis=-1) 
    interpenetration = tf.maximum(eps - distance, 0)
    '''
    vec = va[:, :, None] - vb[:, None] # [batch_size, num_vertices_a, num_vertices_b, 3]
    dist = torch.sum(vec**2, dim=-1)   # [batch_size, num_vertices_a, num_vertices_b]
    closest_vertices = torch.argmin(dist, dim=-1) # [batch_size, num_vertices_a] closest vertices of garment in body
    
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1,1,3)
    vb = torch.gather(vb, 1, closest_vertices)
    nb = torch.gather(nb, 1, closest_vertices)

    distance = (nb*(va - vb)).sum(dim=-1) 
    interpenetration = torch.nn.functional.relu(eps - distance)

    return (interpenetration**3).sum() / batch_size * kcollision
    #return (interpenetration**2).sum() / batch_size * kcollision
    
    
def collision_penalty_mse(va, vb, nb, eps=2e-3, kcollision=2500):#250): # eps=2e-3 ????
    batch_size = va.shape[0]
    '''
    closest_vertices = NearestNeighbour(dtype=va.dtype)(va, vb)
    vb = tf.gather(vb, closest_vertices, batch_dims=1)
    nb = tf.gather(nb, closest_vertices, batch_dims=1)

    distance = tf.reduce_sum(nb*(va - vb), axis=-1) 
    interpenetration = tf.maximum(eps - distance, 0)
    '''
    vec = va[:, :, None] - vb[:, None] # [batch_size, num_vertices_a, num_vertices_b, 3]
    dist = torch.sum(vec**2, dim=-1)   # [batch_size, num_vertices_a, num_vertices_b]
    closest_vertices = torch.argmin(dist, dim=-1) # [batch_size, num_vertices_a] closest vertices of garment in body
    
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1,1,3)
    vb = torch.gather(vb, 1, closest_vertices)
    nb = torch.gather(nb, 1, closest_vertices)

    distance = (nb*(va - vb)).sum(dim=-1) 
    interpenetration = torch.nn.functional.relu(eps - distance)

    return (interpenetration**2).sum() / batch_size * kcollision


def collision_penalty_lite(va, vb, nb, eps=2e-3, kcollision=2500):
    batch_size = va.shape[0]
    num_vertices = va.shape[1]
    
    # 找到每个garment顶点的最近body顶点
    vec = va[:, :, None] - vb[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1, 1, 3)
    
    # 获取最近body顶点和法线
    vb_closest = torch.gather(vb, 1, closest_vertices)
    nb_closest = torch.gather(nb, 1, closest_vertices)
    
    # 计算garment顶点到body表面的距离（带符号）
    distance = (nb_closest * (va - vb_closest)).sum(dim=-1)
    
    # 只对穿透body表面的顶点施加惩罚
    interpenetration = torch.nn.functional.relu(eps - distance)
    
    # 使用二次惩罚（更平滑）
    penalty = (interpenetration**2).sum() / (batch_size * num_vertices) * kcollision
    
    return penalty

def collision_penalty_lite_save(va, vb, nb, eps=2e-3, kcollision=2500):
    """
    va: [B, N, 3]  garment vertices
    vb: [B, M, 3]  body vertices
    nb: [B, M, 3]  body normals
    """
    batch_size, num_vertices = va.shape[:2]

    with torch.no_grad():
        dist = torch.cdist(va, vb)  # [B, N, M]
        closest_idx = dist.argmin(dim=-1)  # [B, N]

    # Get vb_closest and nb_closest
    idx = closest_idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, N, 3]
    vb_closest = torch.gather(vb, 1, idx)
    nb_closest = torch.gather(nb, 1, idx)

    # Compute signed distance
    distance = (nb_closest * (va - vb_closest)).sum(dim=-1)

    # Symmetric penalty
    interpenetration = torch.nn.functional.relu(eps - distance)
    penalty = (interpenetration ** 2).sum() / (batch_size * num_vertices) * kcollision

    return penalty

def waist_penalty_lite(va, vb, nb, eps=2e-3, kcollision=2500):
    batch_size = va.shape[0]
    num_vertices = va.shape[1]
    
    # 找到每个garment顶点的最近body顶点
    vec = va[:, :, None] - vb[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    closest_vertices = closest_vertices.unsqueeze(-1).repeat(1, 1, 3)
    
    # 获取最近body顶点和法线
    vb_closest = torch.gather(vb, 1, closest_vertices)
    nb_closest = torch.gather(nb, 1, closest_vertices)
    
    # 计算garment顶点到body表面的距离（带符号）
    distance = (nb_closest * (va - vb_closest)).sum(dim=-1)
    
    interpenetration = torch.abs(distance) # - eps # torch.nn.functional.relu(distance - eps)
    
    # 使用二次惩罚（更平滑）
    penalty = (interpenetration**2).sum() / (batch_size * num_vertices) * kcollision
    
    return penalty

def shrink_penalty(va, vb):

    vec = va[:, None] - vb[None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    vb = vb[closest_vertices]
    distance = (va-vb).norm(p=2, dim=-1)

    return distance.mean()

partsIdx = load_pkl('/scratch/cvlab/home/ren/code/cloth-from-image/extra-data/parts14_vertex_idx.pkl')
idx_leg = partsIdx[3]+partsIdx[4]+partsIdx[5]+partsIdx[6]+partsIdx[7]+partsIdx[8]
idx_arm = partsIdx[1]+partsIdx[2]+partsIdx[9]+partsIdx[10]+partsIdx[11]+partsIdx[12]
idx_leg = torch.LongTensor(idx_leg).cuda()
idx_arm = torch.LongTensor(idx_arm).cuda()
indicator_leg = torch.zeros(6890).cuda().bool()
indicator_arm = torch.zeros(6890).cuda().bool()
indicator_leg[idx_leg] = 1
indicator_arm[idx_arm] = 1
indicator_rest = torch.ones(6890).cuda().bool()
indicator_rest[idx_leg] = 0
indicator_rest[idx_arm] = 0
def collision_penalty_skirt(va, na, vb, nb, eps=2e-3, kcollision=2500):#250): # eps=2e-3 ????

    vec = va[:, None] - vb[None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    indicator_leg_closest = indicator_leg[closest_vertices]
    
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]
    
    distance_not_leg = (nb[~indicator_leg_closest]*(va[~indicator_leg_closest] - vb[~indicator_leg_closest])).sum(dim=-1)
    interpenetration_not_leg = torch.nn.functional.relu(eps - distance_not_leg)
    distance_leg = (nb[indicator_leg_closest]*(va[indicator_leg_closest] - vb[indicator_leg_closest])).sum(dim=-1)
    interpenetration_leg = torch.nn.functional.relu(eps - distance_leg)


    indicator_flip = torch.logical_and((na[indicator_leg_closest]*nb[indicator_leg_closest]).sum(dim=-1) < 0, distance_leg < 0)
    sign = torch.ones_like(interpenetration_leg)
    sign[indicator_flip] = -1
    interpenetration_leg = interpenetration_leg*sign

    interpenetration = (interpenetration_not_leg**3).sum() + (interpenetration_leg**3).sum()

    return interpenetration * kcollision

partsIdx = np.load('/scratch/cvlab/home/ren/code/cloth-from-video/extra-data/color_smplx.npy')
idx_leg = (partsIdx == 5) + (partsIdx == 6) + (partsIdx == 7) + (partsIdx == 8) + (partsIdx == 9) + (partsIdx == 10)
idx_arm = (partsIdx == 3) + (partsIdx == 4) + (partsIdx == 11) + (partsIdx == 12) + (partsIdx == 13) + (partsIdx == 14)
idx_leg = torch.LongTensor(idx_leg).cuda()
idx_arm = torch.LongTensor(idx_arm).cuda()
indicator_leg_x = torch.zeros(len(partsIdx)).cuda().bool()
indicator_arm_x = torch.zeros(len(partsIdx)).cuda().bool()
indicator_leg_x[idx_leg] = 1
indicator_arm_x[idx_arm] = 1
indicator_rest_x = torch.ones(len(partsIdx)).cuda().bool()
indicator_rest_x[idx_leg] = 0
indicator_rest_x[idx_arm] = 0
def collision_penalty_skirt_smplx(va, na, vb, nb, eps=2e-3, kcollision=2500):#250): # eps=2e-3 ????

    vec = va[:, None] - vb[None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    indicator_leg_closest = indicator_leg_x[closest_vertices]
    
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]
    
    distance_not_leg = (nb[~indicator_leg_closest]*(va[~indicator_leg_closest] - vb[~indicator_leg_closest])).sum(dim=-1)
    interpenetration_not_leg = torch.nn.functional.relu(eps - distance_not_leg)
    distance_leg = (nb[indicator_leg_closest]*(va[indicator_leg_closest] - vb[indicator_leg_closest])).sum(dim=-1)
    interpenetration_leg = torch.nn.functional.relu(eps - distance_leg)


    indicator_flip = torch.logical_and((na[indicator_leg_closest]*nb[indicator_leg_closest]).sum(dim=-1) < 0, distance_leg < 0)
    sign = torch.ones_like(interpenetration_leg)
    sign[indicator_flip] = -1
    interpenetration_leg = interpenetration_leg*sign

    interpenetration = (interpenetration_not_leg**3).sum() + (interpenetration_leg**3).sum()

    return interpenetration * kcollision


def collision_penalty_body_toBottom(va, na, vb, nb, eps=2e-3, kcollision=2500):#250): # eps=2e-3 ????

    vec = va[:, None] - vb[None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    indicator_leg_closest = indicator_leg[closest_vertices]
    
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]
    
    distance_not_leg = (nb[~indicator_leg_closest]*(va[~indicator_leg_closest] - vb[~indicator_leg_closest])).sum(dim=-1)
    interpenetration_not_leg = torch.nn.functional.relu(eps - distance_not_leg)
    distance_leg = (nb[indicator_leg_closest]*(va[indicator_leg_closest] - vb[indicator_leg_closest])).sum(dim=-1)
    interpenetration_leg = torch.nn.functional.relu(eps - distance_leg)


    indicator_flip = torch.logical_and((na[indicator_leg_closest]*nb[indicator_leg_closest]).sum(dim=-1) < 0, distance_leg < 0)
    sign = torch.ones_like(interpenetration_leg)
    sign[indicator_flip] = -1
    interpenetration_leg = interpenetration_leg*sign

    interpenetration = (interpenetration_not_leg**3).sum() + (interpenetration_leg**3).sum()

    return interpenetration * kcollision

def collision_penalty_skirt_arm(va, na, vb, nb, eps=2e-3, kcollision=2500):#250): # eps=2e-3 ????

    vec = va[:, None] - vb[None]
    dist = torch.sum(vec**2, dim=-1)
    closest_vertices = torch.argmin(dist, dim=-1)
    indicator_rest_closest = indicator_rest[closest_vertices]
    indicator_leg_closest = indicator_leg[closest_vertices]
    indicator_arm_closest = indicator_arm[closest_vertices]
    
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]
    
    distance_rest = (nb[indicator_rest_closest]*(va[indicator_rest_closest] - vb[indicator_rest_closest])).sum(dim=-1)
    interpenetration_rest = torch.nn.functional.relu(eps - distance_rest)
    distance_leg = (nb[indicator_leg_closest]*(va[indicator_leg_closest] - vb[indicator_leg_closest])).sum(dim=-1)
    interpenetration_leg = torch.nn.functional.relu(eps - distance_leg)
    distance_arm = (nb[indicator_arm_closest]*(va[indicator_arm_closest] - vb[indicator_arm_closest])).sum(dim=-1)
    interpenetration_arm = torch.nn.functional.relu(eps - distance_arm)


    indicator_flip = torch.logical_and((na[indicator_leg_closest]*nb[indicator_leg_closest]).sum(dim=-1) < 0, distance_leg < 0)
    sign = torch.ones_like(interpenetration_leg)
    sign[indicator_flip] = -1
    interpenetration_leg = interpenetration_leg*sign

    indicator_flip = torch.logical_and((na[indicator_arm_closest]*nb[indicator_arm_closest]).sum(dim=-1) > 0, distance_arm < 0)
    sign = torch.ones_like(interpenetration_arm) 
    sign[indicator_flip] = -1
    interpenetration_arm = interpenetration_arm*sign

    interpenetration = (interpenetration_rest**3).sum() + (interpenetration_leg**3).sum() + (interpenetration_arm**3).sum()

    return interpenetration * kcollision

########################################################################################################################
# batch loss
def inertial_term_batch(x, x_prev, v_prev, mass, time_step, return_average=True):
    batch_size = x.shape[0]*x.shape[1]
    
    x_hat = x_prev + time_step * v_prev
    x_diff = x - x_hat

    #print(mass.shape, x_diff.shape)
    #sys.exit()
    #num = tf.einsum('bvi,bvi->bv', x_diff, mass[:, tf.newaxis] * x_diff)
    num = torch.einsum('btvi,btvi->btv', x_diff, mass[:, None, :, None] * x_diff)
    #print(num.shape)
    den = 2 * time_step ** 2

    if return_average:
        return (num / den).sum() / batch_size

    return num / den

def inertial_term_sequence_batch(x, mass, indicator_v, time_step, return_average=True):
    """
    x: tf.Tensor of shape [batch_size, num_frames, num_vertices, 3]
    """
    #print(x.shape)
    batch_size = x.shape[0]
    num_vertices = x.shape[-2]

    # Compute velocities
    x_current = x[:, 1:]
    x_prev = x[:, :-1] 
    #print(x_current.shape, x_prev.shape, time_step)
    v = (x_current - x_prev) / time_step
    #zeros = tf.zeros([batch_size, 1, num_vertices, 3], x.dtype)
    #v_prev = tf.concat([zeros, v[:, :-1]], axis=1)   
    zeros = torch.zeros([batch_size, 1, num_vertices, 3], device=x.device)
    #print(v.shape, zeros.shape)
    #sys.exit()
    v_prev = torch.cat([zeros, v[:, :-1]], dim=1)   

    # Flatten
    #x_current = tf.reshape(x_current, [-1, num_vertices, 3])   
    #x_prev = tf.reshape(x_prev, [-1, num_vertices, 3])   
    #v_prev = tf.reshape(v_prev, [-1, num_vertices, 3])   
    '''
    x_current = torch.reshape(x_current, [-1, num_vertices, 3])   
    x_prev = torch.reshape(x_prev, [-1, num_vertices, 3])   
    v_prev = torch.reshape(v_prev, [-1, num_vertices, 3])   
    '''

    #mass_reshape = mass.unsqueeze(1).repeat(1, 2, 1).reshape(-1, num_vertices)
    #inertia = inertial_term_batch(x_current, x_prev, v_prev, mass, time_step, return_average=False)

    x_hat = x_prev + time_step * v_prev
    x_diff = x_current - x_hat
    num = torch.einsum('btvi,btvi->btv', x_diff, mass[:, None, :, None] * x_diff)
    den = 2 * time_step ** 2
    inertia = num / den

    
    indicator_v_expand = indicator_v.unsqueeze(1).repeat(1, 2, 1)
    inertia = inertia*indicator_v_expand
    inertia = inertia[indicator_v_expand]
    if return_average:
        return inertia.sum() / batch_size / 2
    
    return inertia.sum(dim=-1)


def stretching_energy_batch(v, f_batch, Dm_inv_batch, f_area_batch, mat, indicator_f, return_average=True): 
    '''
    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant-Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]
    #triangles = gather_triangles(v, cloth.f)
    triangles = gather_triangles_batch(v, f_batch)

    #Dm_inv = tf.repeat([cloth.Dm_inv], tf.shape(v)[0], axis=0)
    #Dm_inv = cloth.Dm_inv.clone()

    F = deformation_gradient(triangles, Dm_inv_batch)
    G = green_strain_tensor(F)

    # Energy
    #mat = cloth.material
    I = torch.eye(2, dtype=G.dtype, device=G.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, G.shape[1], 1, 1)
    trace_G = torch.einsum('bnii->bn', G)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * trace_G[:, :, None, None] * I
    energy_density = torch.einsum('bnij,bnjk->bnik', torch.permute(S, [0, 1, 3, 2]), G)
    energy_density = torch.einsum('bnii->bn', energy_density)
    #energy = cloth.f_area[None] * mat.thickness * energy_density
    energy = f_area_batch * mat.thickness * energy_density
    #print(f_area_batch.shape, energy_density.shape)
    #print(energy.shape, indicator_f.shape)
    energy = energy*indicator_f
    energy = energy[indicator_f]
    #sys.exit()

    if return_average:
        return energy.sum() / batch_size
    
    return energy.sum(dim=-1)

def bending_energy_batch(v, f_batch, f_connectivity_batch, f_connectivity_edges_batch, f_area_batch, indicator_f_connect, mat, theta_rest=None, return_average=True): 
    '''
    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    batch_size = v.shape[0]

    # Compute face normals
    fn = FaceNormals().call_batch(v, f_batch)
    #fn[torch.isnan(fn)] = 0
    #nan_mask = torch.isnan(fn)
    #print(nan_mask.sum())
    
    #face0_idx = cloth.f_connectivity[:, 0]
    #face1_idx = cloth.f_connectivity[:, 1]
    face0_idx = f_connectivity_batch[:, :, 0]
    face1_idx = f_connectivity_batch[:, :, 1]
    #n0 = fn[:, face0_idx]
    #n1 = fn[:, face1_idx]
    n0 = torch.gather(fn, 1, face0_idx.unsqueeze(-1).repeat(1, 1, 3))
    n1 = torch.gather(fn, 1, face1_idx.unsqueeze(-1).repeat(1, 1, 3))

    # Compute edge length
    #v0_idx = cloth.f_connectivity_edges[:, 0]
    #v1_idx = cloth.f_connectivity_edges[:, 1]
    v0_idx = f_connectivity_edges_batch[:, :, 0]
    v1_idx = f_connectivity_edges_batch[:, :, 1]
    #e = v[:, v1_idx] - v[:, v0_idx]
    v0 = torch.gather(v, 1, v0_idx.unsqueeze(-1).repeat(1, 1, 3))
    v1 = torch.gather(v, 1, v1_idx.unsqueeze(-1).repeat(1, 1, 3))
    e = v1 - v0
    l = torch.norm(e, p=2, dim=-1, keepdim=True)
    e_norm = e/l

    # Compute area
    #f_area = cloth.f_area.unsqueeze(0).repeat(batch_size, 1)
    #a0 = f_area[:, face0_idx]
    #a1 = f_area[:, face1_idx]
    a0 = torch.gather(f_area_batch, 1, face0_idx)
    a1 = torch.gather(f_area_batch, 1, face1_idx)
    a = a0 + a1

    # Compute dihedral angle between faces
    cos = (n0 * n1).sum(dim=-1)
    sin = (e_norm * torch.cross(n0, n1, dim=-1)).sum(dim=-1)
    theta = torch.atan2(sin, cos)
    #theta = torch.acos(cos)
    #print(torch.isnan(sin).sum(), torch.isnan(cos).sum())
    
    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    #mat = cloth.material
    #print(a.shape, l.shape)
    #sys.exit()
    scale = l[:, :, 0]**2 / (4*a)

    #nan_mask = torch.isnan(scale)
    #print(indicator_f_connect[nan_mask].sum(), nan_mask.sum())
    # Bending energy
    if theta_rest is None:
        energy = mat.bending_coeff * scale * (theta ** 2) / 2
    else:
        energy = mat.bending_coeff * scale * ((theta-theta_rest) ** 2) / 2

    #nan_mask = torch.isnan(energy)
    #print(indicator_f_connect[nan_mask].sum(), nan_mask.sum())
    #sys.exit()
    #print(energy.shape, indicator_f_connect.shape)
    energy = energy*indicator_f_connect
    energy = energy[indicator_f_connect]
    #sys.exit()

    if return_average:
        return energy.sum() / batch_size

    return energy.sum(dim=-1)

def gravitational_energy_batch(x, mass_batch, indicator_v, g=9.81, return_average=True):
    batch_size = x.shape[0]
    #U = g * mass[None, None, :] * x[:, :, 1]
    U = g * mass_batch * x[:, :, 1]

    #print(U.shape, indicator_v.shape)
    U = U*indicator_v
    U = U[indicator_v]
    #sys.exit()
    if return_average:
        return U.sum() / batch_size

    return U.sum(dim=-1)

def collision_penalty_batch(va, vb, nb, indicator_v, eps=2e-3, kcollision=2500, mean=False):#250): # eps=2e-3 ????
    batch_size = va.shape[0]
    
    with torch.no_grad():
        '''
        vec = va[:, :, None] - vb[:, None]
        dist = torch.sum(vec**2, dim=-1)
        closest_vertices = torch.argmin(dist, dim=-1)
        
        closest_vertices = closest_vertices.unsqueeze(-1).repeat(1,1,3)
        vb = torch.gather(vb, 1, closest_vertices)
        nb = torch.gather(nb, 1, closest_vertices)
        '''
        va_nn = knn_points(va, vb, K=1)
        nb = knn_gather(nb, va_nn.idx)[..., 0, :]
        vb = knn_gather(vb, va_nn.idx)[..., 0, :]
        #print(nb.shape, vb.shape, va.shape)
        #sys.exit()
        del va_nn

    distance = (nb*(va - vb)).sum(dim=-1) 
    interpenetration = torch.nn.functional.relu(eps - distance)
    #print(interpenetration.shape, indicator_v.shape)
    #sys.exit()
    interpenetration = interpenetration*indicator_v
    interpenetration = interpenetration[indicator_v]
    #sys.exit()

    #return (interpenetration**3).sum() / batch_size * kcollision
    if mean:
        return (interpenetration**3).mean() / batch_size * kcollision
    else:
        return (interpenetration**3).sum() / batch_size * kcollision

def pin_penalty_batch(x, flag_torsor, weight=10):
    _B = len(x)
    x_pin = x[flag_torsor]
    #print(x_pin.shape)
    #sys.exit()
    #loss = (x_pin**2).sum()/self._B * weight
    loss = (x_pin[:,1]**2).sum()/_B * weight
    loss += (x_pin[:,0]**2).sum()/_B
    loss += (x_pin[:,2]**2).sum()/_B
    return loss/3

def layer_penalty(down_deform, top_deform, verts_body, lamdba=0.8):
    _B = len(down_deform)
    with torch.no_grad():
        #pose = self.rest_pose.repeat(self._B, 1)
        #_, _, verts_body = self.infer_smpl(pose, beta)

        vec = verts_body[:, :, None] - top_deform[:, None]
        dist = torch.sum(vec**2, dim=-1)
        closest_dist_top, closest_top_vertices = torch.min(dist, dim=-1)
        #print(closest_top_vertices.shape)

        vec_reverse = top_deform[:, :, None] - verts_body[:, None]
        dist_reverse = torch.sum(vec_reverse**2, dim=-1)
        closest_top_reverse_vertices = torch.argmin(dist_reverse, dim=-1)
        #print(closest_top_reverse_vertices.shape)
        body_vertices_mask = torch.zeros_like(closest_top_vertices).bool()
        for b in range(_B):
            body_mask_idx = closest_top_reverse_vertices[b]
            body_vertices_mask[b][body_mask_idx] = 1

        closest_dist_top[~body_vertices_mask] = -100

    vec = down_deform[:, :, None] - verts_body[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_dist, closest_vertices = torch.min(dist, dim=-1)

    closest_dist_top_gt = torch.gather(closest_dist_top, -1, closest_vertices).detach()*lamdba#/3*2
    valid_idx = closest_dist_top_gt>0

    loss = torch.nn.functional.relu(closest_dist-closest_dist_top_gt)[valid_idx].sum()/_B*10

    return loss

def layer_penalty_batch(down_deform, top_deform, down_indicator_v, top_indicator_v, verts_body, lamdba=0.8):
    _B = len(down_deform)
    with torch.no_grad():
        #pose = self.rest_pose.repeat(self._B, 1)
        #_, _, verts_body = self.infer_smpl(pose, beta)

        top_deform[~top_indicator_v] = -100

        vec = verts_body[:, :, None] - top_deform[:, None]
        dist = torch.sum(vec**2, dim=-1)
        closest_dist_top, closest_top_vertices = torch.min(dist, dim=-1)
        #print(closest_top_vertices.shape)

        vec_reverse = top_deform[:, :, None] - verts_body[:, None]
        dist_reverse = torch.sum(vec_reverse**2, dim=-1)
        closest_top_reverse_vertices = torch.argmin(dist_reverse, dim=-1)
        #print(closest_top_reverse_vertices.shape)
        body_vertices_mask = torch.zeros_like(closest_top_vertices).bool()
        for b in range(_B):
            body_mask_idx = closest_top_reverse_vertices[b][top_indicator_v[b]]
            body_vertices_mask[b][body_mask_idx] = 1

        closest_dist_top[~body_vertices_mask] = -100

    vec = down_deform[:, :, None] - verts_body[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_dist, closest_vertices = torch.min(dist, dim=-1)

    closest_dist_top_gt = torch.gather(closest_dist_top, -1, closest_vertices).detach()*lamdba#/3*2
    valid_idx = torch.logical_and(closest_dist_top_gt>0, down_indicator_v)
    #print(valid_idx.shape, down_deform.shape, closest_dist.shape)

    loss = torch.nn.functional.relu(closest_dist-closest_dist_top_gt)[valid_idx].sum()/_B*10

    return loss

def layer_penalty_batch_switch(top_deform, down_deform, top_indicator_v, down_indicator_v, verts_body, lamdba=0.8):
    _B = len(down_deform)
    with torch.no_grad():
        #pose = self.rest_pose.repeat(self._B, 1)
        #_, _, verts_body = self.infer_smpl(pose, beta)

        top_deform[~top_indicator_v] = -100

        vec = verts_body[:, :, None] - top_deform[:, None]
        dist = torch.sum(vec**2, dim=-1)
        closest_dist_top, closest_top_vertices = torch.min(dist, dim=-1)
        #print(closest_top_vertices.shape)

        vec_reverse = top_deform[:, :, None] - verts_body[:, None]
        dist_reverse = torch.sum(vec_reverse**2, dim=-1)
        closest_top_reverse_vertices = torch.argmin(dist_reverse, dim=-1)
        #print(closest_top_reverse_vertices.shape)
        body_vertices_mask = torch.zeros_like(closest_top_vertices).bool()
        for b in range(_B):
            body_mask_idx = closest_top_reverse_vertices[b][top_indicator_v[b]]
            body_vertices_mask[b][body_mask_idx] = 1

        closest_dist_top[~body_vertices_mask] = -100

    vec = down_deform[:, :, None] - verts_body[:, None]
    dist = torch.sum(vec**2, dim=-1)
    closest_dist, closest_vertices = torch.min(dist, dim=-1)

    closest_dist_top_gt = torch.gather(closest_dist_top, -1, closest_vertices).detach()*lamdba#/3*2
    valid_idx = torch.logical_and(closest_dist_top_gt>0, down_indicator_v)
    #print(valid_idx.shape, down_deform.shape, closest_dist.shape)

    loss = torch.nn.functional.relu(closest_dist-closest_dist_top_gt)[valid_idx].sum()/_B*10

    return loss


def reg_batch(v, indicator_v, return_average=True): 

    batch_size = v.shape[0]
    
    loss_reg = v.pow(2).sum(dim=-1)*indicator_v
    loss_reg = loss_reg[indicator_v]
    
    if return_average:
        return loss_reg.sum() / batch_size

    return loss_reg.mean()


'''
def normal_penalty_batch(v, v_eft, f_batch, indicator_f, return_average=True): 

    batch_size = v.shape[0]

    # Compute face normals
    fn = FaceNormals().call_batch(v, f_batch)
    with torch.no_grad():
        fn_eft = FaceNormals().call_batch(v_eft, f_batch)
    
    loss = 1 - (fn*fn_eft).sum(dim=-1)

    loss = loss*indicator_f
    loss = loss[indicator_f]
    
    if return_average:
        #return loss.mean(dim=-1) / batch_size
        return loss.sum() / batch_size

    return loss.mean(dim=-1)
'''
def normal_penalty_batch(v, f_batch, fn_gt, indicator_f, knormal=10, return_average=True): 

    batch_size = v.shape[0]

    # Compute face normals
    fn = FaceNormals().call_batch(v, f_batch)
    
    loss = ((fn - fn_gt)**2).sum(dim=-1)
    loss = loss[indicator_f]
    
    if return_average:
        #return loss.mean(dim=-1) / batch_size
        #return loss.sum() / batch_size / (f_batch.shape[1]//2) * knormal
        #return loss.sum() / batch_size * knormal
        return loss.mean() * knormal

    return loss.mean(dim=-1) * knormal

def relative_location_batch(v_deform, dir_q, indicator_boundary_v, indicator_fit_v, indicator_v, krl=1, return_average=True): 

    batch_size = v_deform.shape[0]

    v_deform_res = v_deform - (v_deform*dir_q).sum(dim=-1, keepdim=True)*dir_q
    loss = (v_deform_res**2).sum(dim=-1)
    alpha = torch.zeros_like(loss)+0.5
    alpha[indicator_boundary_v] = 1000
    alpha[indicator_fit_v] = 1000
    alpha = alpha.detach()
    #loss_rl = (loss_rl*alpha).sum()
    #print(loss.shape)
    loss = (loss*alpha)[indicator_v]

    if return_average:
        return loss.sum() / batch_size * krl

    return loss.mean(dim=-1) * krl

def fit_region_batch(v_deform, dir_q, indicator_fit_v, kfit=1, return_average=True): 

    batch_size = v_deform.shape[0]

    loss = (v_deform*dir_q).sum(dim=-1)**2
    #print(loss.shape)
    loss = loss[indicator_fit_v]
    
    if return_average:
        #return loss.mean(dim=-1) / batch_size
        return loss.sum() / batch_size * kfit

    return loss.mean(dim=-1) * kfit

def edge_boundary_batch(v, boundary_e_idx, boundary_e_dir_gt, indicator_boundary_e, ke=1, return_average=True): 

    batch_size = v.shape[0]

    boundary_e_idx = boundary_e_idx.reshape(batch_size, -1)
    boundary_e_idx_repeat = boundary_e_idx.unsqueeze(-1).repeat(1,1,3).detach().cuda()
    
    edge = torch.gather(v, 1, boundary_e_idx_repeat)
    edge = edge.reshape(batch_size, -1, 2, 3)

    boundary_edge_update_dir = edge[:, :, 0] - edge[:, :, 1]
    boundary_edge_update_dir = boundary_edge_update_dir[:, :, [0,2]]
    boundary_edge_update_dir = boundary_edge_update_dir/torch.norm(boundary_edge_update_dir, dim=-1, keepdim=True, p=2)
    loss = ((boundary_edge_update_dir - boundary_e_dir_gt)**2).sum(dim=-1)

    loss = loss[indicator_boundary_e]
    
    if return_average:
        #return loss.mean(dim=-1) / batch_size
        return loss.mean() * ke

    return loss.mean(dim=-1) * ke