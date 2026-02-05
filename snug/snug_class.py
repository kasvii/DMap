import numpy as np
import torch
import trimesh

def get_shape_matrix(x):
    if x.ndim == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif x.ndim == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError

def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)

def get_vertex_connectivity(faces):
    '''
    Returns a list of unique edges in the mesh. 
    Each edge contains the indices of the vertices it connects
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    return torch.LongTensor(list(edges))


def get_face_connectivity(faces):
    '''
    Returns a list of adjacent face pairs
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    for key in G:
        #assert len(G[key]) < 3
        G[key] = G[key][:2]
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]
   
    return torch.LongTensor(adjacent_faces)


def get_face_connectivity_edges(faces):
    '''
    Returns a list of edges that connect two faces
    (i.e., all the edges except borders)
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_face_edges = []
    for key in G:
        #assert len(G[key]) < 3
        G[key] = G[key][:2]
        if len(G[key]) == 2:
            adjacent_face_edges += [list(key)]

    return torch.LongTensor(adjacent_face_edges)

def get_collar_body_connectivity_indicator(adjacent_face_edges, adjacent_faces, faces, idx_collar_v):
    indicator_idx = []
    for i in range(len(adjacent_face_edges)):
        edge = adjacent_face_edges[i]
        if edge[0] in idx_collar_v and edge[1] in idx_collar_v:
            edge = set(edge.tolist())
            face0 = faces[adjacent_faces[i][0]]
            face1 = faces[adjacent_faces[i][1]]
            face0 = set(face0.tolist())
            face1 = set(face1.tolist())
            v_rest_0 = list(face0 - edge)[0]
            v_rest_1 = list(face1 - edge)[0]
            if (v_rest_0 in idx_collar_v and v_rest_1 in idx_collar_v) or (v_rest_1 in idx_collar_v and v_rest_0 in idx_collar_v):
                indicator_idx.append(i)

    indicator = torch.zeros(len(adjacent_face_edges)).bool()
    indicator[indicator_idx] = 1
    return indicator


def get_vertex_mass(vertices, faces, density):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:,0], triangle_masses/3)
    np.add.at(vertex_masses, faces[:,1], triangle_masses/3)
    np.add.at(vertex_masses, faces[:,2], triangle_masses/3)

    return torch.FloatTensor(vertex_masses)


def get_face_areas(vertices, faces):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()

    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]

    u = v2 - v0
    v = v1 - v0

    return np.linalg.norm(np.cross(u, v), axis=-1) / 2.0


def get_edge_length(vertices, edges):
    #v0 = tf.gather(vertices, edges[:,0], axis=-2) 
    #v1 = tf.gather(vertices, edges[:,1], axis=-2) 
    v0_idx = edges[:, 0]
    v1_idx = edges[:, 1]
    if vertices.ndim == 3:
        v0 = vertices[:, v0_idx]
        v1 = vertices[:, v1_idx]
    elif vertices.ndim == 2:
        v0 = vertices[v0_idx]
        v1 = vertices[v1_idx]
    else:
        raise NotImplementedError
    return torch.norm(v0 - v1, p=2, dim=-1)


def load_obj(filename, tex_coords=False):
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()
            
            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces


def rotate_triangle(triangles):
    num_tri = len(triangles)
    triangles_rotated = torch.zeros_like(triangles)
    e1 = triangles[:, 0] - triangles[:, 2]
    e2 = triangles[:, 1] - triangles[:, 2]
    #print(triangles.shape, e1.shape, e2.shape)
    n = torch.cross(e1, e2, dim=-1)

    x = e1/torch.norm(e1, p=2, dim=-1, keepdim=True)
    n = n/torch.norm(n, p=2, dim=-1, keepdim=True)
    y = torch.cross(n, x, dim=-1)
    y = y/torch.norm(y, p=2, dim=-1, keepdim=True)

    coord_old = torch.stack([x, y, n], dim=-1)
    coord_new = torch.eye(3).unsqueeze(0).repeat(num_tri, 1, 1).cuda()
    matrix_rot = torch.einsum('nij,njk->nik', coord_new, coord_old.permute(0, 2, 1))

    e1_rot = torch.einsum('nij,nj->ni', matrix_rot, e1)
    e2_rot = torch.einsum('nij,nj->ni', matrix_rot, e2)
    #n_rot = torch.einsum('nij,nj->ni', matrix_rot, n)

    shape_matrix = torch.stack([e1_rot, e2_rot], dim=-1)
    return shape_matrix


##################################### Classes #####################################
#                                Modified from SNUG                               #
##################################### Classes #####################################
class Material_original:
    '''
    This class stores parameters for the StVK material model
    '''

    def __init__(self, density,       # Fabric density (kg / m2)
                       thickness,     # Fabric thickness (m)
                       young_modulus, 
                       poisson_ratio,
                       bending_multiplier=1.0,
                       stretch_multiplier=1.0):
                       
        self.density = density
        self.thickness = thickness
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

        self.bending_multiplier = bending_multiplier
        self.stretch_multiplier = stretch_multiplier

        # Bending and stretching coefficients (ARCSim)
        self.A = young_modulus / (1.0 - poisson_ratio**2)
        self.stretch_coeff = self.A
        self.stretch_coeff *= stretch_multiplier

        self.bending_coeff = self.A / 12.0 * (thickness ** 3) 
        self.bending_coeff *= bending_multiplier

        # Lame coefficients
        self.lame_mu =  0.5 * self.stretch_coeff * (1.0 - self.poisson_ratio)
        self.lame_lambda = self.stretch_coeff * self.poisson_ratio

class Material:
    '''
    This class stores parameters for the StVK material model
    '''

    def __init__(self, density=426,       # Fabric density (kg / m2)
                       thickness=4.7e-4     # Fabric thickness (m):
                ):
                       
        self.density = 426
        self.thickness = 0.47e-3 # 0.47 mm
        self.area_density = self.density*self.thickness

        self.young_modulus = 0.7e5
        self.poisson_ratio = 0.485
        self.stretch_multiplier = 1
        self.bending_multiplier = 50
        
        # Bending and stretching coefficients (ARCSim)
        self.A = self.young_modulus / (1.0 - self.poisson_ratio**2)
        self.stretch_coeff = self.A
        self.stretch_coeff *= self.stretch_multiplier
        
        self.bending_coeff = self.A / 12.0 * (self.thickness ** 3) 
        self.bending_coeff *= self.bending_multiplier
        #self.bending_coeff = 3.96e-5

        self.collision_coeff = 250

        # Lame coefficients
        self.lame_mu =  0.5 * self.stretch_coeff * (1.0 - self.poisson_ratio)
        self.lame_lambda = self.stretch_coeff * self.poisson_ratio
        #print(self.lame_mu, self.lame_lambda, self.bending_coeff)
        #sys.exit()
        #self.lame_mu =  2.36e4
        #self.lame_lambda = 4.44e4


class Cloth: 
    '''
    This class stores mesh and material information of the garment
    '''
    
    def __init__(self, path, material, dtype=torch.float32):
        self.dtype = dtype  
        self.material = material

        v, f, vm, fm = load_obj(path, tex_coords=True)

        v = torch.FloatTensor(v).cuda()
        f = torch.LongTensor(f).cuda()
        vm = torch.FloatTensor(vm).cuda()
        fm = torch.LongTensor(fm).cuda()

        # Vertex attributes
        self.v_template = v
        self.v_mass = get_vertex_mass(v, f, self.material.area_density).cuda()
        #print(self.v_mass.shape, v.shape, vm.shape, f.shape, fm.shape)
        #sys.exit()
        self.v_velocity = torch.zeros(1, v.shape[0], 3).cuda() # Vertex velocities in global coordinates
        self.v = torch.zeros(1, v.shape[0], 3).cuda() # Vertex position in global coordinates
        self.v_psd = torch.zeros(1, v.shape[0], 3).cuda() # Pose space deformation of each vertex
        self.v_weights = None # Vertex skinning weights
        self.num_vertices = self.v_template.shape[0]
    
        # Face attributes
        self.f = f
        self.f_connectivity = get_face_connectivity(f).cuda() # Pairs of adjacent faces
        self.f_connectivity_edges = get_face_connectivity_edges(f).cuda() # Edges that connect faces
        self.f_area = torch.FloatTensor(get_face_areas(v, f)).cuda()
        self.num_faces = self.f.shape[0]

        # Edge attributes
        self.e = get_vertex_connectivity(f).cuda() # Pairs of connected vertices
        self.e_rest = get_edge_length(v, self.e) # Rest lenght of the edges (world space)
        self.num_edges = self.e.shape[0]

        '''
        # Rest state of the cloth (computed in material space)
        #tri_m = gather_triangles(vm, fm)
        #print(fm.shape)
        num_faces = len(fm)
        tri_m = vm[fm.reshape(-1)]
        #print(tri_m.shape, vm.shape, fm.shape)
        #sys.exit()
        tri_m = tri_m.reshape(num_faces, 3, 2)
        self.Dm = get_shape_matrix(tri_m).detach()
        self.Dm_inv = torch.linalg.inv(self.Dm).detach()

        #dm1 = self.Dm[1,:,0]
        #dm2 = self.Dm[1,:,1]
        '''

        self.closest_body_vertices = None

        tri = v[f.reshape(-1)]
        tri = tri.reshape(len(f), 3, 3)
        self.Dm = rotate_triangle(tri).detach()[:, :2, :]
        self.Dm_inv = torch.linalg.inv(self.Dm).detach()

        '''
        dm11 = self.Dm[1,:,0]
        dm22 = self.Dm[1,:,1]

        print(torch.nn.functional.cosine_similarity(dm1, dm11, dim=0), torch.nn.functional.cosine_similarity(dm2, dm22, dim=0))
        print(dm1, dm11, torch.norm(dm1, p=2, dim=-1), torch.norm(dm11, p=2, dim=-1))
        print(dm2, dm22, torch.norm(dm2, p=2, dim=-1), torch.norm(dm22, p=2, dim=-1))
        #print(self.Dm.shape, )
        sys.exit()
        '''

    def compute_skinning_weights(self, smpl):
        # self.v_template: numpy.array
        # smpl.template_vertices: numpy.array
        # smpl.skinning_weights: torch.tensor
        if type(self.closest_body_vertices) == type(None):
            self.closest_body_vertices = find_nearest_neighbour(self.v_template, smpl.template_vertices)
        #self.v_weights = tf.gather(smpl.skinning_weights, self.closest_body_vertices).numpy()
        #self.v_weights = tf.convert_to_tensor(self.v_weights, dtype=self.dtype)
        self.closest_body_vertices = torch.LongTensor(self.closest_body_vertices)
        self.v_weights = smpl.skinning_weights[self.closest_body_vertices]
        return self.v_weights

class Cloth_from_NP: 
    '''
    This class stores mesh and material information of the garment
    '''
    
    def __init__(self, v, f, material, dtype=torch.float32):
        self.dtype = dtype  
        self.material = material

        #v, f, vm, fm = load_obj(path, tex_coords=True)

        v = torch.FloatTensor(v).cuda()
        f = torch.LongTensor(f).cuda()
        #vm = torch.FloatTensor(vm).cuda()
        #fm = torch.LongTensor(fm).cuda()

        # Vertex attributes
        self.v_template = v
        self.v_mass = get_vertex_mass(v, f, self.material.area_density).cuda()
        #print(self.v_mass.shape, v.shape, vm.shape, f.shape, fm.shape)
        #sys.exit()
        self.v_velocity = torch.zeros(1, v.shape[0], 3).cuda() # Vertex velocities in global coordinates
        self.v = torch.zeros(1, v.shape[0], 3).cuda() # Vertex position in global coordinates
        self.v_psd = torch.zeros(1, v.shape[0], 3).cuda() # Pose space deformation of each vertex
        self.v_weights = None # Vertex skinning weights
        self.num_vertices = self.v_template.shape[0]
    
        # Face attributes
        self.f = f
        self.f_connectivity = get_face_connectivity(f).cuda() # Pairs of adjacent faces
        self.f_connectivity_edges = get_face_connectivity_edges(f).cuda() # Edges that connect faces
        self.f_area = torch.FloatTensor(get_face_areas(v, f)).cuda()
        self.num_faces = self.f.shape[0]

        # Edge attributes
        self.e = get_vertex_connectivity(f).cuda() # Pairs of connected vertices
        self.e_rest = get_edge_length(v, self.e) # Rest lenght of the edges (world space)
        self.num_edges = self.e.shape[0]

        self.closest_body_vertices = None

        tri = v[f.reshape(-1)]
        tri = tri.reshape(len(f), 3, 3)
        self.tri = tri
        self.Dm = rotate_triangle(tri).detach()[:, :2, :]
        self.Dm_inv = torch.linalg.inv(self.Dm).detach()


    def compute_skinning_weights(self, smpl):
        # self.v_template: numpy.array
        # smpl.template_vertices: numpy.array
        # smpl.skinning_weights: torch.tensor
        if type(self.closest_body_vertices) == type(None):
            self.closest_body_vertices = find_nearest_neighbour(self.v_template, smpl.template_vertices)
        #self.v_weights = tf.gather(smpl.skinning_weights, self.closest_body_vertices).numpy()
        #self.v_weights = tf.convert_to_tensor(self.v_weights, dtype=self.dtype)
        self.closest_body_vertices = torch.LongTensor(self.closest_body_vertices)
        self.v_weights = smpl.skinning_weights[self.closest_body_vertices]
        return self.v_weights

class Body:
    def __init__(self, faces):
        self.f = faces
        self.vb = None
        self.nb = None

    def update_body(self, verts_batch):
        self.vb = verts_batch
        self.nb = self.get_verts_normal(verts_batch)

    def get_verts_normal(self, verts_batch):
        # verts_batch: [batch_size, N, 3]
        nb = []
        for i in range(len(verts_batch)):
            mesh_body = trimesh.Trimesh(verts_batch[i].detach().cpu().numpy(), self.f, process=False)
            nb.append(mesh_body.vertex_normals)

        nb = torch.FloatTensor(nb).cuda()
        return nb


class FaceNormals:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def call(self, vertices, faces):
        v = vertices
        f = faces

        '''
        if v.shape.ndims == (f.shape.ndims + 1):
            f = tf.tile([f], [tf.shape(v)[0], 1, 1])   

        # Warning: tf.gather is prone to memory problems
        triangles = tf.gather(v, f, axis=-2, batch_dims=v.shape.ndims - 2) 

        # Compute face normals
        v0, v1, v2 = tf.unstack(triangles, axis=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = tf.linalg.cross(e2, e1) 

        if self.normalize:
            face_normals = tf.math.l2_normalize(face_normals, axis=-1)
        '''

        # indices: [num_faces, 3]
        # vertices: [batch_size, num_points, 3]
        num_faces = len(f)
        if v.ndim == 3:
            triangles = v[:, f.reshape(-1)]
            triangles = triangles.reshape(-1, num_faces, 3, 3)
        elif v.ndim == 2:
            triangles = v[f.reshape(-1)]
            triangles = triangles.reshape(num_faces, 3, 3)
        else:
            raise NotImplementedError

        #print(triangles.shape, f.shape, f[0])
        #print(triangles[0,0])
        v0, v1, v2 = torch.unbind(triangles, dim=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = torch.cross(e2, e1, dim=-1) 
        
        #print(v0.shape)
        #print(v0[0,0], v1[0,0], v2[0,0])
        #print(face_normals.shape, face_normals[0,0])

        #print(((e1==0).sum(dim=-1) == 3).sum())
        #print(((e2==0).sum(dim=-1) == 3).sum())
        #sys.exit()
        
        #print(torch.isnan(face_normals).sum())
        if self.normalize:
            face_normals = face_normals/(torch.norm(face_normals, p=2, dim=-1, keepdim=True)+1e-6)
            #print(torch.norm(face_normals, p=2, dim=-1)[0, :10])
            #print(face_normals[0,0])
            #sys.exit()

        #print(torch.isnan(face_normals).sum())
        #sys.exit()

        return face_normals

    def call_batch(self, vertices, faces):
        v = vertices
        f = faces

        '''
        if v.shape.ndims == (f.shape.ndims + 1):
            f = tf.tile([f], [tf.shape(v)[0], 1, 1])   

        # Warning: tf.gather is prone to memory problems
        triangles = tf.gather(v, f, axis=-2, batch_dims=v.shape.ndims - 2) 

        # Compute face normals
        v0, v1, v2 = tf.unstack(triangles, axis=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = tf.linalg.cross(e2, e1) 

        if self.normalize:
            face_normals = tf.math.l2_normalize(face_normals, axis=-1)
        '''

        # indices: [batch_size, num_faces, 3]
        # vertices: [batch_size, num_points, 3]
        '''
        num_faces = len(f)
        if v.ndim == 3:
            triangles = v[:, f.reshape(-1)]
            triangles = triangles.reshape(-1, num_faces, 3, 3)
        elif v.ndim == 2:
            triangles = v[f.reshape(-1)]
            triangles = triangles.reshape(num_faces, 3, 3)
        else:
            raise NotImplementedError
        '''
        triangles = self.gather_triangles_batch(v, f)

        #print(triangles.shape, f.shape, f[0])
        #print(triangles[0,0])
        v0, v1, v2 = torch.unbind(triangles, dim=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = torch.cross(e2, e1, dim=-1) 
        
        #print(v0.shape)
        #print(v0[0,0], v1[0,0], v2[0,0])
        #print(face_normals.shape, face_normals[0,0])
        
        if self.normalize:
            face_normals = (face_normals/torch.norm(face_normals, p=2, dim=-1, keepdim=True)+1e-6)
            #print(torch.norm(face_normals, p=2, dim=-1)[0, :10])
            #print(face_normals[0,0])
            #sys.exit()

        return face_normals

    def gather_triangles_batch(self, vertices, faces):
        #if vertices.ndim == (indices.ndim + 1):
        #    indices = indices.unsqueeze(0).repeat(len(vertices), 1, 1)

        #triangles = tf.gather(vertices, indices,
        #                      axis=-2,
        #                      batch_dims=vertices.shape.ndims - 2)

        # indices: [num_faces, 3]
        # vertices: [batch_size, num_points, 3]
        batch_size = faces.shape[0]
        num_faces = faces.shape[1]
        faces = faces.reshape(batch_size, -1)
        faces = faces.unsqueeze(-1).repeat(1, 1, 3)
        triangles = torch.gather(vertices, 1, faces)
        triangles = triangles.reshape(batch_size, num_faces, 3, 3)
        return triangles


'''
class PairwiseDistance(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairwiseDistance, self).__init__(**kwargs)

    def call(self, A, B):
        rA = tf.reduce_sum(tf.square(A), axis=-1)
        rB = tf.reduce_sum(tf.square(B), axis=-1)
        transpose_axes = [0, 2, 1] 
        distances = - 2*tf.matmul(A, tf.transpose(B, transpose_axes)) + rA[:, :, tf.newaxis] + rB[:, tf.newaxis, :]
        return distances


class NearestNeighbour(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NearestNeighbour, self).__init__(**kwargs)

    def call(self, A, B):
        distances = PairwiseDistance(dtype=self.dtype)(A, B)
        nearest_neighbour = tf.argmin(distances, axis=-1)
        return tf.cast(nearest_neighbour, dtype=tf.int32)
'''