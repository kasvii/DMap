import os, sys
import trimesh
import networkx as nx
import numpy as np

def connected_faces(faces):
    connected = {}
    for i in range(len(faces)):
        f = faces[i]
        for v_i in f:
            if v_i in connected:
                connected[v_i].add(i)
            else:
                connected[v_i] = set([i])
    return connected

def get_edges(faces):
    edges = set([])
    for f in faces:
        for i in range(4):
            e = tuple(sorted([f[i], f[(i+1)%4]]))
            edges.add(e)
    edges = list(edges)
    return edges

def get_edges_len(edges, vertices):
    edges_np = np.array(edges)
    length = np.sqrt(((vertices[edges_np[0]] - vertices[edges_np[1]])**2).sum(axis=-1)).tolist()
    return length

def connected_vertices(edges):
    connected = {}
    for e in edges:
        v0_i, v1_i = e
        if v0_i in connected:
            connected[v0_i].add(v1_i)
        else:
            connected[v0_i] = set([v1_i])
        if v1_i in connected:
            connected[v1_i].add(v0_i)
        else:
            connected[v1_i] = set([v0_i])

    return connected

def get_anchor_vertex(faces_connected):
    anchors = []
    anchors_middle = []
    lengths = set([])
    for k,v in faces_connected.items():
        if len(v) == 6:
            anchors.append(k)
        lengths.add(len(v))

        if len(v) == 1:
            anchors_middle.append(k)
    return anchors, anchors_middle, lengths

def get_anchor_vertex_top(faces_connected):
    anchors = []
    anchors_6 = []
    anchors_3 = []
    lengths = set([])
    for k,v in faces_connected.items():
        if len(v) == 5:
            anchors.append(k)
        if len(v) == 6:
            anchors_6.append(k)
        if len(v) == 3:
            anchors_3.append(k)
        lengths.add(len(v))

    return anchors, anchors_6, anchors_3, lengths

def get_anchor_4_top(anchor_6, anchors_3, faces_connected, faces):
    anchor_faces_connected = set(faces_connected[anchor_6])
    anchors_3_neigh = []
    faces_neigh = []
    for anchor in anchors_3:
        fs = set(faces_connected[anchor])
        if len(fs.intersection(anchor_faces_connected)) == 1:
            anchors_3_neigh.append(anchor)
            faces_neigh.append(fs.intersection(anchor_faces_connected).pop())

    assert len(anchors_3_neigh) == 2, 'len(anchors_3_neigh) != 2'

    f0 = set(faces[faces_neigh[0]])
    f1 = set(faces[faces_neigh[1]])
    vs = f0.intersection(f1)
    
    vs.remove(anchor_6)
    anchor_4 = vs.pop()
    return anchor_4

def shortest_path(edges, start_i, end_i, lengths=None):
    # edges without duplication
    #edges = mesh.edges_unique
    # the actual length of each unique edge
    #length = mesh.edges_unique_length
    # create the graph with edge attributes for length
    g = nx.Graph()

    if lengths is None:
        for edge in edges:
            g.add_edge(*edge, length=1)
    else:
        for edge, L in zip(edges, lengths):
            g.add_edge(*edge, length=L)
    # run the shortest path query using length for edge weight
    path = nx.shortest_path(g,
                            source=start_i,
                            target=end_i,
                            weight='length')
    return path

def select_boundary(mesh):
    unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    idx_boundary_v = np.unique(unique_edges.flatten())
    return idx_boundary_v, unique_edges

def connect_2_way(idx_boundary_v_set, one_rings, boundary_edges):
    path = [list(idx_boundary_v_set)[0]]
    idx_boundary_v_set.remove(path[0])
    # connect one way
    while len(idx_boundary_v_set):
        node = path[-1]
        neighbour = one_rings[node]
        for n in neighbour:
            if n in idx_boundary_v_set and tuple(sorted([node, n])) in boundary_edges:
                path.append(n)
                idx_boundary_v_set.remove(n)
                break

        if node == path[-1]:
            break


    # connect the other way
    while len(idx_boundary_v_set):
        node = path[0]
        neighbour = one_rings[node]
        for n in neighbour:
            if n in idx_boundary_v_set and tuple(sorted([node, n])) in boundary_edges:
                path.insert(0, n)
                idx_boundary_v_set.remove(n)
                break

        if node == path[0]:
            break

    return path, idx_boundary_v_set

def one_ring_neighour(idx_v, mesh, is_dic=False, mask_set=None):
    g = nx.from_edgelist(mesh.edges_unique)
    valid_v_i = set(np.unique(mesh.faces.flatten()).tolist())
    one_ring = []
    if mask_set is not None:
        for i in idx_v:
            if i in valid_v_i:
                one_ring.append(set(g[i].keys()).intersection(mask_set))
            else:
                one_ring.append(set([]))
    else:
        for i in idx_v:
            if i in valid_v_i:
                one_ring.append(set(g[i].keys()))
            else:
                one_ring.append(set([]))

    if is_dic:
        one_ring_dic = {}
        for i in range(len(idx_v)):
            one_ring_dic[idx_v[i]] = one_ring[i]

        one_ring = one_ring_dic
    return one_ring

def get_connected_paths_pattern(mesh):
    idx_boundary_v, boundary_edges = select_boundary(mesh)
    boundary_edges = boundary_edges.tolist()
    boundary_edges = set([tuple(sorted(e)) for e in boundary_edges])
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_x_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)
        paths.append(path)
        path_x_mean.append(mesh.vertices[path, 0].mean())

    if len(paths) == 2:
        left_path_i = path_x_mean.index(min(path_x_mean))
        left_path = paths[left_path_i]
        right_path_i = path_x_mean.index(max(path_x_mean))
        right_path = paths[right_path_i]

        return [left_path, right_path]
    elif len(paths) == 1:
        return paths[0]

def get_connected_paths_sleeves(mesh):
    idx_boundary_v, boundary_edges = select_boundary(mesh)
    boundary_edges = boundary_edges.tolist()
    boundary_edges = set([tuple(sorted(e)) for e in boundary_edges])
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_x_mean = []
    path_z_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)
        paths.append(path)
        path_x_mean.append(mesh.vertices[path, 0].mean())
        path_z_mean.append(mesh.vertices[path, -1].mean())

    left_path_i = path_x_mean.index(min(path_x_mean))
    left_path = paths[left_path_i]
    right_path_i = path_x_mean.index(max(path_x_mean))
    right_path = paths[right_path_i]

    is_opening = len(paths) == 3
    _set = set([0,1,2]) if is_opening else set([0,1,2,3])
    _set.remove(left_path_i)
    _set.remove(right_path_i)

    if is_opening:
        middle_path_i = list(_set)[0]
        middle_path = paths[middle_path_i]
    else:
        middle_is = list(_set)
        paths = [paths[i] for i in middle_is]
        path_z_mean = [path_z_mean[i] for i in middle_is]
        down_path_i = path_z_mean.index(min(path_z_mean))
        down_path = paths[down_path_i]
        up_path_i = path_z_mean.index(max(path_z_mean))
        up_path = paths[up_path_i]
        middle_path = [up_path, down_path]

    return left_path, right_path, middle_path, is_opening

def get_connected_paths_pants(mesh):
    idx_boundary_v, boundary_edges = select_boundary(mesh)
    boundary_edges = boundary_edges.tolist()
    boundary_edges = set([tuple(sorted(e)) for e in boundary_edges])
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_z_mean = []
    path_x_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)
        paths.append(path)
        path_z_mean.append(mesh.vertices[path, -1].mean())
        path_x_mean.append(mesh.vertices[path, 0].mean())

    up_path_i = path_z_mean.index(max(path_z_mean))
    up_path = paths[up_path_i]

    _set = set([0,1,2])
    _set.remove(up_path_i)

    bottom_is = list(_set)    
    paths = [paths[i] for i in bottom_is]
    path_x_mean = [path_x_mean[i] for i in bottom_is]
    left_path_i = path_x_mean.index(min(path_x_mean))
    left_path = paths[left_path_i]
    right_path_i = path_x_mean.index(max(path_x_mean))
    right_path = paths[right_path_i]

    return up_path, left_path, right_path

def get_connected_paths_skirt(mesh):
    idx_boundary_v, boundary_edges = select_boundary(mesh)
    boundary_edges = boundary_edges.tolist()
    boundary_edges = set([tuple(sorted(e)) for e in boundary_edges])
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_z_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)
        paths.append(path)
        path_z_mean.append(mesh.vertices[path, -1].mean())

    up_path_i = path_z_mean.index(max(path_z_mean))
    up_path = paths[up_path_i]

    _set = set([0,1])
    _set.remove(up_path_i)

    bottom_path_i = list(_set)[0]
    bottom_path = paths[bottom_path_i] 

    return up_path, bottom_path

def get_connected_paths_top(mesh):
    idx_boundary_v, boundary_edges = select_boundary(mesh)
    boundary_edges = boundary_edges.tolist()
    boundary_edges = set([tuple(sorted(e)) for e in boundary_edges])
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_x_mean = []
    path_z_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)
        paths.append(path)
        path_x_mean.append(mesh.vertices[path, 0].mean())
        path_z_mean.append(mesh.vertices[path, -1].mean())

    is_tube_top = False
    if len(paths) == 2:
        bottom_path_i = path_z_mean.index(min(path_z_mean))
        bottom_path = paths[bottom_path_i]
        up_path_i = path_z_mean.index(max(path_z_mean))
        up_path = paths[up_path_i]
        is_tube_top = True
        return [up_path, bottom_path], is_tube_top

    left_path_i = path_x_mean.index(min(path_x_mean))
    left_path = paths[left_path_i]
    right_path_i = path_x_mean.index(max(path_x_mean))
    right_path = paths[right_path_i]

    _set = set([0,1,2,3])
    _set.remove(left_path_i)
    _set.remove(right_path_i)

    middle_is = list(_set)
    paths = [paths[i] for i in middle_is]
    path_z_mean = [path_z_mean[i] for i in middle_is]
    down_path_i = path_z_mean.index(min(path_z_mean))
    down_path = paths[down_path_i]
    up_path_i = path_z_mean.index(max(path_z_mean))
    up_path = paths[up_path_i]

    return [left_path, right_path, up_path, down_path], is_tube_top

def path_by_shortest(start_i, idx_boundary_v, edges, lengths=None):
    path_shortest = None
    len_shortest = 10000000000000
    for idx_v in idx_boundary_v:
        path = shortest_path(edges, start_i, idx_v, lengths=lengths) # list
        if len(path) < len_shortest:
            len_shortest = len(path)
            path_shortest = path
    return path_shortest

def reduce_potential_ending(anchor_i, idx_boundary_v, vertices, y_offset=0.1, y_center=0.0):
    z_thresh = vertices[anchor_i][-1]
    #y_min_thresh = vertices[anchor_i][1] - y_offset
    #y_max_thresh = vertices[anchor_i][1] + y_offset
    y_min_thresh = y_center - y_offset
    y_max_thresh = y_center + y_offset

    flag_z = vertices[idx_boundary_v, -1] < z_thresh
    flag_y_min = vertices[idx_boundary_v, 1] >= y_min_thresh
    flag_y_max = vertices[idx_boundary_v, 1] <= y_max_thresh

    flag = np.logical_and(np.logical_and(flag_z, flag_y_min), flag_y_max).tolist()

    idx_boundary_v_reduced = [idx_boundary_v[i] for i in range(len(idx_boundary_v)) if flag[i]]
    return idx_boundary_v_reduced

def search(pool, connected, visited):
    while pool:
        idx = pool.pop()
        visited.add(idx)
        for n in connected[idx]:
            if n in visited:
                continue
            pool.add(n)

    return visited

def propogate(idx_cutting_v, vertices, connected):

    front_v = idx_cutting_v.copy() 
    back_v = idx_cutting_v.copy() 

    for idx in idx_cutting_v:
        neighbour = connected[idx]
        for n in neighbour:
            if n in idx_cutting_v:
                continue
            if vertices[n, 1] < vertices[idx, 1]:
                front_v.add(n)
            else:
                back_v.add(n)

    pool_front = front_v - idx_cutting_v
    pool_back = back_v - idx_cutting_v
    front_v = search(pool_front, connected, front_v)
    back_v = search(pool_back, connected, back_v)

    front_v = list(front_v)
    back_v = list(back_v)

    return front_v, back_v


def propogateV2(idx_cutting_v, vertices, connected):

    inner_back_i = np.argmax(vertices[:,1])
    inner_front_i = np.argmin(vertices[:,1])

    front_v = idx_cutting_v.copy() 
    back_v = idx_cutting_v.copy() 

    pool_front = set([inner_front_i])
    pool_back = set([inner_back_i])
    front_v = search(pool_front, connected, front_v)
    back_v = search(pool_back, connected, back_v)

    front_v = list(front_v)
    back_v = list(back_v)

    front_v_new = set(list([i for i in range(len(vertices))])) - set(back_v) 
    front_v_new = front_v_new.union(idx_cutting_v)
    front_v_new = list(front_v_new)
    #print('front_v_new == front_v? ', len(front_v_new), len(front_v))

    return front_v_new, back_v

def filter_faces(faces, idx_v):
    faces_new = []
    for f in faces:
        if f[0] in idx_v and f[1] in idx_v and f[2] in idx_v:
            faces_new.append(f)
    
    faces_new = np.array(faces_new)
    return faces_new

def cut_mesh(idx_front_v, idx_back_v, mesh):
    faces_front = filter_faces(mesh.faces, set(idx_front_v))
    faces_back = filter_faces(mesh.faces, set(idx_back_v))

    mesh_front = trimesh.Trimesh(mesh.vertices, faces_front, validate=False, process=False)
    mesh_back = trimesh.Trimesh(mesh.vertices, faces_back, validate=False, process=False)
    return mesh_front, mesh_back


def get_symmetric_point(anchor, path):
    idx_anchor = path.index(anchor)
    len_path = len(path)
    path_new = [anchor]
    while 1:
        idx_anchor = (idx_anchor+1)%len_path
        next_v = path[idx_anchor]

        path_new.append(next_v)
        if next_v == anchor:
            break

    idx_mid = len(path_new)//2
    
    #path_half_0 = path_new[:idx_mid+1]
    #path_half_1 = path_new[idx_mid:][::-1]

    anchor_sym = path_new[idx_mid]

    return anchor_sym


def seperate_path(anchor_0, anchor_1, path):
    # circle 

    idx_anchor = path.index(anchor_0)
    len_path = len(path)
    path_new_0 = [anchor_0]
    while 1:
        idx_anchor = (idx_anchor+1)%len_path
        next_v = path[idx_anchor]

        path_new_0.append(next_v)
        if next_v == anchor_1:
            break

    
    idx_anchor = path.index(anchor_0)
    path_new_1 = [anchor_0]    
    while 1:
        idx_anchor = (idx_anchor-1)%len_path
        next_v = path[idx_anchor]

        path_new_1.append(next_v)
        if next_v == anchor_1:
            break

    return path_new_0, path_new_1

    
def split_skirt_waist(path_up, idx_middle_front, offset_left, offset_right, len_back, v):
    i = path_up.index(idx_middle_front)
    #print(i, idx_middle_front)
    #print(path_up)

    if i+1 < len(path_up):
        if v[i, 0] > v[i+1, 0]:
            path_up = path_up[::-1]
            i = path_up.index(idx_middle_front)
    else:
        if v[i, 0] < v[i-1, 0]:
            path_up = path_up[::-1]
            i = path_up.index(idx_middle_front)

    path_circle = path_up + path_up + path_up + path_up
    i_in_circle = i + len(path_up)

    path_waist_front = path_circle[i_in_circle-offset_left:i_in_circle+offset_right+1]
    path_waist_back = path_circle[i_in_circle+offset_right:i_in_circle+offset_right+len_back][::-1]
    #print(path_waist_front)
    #print(path_waist_back)

    assert path_waist_front[0] == path_waist_back[0], 'path_waist_front[0] != path_waist_back[0]'
    assert path_waist_front[-1] == path_waist_back[-1], 'path_waist_front[-1] != path_waist_back[-1]'
    assert len(path_waist_front) + len(path_waist_back) == len(path_up) + 2, 'len(path_waist_front) + len(path_waist_back) != len(path_up) + 2'

    return path_waist_front, path_waist_back 

def check_front_back(path_front, path_back, mesh):

    y_front = mesh.vertices[path_front, 1].mean()
    y_back = mesh.vertices[path_back, 1].mean()

    if y_front > y_back:
        return path_back, path_front
    else:
        return path_front, path_back