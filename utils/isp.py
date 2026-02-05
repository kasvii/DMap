import os, sys
import numpy as np 
import trimesh
import cv2
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from shapely.geometry import Polygon, MultiPoint, Point
from shapely.prepared import prep

def create_uv_mesh(x_res, y_res, debug=False):
    x = np.linspace(1, -1, x_res)
    y = np.linspace(1, -1, y_res)

    # exchange x,y to make everything consistent:
    # x is the first coordinate, y is the second!
    xv, yv = np.meshgrid(y, x)
    uv = np.stack((xv, yv), axis=-1)
    '''
    print(uv[0,0])
    print(uv[0,1])
    print(uv[1,0])
    sys.exit()
    '''
    vertices = uv.reshape(-1, 2)
    #print(uv.shape)
    
    tri = Delaunay(vertices)
    faces = tri.simplices
    vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=-1)

    if debug:
        # x in plt is vertical
        # y in plt is horizontal
        plt.figure()
        plt.triplot(vertices[:,0], vertices[:,1], faces)
        plt.plot(vertices[:,0], vertices[:,1], 'o', markersize=2)
        plt.savefig('../tmp/tri.png')

    return vertices, faces

def get_barycentric(mesh, points):
    mesh_base = trimesh.proximity.ProximityQuery(mesh)
    closest_points, _, idx_f = mesh_base.on_surface(points)
    
    triangles = mesh.vertices[mesh.faces[idx_f]]
    barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_points)
    return barycentric, idx_f


def expand_edge_to_face(points_poly, return_mesh=False):
    # points_poly: ordered points on the edge
    num_p = len(points_poly)

    edge_connection = np.zeros((num_p, 2))
    idx_0 = np.arange(num_p)
    idx_1 = (idx_0 + 1)
    idx_1[-1] = 0

    edge_connection[:, 0] = idx_0
    edge_connection[:, 1] = idx_1

    fake_points = (points_poly[idx_0] + points_poly[idx_1])/2

    points_poly_3d = np.concatenate((points_poly, np.zeros((num_p, 1))), axis=-1)
    fake_points_3d = np.concatenate((fake_points, np.zeros((num_p, 1))), axis=-1)
    fake_points_3d[:, -1] = 1

    idx_2 = np.arange(num_p, 2*num_p).reshape(-1, 1)
    faces = np.concatenate((edge_connection, idx_2), axis=-1)

    points_expand = np.concatenate((points_poly_3d, fake_points_3d), axis=0)

    if return_mesh:
        mesh = trimesh.Trimesh(points_expand, faces, validate=False, process=False)
        return mesh

    return points_expand, faces

def closest_point(points1, points2):
    # points: (p, 3)
    # the closest points for points1
    distance = points1[:, None] - points2[None]
    distance = (distance**2).sum(axis=-1)
    closest_id = np.argmin(distance, axis=-1)
    return closest_id
    
def sample_sdf(mesh, v_polygon, label_v, save_fig=False, save_path=None):
    EPS = 0.01
    SURFACE_N = 6000
    CLOSE_N = 12000
    FAR_N = 8000
    RAND_N = 4000

    points_surface, face_id = trimesh.sample.sample_surface(mesh, SURFACE_N)
    points_surface[:, -1] = 0

    # sample points
    # CLOSE_N points EPS-close to the surface
    close_pts = mesh.sample(CLOSE_N) + EPS * (np.random.rand(CLOSE_N,3) - 0.5)
    close_pts[:, -1] = 0
    _, udf_close_pts, _ = mesh.nearest.on_surface(close_pts)
    close_pts = np.hstack((close_pts, udf_close_pts[:,None]))
    # FAR_N points (6*EPS)-close to the surface
    far_pts = mesh.sample(FAR_N) + 6*EPS * (np.random.rand(FAR_N,3) - 0.5)
    far_pts[:, -1] = 0
    _, udf_far_pts, _ = mesh.nearest.on_surface(far_pts)
    far_pts = np.hstack((far_pts, udf_far_pts[:,None]))
    # RAND_N points randomly in unit bbox
    rand_pts = (np.random.rand(RAND_N,3) - 0.5) * 2
    rand_pts[:, -1] = 0
    _, udf_rand_pts, _ = mesh.nearest.on_surface(rand_pts)
    rand_pts = np.hstack((rand_pts, udf_rand_pts[:,None]))
    ## Concatenate samples and save
    samples = np.concatenate((close_pts, far_pts, rand_pts))

    # use shapely polygon for checking the sign of distance instead.
    P=Polygon(v_polygon)
    prepared_polygon = prep(P)
    mp_list = [Point(samples[i,:2]) for i in range((len(samples)))]
    sign_neg = list(map(prepared_polygon.contains, mp_list))
    sign_neg = np.array(sign_neg).astype(bool)
    samples[sign_neg, -1] *= -1


    sdf_surface = np.concatenate((points_surface[:,:2], np.zeros((len(points_surface), 1))), axis=-1)
    sdf_sample = samples[:,[0,1,3]]

    closest_id_surface = closest_point(sdf_surface[:, :2], v_polygon)
    closest_id_sample = closest_point(sdf_sample[:, :2], v_polygon)
    boundary_label_surface = label_v[closest_id_surface]
    boundary_label_sample = label_v[closest_id_sample]

    if save_fig:
        fig, ax = plt.subplots()
        x = sdf_sample[:,0]
        y = sdf_sample[:,1]
        z = boundary_label_sample
        
        df = pandas.DataFrame({
            'x': x,
            'y': y,
            'label': z
        })

        plot = sns.scatterplot(data=df, x='x', y='y', hue='label')
        fig = plot.get_figure()
        #plt.tight_layout()
        fig.savefig(save_path) 

        plt.close('all')

    return sdf_surface, boundary_label_surface, sdf_sample, boundary_label_sample


def sample_sdf_left_right(mesh, v_polygon, v_polygon_left, v_polygon_right, label_v):
    EPS = 0.01
    SURFACE_N = 6000
    CLOSE_N = 12000
    FAR_N = 8000
    RAND_N = 4000

    points_surface, face_id = trimesh.sample.sample_surface(mesh, SURFACE_N)
    points_surface[:, -1] = 0

    # sample points
    # CLOSE_N points EPS-close to the surface
    close_pts = mesh.sample(CLOSE_N) + EPS * (np.random.rand(CLOSE_N,3) - 0.5)
    close_pts[:, -1] = 0
    _, udf_close_pts, _ = mesh.nearest.on_surface(close_pts)
    close_pts = np.hstack((close_pts, udf_close_pts[:,None]))
    # FAR_N points (6*EPS)-close to the surface
    far_pts = mesh.sample(FAR_N) + 6*EPS * (np.random.rand(FAR_N,3) - 0.5)
    far_pts[:, -1] = 0
    _, udf_far_pts, _ = mesh.nearest.on_surface(far_pts)
    far_pts = np.hstack((far_pts, udf_far_pts[:,None]))
    # RAND_N points randomly in unit bbox
    rand_pts = (np.random.rand(RAND_N,3) - 0.5) * 2
    rand_pts[:, -1] = 0
    _, udf_rand_pts, _ = mesh.nearest.on_surface(rand_pts)
    rand_pts = np.hstack((rand_pts, udf_rand_pts[:,None]))
    ## Concatenate samples and save
    samples = np.concatenate((close_pts, far_pts, rand_pts))

    # use shapely polygon for checking the sign of distance instead.
    P_left = Polygon(v_polygon_left)
    P_right = Polygon(v_polygon_right)
    prepared_polygon_left = prep(P_left)
    prepared_polygon_right = prep(P_right)
    mp_list = [Point(samples[i,:2]) for i in range((len(samples)))]
    sign_neg_left = list(map(prepared_polygon_left.contains, mp_list))
    sign_neg_right = list(map(prepared_polygon_right.contains, mp_list))
    sign_neg_left = np.array(sign_neg_left).astype(bool)
    sign_neg_right = np.array(sign_neg_right).astype(bool)
    samples[sign_neg_left, -1] *= -1
    samples[sign_neg_right, -1] *= -1


    sdf_surface = np.concatenate((points_surface[:,:2], np.zeros((len(points_surface), 1))), axis=-1)
    sdf_sample = samples[:,[0,1,3]]

    closest_id_surface = closest_point(sdf_surface[:, :2], v_polygon)
    closest_id_sample = closest_point(sdf_sample[:, :2], v_polygon)
    boundary_label_surface = label_v[closest_id_surface]
    boundary_label_sample = label_v[closest_id_sample]

    return sdf_surface, boundary_label_surface, sdf_sample, boundary_label_sample

def sample_atlas(mesh, mesh_pattern, labels, v_polygon, mesh_poly, uv_vertices, save_fig=False, save_path=None, mask_path=None, x_res=None, y_res=None):
    
    v_inner = mesh.vertices[labels==-1]
    v_inner_uv = mesh_pattern.vertices[labels==-1]
    normal_inner = mesh.vertex_normals[labels==-1]

    _, distance, _ = trimesh.proximity.closest_point(mesh_poly, uv_vertices)
    P=Polygon(v_polygon)
    prepared_polygon = prep(P)
    mp_list = [Point(uv_vertices[i,:2]) for i in range((len(uv_vertices)))]
    sign_neg = list(map(prepared_polygon.contains, mp_list))
    sign_neg = np.array(sign_neg).astype(bool)
    distance[sign_neg] *= -1

    points_boundary = mesh.vertices[labels!=-1]
    points_boundary_uv = mesh_pattern.vertices[labels!=-1, :2]

    if save_fig:
        fig, ax = plt.subplots()
        x = uv_vertices[:, 0].reshape(x_res, y_res)
        y = uv_vertices[:, 1].reshape(x_res, y_res)
        z = distance.reshape(x_res, y_res)
        c = ax.pcolormesh(x, y, z, cmap='RdBu')
        ax.set_title('pcolormesh')
        fig.colorbar(c, ax=ax)
        plt.savefig(save_path)

        fig, ax = plt.subplots()
        z = sign_neg.reshape(x_res, y_res)
        c = ax.pcolormesh(x, y, z, cmap='RdBu')
        ax.set_title('pcolormesh')
        fig.colorbar(c, ax=ax)
        plt.savefig(save_path.replace('sdf.png', 'contain.png'))

        plt.close('all')

        mask = sign_neg.reshape(x_res, y_res).astype(np.uint8)*255
        cv2.imwrite(mask_path, mask)

    return v_inner, v_inner_uv, points_boundary, points_boundary_uv

def sample_atlas_left_right(mesh, mesh_pattern, labels, v_polygon_left, v_polygon_right, mesh_poly, uv_vertices, save_fig=False, save_path=None, mask_path=None, x_res=None, y_res=None):
    
    v_inner = mesh.vertices[labels==-1]
    v_inner_uv = mesh_pattern.vertices[labels==-1]
    normal_inner = mesh.vertex_normals[labels==-1]

    _, distance, _ = trimesh.proximity.closest_point(mesh_poly, uv_vertices)
    P_left = Polygon(v_polygon_left)
    P_right = Polygon(v_polygon_right)
    prepared_polygon_left = prep(P_left)
    prepared_polygon_right = prep(P_right)
    mp_list = [Point(uv_vertices[i,:2]) for i in range((len(uv_vertices)))]
    sign_neg_left = list(map(prepared_polygon_left.contains, mp_list))
    sign_neg_right = list(map(prepared_polygon_right.contains, mp_list))
    sign_neg_left = np.array(sign_neg_left).astype(bool)
    sign_neg_right = np.array(sign_neg_right).astype(bool)
    sign_neg = np.logical_or(sign_neg_left, sign_neg_right)
    distance[sign_neg] *= -1
    #distance[sign_neg_right] *= -1

    points_boundary = mesh.vertices[labels!=-1]
    points_boundary_uv = mesh_pattern.vertices[labels!=-1, :2]

    if save_fig:
        fig, ax = plt.subplots()
        x = uv_vertices[:, 0].reshape(x_res, y_res)
        y = uv_vertices[:, 1].reshape(x_res, y_res)
        z = distance.reshape(x_res, y_res)
        c = ax.pcolormesh(x, y, z, cmap='RdBu')
        ax.set_title('pcolormesh')
        fig.colorbar(c, ax=ax)
        plt.savefig(save_path)

        fig, ax = plt.subplots()
        z = sign_neg.reshape(x_res, y_res)
        c = ax.pcolormesh(x, y, z, cmap='RdBu')
        ax.set_title('pcolormesh')
        fig.colorbar(c, ax=ax)
        plt.savefig(save_path.replace('sdf.png', 'contain.png'))

        plt.close('all')

        mask = sign_neg.reshape(x_res, y_res).astype(np.uint8)*255
        cv2.imwrite(mask_path, mask)

    return v_inner, v_inner_uv, points_boundary, points_boundary_uv


def uv_to_3D(pattern_deform, uv_faces, barycentric_uv, closest_face_idx_uv):
    uv_faces_id = uv_faces[closest_face_idx_uv]
    uv_faces_id = uv_faces_id.reshape(-1)

    pattern_deform_triangles = pattern_deform[uv_faces_id].reshape(-1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv[:, :, None]).sum(axis=-2)
    return pattern_deform_bary

def barycentric_faces(mesh_query, mesh_base):
    v_query = mesh_query.vertices
    base = trimesh.proximity.ProximityQuery(mesh_base)
    closest_pt, _, closest_face_idx = base.on_surface(v_query)
    triangles = mesh_base.triangles[closest_face_idx]
    v_barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_pt)
    return v_barycentric, closest_face_idx