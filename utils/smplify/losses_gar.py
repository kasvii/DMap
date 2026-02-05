import torch
import sys
import trimesh
import numpy as np
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
sys.path.append('../../')
from snug.snug_helper import collision_penalty, collision_penalty_lite

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1], dim=-1)

def get_max_mask_y(mask):
    """
    mask: [H, W], float32 or bool
    return: scalar int, 最大 y（最底下的白点的 y）
    """
    y_indices = torch.arange(mask.shape[0], device=mask.device).view(-1, 1)  # shape [H, 1]
    mask_flat = mask > 0.5  # 二值化

    # 找出每列中为1的点的 y 坐标，然后取最大
    y_masked = y_indices * mask_flat  # shape [H, W]
    max_y = y_masked.max()  # 所有列中最底下的 y
    return max_y

def overlay_images(body_seg, mask_full, alpha=0.5):
    # 1. 确保两张图像大小一致
    if body_seg.shape[:2] != mask_full.shape[:2]:
        mask_full = cv2.resize(mask_full, (body_seg.shape[1], body_seg.shape[0]))

    # 2. 确保 mask_full 是 3 通道（如果是灰度图，则转换为 BGR）
    if len(body_seg.shape) == 2:
        body_seg = cv2.cvtColor(body_seg, cv2.COLOR_GRAY2BGR)

    # 3. 叠加图像
    blended = cv2.addWeighted(body_seg, 1 - alpha, mask_full, alpha, 0)
    return blended

# faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
verts_zero = torch.zeros((6890, 3)).cuda()*0
# cloth_rgb = torch.zeros(len(verts_cloth_zero), 3) + 255 # (1, V, 3)
# verts_rgb = cloth_rgb[None]
# textures = TexturesVertex(verts_features=verts_rgb.cuda())

def get_verts_normal(verts_batch, faces):
    # verts_batch: [batch_size, N, 3]
    nb = []
    faces_np = faces.detach().cpu().numpy()
    for i in range(len(verts_batch)):
        mesh_body = trimesh.Trimesh(verts_batch[i].detach().cpu().numpy(), faces_np, process=False)
        nb.append(mesh_body.vertex_normals)

    # nb = torch.FloatTensor(nb).cuda()
    nb_np = np.stack(nb, axis=0)  # [B, N, 3]
    nb = torch.from_numpy(nb_np).float().to(verts_batch.device)
    return nb
        
def render_body(body_idx, verts, renderer_textured_soft, trans_cam, faces, device, scale=1):
    # 保留 verts 的梯度路径（用于优化 SMPL）
    # verts = smpl_output.vertices[body_idx] * scale
    # joints_smpl = smpl_output.joints[body_idx] * scale
    # verts = verts - joints_smpl[[0]]
    # verts[:, 1:] *= -1  # Y/Z轴反转（右手系 → 图像系）

    # RGB 顶点颜色，不参与训练
    verts_rgb = torch.zeros(len(verts), 3, device=device)
    verts_rgb[:, 1] += 255
    textures = TexturesVertex(verts_features=verts_rgb[None])  # [1, V, 3]

    # 模板 mesh：不参与训练
    mesh_template = Meshes(verts=[verts_zero.detach()], faces=[faces], textures=textures)

    # 使用 offset_verts 添加真实 verts（这个 verts 会传梯度）
    mesh = mesh_template.offset_verts(verts)

    # 渲染输出（image retains gradient！）
    image = renderer_textured_soft(mesh)  # [1, H, W, 3]
    mask = image[0, :, :, 1] / 255.0      # [H, W], float32, requires_grad

    # 仅保存用的 uint8 版本（detach，不留图）
    mask_uint8 = (mask.detach() * 255).cpu().numpy().astype(np.uint8)

    return mask, mask_uint8


class SMPLifyLoss(torch.nn.Module):
    def __init__(self, 
                 res,
                 cam_intrinsics,
                 init_pose, 
                 device,
                 save_path,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.res = res
        self.cam_intrinsics = cam_intrinsics
        self.init_pose = torch.from_numpy(init_pose).float().to(device)
        self.save_path = save_path
        
    def forward(self, iter_num, output, params, input_keypoints, bbox, 
                mask_full=None, renderer_textured_soft=None, transform=None, trans_cams=None, faces=None, scale=1, gar_top=None, gar_bottom=None,
                reprojection_weight=100., regularize_weight=50.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=100.0, sigma=100, mask_weight=50.0, feet_weight=20.0, top_weight=40, bottom_weight=20, eps=1e-3, **kwargs):
        
        pose, shape = params
        joints_smpl = output.joints * scale
        joints_wham = output.joints_wham * scale
        verts = output.vertices * scale
        joints = joints_wham - joints_smpl[:,[0]]
        verts = verts - joints_smpl[:, [0]]
        # print(f"{joints.shape=}")
        joints[:,:,1:] *=-1
        verts[:, :, 1:] *= -1 
        
        # if gar_top is not None or gar_bottom is not None:
        
        pred_keypoints = transform.transform_points(joints).unsqueeze(0)
        pred_keypoints = (-pred_keypoints[:,:,:17,:2] + 1)/2*511
        
        # with torch.no_grad():
        #     kp2d_vis = pred_keypoints[0, 0].detach().cpu().numpy()
        #     img = np.zeros((512, 512, 3), dtype=np.uint8)
        #     for point in kp2d_vis:
        #         cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        #     cv2.imwrite(f'../../tmp/opt_pred_kp2d_{iter_num:04d}.png', img)
        #     # print('ok')
        # sys.exit()
    
        joints_conf = input_keypoints[..., -1:]
        # print(f"{input_keypoints.shape=} {pred_keypoints.shape=}")
        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)
        reprojection_error = ((reprojection_error * joints_conf)).mean() / 512
        # print(f"{reprojection_error.item()=}") # reprojection_error.item()=0.058
        
        # Loss 2. Regularization term
        regularize_error = torch.linalg.norm(pose - self.init_pose, dim=-1).mean()
        # print(f"{regularize_error.item()=}")
        
        # Loss 3. Shape prior and consistency error
        # print(f'{shape.shape=}') # [122, 10]
        consistency_error = shape.std(dim=1).mean()
        sprior_error = torch.linalg.norm(shape, dim=-1).mean() * 0
        shape_error = sprior_weight * sprior_error + consistency_weight * consistency_error
        
        # Loss 4. Smooth loss
        smooth_error = compute_jitter(pose).mean()
        # print(f"{smooth_error.item()=}") # smooth_error.item()=0.09
        
        # Loss 5. Mask loss
        mask_uint8 = None
        loss_mask = torch.tensor(0).float().cuda()
        loss_feet = torch.tensor(0).float().cuda()
        if mask_full is not None:
            # print(f"{output.vertices.shape[0]=}")
            for body_idx in range(output.vertices.shape[0]):
                # with torch.no_grad():
                mask, mask_uint8 = render_body(body_idx, verts[body_idx], renderer_textured_soft, trans_cams[body_idx], faces, self.init_pose.device, scale)
                mask_gt = mask_full[body_idx][..., 0] / 255
                intersection = (mask * mask_gt).sum()
                union = mask.sum() + mask_gt.sum() - intersection
                loss_mask = loss_mask + (1 - intersection/union) #*224
                max_mask_gt_y = get_max_mask_y(mask_gt)
                max_mask_pred_y = get_max_mask_y(mask)
                loss_feet = loss_feet + torch.abs(max_mask_gt_y - max_mask_pred_y) / 256
                
                if body_idx % 30 == 0:
                    with torch.no_grad():
                        cv2.imwrite(f'../../tmp/opt_body_mask_{body_idx:06d}_{iter_num:04d}.png', mask_uint8)
                        blended_img = overlay_images(mask_uint8, mask_full[body_idx].detach().cpu().numpy().astype(np.uint8), alpha=0.5)
                        cv2.imwrite(f'../../tmp/blend_mask_{body_idx:06d}_{iter_num:04d}.png', blended_img)
                
            loss_mask = loss_mask / output.vertices.shape[0]
            loss_feet = loss_feet / output.vertices.shape[0]
        
        # Loss 6. Garment loss
        if gar_top is not None or gar_bottom is not None:
            normals = get_verts_normal(verts, faces).detach()
        if gar_top is not None:
            print(f"gar_top: {gar_top.shape=}") 
            loss_top_collision = 0.0
            num_frames = output.vertices.shape[0]
            count = 0
            batch_size = 10
            # normals = get_verts_normal(verts, faces).detach()
            for i in range(0, num_frames, batch_size):
                end = min(i + batch_size, num_frames)
                gar_batch = gar_top[i:end]
                body_batch = verts[i:end]
                # normals = get_verts_normal(body_batch, faces).detach()
                normal_batch = normals[i:end]

                loss = collision_penalty(
                    va=gar_batch,
                    vb=body_batch,
                    nb=normal_batch,
                    eps=eps,
                )

                loss_top_collision += loss
                count += 1
            loss_top_collision = loss_top_collision / num_frames
        else:
            loss_top_collision = torch.tensor(0).float().cuda()
        if gar_bottom is not None:
            print(f"gar_bottom: {gar_bottom.shape=}") 
            # nb = get_verts_normal(verts, faces)
            # nb = nb.detach()
            # loss_bottom_collision = collision_penalty(gar_bottom, verts, nb, eps=eps
            loss_bottom_collision = 0.0
            num_frames = output.vertices.shape[0]
            count = 0
            batch_size = 10
            # normals = get_verts_normal(verts, faces).detach()

            for i in range(0, num_frames, batch_size):
                end = min(i + batch_size, num_frames)
                gar_batch = gar_bottom[i:end]
                body_batch = verts[i:end]
                # normals = get_verts_normal(body_batch, faces).detach()
                normal_batch = normals[i:end]

                loss = collision_penalty(
                    va=gar_batch,
                    vb=body_batch,
                    nb=normal_batch,
                    eps=eps,
                )

                loss_bottom_collision += loss
                count += 1

            loss_bottom_collision = loss_bottom_collision / num_frames  # 平均每个 batch 的 collision loss
        else:
            loss_bottom_collision = torch.tensor(0).float().cuda()
        
        
        # Sum up losses
        loss = {
            'reprojection': reprojection_weight * reprojection_error,
            'regularize': regularize_weight * regularize_error,
            'shape': shape_error,
            'smooth': smooth_weight * smooth_error,
            'mask': mask_weight * loss_mask,
            'feet': loss_feet * feet_weight,
            'collision_top': loss_top_collision * top_weight,
            'collision_bottom': loss_bottom_collision * bottom_weight,
        }
        
        msg = f'Iter: {iter_num} reprojection: {loss["reprojection"].item():.4f} regularize: {loss["regularize"].item():.4f} shape: {loss["shape"].item():.4f} smooth: {loss["smooth"].item():.4f} mask: {loss["mask"].item():.4f} feet: {loss["feet"].item():.4f} top: {loss["collision_top"].item():.8f} bottom: {loss["collision_bottom"].item():.8f}'
        print(msg)
        with open(f'{self.save_path}', 'a') as f:
            f.write(msg + '\n')
        
        return loss # , mask_uint8
        
    def create_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints,
                       mask_full,
                       renderer_textured_soft,
                       transform,
                       trans_cams, faces, scale, 
                       gar_top, gar_bottom, **kwargs):
        self.iter_num = 0
        
        def closure():
            self.iter_num += 1
            optimizer.zero_grad()
            output = smpl(*params, cam_intrinsics=self.cam_intrinsics, bbox=bbox, res=self.res)
            
            loss_dict = self.forward(self.iter_num, output, params, input_keypoints, bbox, mask_full, renderer_textured_soft, transform, trans_cams, faces, scale, gar_top, gar_bottom, **kwargs)
            loss = sum(loss_dict.values())
            loss.backward()
            # print("Loss:", loss.item())
            # for p in params:
            #     print("Leaf:", p.is_leaf, "Grad:", p.grad)
            # print(loss.grad_fn)
            return loss
        
        return closure