import torch
import sys
import trimesh
import numpy as np
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

def gmof(x, sigma):
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    return torch.linalg.norm(x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1], dim=-1)

def get_max_mask_y(mask):
    y_indices = torch.arange(mask.shape[0], device=mask.device).view(-1, 1)
    mask_flat = mask > 0.5
    y_masked = y_indices * mask_flat
    max_y = y_masked.max()
    return max_y

def overlay_images(body_seg, mask_full, alpha=0.5):
    if body_seg.shape[:2] != mask_full.shape[:2]:
        mask_full = cv2.resize(mask_full, (body_seg.shape[1], body_seg.shape[0]))

    if len(body_seg.shape) == 2:
        body_seg = cv2.cvtColor(body_seg, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(body_seg, 1 - alpha, mask_full, alpha, 0)
    return blended

verts_zero = torch.zeros((6890, 3)).cuda()*0
        
def render_body(body_idx, smpl_output, renderer_textured_soft, trans_cam, faces, device, scale=1):
    verts = smpl_output.vertices[body_idx] * scale
    joints_smpl = smpl_output.joints[body_idx] * scale
    verts = verts - joints_smpl[[0]]
    verts[:, 1:] *= -1

    verts_rgb = torch.zeros(len(verts), 3, device=device)
    verts_rgb[:, 1] += 255
    textures = TexturesVertex(verts_features=verts_rgb[None])

    mesh_template = Meshes(verts=[verts_zero.detach()], faces=[faces], textures=textures)

    mesh = mesh_template.offset_verts(verts)

    image = renderer_textured_soft(mesh)
    mask = image[0, :, :, 1] / 255.0

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
                mask_full=None, renderer_textured_soft=None, transform=None, trans_cams=None, faces=None, scale=1,
                reprojection_weight=100., regularize_weight=5.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=100.0, sigma=100, mask_weight=50.0, feet_weight=100.0, **kwargs):
        
        pose, shape, cam = params
        
        joints_smpl = output.joints * scale
        joints_wham = output.joints_wham * scale
        joints = joints_wham - joints_smpl[:,[0]]
        joints[:,:,1:] *=-1
        pred_keypoints = transform.transform_points(joints).unsqueeze(0)
        pred_keypoints = (-pred_keypoints[:,:,:17,:2] + 1)/2*511
        
        with torch.no_grad():
            kp2d_vis = pred_keypoints[0, 0].detach().cpu().numpy()
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            for point in kp2d_vis:
                cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.imwrite(f'../../tmp/opt_pred_kp2d_{iter_num:04d}.png', img)
    
        joints_conf = input_keypoints[..., -1:]
        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)
        reprojection_error = ((reprojection_error * joints_conf)).mean() / 512
        
        regularize_error = torch.linalg.norm(pose - self.init_pose, dim=-1).mean()
        
        consistency_error = shape.std(dim=1).mean()
        sprior_error = torch.linalg.norm(shape, dim=-1).mean() * 0
        shape_error = sprior_weight * sprior_error + consistency_weight * consistency_error
        
        pose_diff = compute_jitter(pose).mean()
        cam_diff = compute_jitter(cam).mean()
        smooth_error = pose_diff + cam_diff
        
        mask_uint8 = None
        loss_mask = torch.tensor(0).float().cuda()
        loss_feet = torch.tensor(0).float().cuda()
        if mask_full is not None:
            for body_idx in range(output.vertices.shape[0]):
                mask, mask_uint8 = render_body(body_idx, output, renderer_textured_soft, trans_cams[body_idx], faces, self.init_pose.device, scale)
                mask_gt = mask_full[body_idx][..., 0] / 255
                intersection = (mask * mask_gt).sum()
                union = mask.sum() + mask_gt.sum() - intersection
                loss_mask = loss_mask + (1 - intersection/union)
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
            
        loss = {
            'reprojection': reprojection_weight * reprojection_error,
            'regularize': regularize_weight * regularize_error,
            'shape': shape_error,
            'smooth': smooth_weight * smooth_error,
            'mask': mask_weight * loss_mask,
            'feet': loss_feet * feet_weight
        }
        
        msg = f'Iter: {iter_num} reprojection: {loss["reprojection"].item():.4f} regularize: {loss["regularize"].item():.4f} shape: {loss["shape"].item():.4f} smooth: {loss["smooth"].item():.4f} mask: {loss["mask"].item():.4f} feet: {loss["feet"].item():.4f}'
        print(msg)
        with open(f'{self.save_path}', 'a') as f:
            f.write(msg + '\n')
        
        return loss
        
    def create_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints,
                       mask_full,
                       renderer_textured_soft,
                       transform,
                       trans_cams, faces, scale, **kwargs):
        self.iter_num = 0
        
        def closure():
            self.iter_num += 1
            optimizer.zero_grad()
            output = smpl(*params, cam_intrinsics=self.cam_intrinsics, bbox=bbox, res=self.res)
            
            loss_dict = self.forward(self.iter_num, output, params, input_keypoints, bbox, mask_full, renderer_textured_soft, transform, trans_cams, faces, scale, **kwargs)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure