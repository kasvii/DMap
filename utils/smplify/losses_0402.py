import torch
import sys
import trimesh
import numpy as np
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

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
        
def render_body(body_idx, smpl_output, renderer, trans_cam, faces, device):
    verts = smpl_output.vertices[body_idx] #* scale
    joints_wham = smpl_output.joints_wham[body_idx] #* scale
    offset = joints_wham[[11, 12], :].mean(-2)
    # print(f"{verts.shape=} {joints_wham.shape=}") # verts.shape=torch.Size([6890, 3]) joints_wham.shape=torch.Size([31, 3])
    # print(f"{offset=}")
    
    verts = verts - offset
    verts += trans_cam
    
    # body_verts_rgb = torch.ones_like(verts).to(device)  # Default to white if no colors are present
    # body_textures = TexturesVertex(verts_features=body_verts_rgb[None])  # Add batch dimension (N=1)
    verts_rgb = torch.zeros(len(verts), 3)
    verts_rgb[:,1] += 255
    textures_clothed = TexturesVertex(verts_features=verts_rgb[None].to(device))
    
    mesh = Meshes(
        verts=[verts_zero],   
        faces=[faces],
        textures=textures_clothed
    )
    new_mesh = mesh.offset_verts(verts)
    # images_predicted = renderer_textured_soft(new_src_mesh)
    
    # Render
    image = renderer(new_mesh)  # RGB channels
    image = image.flip(1)  # 垂直方向翻转
    image = image.flip(2)  # 水平方向翻转
    mask = image[0, :, :, 1]/255 # [0, 1]
    mask_uint8 = (mask * 255).detach().cpu().numpy().astype(np.uint8)
    # print(f"{mask_uint8.shape=}")
    # image = image[0, ..., :3].cpu().numpy()  # Extract RGB channels and move to CPU
    # image = (image * 255).astype(np.uint8)  # Convert to uint8
    # mask = ~(image == 255).all(axis=-1)     # all white: False, pixel exist: True
    # mask_uint8 = (mask * 255).astype(np.uint8)  # [0, 255]
    
    return mask, mask_uint8


class SMPLifyLoss(torch.nn.Module):
    def __init__(self, 
                 res,
                 cam_intrinsics,
                 init_pose, 
                 device,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.res = res
        self.cam_intrinsics = cam_intrinsics
        self.init_pose = torch.from_numpy(init_pose).float().to(device)
        
    def forward(self, iter_num, output, params, input_keypoints, bbox, 
                mask_full=None, renderer=None, trans_cams=None, faces=None, 
                reprojection_weight=100., regularize_weight=60.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=100.0, sigma=100, mask_weight=20.0):
        
        pose, shape, cam = params
        # print(f"{pose.shape=} {shape.shape=} {cam.shape=}") # pose.shape=torch.Size([1, 122, 144]) shape.shape=torch.Size([1, 122, 10]) cam.shape=torch.Size([1, 122, 3])
        # sys.exit()
        scale = bbox[..., 2:].unsqueeze(-1) * 200.
        
        # Loss 1. Data term
        pred_keypoints = output.full_joints2d[..., :17, :] # torch.Size([1, 122, 17, 2])
        joints_conf = input_keypoints[..., -1:]
        # print(f"{pred_keypoints=} {pred_keypoints.shape=}")
        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)
        reprojection_error = ((reprojection_error * joints_conf) / scale).mean()
        print(f"{reprojection_error.item()=}") # reprojection_error.item()=0.058
        
        # Loss 2. Regularization term
        regularize_error = torch.linalg.norm(pose - self.init_pose, dim=-1).mean() * 0
        # print(f"{regularize_error.item()=}")
        
        # Loss 3. Shape prior and consistency error
        # print(f'{shape.shape=}') # [122, 10]
        consistency_error = shape.std(dim=1).mean()
        sprior_error = torch.linalg.norm(shape, dim=-1).mean() * 0
        shape_error = sprior_weight * sprior_error + consistency_weight * consistency_error
        
        # Loss 4. Smooth loss
        pose_diff = compute_jitter(pose).mean()
        cam_diff = compute_jitter(cam).mean()
        smooth_error = pose_diff + cam_diff
        print(f"{smooth_error.item()=}") # smooth_error.item()=0.09
        
        # Loss 5. Mask loss
        mask_uint8 = None
        loss_mask = torch.tensor(0).float().cuda()
        if mask_full is not None:
            # print(f"{output.vertices.shape[0]=}")
            for body_idx in range(output.vertices.shape[0]):
                mask, mask_uint8 = render_body(body_idx, output, renderer, trans_cams[body_idx], faces, self.init_pose.device)
                # print(f"{mask.max()=} {mask.min()=}") # [1, 0]
                # print(f"{mask_full[body_idx].shape=}") # torch.Size([640, 360, 3])
                # print(f"{mask_full[body_idx].max()=} {mask_full[body_idx].min()=}")
                mask_gt = mask_full[body_idx][..., 0] / 255
                intersection = (mask * mask_gt).sum()
                union = mask.sum() + mask_gt.sum() - intersection
                # print(f"{intersection=} {union=}")
                loss_mask += (1 - intersection/union) #*224
                # print(f"{iter_num=} {body_idx=} {loss_mask.item()=}")
                if body_idx % 30 == 0:
                    with torch.no_grad():
                        cv2.imwrite(f'../../tmp/opt_body_mask_{body_idx:06d}_{iter_num:04d}.png', mask_uint8)
                        blended_img = overlay_images(mask_uint8, mask_full[body_idx].detach().cpu().numpy().astype(np.uint8), alpha=0.5)
                        cv2.imwrite(f'../../tmp/blend_mask_{body_idx:06d}_{iter_num:04d}.png', blended_img)
                
            loss_mask /= output.vertices.shape[0]
            print(f"{loss_mask.item()=}")
            # mask_error = compute_mask_loss(mask, mask_full) * mask_weight
            
        # Sum up losses
        loss = {
            'reprojection': reprojection_weight * reprojection_error,
            'regularize': regularize_weight * regularize_error,
            'shape': shape_error,
            'smooth': smooth_weight * smooth_error,
            'mask': mask_weight * loss_mask
        }
        
        return loss # , mask_uint8
        
    def create_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints,
                       mask_full,
                       renderer,
                       trans_cams, faces):
        self.iter_num = 0
        
        def closure():
            self.iter_num += 1
            optimizer.zero_grad()
            output = smpl(*params, cam_intrinsics=self.cam_intrinsics, bbox=bbox, res=self.res)
            
            loss_dict = self.forward(self.iter_num, output, params, input_keypoints, bbox, mask_full, renderer, trans_cams, faces)
            loss = sum(loss_dict.values())
            loss.backward()
            # print("Loss:", loss.item())
            # for p in params:
            #     print("Leaf:", p.is_leaf, "Grad:", p.grad)
            # print(loss.grad_fn)
            return loss
        
        return closure