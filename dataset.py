import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
import glob
from PIL import Image
from torchvision import transforms as T
import random
from ray_utils import *
from nesf_utils import *

# LLFF dataset
def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378), spheric_poses=False, val_num=1):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                        # load full resolution image then resize
        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        assert H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img

        return sample

# Blender
class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800,800)) -> None:
        # super().__init__()
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.read_meta()
        self.white_back = True


    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r") as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x'])
        self.focal *= self.img_wh[0]/800
        
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        self.directions = get_ray_directions(h, w, self.focal)

        if self.split == 'train':
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3,:4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1,0) # (h*w, 4)
                img = img[:, :3]*img[:, -1:]+(1-img[:,-1:]) # blend A to RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)
                self.all_rays += [torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:,:1]), self.far*torch.ones_like(rays_o[:,:1])],1)] # (h*w, 3+3+2)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames'])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'val':
            return 8 # only validate 8 images
        return len(self.meta['frames'])
    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {
                'rays': self.all_rays[idx],
                'rgbs': self.all_rgbs[idx]
            }
        else:
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3,:4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = self.transform(img)
            valid_mask = (img[-1]>0).flatten()
            img = img.view(4,-1).permute(1,0)
            img = img[:,:3]*img[:,-1:]+(1-img[:,-1:])

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample

# Nesf Klevr

class KlevrDataset(Dataset):
    def __init__(self, root_dir, img_wh, split='train', get_rgb=True, get_semantic=False) -> None:
        # super().__init__()
        self.root_dir = root_dir
        self.get_rgb = get_rgb
        self.get_semantic = get_semantic
        if split == 'train':
            self.split = split
        elif split =='val':
            self.split = 'test'
        else:
            raise KeyError("only train/val split works")
        self.define_transforms()
        self.read_meta()
        self.white_back = True


    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_meta(self):
        with open(os.path.join(self.root_dir, "metadata.json"), "r") as f:
            self.meta = json.load(f)

        w, h = self.meta['metadata']['width'], self.meta['metadata']['width']
        self.img_wh = (w, h)
        self.focal = (self.meta['camera']['focal_length']*w/self.meta['camera']['sensor_width'])
        self.split_ids = self.meta['split_ids'][self.split]

        self.scene_boundaries = np.array([self.meta['scene_boundaries']['min'], self.meta['scene_boundaries']['max']])
        self.segmentation_labels = self.meta['segmentation_labels']
        self.directions = get_ray_directions(h, w, self.focal)

        if self.split == 'train':
            self.poses = []
            self.all_rays_o = []
            self.all_rays_d = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_semantics = []
            camera_positions = np.array(self.meta['camera']['positions'])
            camera_quaternions = np.array(self.meta['camera']['quaternions'])
            for image_id in self.split_ids:
                if self.get_rgb:
                    image_path = os.path.join(self.root_dir, f'rgba_{image_id:05d}.png')
                    img = Image.open(image_path)
                    img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                    img = self.transform(img) # (4, h, w)
                    img = img.view(4, -1).permute(1,0) # (h*w, 4)
                    # not sure, original jax implementation seems not using blend just cut it off, they also /255 to make it [0,1] which I didn't use
                    # img = img[:, :3]*img[:, -1:]+(1-img[:,-1:]) # blend A to RGB
                    img = img[:, :3]
                    self.all_rgbs += [img]
                if self.get_semantic:
                    semantic_path = os.path.join(self.root_dir, f'segmentation_{image_id:05d}.png')
                    sem = Image.open(semantic_path)
                    sem = sem.resize(self.img_wh, Image.Resampling.LANCZOS)
                    sem = torch.from_numpy(np.array(sem)).long() #(h,w) 
                    sem = sem.view(-1) # (h*w)
                    self.all_semantics += [sem]
                pose = np.array(from_position_and_quaternion(camera_positions[image_id:image_id+1,:], camera_quaternions[image_id:image_id+1,:], False))[0,:3,:4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)                    
                rays_o, rays_d = get_rays(self.directions, c2w)
                self.all_rays_o += [rays_o]
                self.all_rays_d += [rays_d]

            self.all_rays_o = torch.cat(self.all_rays_o, 0) # (len(self.split_ids)*h*w, 3)
            self.all_rays_d = torch.cat(self.all_rays_d, 0) # (len(self.split_ids)*h*w, 3)
            self.all_rays_o, self.all_rays_d = scale_rays(self.all_rays_o, self.all_rays_d, self.scene_boundaries, self.img_wh)
            
            
            self.near, self.far = calculate_near_and_far(self.all_rays_o, self.all_rays_d)
            self.all_rays = torch.cat([self.all_rays_o, self.all_rays_d, self.near, self.far],1).float()

            if len(self.all_rgbs) > 0:
                self.all_rgbs = torch.cat(self.all_rgbs, 0)
            if len(self.all_semantics) > 0:
                self.all_semantics = torch.cat(self.all_semantics, 0)

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'val' or self.split == 'test':
            return 8 # only validate 8 images
        return len(self.split_ids)
    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {
                'rays': self.all_rays[idx],
            }
            if self.get_rgb:
                sample['rgbs'] = self.all_rgbs[idx]
            if self.get_semantic:
                sample["semantics"] = self.all_semantics[idx]

        # split of val/test
        else:
            image_id = self.split_ids[idx]

            if self.get_rgb:
                img = Image.open(os.path.join(self.root_dir, f'rgba_{image_id:05d}.png'))
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img)
                valid_mask = (img[-1]>0).flatten()
                img = img.view(4,-1).permute(1,0)
                # img = img[:,:3]*img[:,-1:]+(1-img[:,-1:])
                img = img[:, :3]

            if self.get_semantic:
                sem = Image.open(os.path.join(self.root_dir, f'segmentation_{image_id:05d}.png'))
                sem = sem.resize(self.img_wh, Image.Resampling.LANCZOS)
                sem = torch.from_numpy(np.array(sem)).long().unsqueeze(-1) #(h,w,1) # not sure (h,w,1) or (1, h, w)
                sem = sem.view(-1,1) # (h*w, 1)

            camera_position = np.array(self.meta['camera']['positions'][image_id:image_id+1])
            camera_quaternion = np.array(self.meta['camera']['quaternions'][image_id:image_id+1])

            pose = np.array(from_position_and_quaternion(camera_position, camera_quaternion, False))[0,:3,:4]
            # pose = np.array(from_position_and_quaternion(camera_position[idx:1+idx,:], camera_quaternion[idx:1+idx,:], False))[0,:3,:4]
            c2w = torch.FloatTensor(pose)[:3,:4]
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_o, rays_d = scale_rays(rays_o, rays_d, self.scene_boundaries, self.img_wh)
            
            self.near, self.far = calculate_near_and_far(rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d, 
                              self.near,
                              self.far],
                              1) # (H*W, 8)

            sample = {'rays': rays.float(),
                      'c2w': c2w}
            if self.get_rgb:
                sample['rgbs'] = img
                sample['valid_mask'] = valid_mask
            if self.get_semantic:
                sample["semantics"] = sem

        return sample

dataset_dict = {'blender': BlenderDataset, 'llff': LLFFDataset, 'klevr': KlevrDataset}
