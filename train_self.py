import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import cv2
from mvsnet import MVSNet, mvsnet_loss
import os
import contextlib
import torchvision

SCENES_DIR = "./datasets/NerfSyn/"
TRAIN_SCENES = ["chair", "lego", "materials", "ship"]
TEST_SCENES = ["drums", "hotdog", "ficus",  "mic"]

DEFAULT_VIEW_DIR = [-1., -1., -1.]
BG_COLOR = [1, 1, 1]

IMG_W = 800
IMG_H = 800
PRETRAIN_W = 640
PRETRAIN_H = 512
DEP_L = 2
DEP_R = 6

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 128
mpl.rcParams['savefig.pad_inches'] = 0

def ssim(x, y):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    from skimage.metrics import structural_similarity as ssim
    return ssim(x, y, win_size=11, multichannel=True)

def show(img, save=None):
    # img = img.astype(np.int)
    fig = plt.figure()
    # fig.set_size_inches(height * img.shape[0] / img.shape[1], height, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if isinstance(img, torch.Tensor):
        img = img.cpu()
    # img = torch.from_numpy(cv2.resize(img.float().numpy(), [IMG_W, IMG_H]))
    if len(img.shape) == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')
    if save is not None:
        plt.savefig(save)
    plt.show()

def enhomo(x):
    return torch.cat([x, torch.ones(list(x.shape[:-1]) + [1], device=x.device, dtype=x.dtype)], dim=-1)

def dehomo(x):
    return x[..., :-1] / x[..., [-1]]

def read_pfm(filename):
    import re
    file = open(filename, 'rb')
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def pix2pts_ns(cam, pix_ind, scale=1, rand=False):
    cam_mat_inv = cam[1]
    n, m = IMG_W * scale, IMG_H * scale
    tmp = torch.stack([pix_ind % m, pix_ind.div(m, rounding_mode='floor')], dim=-1).double()
    tmp = (tmp + 0.5).double() 
    if rand:
        mov = (torch.randn_like(tmp) / 16).clamp(min=-1, max=1)
        tmp += mov
    tmp /= scale
    # print(tmp.shape)
    tmp = torch.cat([tmp, torch.ones_like(tmp)], dim=-1).double()
    # print(tmp.shape)
    tmp = dehomo(cam_mat_inv.matmul(tmp.unsqueeze(-1)).squeeze(-1))
    tmp = tmp * 1
    return tmp.float()

def pts2pix_ns(cam, pts):
    pts = pts / 1
    cam_mat = cam[0].to(pts.device)
    proj = dehomo(cam_mat.matmul(enhomo(pts).double().unsqueeze(-1)).squeeze(-1)).float()
    proj = torch.stack([proj[:, 1], proj[:, 0]], dim=-1)
    return proj.float()

def pix2ray_ns(cam, pix_ind, scale=1, rand=False):
    center = cam[2]
    pts = pix2pts_ns(cam, pix_ind, scale=scale, rand=rand)
    dir = (pts - center)
    dir /= dir.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return dir.float()

def intersect_box(line_o, line_d, box_d, box_u, fetch_range=False, positive_only=True):
    import math
    inv_d = 1 / line_d
    A = (box_d - line_o) * inv_d
    B = (box_u - line_o) * inv_d

    def fmx(x):
        x = x.clone()
        x[x.isnan()] = -math.inf
        return x
    def fmn(x):
        x = x.clone()
        x[x.isnan()] = math.inf
        return x

    def pwmin(A, B):
        x = torch.minimum(fmn(A), fmn(B))
        x[A.isnan() & B.isnan()] = math.nan
        return x
    def pwmax(A, B):
        x = torch.maximum(fmx(A), fmx(B))
        x[A.isnan() & B.isnan()] = math.nan
        return x


    pmin = pwmin(A, B)
    pmax = pwmax(A, B)

    vmin = fmx(pmin).max(dim=-1).values
    vmax = fmn(pmax).min(dim=-1).values

    if positive_only:
        vmin = vmin.clamp(min=0)

    intersect = vmin + 1e-6 < vmax

    if fetch_range:
        return intersect, vmin, vmax

    return intersect

def show_cloud(pts, show_norm=False):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    cloud.estimate_normals()
    o3d.visualization.draw_geometries([cloud], point_show_normal=show_norm)

def estimate_norm(pts,param=None):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    if param is None:
        cloud.estimate_normals()
    else:
        cloud.estimate_normals(search_param=param)
    
    return torch.from_numpy(np.asarray(cloud.normals))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
def contour_rgb(pts):
    r = torch.tensor([1, 0, 0]).float().cuda()
    g = torch.tensor([0, 1, 0]).float().cuda()
    b = torch.tensor([0, 0, 1]).float().cuda()
    z = pts[:, 2].clone()
    min, max = z.aminmax()
    z = (z - min) / (max - min)
    z = z.unsqueeze(-1)
    c0 = (z - 0/2) * g + (1/2 - z) * b
    c1 = (z - 1/2) * r + (2/2 - z) * g
    c0 *= 2
    c1 *= 2
    rgb = torch.where(z.expand(-1, 3) < 0.5, c0, c1)
    return rgb.clamp(min=0, max=1)

def plot(pts, conf=None, rgb=None, pov=[0], revcol=[False], dpi=64, save=None, marker='.', size=1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    import numpy as np

    if rgb is None:
        rgb = contour_rgb(pts.cuda()).to(pts.device).to(pts.dtype)

    if len(pts.shape) > 2:
        pts = pts.squeeze(0)
        if conf is not None:
            conf = conf.squeeze(0)
        if rgb is not None:
            rgb = rgb.squeeze(0)
    if conf is None:
        conf = torch.ones_like(pts[:, 0])
    if rgb is None:
        rgb = torch.zeros_like(pts)

    assert len(pts.shape) == 2

    pts = pts.clone().cpu().numpy()
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    x -= x.mean()
    y -= y.mean()
    z -= z.mean()

    for rc in revcol:
        rgba = torch.cat([rgb if not rc else 1 - rgb, conf.unsqueeze(-1)], dim=-1).cpu().numpy()

        def apply_i(pts, i):
            x, y, z = pts[...,0], pts[...,1], pts[...,2]
            if (i & 4) != 0:    x = -x
            if (i & 2) != 0:    y = -y
            if (i & 1) != 0:    z = -z
            return torch.stack([x, y, z], dim=-1)

        for i in pov:
            fig = plt.figure(dpi=dpi)
            fig.set_size_inches(12, 12, forward=False)
            # gs = gridspec.GridSpec(nrows=4, ncols=2, left=0.1, right=2.5, wspace=0.05, hspace=0.1, bottom=0.1, top=4.9)
            gs = gridspec.GridSpec(nrows=1, ncols=1, left=0, right=1, wspace=0, hspace=0, bottom=0.0, top=1)
        
            ax = fig.add_subplot(gs[0, 0], projection='3d')
            if not rc:
                pane_col = (0., 0., 0.5, 0.1)
            else:
                pane_col = (0., 0., 0., 0.95)
            ax.xaxis.set_pane_color(pane_col)
            ax.yaxis.set_pane_color(pane_col)
            ax.zaxis.set_pane_color(pane_col)

            x, y, z = pts[:,0], pts[:,1], pts[:,2]
            labx, laby, labz = 'x', 'y', 'z'
            if (i & 4) != 0:    x = -x; labx = '-x'
            if (i & 2) != 0:    y = -y; laby = '-y'
            if (i & 1) != 0:    z = -z; labz = '-z'

            ax.scatter(x, y, z, c=rgba, marker=marker, s=size)

            xmid = (x.min() + x.max()) / 2
            ymid = (y.min() + y.max()) / 2
            zmid = (z.min() + z.max()) / 2
            gap = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2

            ax.set_xlim(xmid - gap, xmid + gap)
            ax.set_ylim(ymid - gap, ymid + gap)
            ax.set_zlim(zmid - gap, zmid + gap)

            ax.set_xlabel(labx)
            ax.set_ylabel(laby)
            ax.set_zlabel(labz) 

            if save is not None:
                plt.savefig(save)

            plt.show()

def read_data_ns(DATASET, config, i, debug=False):
    import cv2
    import math
    
    img_file = f"{DATASET}/{config['frames'][i]['file_path']}.png"
    cam_ext = torch.tensor(config['frames'][i]['transform_matrix'], dtype=torch.double)
    cam_ext = cam_ext @ torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).double()
    cam_ext = torch.tensor(scipy.linalg.inv(cam_ext)).double()

    focal = 0.5 * IMG_W / math.tan(0.5 * config['camera_angle_x'])
    cam_int = torch.tensor([[focal, 0, IMG_W / 2], [0, focal, IMG_H / 2], [0, 0, 1]], dtype=torch.double)

    raw_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    opa = torch.from_numpy(raw_img[:, :, -1] / 255)
    img = torch.from_numpy(raw_img[:, :, :3][:, :, ::-1] / 255)

    # img2 = torch.from_numpy(cv2.imread(img_file)[:, :, ::-1] / 255)
    # show(img2)
    # show(img)
    # print((img2 - img).abs().max())
    # return

    opa_thres = opa[opa > 0].min() - 1e-6
    if debug:
        print("opa_thres", opa_thres)

    cam_mat = cam_int.double().mm(cam_ext.double()[:-1])
    cam_center = dehomo(torch.tensor(scipy.linalg.null_space(cam_mat.double())).squeeze()).float() * 1
    cam_mat_inv = torch.tensor(scipy.linalg.inv(torch.cat([cam_mat, torch.tensor([[0., 0., 0., 1.]])], dim=0).double()))
    cam = (cam_mat, cam_mat_inv, cam_center)

    img = img * opa.unsqueeze(-1) + torch.tensor(BG_COLOR) * (1 - opa.unsqueeze(-1))

    cam_int[:2] = cam_int[:2] / 4
    cam_int[0] *= PRETRAIN_W / IMG_W
    cam_int[1] *= PRETRAIN_H / IMG_H
    proj = torch.eye(4, dtype=torch.double)
    proj[:3, :4] = cam_int @ cam_ext[:3, :4]

    return cam, proj, img, opa > opa_thres

class MVSDataset(torch.utils.data.Dataset):
    def __init__(self, batches):
        self.batches = batches
        self.mode = -1
    
    def __len__(self):
        return len(self.batches)

    def train(self):
        self.mode = 0
    
    def eval(self):
        self.mode = 1

    def __getitem__(self, idx):
        assert self.mode in [0, 1], "Must be set to train or eval mode."
        proj, img, mask, dep = self.batches[idx]
        dep = dep[..., self.mode]
        return proj, img, mask, dep

    def loader(self, **kwargs):
        return torch.utils.data.DataLoader(self, **kwargs)

    def sample_subset(self, n):
        import random
        return MVSDataset(random.sample(self.batches, n))

def calc_loss(predict, target, mask, no_psnr=False):
    se = (predict - target).pow(2) * mask
    se = se.sum(dim=(-1,-2)) / mask.sum(dim=(-1,-2)).clamp(min=1)
    loss = se.mean()
    if no_psnr:
        return loss
    
    with torch.no_grad():
        psnr = mse2psnr(se).mean().item()
    return loss, psnr

def read_train_data(SCENE, all_views=False, debug=False):
    pairs = torch.load(f"{SCENES_DIR}/mvsnerf_pairs.pth")
    batch = []

    DATASET = f"{SCENES_DIR}/{SCENE}/"
    train_config = json.load(open(f"{DATASET}/transforms_train.json"))
    dep_coa = torch.load(f'{SCENES_DIR}/depth_maps/depth_map_{SCENE}_coarse.pth')
    dep_fin = torch.load(f'{SCENES_DIR}/depth_maps/depth_map_{SCENE}_fine.pth')
    
    train_views = pairs[f"{SCENE}_train"]
    test_views = pairs[f"{SCENE}_test"]
    if all_views:
        train_views = list(range(len(train_config['frames'])))

    
    logging.info(f"Read #{SCENE} #train_views = {len(train_views)}")  

    all_data = []
    cam_centers = []

    dep_psnr = 0.0
    dep_ssim = 0.0

    for i in tqdm(range(len(train_config['frames']))):
        cam, proj, img, mask = read_data_ns(DATASET, train_config, i)
        
        cam_centers.append(cam[2])

        dep = torch.stack([dep_coa[i][..., 0], dep_fin[i][..., 0]], dim=-1)
        dep[~mask] = 0.0

        dep_psnr += calc_loss(dep[..., 0], dep[..., 1], mask)[1]
        dep_ssim += ssim(dep[..., 0], dep[..., 1]).item()

        all_data.append((proj, img, mask, dep))    

    dep_psnr /= len(train_config['frames'])
    dep_ssim /= len(train_config['frames'])
    logging.info(f"Depth psnr = {dep_psnr:.6f} ssim = {dep_ssim:.6f}")

    
    cam_centers = torch.stack(cam_centers)
    train_cam_centers = cam_centers[train_views]
    dist, train_pairs = torch.cdist(train_cam_centers, cam_centers).topk(6, dim=1, largest=False)
    
    if debug:
        for i in range(len(train_views)):
            rgb = torch.zeros_like(cam_centers)
            rgb[:, 1] = 1
            rgb[train_pairs[i], 1] = 0
            rgb[train_pairs[i][:1], 0] = 1
            rgb[train_pairs[i][1:], 2] = 1
            plot(cam_centers, rgb=rgb, marker='o', size=50)

    for train_pair in train_pairs:
        cams = torch.stack([all_data[i][0] for i in train_pair], dim=0)
        imgs = torch.stack([all_data[i][1] for i in train_pair], dim=0)
        masks = torch.stack([all_data[i][2] for i in train_pair], dim=0)
        deps = torch.stack([all_data[i][3] for i in train_pair], dim=0)

        inv_proj = torch.tensor(scipy.linalg.inv(cams[0].numpy()))
        cams = cams @ inv_proj

        batch.append((cams, imgs, masks, deps))

    logging.info("#batch = %d" % len(batch))
    return MVSDataset(batch)

def load_mvsnet(ckpt):
    mvsnet = MVSNet()
    mvsnet.load_state_dict(ckpt)
    mvsnet = mvsnet.cuda()
    return mvsnet

class MVSNetPretrained(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.mvsnet = load_mvsnet(ckpt)

    def forward(self, batch_imgs, batch_cams, simple_output=True, prob_only=False):
        batch_size = batch_imgs.shape[0]

        batch_imgs = batch_imgs.float().permute(0, 1, 4, 2, 3).cuda()
        batch_cams = batch_cams.float().cuda()
        dep_vals = torch.arange(192).cuda().unsqueeze(0).repeat(batch_size, 1) / 192
        dep_vals = dep_vals * (DEP_R - DEP_L) + DEP_L

        batch_imgs = torchvision.transforms.Resize([PRETRAIN_H, PRETRAIN_W])(
            batch_imgs.reshape(-1, *batch_imgs.shape[2:])).reshape(
                batch_size, -1, 3, PRETRAIN_H, PRETRAIN_W)
        
        if prob_only:
            return self.mvsnet(batch_imgs, batch_cams, dep_vals, prob_only=True)
        
        pred_deps, conf, features, prob_volume = self.mvsnet(batch_imgs, batch_cams, dep_vals, prob_only=False)
        pred_deps = 1 - (pred_deps - DEP_L) / (DEP_R - DEP_L)

        pred_deps = torchvision.transforms.Resize([IMG_H, IMG_W])(pred_deps) 

        if simple_output:
            return pred_deps
        return pred_deps, conf, features, prob_volume
        
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(self.dim)

class MVSNetSelfSup(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.mvsnet = MVSNetPretrained(ckpt)
        self.mvsnet.mvsnet.use_native_grid_sample = False
        self.loss_net = nn.Sequential(
            nn.Conv3d(8, 1, 3, stride=1, padding=1),
            nn.ELU(),
            Squeeze(1),
            nn.Conv2d(192, 24, 5, 2, 2),
            nn.ELU(),
            nn.Conv2d(24, 4, 5, 2, 2),
        )

    def forward(self, *args, training=False):
        if training:
            #with torch.no_grad():
            features = self.mvsnet(*args, prob_only=True)[-1]
            maml_loss = self.loss_net(features).mean(dim=(-1,-2)).norm(dim=-1).mean()
            return maml_loss

        pred_deps = self.mvsnet(*args, prob_only=False)[0]
        return pred_deps

def fix_name(name):
    import re
    fixed = re.sub(r"\.(\d{1,})\.", r"[\1].", name)
    return fixed


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
    

def maml_train_step(mvsnet_orig, episode, num_epoch=1, batch_size=2, num_batches=8, alpha=0.002):
    assert num_epoch == 1, "num_epoch must be 1"
    import copy
    mvsnet = copy.deepcopy(mvsnet_orig)
    mvsnet.zero_grad()
    for param in mvsnet.loss_net.parameters():
        param.requires_grad = False
    opt = torch.optim.Adam(mvsnet.mvsnet.parameters(), lr=alpha)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    episode = episode.sample_subset(batch_size * num_batches)

    # 1st run: obtain parameters without 2nd order gradient to save memory
    episode.train()
    mvsnet.eval()
    train_loader = episode.loader(batch_size=batch_size, shuffle=True, pin_memory=True)
    for epoch in range(num_epoch):
        opt.zero_grad()
        for (batch_cams, batch_imgs, _, _) in train_loader:
            loss = mvsnet(batch_imgs, batch_cams, training=True)
            loss = loss * batch_imgs.shape[0] / len(episode)
            loss.backward()
        opt.step()
        sch.step()

    # calculate loss for updated parameters
    episode.eval()
    mvsnet.eval()
    test_loader = episode.loader(batch_size=batch_size, shuffle=True, pin_memory=True)
    test_psnr = 0
    opt.zero_grad()
    for (batch_cams, batch_imgs, batch_masks, batch_deps) in test_loader:
        pred_deps = mvsnet(batch_imgs, batch_cams)
        batch_masks = batch_masks[:, 0].cuda()
        batch_deps = batch_deps[:, 0].cuda()
        loss, psnr = calc_loss(pred_deps, batch_deps, batch_masks)
        loss = loss * batch_imgs.shape[0] / len(episode)
        test_psnr += psnr * batch_imgs.shape[0] / len(episode)
        loss.backward()

    grad_updated_param = []
    for param in mvsnet.mvsnet.parameters():
        grad = param.grad
        if grad is not None:
            grad = grad.detach().clone()
        grad_updated_param.append(grad)

    for param, grad in zip(mvsnet_orig.mvsnet.parameters(), grad_updated_param):
        # 1st-order gradient update
        if grad is not None:
            with torch.no_grad():
                if param.grad is None:
                    param.grad = grad.clone()
                else:
                    param.grad += grad

    
    # 2nd run: obtain parameters with 2nd order gradient
    mvsnet.zero_grad()
    del mvsnet
    mvsnet = copy.deepcopy(mvsnet_orig)
    episode.train()
    mvsnet.eval()
    mvsnet.zero_grad()

    grad_passing_raw = [(grad * -alpha) if grad is not None else None for grad in grad_updated_param]
    del grad_updated_param
    train_loader = episode.loader(batch_size=max(1, batch_size // 2), shuffle=True, pin_memory=True)
    for epoch in range(num_epoch):
        mvsnet.zero_grad()
        for (batch_cams, batch_imgs, _, _) in train_loader:
            loss = mvsnet(batch_imgs, batch_cams, training=True)
            loss = loss * batch_imgs.shape[0] / len(episode)

            update_raw = torch.autograd.grad(
                loss, mvsnet.mvsnet.parameters(), create_graph=True, allow_unused=True)
                
            update = []
            grad_passing = []
            for ug, pg in zip(update_raw, grad_passing_raw):
                if ug is not None and pg is not None:
                    update.append(ug)
                    grad_passing.append(pg)
            
            grad_contribute = torch.autograd.grad(
                update, mvsnet.loss_net.parameters(), grad_passing, allow_unused=True)

            for param, grad in zip(mvsnet_orig.loss_net.parameters(), grad_contribute):
                # 2nd-order gradient update
                if grad is not None:
                    with torch.no_grad():
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad += grad
    

    return test_psnr

def maml_valid_step(mvsnet_orig, episode, num_epoch=40, batch_size=2, alpha=0.002, plot=False):
    import copy
    mvsnet = copy.deepcopy(mvsnet_orig)
    mvsnet.zero_grad()
    # freeze loss_net for validation
    for param in mvsnet.loss_net.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(mvsnet.mvsnet.parameters(), lr=alpha)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    episode.train()
    mvsnet.eval()
    train_loader = episode.loader(batch_size=batch_size, shuffle=True, pin_memory=True)
    for epoch in tqdm(range(num_epoch)):
        opt.zero_grad()
        for (batch_cams, batch_imgs, _, _) in train_loader:
            loss = mvsnet(batch_imgs, batch_cams, training=True)
            loss = loss * batch_imgs.shape[0] / len(episode)
            loss.backward()
        opt.step()
        sch.step()

    episode.eval()
    mvsnet.eval()
    test_loader = episode.loader(batch_size=batch_size if not plot else 1, shuffle=False, pin_memory=True)
    test_psnr = 0
    test_ssim = 0
    for i, (batch_cams, batch_imgs, batch_masks, batch_deps) in enumerate(test_loader):
        with torch.no_grad():
            pred_deps = mvsnet(batch_imgs, batch_cams)
            batch_masks = batch_masks[:, 0].cuda()
            batch_deps = batch_deps[:, 0].cuda()
            loss, psnr = calc_loss(pred_deps, batch_deps, batch_masks)
            loss = loss * batch_imgs.shape[0] / len(episode)
            test_psnr += psnr * batch_imgs.shape[0] / len(episode)

            if plot:
                out = pred_deps[0]
                ans = batch_deps[0]
                ref = episode.batches[i][-1][0, ..., 0].cuda()
                mask = batch_masks[0]
                out[~mask] = 0
                ans[~mask] = 0
                ref[~mask] = 0
                ssim_out = ssim(out, ans)
                ssim_ref = ssim(ref, ans)
                test_ssim += ssim_out / len(episode)
                logging.debug("====================================")
                logging.debug(f"#{i}")
                logging.debug("out:")
                show(out)
                logging.debug("ans:")
                show(ans)
                logging.debug("ref:")
                show(ref)
                logging.debug(f"diff out ans, psnr = {calc_loss(out, ans, mask)[-1]:.4f} ssim = {ssim_out:.4f}")
                show((out - ans).abs())
                logging.debug(f"diff ref ans, psnr = {calc_loss(ref, ans, mask)[-1]:.4f} ssim = {ssim_ref:.4f}")
                show((ref - ans).abs())
                logging.debug("====================================")

    if plot:
        logging.info(f"valid done psnr = {test_psnr:.4f} ssim = {test_ssim:.4f}")

    return test_psnr

def maml_train(mvsnet, episodes, valid_episodes, save_ckpt, 
        batch_size=2, lr=0.001, alpha=0.001, epoch_fact=100):
    assert isinstance(mvsnet, MVSNetSelfSup), "Should be self-supervised MVSNet"
    epochs = epoch_fact * 10
    opt = torch.optim.Adam(mvsnet.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=epoch_fact, gamma=0.75)

    best_valid_psnr = -1
    best_valid_ckpt = None

    mvsnet.eval()
    for epoch in range(0, epochs + 1):
        if epoch > 0:
            epoch_psnr = 0
            opt.zero_grad()
            for i, episode in enumerate(episodes):
                psnr = maml_train_step(mvsnet, episode, batch_size=batch_size, alpha=alpha)
                epoch_psnr = epoch_psnr + psnr
            for param in mvsnet.parameters():
                if param.grad is not None:
                    param.grad /= len(episodes)
            opt.step()
            sch.step()
            epoch_psnr /= len(episodes)            
            logging.info(f"#{epoch} psnr = {epoch_psnr:.8f}")

        if epoch % (epoch_fact // 4) == 0:
            valid_psnr = 0
            for i, episode in enumerate(valid_episodes):
                psnr = maml_valid_step(mvsnet, episode, batch_size=batch_size, alpha=alpha)
                valid_psnr = valid_psnr + psnr
                logging.info(f"valid #{epoch} episode #{i} psnr = {psnr:.6f}")
            valid_psnr /= len(valid_episodes)

            updated = ""
            if valid_psnr > best_valid_psnr:
                import copy
                best_valid_psnr = valid_psnr
                best_valid_ckpt = {a : b.cpu() for a, b in mvsnet.state_dict().items()}
                updated = "updated"

                torch.save(mvsnet.state_dict(), save_ckpt)

            logging.info(f"valid #{epoch} psnr = {valid_psnr:.8f} {updated}")

    mvsnet.load_state_dict(best_valid_ckpt)
    return mvsnet