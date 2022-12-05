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
TRAIN_SCENES = ["chair", "ficus", "materials", "mic"]
TEST_SCENES = ["drums", "hotdog", "lego", "ship"]

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
    
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def loader(self, **kwargs):
        return torch.utils.data.DataLoader(self, **kwargs)

def read_train_data(SCENE, debug=False):
    pairs = torch.load(f"{SCENES_DIR}/mvsnerf_pairs.pth")
    batch = []

    DATASET = f"{SCENES_DIR}/{SCENE}/"

    train_views = pairs[f"{SCENE}_train"]
    test_views = pairs[f"{SCENE}_test"]
    
    logging.info(f"Read #{SCENE} train_views = {train_views}")  

    pts = torch.load(f"{DATASET}/cloud_ref.pth")
    # plot(pts)
    logging.info(f"cloud shape = {pts.shape}")

    train_config = json.load(open(f"{DATASET}/transforms_train.json"))

    all_data = []
    cam_centers = []

    for i in tqdm(range(len(train_config['frames']))):
        cam, proj, img, mask = read_data_ns(DATASET, train_config, i)
        
        cam_centers.append(cam[2])

        W, H, _ = img.shape
        pix = pts2pix_ns(cam, pts)
        pix = pix.long()

        inliers = (pix[:, 0] >= 0) & (pix[:, 1] >= 0) & (pix[:, 0] < H) & (pix[:, 1] < W)
        pix = pix[inliers]

        pix_ind = pix[:, 0] * W + pix[:, 1]
        # pts_proj = pix2pts_ns(cam, pix_ind)
        dep = ((pts[inliers] - cam[2]).norm(dim=1) - DEP_L) / (DEP_R - DEP_L)
        pix_dep = torch.ones(W * H, dtype=torch.float) * 10
        
        # prevent scatter_reduce from printing
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):           
            pix_dep.scatter_reduce_(0, pix_ind, dep, reduce="amin")

        pix_dep = pix_dep.reshape(W, H)

        holes = (pix_dep > 2) & mask
        pix_dep = torch.from_numpy(cv2.inpaint(pix_dep.clamp(0, 1).numpy(), holes.numpy().astype(np.uint8), 3, cv2.INPAINT_TELEA)).float()
        pix_dep = torch.from_numpy(cv2.medianBlur(pix_dep.clamp(0, 1).numpy(), 5)).float()

        pix_dep[~mask] = 1
        dep = 1 - pix_dep.clamp(min=0, max=1)

        if debug:
            # show(img)
            show(1 - dep.clamp(0, 1))

        all_data.append((proj, img, mask, dep))    

    
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

    def forward(self, batch_imgs, batch_cams, prob_only=False):
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
        return pred_deps, conf, features, prob_volume
        


class MVSNetMAML(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.mvsnet = MVSNetPretrained(ckpt)
        self.loss_net = nn.Sequential(
            nn.Conv2d(24, 12, 5, 5, 5),
            nn.ReLU(),
            nn.Conv2d(12, 4, 5, 5, 5),
        )

    def forward(self, *args, training=False):
        if training:
            features = self.mvsnet(*args, prob_only=True)[-1]
            # (B, 192, H, W)
            B, _, H, W = features.shape
            features = features.reshape(B, 24, 8, H, W).mean(dim=2)
            # (B, 24, H, W)

            maml_loss = self.loss_net(features).mean(dim=(-1,-2)).norm(dim=-1).mean()
            return maml_loss

        pred_deps = self.mvsnet(*args, prob_only=False)[0]
        return pred_deps

def fix_name(name):
    import re
    fixed = re.sub(r"\.(\d{1,})\.", r"[\1].", name)
    return fixed

def maml_train_step(mvsnet_orig, episode, batch_size=2, alpha=0.02):
    import copy
    mvsnet = copy.deepcopy(mvsnet_orig)
    mvsnet.zero_grad()

    var_names = [fix_name(name) for name, _ in mvsnet.named_parameters()]

    for name in var_names:
        exec(f"del mvsnet.{name}")
        var = eval(f"mvsnet_orig.{name}")
        exec(f"mvsnet.{name} = var.clone()")

    mvsnet.eval()
    train_loader = tqdm(episode.loader(batch_size=batch_size, shuffle=True, pin_memory=True))
    train_loader.set_description("train")
    for (batch_cams, batch_imgs, batch_masks, batch_deps) in train_loader:
        maml_loss = mvsnet(batch_imgs, batch_cams, training=True)

        params = [eval(f"mvsnet.{name}", {"mvsnet" : mvsnet}) for name in var_names]
        grad = torch.autograd.grad(maml_loss, params, create_graph=True, allow_unused=True)
        for name, g in zip(var_names, grad):
            if g is not None:
                exec(f"mvsnet.{name} = mvsnet.{name} - alpha * g", 
                    {"mvsnet" : mvsnet, "alpha" : alpha, "g" : g})

    mvsnet.eval()
    test_loader = tqdm(episode.loader(batch_size=batch_size * 2, shuffle=False, pin_memory=True))
    test_loader.set_description("test")
    test_loss = 0
    for (batch_cams, batch_imgs, batch_masks, batch_deps) in test_loader:
        count = batch_imgs.shape[0]
        pred_deps = mvsnet(batch_imgs, batch_cams)
        test_loss = test_loss + mvsnet_loss(pred_deps, batch_deps, batch_masks) * count

    test_loss = test_loss / len(episode)

    return test_loss

def maml_train(mvsnet, episodes, batch_size=2, lr=0.01, alpha=0.02, epochs=100):
    opt = torch.optim.Adam(mvsnet.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)

    mvsnet.eval()
    for epoch in range(epochs):
        opt.zero_grad()

        epoch_loss = 0
        for i, episode in enumerate(episodes):
            logging.info(f"train #{epoch} episode #{i}")
            loss = maml_train_step(mvsnet, episode, batch_size=batch_size, alpha=alpha)
            loss.backward()
            epoch_loss = epoch_loss + loss.item()
        epoch_loss /= len(episodes)
        
        opt.step()
        sch.step()

        logging.info(f"train #{epoch} loss = {epoch_loss:.6f}")

