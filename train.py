import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy
import logging

SCENES_DIR = "./datasets/NerfSyn/"
TRAIN_SCENES = ["chair", "ficus", "materials"]
TEST_SCENES = ["drums", "hotdog", "lego", "mic", "ship"]

DEFAULT_VIEW_DIR = [-1., -1., -1.]
BG_COLOR = [1, 1, 1]

IMG_W = 800
IMG_H = 800

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
def plot(pts, conf=None, rgb=None, pov=[0], revcol=[False], dpi=64, save=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    import numpy as np

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

            ax.scatter(x, y, z, c=rgba, marker='.')

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
    return cam, img

def read_train_data():
    pairs = torch.load(f"{SCENES_DIR}/mvsnerf_pairs.pth")
    for SCENE in TRAIN_SCENES:
        DATASET = f"{SCENES_DIR}/{SCENE}/"

        train_views = pairs[f"{SCENE}_train"]
        test_views = pairs[f"{SCENE}_test"]
        
        logging.info(f"Read #{SCENE} train_views = {train_views} test_views = {test_views}")    

        train_config = json.load(open(f"{DATASET}/transforms_train.json"))
        valid_config = json.load(open(f"{DATASET}/transforms_test.json"))

        print("233")