import numpy as np


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)

def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    
    if isinstance(X, np.ndarray):    
        diff = np.square(diff)
        diff = np.sum(diff, 2)
        return np.exp(-diff / (2 * beta**2))
    else:
        diff = torch.square(diff)
        diff = torch.sum(diff, 2)
        return torch.exp(-diff / (2 * beta**2))


def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


''' GMM Registration Helper Functions. Based on:

- OGMM: https://github.com/gfmei/ogmm   
@inproceedings{mei2022overlap,
  title={Overlap-guided Gaussian Mixture Models for Point Cloud Registration},
  author={Mei, Guofeng and Poiesi, Fabio and Saltori, Cristiano and Zhang, Jian and Ricci, Elisa and Sebe, Nicu},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2023},
}

- DeppGMR: https://github.com/wentaoyuan/deepgmr
@inproceedings{yuan2020deepgmr,
  title        = {DeepGMR: Learning Latent Gaussian Mixture Models for Registration},
  author       = {Yuan, Wentao and Eckart, Benjamin and Kim, Kihwan and Jampani, Varun and Fox, Dieter and Kautz, Jan},
  booktitle    = {European Conference on Computer Vision},
  pages        = {733--750},
  year         = {2020},
  organization = {Springer}
}
'''
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, open3d as o3d, matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def numpy_to_torch(inputs, dtype=torch.float32, to_gpu=True):
    if to_gpu:
        return torch.as_tensor(inputs, dtype=dtype).cuda()
    else:
        return torch.as_tensor(inputs, dtype=dtype)

def torch_to_numpy(inputs):
    return inputs.detach().cpu().numpy()

def wkmeans(x, num_clusters=128, dst='feats', iters=5, is_fast=True):
    ''' Wasserstein Weighted K-means clustering. Ref: https://github.com/gfmei/ogmm   
    Inputs:
        - x: [B,N=717,C=3]. Point cloud.
        - num_clusters: int (J=128). Number of clusters.       
        - dst: str. Distance metric. 'feats' for feature space dist or 'eu' for
        euclidean distance.
    Important Intermediate Variables:
        - cost: [B,N,J]. Cost induced by x-to-centroids transportation policy.
        I.e. squared distance between x and centroids.
        - gamma: [B,N,J]. Transportation policy from x to cluster centroids.
        Also can be thought of as soft cluster assignments.
    Outputs:
    '''
    bs, num, dim = x.shape
    if is_fast:
        ids = farthest_point_sample(x, num_clusters, is_center=True)
        centroids = index_points(x, ids)    # (B,M,C=3)
    else:
        ids = torch.randperm(num)[:num_clusters]
        centroids = x[:, ids, :]            # (B,M,C=3)
    gamma, pi = torch.zeros((bs, num, num_clusters), requires_grad=True).to(x), None
    for i in range(iters):
        if dst == 'eu':
            cost = square_distance(x, centroids)
        else:
            x = F.normalize(x, p=2, dim=-1)
            centroids = F.normalize(centroids, p=2, dim=-1)
            cost = 2.0 - 2.0 * torch.einsum('bnd,bmd->bnm', x, centroids)
        gamma = num * sinkhorn(cost, max_iter=10)[0]
        pi, centroids = gmm_params(gamma, x)
    return gamma, pi, centroids

def farthest_point_sample(xyz, npoint, is_center=False):
    """ Ref: https://github.com/gfmei/ogmm   
    Input:
        pts: pointcloud data, [B, N, 3]
        npoint: number of samples
        is_center: if True, the initial farthest point is selected as the centroid 
        of the entire point cloud xyz.
    Return:
        sub_xyz: sampled point cloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(xyz) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    if is_center:
        centroid = xyz.mean(1).view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """ Selects a subset of points from a larger set of points based on idx values. Ref: https://github.com/gfmei/ogmm   
    Input:
        feats: input feats data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed feats data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)    
    view_shape[1:] = [1] * (len(view_shape) - 1)    # (B, 1)
    repeat_shape = list(idx.shape)  
    repeat_shape[0] = 1 # (1, S)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # (B,S)
    new_points = points[batch_indices, idx, :]
    return new_points

def gmm_params(gamma, pts, return_sigma=False):
    """ Return GMM parameters (pi, mu, sigma) given cluster assignments and points. Ref: https://github.com/gfmei/ogmm   
    Input:
        - gamma: [B,N,J=128]. Soft cluster assignments (i.e. transportation policy). 
        J is number of clusters, i.e. number of gaussian components in the GMM. Created by wkeans().
        - pts:   [B,N,D=3].
    Output: GMM parameters
        - pi:    [B,N]
        - mu:    [B,N,3]
        - sigma: [B,N,3,3](optional). Covariance matrices 
    """
    # pi: B feats J
    D = pts.size(-1)
    pi = gamma.mean(dim=1)              # (B,N,J) -> (B,J)
    npi = pi * gamma.shape[1] + 1e-5    # (B,J)
    # p: B feats J feats D
    mu = gamma.transpose(1, 2) @ pts / npi.unsqueeze(2)     # (B,J,N) @ (B,N,D) -> (B,J,D)
    if return_sigma:
        # diff: B feats N feats J feats D
        diff = pts.unsqueeze(2) - mu.unsqueeze(1)
        # sigma: B feats J feats 3 feats 3
        eye = torch.eye(D).unsqueeze(0).unsqueeze(1).to(gamma.device)
        sigma = (((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() *
                  gamma).sum(dim=1) / npi).unsqueeze(2).unsqueeze(3) * eye
        return pi, mu, sigma
    return pi, mu

def square_distance(src, dst, normalize=False):
    """ Calculate Euclid distance between each two src, dst. Ref: https://github.com/gfmei/ogmm   
    src_xyz^T * ref_xyz = xn * xm + yn * ym + zn * zm
    sum(src_xyz^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(ref_xyz^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src_xyz**2,dim=-1)+sum(ref_xyz**2,dim=-1)-2*src_xyz^T*ref_xyz
    Input:
        src_xyz: ref_xyz src_xyz, [B, N, C]
        ref_xyz: target src_xyz, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalize:
        return 2.0 + dist
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.clamp(dist, min=1e-12)
    return dist

def sinkhorn(cost, p=None, q=None, epsilon=1e-2, thresh=1e-2, max_iter=100):
    ''' Sinkhorn-Knopp algorithm for optimal transport. Ref: https://github.com/gfmei/ogmm   
    Call stack: GMMReg.forward() -> get_anchor_corrs() -> wkeans() -> sinkhorn()
    
    Input:
        - cost: Cost matrix.
        - p: [B,N=717]. Source distribution.
        - q: [B,J=128]. Target distribution.
    Output:
        - gamma: Optimal transport matrix.
        - loss: Sinkhorn loss.
    '''
    
    if p is None or q is None:
        batch_size, num_x, num_y = cost.shape
        device = cost.device
        if p is None:
            p = torch.empty(batch_size, num_x, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
        if q is None:
            q = torch.empty(batch_size, num_y, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, gmmlib iterations
    for i in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(cost, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(cost, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff).detach()
        if mean_diff.item() < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(cost, u, v, epsilon)
    gamma = torch.exp(K)
    # Sinkhorn distance
    loss = torch.sum(gamma * cost, dim=(-2, -1))
    return gamma, loss.mean()

def log_boltzmann_kernel(cost, u, v, epsilon):
    kernel = (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel

def gmm_register(pi_s, mu_s, mu_t, sigma_t):
    ''' Ref: https://github.com/wentaoyuan/deepgmr
    Inputs:
        pi: B x J
        mu: B x J x 3
        sigma: B x J x 3 x 3
    '''
    c_s = pi_s.unsqueeze(1) @ mu_s
    c_t = pi_s.unsqueeze(1) @ mu_t
    Ms = torch.sum((pi_s.unsqueeze(2) * (mu_s - c_s)).unsqueeze(3) @
                   (mu_t - c_t).unsqueeze(2) @ sigma_t.inverse(), dim=1)
    U, _, V = torch.svd(Ms.cpu())
    U = U.cuda()
    V = V.cuda()
    S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    S[:, 2, 2] = torch.det(V @ U.transpose(1, 2))
    R = V @ S @ U.transpose(1, 2)
    t = c_t.transpose(1, 2) - R @ c_s.transpose(1, 2)
    bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
    T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
    return T




''' -------- Experiment Helper -------- '''

def sample_pcd(X, num_samples=None):
    assert X.shape[0] == 3, f"X should have shape (3, N), but instead got {X.shape}"
    if num_samples is None: return X
    
    idx = np.random.randint(low = 0, high = X.shape[1], size = num_samples)
    return X[:,idx]

def find_nn_corr(src, tgt):
    ''' Given two input point clouds, find nearest-neighbor correspondence (from source to target) 
    Input:
        - src: Source point cloud (n*3), either array or open3d pcd
        - tgt: Target point cloud (n*3), either array or open3d pcd
    Output:
        - idxs: Array indices corresponds to src points, 
            array elements corresponds to nn in tgt points (n, np.array)
    '''

    ''' Way1: Sklearn'''
    if src.shape[1] != 3: src = src.T
    if tgt.shape[1] != 3: tgt = tgt.T
    
    if not isinstance(src, np.ndarray):
        src = np.asarray(src.points)    # (16384*3)
        tgt = np.asarray(tgt.points)
    
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tgt)
    dists, idxs = neighbors.kneighbors(src)  # (16384*1), (16384*1)
    return idxs.flatten()

def pcd_to_o3d(pcd):
    ''' Convert np array (n,3) to open3d pcd'''
    points = o3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = points
    return points_o3d

def pcd_to_o3d_rgb(pcd, rgb):
    ''' Convert np array (n,3) to open3d pcd'''
    points = o3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    colors = o3d.utility.Vector3dVector(
        rgb.reshape([-1, 3]))  # TODO: What's this
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = points
    pcd_o3d.colors = colors
    return pcd_o3d


''' -------- Arithmetic --------'''
def skew_sym(x):    
    ''' Given a vector x, apply skew-symmetric operator '''
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])    

def vec_to_rot(R0, w):
    ''' Given a initial (3,3) rotation R0 and an (3,) angle w from R0
    compute new rotation matrix'''
    return R0 @ (np.eye(3) + skew_sym(w))

def mse(X, Y): return np.mean((X-Y)**2)

def sse(X,Y): return np.sum((X-Y)**2)


''' --------- Transformation ---------'''

def get_rand_rotation():
    ''' Get a random (3,3) rotation matrix as np.array
    ref: http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html#Rotation
    '''
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    angles = np.random.uniform(-np.pi/2, np.pi/2, size=3)
    return mesh.get_rotation_matrix_from_xyz(angles)

def get_rand_translation(distance=1):
    ''' Get a random (3,) translation vector as np.array ''' 
    t = np.random.uniform(low=-1, high=1, size=3)
    return (t / np.linalg.norm(t)) * distance

def get_rand_transform():
    ''' Get a (4, 4) general transformation matrix in SE(3)'''
    transformation = np.eye(4)
    transformation[:3,:3] = get_rand_rotation()
    transformation[3,:3] = get_rand_translation()
    return transformation
    
def gaussian_corruption(pcd, std=0.03):
    noise = np.random.normal(0, std, size = pcd.shape)
    return pcd + noise


def apply_transform_batch(pts, rot, trans):
    ''' Apply SE(3) transformation to a batch of point clouds
    Input: 
        - pts       (B,3,N)             Source point cloud.
        - rot:      (B,3,3)             Rotation matrices
        - trans:    (B,3,1) or (B,3)    Translation vectors
    Return:
        - rot @ trans + trans for each point cloud in batch B.
    '''
    assert trans.shape[-1]==3, f'trans should be of shape [B,3], but got {trans.shape}'
    assert pts.shape[1]==3, f'pts should be of shape [B,3,N], but got {pts.shape}'
    if trans.shape[-1]==1: trans=trans.squeeze(-1)
    
    try:    return torch.einsum('...ij,...jk->...ik', rot, pts).transpose(-1,-2) + trans.unsqueeze(1)
    except: return np.einsum('...ij,...jk->...ik', rot, pts).transpose(-1,-2) + trans.unsqueeze(1)


def apply_transform(pts, rot, trans):
    ''' Apply SE(3) transformation to point cloud
    Input:
        - pts       (N,3)                   Source point cloud.
        - rot:      (3,3)                   Rotation matrices
        - trans:    (3,1) or (1,3) or (3)   Translation vectors
    Return:
        - (N,3) transformed point cloud
    '''
    assert len(pts.shape)==2 and (pts.shape[-1]==3 or pts.shape[0]==3), \
        f'pts should be of shape [N,3] or [3,N], but got {pts.shape}'
    assert rot.shape==(3,3), f'rot should be of shape [3,3], but got {rot.shape}'
    assert len(trans.shape) <=2 and 3 in trans.shape, \
          f'trans should be of shape [3,1], [1,3] or [3], but got {trans.shape}'

    if pts.shape[0]==3: pts = pts.T
    if len(trans.shape)==2: trans=trans.squeeze()
    return (rot @ pts.T).T + trans
    

''' --------- Visualization --------- '''

def pcd2depth(pcd, collate_uv=True, height=480, width=640, valid_margin=0, round_coords=True, 
    K = np.array([[883.0000,   0.0000, 445.0600,   0.0000],
                [  0.0000, 883.0000, 190.2400,   0.0000],
                [  0.0000,   0.0000,   1.0000,   0.0000],
                [  0.0000,   0.0000,   0.0000,   1.0000]])):
    ''' Convert 3D point cloud to coordinates on image plane (y*HEIGHT+x) 
        Note: A screen coordinate in `coord `is an int. For example:
        ---------------------
        | 0 | 1 | 2 | 3 | 4 | 
        | 5 | 6 | 7 | 8 | 9 |
        |10 |11 |12 |13 |14 |
        ---------------------
    '''
    if torch.is_tensor(pcd): pcd = torch_to_numpy(pcd.squeeze())

    X, Y, Z = pcd[...,0], pcd[...,1], (pcd[...,2] + 1e-8)
    
    u = X * K[0,0] / Z + K[0,2]
    v = Y * K[1,1] / Z + K[1,2]

    if round_coords:
        u = u.round().astype(np.long)      # (N,). Screen coordinate for each point
        v = v.round().astype(np.long)      # (N,). Screen coordinate for each point

    coords = v * width + u
    valid_proj = (v >= valid_margin) & (v < height-1-valid_margin) & \
        (u >= valid_margin) & (u < width-1-valid_margin)    # 
    
    if collate_uv: return np.column_stack([u,v])
    else:          return v, u, coords, valid_proj

def plot_pcd(pcd, title=None, ax_lim=None, path=None): 
    ''' Plot np.array (n,3)
        - ax_lim: 2d List (3,2). Min, max limit for x,y,z axis.
    '''
    if not isinstance(pcd, np.ndarray):
        pcd = np.array(pcd.points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('persp')
    if ax_lim is None:
        ax.set_xlim3d([-3,3]), ax.set_ylim3d([-3,3]), ax.set_zlim3d([-3,3])
    else:
        ax.set_xlim3d(ax_lim[0]); ax.set_ylim3d(ax_lim[1]); ax.set_zlim3d(ax_lim[2])

    ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], marker='.', alpha=1.0, edgecolors='none')
    if title is not None: plt.title(title)
    plt.show()
    if path is not None: plt.savefig(f"./imgs/{path}.png")
    plt.clf()

def compare_pcd(pcds, scale=0.1, title=None, labels=None, path=None):
    if labels is None: labels = ['pts1', 'pts2=r @ pts1 + t', 'pts3']
    dpi = 80
    fig = plt.figure(figsize=(1440/dpi, 720/dpi), dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    axlim = scale * np.array([-1,1])

    ax.set_xlim3d(axlim), ax.set_ylim3d(axlim), ax.set_zlim3d(axlim)
    ax.set_proj_type('persp')

    for points, label in zip(pcds, labels):
        ax.scatter(points[:, 0], points[:, 2], points[:, 1],
                marker='.', alpha=0.5, edgecolors='none', label=label)

    if title is not None: plt.title(title)
    if path is not None: plt.savefig(f"./imgs/{path}.png")
    plt.legend(); plt.show(); plt.clf()

def compare_pcd_2d(pcds, title=None, labels=None, path=None):
    if labels is None: labels = ['pts1', 'pts2=r @ pts1 + t', 'pts3']

    for points, label in zip(pcds, labels):
        if isinstance(points, torch.Tensor): points = torch_to_numpy(points.cpu())
        plt.scatter(points[:,0], points[:,1], marker='.', alpha=0.5, label=label)
    
    plt.legend()
    if title is not None: plt.title(title)
    plt.show()
    if path is not None: plt.savefig(f"./imgs/{path}.png")
    

def plt_gmm(pts, centroids, axlim=0.1, title=None, path=None):
    '''
    Input:
        - pts:          (N, 3) np array 
        - centroids:    (J, 3) np array 
    '''
    dpi = 80
    fig = plt.figure(figsize=(1440/dpi, 720/dpi), dpi=dpi)
    ax = fig.add_subplot(projection='3d')
    
    ax.set_xlim3d([-axlim,axlim]), ax.set_ylim3d([-axlim,axlim]), ax.set_zlim3d([-axlim,axlim])
    ax.set_proj_type('persp')

    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], marker='.', alpha=0.5, edgecolors='none', label='pts')
    ax.scatter(centroids[:, 0], centroids[:, 2], centroids[:, 1], marker='x', alpha=1, edgecolors='none', label='mu')

    if title is not None: plt.title(title)
    if path is not None: plt.savefig(f"./imgs/{path}.png")
    plt.legend(); plt.show(); plt.clf()

def plt_gmm_o3d(pts, centroids, title=None, path=None):
    '''
    Input:
        - pts:          (N, 3) np array 
        - centroids:    (J, 3) np array 
    '''
    pts = pcd_to_o3d_rgb(pts, np.array([[0.1,0.1,0.1]]).repeat(pts.shape[0],axis=0))
    centroids = pcd_to_o3d_rgb(centroids, np.array([[1,0,0]]).repeat(centroids.shape[0],axis=0))
    o3d.visualization.draw_geometries([pts, centroids])


def numpy_to_torch(inputs, dtype=torch.float32, to_gpu=True):
    if to_gpu:
        return torch.as_tensor(inputs, dtype=dtype).cuda()
    else:
        return torch.as_tensor(inputs, dtype=dtype)

def torch_to_numpy(inputs):
    return inputs.detach().cpu().numpy()
