import torch
import random
import statistics
import numpy as np

import poselib

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from utils.rotation_utils import get_random_upward, get_upward_with_dev, get_rt_mtx
from double_sol import SolverPipeline, DisplacementRefinerSolver, Camera
from SolverPipeline import P3PBindingWrapperPipeline
from dataclasses import dataclass

from typing import Union

@dataclass
class Config:
    max_depth: float = 10.
    img_width: int = 640
    img_height: int = 640
    focal_length: int = 3 * (img_width * 0.5) / np.tan(60.0 * np.pi / 180.0);
    min_depth: float = 1.
    max_depth: float = 1.1
    inliers_ratio: float = 1.
    outlier_dist: float = 30.
    
    # [TODO][IMPORTNAT]: not properly tested, be aware of using for
    # some experiments
    pixel_noise: float = 2.

conf = Config()

def get_random_image_point(conf: Config):
    x = random.uniform(0, conf.img_width)
    y = random.uniform(0, conf.img_height)
    x = torch.tensor([x, y], dtype=torch.float64)    
    return x

def to_homogeneous(x):
    return torch.cat([x, torch.ones(1)])

def to_camera_coords(x: torch.tensor, conf: Config = conf):
    x = to_homogeneous(x)
    
    x[0] -= conf.img_width // 2
    x[1] -= conf.img_height // 2
    x[:2] /= conf.focal_length
    x /= x.norm()
    
    return x

def generate_correspondence(x: torch.tensor, conf: Config):
    x = to_camera_coords(x, conf)
    x *= random.uniform(conf.min_depth, conf.max_depth)
    
    assert x.shape == (3,)    
    return x

def transform_correspondence(X: torch.tensor, R: torch.tensor, t: torch.tensor):
    return R.T @ (X - t)

def generate_example(R, t, conf: Config = conf):
    x1, x2 = get_random_image_point(conf), get_random_image_point(conf)
    X1, X2 = generate_correspondence(x1.clone(), conf),\
             generate_correspondence(x2.clone(), conf)
    X1, X2 = transform_correspondence(X1, R, t), transform_correspondence(X2, R, t)
    
    # [TODO][IMPORTNAT]: not properly tested, be aware of using for
    # some experiments
    if conf.pixel_noise != 0:
        x1noise = np.random.normal(0, conf.pixel_noise, 2)
        x2noise = np.random.normal(0, conf.pixel_noise, 2)
        
        if torch.all(x1[0] + x1noise > 0) and torch.all(x1[1] + x1noise < conf.img_width):
            x1 += x1noise
            assert x1[0] > 0 and x1[1] < conf.img_width, f"{x1}"
            
        if torch.all(x2[0] + x2noise > 0) and torch.all(x2[1] + x2noise < conf.img_height):
            x2 += x2noise
            assert x2[0] > 0 and x2[1] < conf.img_height, f"{x2}"
        
        assert x1[0] > 0 and x1[1] < conf.img_width, f"{x1}"
        assert x2[0] > 0 and x2[1] < conf.img_height, f"{x2}"

    return x1, x2, X1, X2 

# x: [2, ]
def generate_outlier(x, conf):
    out = get_random_image_point(conf)
    
    while ((out - x).norm() < conf.outlier_dist):
        out = get_random_image_point(conf)
        
    return out

from typing import Tuple
def generate_examples(num_of_examples: int,
                      dev: Tuple[float, float] = (0., 0.),
                      sim_idx: Union[None, int] = None, 
                      conf: Config = conf
                     ):
    num_of_examples = num_of_examples // 2
    
    num_inliers = num_of_examples * conf.inliers_ratio
    num_outliers = num_of_examples - num_inliers
    
    if num_of_examples == 0:
        num_of_examples, num_inliers, num_outliers = 1, 1, 0
    
    if sim_idx is not None:
        R, rand_angle = get_upward_with_dev(sim_idx, *dev), sim_idx
    else:
        R, rand_angle = get_random_upward(*dev)
    t = torch.rand(3, )

    # [TODO] [IMPORTANT]: under such generation we cannot get model where one of the points is an inlier
    xs, Xs, inliers = [], [], []
    for i in range(num_of_examples):
        x1, x2, X1, X2 = generate_example(R, t)
        Xs.append((X1, X2))

        if i < num_inliers:
            xs.append((x1, x2))
            inliers.append(True)
        else:
            xs.append((generate_outlier(x1, conf), generate_outlier(x2, conf)))
            inliers.append(False)
            
    xs = np.concatenate([[p.numpy() for p in elm] for elm in xs])
    Xs = np.concatenate([[p.numpy() for p in elm] for elm in Xs])
    
            
    return xs, Xs, inliers, R, t, rand_angle

def compute_metric(Rgt, tgt, R, t):
    rot_error = np.arccos((np.trace(np.matmul(Rgt, R)) - 1.0) / 2.0) * 180.0 / np.pi
    if np.isnan(rot_error):
        return 1000000.0, 180.0
    else:
        return np.linalg.norm(tgt - t), rot_error

def print_stats(pose_errors, orientation_errors):
    pos_errors = pose_errors
    orient_errors = orientation_errors
    print(" Couldn't localize " + str(orientation_errors.count(180.0)) + " out of " + str(len(orientation_errors)) + " images") 
    print(" Median position error: " +  str(round(statistics.median(pos_errors),3)) + ", median orientation errors: " + str(round(statistics.median(orient_errors),2)))

    med_pos = statistics.median(pos_errors)
    med_orient = statistics.median(orient_errors)
    counter = 0
    for i in range(0, len(pose_errors)):
        if pose_errors[i] <= med_pos and orientation_errors[i] <= med_orient:
            counter += 1
    print(" Percentage of poses within the median: " + str(100.0 * float(counter) / float(len(pose_errors))) + " % ")


if __name__ == "__main__":
    ref = DisplacementRefinerSolver(verbose=False)            
    p3p_solv_pipe = SolverPipeline()
    p2p_solv_pipe = SolverPipeline(up2p=True)
    p3pwrapper = P3PBindingWrapperPipeline(
            ransac_conf = {
                # 'max_reproj_error': args.ransac_thresh
                'min_iterations': 100,
                'max_iterations': 10000,
                'progressive_sampling': True,
                'max_prosac_iterations': 13
                },
                
                bundle_adj_conf = {
                    'loss_scale' : 1.0,
                }                                              
        )

    camera_dict = {
        "width": conf.img_width, 
        "height": conf.img_height, 
        "params": [conf.focal_length, conf.img_width // 2, conf.img_height // 2, 0]
    }

    camera = Camera.from_camera_dict(camera_dict)

    orientation_errors_p3pr, pose_errors_p3pr = [], []
    orientation_errors_p3p, pose_errors_p3p = [], []
    orientation_errors_np, pose_errors_np = [], []
    orientation_errors_p, pose_errors_p = [], []
        
    CRED = '\033[91m'
    CYELLOW = '\033[43m'
    CEND = '\033[0m'
    PRINT = True
    seed = 13
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for sim_idx in tqdm(range(0, 5)):
        xs, Xs, _, Rgt, tgt, rand_angle = generate_examples(100, (0, 40), sim_idx, conf)
        Rgt, tgt = Rgt.numpy(), tgt.numpy()

        if PRINT: print(CRED, Rotation.from_matrix(Rgt).as_euler("XYZ", degrees=True), tgt, CEND)
        if PRINT: print(Rgt)
        assert np.allclose(xs, camera.cam2pix((Rgt @ Xs.T + tgt[:, None]).T), 2)

            # np.random.seed(seed)
            # torch.manual_seed(seed)
            # R, t = p3pwrapper(xs, Xs, camera_dict)
            # pose_error_p3pr, orient_error_p3pr = dataset.compute_metric(Rgt, tgt, R, t)
            # # print("rp3p[pe, oe]: ", pose_error_p3pr, orient_error_p3pr, Rotation.from_matrix(R).as_euler("XYZ", degrees=True), t)
            # orientation_errors_p3pr.append(orient_error_p3pr)
            # pose_errors_p3pr.append(pose_error_p3pr)

            # np.random.seed(seed)
            # torch.manual_seed(seed)
            # R, t = p3p_solv_pipe(xs, Xs, camera_dict) 
            # pose_error_p3p, orient_error_p3p = compute_metric(Rgt, tgt, R, t)
            # if PRINT: print("p3p[pe, oe]: ", pose_error_p3p, orient_error_p3p, Rotation.from_matrix(R).as_euler("XYZ", degrees=True), t)
            # orientation_errors_p3p.append(orient_error_p3p)
            # pose_errors_p3p.append(pose_error_p3p)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        R, t = p2p_solv_pipe(xs, Xs, camera_dict) 
        pose_error_np, orient_error_np = compute_metric(Rgt, tgt, R, t)
        if PRINT: print(CYELLOW, "np[pe, oe]: ", pose_error_np, orient_error_np, Rotation.from_matrix(R).as_euler("XYZ", degrees=True), t, CEND)
        if PRINT: print(R)
        orientation_errors_np.append(orient_error_np)
        pose_errors_np.append(pose_error_np)

        random.seed(seed)            
        np.random.seed(seed)
        torch.manual_seed(seed)
        R, t = ref(xs, Xs, camera_dict)
        pose_error_p, orient_error_p = compute_metric(Rgt, tgt, R, t)
        if PRINT: print(CYELLOW, "p[pe, oe]: ", pose_error_p, orient_error_p, Rotation.from_matrix(R).as_euler("XYZ", degrees=True), t, CEND)
        orientation_errors_p.append(orient_error_p)
        pose_errors_p.append(pose_error_p)
    else:
        # print("\n\nPure P3P: ")
        # print_stats(pose_errors_p3p, orientation_errors_p3p)
        print("\n\nPure Up2P: ")
        print_stats(pose_errors_np, orientation_errors_np)
        print("\n\nPrerotate: ")
        print_stats(pose_errors_p, orientation_errors_p)
