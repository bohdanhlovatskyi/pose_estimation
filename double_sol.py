import os
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(16,10)})

import torch
import pickle

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_quaternion

import poselib

from solver import Up2P, Solver
from SolverPipeline import P3PBindingWrapperPipeline
from SolverPipeline import SolverPipeline as SP

from typing import Tuple, Dict, List

import warnings
warnings.filterwarnings("ignore")

class Config:
    ransac_thresh = 13.
    max_side_length = 320
    max_ransac_iters = 10000

# convention is that [R t] maps from the world coordinate system into the camera coordinate system
# upright solvers: 
'''
R = [
        a 0 -b
        0 1 0
        b 0 a
    ]
'''
# it necessary to pre-rotate the input such that this is satisfied (?)
# A * [t1,t2,t3,q^2] + b * [q, 1] = 0
# 

# 
#  R t  - view matrix - t - position of world origin in camera coordinates
#  "how world transformed relative to the camera"
#  

def pl_to_scipy(q: List[float]):
    return [*q[1:], q[0]]

class Camera:
    def __init__(self,
                 w: int,
                 h: int,
                 f: float,
                 cc: Tuple[int, int]
                ) -> None:
        self.f = f
        self.cx, self.cy = cc
        self.w = w
        self.h = h
        
    def pix2cam(self, x: np.ndarray):
        x = np.concatenate([x, np.ones(x.shape[0])[:, np.newaxis]], axis=1)
        x[:, 0] -= self.cx
        x[:, 1] -= self.cy
        x[:, :2] /= self.f
        x /= np.linalg.norm(x)
            
        return x
    
    def cam2pix(self, x: np.ndarray):
        x /= x[:, 2][:, None]
        x[:, :2] *= self.f
        x[:, 0] += self.cx
        x[:, 1] += self.cy
        
        return x[:, :2]

    @staticmethod
    def from_camera_dict(camera: Dict):
        w, h = camera["width"], camera["height"]
        params = camera["params"]
        f, cx, cy, _ = params
        
        return Camera(w, h, f, (cx, cy))
    
class Sampler:
    
    def __call__(self, pts: np.array, sample_size: int):
        n = len(pts)
        assert n > sample_size
        
        idcs = np.random.choice(n, sample_size)
        
        return pts[idcs], idcs


class UP2PSolverPipeline(SP):
    
    def __init__(self, num_models_to_eval: int = 20, verbose: bool = False):
        self.solver = Up2P()
        self.sampler = Sampler()
        self.camera: Camera = None
        self.num_models_to_eval = num_models_to_eval
        self.verbose = verbose
    
    def __call__(self, pts2D, pts3D, camera_dict):
        self.camera = Camera.from_camera_dict(camera_dict)
        pts2D = self.camera.pix2cam(pts2D)
        
        solution, sol_err = None, float("+inf")

        iterator = range(self.num_models_to_eval)        
        for i in (tqdm(iterator) if self.verbose else iterator):
            try:
                # +1 for evaluation in here
                min_sample_size = self.solver.get_sample_size()
                pts2d, idcs = self.sampler(pts2D, min_sample_size + 1)
                pts3d = pts3D[idcs]

                solv_res = self.solver(pts2d[:min_sample_size], pts3d[:min_sample_size])
                for sol in solv_res:
                    R, t = sol
                    R, t = R.detach().cpu().numpy(), t.detach().cpu().numpy()
                    translated = R @ (pts3d[min_sample_size] - t)
                    translated = self.camera.cam2pix(translated[None, :])[0]
                    cerr = np.linalg.norm(translated - pts2d[min_sample_size][:2])
                    if cerr < sol_err:
                        sol_err = cerr
                        solution = sol
            except Exception as ex:
                print(ex)
                continue
        
        return solution

class DisplacementRefinerSolver(Solver):
    
    def __init__(self,
     min_sample_size: int = 100,
     outer_models_to_evaluate: int = 20,
     inner_models_to_evaluate: int = 1,
     rotations_to_est: int = 1000,
     verbose: bool = False
     ) -> None:  
        self.min_sample_size = min_sample_size
        self.outer_models_to_evaluate = outer_models_to_evaluate
        self.inner_models_to_evaluate = inner_models_to_evaluate
        self.rotations_to_est = rotations_to_est
        self.internal_solver = Up2P()
        self.sampler: Sampler = Sampler()
        self.verbose = verbose
        self.camera: Camera = None
    
    def get_sample_size(self) -> int:
        return self.min_sample_size

    def get_prerotation_given_guess(self, R, t, X, x):
        # given initial guess of R and t, project Xs so to get the displacements
        projs = []
        for x3d in X:
            proj = R.T @ (x3d - t)
            proj = self.camera.cam2pix(proj[None, :])[0]
            projs.append(proj)

        # get the proposed prerotation angles, given the displacements
        x = self.camera.cam2pix(x)
        _, angles = self._get_rot_angles(np.stack(x)[:, :2], np.stack(projs)[:, :2])
        deg_angles = [angle.as_euler("XYZ", degrees=True)[2] for angle in angles]

        if self.verbose:
            plt.hist(deg_angles, density=True, color='black', bins=np.arange(-180, 180, 5))
            plt.xticks(range(-180, 180, 5))
            plt.show()
        
        try:
            counts = np.bincount([angle + 180.0 for angle in deg_angles])
            counts = np.convolve(counts, np.ones(3), 'same')
            angle_to_prerotate = np.argmax(counts) - 180

            prerotate_with = Rotation.from_euler("XYZ", [0, 0, angle_to_prerotate], degrees=True).as_matrix()
            return prerotate_with
        except:
            return Rotation.identity().as_matrix()

    def get_prerotation_ls(self, R, t, X, x):
        projs = []
        for x3d in X:
            proj = R.T @ (x3d - t)
            proj = self.camera.cam2pix(proj[None, :])[0]
            projs.append(proj)

        x = self.camera.cam2pix(x)

        x = np.stack(x)[:, :2]
        projs = np.stack(projs)[:, :2]

        # normalize
        # TODO: clearly wrong as we do not take into account center of rotation
        x -= np.mean(x, axis=0)
        projs -= np.mean(projs, axis=0)

        # uniform scale
        x /= np.mean(np.square(x))
        projs /= np.mean(np.square(projs))

        # (x, y), (w, z)
        angle_to_prerotate = np.arctan2(
            np.sum(projs[:, 0] * x[:, 1] - projs[:, 1] * x[:, 0]),
            np.sum(projs[:, 0] * x[:, 0] + projs[:, 1] * x[:, 1])
        )

        prerotate_with = Rotation.from_euler("XYZ", [0, 0, angle_to_prerotate], degrees=False).as_matrix()

        # print(np.rad2deg(angle_to_prerotate))
        # print(prerotate_with)

        return prerotate_with

    def get_prerotation_dummy(self, R, t, X, x):
        return Rotation.from_euler("XYZ", [0, 0, 0], degrees=False).as_matrix()

    def inner_solver(self, x, X):
        err, Rf, tf = None, None, None

        for _ in range(self.inner_models_to_evaluate):
            # take the minimal subset from points
            min_sample_size = self.internal_solver.get_sample_size()
            idcs = np.random.choice(len(X), min_sample_size + 1)
            subx, subX = x[idcs], X[idcs] 

            try:
                inner_solv_res = self.internal_solver(subx[:min_sample_size], subX[:min_sample_size])
            except:
                return None, None

            for Ri, ti in inner_solv_res:
                Ri, ti = Ri.cpu().numpy(), ti.cpu().numpy()
                rp = Ri.T @ (subX[min_sample_size] - ti)
                rp = self.camera.cam2pix(rp[None, :])[0]                    
                cerr = np.linalg.norm(subx[min_sample_size, :2] - rp)
                if err is None or cerr < err:
                    err = cerr
                    Rf, tf = Ri, ti

        return Rf, tf
    

    @torch.no_grad()
    def __call__(self, x, X, camera_dict):
        assert x.shape[0] == X.shape[0]
        
        min_sample_size = self.internal_solver.get_sample_size()

        self.camera = Camera.from_camera_dict(camera_dict)
        x = self.camera.pix2cam(x)
        
        err, Rf, tf = None, None, None

        for i in range(self.outer_models_to_evaluate):
            pts2d, idcs = self.sampler(x, min_sample_size + 1)
            pts3d = X[idcs]
            
            # run the solver with default gravity direction so to get an estimate on prerotation
            try:
                solv_res = self.internal_solver(pts2d[:min_sample_size], pts3d[:min_sample_size])
            except:
                continue

            for (R, t) in solv_res:
                R, t = R.detach().cpu().numpy(), t.detach().cpu().numpy()
                # TODO: collect 100 in all of the modles and cmoputer err noot only ono one point

                # prerotate_with = self.get_prerotation_given_guess(R, t, X, x)
                # prerotate_with = self.get_prerotation_ls(R, t, X, x)
                prerotate_with = self.get_prerotation_dummy(R, t, X, x)

                # prerotate, solve, rotate back
                Xs = np.array([prerotate_with.T @ x3d for x3d in X.copy()])
                IR, It = self.inner_solver(x, Xs)
                if IR is None or It is None:
                    continue

                R = prerotate_with @ IR
                # print("hmmm", t, It, -R.T @ It)
                
                # compute quality of the model
                rp = R.T @ (pts3d[min_sample_size] - It)
                rp = self.camera.cam2pix(rp[None, :])[0]
                cerr = np.linalg.norm(rp - pts2d[min_sample_size, :2])
                if err is None or cerr < err:
                    err = cerr
                    Rf, tf = R, It
                
        if Rf is None or tf is None:
            return None

        pose = poselib.CameraPose()
        pose.q = matrix_to_quaternion(torch.tensor(Rf))
        pose.t = tf
        
        return pose
                    
    def _get_rot_angles(self, gt: np.array, proj: np.array):
        centers = []
        indexes = []
        angles = []
        for _ in range(self.rotations_to_est):
            idcs = np.random.choice(len(gt), 2)
            indexes.append(idcs)

            gt1, proj1 = gt[idcs[0]], proj[idcs[0]]
            gt2, proj2 = gt[idcs[1]], proj[idcs[1]]

            c = self._get_center_of_rotation(gt1, proj1, gt2, proj2)
            centers.append(c)
            
        mean_c = np.array([np.median([elm[0] for elm in centers]), np.median([elm[1] for elm in centers])])

        for i in range(self.rotations_to_est):
            idcs = indexes[i]

            gt1, proj1 = gt[idcs[0]], proj[idcs[0]]
            gt2, proj2 = gt[idcs[1]], proj[idcs[1]]

            try:
                angle = self._get_rotation(gt1, proj1, gt2, proj2, mean_c)
            except Exception as ex:
                continue

            angles.append(angle)

        return centers, angles            

    def _get_intersect(self, a1, a2, b1, b2):
        """ 
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1,a2,b1,b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return (x/z, y/z)
    
    def _get_norm_of_disp(self, gt, proj):
        vec = (proj - gt)
        return (-vec[1], vec[0])
    
    def _get_center_of_rotation(self, gt1, proj1, gt2, proj2):
        n1, n2 = self._get_norm_of_disp(gt1, proj1), self._get_norm_of_disp(gt2, proj2)

        first_center = (gt1 + proj1) / 2
        second_center = (gt2 + proj2) / 2

        c = self._get_intersect(
            first_center,
            first_center + n1,
            second_center,
            second_center + n2,
        )


        return c
    
    def _get_rotation(self, gt1, proj1, gt2, proj2, c):
        cgt1 = gt1 - c
        cproj1 = proj1 - c

        cgt2 = gt2 - c
        cproj2 = proj2 - c

        res = Rotation.align_vectors(
            a=np.array(
                [[*cgt1, 0],
                 [*cgt2, 0]]
            ),
            b=np.array(
                [
                    [*cproj1, 0],
                    [*cproj2, 0]
                ]
            )
        )

        return res[0]

class Dataset:

    CAMERAS_PATH = "dataset/StMarysChurch_matches/st_marys_church_list_queries_with_intrinsics_simple_radial_sorted.txt"
    GTS_PATH = "dataset/StMarysChurch_matches/dataset_test.txt"
    BASE = "dataset/StMarysChurch_matches"

    def __init__(self,
        cameras_path: str = CAMERAS_PATH,
        gts_path: str = GTS_PATH,
        verbose: bool = False
     ) -> None:
        self.verbose = verbose

        self.cameras = self._prepare_cameras(cameras_path)
        self.gts = self._prepare_gts(gts_path)
    
        self.idx: int = 0
        # self.seq = [3, 5, 13]
        self.seq = [3]
        self.paths = []
        for s in self.seq:
            for f in os.listdir(f"{self.BASE}/seq{s}"):
                if f.split(".")[1] != "npy":
                    continue

                self.paths.append(f"{self.BASE}/seq{s}/{f}")
        
        print("Created dataset with matches: ", len(self.paths))

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        try:
            path = self.paths[self.idx]
        except IndexError:
            raise StopIteration
        
        data = np.load(path)
        pts2D = list(data[:, :2])
        pts3D = list(data[:, 2:])

        # get gts
        pp = "/".join(path.split("/")[-2:])
        pp = pp.replace("_matches", "")
        pp = pp.replace(".npy", ".png")
        camera_dict = self.cameras[pp]
        gt = self.gts[pp]
        c, r = gt[:3], gt[3:]

        gt_pose = poselib.CameraPose()
        R = Rotation.from_quat(pl_to_scipy(r)).as_matrix()
        t = - R @ c 

        self.idx += 1

        return (R, t, pts2D, pts3D, camera_dict)

    def compute_metric(self, Rgt, tgt, R, t):
        if R is None or t is None:
            return 1000000.0, 180.0
      
        rot_error = np.arccos((np.trace(np.matmul(R.T, Rgt)) - 1.0) / 2.0) * 180.0 / np.pi

        if self.verbose:
            print(" Position error: " + str(np.linalg.norm(tgt - t)) + " orientation error: " + str(rot_error))
        
        if np.isnan(rot_error):
            return 1000000.0, 180.0
        else:
            return np.linalg.norm(tgt - t), rot_error

    def _prepare_cameras(self, cameras_path: str):
        with open(cameras_path) as file:
            data = file.readlines()

        camera_dict = {}
        for _, line in enumerate(data):
            # image width, image height, focal length, x of pp, y of pp, radial distortion factor 
            path, cam_type, w, h, f, x, y, rd = line.split()
            scaling_factor = 320 / max(np.float32(w), np.float32(h))
    
            # camera = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [960, 600, 400]}
            camera_dict[path] = {
            'model': cam_type,
            'width': int(np.float32(w) * scaling_factor),
            'height': int(np.float32(h) * scaling_factor),
            'params': list(map(float, [np.float32(f) * scaling_factor,
                                    np.float32(x) * scaling_factor,
                                    np.float32(y) * scaling_factor,
                                    np.float32(rd)])),
            }

        return camera_dict

    def _prepare_gts(self, path: str):
        with open(path) as file:
            data = file.readlines()

        gts = {}
        # ImageFile, Camera Position [X Y Z W P Q R]
        for _, line in enumerate(data):
            try:
                # seq13/frame00158.png 25.317314 -0.228082 54.493720 0.374564 0.002123 0.915022 -0.149782
                path, x, y, z, w, p, q, r = line.split()
                rest = [x, y, z, w, p, q, r]
                rest = list(map(float, rest))
            except Exception as ex:
                continue
            gts[path] = rest

        return gts

def print_pose(R, t):  
    print(Rotation.from_matrix(R).as_euler("XYZ", degrees=True), t)
    # print(Rotation.from_quat(pose.q).as_matrix())

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
    seed = 13
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    conf = Config()
    dataset = Dataset(verbose=False)

    sampler = Sampler()
    ref = DisplacementRefinerSolver(verbose=False)            
    solv_pipe = UP2PSolverPipeline()
    p3pwrapper = P3PBindingWrapperPipeline(
        ransac_conf = {
            # 'max_reproj_error': args.ransac_thresh
            'min_iterations': min(100, conf.max_ransac_iters),
            'max_iterations': conf.max_ransac_iters,
            'progressive_sampling': True,
            'max_prosac_iterations': conf.max_ransac_iters
            },
            
            bundle_adj_conf = {
                'loss_scale' : 1.0,
            }                                              
    )

    orientation_errors_p3p, pose_errors_p3p = [], []
    orientation_errors_np, pose_errors_np = [], []
    orientation_errors_p, pose_errors_p = [], []
    # iterator = tqdm(enumerate(dataset)) if not dataset.verbose else enumerate(dataset)
    iterator = enumerate(dataset)
    for idx, (Rgt, tgt, pts2D, pts3D, camera_dict) in iterator:
        if idx == 0: # perform sanity check
            camera = Camera.from_camera_dict(camera_dict)

            # print(pts2D[0])
            # print(camera.cam2pix((Rgt @ pts3D[0] + tgt)[None, :]))
            # exit(0)

        print("------------------------ ", idx, " ------------------------------")
        print("gt: ", Rotation.from_matrix(Rgt).as_euler("XYZ", degrees=True), tgt)

        R, t = p3pwrapper(np.stack(pts2D), np.stack(pts3D), camera_dict)
        pose_error_p3p, orient_error_p3p = dataset.compute_metric(Rgt, tgt, R, t)
        orientation_errors_p3p.append(orient_error_p3p)
        pose_errors_p3p.append(pose_error_p3p)
        print_pose(R, t)

        np.random.seed(seed)
        torch.manual_seed(seed)
        R, t = solv_pipe(np.stack(pts2D), np.stack(pts3D), camera_dict) 
        print_pose(R, t.numpy())

        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # estim_pose = ref(np.stack(pts2D), np.stack(pts3D), camera_dict)
        # estim_pose.t = - estim_pose.R.T @ estim_pose.t    
        # print_pose(estim_pose)

        pose_error_np, orient_error_np = dataset.compute_metric(Rgt, tgt, R.numpy(), t.numpy())
        # print("np[pe, oe]: ", pose_error_np, orient_error_np)
        # pose_error_p, orient_error_p = dataset.compute_metric(Rgt, tgt, estim_pose)
        # print("p[pe, oe]: ", pose_error_p, orient_error_p)

        if idx == 5:
            break

        orientation_errors_np.append(orient_error_np)
        pose_errors_np.append(pose_error_np)

        # orientation_errors_p.append(orient_error_p)
        # pose_errors_p.append(pose_error_p)
    else:
        print_stats(pose_errors_p3p, orientation_errors_p3p)
        print_stats(pose_errors_np, orientation_errors_np)
        # print_stats(pose_errors_p, orientation_errors_p)
