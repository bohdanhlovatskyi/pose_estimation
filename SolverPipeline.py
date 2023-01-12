from abc import ABC, abstractmethod

import poselib

from solver import Up2P

class SolverPipeline(ABC):
    
    def __init__(self, ransac_conf: dict, bundle_adj_conf: dict):
        self.ransac_conf = ransac_conf
        self.bundle_adj_conf = bundle_adj_conf
    
    @abstractmethod
    def __call__(self, pts2D, pts3D, camera_dict):
        # (points2D: List[numpy.ndarray[numpy.float64[2, 1]]],
        #       points3D: List[numpy.ndarray[numpy.float64[3, 1]]]
        #       camera_dict: dict, ransac_opt: dict = {},
        #       bundle_opt: dict = {}) -> poselib.CameraPose
        
        raise NotImplemented
    
class P3PBindingWrapperPipeline(SolverPipeline):
    
    def __call__(self, pts2D, pts3D, camera_dict):
        pose, _ = poselib.estimate_absolute_pose(
            pts2D, pts3D, camera_dict, self.ransac_conf, self.bundle_adj_conf
        )
        
        return pose.R, pose.t
    
class UP2PSolverPipeline(SolverPipeline):
    
    def __call__(self, pts2D, pts3D, camera_dict):
        pass