import torch
import numpy as np

from abc import ABC, abstractmethod

class Solver:
    
    @abstractmethod
    def get_sample_size(self) -> int:
        raise NotImplemented

class Up2P(Solver):
    
    def __init__(self):
        self.dtype = torch.float64
        self.min_sample_size: int = 2
        # TODO: add angles to potentially optimize for
        # or use them in a solver itself so not to prerotate the scene
    
    def get_sample_size(self) -> int:
        return self.min_sample_size
    
    # x: [2,3]
    # X: [2,3]
    def __call__(self, x, X):
        assert x.shape == (self.min_sample_size, 3)
        assert X.shape == (self.min_sample_size, 3)
        # [4, 4]
        # should be transposed as in Eigen order of indexation is a different one
        A = torch.tensor([[-x[0, 2], 0, x[0, 0], X[0, 0] * x[0, 2] - X[0, 2] * x[0, 0]],
                          [0, -x[0, 2], x[0, 1], -X[0, 1] * x[0, 2] - X[0, 2] * x[0, 1]],
                          [-x[1, 2], 0, x[1, 0], X[1, 0] * x[1, 2] - X[1, 2] * x[1, 0]],
                          [0, -x[1, 2], x[1, 1], -X[1, 1] * x[1, 2] - X[1, 2] * x[1, 1]]],
                        dtype=self.dtype)
        # [4, 2]                  
        b = torch.cat([torch.tensor([
                                -2 * X[0, 0] * x[0, 0] - 2 * X[0, 2] * x[0, 2],
                                X[0, 2] * x[0, 0] - X[0, 0] * x[0, 2],
                                -2 * X[0, 0] * x[0, 1],
                                X[0, 2] * x[0, 1] - X[0, 1] * x[0, 2]
                           ], dtype=self.dtype),
                       torch.tensor([
                               -2 * X[1, 0] * x[1, 0] - 2 * X[1, 2] * x[1, 2],
                               X[1, 2] * x[1, 0] - X[1, 0] * x[1, 2],
                               -2 * X[1, 0] * x[1, 1],
                               X[1, 2] * x[1, 1] - X[1, 1] * x[1, 2]
                           ], dtype=self.dtype)],
                      dim=-1).reshape((4, 2))
       
        assert A.shape == (4, 4) and b.shape == (4, 2)
        
        b = A.inverse() @ b
        sols = self.solve_quadratic_real(1., b[3, 0], b[3, 1])
        if sols is None:
            return []
        
        res = []
        for q in sols:
            q2 = q ** 2
            inv_norm = 1 / (1 + q2)
            cq = (1 - q2) * inv_norm
            sq = 2 * q * inv_norm
            
            R = torch.eye(3, dtype=self.dtype)
            R[0, 0] = cq
            R[2, 0] = sq
            R[0, 2] = -sq
            R[2, 2] = cq

            t = b[:3, 0] * q + b[:3, 1]
            t *= -inv_norm
            t = -R @ t
            res.append((R, t))
        
        return res
        
    def solve_quadratic_real(self, a, b, c):
        b2m4ac = b * b - 4 * a * c
        if b2m4ac < 0:
            return None
        
        sq = torch.sqrt(b2m4ac)
        roots = []
        if b > 0:
            roots.append((2*c) / (-b - sq))
        else:
            roots.append((2*c) / (-b + sq))
            
        roots.append(c / (a * roots[0]))
        
        return roots
