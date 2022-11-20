'''
## Let's use the binding from pyransaclib

! Note that it is modified version available at my github (uses up2p solver instead of p3p) to perform optimization and so on

```
export CMAKE_PREFIX_PATH=/Users/hlovatskyibohdan/lab/pose_estimation/benchmark/PoseLib/_install/lib/cmake/PoseLib
cmake .. && make
```
'''

import pyransaclib

# TODO: refactor this
def run_ransac(xs, Xs, Rg, tg, rand_angle, conf: Config = conf):
    xs = [(elm[0] - conf.img_width // 2, elm[1] - conf.img_height // 2) for elm in xs]
    xs, Xs = list(itertools.chain(*xs)), list(itertools.chain(*Xs))
    xs = np.array([elm.detach().cpu().numpy().astype(np.float64) for elm in xs])
    Xs = np.array([elm.detach().cpu().numpy().astype(np.float64) for elm in Xs])
    img_name = "test_img.png"
    fl = float(conf.focal_length)
    inlier_threshold = 12
    num_lo = 5
    min_iters = int(1e5)
    max_iters = int(1e9)
    ret = pyransaclib.ransaclib_localization(img_name, fl, fl, xs, Xs,\
                                         inlier_threshold, num_lo,\
                                         min_iters, max_iters)
        
    R, t = quaternion_to_matrix(torch.tensor(ret['qvec'])), torch.tensor(ret['tvec'])
    R = R.T
    t = -R @ t
    
    return R, t, ret['num_inliers']

xs, Xs, inliers, Rg, tg, rand_angle = generate_examples(100, (15, 0))
print(Rg, tg)

R, t, inliers_count = run_ransac(xs, Xs, Rg, tg, rand_angle, conf)
print(R, t, inliers_count)

validate_sol(R,  t, Rg, tg)
