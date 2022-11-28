import os
import argparse
import statistics
import numpy as np

import poselib

VSP = "dataset/StMarysChurch_matches"
seq = [3, 5, 13]
SHOW_SINGLE = False

def process_file(path: str, args, camera_dict, gts):
  data = np.load(path)
  pts2D = list(data[:, :2])
  pts3D = list(data[:, 2:])
  pp = "/".join(path.split("/")[-2:])
  pp = pp.replace("_matches", "")
  pp = pp.replace(".npy", ".png")
  camera_dict = camera_dict[pp]
  gt = gts[pp]
  c, r = gt[:3], gt[3:]
  # print(c, r)

  # (points2D: List[numpy.ndarray[numpy.float64[2, 1]]],
  #       points3D: List[numpy.ndarray[numpy.float64[3, 1]]]
  #       camera_dict: dict, ransac_opt: dict = {},
  #       bundle_opt: dict = {}) -> Tuple[poselib.CameraPose, dict]
  pose, _ = poselib.estimate_absolute_pose(list(pts2D), list(pts3D),
                                                        camera_dict,
                                                        {
                                                        # 'max_reproj_error': args.ransac_thresh
                                                        'min_iterations': min(100, args.max_ransac_iters),
                                                        'max_iterations': args.max_ransac_iters,
                                                        'progressive_sampling': True,
                                                        'max_prosac_iterations': args.max_ransac_iters},
                                                        {'loss_scale' : 1.0})

  pose.t = - pose.R.T @ pose.t          

  gt_pose = poselib.CameraPose()
  gt_pose.q = r / np.linalg.norm(r)

  rot_error = np.arccos((np.trace(np.matmul(gt_pose.R.transpose(), pose.R)) - 1.0) / 2.0) * 180.0 / np.pi

  if SHOW_SINGLE:
    print(np.trace(np.matmul(gt_pose.R.transpose(), pose.R)))
    print(" Position error: " + str(np.linalg.norm(c - pose.t)) + " orientation error: " + str(rot_error))
  if np.isnan(rot_error):
      return 1000000.0, 180.0
  else:
      return np.linalg.norm(c - pose.t), rot_error

def prepare_camera_dict(path: str, args):
  with open(path) as file:
    data = file.readlines()

  camera_dict = {}
  for _, line in enumerate(data):
    # image width, image height, focal length, x of pp, y of pp, radial distortion factor 
    path, cam_type, w, h, f, x, y, rd = line.split()
    scaling_factor = args.max_side_length / max(np.float32(w), np.float32(h))
  
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

def prepare_gts(path: str):
  # ImageFile, Camera Position [X Y Z W P Q R]

  with open(path) as file:
    data = file.readlines()

  gts = {}
  for _, line in enumerate(data):
    try:
      # seq13/frame00158.png 25.317314 -0.228082 54.493720 0.374564 0.002123 0.915022 -0.149782
      path, x, y, z, w, p, q, r = line.split()
      rest = [x, y, z, w, p, q, r]
      rest = list(map(float, rest))
    except Exception as ex:
      # print(ex)
      continue
    gts[path] = rest

  return gts

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # parser.add_argument("--query_list_with_intrinsics", required=True)
  # parser.add_argument("--gt_poses", required=False)
  parser.add_argument("--ransac_thresh", type=float, required=False)
  parser.add_argument("--max_side_length", type=float, required=False, default=320)
  parser.add_argument("--max_ransac_iters", type=int, required=False,
                      default=100000, help="Maximum RANSAC iterations")
  args = parser.parse_args()

  camera_dict = prepare_camera_dict(
    "dataset/StMarysChurch_matches/st_marys_church_list_queries_with_intrinsics_simple_radial_sorted.txt",
    args
  )

  gt_dict = prepare_gts(
    "dataset/StMarysChurch_matches/dataset_test.txt"
  )

  orientation_errors, pose_errors = [], []
  for s in seq:
    p = f"{VSP}/seq{s}"
    for f in os.listdir(p):
      if f.split(".")[1] != "npy":
        continue
      pe, oe = process_file(f"{p}/{f}", args, camera_dict, gt_dict)
      pose_errors.append(pe)
      orientation_errors.append(oe)

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

