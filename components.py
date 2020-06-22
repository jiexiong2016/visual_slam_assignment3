import numpy as np
import cv2
import g2o

from threading import Lock, Thread

def depth_to_3d(depth, coords, cam):
    coords = np.array(coords, dtype=int)
    ix = coords[:, 0]
    iy = coords[:, 1]
    depth = depth[iy, ix]

    zs = depth / cam.scale
    xs = (ix - cam.cx) * zs / cam.fx
    ys = (iy - cam.cy) * zs / cam.fy
    return np.column_stack([xs, ys, zs])

class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height, 
            scale, depth_near, depth_far):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

        self.intrinsic = np.array([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1]])

        self.depth_near = depth_near
        self.depth_far = depth_far

        self.width = width
        self.height = height

class Frame(object):
    def __init__(self, idx, pose, feature, cam, timestamp=None, 
            pose_covariance=np.identity(6)):
        self.idx = idx
        self.pose = pose    # g2o.Isometry3d
        self.feature = feature
        self.cam = cam
        self.timestamp = timestamp
        self.image = feature.image
        
        self.orientation = pose.orientation()
        self.position = pose.position()
        self.pose_covariance = pose_covariance

        self.transform_matrix = pose.inverse().matrix()[:3] # shape: (3, 4)
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))  # from world frame to image
        
    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose
        self.orientation = self.pose.orientation()
        self.position = self.pose.position()

        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))

    def get_color(self, pt):
        return self.feature.get_color(pt)

class RGBDFrame(Frame):
    def __init__(self, idx, pose, feature, depth, cam, timestamp=None, 
            pose_covariance=np.identity(6)):

        super().__init__(idx, pose, feature, cam, timestamp, pose_covariance)
        self.rgb  = Frame(idx, pose, feature, cam, timestamp, pose_covariance)
        self.depth = depth
        self.timestamp = timestamp

        self.mappoints = []

    def update_pose(self, pose):
        super().update_pose(pose)
        self.rgb.update_pose(pose)

    def to_keyframe(self):
        return KeyFrame(
            self.idx, self.pose, 
            self.feature, self.depth, 
            self.cam, self.timestamp, self.pose_covariance)

    def cloudify(self):
        px = np.array([kp.pt for kp in self.rgb.feature.keypoints])
        if len(px) == 0:
            return [], []

        pts = depth_to_3d(self.depth, px, self.cam)
        Rt = self.pose.matrix()[:3]
        R = Rt[:, :3]
        t = Rt[:, 3:]
        points = (R.dot(pts.T) + t).T   # world frame

        mappoints = []
        for i, point in enumerate(points):
            if not (self.cam.depth_near <= pts[i][2] <= self.cam.depth_far):
                continue

            color = self.rgb.get_color(px[i])
            mappoint = MapPoint(point, color)
            self.mappoints.append(mappoint)
        # return mappoints

class KeyFrame(RGBDFrame):
    _id = 0
    _id_lock = Lock()

    def __init__(self, *args, **kwargs):
        RGBDFrame.__init__(self, *args, **kwargs)

        with KeyFrame._id_lock:
            self.id = KeyFrame._id
            KeyFrame._id += 1

        self.reference_keyframe = None
        self.reference_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, KeyFrame) and 
            self.id == rhs.id)

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __le__(self, rhs):
        return self.id <= rhs.id

    def update_reference(self, reference=None):
        if reference is not None:
            self.reference_keyframe = reference
        self.reference_constraint = (
            self.reference_keyframe.pose.inverse() * self.pose)

    def set_loop(self, keyframe, constraint):
        self.loop_keyframe = keyframe
        self.loop_constraint = constraint

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed



class MapPoint(object):
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, color=np.zeros(3)):
        super().__init__()

        with MapPoint._id_lock:
            self.id = MapPoint._id
            MapPoint._id += 1

        self.position = position
        self.color = color

    def update_position(self, position):
        self.position = position

    def set_color(self, color):
        self.color = color