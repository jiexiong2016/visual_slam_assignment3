import numpy as np
import cv2

import OpenGL.GL as gl
import pangolin

import time
from multiprocessing import Process, Queue

class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((1000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])
    
    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self.shape) , refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind+len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind+i] = x
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x

class SimpleMapViewer(object):
    def __init__(self, config=None):
        self.config = config

        self.view_point_cloud = False
        if self.config is not None:
            self.view_point_cloud = self.config.view_point_cloud 
            
        self.saved_keyframes = set()

        # data queue
        self.q_pose = Queue()
        self.q_active = Queue()
        self.q_points = Queue()
        self.q_colors = Queue()
        self.q_graph = Queue()
        self.q_camera = Queue()
        self.q_image = Queue()

        # message queue
        self.q_next = Queue()

        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def show_image(self, image):
        self.q_image.put(image)


    def check_next(self):
        require_next = False
        while not self.q_next.empty():
            require_next = self.q_next.get()
        return require_next

    def update(self, kfs, refresh=False):
        lines = []
        for kf in kfs:
            if kf.reference_keyframe is not None:
                lines.append(([*kf.position, *kf.reference_keyframe.position], 0))
            if kf.loop_keyframe is not None:
                if(abs(kf.id - kf.loop_keyframe.id) > 30):
                    lines.append(([*kf.position, *kf.loop_keyframe.position], 2))
                else:
                    lines.append(([*kf.position, *kf.loop_keyframe.position], 1))

        self.q_graph.put(lines)

        cameras = []
        for kf in kfs:
            cameras.append(kf.pose.matrix())
        self.q_camera.put(cameras)

        if self.view_point_cloud:

            points = []
            colors = []
            if refresh:
                for kf in kfs:
                    for pt in kf.mappoints:
                        points.append(pt.position)
                        colors.append(pt.color)

                if len(points) > 0:
                    self.q_points.put((points, 0))
                    self.q_colors.put((colors, 0))
            else:
                for kf in kfs[-20:]:
                    for pt in kf.mappoints:
                        points.append(pt.position)
                        colors.append(pt.color)

                if len(points) > 0:
                    self.q_points.put((points, 1))
                    self.q_colors.put((colors, 1))

    def stop(self):
        self.view_thread.join()

        qtype = type(Queue())
        for x in self.__dict__.values():
            if isinstance(x, qtype):
                while not x.empty():
                    _ = x.get()
        print('Viewer stopped')

    def view(self):
        pangolin.CreateWindowAndBind('Viewer', 1024, 768)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        panel = pangolin.CreatePanel('menu')
        panel.SetBounds(0.5, 1.0, 0.0, 175 / 1024.)

        # checkbox
        if self.view_point_cloud:
            m_show_points = pangolin.VarBool('menu.Show Points', True, True)
        m_show_keyframes = pangolin.VarBool('menu.Show KeyFrames', True, True)
        m_show_graph = pangolin.VarBool('menu.Show Graph', True, True)
        m_show_image = pangolin.VarBool('menu.Show Image', True, True)

        # button
        m_next_frame = pangolin.VarBool('menu.Next', False, False)

        if self.config is None:
            viewpoint_x = 0
            viewpoint_y = -500   # -10
            viewpoint_z = -100   # -0.1
            viewpoint_f = 2000
            camera_width = 1.
            width, height = 350, 250
        else:
            viewpoint_x = self.config.view_viewpoint_x
            viewpoint_y = self.config.view_viewpoint_y
            viewpoint_z = self.config.view_viewpoint_z
            viewpoint_f = self.config.view_viewpoint_f
            camera_width = self.config.view_camera_width
            width = self.config.view_image_width * 2
            height = self.config.view_image_height

        proj = pangolin.ProjectionMatrix(
            1024, 768, viewpoint_f, viewpoint_f, 512, 389, 0.1, 5000)
        look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)

        # Camera Render Object (for view / scene browsing)
        scam = pangolin.OpenGlRenderState(proj, look_view)

        # Add named OpenGL viewport to window and provide 3D Handler
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 175 / 1024., 1.0, -1024 / 768.)
        dcam.SetHandler(pangolin.Handler3D(scam))

        # Dilay image
        dimg = pangolin.Display('image')
        dimg.SetBounds(0, height / 768., 0.0, width / 1024., 1024 / 768.)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.ones((height, width, 3), 'uint8')

        pose = pangolin.OpenGlMatrix()   # identity matrix

        active = []
        graph = []
        loops = []
        loops_local = []
        mappoints = DynamicArray(shape=(3,))
        colors = DynamicArray(shape=(3,))
        cameras = DynamicArray(shape=(4, 4))

        while not pangolin.ShouldQuit():

            if not self.q_pose.empty():
                pose.m = self.q_pose.get()

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)

            # Show graph
            if not self.q_graph.empty():
                graph = self.q_graph.get()
                loops = np.array([_[0] for _ in graph if _[1] == 2])
                loops_local = np.array([_[0] for _ in graph if _[1] == 1])
                graph = np.array([_[0] for _ in graph if _[1] == 0])
                
            if m_show_graph.Get():
                if len(graph) > 0:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawLines(graph, 3)

                if len(loops) > 0:
                    gl.glLineWidth(2)
                    gl.glColor3f(1.0, 0.0, 1.0)
                    pangolin.DrawLines(loops, 4)
                
                if len(loops_local) > 0:
                    gl.glLineWidth(2)
                    gl.glColor3f(1.0, 1.0, 0.0)
                    pangolin.DrawLines(loops_local, 4)

            if self.view_point_cloud:
                # Show mappoints
                if not self.q_points.empty():
                    pts, code = self.q_points.get()
                    cls, code = self.q_colors.get()
                    if code == 1:     # append new points
                        mappoints.extend(pts)
                        colors.extend(cls)
                    elif code == 0:   # refresh all points
                        mappoints.clear()
                        mappoints.extend(pts)
                        colors.clear()
                        colors.extend(cls)

                if m_show_points.Get():
                    gl.glPointSize(2)
                    # easily draw millions of points
                    pangolin.DrawPoints(mappoints.array(), colors.array())

                    if not self.q_active.empty():
                        active = self.q_active.get()

                    gl.glPointSize(3)
                    gl.glBegin(gl.GL_POINTS)
                    gl.glColor3f(1.0, 0.0, 0.0)
                    for point in active:
                        gl.glVertex3f(*point)
                    gl.glEnd()

            # Show cameras
            if not self.q_camera.empty():
                cams = self.q_camera.get()
                if len(cams) > 20:
                    cameras.clear()
                cameras.extend(cams)
                
            if m_show_keyframes.Get():
                
                if cameras.array().shape[0] > 0:

                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 0.0, 1.0)
                    pangolin.DrawCameras(cameras.array()[:-1], camera_width)

                    gl.glLineWidth(1)
                    gl.glColor3f(1.0, 0.0, 0.0)
                    pangolin.DrawCameras(np.expand_dims(cameras.array()[-1], axis=0), camera_width)

            # Show image
            if not self.q_image.empty():
                image = self.q_image.get()
                if image.ndim == 3:
                    image = image[::-1, :, ::-1]
                else:
                    image = np.repeat(image[::-1, :, np.newaxis], 3, axis=2)
                image = cv2.resize(image, (width, height))

            if m_show_image.Get(): 
                texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()

            if pangolin.Pushed(m_next_frame):
                self.q_next.put(True)

            pangolin.FinishFrame()