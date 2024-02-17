from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation
from pyrr import Matrix44

from manimlib.constants import DEGREES, RADIANS
from manimlib.constants import FRAME_SHAPE
from manimlib.constants import DOWN, LEFT, ORIGIN, OUT, RIGHT, UP
from manimlib.mobject.mobject import Mobject
from manimlib.utils.space_ops import normalize

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import Vect3

# 相机帧类
class CameraFrame(Mobject):
    # 相机帧的初始化方法
    def __init__(
        self,
        # 相机帧的尺寸
        frame_shape: tuple[float, float] = FRAME_SHAPE,
        # 相机帧的中心点
        center_point: Vect3 = ORIGIN,
        # y方向上的视场角
        fovy: float = 45 * DEGREES,
        **kwargs,
    ):
        # 父类的初始化方法
        super().__init__(**kwargs)
        # 设置相机帧的默认旋转
        self.uniforms["orientation"] = Rotation.identity().as_quat()
        # 设置相机帧的默认视场
        self.uniforms["fovy"] = fovy

        self.default_orientation = Rotation.identity()
        self.view_matrix = np.identity(4)
        self.camera_location = OUT  # This will be updated by set_points
        # 设置相机帧的四个顶点
        self.set_points(np.array([ORIGIN, LEFT, RIGHT, DOWN, UP]))
        self.set_width(frame_shape[0], stretch=True)
        self.set_height(frame_shape[1], stretch=True)
        self.move_to(center_point)

    # 设置相机帧的旋转
    def set_orientation(self, rotation: Rotation):
        self.uniforms["orientation"][:] = rotation.as_quat()
        return self

    # 获取相机帧的旋转
    def get_orientation(self):
        return Rotation.from_quat(self.uniforms["orientation"])

    # 设置相机帧的默认旋转
    def make_orientation_default(self):
        self.default_orientation = self.get_orientation()
        return self

    # 重置相机帧的默认状态
    def to_default_state(self):
        self.set_shape(*FRAME_SHAPE)
        self.center()
        self.set_orientation(self.default_orientation)
        return self

    # 获取相机帧的欧拉角
    def get_euler_angles(self) -> np.ndarray:
        orientation = self.get_orientation()
        if all(orientation.as_quat() == [0, 0, 0, 1]):
            return np.zeros(3)
        return orientation.as_euler("zxz")[::-1]

    # 获取相机帧的旋转角度(theta) 
    def get_theta(self):
        return self.get_euler_angles()[0]

    # 获取相机帧的旋转角度(phi)
    def get_phi(self):
        return self.get_euler_angles()[1]

    # 获取相机帧的旋转角度(gamma)
    def get_gamma(self):
        return self.get_euler_angles()[2]

    # 获取相机帧的缩放比例
    def get_scale(self):
        return self.get_height() / FRAME_SHAPE[1]

    # 获取相机帧的逆旋转矩阵
    def get_inverse_camera_rotation_matrix(self):
        return self.get_orientation().as_matrix().T

    # 获取相机帧的视图矩阵
    def get_view_matrix(self, refresh=False):
        # 返回 4x4 的仿射变换映射点，进入相机的内部坐标系
        if self._data_has_changed:
            shift = np.identity(4)
            rotation = np.identity(4)
            scale_mat = np.identity(4)

            shift[:3, 3] = -self.get_center()
            rotation[:3, :3] = self.get_inverse_camera_rotation_matrix()
            scale = self.get_scale()
            if scale > 0:
                scale_mat[:3, :3] /= self.get_scale()

            self.view_matrix = np.dot(scale_mat, np.dot(rotation, shift))

        return self.view_matrix

    # 获取相机帧的逆视图矩阵
    def get_inv_view_matrix(self):
        return np.linalg.inv(self.get_view_matrix())

    # 数据影响装饰器
    @Mobject.affects_data
    # 插值对象的状态
    def interpolate(self, *args, **kwargs):
        super().interpolate(*args, **kwargs)

    # 数据影响装饰器
    @Mobject.affects_data
    # 围绕给定的轴向旋转对象
    def rotate(self, angle: float, axis: np.ndarray = OUT, **kwargs):
        rot = Rotation.from_rotvec(angle * normalize(axis))
        self.set_orientation(rot * self.get_orientation())
        return self

    # 设置相机帧的欧拉角
    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
        units: float = RADIANS
    ):
        eulers = self.get_euler_angles()  # theta, phi, gamma
        for i, var in enumerate([theta, phi, gamma]):
            if var is not None:
                eulers[i] = var * units
        if all(eulers == 0):
            rot = Rotation.identity()
        else:
            rot = Rotation.from_euler("zxz", eulers[::-1])
        self.set_orientation(rot)
        return self

    #
    def reorient(
        self,
        theta_degrees: float | None = None,
        phi_degrees: float | None = None,
        gamma_degrees: float | None = None,
    ):
        """
        Shortcut for set_euler_angles, defaulting to taking
        in angles in degrees
        """
        self.set_euler_angles(theta_degrees, phi_degrees, gamma_degrees, units=DEGREES)
        return self

    # 设置相机帧的旋转角度（theta）
    def set_theta(self, theta: float):
        return self.set_euler_angles(theta=theta)

    # 设置相机帧的旋转角度（phi）
    def set_phi(self, phi: float):
        return self.set_euler_angles(phi=phi)

    # 设置相机帧的旋转角度（gamma）
    def set_gamma(self, gamma: float):
        return self.set_euler_angles(gamma=gamma)

    # 获取相机帧的旋转角度（theta）
    def increment_theta(self, dtheta: float):
        self.rotate(dtheta, OUT)
        return self

    # 获取相机帧的旋转角度（phi）
    def increment_phi(self, dphi: float):
        self.rotate(dphi, self.get_inverse_camera_rotation_matrix()[0])
        return self

    # 获取相机帧的旋转角度（gamma）
    def increment_gamma(self, dgamma: float):
        self.rotate(dgamma, self.get_inverse_camera_rotation_matrix()[2])
        return self

    # 数据影响装饰器
    @Mobject.affects_data
    # 设置相机帧的焦距
    def set_focal_distance(self, focal_distance: float):
        # 根据给定的焦距计算视野角度，并更新uniforms字典中的"fovy"键值对应的值
        self.uniforms["fovy"] = 2 * math.atan(0.5 * self.get_height() / focal_distance)
        return self

    # 数据影响装饰器
    @Mobject.affects_data
    # 设置相机帧的视野
    def set_field_of_view(self, field_of_view: float):
        # 直接将给定的视野角度（field_of_view）设置为"fovy"对应的值，并更新uniforms字典中的"fovy"键值对应的值
        self.uniforms["fovy"] = field_of_view
        return self

    # 获取相机帧的形状（shape）
    def get_shape(self):
        return (self.get_width(), self.get_height())

    # 获取相机帧的宽高比（aspect ratio）
    def get_aspect_ratio(self):
        width, height = self.get_shape()
        return width / height

    # 获取相机帧的中心（center）
    def get_center(self) -> np.ndarray:
        # Assumes first point is at the center
        return self.get_points()[0]

    # 获取相机帧的宽度（width）
    def get_width(self) -> float:
        points = self.get_points()
        return points[2, 0] - points[1, 0]

    # 获取相机帧的高度（height）
    def get_height(self) -> float:
        points = self.get_points()
        return points[4, 1] - points[3, 1]

    # 获取相机帧的焦距（focal distance）
    def get_focal_distance(self) -> float:
        return 0.5 * self.get_height() / math.tan(0.5 * self.uniforms["fovy"])

    # 获取相机帧的视场角（field of view）
    def get_field_of_view(self) -> float:
        return self.uniforms["fovy"]
    
    # 这段代码用于获取相机在场景中的位置。具体步骤如下：
    # 检查数据是否已更改，如果是，则需要重新计算相机位置。
    # 获取相机的旋转矩阵中第三列，该列表示相机的朝向。
    # 获取相机到场景中心的距离（焦距）。
    # 计算相机的位置为场景中心加上相机朝向乘以焦距的结果。
    def get_implied_camera_location(self) -> np.ndarray:
        if self._data_has_changed:
            to_camera = self.get_inverse_camera_rotation_matrix()[2]
            dist = self.get_focal_distance()
            self.camera_location = self.get_center() + dist * to_camera
        return self.camera_location

    # 将一个点（point）从相机视图坐标系转换到固定帧坐标系（fixed-frame coordinate system）中
    def to_fixed_frame_point(self, point: Vect3, relative: bool = False):
        # 获取相机的视图矩阵
        view = self.get_view_matrix()
        # 构造一个四维的点，如果relative为真，则最后一维设为0，否则设为1
        point4d = [*point, 0 if relative else 1]
        # 将四维点与视图矩阵的转置相乘，得到一个新的四维点，并取其前三个元素，即为转换后的点在固定帧坐标系中的位置
        return np.dot(point4d, view.T)[:3]

    # 将一个点从固定帧坐标系转换到相机视图坐标系中
    def from_fixed_frame_point(self, point: Vect3, relative: bool = False):
        # 获取逆视图矩阵
        inv_view = self.get_inv_view_matrix() 
        # 构造一个四维的点，如果relative为真，则最后一维设为0，否则设为1
        point4d = [*point, 0 if relative else 1]
        # 将四维点与逆视图矩阵的转置相乘，得到一个新的四维点，并取其前三个元素，即为转换后的点在相机视图坐标系中的位置
        return np.dot(point4d, inv_view.T)[:3]
