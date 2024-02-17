from __future__ import annotations

import moderngl
import numpy as np
import OpenGL.GL as gl
from PIL import Image

from manimlib.camera.camera_frame import CameraFrame
from manimlib.constants import BLACK
from manimlib.constants import DEFAULT_FPS
from manimlib.constants import DEFAULT_PIXEL_HEIGHT, DEFAULT_PIXEL_WIDTH
from manimlib.constants import FRAME_WIDTH
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.mobject import Point
from manimlib.utils.color import color_to_rgba

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from manimlib.typing import ManimColor, Vect3
    from manimlib.window import Window

# 相机类
class Camera(object):
    # 相机类的构造方法
    def __init__(
        self,
        # 可选的窗口对象，默认为None
        window: Optional[Window] = None,
        # 可选的背景图片路径，默认为None
        background_image: Optional[str] = None,
        # 帧配置参数，默认为空字典
        frame_config: dict = dict(),
        # 帧的实际尺寸可能会根据像素的纵横比进行调整
        # 像素宽度，默认为DEFAULT_PIXEL_WIDTH
        pixel_width: int = DEFAULT_PIXEL_WIDTH,
        # 像素高度，默认为DEFAULT_PIXEL_HEIGHT
        pixel_height: int = DEFAULT_PIXEL_HEIGHT,
        # 帧率，即每秒帧数，默认为DEFAULT_FPS
        fps: int = DEFAULT_FPS,
        # 背景颜色，默认为黑色
        background_color: ManimColor = BLACK,
        # 背景透明度，默认为1.0
        background_opacity: float = 1.0,
        # 当矢量化物体中的点的范数大于指定值时，这些点将被重新缩放
        # 最大允许范数，默认为FRAME_WIDTH
        max_allowable_norm: float = FRAME_WIDTH,
        # 图像模式，默认为"RGBA"
        image_mode: str = "RGBA",
        # 通道数，默认为4
        n_channels: int = 4,
        # 像素数组的类型，默认为np
        pixel_array_dtype: type = np.uint8,
        # 光源位置，默认为[-10, 10, 10]
        light_source_position: Vect3 = np.array([-10, 10, 10]),
        # 在处理矢量图形时，通常不需要使用多重采样来处理抗锯齿，因为矢量图形在处理抗锯齿时表现良好。
        # 但是，在处理3D场景时，可能需要将采样数设置为大于0的值以获得更好的图像质量
        # 采样数，默认为0
        samples: int = 0,
    ):
        # 代入各项参数
        self.window = window
        self.background_image = background_image
        self.default_pixel_shape = (pixel_width, pixel_height)
        self.fps = fps
        self.max_allowable_norm = max_allowable_norm
        self.image_mode = image_mode
        self.n_channels = n_channels
        self.pixel_array_dtype = pixel_array_dtype
        self.light_source_position = light_source_position
        self.samples = samples
        # 像素值在RGB颜色空间中的最大值
        self.rgb_max_val: float = np.iinfo(self.pixel_array_dtype).max
        # 背景颜色的RGBA值
        self.background_rgba: list[float] = list(color_to_rgba(
            background_color, background_opacity
        ))
        # 初始化uniforms
        self.uniforms = dict()
        # 初始化帧
        self.init_frame(**frame_config)
        # 初始化上下文
        self.init_context()
        # 初始化帧缓冲对象
        self.init_fbo()
        # 初始化光源
        self.init_light_source()

    # 初始化帧
    def init_frame(self, **config) -> None:
        self.frame = CameraFrame(**config)

    # 初始化上下文
    def init_context(self) -> None:
        # 
        if self.window is None:
            self.ctx: moderngl.Context = moderngl.create_standalone_context()
        else:
            self.ctx: moderngl.Context = self.window.ctx

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)

    # 初始化帧缓冲对象
    def init_fbo(self) -> None:
        # 创建一个用于写入视频/图像文件的帧缓冲对象
        self.fbo_for_files = self.get_fbo(self.samples)
        # 创建一个用于绘制帧的帧缓冲对象
        self.draw_fbo = self.get_fbo(samples=0)
        # 如果没有窗口对象
        if self.window is None:
            # 不设置窗口帧缓冲对象
            self.window_fbo = None
            # 后续绘制操作将会作用在这个帧缓冲对象上
            self.fbo = self.fbo_for_files
        else:
            # 检测当前OpenGL上下文中的帧缓冲对象，用于窗口显示
            self.window_fbo = self.ctx.detect_framebuffer()
            # 将self.fbo指向窗口帧缓冲对象，用于实时显示场景
            self.fbo = self.window_fbo
        # 激活当前帧缓冲对象，使得后续的绘制操作都会作用在这个帧缓冲对象上
        self.fbo.use()

    # 初始化光源
    def init_light_source(self) -> None:
        self.light_source = Point(self.light_source_position)

    # 使用窗口的帧缓冲对象
    def use_window_fbo(self, use: bool = True):
        assert(self.window is not None)
        if use:
            self.fbo = self.window_fbo
        else:
            self.fbo = self.fbo_for_files

    # 帧缓冲对象的有关方法

    # 获取帧缓冲对象
    def get_fbo(
        self,
        samples: int = 0
    ) -> moderngl.Framebuffer:
        return self.ctx.framebuffer(
            color_attachments=self.ctx.texture(
                self.default_pixel_shape,
                components=self.n_channels,
                samples=samples,
            ),
            depth_attachment=self.ctx.depth_renderbuffer(
                self.default_pixel_shape,
                samples=samples
            )
        )

    # 清空帧缓冲对象
    def clear(self) -> None:
        self.fbo.clear(*self.background_rgba)

    # 从一个帧缓冲对象复制到另一个帧缓冲对象
    def blit(self, src_fbo, dst_fbo):
        # 使用 Blit（位块传输）将一个帧缓冲对象（fbo）中的内容复制到另一个帧缓冲对象中
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo.glo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, dst_fbo.glo)
        gl.glBlitFramebuffer(
            *src_fbo.viewport,
            *dst_fbo.viewport,
            gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        )

    # 获取帧缓冲对象的原始数据
    def get_raw_fbo_data(self, dtype: str = 'f1') -> bytes:
        # 
        self.blit(self.fbo, self.draw_fbo)
        # 
        return self.draw_fbo.read(
            viewport=self.draw_fbo.viewport,
            components=self.n_channels,
            dtype=dtype,
        )

    # 获取图像
    def get_image(self) -> Image.Image:
        return Image.frombytes(
            # 图像模式
            'RGBA',
            # 图像的像素尺寸
            self.get_pixel_shape(),
            # 帧缓冲对象的原始数据
            self.get_raw_fbo_data(),
            # 数据来源、像素格式、扫描方向（0表示自上而下扫描）、行距（-1表示使用默认值）
            'raw', 'RGBA', 0, -1
        )

    # 获取像素数组
    def get_pixel_array(self) -> np.ndarray:
        # 获取帧缓冲对象的原始数据
        raw = self.get_raw_fbo_data(dtype='f4')
        # 将原始数据转换为NumPy数组
        flat_arr = np.frombuffer(raw, dtype='f4')
        # 将flat_arr按照帧缓冲对象的尺寸和通道数重新形状为三维数组arr
        arr = flat_arr.reshape([*reversed(self.draw_fbo.size), self.n_channels])
        # 将arr沿着垂直方向翻转，即将数组的行逆序排列
        arr = arr[::-1]
        # 将数组中的值从浮点数转换为整数
        return (self.rgb_max_val * arr).astype(self.pixel_array_dtype)

    # 需要吗?
    # 将帧缓冲对象的内容转换为纹理对象
    def get_texture(self) -> moderngl.Texture:
        texture = self.ctx.texture(
            # 获取帧缓冲对象的尺寸
            size=self.fbo.size,
            # 颜色组件数，默认为4，表示RGBA
            components=4,
            # 纹理数据
            data=self.get_raw_fbo_data(),
            # 数据类型
            dtype='f4'
        )
        return texture

    # 获取相机属性

    # 获取像素大小
    def get_pixel_size(self) -> float:
        return self.frame.get_width() / self.get_pixel_shape()[0]

    # 获取像素形状
    def get_pixel_shape(self) -> tuple[int, int]:
        return self.fbo.size

    # 获取像素宽度
    def get_pixel_width(self) -> int:
        return self.get_pixel_shape()[0]

    # 获取像素高度
    def get_pixel_height(self) -> int:
        return self.get_pixel_shape()[1]

    # 获取宽高比
    def get_aspect_ratio(self):
        pw, ph = self.get_pixel_shape()
        return pw / ph

    # 获取帧的高度
    def get_frame_height(self) -> float:
        return self.frame.get_height()

    # 获取帧的宽度
    def get_frame_width(self) -> float:
        return self.frame.get_width()

    # 获取帧的形状
    def get_frame_shape(self) -> tuple[float, float]:
        return (self.get_frame_width(), self.get_frame_height())

    # 获取帧中心
    def get_frame_center(self) -> np.ndarray:
        return self.frame.get_center()

    # 获取帧位置
    def get_location(self) -> tuple[float, float, float]:
        return self.frame.get_implied_camera_location()

    # 重置帧形状
    def resize_frame_shape(self, fixed_dimension: bool = False) -> None:
        # 将帧的形状调整为与像素的宽高比相匹配
        # fixed_dimension参数确定了在另一个维度发生变化时，帧的高度或宽度是否保持固定
        frame_height = self.get_frame_height()
        frame_width = self.get_frame_width()
        aspect_ratio = self.get_aspect_ratio()
        if not fixed_dimension:
            frame_height = frame_width / aspect_ratio
        else:
            frame_width = aspect_ratio * frame_height
        self.frame.set_height(frame_height, stretch=true)
        self.frame.set_width(frame_width, stretch=true)

    # 渲染

    # 捕获指定的Mobject对象
    def capture(self, *mobjects: Mobject) -> None:
        # 清除当前帧缓冲对象的内容
        self.clear()
        # 刷新uniform变量
        self.refresh_uniforms()
        # 将当前帧缓冲对象设置为渲染目标
        self.fbo.use()
        # 对于传入的每个Mobject对象
        for mobject in mobjects:
            # 调用其render方法进行渲染，使用当前的渲染上下文和uniform变量
            mobject.render(self.ctx, self.uniforms)
        # 如果存在窗口并且帧缓冲对象不是窗口的帧缓冲对象
        if self.window is not None and self.fbo is not self.window_fbo:
            # 则将当前帧缓冲对象的内容复制到窗口的帧缓冲对象中，以便在窗口中显示
            self.blit(self.fbo, self.window_fbo)

    # 刷新uniform变量
    def refresh_uniforms(self) -> None:
        # 获取当前的帧对象
        frame = self.frame
        # 获取当前的帧对象的视图矩阵
        view_matrix = frame.get_view_matrix()
        # 获取当前的帧对象的光源位置
        light_pos = self.light_source.get_location()
        # 获取当前的帧对象的相机位置
        cam_pos = self.frame.get_implied_camera_location()

        # 更新uniforms属性
        self.uniforms.update(
            # 视图矩阵
            view=tuple(view_matrix.T.flatten()),
            # 焦距
            focal_distance=frame.get_focal_distance() / frame.get_scale(),
            # 帧形状
            frame_scale=frame.get_scale(),
            # 像素大小
            pixel_size=self.get_pixel_size(),
            # 相机位置
            camera_position=tuple(cam_pos),
            # 光源位置
            light_position=tuple(light_pos),
        )


# 大部分只是定义，所以旧场景不会被破坏
# 3D场景的相机对象
class ThreeDCamera(Camera):
    # 3D场景相机的构造方法
    def __init__(self, 
        # 指定采样数为4
        samples: int = 4, 
        **kwargs
        ):
        # 只更改采样数，其他参数以父类为准
        super().__init__(samples=samples, **kwargs)
