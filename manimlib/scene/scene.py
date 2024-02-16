from __future__ import annotations

from collections import OrderedDict
import inspect
import os
import platform
import pyperclip
import random
import time
from functools import wraps

from IPython.terminal import pt_inputhooks
from IPython.terminal.embed import InteractiveShellEmbed
from IPython.core.getipython import get_ipython

import numpy as np
from tqdm.auto import tqdm as ProgressDisplay

from manimlib.animation.animation import prepare_animation
from manimlib.animation.fading import VFadeInThenOut
from manimlib.camera.camera import Camera
from manimlib.camera.camera_frame import CameraFrame
from manimlib.config import get_module
from manimlib.constants import ARROW_SYMBOLS
from manimlib.constants import DEFAULT_WAIT_TIME
from manimlib.constants import COMMAND_MODIFIER
from manimlib.constants import SHIFT_MODIFIER
from manimlib.constants import RED
from manimlib.event_handler import EVENT_DISPATCHER
from manimlib.event_handler.event_type import EventType
from manimlib.logger import log
from manimlib.mobject.frame import FullScreenRectangle
from manimlib.mobject.mobject import _AnimationBuilder
from manimlib.mobject.mobject import Group
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.mobject import Point
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.scene.scene_file_writer import SceneFileWriter
from manimlib.utils.family_ops import extract_mobject_family_members
from manimlib.utils.family_ops import recursive_mobject_remove
from manimlib.utils.iterables import batch_by_property

from typing import TYPE_CHECKING

# 在静态类型检查时导入所需的模块和类型
if TYPE_CHECKING:
    from typing import Callable, Iterable
    from manimlib.typing import Vect3
    from PIL.Image import Image
    from manimlib.animation.animation import Animation


# 3D场景平移操作
PAN_3D_KEY = 'd'
# 切换场景帧
FRAME_SHIFT_KEY = 'f'
# 重置场景帧
RESET_FRAME_KEY = 'r'
# 退出交互模式
QUIT_KEY = 'q'


class Scene(object):
    # 随机种子，初始为0
    random_seed: int = 0
    # 平移灵敏度，初始为0.5
    pan_sensitivity: float = 0.5
    # 滚动灵敏度，初始为0.5
    scroll_sensitivity: float = 20
    # 拖动平移开关，初始打开
    drag_to_pan: bool = True
    # 最大保存状态数，初始为50
    max_num_saved_states: int = 50
    # 默认摄像头配置，初始为空字典
    default_camera_config: dict = dict()
    # 默认窗口配置，初始为空字典
    default_window_config: dict = dict()
    # 默认文件写入器配置，初始为空字典
    default_file_writer_config: dict = dict()
    # 样本数，初始为0
    samples = 0
    # 默认帧方向，欧拉角，以度为单位，初始为(0, 0)，即不进行任何旋转
    default_frame_orientation = (0, 0)

    # 创建一个Scene场景对象，该对象包含了一系列配置参数，用于控制动画的播放和展示效果。
    def __init__(
        self,
        # 窗口配置参数，默认为空字典
        window_config: dict = dict(),
        # 摄像头配置参数，默认为空字典
        camera_config: dict = dict(),
        # 文件写入器配置参数，默认为空字典
        file_writer_config: dict = dict(),
        # 是否跳过动画，默认为False
        skip_animations: bool = False,
        # 是否始终更新对象，默认为False
        always_update_mobjects: bool = False,
        # 开始动画的编号，默认为None
        start_at_animation_number: int | None = None,
        # 结束动画的编号，默认为None
        end_at_animation_number: int | None = None,
        # 是否保留进度条，默认为False
        leave_progress_bars: bool = False,
        # 是否预览，默认为True
        preview: bool = True,
        # 演示者模式，默认为False
        presenter_mode: bool = False,
        # 是否显示动画进度，默认为False
        show_animation_progress: bool = False,
        # 嵌入式异常模式，默认为空字符串
        embed_exception_mode: str = "",
        # 是否嵌入错误声音，默认为False
        embed_error_sound: bool = False,
    ):
        # 将默认窗口配置和传入的窗口配置合并
        self.window_config = {**self.default_window_config, **window_config}
        # 将默认相机配置和传入的相机配置合并
        self.camera_config = {**self.default_camera_config, **camera_config}
        # 将默认文件写入器配置和传入的文件写入器配置合并
        self.file_writer_config = {**self.default_file_writer_config, **file_writer_config}
        # 依次代入各项参数
        self.skip_animations = skip_animations
        self.always_update_mobjects = always_update_mobjects
        self.start_at_animation_number = start_at_animation_number
        self.end_at_animation_number = end_at_animation_number
        self.leave_progress_bars = leave_progress_bars
        self.preview = preview
        self.presenter_mode = presenter_mode
        self.show_animation_progress = show_animation_progress
        self.embed_exception_mode = embed_exception_mode
        self.embed_error_sound = embed_error_sound

        # 为相机配置、窗口配置和文件写入器配置设置一些默认值
        for config in self.camera_config, self.window_config:
            # 控制渲染时的采样级别，影响渲染质量
            config["samples"] = self.samples

        # 是否需要预览场景
        if self.preview:
            from manimlib.window import Window
            # 将当前场景传递给窗口对象，将窗口配置传递给窗口对象的构造函数
            self.window = Window(scene=self, **self.window_config)
            # 将窗口对象添加到相机配置参数中，以便在渲染时使用
            self.camera_config["window"] = self.window
            # 相机的帧率为30帧每秒
            self.camera_config["fps"] = 30  # 30从哪里来的？
        else:
            # 如果不需要预览场景，则没有窗口对象
            self.window = None

        # 场景类的核心状态定义，包括摄像头、帧、文件写入器等。
        # 这些状态记录了场景的基本信息和状态，用于控制场景的展示和交互。
        # camera：摄像头对象，用于控制场景的视角和视野。
        # frame：摄像头的帧，描述了场景中物体的位置和视角。
        # 根据字典中的配置信息创建一个相机对象
        self.camera: Camera = Camera(**self.camera_config)
        # 获取相机的帧对象
        self.frame: CameraFrame = self.camera.frame
        # 将相机帧的方向设置为默认帧方向
        self.frame.reorient(*self.default_frame_orientation)
        # 将当前方向设置为默认方向
        self.frame.make_orientation_default()
        # 文件写入器，用于将场景渲染为视频或图片。
        self.file_writer = SceneFileWriter(self, **self.file_writer_config)
        # 场景中的物体列表，包括摄像头帧和其他添加的物体
        self.mobjects: list[Mobject] = [self.camera.frame]
        # 渲染组，用于控制物体的渲染顺序和分组
        self.render_groups: list[Mobject] = []
        # 物体ID映射表，记录了物体和其对应的ID的关系
        self.id_to_mobject_map: dict[int, Mobject] = dict()
        # 场景播放次数计数器
        self.num_plays: int = 0
        # 场景播放的时间
        self.time: float = 0
        # 跳过动画的时间
        self.skip_time: float = 0
        # 初始跳过动画的状态
        self.original_skipping_status: bool = self.skip_animations
        # 场景状态检查点，记录场景的历史状态
        self.checkpoint_states: dict[str, list[tuple[Mobject, Mobject]]] = dict()
        # 撤销和重做栈，用于撤销和重做操作
        self.undo_stack = []
        self.redo_stack = []

        # 如果在场景配置中指定了开始动画的序号，则在渲染场景时会跳过前面的动画，直接从指定序号开始播放动画
        if self.start_at_animation_number is not None:
            self.skip_animations = True
        # 如果场景配置中指定了显示进度条，则不显示动画进度条
        if self.file_writer.has_progress_display():
            self.show_animation_progress = False

        # 一些交互相关物件
        # 鼠标的当前位置
        self.mouse_point = Point()
        # 鼠标拖拽的起始位置
        self.mouse_drag_point = Point()
        # 是否在演示者模式下暂停场景，否，则不暂停
        self.hold_on_wait = self.presenter_mode
        # 是否退出交互模式，否，则继续交互
        self.quit_interaction = False

        # # 设置随机数生成的种子，以确保在需要确定性行为的场景中能够重现相同的结果
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            
    # 获取场景类名
    def __str__(self) -> str:
        return self.__class__.__name__

    # 运行动画
    def run(self) -> None:
        # 动画开始的虚拟时间
        self.virtual_animation_start_time: float = 0
        # 动画开始的真实时间
        self.real_animation_start_time: float = time.time()
        # 开始写入动画的帧到文件中，生成视频文件
        self.file_writer.begin()
        # 准备动画的执行环境
        self.setup()
        # 监视异常
        try:
            # 构建动画
            self.construct()
            # 交互响应
            self.interact()
        # 结束场景，跳过
        except EndScene:
            pass
        # 键盘中断键
        except KeyboardInterrupt:
            # 清除中断键内容（如清除'^C'字样）
            print("", end="\r")
            # 动画以键盘中断的方式结束
            self.file_writer.ended_with_interrupt = True
        # 执行动画结束的清理工作，如释放资源或关闭文件
        self.tear_down()

    # 接口方法，必须由子类重写
    def setup(self) -> None:
        pass
        
    # 所有动画发生的地方，也是个接口方法
    def construct(self) -> None:
        pass
        
    # 结束后的清理工作
    def tear_down(self) -> None:
        # 停止跳过动画
        self.stop_skipping()
        # 完成文件写入操作
        self.file_writer.finish()
        # 如果窗口存在，则销毁窗口并释放引用，以便回收
        if self.window:
            self.window.destroy()
            self.window = None
            
    # 场景交互
    def interact(self) -> None:
        # 如果没有窗口则直接返回
        if self.window is None:
            return
        # 使用按键d、f或z与场景交互，按下command + q或esc退出
        log.info(
            "\nTips: Using the keys `d`, `f`, or `z` " +
            "you can interact with the scene. " +
            "Press `command + q` or `esc` to quit"
        )
        # 不会跳过动画
        self.skip_animations = False
        # 在窗口没有关闭的情况下不断更新帧
        while not self.is_window_closing():
            # 更新帧，参数为每秒帧数的倒数
            self.update_frame(1 / self.camera.fps)

    # 嵌入动画
    def embed(
        self,
        # 在退出时是否关闭场景，默认要关闭
        close_scene_on_exit: bool = True,
        # 是否显示动画进度，默认不显示
        show_animation_progress: bool = False,
    ) -> None:
        # 只有在有预览时才需要嵌入动画
        if not self.preview:
            # 如果不预览，跳过
            return  
        # 停止跳过动画设置
        self.stop_skipping()
        # 更新当前帧的内容
        self.update_frame()
        # 保存当前动画状态
        self.save_state()
        # 控制是否显示动画进度
        self.show_animation_progress = show_animation_progress

        # 创建一个嵌入式的IPython终端
        shell = InteractiveShellEmbed.instance()
        # 获取当前函数调用栈的前一帧
        caller_frame = inspect.currentframe().f_back
        # 将调用者的局部命名空间转换为字典，以便在嵌入的IPython终端中使用
        local_ns = dict(caller_frame.f_locals)

        # 将一些自定义的快捷方式添加到了嵌入的IPython终端中
        local_ns.update(
            # 播放动画
            play=self.play,
            # 等待一段时间
            wait=self.wait,
            # 向场景中添加物体
            add=self.add,
            # 从场景中移除物体
            remove=self.remove,
            # 清空场景
            clear=self.clear,
            # 保存场景的状态
            save_state=self.save_state,
            # 撤销操作
            undo=self.undo,
            # 重做操作
            redo=self.redo,
            # ID转Group操作
            i2g=self.i2g,
            # ID转Mobject操作
            i2m=self.i2m,
            checkpoint_paste=self.checkpoint_paste,
        )

        # Enables gui interactions during the embed
        def inputhook(context):
            while not context.input_is_ready():
                if not self.is_window_closing():
                    self.update_frame(dt=0)
            if self.is_window_closing():
                shell.ask_exit()

        pt_inputhooks.register("manim", inputhook)
        shell.enable_gui("manim")

        # This is hacky, but there's an issue with ipython which is that
        # when you define lambda's or list comprehensions during a shell session,
        # they are not aware of local variables in the surrounding scope. Because
        # That comes up a fair bit during scene construction, to get around this,
        # we (admittedly sketchily) update the global namespace to match the local
        # namespace, since this is just a shell session anyway.
        shell.events.register(
            "pre_run_cell",
            lambda: shell.user_global_ns.update(shell.user_ns)
        )

        # Operation to run after each ipython command
        def post_cell_func():
            if not self.is_window_closing():
                self.update_frame(dt=0, ignore_skipping=True)
            self.save_state()

        shell.events.register("post_run_cell", post_cell_func)

        # Flash border, and potentially play sound, on exceptions
        def custom_exc(shell, etype, evalue, tb, tb_offset=None):
            # still show the error don't just swallow it
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
            if self.embed_error_sound:
                os.system("printf '\a'")
            rect = FullScreenRectangle().set_stroke(RED, 30).set_fill(opacity=0)
            rect.fix_in_frame()
            self.play(VFadeInThenOut(rect, run_time=0.5))

        shell.set_custom_exc((Exception,), custom_exc)

        # Set desired exception mode
        shell.magic(f"xmode {self.embed_exception_mode}")

        # Launch shell
        shell(
            local_ns=local_ns,
            # Pretend like we're embeding in the caller function, not here
            stack_depth=2,
            # Specify that the present module is the caller's, not here
            module=get_module(caller_frame.f_globals["__file__"])
        )

        # End scene when exiting an embed
        if close_scene_on_exit:
            raise EndScene()

    # Only these methods should touch the camera

    def get_image(self) -> Image:
        if self.window is not None:
            self.camera.use_window_fbo(False)
            self.camera.capture(*self.render_groups)
        image = self.camera.get_image()
        if self.window is not None:
            self.camera.use_window_fbo(True)
        return image

    def show(self) -> None:
        self.update_frame(ignore_skipping=True)
        self.get_image().show()

    def update_frame(self, dt: float = 0, ignore_skipping: bool = False) -> None:
        self.increment_time(dt)
        self.update_mobjects(dt)
        if self.skip_animations and not ignore_skipping:
            return

        if self.is_window_closing():
            raise EndScene()

        if self.window:
            self.window.clear()
        self.camera.capture(*self.render_groups)

        if self.window:
            self.window.swap_buffers()
            vt = self.time - self.virtual_animation_start_time
            rt = time.time() - self.real_animation_start_time
            if rt < vt:
                self.update_frame(0)

    def emit_frame(self) -> None:
        if not self.skip_animations:
            self.file_writer.write_frame(self.camera)

    # 更新相关

    # 更新场景中的所有物体
    def update_mobjects(self, dt: float) -> None:
        # 遍历场景中的每个物体
        for mobject in self.mobjects:
            # 传递时间增量实现物体的更新
            mobject.update(dt)

    # 检查是否应该更新物体
    def should_update_mobjects(self) -> bool:
        # 如果要总是更新，则更新
        return self.always_update_mobjects or any([
            # 否则，检查场景中每个物体是否有与时间相关的更新器
            len(mob.get_family_updaters()) > 0
            # 如果有任何一个物体有时间相关的更新器，则更新，否则不更新
            for mob in self.mobjects
        ])

    # 检查场景中是否有与时间相关的更新器
    def has_time_based_updaters(self) -> bool:
        return any([
            # 遍历场景中的每个物体及其子物体，检查是否有任何一个更新器与时间相关
            sm.has_time_based_updater()
            for mob in self.mobjects()
            for sm in mob.get_family()
        ])

    # 时间相关

    # 获取场景当前时间
    def get_time(self) -> float:
        return self.time

    # 实现对时间的增量操作
    def increment_time(self, dt: float) -> None:
        self.time += dt

    # Related to internal mobject organization

    def get_top_level_mobjects(self) -> list[Mobject]:
        # Return only those which are not in the family
        # of another mobject from the scene
        mobjects = self.get_mobjects()
        families = [m.get_family() for m in mobjects]

        def is_top_level(mobject):
            num_families = sum([
                (mobject in family)
                for family in families
            ])
            return num_families == 1
        return list(filter(is_top_level, mobjects))

    def get_mobject_family_members(self) -> list[Mobject]:
        return extract_mobject_family_members(self.mobjects)

    def assemble_render_groups(self):
        """
        Rendering can be more efficient when mobjects of the
        same type are grouped together, so this function creates
        Groups of all clusters of adjacent Mobjects in the scene
        """
        batches = batch_by_property(
            self.mobjects,
            lambda m: str(type(m)) + str(m.get_uniforms())
        )

        for group in self.render_groups:
            group.clear()
        self.render_groups = [
            batch[0].get_group_class()(*batch)
            for batch, key in batches
        ]

    def affects_mobject_list(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.assemble_render_groups()
            return self
        return wrapper

    @affects_mobject_list
    def add(self, *new_mobjects: Mobject):
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.
        """
        self.remove(*new_mobjects)
        self.mobjects += new_mobjects
        self.id_to_mobject_map.update({
            id(sm): sm
            for m in new_mobjects
            for sm in m.get_family()
        })
        return self

    def add_mobjects_among(self, values: Iterable):
        """
        This is meant mostly for quick prototyping,
        e.g. to add all mobjects defined up to a point,
        call self.add_mobjects_among(locals().values())
        """
        self.add(*filter(
            lambda m: isinstance(m, Mobject),
            values
        ))
        return self

    @affects_mobject_list
    def replace(self, mobject: Mobject, *replacements: Mobject):
        if mobject in self.mobjects:
            index = self.mobjects.index(mobject)
            self.mobjects = [
                *self.mobjects[:index],
                *replacements,
                *self.mobjects[index + 1:]
            ]
        return self

    @affects_mobject_list
    def remove(self, *mobjects_to_remove: Mobject):
        """
        Removes anything in mobjects from scenes mobject list, but in the event that one
        of the items to be removed is a member of the family of an item in mobject_list,
        the other family members are added back into the list.

        For example, if the scene includes Group(m1, m2, m3), and we call scene.remove(m1),
        the desired behavior is for the scene to then include m2 and m3 (ungrouped).
        """
        to_remove = set(extract_mobject_family_members(mobjects_to_remove))
        new_mobjects, _ = recursive_mobject_remove(self.mobjects, to_remove)
        self.mobjects = new_mobjects

    def bring_to_front(self, *mobjects: Mobject):
        self.add(*mobjects)
        return self

    @affects_mobject_list
    def bring_to_back(self, *mobjects: Mobject):
        self.remove(*mobjects)
        self.mobjects = list(mobjects) + self.mobjects
        return self

    @affects_mobject_list
    def clear(self):
        self.mobjects = []
        return self

    def get_mobjects(self) -> list[Mobject]:
        return list(self.mobjects)

    def get_mobject_copies(self) -> list[Mobject]:
        return [m.copy() for m in self.mobjects]

    def point_to_mobject(
        self,
        point: np.ndarray,
        search_set: Iterable[Mobject] | None = None,
        buff: float = 0
    ) -> Mobject | None:
        """
        E.g. if clicking on the scene, this returns the top layer mobject
        under a given point
        """
        if search_set is None:
            search_set = self.mobjects
        for mobject in reversed(search_set):
            if mobject.is_point_touching(point, buff=buff):
                return mobject
        return None

    def get_group(self, *mobjects):
        if all(isinstance(m, VMobject) for m in mobjects):
            return VGroup(*mobjects)
        else:
            return Group(*mobjects)
            
    # 根据一个ID值返回所对应的物体对象
    def id_to_mobject(self, id_value):
        return self.id_to_mobject_map[id_value]

    # 根据多个ID值返回所对应的多个物体对象所组成的组对象
    def ids_to_group(self, *id_values):
        # 遍历ID列表中的每个ID值
        return self.get_group(*filter(
            # 如果ID值不为None
            lambda x: x is not None,
            # 则将对应的物体对象添加到mobjects列表中，并将mobjects列表中的物体对象作为参数传递给get_group方法，以创建组对象
            map(self.id_to_mobject, id_values)
        ))
        
    # 将传入的多个ID值作为参数，转换为一个组对象（Group）
    def i2g(self, *id_values):
        return self.ids_to_group(*id_values)
    # 将传入的ID值转换为一个物体对象（Mobject）
    def i2m(self, id_value):
        return self.id_to_mobject(id_value)

    # 跳过相关

    # 
    def update_skipping_status(self) -> None:
        if self.start_at_animation_number is not None:
            if self.num_plays == self.start_at_animation_number:
                self.skip_time = self.time
                if not self.original_skipping_status:
                    self.stop_skipping()
        if self.end_at_animation_number is not None:
            if self.num_plays >= self.end_at_animation_number:
                raise EndScene()

    def stop_skipping(self) -> None:
        self.virtual_animation_start_time = self.time
        self.skip_animations = False

    # 运行动画相关

    # 
    def get_time_progression(
        self,
        run_time: float,
        n_iterations: int | None = None,
        desc: str = "",
        override_skip_animations: bool = False
    ) -> list[float] | np.ndarray | ProgressDisplay:
        if self.skip_animations and not override_skip_animations:
            return [run_time]

        times = np.arange(0, run_time, 1 / self.camera.fps)

        self.file_writer.set_progress_display_description(sub_desc=desc)

        if self.show_animation_progress:
            return ProgressDisplay(
                times,
                total=n_iterations,
                leave=self.leave_progress_bars,
                ascii=True if platform.system() == 'Windows' else None,
                desc=desc,
                bar_format="{l_bar} {n_fmt:3}/{total_fmt:3} {rate_fmt}{postfix}",
            )
        else:
            return times

    def get_run_time(self, animations: Iterable[Animation]) -> float:
        return np.max([animation.get_run_time() for animation in animations])

    def get_animation_time_progression(
        self,
        animations: Iterable[Animation]
    ) -> list[float] | np.ndarray | ProgressDisplay:
        animations = list(animations)
        run_time = self.get_run_time(animations)
        description = f"{self.num_plays} {animations[0]}"
        if len(animations) > 1:
            description += ", etc."
        time_progression = self.get_time_progression(run_time, desc=description)
        return time_progression

    def get_wait_time_progression(
        self,
        duration: float,
        stop_condition: Callable[[], bool] | None = None
    ) -> list[float] | np.ndarray | ProgressDisplay:
        kw = {"desc": f"{self.num_plays} Waiting"}
        if stop_condition is not None:
            kw["n_iterations"] = -1  # So it doesn't show % progress
            kw["override_skip_animations"] = True
        return self.get_time_progression(duration, **kw)

    def pre_play(self):
        if self.presenter_mode and self.num_plays == 0:
            self.hold_loop()

        self.update_skipping_status()

        if not self.skip_animations:
            self.file_writer.begin_animation()

        if self.window:
            self.real_animation_start_time = time.time()
            self.virtual_animation_start_time = self.time

    def post_play(self):
        if not self.skip_animations:
            self.file_writer.end_animation()

        if self.skip_animations and self.window is not None:
            # Show some quick frames along the way
            self.update_frame(dt=0, ignore_skipping=True)

        self.num_plays += 1

    def begin_animations(self, animations: Iterable[Animation]) -> None:
        for animation in animations:
            animation.begin()
            # Anything animated that's not already in the
            # scene gets added to the scene.  Note, for
            # animated mobjects that are in the family of
            # those on screen, this can result in a restructuring
            # of the scene.mobjects list, which is usually desired.
            if animation.mobject not in self.mobjects:
                self.add(animation.mobject)

    def progress_through_animations(self, animations: Iterable[Animation]) -> None:
        last_t = 0
        for t in self.get_animation_time_progression(animations):
            dt = t - last_t
            last_t = t
            for animation in animations:
                animation.update_mobjects(dt)
                alpha = t / animation.run_time
                animation.interpolate(alpha)
            self.update_frame(dt)
            self.emit_frame()

    def finish_animations(self, animations: Iterable[Animation]) -> None:
        for animation in animations:
            animation.finish()
            animation.clean_up_from_scene(self)
        if self.skip_animations:
            self.update_mobjects(self.get_run_time(animations))
        else:
            self.update_mobjects(0)

    @affects_mobject_list
    def play(
        self,
        *proto_animations: Animation | _AnimationBuilder,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
    ) -> None:
        if len(proto_animations) == 0:
            log.warning("Called Scene.play with no animations")
            return
        animations = list(map(prepare_animation, proto_animations))
        for anim in animations:
            anim.update_rate_info(run_time, rate_func, lag_ratio)
        self.pre_play()
        self.begin_animations(animations)
        self.progress_through_animations(animations)
        self.finish_animations(animations)
        self.post_play()

    def wait(
        self,
        duration: float = DEFAULT_WAIT_TIME,
        stop_condition: Callable[[], bool] = None,
        note: str = None,
        ignore_presenter_mode: bool = False
    ):
        self.pre_play()
        self.update_mobjects(dt=0)  # Any problems with this?
        if self.presenter_mode and not self.skip_animations and not ignore_presenter_mode:
            if note:
                log.info(note)
            self.hold_loop()
        else:
            time_progression = self.get_wait_time_progression(duration, stop_condition)
            last_t = 0
            for t in time_progression:
                dt = t - last_t
                last_t = t
                self.update_frame(dt)
                self.emit_frame()
                if stop_condition is not None and stop_condition():
                    break
        self.post_play()

    def hold_loop(self):
        while self.hold_on_wait:
            self.update_frame(dt=1 / self.camera.fps)
        self.hold_on_wait = True

    def wait_until(
        self,
        stop_condition: Callable[[], bool],
        max_time: float = 60
    ):
        self.wait(max_time, stop_condition=stop_condition)

    def force_skipping(self):
        self.original_skipping_status = self.skip_animations
        self.skip_animations = True
        return self

    def revert_to_original_skipping_status(self):
        if hasattr(self, "original_skipping_status"):
            self.skip_animations = self.original_skipping_status
        return self

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        gain_to_background: float | None = None
    ):
        if self.skip_animations:
            return
        time = self.get_time() + time_offset
        self.file_writer.add_sound(sound_file, time, gain, gain_to_background)

    # Helpers for interactive development

    def get_state(self) -> SceneState:
        return SceneState(self)

    @affects_mobject_list
    def restore_state(self, scene_state: SceneState):
        scene_state.restore_scene(self)

    def save_state(self) -> None:
        if not self.preview:
            return
        state = self.get_state()
        if self.undo_stack and state.mobjects_match(self.undo_stack[-1]):
            return
        self.redo_stack = []
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_num_saved_states:
            self.undo_stack.pop(0)

    # 撤销操作
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.get_state())
            self.restore_state(self.undo_stack.pop())

    # 重做操作
    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.get_state())
            self.restore_state(self.redo_stack.pop())

    # 在交互式开发过程中运行（或重新运行）一个场景代码块
    def checkpoint_paste(
        self,
        # 默认不跳过
        skip: bool = False,
        # 默认不记录
        record: bool = False,
        # 默认加进度条
        progress_bar: bool = True
    ):
        # 获取当前的IPython shell对象
        shell = get_ipython()
        # 如果shell或window为空，则抛出异常，表示无法在非IPython shell环境或没有窗口的情况下调用此方法
        if shell is None or self.window is None:
            raise Exception(
                "Scene.checkpoint_paste cannot be called outside of an ipython shell"
            )
        # 从剪贴板中获取内容
        pasted = pyperclip.paste()
        # 获取pasted的第一行内容，并将其存储在变量line0中。lstrip()用于删除字符串开头的空格和换行符
        line0 = pasted.lstrip().split("\n")[0]
        # 如果第一行内容以#开头
        if line0.startswith("#"):
            # 如果以#开头的第一行内容不在字典中
            if line0 not in self.checkpoint_states:
                # 则将以该行内容为键的检查点状态保存到self.checkpoint_states字典中
                self.checkpoint(line0)
            else:
                # 否则将场景恢复到以该行内容为键的检查点状态
                self.revert_to_checkpoint(line0)
                
        # 保存当前的跳过动画设置
        prev_skipping = self.skip_animations
        # 跳过动画设置为skip指定的值
        self.skip_animations = skip
        # 保存当前的显示动画进度设置
        prev_progress = self.show_animation_progress
        # 显示动画进度设置为progress_bar指定的值
        self.show_animation_progress = progress_bar
        
        # 如果有记录
        if record:
            # 禁用窗口帧缓冲区
            self.camera.use_window_fbo(False)
            # 开始录制动画
            self.file_writer.begin_insert()
            
        # 在IPython shell中运行剪贴板中的代码块
        shell.run_cell(pasted)
        
        # 如果有记录
        if record:
            # 结束录制动画
            self.file_writer.end_insert()
            # 重新启用窗口帧缓冲区
            self.camera.use_window_fbo(True)
            
        # 恢复先前保存的跳过动画设置
        self.skip_animations = prev_skipping
        # 恢复先前保存的显示动画进度设置
        self.show_animation_progress = prev_progress

    # 将当前场景状态保存为一个检查点
    def checkpoint(self, key: str):
        # 以key为键，以当前场景状态为值
        self.checkpoint_states[key] = self.get_state()
        
    # 将场景恢复到之前保存的某个检查点的状态
    def revert_to_checkpoint(self, key: str):
        # 若key不存在，即没有找到对应的检查点
        if key not in self.checkpoint_states:
            # 报错
            log.error(f"No checkpoint at {key}")
            return
        # 将self.checkpoint_states字典中所有的键（即检查点的标识）转换为列表
        all_keys = list(self.checkpoint_states.keys())
        # 获取key在列表中的索引，即找到指定检查点在所有检查点中的位置
        index = all_keys.index(key)
        # 遍历all_keys列表中从index+1开始的所有元素，这些元素表示比指定检查点更新的检查点
        for later_key in all_keys[index + 1:]:
            # 移除比指定检查点更新的所有检查点，保留指定检查点及之前的检查点
            self.checkpoint_states.pop(later_key)
        # 将场景恢复到指定检查点的状态
        self.restore_state(self.checkpoint_states[key])
        
    # 清除所有的检查点
    def clear_checkpoints(self):
        # 将self.checkpoint_states字典清空，重新开始记录新的检查点
        self.checkpoint_states = dict()

    def save_mobject_to_file(self, mobject: Mobject, file_path: str | None = None) -> None:
        if file_path is None:
            file_path = self.file_writer.get_saved_mobject_path(mobject)
            if file_path is None:
                return
        mobject.save_to_file(file_path)

    def load_mobject(self, file_name):
        if os.path.exists(file_name):
            path = file_name
        else:
            directory = self.file_writer.get_saved_mobject_directory()
            path = os.path.join(directory, file_name)
        return Mobject.load(path)

    def is_window_closing(self):
        return self.window and (self.window.is_closing or self.quit_interaction)

    # Event handling

    def on_mouse_motion(
        self,
        point: Vect3,
        d_point: Vect3
    ) -> None:
        assert(self.window is not None)
        self.mouse_point.move_to(point)

        event_data = {"point": point, "d_point": d_point}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseMotionEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        frame = self.camera.frame
        # Handle perspective changes
        if self.window.is_key_pressed(ord(PAN_3D_KEY)):
            ff_d_point = frame.to_fixed_frame_point(d_point, relative=True)
            ff_d_point *= self.pan_sensitivity
            frame.increment_theta(-ff_d_point[0])
            frame.increment_phi(ff_d_point[1])
        # Handle frame movements
        elif self.window.is_key_pressed(ord(FRAME_SHIFT_KEY)):
            frame.shift(-d_point)

    def on_mouse_drag(
        self,
        point: Vect3,
        d_point: Vect3,
        buttons: int,
        modifiers: int
    ) -> None:
        self.mouse_drag_point.move_to(point)
        if self.drag_to_pan:
            self.frame.shift(-d_point)

        event_data = {"point": point, "d_point": d_point, "buttons": buttons, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseDragEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_press(
        self,
        point: Vect3,
        button: int,
        mods: int
    ) -> None:
        self.mouse_drag_point.move_to(point)
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MousePressEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_release(
        self,
        point: Vect3,
        button: int,
        mods: int
    ) -> None:
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseReleaseEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_scroll(
        self,
        point: Vect3,
        offset: Vect3,
        x_pixel_offset: float,
        y_pixel_offset: float
    ) -> None:
        event_data = {"point": point, "offset": offset}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.MouseScrollEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        rel_offset = y_pixel_offset / self.camera.get_pixel_height()
        self.frame.scale(
            1 - self.scroll_sensitivity * rel_offset,
            about_point=point
        )

    def on_key_release(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.KeyReleaseEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

    def on_key_press(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        try:
            char = chr(symbol)
        except OverflowError:
            log.warning("The value of the pressed key is too large.")
            return

        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(EventType.KeyPressEvent, **event_data)
        if propagate_event is not None and propagate_event is False:
            return

        if char == RESET_FRAME_KEY:
            self.play(self.camera.frame.animate.to_default_state())
        elif char == "z" and modifiers == COMMAND_MODIFIER:
            self.undo()
        elif char == "z" and modifiers == COMMAND_MODIFIER | SHIFT_MODIFIER:
            self.redo()
        # command + q
        elif char == QUIT_KEY and modifiers == COMMAND_MODIFIER:
            self.quit_interaction = True
        # Space or right arrow
        elif char == " " or symbol == ARROW_SYMBOLS[2]:
            self.hold_on_wait = False

    def on_resize(self, width: int, height: int) -> None:
        pass

    def on_show(self) -> None:
        pass

    def on_hide(self) -> None:
        pass

    def on_close(self) -> None:
        pass


class SceneState():
    def __init__(self, scene: Scene, ignore: list[Mobject] | None = None):
        self.time = scene.time
        self.num_plays = scene.num_plays
        self.mobjects_to_copies = OrderedDict.fromkeys(scene.mobjects)
        if ignore:
            for mob in ignore:
                self.mobjects_to_copies.pop(mob, None)

        last_m2c = scene.undo_stack[-1].mobjects_to_copies if scene.undo_stack else dict()
        for mob in self.mobjects_to_copies:
            # If it hasn't changed since the last state, just point to the
            # same copy as before
            if mob in last_m2c and last_m2c[mob].looks_identical(mob):
                self.mobjects_to_copies[mob] = last_m2c[mob]
            else:
                self.mobjects_to_copies[mob] = mob.copy()

    def __eq__(self, state: SceneState):
        return all((
            self.time == state.time,
            self.num_plays == state.num_plays,
            self.mobjects_to_copies == state.mobjects_to_copies
        ))

    def mobjects_match(self, state: SceneState):
        return self.mobjects_to_copies == state.mobjects_to_copies

    def n_changes(self, state: SceneState):
        m2c = state.mobjects_to_copies
        return sum(
            1 - int(mob in m2c and mob.looks_identical(m2c[mob]))
            for mob in self.mobjects_to_copies
        )

    def restore_scene(self, scene: Scene):
        scene.time = self.time
        scene.num_plays = self.num_plays
        scene.mobjects = [
            mob.become(mob_copy)
            for mob, mob_copy in self.mobjects_to_copies.items()
        ]

# 在某些场景结束时捕获该异常并执行一些清理工作或者切换到下一个场景
class EndScene(Exception):
    pass

# 创建一个三维场景
class ThreeDScene(Scene):
    # 渲染三维对象时用于采样的数量。增加采样数量可以提高图像质量，但也会增加计算量和渲染时间
    samples = 4
    # 默认的摄像头视角，是一个二元组，表示摄像头的旋转角度，通常用来调整场景的初始视角
    default_frame_orientation = (-30, 70)
    # 是否始终启用深度测试
    always_depth_test = True
    # 向场景中添加三维对象
    def add(self, *mobjects, set_depth_test: bool = True):
        # 遍历所有传入的物体 mobjects，检查是否需要对深度进行测试
        for mob in mobjects:
            # 如果 set_depth_test 为 True，并且物体 mob 不是固定在帧中的，并且 always_depth_test 属性为 True:
            if set_depth_test and not mob.is_fixed_in_frame() and self.always_depth_test:
                # 则调用 mob.apply_depth_test() 方法对物体应用深度测试
                mob.apply_depth_test()
        # 将所有物体添加到场景中
        super().add(*mobjects)
