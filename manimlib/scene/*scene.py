"""Basic canvas for animations."""

from __future__ import annotations

from manim.utils.parameter_parsing import flatten_iterable_parameters

__all__ = ["Scene"]

import copy
import datetime
import inspect
import platform
import random
import threading
import time
import types
from queue import Queue

import srt

from manim.scene.section import DefaultSectionType

try:
    import dearpygui.dearpygui as dpg

    dearpygui_imported = True
except ImportError:
    dearpygui_imported = False
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_mobject import OpenGLPoint

from .. import config, logger
from ..animation.animation import Animation, Wait, prepare_animation
from ..camera.camera import Camera
from ..constants import *
from ..gui.gui import configure_pygui
from ..renderer.cairo_renderer import CairoRenderer
from ..renderer.opengl_renderer import OpenGLRenderer
from ..renderer.shader import Object3D
from ..utils import opengl, space_ops
from ..utils.exceptions import EndSceneEarlyException, RerunSceneException
from ..utils.family import extract_mobject_family_members
from ..utils.family_ops import restructure_list_to_exclude_certain_family_members
from ..utils.file_ops import open_media_file
from ..utils.iterables import list_difference_update, list_update

if TYPE_CHECKING:
    from typing import Callable, Iterable


# （修改文件后）重新运行场景的处理器类
class RerunSceneHandler(FileSystemEventHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
    # 当文件或目录被修改时调用
    def on_modified(self, event):
        # 将重新运行场景的指令放入队列中
        self.queue.put(("rerun_file", [], {}))
        # 这里的事件包括DirModifiedEvent（目录修改事件）或FileModifiedEvent（文件修改事件）

# Scene是动画的画布，它提供了管理图形对象和动画的工具
# 通常，一个manim脚本包含一个继承自Scene的类
# 用户会在这个类中重写Scene.construct方法
# 通过调用Scene.add可以将图形对象显示在屏幕上
# 通过调用Scene.remove可以将图形对象从屏幕上移除
# 所有当前在屏幕上的图形对象都存储在Scene.mobjects属性中
# 通过调用Scene.play可以播放动画
# Scene内部通过调用Scene.render来进行渲染。这个过程中会依次调用Scene.setup、Scene.construct和Scene.tear_down
# 不建议在用户自定义的Scene中重写__init__方法。如果需要在Scene渲染之前运行一些代码，可以使用Scene.setup方法
# 示例代码：
#         class MyScene(Scene):
#             def construct(self):
#                 self.play(Write(Text("Hello World!")))
class Scene:
    def __init__(
        self,
        # 渲染器
        renderer=None,
        # 相机
        camera_class=Camera,
        # 是否总是更新mobjects，默认为False
        always_update_mobjects=False,
        # 随机数种子
        random_seed=None,
        # 是否跳过动画，默认为False
        skip_animations=False,
    ):
        self.camera_class = camera_class
        self.always_update_mobjects = always_update_mobjects
        self.random_seed = random_seed
        self.skip_animations = skip_animations
        # 动画
        self.animations = None
        # 停止条件
        self.stop_condition = None
        # 动态mobject列表
        self.moving_mobjects = []
        # 静态mobject列表
        self.static_mobjects = []
        # 时间进度
        self.time_progression = None
        # 动画持续时间
        self.duration = None
        # 上次的时间进度
        self.last_t = None
        # 重新运行场景的队列
        self.queue = Queue()
        # 是否跳过动画预览，默认为False
        self.skip_animation_preview = False
        # 网格
        self.meshes = []
        # 相机目标位置
        self.camera_target = ORIGIN
        # 控件列表
        self.widgets = []
        # 是否导入dearpygui
        self.dearpygui_imported = dearpygui_imported
        # 更新器列表
        self.updaters = []
        # 点光源列表
        self.point_lights = []
        # 环境光
        self.ambient_light = None
        # 不同的键映射到不同的函数上的字典
        self.key_to_function_map = {}
        # 鼠标按下回调列表
        self.mouse_press_callbacks = []
        # 交互模式
        self.interactive_mode = False
        # 检查配置中的渲染器是否为OpenGL类型
        if config.renderer == RendererType.OPENGL:
            # 交互相关的物件，如果是，它将创建两个点mouse_point和mouse_drag_point
            self.mouse_point = OpenGLPoint()
            self.mouse_drag_point = OpenGLPoint()
            # 如果未提供渲染器，则会创建一个OpenGL渲染器
            if renderer is None:
                renderer = OpenGLRenderer()
        # 检查是否已经提供了渲染器。如果未提供，则创建一个CairoRenderer实例，并根据情况传递相机类和跳过动画的设置
        if renderer is None:
            # 然后，它将初始化创建的渲染器，并将其存储在场景的renderer属性中
            self.renderer = CairoRenderer(
                camera_class=self.camera_class,
                skip_animations=self.skip_animations,
            )
        # 如果已经提供了渲染器，则直接使用提供的渲染器，并初始化它
        else:
            self.renderer = renderer
        self.renderer.init_scene(self)

        self.mobjects = []
        # TODO, remove need for foreground mobjects
        self.foreground_mobjects = []
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    @property
    def camera(self):
        return self.renderer.camera

    def __deepcopy__(self, clone_from_id):
        cls = self.__class__
        result = cls.__new__(cls)
        clone_from_id[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["renderer", "time_progression"]:
                continue
            if k == "camera_class":
                setattr(result, k, v)
            setattr(result, k, copy.deepcopy(v, clone_from_id))
        result.mobject_updater_lists = []

        # Update updaters
        for mobject in self.mobjects:
            cloned_updaters = []
            for updater in mobject.updaters:
                # Make the cloned updater use the cloned Mobjects as free variables
                # rather than the original ones. Analyzing function bytecode with the
                # dis module will help in understanding this.
                # https://docs.python.org/3/library/dis.html
                # TODO: Do the same for function calls recursively.
                free_variable_map = inspect.getclosurevars(updater).nonlocals
                cloned_co_freevars = []
                cloned_closure = []
                for free_variable_name in updater.__code__.co_freevars:
                    free_variable_value = free_variable_map[free_variable_name]

                    # If the referenced variable has not been cloned, raise.
                    if id(free_variable_value) not in clone_from_id:
                        raise Exception(
                            f"{free_variable_name} is referenced from an updater "
                            "but is not an attribute of the Scene, which isn't "
                            "allowed.",
                        )

                    # Add the cloned object's name to the free variable list.
                    cloned_co_freevars.append(free_variable_name)

                    # Add a cell containing the cloned object's reference to the
                    # closure list.
                    cloned_closure.append(
                        types.CellType(clone_from_id[id(free_variable_value)]),
                    )

                cloned_updater = types.FunctionType(
                    updater.__code__.replace(co_freevars=tuple(cloned_co_freevars)),
                    updater.__globals__,
                    updater.__name__,
                    updater.__defaults__,
                    tuple(cloned_closure),
                )
                cloned_updaters.append(cloned_updater)
            mobject_clone = clone_from_id[id(mobject)]
            mobject_clone.updaters = cloned_updaters
            if len(cloned_updaters) > 0:
                result.mobject_updater_lists.append((mobject_clone, cloned_updaters))
        return result

    def render(self, preview: bool = False):
        """
        Renders this Scene.

        Parameters
        ---------
        preview
            If true, opens scene in a file viewer.
        """
        self.setup()
        try:
            self.construct()
        except EndSceneEarlyException:
            pass
        except RerunSceneException as e:
            self.remove(*self.mobjects)
            self.renderer.clear_screen()
            self.renderer.num_plays = 0
            return True
        self.tear_down()
        # We have to reset these settings in case of multiple renders.
        self.renderer.scene_finished(self)

        # Show info only if animations are rendered or to get image
        if (
            self.renderer.num_plays
            or config["format"] == "png"
            or config["save_last_frame"]
        ):
            logger.info(
                f"Rendered {str(self)}\nPlayed {self.renderer.num_plays} animations",
            )

        # If preview open up the render after rendering.
        if preview:
            config["preview"] = True

        if config["preview"] or config["show_in_file_browser"]:
            open_media_file(self.renderer.file_writer)

    def setup(self):
        """
        This is meant to be implemented by any scenes which
        are commonly subclassed, and have some common setup
        involved before the construct method is called.
        """
        pass

    def tear_down(self):
        """
        This is meant to be implemented by any scenes which
        are commonly subclassed, and have some common method
        to be invoked before the scene ends.
        """
        pass

    def construct(self):
        """Add content to the Scene.

        From within :meth:`Scene.construct`, display mobjects on screen by calling
        :meth:`Scene.add` and remove them from screen by calling :meth:`Scene.remove`.
        All mobjects currently on screen are kept in :attr:`Scene.mobjects`.  Play
        animations by calling :meth:`Scene.play`.

        Notes
        -----
        Initialization code should go in :meth:`Scene.setup`.  Termination code should
        go in :meth:`Scene.tear_down`.

        Examples
        --------
        A typical manim script includes a class derived from :class:`Scene` with an
        overridden :meth:`Scene.contruct` method:

        .. code-block:: python

            class MyScene(Scene):
                def construct(self):
                    self.play(Write(Text("Hello World!")))

        See Also
        --------
        :meth:`Scene.setup`
        :meth:`Scene.render`
        :meth:`Scene.tear_down`

        """
        pass  # To be implemented in subclasses

    def next_section(
        self,
        name: str = "unnamed",
        type: str = DefaultSectionType.NORMAL,
        skip_animations: bool = False,
    ) -> None:
        """Create separation here; the last section gets finished and a new one gets created.
        ``skip_animations`` skips the rendering of all animations in this section.
        Refer to :doc:`the documentation</tutorials/output_and_config>` on how to use sections.
        """
        self.renderer.file_writer.next_section(name, type, skip_animations)

    def __str__(self):
        return self.__class__.__name__

    def get_attrs(self, *keys: str):
        """
        Gets attributes of a scene given the attribute's identifier/name.

        Parameters
        ----------
        *keys
            Name(s) of the argument(s) to return the attribute of.

        Returns
        -------
        list
            List of attributes of the passed identifiers.
        """
        return [getattr(self, key) for key in keys]

    def update_mobjects(self, dt: float):
        """
        Begins updating all mobjects in the Scene.

        Parameters
        ----------
        dt
            Change in time between updates. Defaults (mostly) to 1/frames_per_second
        """
        for mobject in self.mobjects:
            mobject.update(dt)

    def update_meshes(self, dt):
        for obj in self.meshes:
            for mesh in obj.get_family():
                mesh.update(dt)

    def update_self(self, dt: float):
        """Run all scene updater functions.

        Among all types of update functions (mobject updaters, mesh updaters,
        scene updaters), scene update functions are called last.

        Parameters
        ----------
        dt
            Scene time since last update.

        See Also
        --------
        :meth:`.Scene.add_updater`
        :meth:`.Scene.remove_updater`
        """
        for func in self.updaters:
            func(dt)

    def should_update_mobjects(self) -> bool:
        """
        Returns True if the mobjects of this scene should be updated.

        In particular, this checks whether

        - the :attr:`always_update_mobjects` attribute of :class:`.Scene`
          is set to ``True``,
        - the :class:`.Scene` itself has time-based updaters attached,
        - any mobject in this :class:`.Scene` has time-based updaters attached.

        This is only called when a single Wait animation is played.
        """
        wait_animation = self.animations[0]
        if wait_animation.is_static_wait is None:
            should_update = (
                self.always_update_mobjects
                or self.updaters
                or wait_animation.stop_condition is not None
                or any(
                    mob.has_time_based_updater()
                    for mob in self.get_mobject_family_members()
                )
            )
            wait_animation.is_static_wait = not should_update
        return not wait_animation.is_static_wait

    def get_top_level_mobjects(self):
        """
        Returns all mobjects which are not submobjects.

        Returns
        -------
        list
            List of top level mobjects.
        """
        # Return only those which are not in the family
        # of another mobject from the scene
        families = [m.get_family() for m in self.mobjects]

        def is_top_level(mobject):
            num_families = sum((mobject in family) for family in families)
            return num_families == 1

        return list(filter(is_top_level, self.mobjects))

    def get_mobject_family_members(self):
        """
        Returns list of family-members of all mobjects in scene.
        If a Circle() and a VGroup(Rectangle(),Triangle()) were added,
        it returns not only the Circle(), Rectangle() and Triangle(), but
        also the VGroup() object.

        Returns
        -------
        list
            List of mobject family members.
        """
        if config.renderer == RendererType.OPENGL:
            family_members = []
            for mob in self.mobjects:
                family_members.extend(mob.get_family())
            return family_members
        elif config.renderer == RendererType.CAIRO:
            return extract_mobject_family_members(
                self.mobjects,
                use_z_index=self.renderer.camera.use_z_index,
            )

    def add(self, *mobjects: Mobject):
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.

        Parameters
        ---------
        *mobjects
            Mobjects to add.

        Returns
        -------
        Scene
            The same scene after adding the Mobjects in.

        """
        if config.renderer == RendererType.OPENGL:
            new_mobjects = []
            new_meshes = []
            for mobject_or_mesh in mobjects:
                if isinstance(mobject_or_mesh, Object3D):
                    new_meshes.append(mobject_or_mesh)
                else:
                    new_mobjects.append(mobject_or_mesh)
            self.remove(*new_mobjects)
            self.mobjects += new_mobjects
            self.remove(*new_meshes)
            self.meshes += new_meshes
        elif config.renderer == RendererType.CAIRO:
            mobjects = [*mobjects, *self.foreground_mobjects]
            self.restructure_mobjects(to_remove=mobjects)
            self.mobjects += mobjects
            if self.moving_mobjects:
                self.restructure_mobjects(
                    to_remove=mobjects,
                    mobject_list_name="moving_mobjects",
                )
                self.moving_mobjects += mobjects
        return self

    def add_mobjects_from_animations(self, animations):
        curr_mobjects = self.get_mobject_family_members()
        for animation in animations:
            if animation.is_introducer():
                continue
            # Anything animated that's not already in the
            # scene gets added to the scene
            mob = animation.mobject
            if mob is not None and mob not in curr_mobjects:
                self.add(mob)
                curr_mobjects += mob.get_family()

    def remove(self, *mobjects: Mobject):
        """
        Removes mobjects in the passed list of mobjects
        from the scene and the foreground, by removing them
        from "mobjects" and "foreground_mobjects"

        Parameters
        ----------
        *mobjects
            The mobjects to remove.
        """
        if config.renderer == RendererType.OPENGL:
            mobjects_to_remove = []
            meshes_to_remove = set()
            for mobject_or_mesh in mobjects:
                if isinstance(mobject_or_mesh, Object3D):
                    meshes_to_remove.add(mobject_or_mesh)
                else:
                    mobjects_to_remove.append(mobject_or_mesh)
            self.mobjects = restructure_list_to_exclude_certain_family_members(
                self.mobjects,
                mobjects_to_remove,
            )
            self.meshes = list(
                filter(lambda mesh: mesh not in set(meshes_to_remove), self.meshes),
            )
            return self
        elif config.renderer == RendererType.CAIRO:
            for list_name in "mobjects", "foreground_mobjects":
                self.restructure_mobjects(mobjects, list_name, False)
            return self

    def replace(self, old_mobject: Mobject, new_mobject: Mobject) -> None:
        """Replace one mobject in the scene with another, preserving draw order.

        If ``old_mobject`` is a submobject of some other Mobject (e.g. a
        :class:`.Group`), the new_mobject will replace it inside the group,
        without otherwise changing the parent mobject.

        Parameters
        ----------
        old_mobject
            The mobject to be replaced. Must be present in the scene.
        new_mobject
            A mobject which must not already be in the scene.

        """
        if old_mobject is None or new_mobject is None:
            raise ValueError("Specified mobjects cannot be None")

        def replace_in_list(
            mobj_list: list[Mobject], old_m: Mobject, new_m: Mobject
        ) -> bool:
            # We use breadth-first search because some Mobjects get very deep and
            # we expect top-level elements to be the most common targets for replace.
            for i in range(0, len(mobj_list)):
                # Is this the old mobject?
                if mobj_list[i] == old_m:
                    # If so, write the new object to the same spot and stop looking.
                    mobj_list[i] = new_m
                    return True
            # Now check all the children of all these mobs.
            for mob in mobj_list:  # noqa: SIM110
                if replace_in_list(mob.submobjects, old_m, new_m):
                    # If we found it in a submobject, stop looking.
                    return True
            # If we did not find the mobject in the mobject list or any submobjects,
            # (or the list was empty), indicate we did not make the replacement.
            return False

        # Make use of short-circuiting conditionals to check mobjects and then
        # foreground_mobjects
        replaced = replace_in_list(
            self.mobjects, old_mobject, new_mobject
        ) or replace_in_list(self.foreground_mobjects, old_mobject, new_mobject)

        if not replaced:
            raise ValueError(f"Could not find {old_mobject} in scene")

    def add_updater(self, func: Callable[[float], None]) -> None:
        """Add an update function to the scene.

        The scene updater functions are run every frame,
        and they are the last type of updaters to run.

        .. WARNING::

            When using the Cairo renderer, scene updaters that
            modify mobjects are not detected in the same way
            that mobject updaters are. To be more concrete,
            a mobject only modified via a scene updater will
            not necessarily be added to the list of *moving
            mobjects* and thus might not be updated every frame.

            TL;DR: Use mobject updaters to update mobjects.

        Parameters
        ----------
        func
            The updater function. It takes a float, which is the
            time difference since the last update (usually equal
            to the frame rate).

        See also
        --------
        :meth:`.Scene.remove_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters.append(func)

    def remove_updater(self, func: Callable[[float], None]) -> None:
        """Remove an update function from the scene.

        Parameters
        ----------
        func
            The updater function to be removed.

        See also
        --------
        :meth:`.Scene.add_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters = [f for f in self.updaters if f is not func]

    def restructure_mobjects(
        self,
        to_remove: Mobject,
        mobject_list_name: str = "mobjects",
        extract_families: bool = True,
    ):
        """
        tl:wr
            If your scene has a Group(), and you removed a mobject from the Group,
            this dissolves the group and puts the rest of the mobjects directly
            in self.mobjects or self.foreground_mobjects.

        In cases where the scene contains a group, e.g. Group(m1, m2, m3), but one
        of its submobjects is removed, e.g. scene.remove(m1), the list of mobjects
        will be edited to contain other submobjects, but not m1, e.g. it will now
        insert m2 and m3 to where the group once was.

        Parameters
        ----------
        to_remove
            The Mobject to remove.

        mobject_list_name
            The list of mobjects ("mobjects", "foreground_mobjects" etc) to remove from.

        extract_families
            Whether the mobject's families should be recursively extracted.

        Returns
        -------
        Scene
            The Scene mobject with restructured Mobjects.
        """
        if extract_families:
            to_remove = extract_mobject_family_members(
                to_remove,
                use_z_index=self.renderer.camera.use_z_index,
            )
        _list = getattr(self, mobject_list_name)
        new_list = self.get_restructured_mobject_list(_list, to_remove)
        setattr(self, mobject_list_name, new_list)
        return self

    def get_restructured_mobject_list(self, mobjects: list, to_remove: list):
        """
        Given a list of mobjects and a list of mobjects to be removed, this
        filters out the removable mobjects from the list of mobjects.

        Parameters
        ----------

        mobjects
            The Mobjects to check.

        to_remove
            The list of mobjects to remove.

        Returns
        -------
        list
            The list of mobjects with the mobjects to remove removed.
        """

        new_mobjects = []

        def add_safe_mobjects_from_list(list_to_examine, set_to_remove):
            for mob in list_to_examine:
                if mob in set_to_remove:
                    continue
                intersect = set_to_remove.intersection(mob.get_family())
                if intersect:
                    add_safe_mobjects_from_list(mob.submobjects, intersect)
                else:
                    new_mobjects.append(mob)

        add_safe_mobjects_from_list(mobjects, set(to_remove))
        return new_mobjects

    # TODO, remove this, and calls to this
    def add_foreground_mobjects(self, *mobjects: Mobject):
        """
        Adds mobjects to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        *mobjects
            The Mobjects to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects added.
        """
        self.foreground_mobjects = list_update(self.foreground_mobjects, mobjects)
        self.add(*mobjects)
        return self

    def add_foreground_mobject(self, mobject: Mobject):
        """
        Adds a single mobject to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        mobject
            The Mobject to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject added.
        """
        return self.add_foreground_mobjects(mobject)

    def remove_foreground_mobjects(self, *to_remove: Mobject):
        """
        Removes mobjects from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        *to_remove
            The mobject(s) to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects removed.
        """
        self.restructure_mobjects(to_remove, "foreground_mobjects")
        return self

    def remove_foreground_mobject(self, mobject: Mobject):
        """
        Removes a single mobject from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        mobject
            The mobject to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject removed.
        """
        return self.remove_foreground_mobjects(mobject)

    def bring_to_front(self, *mobjects: Mobject):
        """
        Adds the passed mobjects to the scene again,
        pushing them to he front of the scene.

        Parameters
        ----------
        *mobjects
            The mobject(s) to bring to the front of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects brought to the front
            of the scene.
        """
        self.add(*mobjects)
        return self

    def bring_to_back(self, *mobjects: Mobject):
        """
        Removes the mobject from the scene and
        adds them to the back of the scene.

        Parameters
        ----------
        *mobjects
            The mobject(s) to push to the back of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects pushed to the back
            of the scene.
        """
        self.remove(*mobjects)
        self.mobjects = list(mobjects) + self.mobjects
        return self

    def clear(self):
        """
        Removes all mobjects present in self.mobjects
        and self.foreground_mobjects from the scene.

        Returns
        ------
        Scene
            The Scene, with all of its mobjects in
            self.mobjects and self.foreground_mobjects
            removed.
        """
        self.mobjects = []
        self.foreground_mobjects = []
        return self

    def get_moving_mobjects(self, *animations: Animation):
        """
        Gets all moving mobjects in the passed animation(s).

        Parameters
        ----------
        *animations
            The animations to check for moving mobjects.

        Returns
        ------
        list
            The list of mobjects that could be moving in
            the Animation(s)
        """
        # Go through mobjects from start to end, and
        # as soon as there's one that needs updating of
        # some kind per frame, return the list from that
        # point forward.
        animation_mobjects = [anim.mobject for anim in animations]
        mobjects = self.get_mobject_family_members()
        for i, mob in enumerate(mobjects):
            update_possibilities = [
                mob in animation_mobjects,
                len(mob.get_family_updaters()) > 0,
                mob in self.foreground_mobjects,
            ]
            if any(update_possibilities):
                return mobjects[i:]
        return []

    def get_moving_and_static_mobjects(self, animations):
        all_mobjects = list_update(self.mobjects, self.foreground_mobjects)
        all_mobject_families = extract_mobject_family_members(
            all_mobjects,
            use_z_index=self.renderer.camera.use_z_index,
            only_those_with_points=True,
        )
        moving_mobjects = self.get_moving_mobjects(*animations)
        all_moving_mobject_families = extract_mobject_family_members(
            moving_mobjects,
            use_z_index=self.renderer.camera.use_z_index,
        )
        static_mobjects = list_difference_update(
            all_mobject_families,
            all_moving_mobject_families,
        )
        return all_moving_mobject_families, static_mobjects

    def compile_animations(
        self,
        *args: Animation | Iterable[Animation] | types.GeneratorType[Animation],
        **kwargs,
    ):
        """
        Creates _MethodAnimations from any _AnimationBuilders and updates animation
        kwargs with kwargs passed to play().

        Parameters
        ----------
        *args
            Animations to be played.
        **kwargs
            Configuration for the call to play().

        Returns
        -------
        Tuple[:class:`Animation`]
            Animations to be played.
        """
        animations = []
        arg_anims = flatten_iterable_parameters(args)
        # Allow passing a generator to self.play instead of comma separated arguments
        for arg in arg_anims:
            try:
                animations.append(prepare_animation(arg))
            except TypeError:
                if inspect.ismethod(arg):
                    raise TypeError(
                        "Passing Mobject methods to Scene.play is no longer"
                        " supported. Use Mobject.animate instead.",
                    )
                else:
                    raise TypeError(
                        f"Unexpected argument {arg} passed to Scene.play().",
                    )

        for animation in animations:
            for k, v in kwargs.items():
                setattr(animation, k, v)

        return animations

    def _get_animation_time_progression(
        self, animations: list[Animation], duration: float
    ):
        """
        You will hardly use this when making your own animations.
        This method is for Manim's internal use.

        Uses :func:`~.get_time_progression` to obtain a
        CommandLine ProgressBar whose ``fill_time`` is
        dependent on the qualities of the passed Animation,

        Parameters
        ----------
        animations
            The list of animations to get
            the time progression for.

        duration
            duration of wait time

        Returns
        -------
        time_progression
            The CommandLine Progress Bar.
        """
        if len(animations) == 1 and isinstance(animations[0], Wait):
            stop_condition = animations[0].stop_condition
            if stop_condition is not None:
                time_progression = self.get_time_progression(
                    duration,
                    f"Waiting for {stop_condition.__name__}",
                    n_iterations=-1,  # So it doesn't show % progress
                    override_skip_animations=True,
                )
            else:
                time_progression = self.get_time_progression(
                    duration,
                    f"Waiting {self.renderer.num_plays}",
                )
        else:
            time_progression = self.get_time_progression(
                duration,
                "".join(
                    [
                        f"Animation {self.renderer.num_plays}: ",
                        str(animations[0]),
                        (", etc." if len(animations) > 1 else ""),
                    ],
                ),
            )
        return time_progression

    def get_time_progression(
        self,
        run_time: float,
        description,
        n_iterations: int | None = None,
        override_skip_animations: bool = False,
    ):
        """
        You will hardly use this when making your own animations.
        This method is for Manim's internal use.

        Returns a CommandLine ProgressBar whose ``fill_time``
        is dependent on the ``run_time`` of an animation,
        the iterations to perform in that animation
        and a bool saying whether or not to consider
        the skipped animations.

        Parameters
        ----------
        run_time
            The ``run_time`` of the animation.

        n_iterations
            The number of iterations in the animation.

        override_skip_animations
            Whether or not to show skipped animations in the progress bar.

        Returns
        -------
        time_progression
            The CommandLine Progress Bar.
        """
        if self.renderer.skip_animations and not override_skip_animations:
            times = [run_time]
        else:
            step = 1 / config["frame_rate"]
            times = np.arange(0, run_time, step)
        time_progression = tqdm(
            times,
            desc=description,
            total=n_iterations,
            leave=config["progress_bar"] == "leave",
            ascii=True if platform.system() == "Windows" else None,
            disable=config["progress_bar"] == "none",
        )
        return time_progression

    def get_run_time(self, animations: list[Animation]):
        """
        Gets the total run time for a list of animations.

        Parameters
        ----------
        animations
            A list of the animations whose total
            ``run_time`` is to be calculated.

        Returns
        -------
        float
            The total ``run_time`` of all of the animations in the list.
        """

        if len(animations) == 1 and isinstance(animations[0], Wait):
            return animations[0].duration

        else:
            return np.max([animation.run_time for animation in animations])

    def play(
        self,
        *args: Animation | Iterable[Animation] | types.GeneratorType[Animation],
        subcaption=None,
        subcaption_duration=None,
        subcaption_offset=0,
        **kwargs,
    ):
        r"""Plays an animation in this scene.

        Parameters
        ----------

        args
            Animations to be played.
        subcaption
            The content of the external subcaption that should
            be added during the animation.
        subcaption_duration
            The duration for which the specified subcaption is
            added. If ``None`` (the default), the run time of the
            animation is taken.
        subcaption_offset
            An offset (in seconds) for the start time of the
            added subcaption.
        kwargs
            All other keywords are passed to the renderer.

        """
        # If we are in interactive embedded mode, make sure this is running on the main thread (required for OpenGL)
        if (
            self.interactive_mode
            and config.renderer == RendererType.OPENGL
            and threading.current_thread().name != "MainThread"
        ):
            kwargs.update(
                {
                    "subcaption": subcaption,
                    "subcaption_duration": subcaption_duration,
                    "subcaption_offset": subcaption_offset,
                }
            )
            self.queue.put(
                (
                    "play",
                    args,
                    kwargs,
                )
            )
            return

        start_time = self.renderer.time
        self.renderer.play(self, *args, **kwargs)
        run_time = self.renderer.time - start_time
        if subcaption:
            if subcaption_duration is None:
                subcaption_duration = run_time
            # The start of the subcaption needs to be offset by the
            # run_time of the animation because it is added after
            # the animation has already been played (and Scene.renderer.time
            # has already been updated).
            self.add_subcaption(
                content=subcaption,
                duration=subcaption_duration,
                offset=-run_time + subcaption_offset,
            )

    def wait(
        self,
        duration: float = DEFAULT_WAIT_TIME,
        stop_condition: Callable[[], bool] | None = None,
        frozen_frame: bool | None = None,
    ):
        """Plays a "no operation" animation.

        Parameters
        ----------
        duration
            The run time of the animation.
        stop_condition
            A function without positional arguments that is evaluated every time
            a frame is rendered. The animation only stops when the return value
            of the function is truthy, or when the time specified in ``duration``
            passes.
        frozen_frame
            If True, updater functions are not evaluated, and the animation outputs
            a frozen frame. If False, updater functions are called and frames
            are rendered as usual. If None (the default), the scene tries to
            determine whether or not the frame is frozen on its own.

        See also
        --------
        :class:`.Wait`, :meth:`.should_mobjects_update`
        """
        self.play(
            Wait(
                run_time=duration,
                stop_condition=stop_condition,
                frozen_frame=frozen_frame,
            )
        )

    def pause(self, duration: float = DEFAULT_WAIT_TIME):
        """Pauses the scene (i.e., displays a frozen frame).

        This is an alias for :meth:`.wait` with ``frozen_frame``
        set to ``True``.

        Parameters
        ----------
        duration
            The duration of the pause.

        See also
        --------
        :meth:`.wait`, :class:`.Wait`
        """
        self.wait(duration=duration, frozen_frame=True)

    def wait_until(self, stop_condition: Callable[[], bool], max_time: float = 60):
        """Wait until a condition is satisfied, up to a given maximum duration.

        Parameters
        ----------
        stop_condition
            A function with no arguments that determines whether or not the
            scene should keep waiting.
        max_time
            The maximum wait time in seconds.
        """
        self.wait(max_time, stop_condition=stop_condition)

    def compile_animation_data(
        self,
        *animations: Animation | Iterable[Animation] | types.GeneratorType[Animation],
        **play_kwargs,
    ):
        """Given a list of animations, compile the corresponding
        static and moving mobjects, and gather the animation durations.

        This also begins the animations.

        Parameters
        ----------
        animations
            Animation or mobject with mobject method and params
        play_kwargs
            Named parameters affecting what was passed in ``animations``,
            e.g. ``run_time``, ``lag_ratio`` and so on.

        Returns
        -------
        self, None
            None if there is nothing to play, or self otherwise.
        """
        # NOTE TODO : returns statement of this method are wrong. It should return nothing, as it makes a little sense to get any information from this method.
        # The return are kept to keep webgl renderer from breaking.
        if len(animations) == 0:
            raise ValueError("Called Scene.play with no animations")

        self.animations = self.compile_animations(*animations, **play_kwargs)
        self.add_mobjects_from_animations(self.animations)

        self.last_t = 0
        self.stop_condition = None
        self.moving_mobjects = []
        self.static_mobjects = []

        if len(self.animations) == 1 and isinstance(self.animations[0], Wait):
            if self.should_update_mobjects():
                self.update_mobjects(dt=0)  # Any problems with this?
                self.stop_condition = self.animations[0].stop_condition
            else:
                self.duration = self.animations[0].duration
                # Static image logic when the wait is static is done by the renderer, not here.
                self.animations[0].is_static_wait = True
                return None
        self.duration = self.get_run_time(self.animations)
        return self

    def begin_animations(self) -> None:
        """Start the animations of the scene."""
        for animation in self.animations:
            animation._setup_scene(self)
            animation.begin()

        if config.renderer == RendererType.CAIRO:
            # Paint all non-moving objects onto the screen, so they don't
            # have to be rendered every frame
            (
                self.moving_mobjects,
                self.static_mobjects,
            ) = self.get_moving_and_static_mobjects(self.animations)

    def is_current_animation_frozen_frame(self) -> bool:
        """Returns whether the current animation produces a static frame (generally a Wait)."""
        return (
            isinstance(self.animations[0], Wait)
            and len(self.animations) == 1
            and self.animations[0].is_static_wait
        )

    def play_internal(self, skip_rendering: bool = False):
        """
        This method is used to prep the animations for rendering,
        apply the arguments and parameters required to them,
        render them, and write them to the video file.

        Parameters
        ----------
        skip_rendering
            Whether the rendering should be skipped, by default False
        """
        self.duration = self.get_run_time(self.animations)
        self.time_progression = self._get_animation_time_progression(
            self.animations,
            self.duration,
        )
        for t in self.time_progression:
            self.update_to_time(t)
            if not skip_rendering and not self.skip_animation_preview:
                self.renderer.render(self, t, self.moving_mobjects)
            if self.stop_condition is not None and self.stop_condition():
                self.time_progression.close()
                break

        for animation in self.animations:
            animation.finish()
            animation.clean_up_from_scene(self)
        if not self.renderer.skip_animations:
            self.update_mobjects(0)
        self.renderer.static_image = None
        # Closing the progress bar at the end of the play.
        self.time_progression.close()

    def check_interactive_embed_is_valid(self):
        if config["force_window"]:
            return True
        if self.skip_animation_preview:
            logger.warning(
                "Disabling interactive embed as 'skip_animation_preview' is enabled",
            )
            return False
        elif config["write_to_movie"]:
            logger.warning("Disabling interactive embed as 'write_to_movie' is enabled")
            return False
        elif config["format"]:
            logger.warning(
                "Disabling interactive embed as '--format' is set as "
                + config["format"],
            )
            return False
        elif not self.renderer.window:
            logger.warning("Disabling interactive embed as no window was created")
            return False
        elif config.dry_run:
            logger.warning("Disabling interactive embed as dry_run is enabled")
            return False
        return True

    def interactive_embed(self):
        """
        Like embed(), but allows for screen interaction.
        """
        if not self.check_interactive_embed_is_valid():
            return
        self.interactive_mode = True

        def ipython(shell, namespace):
            import manim.opengl

            def load_module_into_namespace(module, namespace):
                for name in dir(module):
                    namespace[name] = getattr(module, name)

            load_module_into_namespace(manim, namespace)
            load_module_into_namespace(manim.opengl, namespace)

            def embedded_rerun(*args, **kwargs):
                self.queue.put(("rerun_keyboard", args, kwargs))
                shell.exiter()

            namespace["rerun"] = embedded_rerun

            shell(local_ns=namespace)
            self.queue.put(("exit_keyboard", [], {}))

        def get_embedded_method(method_name):
            return lambda *args, **kwargs: self.queue.put((method_name, args, kwargs))

        local_namespace = inspect.currentframe().f_back.f_locals
        for method in ("play", "wait", "add", "remove"):
            embedded_method = get_embedded_method(method)
            # Allow for calling scene methods without prepending 'self.'.
            local_namespace[method] = embedded_method

        from sqlite3 import connect

        from IPython.core.getipython import get_ipython
        from IPython.terminal.embed import InteractiveShellEmbed
        from traitlets.config import Config

        cfg = Config()
        cfg.TerminalInteractiveShell.confirm_exit = False
        if get_ipython() is None:
            shell = InteractiveShellEmbed.instance(config=cfg)
        else:
            shell = InteractiveShellEmbed(config=cfg)
        hist = get_ipython().history_manager
        hist.db = connect(hist.hist_file, check_same_thread=False)

        keyboard_thread = threading.Thread(
            target=ipython,
            args=(shell, local_namespace),
        )
        # run as daemon to kill thread when main thread exits
        if not shell.pt_app:
            keyboard_thread.daemon = True
        keyboard_thread.start()

        if self.dearpygui_imported and config["enable_gui"]:
            if not dpg.is_dearpygui_running():
                gui_thread = threading.Thread(
                    target=configure_pygui,
                    args=(self.renderer, self.widgets),
                    kwargs={"update": False},
                )
                gui_thread.start()
            else:
                configure_pygui(self.renderer, self.widgets, update=True)

        self.camera.model_matrix = self.camera.default_model_matrix

        self.interact(shell, keyboard_thread)

    def interact(self, shell, keyboard_thread):
        event_handler = RerunSceneHandler(self.queue)
        file_observer = Observer()
        file_observer.schedule(event_handler, config["input_file"], recursive=True)
        file_observer.start()

        self.quit_interaction = False
        keyboard_thread_needs_join = shell.pt_app is not None
        assert self.queue.qsize() == 0

        last_time = time.time()
        while not (self.renderer.window.is_closing or self.quit_interaction):
            if not self.queue.empty():
                tup = self.queue.get_nowait()
                if tup[0].startswith("rerun"):
                    # Intentionally skip calling join() on the file thread to save time.
                    if not tup[0].endswith("keyboard"):
                        if shell.pt_app:
                            shell.pt_app.app.exit(exception=EOFError)
                        file_observer.unschedule_all()
                        raise RerunSceneException
                    keyboard_thread.join()

                    kwargs = tup[2]
                    if "from_animation_number" in kwargs:
                        config["from_animation_number"] = kwargs[
                            "from_animation_number"
                        ]
                    # # TODO: This option only makes sense if interactive_embed() is run at the
                    # # end of a scene by default.
                    # if "upto_animation_number" in kwargs:
                    #     config["upto_animation_number"] = kwargs[
                    #         "upto_animation_number"
                    #     ]

                    keyboard_thread.join()
                    file_observer.unschedule_all()
                    raise RerunSceneException
                elif tup[0].startswith("exit"):
                    # Intentionally skip calling join() on the file thread to save time.
                    if not tup[0].endswith("keyboard") and shell.pt_app:
                        shell.pt_app.app.exit(exception=EOFError)
                    keyboard_thread.join()
                    # Remove exit_keyboard from the queue if necessary.
                    while self.queue.qsize() > 0:
                        self.queue.get()
                    keyboard_thread_needs_join = False
                    break
                else:
                    method, args, kwargs = tup
                    getattr(self, method)(*args, **kwargs)
            else:
                self.renderer.animation_start_time = 0
                dt = time.time() - last_time
                last_time = time.time()
                self.renderer.render(self, dt, self.moving_mobjects)
                self.update_mobjects(dt)
                self.update_meshes(dt)
                self.update_self(dt)

        # Join the keyboard thread if necessary.
        if shell is not None and keyboard_thread_needs_join:
            shell.pt_app.app.exit(exception=EOFError)
            keyboard_thread.join()
            # Remove exit_keyboard from the queue if necessary.
            while self.queue.qsize() > 0:
                self.queue.get()

        file_observer.stop()
        file_observer.join()

        if self.dearpygui_imported and config["enable_gui"]:
            dpg.stop_dearpygui()

        if self.renderer.window.is_closing:
            self.renderer.window.destroy()

    def embed(self):
        if not config["preview"]:
            logger.warning("Called embed() while no preview window is available.")
            return
        if config["write_to_movie"]:
            logger.warning("embed() is skipped while writing to a file.")
            return

        self.renderer.animation_start_time = 0
        self.renderer.render(self, -1, self.moving_mobjects)

        # Configure IPython shell.
        from IPython.terminal.embed import InteractiveShellEmbed

        shell = InteractiveShellEmbed()

        # Have the frame update after each command
        shell.events.register(
            "post_run_cell",
            lambda *a, **kw: self.renderer.render(self, -1, self.moving_mobjects),
        )

        # Use the locals of the caller as the local namespace
        # once embedded, and add a few custom shortcuts.
        local_ns = inspect.currentframe().f_back.f_locals
        # local_ns["touch"] = self.interact
        for method in (
            "play",
            "wait",
            "add",
            "remove",
            "interact",
            # "clear",
            # "save_state",
            # "restore",
        ):
            local_ns[method] = getattr(self, method)
        shell(local_ns=local_ns, stack_depth=2)

        # End scene when exiting an embed.
        raise Exception("Exiting scene.")

    def update_to_time(self, t):
        dt = t - self.last_t
        self.last_t = t
        for animation in self.animations:
            animation.update_mobjects(dt)
            alpha = t / animation.run_time
            animation.interpolate(alpha)
        self.update_mobjects(dt)
        self.update_meshes(dt)
        self.update_self(dt)

    def add_subcaption(
        self, content: str, duration: float = 1, offset: float = 0
    ) -> None:
        r"""Adds an entry in the corresponding subcaption file
        at the current time stamp.

        The current time stamp is obtained from ``Scene.renderer.time``.

        Parameters
        ----------

        content
            The subcaption content.
        duration
            The duration (in seconds) for which the subcaption is shown.
        offset
            This offset (in seconds) is added to the starting time stamp
            of the subcaption.

        Examples
        --------

        This example illustrates both possibilities for adding
        subcaptions to Manimations::

            class SubcaptionExample(Scene):
                def construct(self):
                    square = Square()
                    circle = Circle()

                    # first option: via the add_subcaption method
                    self.add_subcaption("Hello square!", duration=1)
                    self.play(Create(square))

                    # second option: within the call to Scene.play
                    self.play(
                        Transform(square, circle),
                        subcaption="The square transforms."
                    )

        """
        subtitle = srt.Subtitle(
            index=len(self.renderer.file_writer.subcaptions),
            content=content,
            start=datetime.timedelta(seconds=float(self.renderer.time + offset)),
            end=datetime.timedelta(
                seconds=float(self.renderer.time + offset + duration)
            ),
        )
        self.renderer.file_writer.subcaptions.append(subtitle)

    # 添加声音
    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        **kwargs,
    ):
        """
        This method is used to add a sound to the animation.

        Parameters
        ----------

        sound_file
            The path to the sound file.
        time_offset
            The offset in the sound file after which
            the sound can be played.
        gain
            Amplification of the sound.

        Examples
        --------
        .. manim:: SoundExample
            :no_autoplay:

            class SoundExample(Scene):
                # Source of sound under Creative Commons 0 License. https://freesound.org/people/Druminfected/sounds/250551/
                def construct(self):
                    dot = Dot().set_color(GREEN)
                    self.add_sound("click.wav")
                    self.add(dot)
                    self.wait()
                    self.add_sound("click.wav")
                    dot.set_color(BLUE)
                    self.wait()
                    self.add_sound("click.wav")
                    dot.set_color(RED)
                    self.wait()

        Download the resource for the previous example `here <https://github.com/ManimCommunity/manim/blob/main/docs/source/_static/click.wav>`_ .
        """
        if self.renderer.skip_animations:
            return
        time = self.renderer.time + time_offset
        self.renderer.file_writer.add_sound(sound_file, time, gain, **kwargs)

    # 处理鼠标移动事件
    def on_mouse_motion(self, point, d_point):
        self.mouse_point.move_to(point)
        if SHIFT_VALUE in self.renderer.pressed_keys:
            shift = -d_point
            shift[0] *= self.camera.get_width() / 2
            shift[1] *= self.camera.get_height() / 2
            transform = self.camera.inverse_rotation_matrix
            shift = np.dot(np.transpose(transform), shift)
            self.camera.shift(shift)

    # 处理鼠标滚轮事件
    def on_mouse_scroll(self, point, offset):
        # 确定是否启用投影笔触着色器
        if not config.use_projection_stroke_shaders:
            # 如果启用了这个功能，它会计算一个缩放因子 factor
            factor = 1 + np.arctan(-2.1 * offset[1])
            # 以 self.camera_target 为中心缩放相机
            self.camera.scale(factor, about_point=self.camera_target)
        # 处理鼠标滚轮控制
        self.mouse_scroll_orbit_controls(point, offset)

    # 定义键盘按下时的行为
    def on_key_press(self, symbol, modifiers):
        # 尝试将按键符号转换为相应的字符
        try:
            char = chr(symbol)
        # 果按键符号太大而无法转换为字符，则记录一个警告消息，表示按键值过大。
        except OverflowError:
            logger.warning("按键值过大！")
            # 退出当前函数，避免继续执行后续代码
            return
        # 按下'r'键可以将相机状态重置为默认状态，并将相机的目标位置设置为原点
        if char == "r":
            self.camera.to_default_state()
            self.camera_target = np.array([0, 0, 0], dtype=np.float32)
        # 按下'q'键可以退出交互模式
        elif char == "q":
            self.quit_interaction = True
        # 如果按下的键有特定的函数映射到它，那么将执行该函数
        else:
            if char in self.key_to_function_map:
                self.key_to_function_map[char]()
    
    # 定义键盘松开时的行为
    def on_key_release(self, symbol, modifiers):
        pass

    # 定义鼠标拖拽时的行为
    def on_mouse_drag(self, point, d_point, buttons, modifiers):
        self.mouse_drag_point.move_to(point)
        # 鼠标左键按下时（buttons == 1），相机会根据鼠标的水平和垂直移动改变水平角度（theta）和垂直角度（phi）
        if buttons == 1:
            self.camera.increment_theta(-d_point[0])
            self.camera.increment_phi(d_point[1])
        # 鼠标右键按下时（buttons == 4），相机会沿着相机自身的X轴和与OUT方向的叉乘方向移动，以实现平移视角的效果
        elif buttons == 4:
            camera_x_axis = self.camera.model_matrix[:3, 0]
            horizontal_shift_vector = -d_point[0] * camera_x_axis
            vertical_shift_vector = -d_point[1] * np.cross(OUT, camera_x_axis)
            total_shift_vector = horizontal_shift_vector + vertical_shift_vector
            self.camera.shift(1.1 * total_shift_vector)
        self.mouse_drag_orbit_controls(point, d_point, buttons, modifiers)

    # 处理鼠标滚轮控制相机轨道运动，根据鼠标滚轮的偏移量offset来确定相机沿着视线方向移动的距离
    def mouse_scroll_orbit_controls(self, point, offset):
        # 首先计算相机到目标点的向量
        camera_to_target = self.camera_target - self.camera.get_position()
        # 滚轮向上滚动为正，向下滚动为负，调整这个向量的方向
        camera_to_target *= np.sign(offset[1])
        # 乘以一个固定的缩放因子
        shift_vector = 0.01 * camera_to_target
        # 将相机的模型矩阵应用这个位移来实现相机的移动
        self.camera.model_matrix = (
            opengl.translation_matrix(*shift_vector) @ self.camera.model_matrix
        )

    # 鼠标拖动控制相机，允许用户通过鼠标左键和右键进行不同的操作
    def mouse_drag_orbit_controls(self, point, d_point, buttons, modifiers):
        # 鼠标左键拖动时，相机围绕目标点旋转
        if buttons == 1:
            # 将操作点转换为以原点为中心，围绕 z 轴旋转
            self.camera.model_matrix = (
                # 围绕 z 轴以 d_point[0] 的负值为角度进行旋转。这里的 d_point[0] 可能是鼠标在 x 方向上的拖动距离
                opengl.rotation_matrix(z=-d_point[0])
                # 将相机位置平移到目标点的负值位置。这样可以使相机绕目标点旋转，而不是围绕自己的原点
                @ opengl.translation_matrix(*-self.camera_target)
                # 将上述旋转和平移操作应用到相机的模型矩阵上，从而改变相机的位置和方向
                @ self.camera.model_matrix
            )
            # 处理鼠标拖拽导致的相机绕特定轴的旋转
            # 获取相机当前的位置
            camera_position = self.camera.get_position()
            # 获取相机当前的 y 轴方向，即相机模型矩阵的第二列（索引为 1）
            camera_y_axis = self.camera.model_matrix[:3, 1]
            # 计算旋转轴，即相机的 y 轴和相机位置向量的叉乘结果，然后进行归一化处理
            axis_of_rotation = space_ops.normalize(
                np.cross(camera_y_axis, camera_position),
            )
            # 根据鼠标在垂直方向上的拖拽量 d_point[1] 和旋转轴 axis_of_rotation，生成一个绕该轴旋转的旋转矩阵
            rotation_matrix = space_ops.rotation_matrix(
                d_point[1],
                axis_of_rotation,
                homogeneous=True,
            )
            # 获取相机的最大极角和最小极角，限制相机绕某些轴的旋转范围，以确保相机的视角不会超出特定范围
            maximum_polar_angle = self.camera.maximum_polar_angle
            minimum_polar_angle = self.camera.minimum_polar_angle
            # 计算了潜在的相机位置和旋转后的模型矩阵
            potential_camera_model_matrix = rotation_matrix @ self.camera.model_matrix
            potential_camera_location = potential_camera_model_matrix[:3, 3]
            potential_camera_y_axis = potential_camera_model_matrix[:3, 1]
            # 确定角度符号的值，保证了sign的值在potential_camera_y_axis[2]为零时仍然有效，避免了除以零的错误
            sign = (
                np.sign(potential_camera_y_axis[2])
                if potential_camera_y_axis[2] != 0
                else 1
            )
            # 计算了潜在的极角，用于确定相机绕其观察目标点的旋转角度
            potential_polar_angle = sign * np.arccos(
                potential_camera_location[2]
                / np.linalg.norm(potential_camera_location),
            )
            # 检查这个极角是否在允许的范围内
            if minimum_polar_angle <= potential_polar_angle <= maximum_polar_angle:
                # 如果在范围内，则直接更新相机的模型矩阵为潜在的模型矩阵
                self.camera.model_matrix = potential_camera_model_matrix
            # 如果不在范围内，则计算需要旋转的角度，并重新计算旋转矩阵并应用到相机的模型矩阵上
            else:
                sign = np.sign(camera_y_axis[2]) if camera_y_axis[2] != 0 else 1
                current_polar_angle = sign * np.arccos(
                    camera_position[2] / np.linalg.norm(camera_position),
                )
                if potential_polar_angle > maximum_polar_angle:
                    polar_angle_delta = maximum_polar_angle - current_polar_angle
                else:
                    polar_angle_delta = minimum_polar_angle - current_polar_angle
                rotation_matrix = space_ops.rotation_matrix(
                    polar_angle_delta,
                    axis_of_rotation,
                    homogeneous=True,
                )
                self.camera.model_matrix = rotation_matrix @ self.camera.model_matrix
            # 将相机平移回原始的目标点位置，以保持相机视线方向的稳定
            self.camera.model_matrix = (
                # 根据相机的目标点位置 self.camera_target，创建一个平移矩阵，使得相机的目标点移动到世界坐标系的原点位置
                opengl.translation_matrix(*self.camera_target)
                # 与当前的相机模型矩阵相乘
                @ self.camera.model_matrix
            )
        # 鼠标右键拖动时，相机在平面上的平移
        elif buttons == 4:
            # 获取相机模型矩阵的第一列，即相机的 x 轴方向
            camera_x_axis = self.camera.model_matrix[:3, 0]
            # 根据鼠标水平方向上的移动距离 d_point[0]，计算出相机需要水平平移的向量，即相机 x 轴方向上的移动
            horizontal_shift_vector = -d_point[0] * camera_x_axis
            # 根据鼠标垂直方向上的移动距离 d_point[1]，通过计算相机的 y 轴方向与世界坐标系的 z 轴方向的叉乘，计算出相机需要垂直平移的向量，即相机 y 轴方向上的移动
            vertical_shift_vector = -d_point[1] * np.cross(OUT, camera_x_axis)
            # 将水平和垂直方向上的移动向量相加，得到总的平移向量
            total_shift_vector = horizontal_shift_vector + vertical_shift_vector
            # 将总的平移向量应用到相机的模型矩阵上，实现相机的平移操作
            self.camera.model_matrix = (
                opengl.translation_matrix(*total_shift_vector)
                @ self.camera.model_matrix
            )
            # 同时更新相机的目标点，保持相机视线的方向不变，但相机的位置发生了变化，从而实现平移效果
            self.camera_target += total_shift_vector

    # 将按键与函数关联起来，当按下特定按键时，将执行相应的函数
    def set_key_function(self, char, func):
        self.key_to_function_map[char] = func

    # 处理鼠标按下事件，依次调用已注册的鼠标按下回调函数
    def on_mouse_press(self, point, button, modifiers):
        for func in self.mouse_press_callbacks:
            func()
