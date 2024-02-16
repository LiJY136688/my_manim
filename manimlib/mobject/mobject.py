from __future__ import annotations

import copy
from functools import wraps
import itertools as it
import os
import pickle
import random
import sys

import moderngl
import numbers
import numpy as np

from manimlib.constants import DEFAULT_MOBJECT_TO_EDGE_BUFFER
from manimlib.constants import DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
from manimlib.constants import DOWN, IN, LEFT, ORIGIN, OUT, RIGHT, UP
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from manimlib.constants import MED_SMALL_BUFF
from manimlib.constants import TAU
from manimlib.constants import WHITE
from manimlib.event_handler import EVENT_DISPATCHER
from manimlib.event_handler.event_listner import EventListener
from manimlib.event_handler.event_type import EventType
from manimlib.logger import log
from manimlib.shader_wrapper import ShaderWrapper
from manimlib.utils.color import color_gradient
from manimlib.utils.color import color_to_rgb
from manimlib.utils.color import get_colormap_list
from manimlib.utils.color import rgb_to_hex
from manimlib.utils.iterables import arrays_match
from manimlib.utils.iterables import array_is_constant
from manimlib.utils.iterables import batch_by_property
from manimlib.utils.iterables import list_update
from manimlib.utils.iterables import listify
from manimlib.utils.iterables import resize_array
from manimlib.utils.iterables import resize_preserving_order
from manimlib.utils.iterables import resize_with_interpolation
from manimlib.utils.bezier import integer_interpolate
from manimlib.utils.bezier import interpolate
from manimlib.utils.paths import straight_path
from manimlib.utils.simple_functions import get_parameters
from manimlib.utils.shaders import get_colormap_code
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import rotation_matrix_transpose

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Iterable, Iterator, Union, Tuple, Optional
    import numpy.typing as npt
    from manimlib.typing import ManimColor, Vect3, Vect4, Vect3Array, UniformDict, Self
    from moderngl.context import Context

    TimeBasedUpdater = Callable[["Mobject", float], "Mobject" | None]
    NonTimeUpdater = Callable[["Mobject"], "Mobject" | None]
    Updater = Union[TimeBasedUpdater, NonTimeUpdater]

# 数学对象类
class Mobject(object):
    # 数学对象的维度，默认为3维
    dim: int = 3
    # 用于此数学对象的着色器文件夹的路径，默认为空字符串
    shader_folder: str = ""
    # 渲染时使用的图元类型
    render_primitive: int = moderngl.TRIANGLE_STRIP
    # 顶点着色器的数据类型，必须与垂直着色器的属性匹配
    shader_dtype: np.dtype = np.dtype([
        # 顶点的位置（三维坐标）
        ('point', np.float32, (3,)),
        # 颜色（四维RGBA值）
        ('rgba', np.float32, (4,)),
    ])
    # 对齐数据键
    aligned_data_keys = ['point']
    # 点类似数据键
    pointlike_data_keys = ['point']

    # 数学对象类的初始化方法
    def __init__(
        self,
        # 颜色，默认为WHITE
        color: ManimColor = WHITE,
        # 不透明度，默认为1.0
        opacity: float = 1.0,
        # 阴影颜色，默认为(0.0, 0.0, 0.0)
        shading: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        # 纹理路径，不同部分使用不同的纹理，默认为空字典
        texture_paths: dict[str, str] | None = None,
        # 是否固定在帧中，默认不会随着相机位置的旋转而旋转
        is_fixed_in_frame: bool = False,
        # 是否进行深度测试，默认不进行
        depth_test: bool = False,
    ):
        # 依次代入各项参数
        self.color = color
        self.opacity = opacity
        self.shading = shading
        self.texture_paths = texture_paths
        self._is_fixed_in_frame = is_fixed_in_frame
        self.depth_test = depth_test

        # 内部状态
        # 当前对象的所有子对象，默认为空字典
        self.submobjects: list[Mobject] = []
        # 当前对象的所有父对象，默认为空字典
        self.parents: list[Mobject] = []
        # 当前对象及其所有相关对象组成的对象组，默认为只有自己的字典
        self.family: list[Mobject] = [self]
        # 锁定的数据键集合，表示不能修改的数据键，默认为空集合
        self.locked_data_keys: set[str] = set()
        # 常量数据键集合，表示不会在运行时更改的数据键，默认为空集合
        self.const_data_keys: set[str] = set()
        # 锁定的统一键集合，表示不能修改的统一键，默认为空集合
        self.locked_uniform_keys: set[str] = set()
        # 是否需要计算新的边界框，默认需要
        self.needs_new_bounding_box: bool = True
        # 对象是否正在进行动画，默认没有
        self._is_animating: bool = False
        # 保存的对象状态，默认还没东西
        self.saved_state = None
        # 对象的目标状态，默认还没东西
        self.target = None
        # 对象的边界框，默认为3×3的零矩阵
        self.bounding_box: Vect3Array = np.zeros((3, 3))
        # 着色器是否已初始化，默认还没有
        self._shaders_initialized: bool = False
        # 数据是否已更改，默认已更改
        self._data_has_changed: bool = True
        # 着色器代码替换字典，用于替换着色器中的代码，默认为空字典
        self.shader_code_replacements: dict[str, str] = dict()

        # 初始化对象的数据
        self.init_data()
        # 初始化对象的默认数据为一个值为1的数组，数组长度为1，数据类型与对象的数据类型一致
        self._data_defaults = np.ones(1, dtype=self.data.dtype)
        # 初始化对象的统一键
        self.init_uniforms()
        # 初始化对象的更新器
        self.init_updaters()
        # 初始化对象的事件监听器
        self.init_event_listners()
        # 初始化对象的点
        self.init_points()
        # 初始化对象的颜色
        self.init_colors()

        # 如果需要深度测试
        if self.depth_test:
            # 就进行深度测试
            self.apply_depth_test()

    # 获取对象的类名
    def __str__(self):
        return self.__class__.__name__

    # 对象之间的合并或组合
    def __add__(self, other: Mobject) -> Mobject:
        assert(isinstance(other, Mobject))
        # 创建一个新的群组对象，包含对象自身和新加入的对象
        return self.get_group_class()(self, other)

    # 对象自身的复制或重复
    def __mul__(self, other: int) -> Mobject:
        assert(isinstance(other, int))
        # 根据整数多少复制几个对象
        return self.replicate(other)

    # 初始化对象的数据
    def init_data(self, length: int = 0):
        # 创建一个指定长度的全零数组
        self.data = np.zeros(length, dtype=self.shader_dtype)

    # 初始化对象的统一键
    def init_uniforms(self):
        self.uniforms: UniformDict = {
            # 是否固定在帧中
            "is_fixed_in_frame": float(self._is_fixed_in_frame),
            # 对象的阴影效果
            "shading": np.array(self.shading, dtype=float),
        }
        
    # 初始化对象的颜色
    def init_colors(self):
        # 根据对象的颜色和不透明度设置其颜色
        self.set_color(self.color, self.opacity)

    # 初始化对象的点，子类中实现
    def init_points(self):
        pass

    # 设置对象的统一键
    def set_uniforms(self, uniforms: dict) -> Self:
        # 遍历uniforms字典的每个键值对
        for key, value in uniforms.items():
            # 如果值是NumPy数组
            if isinstance(value, np.ndarray):
                # 创建其副本以避免修改原数组
                value = value.copy()
            # 将键值对添加到统一键键值对属性中
            self.uniforms[key] = value
        return self

    # @property用于将一个方法转换为属性，使得可以像访问属性一样访问该方法
    @property
    # 构建对象的动画
    def animate(self) -> _AnimationBuilder:
        return _AnimationBuilder(self)

    def note_changed_data(self, recurse_up: bool = True) -> Self:
        self._data_has_changed = True
        if recurse_up:
            for mob in self.parents:
                mob.note_changed_data()
        return self

    def affects_data(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.note_changed_data()
        return wrapper

    def affects_family_data(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            for mob in self.family_members_with_points():
                mob.note_changed_data()
            return self
        return wrapper

    # Only these methods should directly affect points
    @affects_data
    def set_data(self, data: np.ndarray) -> Self:
        assert(data.dtype == self.data.dtype)
        self.resize_points(len(data))
        self.data[:] = data
        return self

    @affects_data
    def resize_points(
        self,
        new_length: int,
        resize_func: Callable[[np.ndarray, int], np.ndarray] = resize_array
    ) -> Self:
        if new_length == 0:
            if len(self.data) > 0:
                self._data_defaults[:1] = self.data[:1]
        elif self.get_num_points() == 0:
            self.data = self._data_defaults.copy()

        self.data = resize_func(self.data, new_length)
        self.refresh_bounding_box()
        return self

    @affects_data
    def set_points(self, points: Vect3Array | list[Vect3]) -> Self:
        self.resize_points(len(points), resize_func=resize_preserving_order)
        self.data["point"][:] = points
        return self

    @affects_data
    def append_points(self, new_points: Vect3Array) -> Self:
        n = self.get_num_points()
        self.resize_points(n + len(new_points))
        # Have most data default to the last value
        self.data[n:] = self.data[n - 1]
        # Then read in new points
        self.data["point"][n:] = new_points
        self.refresh_bounding_box()
        return self

    @affects_family_data
    def reverse_points(self) -> Self:
        for mob in self.get_family():
            mob.data[:] = mob.data[::-1]
        return self

    @affects_family_data
    def apply_points_function(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        about_point: Vect3 | None = None,
        about_edge: Vect3 = ORIGIN,
        works_on_bounding_box: bool = False
    ) -> Self:
        if about_point is None and about_edge is not None:
            about_point = self.get_bounding_box_point(about_edge)

        for mob in self.get_family():
            arrs = []
            if mob.has_points():
                for key in mob.pointlike_data_keys:
                    arrs.append(mob.data[key])
            if works_on_bounding_box:
                arrs.append(mob.get_bounding_box())

            for arr in arrs:
                if about_point is None:
                    arr[:] = func(arr)
                else:
                    arr[:] = func(arr - about_point) + about_point

        if not works_on_bounding_box:
            self.refresh_bounding_box(recurse_down=True)
        else:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    # Others related to points

    def match_points(self, mobject: Mobject) -> Self:
        self.set_points(mobject.get_points())
        return self

    def get_points(self) -> Vect3Array:
        return self.data["point"]

    def clear_points(self) -> Self:
        self.resize_points(0)
        return self

    def get_num_points(self) -> int:
        return len(self.get_points())

    def get_all_points(self) -> Vect3Array:
        if self.submobjects:
            return np.vstack([sm.get_points() for sm in self.get_family()])
        else:
            return self.get_points()

    def has_points(self) -> bool:
        return len(self.get_points()) > 0

    def get_bounding_box(self) -> Vect3Array:
        if self.needs_new_bounding_box:
            self.bounding_box[:] = self.compute_bounding_box()
            self.needs_new_bounding_box = False
        return self.bounding_box

    def compute_bounding_box(self) -> Vect3Array:
        all_points = np.vstack([
            self.get_points(),
            *(
                mob.get_bounding_box()
                for mob in self.get_family()[1:]
                if mob.has_points()
            )
        ])
        if len(all_points) == 0:
            return np.zeros((3, self.dim))
        else:
            # Lower left and upper right corners
            mins = all_points.min(0)
            maxs = all_points.max(0)
            mids = (mins + maxs) / 2
            return np.array([mins, mids, maxs])

    def refresh_bounding_box(
        self,
        recurse_down: bool = False,
        recurse_up: bool = True
    ) -> Self:
        for mob in self.get_family(recurse_down):
            mob.needs_new_bounding_box = True
        if recurse_up:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    def are_points_touching(
        self,
        points: Vect3Array,
        buff: float = 0
    ) -> np.ndarray:
        bb = self.get_bounding_box()
        mins = (bb[0] - buff)
        maxs = (bb[2] + buff)
        return ((points >= mins) * (points <= maxs)).all(1)

    def is_point_touching(
        self,
        point: Vect3,
        buff: float = 0
    ) -> bool:
        return self.are_points_touching(np.array(point, ndmin=2), buff)[0]

    def is_touching(self, mobject: Mobject, buff: float = 1e-2) -> bool:
        bb1 = self.get_bounding_box()
        bb2 = mobject.get_bounding_box()
        return not any((
            (bb2[2] < bb1[0] - buff).any(),  # E.g. Right of mobject is left of self's left
            (bb2[0] > bb1[2] + buff).any(),  # E.g. Left of mobject is right of self's right
        ))

    # Family matters

    def __getitem__(self, value: int | slice) -> Self:
        if isinstance(value, slice):
            GroupClass = self.get_group_class()
            return GroupClass(*self.split().__getitem__(value))
        return self.split().__getitem__(value)

    def __iter__(self) -> Iterator[Self]:
        return iter(self.split())

    def __len__(self) -> int:
        return len(self.split())

    def split(self) -> list[Self]:
        return self.submobjects

    @affects_data
    def assemble_family(self) -> Self:
        sub_families = (sm.get_family() for sm in self.submobjects)
        self.family = [self, *it.chain(*sub_families)]
        self.refresh_has_updater_status()
        self.refresh_bounding_box()
        for parent in self.parents:
            parent.assemble_family()
        return self

    def get_family(self, recurse: bool = True) -> list[Self]:
        if recurse:
            return self.family
        else:
            return [self]

    def family_members_with_points(self) -> list[Self]:
        return [m for m in self.family if len(m.data) > 0]

    def get_ancestors(self, extended: bool = False) -> list[Mobject]:
        """
        Returns parents, grandparents, etc.
        Order of result should be from higher members of the hierarchy down.

        If extended is set to true, it includes the ancestors of all family members,
        e.g. any other parents of a submobject
        """
        ancestors = []
        to_process = list(self.get_family(recurse=extended))
        excluded = set(to_process)
        while to_process:
            for p in to_process.pop().parents:
                if p not in excluded:
                    ancestors.append(p)
                    to_process.append(p)
        # Ensure mobjects highest in the hierarchy show up first
        ancestors.reverse()
        # Remove list redundancies while preserving order
        return list(dict.fromkeys(ancestors))

    def add(self, *mobjects: Mobject) -> Self:
        if self in mobjects:
            raise Exception("Mobject cannot contain self")
        for mobject in mobjects:
            if mobject not in self.submobjects:
                self.submobjects.append(mobject)
            if self not in mobject.parents:
                mobject.parents.append(self)
        self.assemble_family()
        return self

    def remove(
        self,
        *to_remove: Mobject,
        reassemble: bool = True,
        recurse: bool = True
    ) -> Self:
        for parent in self.get_family(recurse):
            for child in to_remove:
                if child in parent.submobjects:
                    parent.submobjects.remove(child)
                if parent in child.parents:
                    child.parents.remove(parent)
            if reassemble:
                parent.assemble_family()
        return self

    def clear(self) -> Self:
        self.remove(*self.submobjects, recurse=False)
        return self

    def add_to_back(self, *mobjects: Mobject) -> Self:
        self.set_submobjects(list_update(mobjects, self.submobjects))
        return self

    def replace_submobject(self, index: int, new_submob: Mobject) -> Self:
        old_submob = self.submobjects[index]
        if self in old_submob.parents:
            old_submob.parents.remove(self)
        self.submobjects[index] = new_submob
        new_submob.parents.append(self)
        self.assemble_family()
        return self

    def insert_submobject(self, index: int, new_submob: Mobject) -> Self:
        self.submobjects.insert(index, new_submob)
        self.assemble_family()
        return self

    def set_submobjects(self, submobject_list: list[Mobject]) -> Self:
        if self.submobjects == submobject_list:
            return self
        self.clear()
        self.add(*submobject_list)
        return self

    def digest_mobject_attrs(self) -> Self:
        """
        Ensures all attributes which are mobjects are included
        in the submobjects list.
        """
        mobject_attrs = [x for x in list(self.__dict__.values()) if isinstance(x, Mobject)]
        self.set_submobjects(list_update(self.submobjects, mobject_attrs))
        return self

    # 子对象组织

    # 排列
    def arrange(
        self,
        # 子对象排列的方向，默认为右方向
        direction: Vect3 = RIGHT,
        # 是否居中排列，默认居中排列
        center: bool = True,
        # 其他参数
        **kwargs
    ) -> Self:
        # 遍历子对象
        for m1, m2 in zip(self.submobjects, self.submobjects[1:]):
            # 子对象的下一个子对象
            m2.next_to(m1, direction, **kwargs)
        # 如果需要居中
        if center:
            # 居中
            self.center()
        return self

    # 将子对象以网格形式排列
    def arrange_in_grid(
        self,
        # 网格的行数，默认为None
        n_rows: int | None = None,
        # 网格的列数，默认为None
        n_cols: int | None = None,
        # 子对象之间的间距，默认为None
        buff: float | None = None,
        # 子对象水平方向的间距，默认为None
        h_buff: float | None = None,
        # 子对象垂直方向的间距，默认为None
        v_buff: float | None = None,
        # 子对象之间的间距比例，默认为None
        buff_ratio: float | None = None,
        # 子对象水平方向的间距比例，默认为0.5
        h_buff_ratio: float = 0.5,
        # 子对象垂直方向的间距比例，默认为0.5
        v_buff_ratio: float = 0.5,
        # 子对象的对齐方向，默认为ORIGIN
        aligned_edge: Vect3 = ORIGIN,
        # 是否填充网格，默认为True，先填满行再填充列
        fill_rows_first: bool = True
    ) -> Self:

        submobs = self.submobjects
        if n_rows is None and n_cols is None:
            n_rows = int(np.sqrt(len(submobs)))
        if n_rows is None:
            n_rows = len(submobs) // n_cols
        if n_cols is None:
            n_cols = len(submobs) // n_rows

        if buff is not None:
            h_buff = buff
            v_buff = buff
        else:
            if buff_ratio is not None:
                v_buff_ratio = buff_ratio
                h_buff_ratio = buff_ratio
            if h_buff is None:
                h_buff = h_buff_ratio * self[0].get_width()
            if v_buff is None:
                v_buff = v_buff_ratio * self[0].get_height()

        x_unit = h_buff + max([sm.get_width() for sm in submobs])
        y_unit = v_buff + max([sm.get_height() for sm in submobs])

        for index, sm in enumerate(submobs):
            if fill_rows_first:
                x, y = index % n_cols, index // n_cols
            else:
                x, y = index // n_rows, index % n_rows
            sm.move_to(ORIGIN, aligned_edge)
            sm.shift(x * x_unit * RIGHT + y * y_unit * DOWN)
        self.center()
        return self

    def arrange_to_fit_dim(self, length: float, dim: int, about_edge=ORIGIN) -> Self:
        ref_point = self.get_bounding_box_point(about_edge)
        n_submobs = len(self.submobjects)
        if n_submobs <= 1:
            return
        total_length = sum(sm.length_over_dim(dim) for sm in self.submobjects)
        buff = (length - total_length) / (n_submobs - 1)
        vect = np.zeros(self.dim)
        vect[dim] = 1
        x = 0
        for submob in self.submobjects:
            submob.set_coord(x, dim, -vect)
            x += submob.length_over_dim(dim) + buff
        self.move_to(ref_point, about_edge)
        return self

    def arrange_to_fit_width(self, width: float, about_edge=ORIGIN) -> Self:
        return self.arrange_to_fit_dim(width, 0, about_edge)

    def arrange_to_fit_height(self, height: float, about_edge=ORIGIN) -> Self:
        return self.arrange_to_fit_dim(height, 1, about_edge)

    def arrange_to_fit_depth(self, depth: float, about_edge=ORIGIN) -> Self:
        return self.arrange_to_fit_dim(depth, 2, about_edge)

    def sort(
        self,
        point_to_num_func: Callable[[np.ndarray], float] = lambda p: p[0],
        submob_func: Callable[[Mobject]] | None = None
    ) -> Self:
        if submob_func is not None:
            self.submobjects.sort(key=submob_func)
        else:
            self.submobjects.sort(key=lambda m: point_to_num_func(m.get_center()))
        self.assemble_family()
        return self

    def shuffle(self, recurse: bool = False) -> Self:
        if recurse:
            for submob in self.submobjects:
                submob.shuffle(recurse=True)
        random.shuffle(self.submobjects)
        self.assemble_family()
        return self

    def reverse_submobjects(self) -> Self:
        self.submobjects.reverse()
        self.assemble_family()
        return self

    # Copying and serialization

    def stash_mobject_pointers(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            uncopied_attrs = ["parents", "target", "saved_state"]
            stash = dict()
            for attr in uncopied_attrs:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    stash[attr] = value
                    null_value = [] if isinstance(value, list) else None
                    setattr(self, attr, null_value)
            result = func(self, *args, **kwargs)
            self.__dict__.update(stash)
            return result
        return wrapper

    @stash_mobject_pointers
    def serialize(self) -> bytes:
        return pickle.dumps(self)

    def deserialize(self, data: bytes) -> Self:
        self.become(pickle.loads(data))
        return self

    def deepcopy(self) -> Self:
        result = copy.deepcopy(self)
        result._shaders_initialized = False
        result._data_has_changed = True
        return result

    def copy(self, deep: bool = False) -> Self:
        if deep:
            return self.deepcopy()

        result = copy.copy(self)

        result.parents = []
        result.target = None
        result.saved_state = None

        # copy.copy is only a shallow copy, so the internal
        # data which are numpy arrays or other mobjects still
        # need to be further copied.
        result.uniforms = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.uniforms.items()
        }

        # Instead of adding using result.add, which does some checks for updating
        # updater statues and bounding box, just directly modify the family-related
        # lists
        result.submobjects = [sm.copy() for sm in self.submobjects]
        for sm in result.submobjects:
            sm.parents = [result]
        result.family = [result, *it.chain(*(sm.get_family() for sm in result.submobjects))]

        # Similarly, instead of calling match_updaters, since we know the status
        # won't have changed, just directly match.
        result.non_time_updaters = list(self.non_time_updaters)
        result.time_based_updaters = list(self.time_based_updaters)
        result._data_has_changed = True
        result._shaders_initialized = False

        family = self.get_family()
        for attr, value in self.__dict__.items():
            if isinstance(value, Mobject) and value is not self:
                if value in family:
                    setattr(result, attr, result.family[self.family.index(value)])
            elif isinstance(value, np.ndarray):
                setattr(result, attr, value.copy())
        return result

    def generate_target(self, use_deepcopy: bool = False) -> Self:
        self.target = self.copy(deep=use_deepcopy)
        self.target.saved_state = self.saved_state
        return self.target

    def save_state(self, use_deepcopy: bool = False) -> Self:
        self.saved_state = self.copy(deep=use_deepcopy)
        self.saved_state.target = self.target
        return self

    def restore(self) -> Self:
        if not hasattr(self, "saved_state") or self.saved_state is None:
            raise Exception("Trying to restore without having saved")
        self.become(self.saved_state)
        return self

    def save_to_file(self, file_path: str) -> Self:
        with open(file_path, "wb") as fp:
            fp.write(self.serialize())
        log.info(f"Saved mobject to {file_path}")
        return self

    @staticmethod
    def load(file_path) -> Mobject:
        if not os.path.exists(file_path):
            log.error(f"No file found at {file_path}")
            sys.exit(2)
        with open(file_path, "rb") as fp:
            mobject = pickle.load(fp)
        return mobject

    def become(self, mobject: Mobject, match_updaters=False) -> Self:
        """
        Edit all data and submobjects to be idential
        to another mobject
        """
        self.align_family(mobject)
        family1 = self.get_family()
        family2 = mobject.get_family()
        for sm1, sm2 in zip(family1, family2):
            sm1.set_data(sm2.data)
            sm1.set_uniforms(sm2.uniforms)
            sm1.bounding_box[:] = sm2.bounding_box
            sm1.shader_folder = sm2.shader_folder
            sm1.texture_paths = sm2.texture_paths
            sm1.depth_test = sm2.depth_test
            sm1.render_primitive = sm2.render_primitive
            sm1.needs_new_bounding_box = sm2.needs_new_bounding_box
        # Make sure named family members carry over
        for attr, value in list(mobject.__dict__.items()):
            if isinstance(value, Mobject) and value in family2:
                setattr(self, attr, family1[family2.index(value)])
        if match_updaters:
            self.match_updaters(mobject)
        return self

    def looks_identical(self, mobject: Mobject) -> bool:
        fam1 = self.family_members_with_points()
        fam2 = mobject.family_members_with_points()
        if len(fam1) != len(fam2):
            return False
        for m1, m2 in zip(fam1, fam2):
            if m1.get_num_points() != m2.get_num_points():
                return False
            if not m1.data.dtype == m2.data.dtype:
                return False
            for key in m1.data.dtype.names:
                if not np.isclose(m1.data[key], m2.data[key]).all():
                    return False
            if set(m1.uniforms).difference(m2.uniforms):
                return False
            for key in m1.uniforms:
                value1 = m1.uniforms[key]
                value2 = m2.uniforms[key]
                if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray) and not value1.size == value2.size:
                    return False
                if not np.isclose(value1, value2).all():
                    return False
        return True

    def has_same_shape_as(self, mobject: Mobject) -> bool:
        # Normalize both point sets by centering and making height 1
        points1, points2 = (
            (m.get_all_points() - m.get_center()) / m.get_height()
            for m in (self, mobject)
        )
        if len(points1) != len(points2):
            return False
        return bool(np.isclose(points1, points2, atol=self.get_width() * 1e-2).all())

    # Creating new Mobjects from this one

    def replicate(self, n: int) -> Self:
        group_class = self.get_group_class()
        return group_class(*(self.copy() for _ in range(n)))

    def get_grid(
        self,
        n_rows: int,
        n_cols: int,
        height: float | None = None,
        width: float | None = None,
        group_by_rows: bool = False,
        group_by_cols: bool = False,
        **kwargs
    ) -> Self:
        """
        Returns a new mobject containing multiple copies of this one
        arranged in a grid
        """
        total = n_rows * n_cols
        grid = self.replicate(total)
        if group_by_cols:
            kwargs["fill_rows_first"] = False
        grid.arrange_in_grid(n_rows, n_cols, **kwargs)
        if height is not None:
            grid.set_height(height)
        if width is not None:
            grid.set_height(width)

        group_class = self.get_group_class()
        if group_by_rows:
            return group_class(*(grid[n:n + n_cols] for n in range(0, total, n_cols)))
        elif group_by_cols:
            return group_class(*(grid[n:n + n_rows] for n in range(0, total, n_rows)))
        else:
            return grid

    # Updating

    def init_updaters(self):
        self.time_based_updaters: list[TimeBasedUpdater] = []
        self.non_time_updaters: list[NonTimeUpdater] = []
        self.has_updaters: bool = False
        self.updating_suspended: bool = False

    def update(self, dt: float = 0, recurse: bool = True) -> Self:
        if not self.has_updaters or self.updating_suspended:
            return self
        if recurse:
            for submob in self.submobjects:
                submob.update(dt, recurse)
        for updater in self.time_based_updaters:
            updater(self, dt)
        for updater in self.non_time_updaters:
            updater(self)
        return self

    def get_time_based_updaters(self) -> list[TimeBasedUpdater]:
        return self.time_based_updaters

    def has_time_based_updater(self) -> bool:
        return len(self.time_based_updaters) > 0

    def get_updaters(self) -> list[Updater]:
        return self.time_based_updaters + self.non_time_updaters

    def get_family_updaters(self) -> list[Updater]:
        return list(it.chain(*[sm.get_updaters() for sm in self.get_family()]))

    def add_updater(
        self,
        update_function: Updater,
        index: int | None = None,
        call_updater: bool = True
    ) -> Self:
        if "dt" in get_parameters(update_function):
            updater_list = self.time_based_updaters
        else:
            updater_list = self.non_time_updaters

        if index is None:
            updater_list.append(update_function)
        else:
            updater_list.insert(index, update_function)

        self.refresh_has_updater_status()
        for parent in self.parents:
            parent.has_updaters = True
        if call_updater:
            self.update(dt=0)
        return self

    def remove_updater(self, update_function: Updater) -> Self:
        for updater_list in [self.time_based_updaters, self.non_time_updaters]:
            while update_function in updater_list:
                updater_list.remove(update_function)
        self.refresh_has_updater_status()
        return self

    def clear_updaters(self, recurse: bool = True) -> Self:
        self.time_based_updaters = []
        self.non_time_updaters = []
        if recurse:
            for submob in self.submobjects:
                submob.clear_updaters()
        self.refresh_has_updater_status()
        return self

    def match_updaters(self, mobject: Mobject) -> Self:
        self.clear_updaters()
        for updater in mobject.get_updaters():
            self.add_updater(updater)
        return self

    def suspend_updating(self, recurse: bool = True) -> Self:
        self.updating_suspended = True
        if recurse:
            for submob in self.submobjects:
                submob.suspend_updating(recurse)
        return self

    def resume_updating(self, recurse: bool = True, call_updater: bool = True) -> Self:
        self.updating_suspended = False
        if recurse:
            for submob in self.submobjects:
                submob.resume_updating(recurse)
        for parent in self.parents:
            parent.resume_updating(recurse=False, call_updater=False)
        if call_updater:
            self.update(dt=0, recurse=recurse)
        return self

    def refresh_has_updater_status(self) -> Self:
        self.has_updaters = any(mob.get_updaters() for mob in self.get_family())
        return self

    # Check if mark as static or not for camera

    def is_changing(self) -> bool:
        return self._is_animating or self.has_updaters

    def set_animating_status(self, is_animating: bool, recurse: bool = True) -> Self:
        for mob in (*self.get_family(recurse), *self.get_ancestors()):
            mob._is_animating = is_animating
        return self

    # Transforming operations

    def shift(self, vector: Vect3) -> Self:
        self.apply_points_function(
            lambda points: points + vector,
            about_edge=None,
            works_on_bounding_box=True,
        )
        return self

    def scale(
        self,
        scale_factor: float | npt.ArrayLike,
        min_scale_factor: float = 1e-8,
        about_point: Vect3 | None = None,
        about_edge: Vect3 = ORIGIN
    ) -> Self:
        """
        Default behavior is to scale about the center of the mobject.
        The argument about_edge can be a vector, indicating which side of
        the mobject to scale about, e.g., mob.scale(about_edge = RIGHT)
        scales about mob.get_right().

        Otherwise, if about_point is given a value, scaling is done with
        respect to that point.
        """
        if isinstance(scale_factor, numbers.Number):
            scale_factor = max(scale_factor, min_scale_factor)
        else:
            scale_factor = np.array(scale_factor).clip(min=min_scale_factor)
        self.apply_points_function(
            lambda points: scale_factor * points,
            about_point=about_point,
            about_edge=about_edge,
            works_on_bounding_box=True,
        )
        for mob in self.get_family():
            mob._handle_scale_side_effects(scale_factor)
        return self

    def _handle_scale_side_effects(self, scale_factor):
        # In case subclasses, such as DecimalNumber, need to make
        # any other changes when the size gets altered
        pass

    def stretch(self, factor: float, dim: int, **kwargs) -> Self:
        def func(points):
            points[:, dim] *= factor
            return points
        self.apply_points_function(func, works_on_bounding_box=True, **kwargs)
        return self

    def rotate_about_origin(self, angle: float, axis: Vect3 = OUT) -> Self:
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(
        self,
        angle: float,
        axis: Vect3 = OUT,
        about_point: Vect3 | None = None,
        **kwargs
    ) -> Self:
        rot_matrix_T = rotation_matrix_transpose(angle, axis)
        self.apply_points_function(
            lambda points: np.dot(points, rot_matrix_T),
            about_point,
            **kwargs
        )
        return self

    def flip(self, axis: Vect3 = UP, **kwargs) -> Self:
        return self.rotate(TAU / 2, axis, **kwargs)

    def apply_function(self, function: Callable[[np.ndarray], np.ndarray], **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if len(kwargs) == 0:
            kwargs["about_point"] = ORIGIN
        self.apply_points_function(
            lambda points: np.array([function(p) for p in points]),
            **kwargs
        )
        return self

    def apply_function_to_position(self, function: Callable[[np.ndarray], np.ndarray]) -> Self:
        self.move_to(function(self.get_center()))
        return self

    def apply_function_to_submobject_positions(
        self,
        function: Callable[[np.ndarray], np.ndarray]
    ) -> Self:
        for submob in self.submobjects:
            submob.apply_function_to_position(function)
        return self

    def apply_matrix(self, matrix: npt.ArrayLike, **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if ("about_point" not in kwargs) and ("about_edge" not in kwargs):
            kwargs["about_point"] = ORIGIN
        full_matrix = np.identity(self.dim)
        matrix = np.array(matrix)
        full_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        self.apply_points_function(
            lambda points: np.dot(points, full_matrix.T),
            **kwargs
        )
        return self

    def apply_complex_function(self, function: Callable[[complex], complex], **kwargs) -> Self:
        def R3_func(point):
            x, y, z = point
            xy_complex = function(complex(x, y))
            return [
                xy_complex.real,
                xy_complex.imag,
                z
            ]
        return self.apply_function(R3_func, **kwargs)

    def wag(
        self,
        direction: Vect3 = RIGHT,
        axis: Vect3 = DOWN,
        wag_factor: float = 1.0
    ) -> Self:
        for mob in self.family_members_with_points():
            alphas = np.dot(mob.get_points(), np.transpose(axis))
            alphas -= min(alphas)
            alphas /= max(alphas)
            alphas = alphas**wag_factor
            mob.set_points(mob.get_points() + np.dot(
                alphas.reshape((len(alphas), 1)),
                np.array(direction).reshape((1, mob.dim))
            ))
        return self

    # 方向与定位

    # 将物体移动到场景的中心位置
    def center(self) -> Self:
        # 获取物体的中心位置，将物体向其相反方向平移，使其移动到场景的中心位置
        self.shift(-self.get_center())
        return self

    # 将物体沿着指定方向对齐到边界
    def align_on_border(
        self,
        # 指定对齐的方向向量
        direction: Vect3,
        # 指定对齐时的间距，默认采用默认间距
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER
    ) -> Self:
        # “方向”只需要是一个指向二维平面上某一侧面或角落的向量即可
        # 计算目标点
        target_point = np.sign(direction) * (FRAME_X_RADIUS, FRAME_Y_RADIUS, 0)
        # 获取物体边界框上指定方向上的点
        point_to_align = self.get_bounding_box_point(direction)
        # 计算需要平移的距离
        shift_val = target_point - point_to_align - buff * np.array(direction)
        # 根据direction的符号调整平移向量的方向
        shift_val = shift_val * abs(np.sign(direction))
        # 将物体按计算得到的平移向量移动
        self.shift(shift_val)
        return self

    # 将物体对齐到指定角落
    def to_corner(
        self,
        # 指定要对齐到的角落，默认为左下角
        corner: Vect3 = LEFT + DOWN,
        # 间距，默认为默认对象间距离
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER
    ) -> Self:
        return self.align_on_border(corner, buff)

    # 将物体对齐到指定边缘
    def to_edge(
        self,
        # 指定要对齐到的边缘，默认为左边缘
        edge: Vect3 = LEFT,
        # 指定对齐时的间距，默认为默认对象间距离
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER
    ) -> Self:
        return self.align_on_border(edge, buff)

    # 将对象移动到另一个对象
    def next_to(
        self,
        # 移动到的对象，一个对象或者一个点
        mobject_or_point: Mobject | Vect3,
        # 移动方向，默认向右
        direction: Vect3 = RIGHT,
        # 间距，采用默认对象间距离
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        # 对齐的边，默认为ORIGIN
        aligned_edge: Vect3 = ORIGIN,
        # 传入一个子对象，默认为None
        submobject_to_align: Mobject | None = None,
        # 传入一个子对象的索引，默认为None
        index_of_submobject_to_align: int | slice | None = None,
        # 传入一个坐标掩码，默认为np.array([1, 1, 1])
        coor_mask: Vect3 = np.array([1, 1, 1]),
    ) -> Self:
        # 如果移动到的对象是Mobject对象
        if isinstance(mobject_or_point, Mobject):
            # 获取移动到的对象
            mob = mobject_or_point
            # 如果对齐子对象的索引不为空
            if index_of_submobject_to_align is not None:
                # 则获取目标对齐器
                target_aligner = mob[index_of_submobject_to_align]
            else:
                # 如果未提供索引，则使用整个Mobject作为目标对齐器
                target_aligner = mob
            # 根据对齐边缘和方向，获取目标对齐点
            target_point = target_aligner.get_bounding_box_point(aligned_edge + direction)
        # 如果移动到的对象是坐标点，则直接将其作为目标点
        else:
            target_point = mobject_or_point
        # 确定对齐器
        if submobject_to_align is not None:
            # 如果提供了待对齐的子对象，则使用其作为对齐器
            aligner = submobject_to_align
        elif index_of_submobject_to_align is not None:
            # 如果未提供待对齐的子对象但提供了索引，则使用相应子对象作为对齐器
            aligner = self[index_of_submobject_to_align]
        else:
            # 如果既未提供待对齐的子对象也未提供索引，则使用整个Mobject作为对齐器
            aligner = self
        # 获取待对齐点，根据对齐边缘和方向对齐器进行计算得到
        point_to_align = aligner.get_bounding_box_point(aligned_edge - direction)
        # 移动Mobject，使目标点与待对齐点对齐，并考虑缓冲距离（buff * direction）
        self.shift((target_point - point_to_align + buff * direction) * coor_mask)
        return self

    # 将物体移动到屏幕内部
    def shift_onto_screen(self, **kwargs) -> Self:
        # 定义了场景的长度和宽度
        space_lengths = [FRAME_X_RADIUS, FRAME_Y_RADIUS]
        # 遍历上、下、左、右四个方向
        for vect in UP, DOWN, LEFT, RIGHT:
            # 获取向量中绝对值最大的分量所在的维度
            dim = np.argmax(np.abs(vect))
            # 获取缓冲区间距，如果未指定则使用默认值
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_EDGE_BUFFER)
            # 计算在当前维度上可以移动的最大值
            max_val = space_lengths[dim] - buff
            # 获取沿着当前方向的边缘中心点
            edge_center = self.get_edge_center(vect)
            # 检查当前边缘中心点是否超出了屏幕范围
            if np.dot(edge_center, vect) > max_val:
                # 如果超出了屏幕范围，则将物体对齐到当前方向的边缘
                self.to_edge(vect, **kwargs)
        return self

    # 判断物体是否在超出屏幕范围
    def is_off_screen(self) -> bool:
        # 如果物体最左边的点的x坐标大于FRAME_X_RADIUS，则超出
        if self.get_left()[0] > FRAME_X_RADIUS:
            return True
        # 如果物体最右边的点的x坐标小于-FRAME_X_RADIUS，则超出
        if self.get_right()[0] < -FRAME_X_RADIUS:
            return True
        # 如果物体最下边的点的x坐标大于FRAME_Y_RADIUS，则超出
        if self.get_bottom()[1] > FRAME_Y_RADIUS:
            return True
        # 如果物体最上边的点的x坐标小于-FRAME_Y_RADIUS，则超出
        if self.get_top()[1] < -FRAME_Y_RADIUS:
            return True
        # 均不超出
        return False

    # 围绕指定点按照指定因子进行拉伸
    def stretch_about_point(self, 
        # 拉伸因子
        factor: float, 
        # 拉伸的维度
        dim: int, 
        # 围绕的点
        point: Vect3
        ) -> Self:
        return self.stretch(factor, dim, about_point=point)

    # 在原地按照指定因子进行拉伸
    def stretch_in_place(self, 
        factor: float, 
        dim: int
        ) -> Self:
        # 目前已经被stretch方法取代
        return self.stretch(factor, dim)

    # 将物体按照指定维度的长度重新缩放到指定长度
    def rescale_to_fit(self, 
        length: float, 
        dim: int, 
        stretch: bool = False, 
        **kwargs
        ) -> Self:
        old_length = self.length_over_dim(dim)
        if old_length == 0:
            return self
        if stretch:
            self.stretch(length / old_length, dim, **kwargs)
        else:
            self.scale(length / old_length, **kwargs)
        return self

    def stretch_to_fit_width(self, width: float, **kwargs) -> Self:
        return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

    def stretch_to_fit_height(self, height: float, **kwargs) -> Self:
        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

    def stretch_to_fit_depth(self, depth: float, **kwargs) -> Self:
        return self.rescale_to_fit(depth, 2, stretch=True, **kwargs)

    def set_width(self, width: float, stretch: bool = False, **kwargs) -> Self:
        return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

    def set_height(self, height: float, stretch: bool = False, **kwargs) -> Self:
        return self.rescale_to_fit(height, 1, stretch=stretch, **kwargs)

    def set_depth(self, depth: float, stretch: bool = False, **kwargs) -> Self:
        return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

    def set_max_width(self, max_width: float, **kwargs) -> Self:
        if self.get_width() > max_width:
            self.set_width(max_width, **kwargs)
        return self

    def set_max_height(self, max_height: float, **kwargs) -> Self:
        if self.get_height() > max_height:
            self.set_height(max_height, **kwargs)
        return self

    def set_max_depth(self, max_depth: float, **kwargs) -> Self:
        if self.get_depth() > max_depth:
            self.set_depth(max_depth, **kwargs)
        return self

    def set_min_width(self, min_width: float, **kwargs) -> Self:
        if self.get_width() < min_width:
            self.set_width(min_width, **kwargs)
        return self

    def set_min_height(self, min_height: float, **kwargs) -> Self:
        if self.get_height() < min_height:
            self.set_height(min_height, **kwargs)
        return self

    def set_min_depth(self, min_depth: float, **kwargs) -> Self:
        if self.get_depth() < min_depth:
            self.set_depth(min_depth, **kwargs)
        return self

    def set_shape(
        self,
        width: Optional[float] = None,
        height: Optional[float] = None,
        depth: Optional[float] = None,
        **kwargs
    ) -> Self:
        if width is not None:
            self.set_width(width, stretch=True, **kwargs)
        if height is not None:
            self.set_height(height, stretch=True, **kwargs)
        if depth is not None:
            self.set_depth(depth, stretch=True, **kwargs)
        return self

    def set_coord(self, value: float, dim: int, direction: Vect3 = ORIGIN) -> Self:
        curr = self.get_coord(dim, direction)
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        self.shift(shift_vect)
        return self

    def set_x(self, x: float, direction: Vect3 = ORIGIN) -> Self:
        return self.set_coord(x, 0, direction)

    def set_y(self, y: float, direction: Vect3 = ORIGIN) -> Self:
        return self.set_coord(y, 1, direction)

    def set_z(self, z: float, direction: Vect3 = ORIGIN) -> Self:
        return self.set_coord(z, 2, direction)

    def space_out_submobjects(self, factor: float = 1.5, **kwargs) -> Self:
        self.scale(factor, **kwargs)
        for submob in self.submobjects:
            submob.scale(1. / factor)
        return self

    def move_to(
        self,
        point_or_mobject: Mobject | Vect3,
        aligned_edge: Vect3 = ORIGIN,
        coor_mask: Vect3 = np.array([1, 1, 1])
    ) -> Self:
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_bounding_box_point(aligned_edge)
        else:
            target = point_or_mobject
        point_to_align = self.get_bounding_box_point(aligned_edge)
        self.shift((target - point_to_align) * coor_mask)
        return self

    def replace(self, mobject: Mobject, dim_to_match: int = 0, stretch: bool = False) -> Self:
        if not mobject.get_num_points() and not mobject.submobjects:
            self.scale(0)
            return self
        if stretch:
            for i in range(self.dim):
                self.rescale_to_fit(mobject.length_over_dim(i), i, stretch=True)
        else:
            self.rescale_to_fit(
                mobject.length_over_dim(dim_to_match),
                dim_to_match,
                stretch=False
            )
        self.shift(mobject.get_center() - self.get_center())
        return self

    def surround(
        self,
        mobject: Mobject,
        dim_to_match: int = 0,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF
    ) -> Self:
        self.replace(mobject, dim_to_match, stretch)
        length = mobject.length_over_dim(dim_to_match)
        self.scale((length + buff) / length)
        return self

    def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
        curr_start, curr_end = self.get_start_and_end()
        curr_vect = curr_end - curr_start
        if np.all(curr_vect == 0):
            raise Exception("Cannot position endpoints of closed loop")
        target_vect = end - start
        self.scale(
            get_norm(target_vect) / get_norm(curr_vect),
            about_point=curr_start,
        )
        self.rotate(
            angle_of_vector(target_vect) - angle_of_vector(curr_vect),
        )
        self.rotate(
            np.arctan2(curr_vect[2], get_norm(curr_vect[:2])) - np.arctan2(target_vect[2], get_norm(target_vect[:2])),
            axis=np.array([-target_vect[1], target_vect[0], 0]),
        )
        self.shift(start - self.get_start())
        return self

    # Color functions

    @affects_family_data
    def set_rgba_array(
        self,
        rgba_array: npt.ArrayLike,
        name: str = "rgba",
        recurse: bool = False
    ) -> Self:
        for mob in self.get_family(recurse):
            data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
            data[name][:] = rgba_array
        return self

    def set_color_by_rgba_func(
        self,
        func: Callable[[Vect3], Vect4],
        recurse: bool = True
    ) -> Self:
        """
        Func should take in a point in R3 and output an rgba value
        """
        for mob in self.get_family(recurse):
            rgba_array = [func(point) for point in mob.get_points()]
            mob.set_rgba_array(rgba_array)
        return self

    def set_color_by_rgb_func(
        self,
        func: Callable[[Vect3], Vect3],
        opacity: float = 1,
        recurse: bool = True
    ) -> Self:
        """
        Func should take in a point in R3 and output an rgb value
        """
        for mob in self.get_family(recurse):
            rgba_array = [[*func(point), opacity] for point in mob.get_points()]
            mob.set_rgba_array(rgba_array)
        return self

    @affects_family_data
    def set_rgba_array_by_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None = None,
        opacity: float | Iterable[float] | None = None,
        name: str = "rgba",
        recurse: bool = True
    ) -> Self:
        for mob in self.get_family(recurse):
            data = mob.data if mob.has_points() > 0 else mob._data_defaults
            if color is not None:
                rgbs = np.array(list(map(color_to_rgb, listify(color))))
                if 1 < len(rgbs):
                    rgbs = resize_with_interpolation(rgbs, len(data))
                data[name][:, :3] = rgbs
            if opacity is not None:
                if not isinstance(opacity, (float, int)):
                    opacity = resize_with_interpolation(np.array(opacity), len(data))
                data[name][:, 3] = opacity
        return self

    def set_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None,
        opacity: float | Iterable[float] | None = None,
        recurse: bool = True
    ) -> Self:
        self.set_rgba_array_by_color(color, opacity, recurse=False)
        # Recurse to submobjects differently from how set_rgba_array_by_color
        # in case they implement set_color differently
        if recurse:
            for submob in self.submobjects:
                submob.set_color(color, recurse=True)
        return self

    def set_opacity(
        self,
        opacity: float | Iterable[float] | None,
        recurse: bool = True
    ) -> Self:
        self.set_rgba_array_by_color(color=None, opacity=opacity, recurse=False)
        if recurse:
            for submob in self.submobjects:
                submob.set_opacity(opacity, recurse=True)
        return self

    def get_color(self) -> str:
        return rgb_to_hex(self.data["rgba"][0, :3])

    def get_opacity(self) -> float:
        return self.data["rgba"][0, 3]

    def set_color_by_gradient(self, *colors: ManimColor) -> Self:
        if self.has_points():
            self.set_color(colors)
        else:
            self.set_submobject_colors_by_gradient(*colors)
        return self

    def set_submobject_colors_by_gradient(self, *colors: ManimColor) -> Self:
        if len(colors) == 0:
            raise Exception("Need at least one color")
        elif len(colors) == 1:
            return self.set_color(*colors)

        # mobs = self.family_members_with_points()
        mobs = self.submobjects
        new_colors = color_gradient(colors, len(mobs))

        for mob, color in zip(mobs, new_colors):
            mob.set_color(color)
        return self

    def fade(self, darkness: float = 0.5, recurse: bool = True) -> Self:
        self.set_opacity(1.0 - darkness, recurse=recurse)

    def get_shading(self) -> np.ndarray:
        return self.uniforms["shading"]

    def set_shading(
        self,
        reflectiveness: float | None = None,
        gloss: float | None = None,
        shadow: float | None = None,
        recurse: bool = True
    ) -> Self:
        """
        Larger reflectiveness makes things brighter when facing the light
        Larger shadow makes faces opposite the light darker
        Makes parts bright where light gets reflected toward the camera
        """
        for mob in self.get_family(recurse):
            for i, value in enumerate([reflectiveness, gloss, shadow]):
                if value is not None:
                    mob.uniforms["shading"][i] = value
        return self

    def get_reflectiveness(self) -> float:
        return self.get_shading()[0]

    def get_gloss(self) -> float:
        return self.get_shading()[1]

    def get_shadow(self) -> float:
        return self.get_shading()[2]

    def set_reflectiveness(self, reflectiveness: float, recurse: bool = True) -> Self:
        self.set_shading(reflectiveness=reflectiveness, recurse=recurse)
        return self

    def set_gloss(self, gloss: float, recurse: bool = True) -> Self:
        self.set_shading(gloss=gloss, recurse=recurse)
        return self

    def set_shadow(self, shadow: float, recurse: bool = True) -> Self:
        self.set_shading(shadow=shadow, recurse=recurse)
        return self

    # Background rectangle

    def add_background_rectangle(
        self,
        color: ManimColor | None = None,
        opacity: float = 1.0,
        **kwargs
    ) -> Self:
        from manimlib.mobject.shape_matchers import BackgroundRectangle
        self.background_rectangle = BackgroundRectangle(
            self, color=color,
            fill_opacity=opacity,
            **kwargs
        )
        self.add_to_back(self.background_rectangle)
        return self

    def add_background_rectangle_to_submobjects(self, **kwargs) -> Self:
        for submobject in self.submobjects:
            submobject.add_background_rectangle(**kwargs)
        return self

    def add_background_rectangle_to_family_members_with_points(self, **kwargs) -> Self:
        for mob in self.family_members_with_points():
            mob.add_background_rectangle(**kwargs)
        return self

    # Getters

    def get_bounding_box_point(self, direction: Vect3) -> Vect3:
        bb = self.get_bounding_box()
        indices = (np.sign(direction) + 1).astype(int)
        return np.array([
            bb[indices[i]][i]
            for i in range(3)
        ])

    def get_edge_center(self, direction: Vect3) -> Vect3:
        return self.get_bounding_box_point(direction)

    def get_corner(self, direction: Vect3) -> Vect3:
        return self.get_bounding_box_point(direction)

    def get_all_corners(self):
        bb = self.get_bounding_box()
        return np.array([
            [bb[indices[-i + 1]][i] for i in range(3)]
            for indices in it.product([0, 2], repeat=3)
        ])

    def get_center(self) -> Vect3:
        return self.get_bounding_box()[1]

    def get_center_of_mass(self) -> Vect3:
        return self.get_all_points().mean(0)

    def get_boundary_point(self, direction: Vect3) -> Vect3:
        all_points = self.get_all_points()
        boundary_directions = all_points - self.get_center()
        norms = np.linalg.norm(boundary_directions, axis=1)
        boundary_directions /= np.repeat(norms, 3).reshape((len(norms), 3))
        index = np.argmax(np.dot(boundary_directions, np.array(direction).T))
        return all_points[index]

    def get_continuous_bounding_box_point(self, direction: Vect3) -> Vect3:
        dl, center, ur = self.get_bounding_box()
        corner_vect = (ur - center)
        return center + direction / np.max(np.abs(np.true_divide(
            direction, corner_vect,
            out=np.zeros(len(direction)),
            where=((corner_vect) != 0)
        )))

    def get_top(self) -> Vect3:
        return self.get_edge_center(UP)

    def get_bottom(self) -> Vect3:
        return self.get_edge_center(DOWN)

    def get_right(self) -> Vect3:
        return self.get_edge_center(RIGHT)

    def get_left(self) -> Vect3:
        return self.get_edge_center(LEFT)

    def get_zenith(self) -> Vect3:
        return self.get_edge_center(OUT)

    def get_nadir(self) -> Vect3:
        return self.get_edge_center(IN)

    def length_over_dim(self, dim: int) -> float:
        bb = self.get_bounding_box()
        return abs((bb[2] - bb[0])[dim])

    def get_width(self) -> float:
        return self.length_over_dim(0)

    def get_height(self) -> float:
        return self.length_over_dim(1)

    def get_depth(self) -> float:
        return self.length_over_dim(2)

    def get_shape(self) -> Tuple[float]:
        return tuple(self.length_over_dim(dim) for dim in range(3))

    def get_coord(self, dim: int, direction: Vect3 = ORIGIN) -> float:
        """
        Meant to generalize get_x, get_y, get_z
        """
        return self.get_bounding_box_point(direction)[dim]

    def get_x(self, direction=ORIGIN) -> float:
        return self.get_coord(0, direction)

    def get_y(self, direction=ORIGIN) -> float:
        return self.get_coord(1, direction)

    def get_z(self, direction=ORIGIN) -> float:
        return self.get_coord(2, direction)

    def get_start(self) -> Vect3:
        self.throw_error_if_no_points()
        return self.get_points()[0].copy()

    def get_end(self) -> Vect3:
        self.throw_error_if_no_points()
        return self.get_points()[-1].copy()

    def get_start_and_end(self) -> tuple[Vect3, Vect3]:
        self.throw_error_if_no_points()
        points = self.get_points()
        return (points[0].copy(), points[-1].copy())

    def point_from_proportion(self, alpha: float) -> Vect3:
        points = self.get_points()
        i, subalpha = integer_interpolate(0, len(points) - 1, alpha)
        return interpolate(points[i], points[i + 1], subalpha)

    def pfp(self, alpha):
        """Abbreviation for point_from_proportion"""
        return self.point_from_proportion(alpha)

    def get_pieces(self, n_pieces: int) -> Group:
        template = self.copy()
        template.set_submobjects([])
        alphas = np.linspace(0, 1, n_pieces + 1)
        return Group(*[
            template.copy().pointwise_become_partial(
                self, a1, a2
            )
            for a1, a2 in zip(alphas[:-1], alphas[1:])
        ])

    def get_z_index_reference_point(self) -> Vect3:
        # TODO, better place to define default z_index_group?
        z_index_group = getattr(self, "z_index_group", self)
        return z_index_group.get_center()

    # Match other mobject properties

    def match_color(self, mobject: Mobject) -> Self:
        return self.set_color(mobject.get_color())

    def match_style(self, mobject: Mobject) -> Self:
        self.set_color(mobject.get_color())
        self.set_opacity(mobject.get_opacity())
        self.set_shading(*mobject.get_shading())
        return self

    def match_dim_size(self, mobject: Mobject, dim: int, **kwargs) -> Self:
        return self.rescale_to_fit(
            mobject.length_over_dim(dim), dim,
            **kwargs
        )

    def match_width(self, mobject: Mobject, **kwargs) -> Self:
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_height(self, mobject: Mobject, **kwargs) -> Self:
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_depth(self, mobject: Mobject, **kwargs) -> Self:
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_coord(
        self,
        mobject_or_point: Mobject | Vect3,
        dim: int,
        direction: Vect3 = ORIGIN
    ) -> Self:
        if isinstance(mobject_or_point, Mobject):
            coord = mobject_or_point.get_coord(dim, direction)
        else:
            coord = mobject_or_point[dim]
        return self.set_coord(coord, dim=dim, direction=direction)

    def match_x(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        return self.match_coord(mobject_or_point, 0, direction)

    def match_y(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        return self.match_coord(mobject_or_point, 1, direction)

    def match_z(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        return self.match_coord(mobject_or_point, 2, direction)

    def align_to(
        self,
        mobject_or_point: Mobject | Vect3,
        direction: Vect3 = ORIGIN
    ) -> Self:
        """
        Examples:
        mob1.align_to(mob2, UP) moves mob1 vertically so that its
        top edge lines ups with mob2's top edge.

        mob1.align_to(mob2, alignment_vect = RIGHT) moves mob1
        horizontally so that it's center is directly above/below
        the center of mob2
        """
        if isinstance(mobject_or_point, Mobject):
            point = mobject_or_point.get_bounding_box_point(direction)
        else:
            point = mobject_or_point

        for dim in range(self.dim):
            if direction[dim] != 0:
                self.set_coord(point[dim], dim, direction)
        return self

    def get_group_class(self):
        return Group

    # Alignment

    def is_aligned_with(self, mobject: Mobject) -> bool:
        if len(self.data) != len(mobject.data):
            return False
        if len(self.submobjects) != len(mobject.submobjects):
            return False
        return all(
            sm1.is_aligned_with(sm2)
            for sm1, sm2 in zip(self.submobjects, mobject.submobjects)
        )

    def align_data_and_family(self, mobject: Mobject) -> Self:
        self.align_family(mobject)
        self.align_data(mobject)
        return self

    def align_data(self, mobject: Mobject) -> Self:
        for mob1, mob2 in zip(self.get_family(), mobject.get_family()):
            mob1.align_points(mob2)
        return self

    def align_points(self, mobject: Mobject) -> Self:
        max_len = max(self.get_num_points(), mobject.get_num_points())
        for mob in (self, mobject):
            mob.resize_points(max_len, resize_func=resize_preserving_order)
        return self

    def align_family(self, mobject: Mobject) -> Self:
        mob1 = self
        mob2 = mobject
        n1 = len(mob1)
        n2 = len(mob2)
        if n1 != n2:
            mob1.add_n_more_submobjects(max(0, n2 - n1))
            mob2.add_n_more_submobjects(max(0, n1 - n2))
        # Recurse
        for sm1, sm2 in zip(mob1.submobjects, mob2.submobjects):
            sm1.align_family(sm2)
        return self

    def push_self_into_submobjects(self) -> Self:
        copy = self.copy()
        copy.set_submobjects([])
        self.resize_points(0)
        self.add(copy)
        return self

    def add_n_more_submobjects(self, n: int) -> Self:
        if n == 0:
            return self

        curr = len(self.submobjects)
        if curr == 0:
            # If empty, simply add n point mobjects
            null_mob = self.copy()
            null_mob.set_points([self.get_center()])
            self.set_submobjects([
                null_mob.copy()
                for k in range(n)
            ])
            return self
        target = curr + n
        repeat_indices = (np.arange(target) * curr) // target
        split_factors = [
            (repeat_indices == i).sum()
            for i in range(curr)
        ]
        new_submobs = []
        for submob, sf in zip(self.submobjects, split_factors):
            new_submobs.append(submob)
            for k in range(1, sf):
                new_submobs.append(submob.invisible_copy())
        self.set_submobjects(new_submobs)
        return self

    def invisible_copy(self) -> Self:
        return self.copy().set_opacity(0)

    # Interpolate

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = straight_path
    ) -> Self:
        keys = [k for k in self.data.dtype.names if k not in self.locked_data_keys]
        if keys:
            self.note_changed_data()
        for key in keys:
            func = path_func if key in self.pointlike_data_keys else interpolate
            md1 = mobject1.data[key]
            md2 = mobject2.data[key]
            if key in self.const_data_keys:
                md1 = md1[0]
                md2 = md2[0]
            self.data[key] = func(md1, md2, alpha)

        keys = [k for k in self.uniforms if k not in self.locked_uniform_keys]
        for key in keys:
            if key not in mobject1.uniforms or key not in mobject2.uniforms:
                continue
            self.uniforms[key] = interpolate(
                mobject1.uniforms[key],
                mobject2.uniforms[key],
                alpha
            )
        self.bounding_box[:] = path_func(
            mobject1.bounding_box, mobject2.bounding_box, alpha
        )
        return self

    def pointwise_become_partial(self, mobject, a, b) -> Self:
        """
        Set points in such a way as to become only
        part of mobject.
        Inputs 0 <= a < b <= 1 determine what portion
        of mobject to become.
        """
        # To be implemented in subclass
        return self

    # Locking data

    def lock_data(self, keys: Iterable[str]) -> Self:
        """
        To speed up some animations, particularly transformations,
        it can be handy to acknowledge which pieces of data
        won't change during the animation so that calls to
        interpolate can skip this, and so that it's not
        read into the shader_wrapper objects needlessly
        """
        if self.has_updaters:
            return self
        self.locked_data_keys = set(keys)
        return self

    def lock_uniforms(self, keys: Iterable[str]) -> Self:
        if self.has_updaters:
            return self
        self.locked_uniform_keys = set(keys)
        return self

    def lock_matching_data(self, mobject1: Mobject, mobject2: Mobject) -> Self:
        tuples = zip(
            self.get_family(),
            mobject1.get_family(),
            mobject2.get_family(),
        )
        for sm, sm1, sm2 in tuples:
            if not sm.data.dtype == sm1.data.dtype == sm2.data.dtype:
                continue
            sm.lock_data(
                key for key in sm.data.dtype.names
                if arrays_match(sm1.data[key], sm2.data[key])
            )
            sm.lock_uniforms(
                key for key in self.uniforms
                if all(listify(mobject1.uniforms.get(key, 0) == mobject2.uniforms.get(key, 0)))
            )
            sm.const_data_keys = set(
                key for key in sm.data.dtype.names
                if key not in sm.locked_data_keys
                if all(
                    array_is_constant(mob.data[key])
                    for mob in (sm, sm1, sm2)
                )
            )

        return self

    def unlock_data(self) -> Self:
        for mob in self.get_family():
            mob.locked_data_keys = set()
            mob.const_data_keys = set()
            mob.locked_uniform_keys = set()
        return self

    # Operations touching shader uniforms

    def affects_shader_info_id(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.refresh_shader_wrapper_id()
            return result
        return wrapper

    @affects_shader_info_id
    def set_uniform(self, recurse: bool = True, **new_uniforms) -> Self:
        for mob in self.get_family(recurse):
            mob.uniforms.update(new_uniforms)
        return self

    @affects_shader_info_id
    def fix_in_frame(self, recurse: bool = True) -> Self:
        self.set_uniform(recurse, is_fixed_in_frame=1.0)
        return self

    @affects_shader_info_id
    def unfix_from_frame(self, recurse: bool = True) -> Self:
        self.set_uniform(recurse, is_fixed_in_frame=0.0)
        return self

    def is_fixed_in_frame(self) -> bool:
        return bool(self.uniforms["is_fixed_in_frame"])

    @affects_shader_info_id
    def apply_depth_test(self, recurse: bool = True) -> Self:
        for mob in self.get_family(recurse):
            mob.depth_test = True
        return self

    @affects_shader_info_id
    def deactivate_depth_test(self, recurse: bool = True) -> Self:
        for mob in self.get_family(recurse):
            mob.depth_test = False
        return self

    # Shader code manipulation

    @affects_data
    def replace_shader_code(self, old: str, new: str) -> Self:
        self.shader_code_replacements[old] = new
        self._shaders_initialized = False
        for mob in self.get_ancestors():
            mob._shaders_initialized = False
        return self

    def set_color_by_code(self, glsl_code: str) -> Self:
        """
        Takes a snippet of code and inserts it into a
        context which has the following variables:
        vec4 color, vec3 point, vec3 unit_normal.
        The code should change the color variable
        """
        self.replace_shader_code(
            "///// INSERT COLOR FUNCTION HERE /////",
            glsl_code
        )
        return self

    def set_color_by_xyz_func(
        self,
        glsl_snippet: str,
        min_value: float = -5.0,
        max_value: float = 5.0,
        colormap: str = "viridis"
    ) -> Self:
        """
        Pass in a glsl expression in terms of x, y and z which returns
        a float.
        """
        # TODO, add a version of this which changes the point data instead
        # of the shader code
        for char in "xyz":
            glsl_snippet = glsl_snippet.replace(char, "point." + char)
        rgb_list = get_colormap_list(colormap)
        self.set_color_by_code(
            "color.rgb = float_to_color({}, {}, {}, {});".format(
                glsl_snippet,
                float(min_value),
                float(max_value),
                get_colormap_code(rgb_list)
            )
        )
        return self

    # For shader data

    def init_shader_data(self, ctx: Context):
        self.shader_indices = np.zeros(0)
        self.shader_wrapper = ShaderWrapper(
            ctx=ctx,
            vert_data=self.data,
            shader_folder=self.shader_folder,
            texture_paths=self.texture_paths,
            depth_test=self.depth_test,
            render_primitive=self.render_primitive,
        )

    def refresh_shader_wrapper_id(self):
        if self._shaders_initialized:
            self.shader_wrapper.refresh_id()
        return self

    def get_shader_wrapper(self, ctx: Context) -> ShaderWrapper:
        if not self._shaders_initialized:
            self.init_shader_data(ctx)
            self._shaders_initialized = True

        self.shader_wrapper.vert_data = self.get_shader_data()
        self.shader_wrapper.vert_indices = self.get_shader_vert_indices()
        self.shader_wrapper.bind_to_mobject_uniforms(self.get_uniforms())
        self.shader_wrapper.depth_test = self.depth_test
        for old, new in self.shader_code_replacements.items():
            self.shader_wrapper.replace_code(old, new)
        return self.shader_wrapper

    def get_shader_wrapper_list(self, ctx: Context) -> list[ShaderWrapper]:
        shader_wrappers = it.chain(
            [self.get_shader_wrapper(ctx)],
            *[sm.get_shader_wrapper_list(ctx) for sm in self.submobjects]
        )
        batches = batch_by_property(shader_wrappers, lambda sw: sw.get_id())

        result = []
        for wrapper_group, sid in batches:
            shader_wrapper = wrapper_group[0]
            if not shader_wrapper.is_valid():
                continue
            shader_wrapper.combine_with(*wrapper_group[1:])
            if len(shader_wrapper.vert_data) > 0:
                result.append(shader_wrapper)
        return result

    def get_shader_data(self):
        return self.data

    def get_uniforms(self):
        return self.uniforms

    def get_shader_vert_indices(self):
        return self.shader_indices

    def render(self, ctx: Context, camera_uniforms: dict):
        if self._data_has_changed:
            self.shader_wrappers = self.get_shader_wrapper_list(ctx)
            for shader_wrapper in self.shader_wrappers:
                shader_wrapper.generate_vao()
            self._data_has_changed = False
        for shader_wrapper in self.shader_wrappers:
            shader_wrapper.update_program_uniforms(camera_uniforms)
            shader_wrapper.pre_render()
            shader_wrapper.render()

    # 事件处理
    
    # 该函数是一个事件处理函数，它遵循JavaScript中DOM的事件冒泡模型。
    # 它接受一个可调用函数作为回调参数。
    # 该函数接受两个参数：一个Mobject对象和一个EventData对象。
    # 如果回调函数返回false，则事件冒泡将停止。
    # 参见https://www.quirksmode.org/js/events_order.html

    # 初始化事件监听器
    def init_event_listners(self):
        # 创建一个空的事件监听器列表
        self.event_listners: list[EventListener] = []

    # 添加事件监听器
    def add_event_listner(
        self,
        # 事件类型
        event_type: EventType,
        # 事件回调函数
        event_callback: Callable[[Mobject, dict[str]]]
    ):
        # 创建一个事件监听器
        event_listner = EventListener(self, event_type, event_callback)
        # 添加到事件监听器列表中
        self.event_listners.append(event_listner)
        # 将事件监听器添加到事件分发器中
        EVENT_DISPATCHER.add_listner(event_listner)
        return self

    # 移除事件监听器
    def remove_event_listner(
        self,
        # 事件类型
        event_type: EventType,
        # 事件回调函数
        event_callback: Callable[[Mobject, dict[str]]]
    ):
        # 创建一个事件监听器
        event_listner = EventListener(self, event_type, event_callback)
        # 从事件监听器列表中移除事件监听器
        while event_listner in self.event_listners:
            self.event_listners.remove(event_listner)
        # 
        EVENT_DISPATCHER.remove_listner(event_listner)
        return self

    # 清空事件监听器
    def clear_event_listners(self, recurse: bool = True):
        # 清空事件监听器列表
        self.event_listners = []
        if recurse:
            for submob in self.submobjects:
                submob.clear_event_listners(recurse=recurse)
        return self

    # 获取事件监听器
    def get_event_listners(self):
        return self.event_listners

    # 获取事件监听器的家族
    def get_family_event_listners(self):
        return list(it.chain(*[sm.get_event_listners() for sm in self.get_family()]))

    # 判断是否有事件监听器
    def get_has_event_listner(self):
        # 遍历事件监听器列表
        return any(
            # 判断是否有事件监听器
            mob.get_event_listners()
            # 遍历事件监听器的家族
            for mob in self.get_family()
        )

    # 添加鼠标移动监听器
    def add_mouse_motion_listner(self, callback):
        self.add_event_listner(EventType.MouseMotionEvent, callback)

    # 移除鼠标移动监听器
    def remove_mouse_motion_listner(self, callback):
        self.remove_event_listner(EventType.MouseMotionEvent, callback)

    # 添加鼠标按下监听器
    def add_mouse_press_listner(self, callback):
        self.add_event_listner(EventType.MousePressEvent, callback)

    # 移除鼠标按下监听器
    def remove_mouse_press_listner(self, callback):
        self.remove_event_listner(EventType.MousePressEvent, callback)

    # 添加鼠标释放监听器
    def add_mouse_release_listner(self, callback):
        self.add_event_listner(EventType.MouseReleaseEvent, callback)

    # 移除鼠标释放监听器
    def remove_mouse_release_listner(self, callback):
        self.remove_event_listner(EventType.MouseReleaseEvent, callback)

    # 添加鼠标拖拽监听器
    def add_mouse_drag_listner(self, callback):
        self.add_event_listner(EventType.MouseDragEvent, callback)

    # 移除鼠标拖拽监听器
    def remove_mouse_drag_listner(self, callback):
        self.remove_event_listner(EventType.MouseDragEvent, callback)

    # 
    def add_mouse_scroll_listner(self, callback):
        self.add_event_listner(EventType.MouseScrollEvent, callback)

    def remove_mouse_scroll_listner(self, callback):
        self.remove_event_listner(EventType.MouseScrollEvent, callback)

    def add_key_press_listner(self, callback):
        self.add_event_listner(EventType.KeyPressEvent, callback)

    def remove_key_press_listner(self, callback):
        self.remove_event_listner(EventType.KeyPressEvent, callback)

    def add_key_release_listner(self, callback):
        self.add_event_listner(EventType.KeyReleaseEvent, callback)

    def remove_key_release_listner(self, callback):
        self.remove_event_listner(EventType.KeyReleaseEvent, callback)

    # Errors

    def throw_error_if_no_points(self):
        if not self.has_points():
            message = "Cannot call Mobject.{} " +\
                      "for a Mobject with no points"
            caller_name = sys._getframe(1).f_code.co_name
            raise Exception(message.format(caller_name))

# 组类
class Group(Mobject):
    # 组类的构造方法
    def __init__(self, *mobjects: Mobject, **kwargs):
        # 检查所有子对象是否为数学对象
        if not all([isinstance(m, Mobject) for m in mobjects]):
            # 抛出异常
            raise Exception("所有子对象必须为Mobject类型")
        # 使用父类的构造方法
        Mobject.__init__(self, **kwargs)
        # 添加子对象
        self.add(*mobjects)
        if any(m.is_fixed_in_frame() for m in mobjects):
            # 
            self.fix_in_frame()
    # 向当前组类中添加另一个数学对象或组对象
    def __add__(self, other: Mobject | Group) -> Self:
        # 判断外来者是否为数学对象
        assert(isinstance(other, Mobject))
        # 返回当前组对象
        return self.add(other)

# 点类
class Point(Mobject):
    # 点类的构造方法
    def __init__(
        self,
        # 点的坐标，默认为ORIGIN
        location: Vect3 = ORIGIN,
        # 点的宽度，默认为1e-6
        artificial_width: float = 1e-6,
        # 点的高度，默认为1e-6
        artificial_height: float = 1e-6,
        # 其他关键字参数
        **kwargs
    ):
        # 点的宽度和高度
        self.artificial_width = artificial_width
        self.artificial_height = artificial_height
        # 调用父类Mobject的构造方法
        super().__init__(**kwargs)
        # 设置点的位置
        self.set_location(location)

    # 获取点的宽度
    def get_width(self) -> float:
        return self.artificial_width

    # 获取点的高度
    def get_height(self) -> float:
        return self.artificial_height

    # 获取点的位置，三维向量
    def get_location(self) -> Vect3:
        return self.get_points()[0].copy()
        
    # 获取包围盒的点，即返回点的位置，三维向量
    def get_bounding_box_point(self, *args, **kwargs) -> Vect3:
        return self.get_location()
        
    # 设置点的位置，即设置点的顶点坐标为new_loc
    def set_location(self, new_loc: npt.ArrayLike) -> Self:
        self.set_points(np.array(new_loc, ndmin=2, dtype=float))
        return self

# 动画构造器类
class _AnimationBuilder:
    # 动画构造器类的构造方法
    def __init__(self, mobject: Mobject):
        # 保存传入的Mobject对象
        self.mobject = mobject
        # 保存要覆盖的动画，默认为空
        self.overridden_animation = None
        # 生成动画的目标状态，用于后续动画效果的计算
        self.mobject.generate_target()
        # 标记是否正在链式调用动画方法，默认没有
        self.is_chaining = False
        # 保存要调用的动画方法，默认为空列表
        self.methods: list[Callable] = []
        # 保存动画方法的参数，默认为空字典
        self.anim_args = {}
        # 标记是否可以传递参数给动画方法，默认可以
        self.can_pass_args = True

    # 在访问实例属性时动态地获取属性值，method_name 表示要访问的属性名
    def __getattr__(self, method_name: str):
        # 获取self.mobject.target对象的属性method_name
        method = getattr(self.mobject.target, method_name)
        # 获取到的属性 method 添加到self.methods列表中
        self.methods.append(method)
        # 检查获取到的属性method是否具有_override_animate属性，如果有则将其赋值给has_overridden_animation
        has_overridden_animation = hasattr(method, "_override_animate")
        # 检查是否存在方法链和是否存在被覆盖的动画
        if (self.is_chaining and has_overridden_animation) or self.overridden_animation:
            # 如果是则抛出NotImplementedError异常
            raise NotImplementedError(
                "目前不支持被覆盖动画的方法链"
            )
        # 更新动画的目标对象
        def update_target(*method_args, **method_kwargs):
            # 检查是否存在已被覆盖的动画
            if has_overridden_animation:
                # 如果存在，将已被覆盖的动画添加到动画构造器
                self.overridden_animation = method._override_animate(self.mobject, *method_args, **method_kwargs)
            else:
                # 如果不存在，调用原始方法，更新动画的属性或状态
                method(*method_args, **method_kwargs)
            # 返回动画构造器
            return self
        # 方法链正在进行中
        self.is_chaining = True
        # 返回目标更新函数
        return update_target

    # 对象实例可以像函数一样被调用，接受任意关键字参数kwargs
    def __call__(self, **kwargs):
        # 然后将这些关键字参数传递给set_anim_args方法，并返回对象实例本身
        return self.set_anim_args(**kwargs)

    # 设置动画构造器的动画参数
    def set_anim_args(self, **kwargs):
        # 可以更改 :class:`~manimlib.animation.transform.Transform` 的参数，比如：
        # - ``run_time``（运行时间）
        # - ``time_span``（时间跨度）
        # - ``rate_func``（缓动函数，控制动画的速度变化）
        # - ``lag_ratio``（延迟比例，用于控制动画的延迟）
        # - ``path_arc``（路径弧度，用于指定动画对象在运动过程中的路径弧度）
        # - ``path_func``（路径函数，用于控制动画对象在运动过程中的路径形状）
        # 等等
        # 如果不允许传递参数，说明已经设置过动画参数，
        if not self.can_pass_args:
            # 抛出ValueError异常
            raise ValueError(
                "动画参数只能通过调用 'animate' 或 'set_anim_args' 来传递，且只能被传递一次"
            )
        # 动画参数设置为传入的参数
        self.anim_args = kwargs
        # 已经设置过动画参数，不允许再次传递
        self.can_pass_args = False
        # 返回该动画构造器
        return self

    # 构建动画构造器
    def build(self):
        from manimlib.animation.transform import _MethodAnimation
        # 如果存在被覆盖的动画对象，则返回该动画对象
        if self.overridden_animation:
            return self.overridden_animation
        # 否则，根据动画构造器的各项信息创建一个新的动画对象
        return _MethodAnimation(self.mobject, self.methods, **self.anim_args)

# 通过使用@override_animate装饰器，可以在定义动画方法时指定一个重写的动画方法
# 这样，在调用动画方法时，实际执行的将是重写的动画方法
# 重写动画方法的装饰器函数
def override_animate(method):
    # 装饰一个动画方法
    def decorator(animation_method):
        # 将该动画方法赋值给重写动画
        method._override_animate = animation_method
        # 返回动画方法
        return animation_method
    # 返回用于装饰需要重写的动画方法
    return decorator
