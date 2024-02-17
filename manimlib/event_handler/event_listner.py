from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.event_handler.event_type import EventType
    from manimlib.mobject.mobject import Mobject

# 事件监听器类
class EventListener(object):
    # 初始化
    def __init__(
        self,
        # 监听的mobject
        mobject: Mobject,
        # 监听的事件类型
        event_type: EventType,
        # 监听的回调函数
        event_callback: Callable[[Mobject, dict[str]]]
    ):
        self.mobject = mobject
        self.event_type = event_type
        self.callback = event_callback

    # 判断两个监听器是否相等
    def __eq__(self, o: object) -> bool:
        return_val = False
        try:
            # 比较的标准是监听的对象、事件类型和回调函数是否相等
            return_val = self.callback == o.callback \
                and self.mobject == o.mobject \
                and self.event_type == o.event_type
        except:
            pass
        # 只要有不同，都返回False
        return return_val
