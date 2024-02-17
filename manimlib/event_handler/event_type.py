from enum import Enum

# 事件类型的枚举类
class EventType(Enum):
    # 鼠标移动事件
    MouseMotionEvent = 'mouse_motion_event'
    # 鼠标点击事件
    MousePressEvent = 'mouse_press_event'
    # 鼠标释放事件
    MouseReleaseEvent = 'mouse_release_event'
    # 鼠标拖拽事件
    MouseDragEvent = 'mouse_drag_event'
    # 鼠标滚动事件
    MouseScrollEvent = 'mouse_scroll_event'
    # 键盘按下事件
    KeyPressEvent = 'key_press_event'
    # 键盘释放事件
    KeyReleaseEvent = 'key_release_event'
