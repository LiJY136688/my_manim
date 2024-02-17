from manimlib.event_handler.event_dispatcher import EventDispatcher

# 这应该是一个单例，即在运行期间应该只有一个事件调度对象
EVENT_DISPATCHER = EventDispatcher()
