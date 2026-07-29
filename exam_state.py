"""考试状态机：统一管理读帧/推理/录制的生命周期

IDLE      待考：读帧线程睡眠，不推理不录制
STARTING  /start 进行中：读帧线程读帧供三级校验，但跳过推理和录制
RUNNING   考试中：完整推理 + 录制
"""


class ExamState:
    IDLE = 0
    STARTING = 1
    RUNNING = 2
