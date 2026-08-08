"""考试状态机：统一管理读帧/推理的生命周期

IDLE      待考：读帧线程睡眠，不推理
STARTING  /start 进行中：读帧线程读帧供就绪校验，但跳过推理
RUNNING   考试中：完整推理
"""


class ExamState:
    IDLE = 0
    STARTING = 1
    RUNNING = 2
