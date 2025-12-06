import os
import fcntl
from contextlib import contextmanager


@contextmanager
def file_lock(file):
    """文件级排他锁"""
    fcntl.flock(file, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(file, fcntl.LOCK_UN)


class SafeOffsetFileQueue:
    """基于偏移量指针 + 文件锁的多进程安全队列"""
    def __init__(self, queue_file="queue.txt", offset_file="offset.txt"):
        self.queue_file = queue_file
        self.offset_file = offset_file
        if not os.path.exists(self.queue_file):
            open(self.queue_file, "w").close()
        if not os.path.exists(self.offset_file):
            with open(self.offset_file, "w") as f:
                f.write("0")

    def put(self, item):
        """入队：追加到文件末尾"""
        with open(self.queue_file, "a") as f:
            with file_lock(f):
                f.write(item + "\n")

    def get(self):
        """出队：根据偏移量读取一条新任务并更新指针"""
        # 先锁 offset 文件，确保只有一个进程读取更新偏移量
        with open(self.offset_file, "r+") as ofs:
            with file_lock(ofs):
                offset = int(ofs.read().strip() or 0)

                # 再锁任务文件，读取当前任务
                with open(self.queue_file, "r") as qf:
                    with file_lock(qf):
                        lines = qf.readlines()

                if offset >= len(lines):
                    return None  # 没有新任务

                item = lines[offset].strip()

                # 更新偏移量
                ofs.seek(0)
                ofs.write(str(offset + 1))
                ofs.truncate()

                return item

    def empty(self):
        """判断是否还有未处理任务"""
        with open(self.offset_file, "r") as ofs:
            offset = int(ofs.read().strip() or 0)
        with open(self.queue_file, "r") as qf:
            lines = qf.readlines()
        return offset >= len(lines)
    
    def truncate(self, num_items: int = 0):
        """
        截断队列，保留前 num_items 条任务，并更新偏移量。
        """
        # 更新偏移量
        with open(self.offset_file, "r+") as ofs:
            with file_lock(ofs):
                current_offset = int(ofs.read().strip() or 0)
                new_offset = min(current_offset, num_items)
                ofs.seek(0)
                ofs.write(str(new_offset))
                ofs.truncate()
                
        # 先锁队列文件
        with open(self.queue_file, "r+") as qf:
            with file_lock(qf):
                lines = qf.readlines()
                qf.seek(0)
                qf.writelines(lines[:num_items])
                qf.truncate()  # 截掉多余内容

        

