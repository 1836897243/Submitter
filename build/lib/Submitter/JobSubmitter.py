import time
import subprocess
import os
from abc import ABC, abstractmethod
from datetime import datetime
from .SafeOffsetFileQueue import SafeOffsetFileQueue

class BaseJobSubmitter(ABC):
    """抽象基类：定义任务提交和调度的接口"""
    def __init__(self, file_prefix, logfile=None):
        self.queue = SafeOffsetFileQueue(queue_file=f"{file_prefix}_queue.txt",offset_file=f"{file_prefix}_offset.txt")
        self.logfile = logfile if logfile else f"Submiter/job_submitter_{file_prefix}.log"
        os.makedirs(os.path.dirname(self.logfile), exist_ok=True)
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

    def truncate(self, num_items):
        self.queue.truncate(num_items)


    def addJobs(self, commands):
        """
        批量添加任务到文件队列中
        :param commands: list[str] - 命令字符串列表
        """
        if not isinstance(commands, (list, tuple)):
            raise TypeError("commands 必须是列表或元组类型。")

        added = 0
        for cmd in commands:
            if not isinstance(cmd, str):
                self._log(f"⚠️ 忽略非法任务（非字符串类型）: {cmd}")
                continue
            self.queue.put(cmd)
            added += 1

        self._log(f"🧾 成功添加 {added} 条任务到队列中。")

    def submit(self, repeat_last=False):
        """
        从文件队列中持续取任务直到完成。

        :param repeat_last: bool
        """

        assert not self.queue.empty(), "提交失败，任务为空"
        last_command = None

        while True:
            command = self.queue.get()

            if command is None:
                if repeat_last and last_command is not None:
                    # 队列空了，但需要重复最后一条任务
                    command = last_command
                else:
                    # 不重复 → 进入正常清理流程
                    if self._is_running():
                        time.sleep(5)
                        continue
                    break

            # 保存最后一条任务
            last_command = command

            # 等待资源可用
            while self._get_available_resource() is None:
                time.sleep(5)

            resource = self._get_available_resource()
            self._submit(command, resource)

        if not repeat_last:
            while self._is_running():
                time.sleep(6)
            self._log("✅ All jobs are done.")

    @abstractmethod
    def _get_available_resource(self):
        pass

    @abstractmethod
    def _submit(self, command, resource):
        pass

    @abstractmethod
    def _is_running(self):
        pass

    @abstractmethod
    def _clean_resources(self):
        pass

    def _log(self, message):
        """统一日志输出，带时间戳"""
        with open(self.logfile, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


# ===========================================================
# GPU 调度器
# ===========================================================
class CudaJobSubmitter(BaseJobSubmitter):
    """基于 GPU 的任务提交器，按 GPU ID 调度任务"""
    def __init__(self, file_prefix, gpu_ids):
        super().__init__(file_prefix)
        self.gpu_ids = gpu_ids
        self.cuda_processes = {gpu_id: None for gpu_id in gpu_ids}

    def _clean_resources(self):
        for gpu_id, proc in self.cuda_processes.items():
            if proc is not None and proc.poll() is not None:
                if proc.returncode == 0:
                    self._log(f"[info][command: {proc.args}]")
                else:
                    self._log(f"[error][command: {proc.args}]")
                self.cuda_processes[gpu_id] = None

    def _get_available_resource(self):
        self._clean_resources()
        for gpu_id, proc in self.cuda_processes.items():
            if proc is None:
                return gpu_id
        return None

    def _is_running(self):
        self._clean_resources()
        return any(proc is not None for proc in self.cuda_processes.values())

    def _submit(self, command, gpu_id):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        proc = subprocess.Popen(command, shell=True, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.cuda_processes[gpu_id] = proc


# ===========================================================
# 并发任务提交器
# ===========================================================
class ConcurrentJobSubmitter(BaseJobSubmitter):
    """限制最大并发任务数的任务提交器"""
    def __init__(self, file_prefix, max_jobs):
        super().__init__(file_prefix)
        self.max_jobs = max_jobs
        self.processes = []

    def _clean_resources(self):
        for proc in list(self.processes):
            if proc.poll() is not None:
                if proc.returncode == 0:
                    self._log(f"[info][command: {proc.args}]")
                else:
                    self._log(f"[error][command: {proc.args}]")
                self.processes.remove(proc)

    def _get_available_resource(self):
        self._clean_resources()
        if len(self.processes) < self.max_jobs:
            return len(self.processes)
        return None

    def _is_running(self):
        self._clean_resources()
        return len(self.processes) > 0

    def _submit(self, command, resource):
        proc = subprocess.Popen(command, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.processes.append(proc)
    
if __name__ == "__main__":
    # 示例用法
    submitter = CudaJobSubmitter(file_prefix='test_jobs', gpu_ids=[0, 1])
    submitter.addJobs(['python test.py', 'python test2.py'])
    submitter.submit()
