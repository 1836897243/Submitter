import time
import subprocess
import os
from abc import ABC, abstractmethod
from datetime import datetime
from  tqdm import tqdm

class BaseJobSubmitter(ABC):
    """抽象基类：定义任务提交和调度的接口"""
    def __init__(self, commands, logfile=None):
        self.commands = commands
        self.logfile = logfile if logfile else "Submiter/job_submitter.log"
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

    def submit(self):
        for command in tqdm(self.commands, desc="Submitting jobs"):
            while self._get_available_resource() is None:
                time.sleep(10)
            resource = self._get_available_resource()
            self._submit(command, resource)

        while self._is_running():
            time.sleep(6)
        self._log("✅ All jobs are done.")

    @abstractmethod
    def _get_available_resource(self):
        """返回一个可用资源标识，如果没有资源则返回 None"""
        pass

    @abstractmethod
    def _submit(self, command, resource):
        """提交任务到指定资源"""
        pass

    @abstractmethod
    def _is_running(self):
        """判断是否还有任务在运行"""
        pass

    @abstractmethod
    def _clean_resources(self):
        """清理已完成任务"""
        pass

    def _log(self, message):
        """统一日志输出，带时间戳"""
        with open(self.logfile, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

class CudaJobSubmitter(BaseJobSubmitter):
    """基于 GPU 的任务提交器，按 GPU ID 调度任务"""
    def __init__(self, commands, gpu_ids):
        super().__init__(commands)
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
        proc = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.cuda_processes[gpu_id] = proc


class ConcurrentJobSubmitter(BaseJobSubmitter):
    """限制最大并发任务数的任务提交器"""
    def __init__(self, commands, max_jobs):
        super().__init__(commands)
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
            return len(self.processes)  # 返回当前任务槽编号
        return None

    def _is_running(self):
        self._clean_resources()
        return len(self.processes) > 0

    def _submit(self, command, resource):
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.processes.append(proc)
