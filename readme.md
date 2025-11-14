# 本地多线程
## 首次提交任务
```python
from Submitter import CudaJobSubmitter, ConcurrentJobSubmitter
# cuda
submitter = CudaJobSubmitter(file_prefix='command_file_prefix', gpu_ids=[0,1,2,3])
submitter.addJobs(cmds)
submitter.submit(repeat_last=False)

# just cpu
submitter = CudaJobSubmitter(file_prefix='command_file_prefix', max_jobs=10)
submitter.addJobs(cmds)
submitter.submit(repeat_last=False)
```

## 添加更多任务
```python
submitter.addJobs(cmds) #任务会加入file_prefix为前缀的文件中，供submitter读取
```

# Slurm
## 首次提交任务
```python
from Submitter import SlurmJobSubmitter
submitter = SlurmJobSubmitter(file_prefix='test_jobs', ntasks=2, require_gpu=False, mem=16, partition='gpujl')
submitter.addJobs(cmds)
submitter.submit(job_name='test_job',repeat_last=True)
```

## 添加更多任务
```python
submitter.addJobs(cmds) #任务会加入file_prefix为前缀的文件中，供submitter读取
```