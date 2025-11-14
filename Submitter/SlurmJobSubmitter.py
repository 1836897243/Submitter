import os
import subprocess
from .JobSubmitter import CudaJobSubmitter, ConcurrentJobSubmitter
class SlurmJobSubmitter:
    '''
    only support for slurm
    only a gpu per job
    '''
    def __init__(self,file_prefix='',ntasks=70,require_gpu=True, mem=475, partition='gpujl'):   
        '''
        for gpu job, only one gpu per job, and execute commands sequentially
        for cpu job, zero gpu per job, execute commands in parallel, and each job use 10 cpus
        '''
        self.ntasks = ntasks
        self.require_gpu = require_gpu
        self.file_prefix = file_prefix

        self.cpus_per_task = 1
        self.mem = mem 
        self.partition = partition
        self.gpu_cnt = 4 if require_gpu else 0
        self.outdir = 'Submiter/slurm_logs'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def addJobs(self, commands):
        if self.require_gpu:
            submitter = CudaJobSubmitter(file_prefix=self.file_prefix, gpu_ids=[0,1,2,3])
        else:
            submitter = ConcurrentJobSubmitter(file_prefix=self.file_prefix, max_jobs=self.ntasks)
        submitter.addJobs(commands)

    def submit(self,job_name, repeat_last=False):
        # generate python script
        python_script = f'''import time
import subprocess
import os
from Submitter import CudaJobSubmitter, ConcurrentJobSubmitter
'''
        if self.require_gpu:
            python_script += f'''submitter = CudaJobSubmitter(file_prefix='{self.file_prefix}', gpu_ids=[0,1,2,3])
submitter.submit(repeat_last={str(repeat_last)})'''
        else:
            python_script += f'''submitter = ConcurrentJobSubmitter(file_prefix='{self.file_prefix}', max_jobs={self.ntasks})
submitter.submit(repeat_last={str(repeat_last)})'''
        with open(f'{self.outdir}/python_script_{job_name}.py','w') as f:
            f.write(python_script)
        job_command = f'python {self.outdir}/python_script_{job_name}.py'

        script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.outdir}/{job_name}.out
#SBATCH --time=5-15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --mem={self.mem}G
#SBATCH --partition={self.partition}
#SBATCH --gres=gpu:{self.gpu_cnt}
{job_command}'''
        
        with open(f'{self.outdir}/script_{job_name}.sh','w') as f:
            f.write(script)
        result = subprocess.run(['sbatch', f'{self.outdir}/script_{job_name}.sh'], capture_output=True, text=True)
    

if __name__ == '__main__':
    commands = [f'python smote.py --seed {i} --dataset \'credit-g\' --sampler TreeSMOTE2 --model DecisionTree' for i in range(20)]
    submitter = SlurmJobSubmitter()
    submitter.submit(job_name="test")
