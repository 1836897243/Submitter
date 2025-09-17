import os
import subprocess
from io import StringIO
import pandas as pd
import numpy as np

import random   
class SlurmJobSubmitter:
    '''
    only support for slurm
    only a gpu per job
    '''
    def __init__(self,commands,job_name,max_jobs=8,require_gpu=True, mem=64):   
        '''
        for gpu job, only one gpu per job, and execute commands sequentially
        for cpu job, zero gpu per job, execute commands in parallel, and each job use 10 cpus
        '''
        self.commands = np.random.permutation(commands).tolist()
        
        self.job_name = job_name
        self.max_jobs = max_jobs
        self.require_gpu = require_gpu

        self.max_nodes = 2 # max nodes to use

        self.cpus_per_task = 4 if require_gpu else 10
        self.mem = mem 
        self.gpu_cnt = 1 if require_gpu else 0
        self.nodes, self.avail_cpu, self.avail_gpu = self._get_available_node()
        self.outdir = 'Submiter/slurm_logs'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
    def submit(self):
        '''
        submit job to slurm
        '''
        if len(self.commands) == 0:
            print('no job to run')
            return
        job_cnt = 0
        for node,cpu,gpu in zip(self.nodes,self.avail_cpu,self.avail_gpu):
            if self.require_gpu:
                for i in range(gpu):
                    job_cnt += 1
            else:
                for i in range(cpu//self.cpus_per_task):
                    job_cnt += 1
        max_jobs = min(job_cnt,self.max_jobs)
        commands_per_job = len(self.commands)//(max_jobs)+1

        job_cnt = 0
        for node,cpu,gpu in zip(self.nodes,self.avail_cpu,self.avail_gpu):
            
            if self.require_gpu:
                for i in range(gpu):
                    commands = self.commands[job_cnt*commands_per_job:(job_cnt+1)*commands_per_job]
                    self._submit(commands,node,self.job_name+f'-{job_cnt}-gpu')
                    job_cnt += 1
                    if job_cnt >=max_jobs:
                        return
            else:
                for i in range(cpu//self.cpus_per_task):
                    commands = self.commands[job_cnt*commands_per_job:(job_cnt+1)*commands_per_job]
                    self._submit(commands,node,self.job_name+f'-{job_cnt}-cpu')
                    job_cnt += 1
                    if job_cnt >=max_jobs:
                        return

    
    def _get_my_node(self):
        # get user name
        user_name = os.environ['USER']
        # get node of user if exist
        print('-'*50)
        print(f'get node of user {user_name}:')
        output = subprocess.getoutput(f'squeue -u {user_name}')
        print(output)
        data = StringIO(output)# convert string to file-like object
        df = pd.read_csv(data, delim_whitespace=True)
        data.close()
        nodes = np.unique(df['NODELIST(REASON)']).tolist()
        for node in nodes:
            if not node.startswith('node'):
                nodes.remove(node)
        print(f'get nodes:{nodes}')
        return nodes
    
    def _get_available_node(self):
        '''
        squeue -u username # get the node of user
        sinfo # get idle node and mix node
        '''
        nodes = self._get_my_node()
        avail_cpu = []
        avail_gpu = []
        for node in nodes:
            cpu,gpu = self._get_cpu_gpu_avail(node)
            avail_cpu.append(cpu)
            avail_gpu.append(gpu)
        # if no new node available
        if len(nodes) >= self.max_nodes:
            return nodes, avail_cpu, avail_gpu
        else:
            print('-'*50)
            print('get available node:')
            output = subprocess.getoutput('sinfo')
            print(output)
            data = StringIO(output)
            df = pd.read_csv(data, delim_whitespace=True)
            data.close()
            if 'idle' in df['STATE'].values:
                idel_nodes = parser_nodes(df['NODELIST'][df['STATE']=='idle'].values[0])
                for i in range(self.max_nodes-len(nodes)):
                    if i<len(idel_nodes):
                        nodes.append(idel_nodes[i])
                        cpu,gpu = self._get_cpu_gpu_avail(idel_nodes[i])
                        avail_cpu.append(cpu)
                        avail_gpu.append(gpu)
                if len(nodes) == self.max_nodes:
                    return nodes, avail_cpu, avail_gpu
            if 'mix' in df['STATE'].values:
                mixed_nodes = parser_nodes(df['NODELIST'][df['STATE']=='mix'].values[0])
                avail_cpu_gpu = [self._get_cpu_gpu_avail(node) for node in mixed_nodes]
                avail_cpu_mix = [cpu for cpu,gpu in avail_cpu_gpu]
                avail_gpu_mix = [gpu for cpu,gpu in avail_cpu_gpu]
                if self.require_gpu:
                    args_sort = np.argsort(avail_gpu_mix)[::-1]
                else:
                    args_sort =  np.argsort(avail_cpu_mix)[::-1]
                for i in range(self.max_nodes-len(nodes)):
                    nodes.append(mixed_nodes[args_sort[i]])
                    avail_cpu.append(avail_cpu_mix[args_sort[i]])
                    avail_gpu.append(avail_gpu_mix[args_sort[i]])
                    if len(nodes) == self.max_nodes:
                        return nodes, avail_cpu, avail_gpu
            return nodes, avail_cpu, avail_gpu

    def _get_cpu_gpu_avail(self, node):
        print('-'*50)
        print(f'get resource of node {node}:')
        output = subprocess.getoutput(f'scontrol show node {node}').split('\n')
        print(output)
        
        find_max_resource = False
        find_alloc_resource = False
        for line in output:
            if line.find('CfgTRES') != -1:
                idx_cpu = line.find('cpu=')
                max_cpu_cnt = get_int_digits(line,idx_cpu+4)
                idx_gpu = line.find('gres/gpu=')
                max_gpu_cnt = get_int_digits(line,idx_gpu+9)
                find_max_resource = True
            if line.find('AllocTRES') !=-1:
                idx_cpu = line.find('cpu=')
                idx_gpu = line.find('gres/gpu=')
                if idx_cpu != -1 and idx_gpu != -1:
                    alloc_cpu_cnt = get_int_digits(line,idx_cpu+4)
                    alloc_gpu_cnt = get_int_digits(line,idx_gpu+9)
                else:
                    alloc_cpu_cnt = 0
                    alloc_gpu_cnt = 0
                find_alloc_resource = True
        assert find_max_resource and find_alloc_resource, 'no max resource or alloc resource found'
        return max_cpu_cnt-alloc_cpu_cnt, max_gpu_cnt-alloc_gpu_cnt

    def _submit(self,commands,node,job_name):
        # generate python script
        python_script = f'''import time
import subprocess
import os
class JobSubmitter:
    def __init__(self,commands, max_jobs):
        self.commands = commands
        self.max_jobs = max_jobs
        self.processes = []
    def run(self):
        for i,command in enumerate(self.commands):
            while self._is_full():
                time.sleep(5)
            print(f'run command:{{command}}')
            self._submit(command)
        while self._is_running():
            time.sleep(6)
        print('all jobs are done')
    def _is_running(self):
        for proc in self.processes:
            if proc.poll() is None:
                return True
        return False
    
    def _submit(self,command):
        proc = subprocess.Popen(command, shell=True)
        self.processes.append(proc)

    def _is_full(self):
        for proc in self.processes:
            if proc.poll() is not None:# process is done
                self.processes.remove(proc)
        return len(self.processes) >= self.max_jobs
commands = {commands}
Submitter = JobSubmitter(commands, max_jobs={self.cpus_per_task})
Submitter.run()'''
        
        
        if self.require_gpu:
            job_commands = '\n'.join(commands)
        else:
            with open(f'{self.outdir}/python_script_{job_name}.py','w') as f:
                f.write(python_script)
            job_commands = f'python {self.outdir}/python_script_{job_name}.py'

        script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.outdir}/{job_name}.out
#SBATCH --time=5-15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --mem={self.mem}G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:{self.gpu_cnt}
#SBATCH --nodelist={node}
{job_commands}'''
        
        with open(f'{self.outdir}/script_{job_name}.sh','w') as f:
            f.write(script)
        # print(f'{python_script}')
        result = subprocess.run(['sbatch', f'{self.outdir}/script_{job_name}.sh'], capture_output=True, text=True)
        print('-'*50)
        print(f'submit job to node {node} with result:{result.stdout}')
        # if os.path.exists('python_script.py'):
        #     os.remove('python_script.py')
        # os.remove('script.sh')

def get_int_digits(string, start_idx):
            idx = start_idx
            while idx < len(string):
                if not string[idx].isdigit():
                    break
                idx += 1
            assert idx >= start_idx, 'no digit found'
            return int(string[start_idx:idx])
def parser_nodes(str_nodes):
    '''
        str_nodes: 
            1. node01
            2. node[01-03,05]
    '''
    return_nodes = []
    if '[' in str_nodes and ']' in str_nodes:
        idx_s = str_nodes.find('[')
        idx_e = str_nodes.find(']')
        nodes = str_nodes[idx_s+1:idx_e].split(',')
        for node in nodes:
            if '-' in node:
                idx = node.find('-')
                start = int(node[:idx])
                end = int(node[idx+1:])
                return_nodes = return_nodes + [f'node{str(i).zfill(2)}' for i in range(start,end+1)]
            else:
                return_nodes.append(f'node{str(node).zfill(2)}')
        return return_nodes
    else:
        return [str_nodes]