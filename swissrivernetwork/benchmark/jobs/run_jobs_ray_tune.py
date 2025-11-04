"""
run_jobs_ray_tune



@Author: linlin
@Date: Sep 05 2025

Notice: Use same Python and Ray versions for training and evaluation. Otherwise, the checkpoint may not be loaded correctly.
E.g., Python 3.10.4-GCCcore-11.3.0 trained and saved checkpoints can not be loaded by Python 3.12.3.
"""

import os
import re

from swissrivernetwork.benchmark.util import is_transformer_model

cur_path = os.path.dirname(os.path.abspath(__file__))

# prefix keyword # TODO
prefix_kw = 'srn'  # swiss river network project

# The cluster used
infras = 'ubelix'  # 'criann'
"""Change the following script parts according to the cluster:
	- `--partition`: e.g., `tlong` for CRIANN, `epyc2` for UBELIX.
	- module loaded:
		- CRIANN/CPU: 
			module load python3-DL/keras/2.4.3-cuda10.1
			# CMake is needed for CRIANN as well.
		- UBELIX/CPU: 
			module load Python/3.12.3-GCCcore-13.3.0  # previous version: 3.10.4-GCCcore-11.3.0
			module load CMake
"""


def get_job_script(
        job_name: str,
        py_file: str,
        params: dict = {},
        device: str = 'gpu'
) -> str:
    if device == 'gpu':
        script = get_job_script_gpu(job_name)


        def get_command(s):
            return 'sbatch <<EOF\n' + s + '\nEOF'

    elif device == 'cpu':
        script = get_job_script_cpu(job_name)


        def get_command(s):
            return 'sbatch <<EOF\n' + s + '\nEOF'

    elif device == 'cpu_local':
        print(job_name)
        script = get_job_script_cpu_local()
        import datetime
        now = datetime.datetime.now()
        fn_op = os.path.join(
            cur_path,
            'outputs/' + prefix_kw + '.' + job_name + '.o' + now.strftime(
                '%Y%m%d%H%M%S'
            )
        )


        def get_command(s):
            return s + ' > ' + fn_op

    elif device is None:
        # script = ''
        raise ValueError('Device not specified.')

    #     script += r"""
    # python3 """ + py_file
    script += r"""
python3 """ + py_file + ' ' + ' '.join([r"""--""" + k + r""" """ + str(v) for k, v in params.items()])
    script = script.strip()
    script = re.sub('\n\t+', '\n', script)
    script = re.sub('\n +', '\n', script)

    return get_command(script)


def get_job_script_gpu(id_str):
    # ubelix
    script = r"""
#!/bin/bash

# Not shared resources
##SBATCH --exclusive
#SBATCH --job-name=""" + '"' + prefix_kw + r""".""" + id_str + r""""
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/""" + prefix_kw + r""".""" + id_str + r""".o%J"
#SBATCH --error="errors/""" + prefix_kw + r""".""" + id_str + r""".e%J"
#
# GPUs architecture and number
# ----------------------------
#SBATCH --partition=gpu # @todo: to change it back p100, v100
##SBATCH --qos=job_gpu_preemptable # @fixme: this is a must if one wants to use a100 gpu on UBELIX cluster, but it can be preempted by investors' tasks.
##SBATCH --no-requeue  # this option ensure that the job, if preempted, won't be re-queued but canceled instead: 
## GPUs per compute node
##   gpu:4 (maximum) for gpu_k80
##   gpu:2 (maximum) for gpu_p100
##SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --gres=gpu:rtx4090:1  # @fixme: change as needed. Can use multiple GPUs so that Ray Tune can run more trials in parallel to better prune.
##SBATCH --gpus-per-node=a100:1  # may needed for 1024 px)
# ----------------------------
# Job time (hh:mm:ss)
#SBATCH --time=24:00:00  # @fixme: change as needed
##SBATCH --ntasks=1
##SBATCH --nodes=1
#SBATCH --cpus-per-task=4  # for UBELIX: per rtx4090 GPU (CPU: 16, RAM: 92160 MB) # When cpu = 1 and gpu = 0.25 per trial, num_cpus = 4 allows for 4 trials to run in parallel on a machine with 4 cpus and 1 gpu.
#SBATCH --mem-per-cpu=10G  # @fixme: change as needed

# environments
# ---------------------------------
# cuDNN is needed when using Python/3.10.4-GCCcore-11.3.0. When using Python/3.12.3-GCCcore-13.3.0 and torch >= 2.6,
# torch comes with its own cuDNN, so no need to load it separately. If loading cuDNN incorrectly, it may cause 
# discrepancy between torch and cuDNN versions, leading to errors (e.g., for lstm).
# module load cuDNN/9.5.0.50-CUDA-12.6.0  # previous version: cuDNN/8.9.2.26-CUDA-12.2.0
module load Python/3.12.3-GCCcore-13.3.0
##module load CMake
source """ + cur_path + r"""/../../../.venv/bin/activate  # change as needed
python3 --version
module list

echo hostname
cd """ + cur_path + r"""/../
echo Working directory : $PWD
echo Local work dir_file : $LOCAL_WORK_DIR
"""
    return script


def get_job_script_cpu(id_str):
    script = r"""
#!/bin/bash

##SBATCH --exclusive
#SBATCH --job-name=""" + '"' + prefix_kw + r""".cpu.""" + id_str + r""""
#SBATCH --partition=epyc2,bdw # @todo: to change it back
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/""" + prefix_kw + r""".cpu.""" + id_str + r""".o%J"
#SBATCH --error="errors/""" + prefix_kw + r""".cpu.""" + id_str + r""".e%J"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50 # @todo: to change it, max 128 on UBELIX, 28 on CRIANN
#SBATCH --time=90:00:00 # @todo: to change it back
# Do not use values without a unit. In CRIANN, the default unit is MB; while in UBELIX, it is GB.
#SBATCH --mem-per-cpu=10G  # This value can not exceed 4GB on CRIANN. on UBELIX, 256G for 1 CPU, 7G for 128 CPUs each.

module load Python/3.12.3-GCCcore-13.3.0
source """ + cur_path + r"""/../../../.venv/bin/activate  # change as needed
python3 --version
# module load CMake # This is useful to load GLIBCXX_3.4.29 for GED computation.
module list

echo hostname
cd """ + cur_path + r"""/../
echo Working directory : $PWD
echo Local work dir_file : $LOCAL_WORK_DIR
"""

    return script


def get_job_script_cpu_local():
    script = r"""
cd """ + cur_path + r"""/../generation/
echo Working directory : $PWD
echo Local work dir_file : $LOCAL_WORK_DIR
"""
    return script


# TODO: this is not working correctly.
def check_job_script(script, user='lj22u267'):
    """
    Check the job name in the given script, to see if it is already submitted in
    the cluster by SLURM.
    """
    import re
    pattern = re.compile(r"""--job-name="(.*)" """)
    match = pattern.search(script)
    if match is None:
        return False
    job_name = match.group(1)
    import subprocess
    cmd = 'squeue -u ' + user
    output = subprocess.check_output(cmd, shell=True)
    output = output.decode('utf-8')
    if job_name in output:
        return True
    else:
        return False


if __name__ == '__main__':
    os.makedirs('outputs/', exist_ok=True)
    os.makedirs('errors/', exist_ok=True)

    # fixme debug: change these as needed.

    # general settings:
    # It seems GPUs should be used for transformer model:
    # for transformer_embedding, GPU per epoch is around 50 seconds, CPU around 14-40 minutes
    # Using GPU is much faster. Notice when only one GPU is present, set the `resources_per_trial['gpu']` to a float
    # between 0 and 1 to enable multiple trials on GPU, otherwise it seems that Ray Tune just simply runs each trial
    # until the end.
    device = 'gpu'

    # methods = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn', 'transformer_embedding']
    methods = [
        'transformer_embedding', 'transformer_stgnn', 'transformer', 'transformer_graphlet',
        'lstm_embedding', 'stgnn', 'lstm', 'graphlet'
    ][1:2]
    graphs = ['swiss-1990', 'swiss-2010', 'zurich'][0:2]  # fixme
    # positional_encodings = ['none']  # fixme: for lstm
    positional_encodings = ['learnable', 'sinusoidal', 'rope', 'none'][0:3]  # Only for transformer
    window_lens = [90]  # [90, 366, 731, 10000]
    # max_len = [500]
    missing_value_methods = ['none']  # ['mask_embedding']  # 'mask_embedding' or 'interpolation' or 'zero' or 'none'
    use_current_xs = [True]  # whether to use current time step features (e.g., air temperature) as input
    short_subsequence_methods = ['drop']  # 'pad' or 'drop', how to deal with short subsequences
    max_mask_consecutives = [0]  # only used when missing_value_method is 'mask_embedding'
    # max_mask_ratios = [0.5]  # only used when missing_value_method is 'mask_embedding'
    # Other settings:
    resumes = [True]  # fixme: depends on exps. Whether to resume from previous checkpoints

    params_list = {
        'method': methods,
        'graph': graphs,
        'positional_encoding': positional_encodings,
        'window_len': window_lens,
        'missing_value_method': missing_value_methods,
        'use_current_x': use_current_xs,
        'short_subsequence_method': short_subsequence_methods,
        'max_mask_consecutive': max_mask_consecutives,
        'resume': resumes,
    }

    from sklearn.model_selection import ParameterGrid

    params_list = list(ParameterGrid(params_list))
    for i, params in enumerate(params_list):
        if is_transformer_model(params['method']):
            device = 'gpu'
        # elif params['method'] in ['lstm', 'graphlet', 'lstm_embedding', 'stgnn']:
        #     device = 'cpu'  # todo: seems with ray tune, cpu is not working properly.
        else:
            device = 'gpu'
        # else:
        #     raise NotImplementedError(f'Method {params["method"]} not implemented.')

        exp_key = '__'.join([f'{v}' for _, v in params.items()])
        print(f'[{i + 1}/{len(params_list)}] Experiment: {exp_key}')

        config = f'{cur_path}/../configs/{params["method"]}.yaml'

        command = get_job_script(
            job_name=exp_key,
            py_file='ray_tune.py',
            params={**{'config': config}, **params},
            device=device,
        )

        if check_job_script(command, user='lj24u267'):
            print('Job already submitted.')
        else:
            output = os.system(command)
