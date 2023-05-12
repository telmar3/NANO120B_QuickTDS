**In Anaconda, port forward to Expanse and login as usual:**
```bash
ssh -Y -l <USERNAME> login.expanse.sdsc.edu 
```

In order to run jupyter notebooks through the Expanse shell, you must run the `galyleo` script from https://github.com/mkandes/galyleo that allows you to schedule a batch job via slurm that will run a jupyterlab server on Expanse that you can securely open in your browser.

**Prepend galyleo install location:**
```bash
export PATH="/cm/shared/apps/sdsc/galyleo:${PATH}"
```

**Launching a notebook server:**
```bash
galyleo launch --account <abc123> --partition shared --cpus 1 --memory 2 --time-limit 00:45:00 --env-modules cpu/0.17.3b,gcc/10.2.0,anaconda3/2021.05
```
Alternatively the following combines the two prior steps:
```
/cm/shared/apps/sdsc/galyleo/galyleo.sh launch --account xyz123 --partition shared --cpus 3 --memory 32 --time-limit 01:30:00 --env-modules cpu/0.17.3b,gcc/10.2.0,anaconda3/2021.05
```

It will generate a randomized url that you can copy and paste into your web browser of choice. If I understood the docs correctly, then the available python libraries depend on the virtual environment on your local machine, rather than from SDSC using the above method. 


```bash
galyleo launch -a csd709 -p shared --cpus 1 --memory 2 --gpus 1 --time-limit 01:30:00 --env-modules cpu,gpu,anaconda3
```

---

Another option is using a Singularity container that has all the python packages included (+ GPU drivers and CUDA libraries) found here `/cm/shared/apps/containers/singularity/`

More information on accessing Singularity using Expanse:
https://cvw.cac.cornell.edu/ExpanseSing/runcontainer 

https://github.com/sdsc-hpc-training-org/hpc-training-2022/blob/main/week07_ml_tensorflow_pytorch/Intro_TensorFlow_PyTorch_Usage.pdf



I have also uploaded the TDS training data. Just `cp` it into your own directories.
```bash
 cp /home/bbragado/NNN_120B/'100 files'/
 cp /home/bbragado/NNN_120B/'15K files'/
 cp /home/bbragado/NNN_120B/TDS_combined_raw
```



### Enabling TensorFlow Example

Change allocation "xyz123", number of nodes, time limit, and other launch params as needed. 

```
[1] CPU node:
/cm/shared/apps/sdsc/galyleo/galyleo.sh launch -A xyz123 -p shared -n 16 -M 32 -t 00:30:00 -e singularitypro/ -s /cm/shared/apps/containers/singularity/keras-v2.2.4-tensorflow-v1.12-gpu-20190214.simg -d /home/$USER/tensorflow


[2] GPU node:
/cm/shared/apps/sdsc/galyleo/galyleo.sh launch -A xyz123 -p gpu-shared -n 10 -M 93 -G 1 -t 00:30:00 -e singularitypro/ -s /cm/shared/apps/containers/singularity/tensorflow/tensorflow-2.3.0-gpu-20200929.simg -d /home/$USER/tensorflow


---

**Resources:**

https://github.com/sdsc-hpc-training-org/basic_skills/tree/master/how_to_run_notebooks_on_expanse#galyleo

https://hpc-training.sdsc.edu/notebooks-101/notebook-101.html#download-example-notebooks

https://github.com/sdsc-hpc-training-org/hpc-training-2022

HPC (High powered computing) Examples:
` git clone https://github.com/sdsc-hpc-training-org/notebook-examples.git `




Theoretically, we can convert .ipynb notebooks into .py and run that through `sbatch` or `srun`, but I can not get it to work due to not detecting the python libraries. If anyone else has any ideas of how this can work (i.e loading specific modules through the console or through editing the `f.write` lines below), please let me know! Thanks
Usage: Add/edit the following block into the python script and run the code block. It should automatically send it as a job through EXPANSE. 
```python
import multiprocessing
print("num of cpus:", multiprocessing.cpu_count()) #num of available cpus
import os
os.system('jupyter nbconvert --to python sklearn-parallel.ipynb') # Converts the nb into .py
job_name = "rfr_job"
sbatch_file = f"{job_name}.slurm"
### Creating a sbatch file to run the .py 
with open(sbatch_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --job-name={job_name}\n")
    f.write("#SBATCH --output={job_name}.out\n")
    f.write("#SBATCH --account=csd709\n") # Change account number to new one
    f.write("#SBATCH --nodes=3\n")
    f.write("#SBATCH --ntasks-per-node=16\n") # number of cores used
    f.write("#SBATCH --cpus-per-task=3\n")
    f.write("#SBATCH --time=01:30:00\n")
    f.write("#SBATCH --mem=32G\n")
    f.write("#SBATCH --partition=shared\n")
    f.write("\n")
    f.write("module purge\n")
    f.write("module load slurm\n")
    f.write("module load cpu/0.17.3b gcc/10.2.0") 
    #f.write("module load gpu/0.15.4\n") 
    f.write("module load python/3.8.5\n")
    f.write("module load singularitypro/3.7\n")
    f.write("module load anaconda3/2020.11/conda\n")
    f.write("echo $CONDA_PYTHON_EXE\n") # finds and launches Anaconda 
    f.write("conda activate base\n")
    f.write("\n")    
    f.write("python ./sklearn-parallel.py\n")
# Python outputs should be found in the *.out file  
os.system(f"sbatch {sbatch_file}")
```





