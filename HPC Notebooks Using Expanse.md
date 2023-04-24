In Anaconda, port forward to Expanse and login as usual:
```bash
ssh -Y -l <USERNAME> login.expanse.sdsc.edu 
```

In order to run jupyter notebooks through the Expanse shell, you must run the `galyleo` script from https://github.com/mkandes/galyleo that allows you to schedule a batch job via slurm that will run a jupyterlab server on Expanse that you can securely open in your browser.

Prepend galyleo install location:
```bash
export PATH="/cm/shared/apps/sdsc/galyleo:${PATH}"
```

Launching a notebook server:
```bash
galyleo launch --account <abc123> --partition shared --cpus 1 --memory 2 --time-limit 00:45:00 --env-modules cpu,anaconda3
```
It will generate a randomized url that you can copy and paste into your web browser of choice. If I understood the docs correctly, then the available python libraries depend on the virtual environment on your local machine, rather than from SDSC using the above method.


I've only tested this using the previous accounts from NANO110/120A. Normal Python scripts work, though CUDA-based GPU-computing scripts do not since prior ACCESS requested CPU-nodes only ("sbatch: error: Batch job submission failed: Requested node configuration is not available"):

```bash
galyleo launch -a csd709 -p shared --cpus 1 --memory 2 --gpus 1 --time-limit 01:30:00 --env-modules cpu,gpu,anaconda3
```


---

Another option is using a Singularity container that has all the python packages included (+ GPU drivers and CUDA libraries) found here `/cm/shared/apps/containers/singularity/`

More information on accessing Singularity using Expanse (I have not tried this method): https://cvw.cac.cornell.edu/ExpanseSing/runcontainer 

https://github.com/sdsc-hpc-training-org/hpc-training-2022/blob/main/week07_ml_tensorflow_pytorch/Intro_TensorFlow_PyTorch_Usage.pdf



---

Resources:

https://github.com/sdsc-hpc-training-org/basic_skills/tree/master/how_to_run_notebooks_on_expanse#galyleo

https://hpc-training.sdsc.edu/notebooks-101/notebook-101.html#download-example-notebooks

https://github.com/sdsc-hpc-training-org/hpc-training-2022

HPC (High powered computing) Examples:
` git clone https://github.com/sdsc-hpc-training-org/notebook-examples.git `

