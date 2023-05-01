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
Alternatively the following combines the two prior steps:
```
/cm/shared/apps/sdsc/galyleo/galyleo.sh launch --account csd709 --partition shared --cpus 3 --memory 32 --time-limit 01:30:00 --env-modules cpu,anaconda3
```

It will generate a randomized url that you can copy and paste into your web browser of choice. If I understood the docs correctly, then the available python libraries depend on the virtual environment on your local machine, rather than from SDSC using the above method.


```bash
galyleo launch -a csd709 -p shared --cpus 1 --memory 2 --gpus 1 --time-limit 01:30:00 --env-modules cpu,gpu,anaconda3
```


---

Another option is using a Singularity container that has all the python packages included (+ GPU drivers and CUDA libraries) found here `/cm/shared/apps/containers/singularity/`

More information on accessing Singularity using Expanse (I have not tried this method): https://cvw.cac.cornell.edu/ExpanseSing/runcontainer 

https://github.com/sdsc-hpc-training-org/hpc-training-2022/blob/main/week07_ml_tensorflow_pytorch/Intro_TensorFlow_PyTorch_Usage.pdf

### Enabling TensorFlow Example

Change allocation "xyz123", number of nodes, time limit, and other launch params as needed. 

```
[1] CPU node:
/cm/shared/apps/sdsc/galyleo/galyleo.sh launch -A xyz123 -p shared -n 16 -M 32 -t 00:30:00 -e singularitypro/ -s /cm/shared/apps/containers/singularity/keras-v2.2.4-tensorflow-v1.12-gpu-20190214.simg -d /home/$USER/tensorflow


[2] GPU node:
/cm/shared/apps/sdsc/galyleo/galyleo.sh launch -A xyz123 -p gpu-shared -n 10 -M 93 -G 1 -t 00:30:00 -e singularitypro/ -s /cm/shared/apps/containers/singularity/tensorflow/tensorflow-2.3.0-gpu-20200929.simg -d /home/$USER/tensorflow



---

Resources:

https://github.com/sdsc-hpc-training-org/basic_skills/tree/master/how_to_run_notebooks_on_expanse#galyleo

https://hpc-training.sdsc.edu/notebooks-101/notebook-101.html#download-example-notebooks

https://github.com/sdsc-hpc-training-org/hpc-training-2022

HPC (High powered computing) Examples:
` git clone https://github.com/sdsc-hpc-training-org/notebook-examples.git `


```

`ls /cm/shared/apps/containers/singularity/`



```python
import multiprocessing
multiprocessing.cpu_count()
```




    128




```python

```
