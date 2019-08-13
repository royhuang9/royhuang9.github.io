Install Nvidia Docker 19.03 on Ubuntu 16.04 with GTX 960M
======
Eventually I successfully made my old GTX 960M work with nvidia docker 19.03 after suffering a whole day's struggling. Here is what I did and the problems I encountered.

## Preparation
I am using the integrated graphic card for display and this old 4GB GTX 960M nvidia graphic card to run some simple deep learning models. If pytorch or tensorflow need to run on it, I will run the command `nvidia-smi` first to trigger the loading of nvidia drivers.

Make sure the nvidia driver works by `nvidia-smi`:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 960M    Off  | 00000000:02:00.0 Off |                  N/A |
| N/A   40C    P0    N/A /  N/A |      0MiB /  4046MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
So the driver version is 418.56 and memroy size is 4046MiB.

## Step 1 Install Docker
## Step 2 Install Nvidia Docker 19.03
follow the instructions on the github page
```
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

But when I tested it with nvidia-smi, I encountered my the first problem.
```
$ docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```
The error messages are:
```
docker: Error response from daemon: OCI runtime create failed: container_linux.go:345: starting container process caused "process_linux.go:430: container init caused \"process_linux.go:413: running prestart hook 0 caused \\\"error running hook: exit status 1, stdout: , stderr: exec command: [/usr/bin/nvidia-container-cli --load-kmods configure --ldconfig=@/sbin/ldconfig.real --device=all --compute --utility --require=cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411 --pid=23943 /var/lib/docker/overlay2/64d7664d70450002cf8b8827085b611df730e19e34c341ebd1d86ed45e76822a/merged]\\\\nnvidia-container-cli: initialization error: driver error: failed to process request\\\\n\\\"\"": unknown.
ERRO[0002] error waiting for container: context canceled
```
## Step 3 Fix initialization error
After searching the issues in nvidia-docker github, someone suggests run command `sudo nvidia-container-cli -k -d /dev/tty info` to debug it and I got the following:

```
-- WARNING, the following logs are for debugging purposes only --

I0812 23:50:33.026178 24180 nvc.c:281] initializing library context (version=1.0.3, build=bfb23a2ed0c2e045f4586a367e989883f2752821)
I0812 23:50:33.026227 24180 nvc.c:255] using root /
I0812 23:50:33.026232 24180 nvc.c:256] using ldcache /etc/ld.so.cache
I0812 23:50:33.026236 24180 nvc.c:257] using unprivileged user 65534:65534
I0812 23:50:33.027200 24181 nvc.c:191] loading kernel module nvidia
I0812 23:50:33.027330 24181 nvc.c:203] loading kernel module nvidia_uvm
I0812 23:50:33.027416 24181 nvc.c:211] loading kernel module nvidia_modeset
I0812 23:50:33.027643 24182 driver.c:133] starting driver service
E0812 23:50:33.027990 24182 driver.c:197] could not start driver service: load library failed: libnvidia-fatbinaryloader.so.418.56: cannot open shared object file: no such file or directory
I0812 23:50:33.028067 24180 driver.c:233] driver service terminated successfully
nvidia-container-cli: initialization error: driver error: failed to process request
```
It seems like nvidia-docker cannot find the nvidia library `libnvidia-fatbinaryloader.so.418.56`. But nvidia-smi also need these libraries, why couldn't nvidia-docker find it?

This library is located in `/usr/lib/nvidia-418/`
I have added this path to LD_LIBRARY_PATH, run 
```
echo "$LD_LIBRARY_PATH"
```
The result is:
```
/usr/local/cuda/lib64:/usr/lib/nvidia-418/
```
It seems like nvidia-docker does not check LD_LIBRARY_PATH.

Actually, nvidia-docker depends on /etc/ld.so.cache, which is a temporary file generated by `ldconfig` during system boot.

You can check the content of ld.so.cache by `ldconfig -p | grep nvidia` to find out whether `libnvidia-fatbinaryloader.so.418.56` is there:

```
	libnvidia-opencl.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
	libnvidia-gtk3.so.418.56 (libc6,x86-64) => /usr/lib/libnvidia-gtk3.so.418.56
	libnvidia-gtk3.so.375.26 (libc6,x86-64) => /usr/lib/libnvidia-gtk3.so.375.26
	libnvidia-gtk2.so.418.56 (libc6,x86-64) => /usr/lib/libnvidia-gtk2.so.418.56
	libnvidia-gtk2.so.375.26 (libc6,x86-64) => /usr/lib/libnvidia-gtk2.so.375.26
	libnvidia-container.so.1 (libc6,x86-64, OS ABI: Linux 3.10.0) => /usr/lib/x86_64-linux-gnu/libnvidia-container.so.1
```

In order to make ld.so.cache include the librararies in directory /usr/lib/nvidia-418, I created a file name `cuda.conf` under /etc/ld.so.conf.d/.
```
# library for nvidia driver
/usr/local/cuda/lib64
/usr/lib/nvidia-418

```
And run `sudo ldconfig` to regenerate `ld.so.cache`, then `sudo ldconfig -p | grep nvidia`. I got:
```
	libnvoptix.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvoptix.so.1
	libnvidia-tls.so.418.56 (libc6,x86-64, OS ABI: Linux 2.3.99) => /usr/lib/nvidia-418/libnvidia-tls.so.418.56
	libnvidia-rtcore.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-rtcore.so.418.56
	libnvidia-ptxjitcompiler.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-ptxjitcompiler.so.1
	libnvidia-opticalflow.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-opticalflow.so.1
	libnvidia-opencl.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
	libnvidia-ml.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-ml.so.1
	libnvidia-ml.so (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-ml.so
	libnvidia-ifr.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-ifr.so.1
	libnvidia-ifr.so (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-ifr.so
	libnvidia-gtk3.so.418.56 (libc6,x86-64) => /usr/lib/libnvidia-gtk3.so.418.56
	libnvidia-gtk3.so.375.26 (libc6,x86-64) => /usr/lib/libnvidia-gtk3.so.375.26
	libnvidia-gtk2.so.418.56 (libc6,x86-64) => /usr/lib/libnvidia-gtk2.so.418.56
	libnvidia-gtk2.so.375.26 (libc6,x86-64) => /usr/lib/libnvidia-gtk2.so.375.26
	libnvidia-glvkspirv.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-glvkspirv.so.418.56
	libnvidia-glsi.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-glsi.so.418.56
	libnvidia-glcore.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-glcore.so.418.56
	libnvidia-fbc.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-fbc.so.1
	libnvidia-fbc.so (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-fbc.so
	libnvidia-fatbinaryloader.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-fatbinaryloader.so.418.56
	libnvidia-encode.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-encode.so.1
	libnvidia-encode.so (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-encode.so
	libnvidia-eglcore.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-eglcore.so.418.56
	libnvidia-egl-wayland.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-egl-wayland.so.1
	libnvidia-container.so.1 (libc6,x86-64, OS ABI: Linux 3.10.0) => /usr/lib/x86_64-linux-gnu/libnvidia-container.so.1
	libnvidia-compiler.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-compiler.so.418.56
	libnvidia-compiler.so (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-compiler.so
	libnvidia-cfg.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-cfg.so.1
	libnvidia-cfg.so (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-cfg.so
	libnvidia-cbl.so.418.56 (libc6,x86-64) => /usr/lib/nvidia-418/libnvidia-cbl.so.418.56
	libnvcuvid.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libnvcuvid.so.1
	libnvcuvid.so (libc6,x86-64) => /usr/lib/nvidia-418/libnvcuvid.so
	libOpenGL.so.0 (libc6,x86-64) => /usr/lib/nvidia-418/libOpenGL.so.0
	libOpenGL.so (libc6,x86-64) => /usr/lib/nvidia-418/libOpenGL.so
	libGLdispatch.so.0 (libc6,x86-64) => /usr/lib/nvidia-418/libGLdispatch.so.0
	libGLX_nvidia.so.0 (libc6,x86-64) => /usr/lib/nvidia-418/libGLX_nvidia.so.0
	libGLX.so.0 (libc6,x86-64) => /usr/lib/nvidia-418/libGLX.so.0
	libGLX.so (libc6,x86-64) => /usr/lib/nvidia-418/libGLX.so
	libGLESv2_nvidia.so.2 (libc6,x86-64) => /usr/lib/nvidia-418/libGLESv2_nvidia.so.2
	libGLESv2.so.2 (libc6,x86-64) => /usr/lib/nvidia-418/libGLESv2.so.2
	libGLESv2.so (libc6,x86-64) => /usr/lib/nvidia-418/libGLESv2.so
	libGLESv1_CM_nvidia.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libGLESv1_CM_nvidia.so.1
	libGLESv1_CM.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libGLESv1_CM.so.1
	libGLESv1_CM.so (libc6,x86-64) => /usr/lib/nvidia-418/libGLESv1_CM.so
	libGL.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libGL.so.1
	libGL.so (libc6,x86-64) => /usr/lib/nvidia-418/libGL.so
	libEGL_nvidia.so.0 (libc6,x86-64) => /usr/lib/nvidia-418/libEGL_nvidia.so.0
	libEGL.so.1 (libc6,x86-64) => /usr/lib/nvidia-418/libEGL.so.1
	libEGL.so (libc6,x86-64) => /usr/lib/nvidia-418/libEGL.so
```
Even nvidia-docker works:

```
$ docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
Tue Aug 13 00:06:54 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 960M    Off  | 00000000:02:00.0 Off |                  N/A |
| N/A   40C    P0    N/A /  N/A |      0MiB /  4046MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Hooray!!!

## Step 4 Fix login problem
There is still a big problem to fix: after logout or reboot, I cannot login into my desktop. Every time when I enter the password, it just comes back. Switch to console by "CTRL + ALT + F1", I can login console.
There is a .xsession-errors file under home directory. The content is:

```
X Error of failed request:  BadValue (integer parameter out of range for operation)
  Major opcode of failed request:  154 (GLX)
  Minor opcode of failed request:  3 (X_GLXCreateContext)
  Value in failed request:  0x0
  Serial number of failed request:  27
  Current serial number in output stream:  28
openConnection: connect: No such file or directory
cannot connect to brltty at :0
```

After study related page on github, I lost in the GLX support problem with nvidia-docker. I almost give up. Finally I decided to rename the file `/etc/ld.so.conf.d/cuda.conf`  to `/etc/ld.so.conf.d/zzz_nvidia.conf`. It will make the libraries under `/usr/local/cuda/lib64 and /usr/lib/nvidia-418/` loaded at last.

Anyway it works. Nvidia-docker works and I can login successully.

Magic!!!