#!/bin/bash
### These are specific to my setup. I have a conda environment "python310" I use, then I 
### mount an NFS share, and I slightly lower the clock speed for my GPUs to avoid stability issues
#conda activate python310
#mount file-server.lan:/ /media/import
#nvidia-smi -i 0,1,2,3 -pm 1
#nvidia-smi -i 0,1,2,3 -ac 3615,1442