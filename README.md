#Interrupt Training and when the CUDA is out of Memory
1. Make sure the batch size is below 128.
2. type
```
nvidia-smi
```
to check the proccess in CUDA and remember the job id.
3. type 
```
sudo kill -9 [PID]
```
