# trt_crash_efficientnet
Test repository to show TensorRT 6/7 crash during conversion

Steps to reproduce:
```
$ docker run --ulimit core=-1 --network=host -ti --user=`id -u`:`id -g` --runtime=nvidia -v `pwd`:`pwd` -w `pwd` --rm -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0" nvcr.io/nvidia/tensorflow:20.02-tf2-py3 python3 ./test.py --output_dir test/test --num_classes 8 --model_name efficientnet-b0
```

Can be reproduced both with `nvcr.io/nvidia/tensorflow:20.02-tf2-py3` and `tensorflow/tensorflow:2.1.0-gpu-py3`
