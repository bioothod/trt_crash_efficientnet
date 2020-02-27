# trt_crash_efficientnet
Test repository to show TensorRT 6/7 crash during conversion

Steps to reproduce:
```
$ docker run --ulimit core=-1 --network=host -ti --user=`id -u`:`id -g` --runtime=nvidia -v `pwd`:`pwd` -w `pwd` --rm -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0" nvcr.io/nvidia/tensorflow:20.02-tf2-py3 python3 ./test.py --output_dir test/test --num_classes 8 --model_name efficientnet-b0
```

Can be reproduced both with `nvcr.io/nvidia/tensorflow:20.02-tf2-py3` and `tensorflow/tensorflow:2.1.0-gpu-py3`

Stack trace (`tensorflow/tensorflow:2.1.0-gpu-py3` image):
```
(gdb) bt
#0  0x00007fff60219560 in tensorflow::Node::name() const () from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/../libtensorflow_framework.so.2
#1  0x00007fff6796af66 in tensorflow::tensorrt::convert::UpdateToEngineNode(std::vector<tensorflow::tensorrt::convert::EngineInfo, std::allocator<tensorflow::tensorrt::convert::EngineInfo> > const&, unsigned long, std::vector<tensorflow::Node*, std::allocator<tensorflow::Node*> > const&, bool, std::string const&, tensorflow::Node**, int*) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#2  0x00007fff6796cfc5 in tensorflow::tensorrt::convert::CreateTRTNode(tensorflow::tensorrt::convert::ConversionParams const&, std::vector<tensorflow::tensorrt::convert::EngineInfo, std::allocator<tensorflow::tensorrt::convert::EngineInfo> > const&, int, int, tensorflow::Graph*, nvinfer1::IGpuAllocator*, std::vector<tensorflow::Node*, std::allocator<tensorflow::Node*> >*) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#3  0x00007fff67972041 in tensorflow::tensorrt::convert::ConvertAfterShapes(tensorflow::tensorrt::convert::ConversionParams const&) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#4  0x00007fff679ac34b in tensorflow::tensorrt::convert::TRTOptimizationPass::Optimize(tensorflow::grappler::Cluster*, tensorflow::grappler::GrapplerItem const&, tensorflow::GraphDef*) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#5  0x00007fff6a31f937 in tensorflow::grappler::MetaOptimizer::RunOptimizer(tensorflow::grappler::GraphOptimizer*, tensorflow::grappler::Cluster*, tensorflow::grappler::GrapplerItem*, tensorflow::GraphDef*, tensorflow::grappler::MetaOptimizer::GraphOptimizationResult*) () from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#6  0x00007fff6a320ccd in tensorflow::grappler::MetaOptimizer::OptimizeGraph(tensorflow::grappler::Cluster*, tensorflow::grappler::GrapplerItem const&, tensorflow::GraphDef*) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#7  0x00007fff6a322714 in tensorflow::grappler::MetaOptimizer::Optimize(tensorflow::grappler::Cluster*, tensorflow::grappler::GrapplerItem const&, tensorflow::GraphDef*) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#8  0x00007fff6371cc07 in TF_OptimizeGraph(GCluster, tensorflow::ConfigProto const&, tensorflow::MetaGraphDef const&, bool, std::string const&, bool, TF_Status*) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#9  0x00007fff637217d6 in _wrap_TF_OptimizeGraph () from /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/_pywrap_tensorflow_internal.so
#10 0x000000000050a8af in ?? ()
#11 0x000000000050c5b9 in _PyEval_EvalFrameDefault ()
#12 0x0000000000508245 in ?? ()
#13 0x000000000050a080 in ?? ()
#14 0x000000000050aa7d in ?? ()
```
