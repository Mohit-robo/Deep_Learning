#### 1. RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same

  Cause: This means that input --> cpu and model, i.e weigths --> gpu.
  
  Solution: [Stackoverflow](https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte)

#### 2. [Multiple GPU's] RuntimeError: module must have its parameters and buffers on device cuda:1 (device_ids[0]) but found one of them on device: cuda:2

  Cause: DataParallel requires every input tensor be provided on the first device in its device_ids list.
  
  Solution: [Stackoverflow](https://stackoverflow.com/questions/59249563/runtimeerror-module-must-have-its-parameters-and-buffers-on-device-cuda1-devi)
 
