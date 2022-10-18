#### 1. RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same

  Cause: This means that input --> cpu and model, i.e weigths --> gpu.
  
  Solution: [Stackoverflow](https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte)
