## 1.Basics 

  [Notebook](https://github.com/Mohit-robo/Deep_Learning/blob/main/Pytorch_series/Advance/basics.ipynb)
  
  #### a. Switching from Torch Tensor to Numpy
  
    if torch.cuda.is_available():
      device = torch.device('cuda')

      x = torch.zeros(1,device= device)
      y = torch.zeros(1)
      y = y.to(device)    ## Assigned to the GPU, or what device it is
      z = x + y           ## Performed on GPU
      z = z.numpy()       ## Numpy can only handle tensors on CPU memory but z is on GPU memory, so this line gives an error as below.
    print(z)
  ##### Error
  
    TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
  ##### Solution
  
    z = z.cpu().numpy()   ## cpu() copies the same torch tensor to cpu memory so that Numpy is OK with it. 
  
 #### b. requires_grad
 
    a = torch.zeros_like(3,requires_grad = True)
    a.backward()
    
    '''
    Say variable a is declared and it's values need to be updated in the code later on, so set requires_grad = True. 
    
    '''
  PyTorch will automatically track and calculate gradients for that tensor. 
  Setting `requires_grad=True` tells PyTorch that this parameter should be optimized during the training process using backpropagation, when gradients are used
  to update weights. This is done with the `tensor.backward()` method; during this operation tensors with `requires_grad=True` will be used along with the tensor used     to call `tensor.backward()` to calculate the gradients.
  ##### Error
  
    a = torch.zeros_like(3,requires_grad = False)    ## Disable Differentiation
    a.backward()
    
    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    
  By switching the `requires_grad` flags to `False`, you can freeze part of your model and train the rest, no intermediate buffers will be saved, until the computation   gets to some point where one of the inputs of the operation requires the gradient. [Disabling Automatic Differentiation](https://aman.ai/primers/pytorch/#disabling-automatic-differentiation)
  
  ##### Using `torch.no_grad()`
  
  Using the context manager `torch.no_grad()` is a different way to achieve that goal: in the `no_grad` context, all the results of the computations will have        `requires_grad=False`, even if the inputs have `requires_grad=True`.
  Notice that you won’t be able to backpropagate the gradient to layers before the `torch.no_grad`. [Using `torch.no_grad()`](https://aman.ai/primers/pytorch/#using-torchno_grad)
  
##### Using `model.eval()`

If your goal is not to finetune, but to set your model in inference mode, the most convenient way is to use the `torch.no_grad` context manager. In this case you also have to set your model to evaluation mode, this is achieved by calling `eval()` on the `nn.Module`, for example:

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    
This operation sets the attribute `self.training` of the layers to `False`, which changes the behavior of operations like `Dropout` or `BatchNorm` that behave differently at training vs. test time.

## 2.Autograd

[Notebook](https://github.com/Mohit-robo/Deep_Learning/blob/main/Pytorch_series/Advance/autograd.ipynb)

#### Backward Function
    x = torch.ones(5,requires_grad= True)
    y = x+2
    
    Output: tensor([3., 3., 3., 3., 3.], grad_fn=<AddBackward0>)
    
  This happens due to `requires_grad = True` and `x` is added with some value to get the new variable that is a `gradient function`, that hold gradients of `x`.  
  ``requires_grad = True` cause a computational graph to be constructed, allowing us to later perform backpropagation through the graph. 
  
  In the above cell code:
  `x` is a Tensor with `requires_grad = True`, then after backpropagation `x.grad (y)` will be another Tensor holding the gradient of `x` with respect to some scalar     value.
  
    The graph for updating X 
        
              ---------------> Forward Prop
              
              X -------------
                            |
                            |-----> y
                            |
              2 -------------
              
         Backward Prop -------------------->                     
                            
                           
## 3. Torch.nn
       
   The `nn` package defines a set of Modules, which are roughly equivalent to neural network layers. A Module receives input Tensors and computes output Tensors, but may also hold internal state such as Tensors containing learnable parameters. The `nn` package also defines a set of useful loss functions that are commonly used when training neural networks.

#### `nn.Sequential`

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # After constructing the model we use the .to() method to move it to the
    # desired device.
    
    model = torch.nn.Sequential(
              torch.nn.Linear(D_in, H),
              torch.nn.ReLU(),
              torch.nn.Linear(H, D_out),
            ).to(device)           

#### Loss Functions
     
     torch.nn.MSELoss
              CrossEntropyLoss
              BCELoss
              L1Loss
              
  Apart from just `nn.Sequential`, `torch.nn` is also the parent class when creating models with Pytorch, i.e the model class inherits attributes from the `nn` Module.

## 4. torch.optim
  
  The optim package in PyTorch abstracts the idea of an optimization algorithm and provides implementations of commonly used optimization algorithms.
  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                            Adagrad
                            SGD
                            RMSProp

      # Before the backward pass, use the optimizer object to zero all of the
      # gradients for the Tensors it will update (which are the learnable weights
      # of the model)
      optimizer.zero_grad()

      # Backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()

      # Calling the step function on an Optimizer makes an update to its parameters
      optimizer.step()

## 5. The training loop

     for epoch in range(epochs):
        model.train()
        preds = model(X_train)
        loss = loss_fxn(preds, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

  #### Step 1: The Forward pass.
  
  Here, the model takes your data, feeds it forward through your network architecture, and comes up with a prediction.
  First, put the model in training mode using `model.train()`.
  Second, make predictions: `predictions = model(training_data)`.
  
  #### Step 2: Calculate the loss.
  
  Your model will start off making errors.
  These errors are the difference between your prediction and the ground truth.
  You can calculate this as: `loss = loss_fxn(predictions, ground_truth)`.
  
  #### Step 3: Zero gradients.
  
  You need to zero out the gradients for the optimizer prior to performing back propagation.
  If gradients accumulate across iterations, then your model won’t train properly.
  You can do this via `optimizer.zero_grad()`.
  
  #### Step 4: Backprop.
  
  Next, you compute the gradient of the loss with respect to model parameter via backprop.
  Only parameters with `requires_grad = True` will be updated.
  This is where the learning starts to happen.
  PyTorch makes it easy, all you do is call: `loss.backward()`.
  
  #### Step 5: Update the optimizer (gradient descent).
  
  Now it’s time to update your trainable parameters so that you can make better predictions.
  Remember, trainable means that the parameter has `requires_grad=True`.
  To update your parameters, all you do is call: `optimizer.step()`.

  
