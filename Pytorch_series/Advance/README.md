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

## 2. PyTorch Dataset
 
 We will used the **SIGNS dataset**. The dataset is hosted on google drive, download it [here](https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view).
 This will download the SIGNS dataset (~1.1 GB) contain photos of hands signs representing numbers between 0 and 5. Here is the structure of the data:

      SIGNS/
        train_signs/
            0_IMG_5864.jpg
            ...
        test_signs/
            0_IMG_5942.jpg
            ...
The images are named following {label}_IMG_{id}.jpg where the label is in [0, 5].

Once the download is complete, move the dataset into the data/SIGNS folder. Run python build_dataset.py which will resize the images to size (64,64). The new resized dataset will be located by default in data/64x64_SIGNS

#### Creating a PyTorch Dataset

`torch.utils.data` provides some nifty functionality for loading data. We use `torch.utils.data.Dataset`, which is an abstract class representing a dataset. To make our own `SIGNSDataset` class, we need to inherit the `Dataset` class and override the following methods:
`__len__`: so that len(dataset) returns the size of the dataset
`__getitem__`: to support indexing using dataset[i] to get the ith image

    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    class SIGNSDataset(Dataset):
        def __init__(self, data_dir, transform):      
            # store filenames
            # self.filenames = os.listdir(data_dir) or ...
            self.filenames = [os.path.join(data_dir, f) for f in self.filenames]

        # the first character of the filename contains the label
        self.labels = [int(filename.split('/')[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]
   
#### Transforms

When we return an image-label pair using `__getitem__` we apply a `transform` on the image. These transformations are a part of the `torchvision.transforms` package, that allow us to annotate the images easily. Consider the following composition of multiple transforms:

    train_transformer = transforms.Compose([
        transforms.Resize(64),              # resize the image to 64x64 
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor()])             # transform it into a PyTorch Tensor

When we apply `self.transform(image)` in `__getitem__`, we pass it through the above transformations before using it as a training example. The final output is a PyTorch Tensor. To augment the dataset during training, we also use the `RandomHorizontalFlip` transform when loading the image.

    train_dataset = SIGNSDataset(train_data_path, train_transformer)
    val_dataset = SIGNSDataset(val_data_path, eval_transformer)
    test_dataset = SIGNSDataset(test_data_path, eval_transformer)

#### Loading Data Batches

`torch.utils.data.DataLoader` provides an iterator that takes in a `Dataset` object and performs batching, shuffling and loading of the data. This is crucial when images are big in size and take time to load. In such cases, the GPU can be left idling while the CPU fetches the images from file and then applies the transforms.

In contrast, the `DataLoader` class (using multiprocessing) fetches the data asynchronously and prefetches batches to be sent to the GPU. Initializing the `DataLoader` is quite easy:

    train_dataloader = DataLoader(SIGNSDataset(train_data_path, train_transformer), 
                       batch_size=hyperparams.batch_size, shuffle=True,
                       num_workers=hyperparams.num_workers)

We can then iterate through batches of examples as follows:

    for train_batch, labels_batch in train_dataloader:
        # wrap Tensors in Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # pass through model, perform backpropagation and updates
        output_batch = model(train_batch)

Applying transformations on the data loads them as PyTorch `Tensors`. The for loop ends after one pass over the data, i.e., after one epoch. It can be reused again for another epoch without any changes. We can use similar data loaders for validation and test data.

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
  
![ptl](https://user-images.githubusercontent.com/82194525/196495949-b5c1e70f-dbd0-43b7-832a-6fa73a0c47c3.jpeg)

## 6. Using Multiple GPU's

We want to distribute the data across the available GPUs (If you have batch size of 16, and 2 GPUs, you might be looking providing the 8 samples to each of the GPUs), and not really spread out the parts of models across difference GPU's. This can be done as follows:

If you want to use all the available GPUs:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CreateModel()

    model= nn.DataParallel(model)
    model.to(device)

If you want to use specific GPUs: (For example, using 2 out of 4 GPUs)

    device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

    model = CreateModel()

    model= nn.DataParallel(model,device_ids = [1, 3])
    model.to(device)

To use the specific GPU's by setting OS environment variable:

Before executing the program, set CUDA_VISIBLE_DEVICES variable as follows:

    export CUDA_VISIBLE_DEVICES=1,3 (Assuming you want to select 2nd and 4th GPU)

Then, within program, you can just use DataParallel() as though you want to use all the GPUs. (similar to 1st case). Here the GPUs available for the program is restricted by the OS environment variable.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CreateModel()

    model= nn.DataParallel(model)
    model.to(device)

In all of these cases, the data has to be mapped to the device.

If X and y are the data:

    X.to(device)
    y.to(device)



  
