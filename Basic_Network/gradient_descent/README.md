### Background

Logistic regression is a supervised learning algorithm, which implies that the expected outputs, i.e., 
ground-truth labels Y corresponding to each training sample in X are available during training.

Logistic regression takes a regular linear regression z and applies a hypothesis function on top of it (more on the hypothesis
function in the section on sigmoid function).

#### Formally:

    h(z)=11+e−zwhere, z=θ0x0+θ1x1+θ2x2+…+θNxN=θTx(i)

Note that in some literature, z is referred to as the ‘logit’, which essentially serves as the input to the hypothesis function.

Note that the parameter vector θ is also referred to simply, as “weights”. In some contexts, weights are referred to using a w vector or W matrix.

### Sigmoid Function

In the figure below from the section on supervised machine learning, the transformation function F(⋅) that operates on parameters θ
and the feature vector X(i) to yield the output Y^, is also known as the hypothesis function.

![25](https://user-images.githubusercontent.com/82194525/196841617-e06c3c1b-63e6-4067-903b-fcb0c9e04759.jpg)

Specifically, h(x(i);θ) denotes the hypothesis function, where i is used to denote the ith observation or data point.
In the context of tweets, it’s the ith tweet in the dataset.

In logistic regression, the sigmoid function serves as the hypothesis function. Logistic regression makes use of the sigmoid function which
squeezes/collapses/transforms the input into an output range of [0,1],and can interpreted as a probability measure.
Note that outside of the context of logistic regression, specifically in neural networks, the sigmoid function serves as a
non-linearity and is denoted by σ(⋅).

#### Formally:

    h(x(i);θ)=σ(x(i);θ)=11+e−θTx(i)

Visually, the sigmoid function is shown below – it approaches 0 as the dot product of θTx(i) approaches −∞ and 1 as the dot product approaches ∞.

![image](https://user-images.githubusercontent.com/82194525/196841883-62ad7b6e-cc68-45a3-b85b-7c1b5bd72444.png)

### Classification Using Logistic Regression

For classification, a threshold is needed. Usually, it is set to be 0.5 and this value corresponds to a dot product θTx(i) equal to 0.

As shown below, whenever the dot product is ≥ 0, the prediction is positive, and whenever the dot product is < 0, the prediction is negative.
Formally,

  ![image](https://user-images.githubusercontent.com/82194525/196842104-13d2bb84-f450-4b32-9570-f02ba90be536.png)

![image](https://user-images.githubusercontent.com/82194525/196842146-e5b811da-838a-4cde-9389-617f97cf9432.png)

### Computing Gradients

Since computing gradients are the biggest component of the overall training process, we detail the process here.
To update your weight vector θ, you will apply gradient descent to iteratively improve your model’s predictions.

The gradient of the cost function `J(θ)` with respect to one of the weights `θj` is denoted by `∂J(θ) / ∂θj` (or equivalently, `∇θjJ(θ))` as below. 
“Partial Gradient of the Cost Function for Logistic Regression” delves into the derivation of this equation.

![image](https://user-images.githubusercontent.com/82194525/196842711-a570dda7-6122-4e03-a29b-a9fb54c4fa1a.png)

where,

    i is the index across all m training examples.
    j is the index of the weight θj, so xj is the feature associated with weight θj.
    y^(i) (or equivalently, h(X(i);θ)) is the model’s prediction for the ith sample.

### Update Rule

To update the weight θj, we adjust it by subtracting a fraction of the gradient determined by α:

    θj:=θj−α×∇θjJ(θ)
    
    where := denotes a paramater/variable update.

Note that the above weight-update equation is called as the update rule in some literature.
The learning rate α, or the step size, is a value that helps us choose to control how big of a step a single update during our training process would involve.

To train your logistic regression classifier, i.e.,learn θ from scratch, iterate until you find the set of parameters θ that minimize your cost function
until convergence.

### In a nutshell

1. **Initialize parameters** : Initialize the parameter vector θ randomly or with zeros.

2. **Calculate logits**: The logits z are calculated using the dot product of the feature vector x with the weight vector θ, i.e., z=θTx.

3. **Apply a hypothesis function**: Use the sigmoid function h(⋅) element-wise to transform each feature of each sample.

4. **Generate the final prediction**: Generate the final prediction y^ by comparing each element obtained at the output of the prior step with a static threshold, of say 0.5 and inferring a 1 if the element ≥ 0, else 0.

5. **Compute the cost function**: Compute your cost function J(θ) to determine how far we are from our expected output (note that J(θ) can be viewed as a measure of our unhappiness quotient towards the model’s accuracy).
    
    The cost function basically tells us if we need more iterations before we converge to a minimized loss (or can be bounded by a “maximum” cap on the number of iterations). Typically, after several iterations, you reach your minima where your cost is at its minimum and you would end your training here.

6. **Calculate gradients**: Calculate the gradients of your cost function w.r.t. each parameter in the parameter vector θ.

7. **Apply update rule**: Update your parameter vector θ in the direction of the gradient of your cost function using your update rule.

After a set of iterations, you traverse along the loss curve (shown by the green dots in the figure above) and get closer to your minima where you would end your training here when you converge (shown by the orange dot).

![image](https://user-images.githubusercontent.com/82194525/196843936-6ee2ea3d-fec5-4c4d-bf7a-ba02eec243fd.png)
