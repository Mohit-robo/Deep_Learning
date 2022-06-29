

from torchvision import models

## List all the  models  present
dir(models)

## View the backend architecture of Alxenet
alexnet = models.AlexNet()
alexnet

## View the backend architecture of resnet
resnet = models.resnet101(pretrained = True) 
resnet

# Loading instance for applying augmentation on images

from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

from PIL import Image
img = Image.open("/content/temp.jpg")
img

# Apply transformations on images
img_t = preprocess(img)

import torch
batch_t = torch.unsqueeze(img_t, 0)   ## Refer: https://pytorch.org/docs/stable/generated/torch.squeeze.html

resnet.eval()

"""With Resnet

vector of 1,000 scores, one per ImageNet class.
"""

res = resnet(batch_t)

# Load all the class labels from the file
with open('/content/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

#Returns the maximum value of all elements in the input tensor.
_, index = torch.max(res, 1)

percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

_, indices = torch.sort(res, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

"""With AlexNet"""

alex = alexnet(batch_t)
# Load all the class labels from the file
with open('/content/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

#Returns the maximum value of all elements in the input tensor.
_, index = torch.max(alex, 1)

percentage = torch.nn.functional.softmax(alex, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

_, indices = torch.sort(alex, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

