# Learning to Visualize Attribute Maps of trained models

Disclosing the Magic of Convolutional Neural Networks. While traditional algorithms might leave us with clear roadmaps to their conclusions, deep learning models often act like mysterious oracles, spitting out accurate results without revealing their inner workings. But don't despair; convolutional neural networks (CNNs) are just partially mysterious! Let's peek under the hood and see how these AI wizards learn to see the world.

![Alt Text](/Visualizacao/gandalf.gif)

To do that, we shall look into a CNN designed to perform semantic segmentation (pixel-wise classification), which is a task where every pixel of an image is classified within a class. To demonstrate this, we will use a pre-trained Fully Convolutional network capable of roughly identifying lithofacies from seismic images.

![alt text](/Visualizacao/sismica.png)

Imagine that a CNN is like a famous sorcerer, Sr. Network, with special skills to identify valuable resources and medicinal spices. Many people would like to hear what he says but distrust him, for they do not understand how this old creepy can tell a mushroom from another.

So you, a young mage adventurer, become his pupil and realize that this mysterious creature is not much more than a clever gentleman stuffed with equipment. You then realize that he is a pattern recognition master.

You discover that he has a stack of specialized lenses. Each lens, called a filter, focuses on a particular feature – like edges, textures, or even entire objects. As he scans an image, these filters slide across it, picking up on these visual clues. The resulting output, a feature map, reveals how strongly each feature appears in different image parts.

But the journey doesn't end there. Sometimes, like a detective zooming in on critical details, the Sr. Network shrinks the image while preserving crucial spatial information. This downsampling, or pooling, helps the Sr. Network focus on the bigger picture, making it less sensitive to minor shifts or rotations in the image. This allows him to identify the herbs even if they're standing at a different angle.

Stacking these filter and pooling layers, the Sr. Network builds a rich understanding of the image, from primary lines and shapes to complex objects and their relationships. This allows him to identify which plant is which and separate medicine from poison.

![alt text](/Visualizacao/planta.jpg)
Image Source: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.864486/full

So, while CNNs might not give us step-by-step instructions, they offer insights into their thought process through these feature maps. It's like looking at the sorcerer's notes and understanding the key pieces of evidence they considered before reaching their conclusion.

These feature maps hold the secrets to what the model pays attention to and how it builds its understanding of the observed objects. But just like old scrolls, they might not be instantly recognizable. Here's where we, as witchcraft apprentices, step in:

> Filters: The Specialized Lenses: Imagine the model has a collection of special lenses, each trained to detect specific visual clues. These filters come in different shapes and sizes, focusing on edges, textures, or even complete objects. As they scan an image, these filters leave behind maps highlighting where they found their clues – the feature maps.

>Understanding the Bigger Picture: However, directly interpreting these maps, especially with numerous filters and channels, can be like sifting through mountains of forensic data. This is where downsampling comes in, zooming out to see the bigger picture. By shrinking the image while preserving essential information, the model focuses on broader features, making it less sensitive to minor details.

> Feature Maps Tell the Story: Instead of deciphering individual maps, we can focus on their evolution across the network. Early maps capture intricate details, like individual leaves in a plant. As we move deeper, the maps reveal more abstract features, like the overall composition of the plant species. This progression tells how the model builds its understanding, from basic building blocks to complex interpretations.

> More Than Just Numbers: While the raw data might seem overwhelming, visualization tools come to the rescue. By converting the maps into heat maps or generating images based on their patterns, we can "see" what the model sees. Edges might glow, textures emerge, and objects take shape, helping us bridge the gap between the cold numbers and the visual world the model navigates. Composing these attributes in a model trained for segmentation, we can see the classes being slowly built up as the model develops.

#  Demonstration of visualization of feature maps using a model trained for semantic segmentation of seismic facies

We will need to import the necessary data, model and codes.

First of all import the dataset and trained models

The official codes are available at: https://github.com/brunoaugustoam/SSL_Seismic_Images

```python
# Downloading the sample of F3_netherlands dataset
!wget https://www.dropbox.com/sh/sake8dobi53r4l3/AAAhjUFtsKACXEC1t6kA0JuZa -O F3_netherlands.zip
!unzip F3_netherlands.zip

# Downloading the models trained for the F3_netherlands dataset
!wget https://www.dropbox.com/s/m2h4sgg01qh43w5/demo_models_ssl%20Bruno%20Monteiro.zip?dl=0 -O demo_models_ssl.zip
!unzip demo_models_ssl.zip

#Clone github repository
!git clone https://github.com/brunoaugustoam/SSL_Seismic_Images.git
!mv /content/SSL_Seismic_Images/demo/* /content/

#Install Required libraries
!pip install segyio
!pip install torchinfo
```
```python
from aux_functions_demo import *
from plots_demo import *
from dataloader_seismic_demo import *
from fcn_demo import *
from train_segmentation_demo import *

from pathlib import Path
from torchinfo import summary
import os
```

Lets define the set of parameters necessary to run this pretrained model.

It has a set of non obvious parameters as this model was pre-trained in a self-supervised manner and later on fine-tuned for segmenting facies.

This is a demo of the experiments performed for the paper "Self-Supervised Learning for Seismic Image Segmentation from Few-Labeled Samples", published at IEEE Geoscience and Remote Sensing letters, volume 19, 2022.

```python
root = '/content'

dataset = 'F3_netherlands'
pretask = 'jigsaw'

args = {
    'dataset'   : dataset,
    'pretask' : pretask,    
    'task'      : 'segmentation',
    'batch_size' : 1,
    'num_workers' : 2,
    'n_channels' : 1,
    'n_classes' : 10,
    'height' : 448,
    'width' : 448,
    'train_type' : 'fine_tune'
}

if args['pretask'] == 'jigsaw':
    args['pre_classes'] =  9
elif args['pretask'] == 'rotation':
    args['pre_classes'] =  5

#Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

n_few_shot = 5 


#For testing the fine-tuned model or only evaluating the loaded one, the test_set is used
test_set = SeismicDataset(root=root, dataset_name=dataset, split='test',
                            task=args['task'],train_type=args['train_type'],n_few_shot=n_few_shot)

dataloader_test = DataLoader(test_set,
                              batch_size=args['batch_size'],
                              shuffle=False,num_workers=args['num_workers'])


```
## Load fine-tuned model

```python
#If you just want to evaluate the already trained model, the following model can be used
finetuned_model = f'{dataset}_{pretask}_{n_few_shot}shot'
finetuned_model

# loss definition
criterion = set_criterion(args['task'],device)  #For Jigsaw, Rotation and Segmentation, will be set to CrossEntropyLoss


# Instantiating architecture - first with the pretext task
model = FCN(num_classes=args['pre_classes'], in_channels=args['n_channels'],task=pretask).to(device)

#Update the task for segmentation (used downstream task)
model.update_task(args['task'], args['n_classes'])

#After defining the correct model, upload the weights
model = model.to(device) #must be on GPU, as the trained models are cast to GPU
model.load_state_dict(torch.load(finetuned_model))

summary(model,input_size=(args['batch_size'],args['n_channels'],args['width'],args['height'] ))
```
![alt text](/Visualizacao/network_print.png)

## Now let us visualize the and the activation maps produced when our model sees an image

First we extract one image from our loader

```python
with torch.no_grad():
      for idx, (test_images, test_labels, name) in enumerate(dataloader_test):
        #Cast images to device
        test_images =test_images.to(device)
        test_labels =test_labels.to(device)
        y_hat = model(test_images)

        break
```
## Run only first convolutional layer

Here, we forward our model to the very first convolutional layer. The output of this layer consists of 64 channels, which are shown below. As we can see, this layer performs a simple convolution with downsampling. The feature map obtained is half the size of the original image, and the highlighted regions recognize features similar to the original texture of the image. As this is still the very first layer, the model is yet closer to the original image and sees raw features that resemble the image. 


```python
out = model.conv1(test_images)

n_rows = out.shape[1]//4
n_cols = 4
idx=0
for i in range(n_rows):

  fig, ax = plt.subplots(1,n_cols, figsize=(12,6))
  for j in range(n_cols):
    ax[j].imshow(out[0][idx].detach().cpu().numpy(), cmap='Greys')
    idx+=1
plt.show()
```
![alt text](/Visualizacao/out_conv1.png)

## Foward the image through the model until the end of the first set of convolutional layers

Here we can already see that the model is outputting a more coarse feature map. We can still see some channels highlighting distinct areas that resemble classes, others accentuating borders or similar features. 

```python
sub_model = nn.Sequential(model.conv1,
model.bn1,
model.relu,
model.dropout,
model.maxpool,
model.layer1,
)

out = sub_model(test_images)

n_rows = 18
n_cols = 6
idx=0
for i in range(n_rows):

  fig, ax = plt.subplots(1,n_cols, figsize=(12,6))
  for j in range(n_cols):
    ax[j].imshow(out[0][idx].detach().cpu().numpy(), cmap='Greys')
    idx+=1
plt.show()
```
![alt text](/Visualizacao/out_layer1.png)

## Foward the image through the model until the end of the second and third set of convolutional layers

This time, the features are even more abstract, but we can still notice some of the original texture in the end of the second set of layers. By the time the model outputs from the third set, the feature maps are more difficult to interpretate and start to create more complex features that are more similar to the disposal of classes and its borders. 

```python
sub_model = nn.Sequential(model.conv1,
model.bn1,
model.relu,
model.dropout,
model.maxpool,
model.layer1,
model.dropout,
model.layer2,
)

out = sub_model(test_images)

n_rows = 18
n_cols = 6
idx=0
for i in range(n_rows):

  fig, ax = plt.subplots(1,n_cols, figsize=(12,6))
  for j in range(n_cols):
    ax[j].imshow(out[0][idx].detach().cpu().numpy(), cmap='Greys')
    idx+=1
plt.show()
```
![alt text](/Visualizacao/out_layer2.png)

```python
sub_model = nn.Sequential(model.conv1,
model.bn1,
model.relu,
model.dropout,
model.maxpool,
model.layer1,
model.dropout,
model.layer2,
model.dropout,
model.layer3,
)

out = sub_model(test_images)

n_rows = 18
n_cols = 6
idx=0
for i in range(n_rows):

  fig, ax = plt.subplots(1,n_cols, figsize=(12,6))
  for j in range(n_cols):
    ax[j].imshow(out[0][idx].detach().cpu().numpy(), cmap='Greys')
    idx+=1
plt.show()
```
![alt text](/Visualizacao/out_layer3.png)

## Foward the image through the model until the end of the last set of convolutional layers, only skipping the final segmentation output

This time we fowarded the image through all our model, except the final segmentation layer. We can see that the feature maps are abstract and no longer resemble the original structure of the image. Instead, the map is more similar to the disposal of the classes in the seismic image. 


```python
sub_model = nn.Sequential(model.conv1,
model.bn1,
model.relu,
model.dropout,
model.maxpool,
model.layer1,
model.dropout,
model.layer2,
model.dropout,
model.layer3,
model.dropout,
model.layer4,)

out = sub_model(test_images)

n_rows = 18
n_cols = 6
idx=0
for i in range(n_rows):

  fig, ax = plt.subplots(1,n_cols, figsize=(12,6))
  for j in range(n_cols):
    ax[j].imshow(out[0][idx].detach().cpu().numpy(), cmap='Greys')
    idx+=1
plt.show()
```

![alt text](/Visualizacao/out_layer4.png)

If we compare against the final segmentation, we can (kind of) see the resemblance in the disposal of the feature maps and the disposal of the classes.

![alt text](/Visualizacao/sism_pred_mask.png)

### Fake Heatmap


Note: This example was built using a segmentation network, where in almost all images, every class is also present. Therefore, it makes no sense in creating a heatmap (Class Activation Map), for every class would be active in this case and we would no be able to make any sense of such heatmap. But would be a good sanity conference when dealing with classification or detection tasks. 
 
If we combine some of the feature maps, we can see how it "cluster" around some of the most activated classes, but as mentioned, this is not the best approach to analyse this kind of task.

![alt text](/Visualizacao/heat_fake.png)