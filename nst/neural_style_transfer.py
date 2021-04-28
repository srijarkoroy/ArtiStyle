'''
This file contains the main Neural Style Transfer module. 

The results may be obtained by running 'run_nst()'
'''

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms

import os
import errno
import time
import copy

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from nst.loss_norm import ContentLoss, StyleLoss, Normalization

class NST(object):

    def __init__(self, content_layers = ['conv_4'], style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], num_steps = 300, style_weight = 1000000, content_weight = 1,gpu = False, path = None):

        '''
        The constructor has the Parameters which are going to be used to generate the resulting image from the content and style image.

        Parameters:

        - content_layers(default: ['conv_4']): layer that extracts the content features

        - style_layers(default: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']): layer   that extracts style features

        - num_steps(default: 300): the number of epochs to run neural style transfer

        - style_weight(default: 1000000): the weight given to the style that is extracted from style image

        - content_weight(default: 1): the weight given to the content that is extracted from content image

        - gpu(default: False): run neural_style_transfer on gpu if True and is available

        - path(default: None): path to save the resulting image in jpg format 
        '''

        if gpu and not torch.cuda.is_available():
            raise ValueError('gpu is True but cuda not available')

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.path = path

        '''
        choosing which device to run the network on and setting image size accordingly
        '''

        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.img_size = 512 if torch.cuda.is_available() else 128

        '''
        importing the pre-trained 19 layer VGG network

        using the features module because we need the output of the individual convolution layers to measure content and style loss

        setting the network to evaluation mode using .eval() because layers have different behavior during training than evaluation
        '''

        self.cnn = models.vgg19(pretrained = True).features.to(self.device).eval()

        '''
        normalizing the image before sending it into the network, VGG networks are trained on images with each channel normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        '''

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    ''' 
    getting content and style image paths and loading them, creating a white noise input image, and returning the three tensors
    '''

    def image_loader(self, content, style):

        ''' the content and style image 'paths' must be entered '''

        ''' loading the content image '''

        if os.path.exists(content):
            content_img = Image.open(content)

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), content)

        ''' loading the style image '''
        
        if os.path.exists(style):
            style_img = Image.open(style)

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), style)

        ''' resizing the style image according to the content image '''

        width, height = content_img.size
        style_img = style_img.resize((width, height))

        loader = transforms.Compose([transforms.ToTensor()])

        '''
        transforming the content and style images to tensors and creating a white noise input image
        ''' 

        content_img = loader(content_img).unsqueeze(0)
        style_img = loader(style_img).unsqueeze(0)
        input_img = torch.randn(content_image.data.size(), device = self.device)

        content_img = content_img.to(self.device, torch.float)
        style_img = style_img.to(self.device, torch.float)
        input_img = input_img.to(self.device, torch.float)

        return content_img, style_img, input_img

    def style_model_and_losses(self, cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img):

        '''
        adding our content loss and style loss layers immediately after the convolution layer they are detecting by creating a new Sequential module that has content loss and style loss modules correctly inserted
        '''

        content_layers = self.content_layers
        style_layers = self.style_layers

        cnn = copy.deepcopy(self.cnn)

        ''' normalization module '''
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)

        '''
        declaring content_loss and style_loss lists to have an iterable access to content and style losses
        '''

        content_loss = []
        style_loss = []

        '''
        assuming cnn to be sequential we create a new Sequential module that has content loss and style loss modules correctly inserted to go with other modules activated sequentially
        '''

        model = nn.Sequential(normalization)

        i = 0   # increment when we find a conv layer
        for layer in self.cnn.children():

            if isinstance(layer, nn.Conv2d):
                i+=1
                name = 'conv_{}'.format(i)
            
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                '''
                in-place version doesn't play very nicely with the ContentLoss and StyleLoss we insert below. So we replace with out-of-place ones here
                '''
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i) 

            else:
                raise RuntimeError('Layer {} is not recognized'.format(layer.__class__.__name__))            

            model.add_module(name, layer)

            ''' adding content loss '''
            for name in content_layers:
                target = model(content_img).detach()
                cnt_loss = ContentLoss(target)
                model.add_module('content_loss_{}'.format(i), cnt_loss)
                content_loss.append(cnt_loss)

            ''' adding style loss '''
            for name in style_layers:
                target_feature = model(style_img).detach()
                styl_loss = StyleLoss(target_feature)
                model.add_module('style_loss_{}'.format(i), styl_loss)
                style_loss.append(styl_loss)

        ''' trimming off layers present after the last content and style loss '''
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i+1)]

        return model, content_loss, style_loss
    
    def run_nst(self, content_path, style_path):
    
        ''' initialising the variables '''

        num_steps = self.num_steps
        style_weight = self.style_weight
        content_weight = self.content_weight

        ''' loading the images using the image_loader function '''

        content_img, style_img, input_img = image_loader(content_path, style_path)

        ''' building the style transfer model '''

        model , content_loss, style_loss = style_model_and_losses(self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std, content_img, style_img)

        ''' using LBFGS optimizer to run gradient descent on the input image in order to minimise the content and the style losses '''

        optimizer = optim.LBFGS([input_img.requires_grad_()])

        print('Optimizing')
        run = [0]
        while run[0] <= num_steps:

            ''' The function closure computes the losses multiple times after every step and updates the model after every iteration '''

            def closure():

                ''' restricting the pixel values of the input image between 0 and 1 '''

                input_img.clamp(0,1)

                optimzer.zero_grad()
                model(input_img)
                content_score = 0
                style_score = 0

                ''' updating the content and the style error by adding the losses returned after every iteration '''

                for cl in content_loss:
                    content_score+=cl
                for sl in style_loss:
                    style_score+=sl

                ''' multipying the errors generated with the weight assigned to the content and the style model respectively '''

                style_score *= style_weight
                content_score *= content_weight 

                ''' calculating the total error by adding the style and the content errors '''

                loss = content_score + style_score

                ''' computing the gradient with respect to all the parameters associated with the loss function '''

                loss.backward()

                run[0]+=1

                ''' displaying the loss after every 50th iteration '''

                if run[0] % 50 == 0:
                    print("Run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            ''' running the gradient descent on each iteration '''

            optimizer.step(closure)

            ''' restricting the pixel values of the input image between 0 and 1 '''

            input_img.clamp(0,1)

            ''' converting the tensor to a PIL Image '''
            
            unloader = transforms.ToPILImage()

            output_img = input_img.cpu().clone()  # cloning the tensor to not do changes on it
            output_img = output_img.squeeze(0)      # removing the fake batch dimension
            output_img = unloader(output_img)

            ''' saving the image if the user wants to '''
            
            if self.path is not None:
                output_img.save(self.path)

            return output_img
