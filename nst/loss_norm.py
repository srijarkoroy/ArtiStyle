'''
This file contains classes for ContentLoss, StyleLoss and Normalization.

ContentLoss: takes the feature maps of a layer in a network processing input and returns the weighted content distance

StyleLoss: computes feature correlations given by the Gram matrix where the given matrix is a reshaped version of the feature maps, returns the style distance

Normalization: returns normalized input image which can easily be put it in nn.Sequential
'''

import torch

import torch.nn as nn
import torch.nn.functional as F

# content loss
class ContentLoss(nn.Module):

    '''
    represents a weighted version of the content distance for an individual layer
    '''

    def __init__(self, target,):
        super(ContentLoss, self).__init__()

        '''
        We perform 'detach' on the target content from the tree used to dynamically compute the gradient to prevent the forward method from throwing an error
        '''

        self.target = target.detach()

    def forward(self, input):

        '''
        computes the mean square error between the two sets of feature maps and is saved as a parameter of the module, returns the convolution layer’s input
        '''

        self.loss = F.mse_loss(input, self.target)
        return input

# style loss
class StyleLoss(nn.Module):

    '''
    computes the correlations between the different filter responses given by the Gram Matrix and act as a transparent layer in a network that computes the style loss of that layer
    '''

    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_features).detach()

    def gram_matrix(self, input):

        '''
        result of multiplying a given matrix by its transposed matrix, the given matrix here being a reshaped version of the feature maps F_XL.
        G(l){i j} is the inner product between the vectorised feature map i and j in layer l
        '''

        batch_size, num_feature_maps, dim1, dim2 = input.size()
        
        '''
        resizing F_XL into a KxN matrix, where K is the number of feature maps at layer L and N is the length of any vectorized feature map
        '''

        features = input.view(batch_size * num_feature_maps, dim1 * dim2)

        '''
        computing the gram_matrix elements and normalizing them by dividing by the number of elements in each feature map
        '''

        gram_prod = torch.mm(features, features.t())
        return gram_prod.div(batch_size * num_feature_maps * dim1 * dim2)

    def forward(self, input):

        '''
        computes the mean-squared distance between the entries of the Gram matrix from the original image and the Gram matrix of the image to be generated and is saved as a parameter of the module, returns the convolution layer’s input
        '''

        gram_prod = self.gram_matrix(input)
        self.loss = F.mse_loss(gram_prod, self.target)
        return input

# normalization
class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        '''
        we have to make mean and std [num_channels x 1 x 1] so that they may directly work with image tensor of shape [batch_size x num_channels x height x width]
        '''

        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):

        '''
        normalizing the image using the normalization formula
        '''
        return (img - self.mean) / self.std
