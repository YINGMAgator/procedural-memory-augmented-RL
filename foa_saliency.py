# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:45:00 2019

@author: meser

foa_saliency.py - Contains functions for generating saliency maps and
searching for regions of interest to be bounded.

"""

# Standard Library Imports

# 3P Imports
import numpy as np
import matplotlib

# Local Imports
from foa_image import ImageObject


def build_bounding_box(center, boundLength=32):

    ''' Generate the coordinates that defined a square region
    around the detected region of interest.
    center - Center coords of the object
    boundLength - Length and width of the square saliency region '''

    # Take most recent center coordinate
    R = center[0]
    C = center[1]

    # Dictionary for Clarity
    boxSize = {'Row': boundLength, 'Column': boundLength}

    # Derive upper left coordinate of bounding region
    R1 = int(R - (boxSize['Row'] / 2))
    if (R1 < 0):
        R1 = 0
    C1 = int(C - (boxSize['Column'] / 2))
    if (C1 < 0):
        C1 = 0

    # Derive lower right coordinate of bounding region
    R2 = int(R + (boxSize['Row'] / 2))
    if (R2 > 223):
        R2 = 223
    C2 = int(C + (boxSize['Column'] / 2))
    if (C2 > 255):
        C2 = 255

    return [R1, R2, C1, C2]


def salience_scan(image=ImageObject, rankCount=8, boundLength=32):

    ''' Saliency Map Scan

    Scan through the saliency map with a square region to find the
    most salient pieces of the image. Done by picking the maximally intense
    picture and bounding the area around it

    image - ImageObject being scanned
    rankCount - Number of objects to acquire before stopping
    boundLength - Length and width of the square saliency region '''

    # Copy salience map for processing
    smap = np.copy(image.salience_map)
    image.patched_sequence = np.empty((0, smap.shape[0], smap.shape[1]))
    
    
    ori = np.copy(image.original)
    img_sz = 32
    #ori = np.ones(smap.shape)
    image.patch = np.empty((0, img_sz, img_sz, 3))
    image.location = np.empty((0, 2))
    #image.patch = np.empty((0, smap.shape[0], smap.shape[1]))

    # Pick out the top 'rankCount' maximally intense regions
    map1 = []
    for i in range(rankCount):
        
        # Copy and Reshape saliency map
        temp_smap = np.copy(smap)
        map1.append(temp_smap)
        ori_in = np.copy(ori)
        
        temp_smap = np.reshape(temp_smap, (1, smap.shape[0], smap.shape[1]))
        

        # Append modified saliency map
        image.patched_sequence = np.vstack((image.patched_sequence, temp_smap))
        

        # Grab Maximally Intense Pixel Coordinates (Object Center)
        indices = np.where(smap == smap.max())
        try:
            R = indices[0][0]  # Row
            C = indices[1][0]  # Column
        except IndexError:
            if (i == 1):
                print("Image has no variation, might just be black")
            R = boundLength
            C = boundLength

        # Get bounding box coordinates for object
        coords = build_bounding_box([R, C], boundLength)

        # Add coordinates to member list on the image object
        image.bb_coords.append(coords)

        # "Zero" the maximally intense region to avoid selecting it again
        R1 = coords[0]
        R2 = coords[1]
        C1 = coords[2]
        C2 = coords[3]
        
        if R1 >=0 and R1<=220-(img_sz-20) and C1 >= 0 and C1 <=236-(img_sz-12):
            patch = ori[R1:R1+img_sz,C1:C1+img_sz,:]
            patch = np.reshape(patch, (1, img_sz, img_sz, 3))
        else:
            patch = np.ones([img_sz,img_sz,3])#np.random.normal(size = [32,32,3])#
            patch = np.reshape(patch, (1, img_sz, img_sz, 3))
        
        location = np.array([R1,C1])
        location = np.reshape(location,[1,2])
        image.patch = np.vstack((image.patch, patch))
        image.location = np.vstack((image.location, location))
            
        
        
        

        # Sum up and find the average intensity of the region
        pixel_intensity_sum = 0

        # Traverse through identified region
        for j in range(R1, R2):
            for k in range(C1, C2):
                x_dim = image.original.shape[0]
                y_dim = image.original.shape[1]
                if ((j < x_dim) and (k < y_dim)):
                    pixel_intensity_sum += image.salience_map[j][k]
                    smap[j][k] = 0  # Zero out pixel
                    #ori_in[j][k] = 0
#        temp_ori_in = np.reshape(ori_in, (1, smap.shape[0], smap.shape[1]))
#        image.patch = np.vstack((image.patch, temp_ori_in))
    return map1                
                    
