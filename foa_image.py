# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

focus_of_attention.py - Initial front end system gamma kernel implemenation in
Python. Contains the current functional breakdown of the front end system (FES)
gamma kernel. Current implementation is intended to mimic the initial MATLAB
implementation and results created by Ryan Burt.

"""

# Standard Library Imports

# 3P Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

# Local Imports

plt.rcParams.update({'font.size': 22})


class ImageObject():

    ''' Image Object encapsulates the meta data related to each image
    being processed by the front end system '''

    path = ''   # Folder Path
    name = ''   # Image Name
    ext = '.'   # Image Extension
    rgb = True  # RGB Boolean
    fc = 0      # Frame Count

    # Image Maps
    original = np.array([])  # Original Image
    modified = np.array([])  # Modified Image - Processed by Gamma Kernel
    patched = np.array([])  # Original Image with image patch bounding boxes
    patched_sequence = np.array([])
    ground_truth = np.array([])  # Ground Truth Map
    salience_map = np.array([])  # Saliency Map

    # Objects
    patch_list = []  # List of objects in the frame

    # Bounding Box Metadata
    gt_coords = []  # Coordinates derived from ground truth data
    bb_coords = []  # Bounding box coordinates by FoA - Ranked by intensity

    def __init__(self, img, rgb=rgb, fc=0):

        ''' Initialization '''

        if (isinstance(img, str)):
            # Extract image name, path, and extension
            self.file_parse(img)
            self.original = cv2.imread(img, cv2.IMREAD_COLOR)
        else:
            self.original = img

        # Convert image to CIELAB Color Space
        self.image_convert()

        self.bb_coords = []
        self.gt_coords = []
        self.patched_sequence = np.array([])
        self.rgb = rgb
        self.fc = fc

    def file_parse(self, file):

        ''' Discretize file path, name, and extension '''

        # Current Directory
        if (file[0] == '.'):
            file = file[1:]

        # Extension
        s1 = file.split('.')
        self.ext += s1[-1]

        # Name
        s2 = s1[0].split('/')
        self.name = s2[-1]

        # Path
        s2.remove(s2[-1])
        for f in s2:
            self.path += (f + '/')
        if (self.path[0] == '/'):
            self.path = '.' + self.path

    def image_convert(self):

        ''' Convert RGB image to CIELAB Color Space '''

        if (self.rgb):  # Image is RGB
            # Read original image into object
            self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)

            # Convert RGB to CIELAB
            self.modified = color.rgb2lab(self.original)

    def display_map(self, img, title):

        ''' Plot input image '''

        plt.figure()
        plt.title(title)
        plt.imshow(img)
        plt.show()

    def plot_original_map(self):
        self.display_map(self.original, "Original Image")

    def plot_modified_map(self):
        self.display_map(self.modified, "Original Image")

    def plot_saliency_map(self):
        self.display_map(self.salience_map, "Saliency Map")

    def draw_image_patches(self, bbt=1, salmap=False):

        ''' Apply bounding boxes to original image in the patched map.
        bbt - Bounding box line thickness
        salmap - Apply bounding boxes to saliency map '''

        # Copy the original image
        self.patched = np.copy(self.original)

        # Draw bounding boxes for ground truth
        for o in self.gt_coords:
            lc = [100, 150, 255]  # Line Color - Blue
            self._draw(o, lc, bbt, salmap)

        # Draw bounding boxes on max intensity regions
        for o in self.bb_coords:
            lc = [255, 150, 100]  # Line Color - Red
            self.__draw(o, lc, bbt, salmap)

    def __draw(self, obj, lc, bbt=int, salmap=bool):

        ''' Helper method for drawing bounding boxes '''

        # Get bounding coordinates
        a = obj[0]  # R1
        b = obj[1]  # R2
        c = obj[2]  # C1
        d = obj[3]  # C2

        # Draw bounding boxes on patched map
        if (self.rgb):  # RGB
            self.patched[a:b, c-bbt:c] = lc  # Left
            self.patched[a:b, d:d+bbt] = lc  # Right
            self.patched[a-bbt:a, c:d] = lc  # Top
            self.patched[b:b+bbt, c:d] = lc  # Bottom

            # Salience Map
            if (salmap):
                self.salience_map[a:b, c-bbt:c] = 1.0  # Left
                self.salience_map[a:b, d:d+bbt] = 1.0  # Right
                self.salience_map[a-bbt:a, c:d] = 1.0  # Top
                self.salience_map[b:b+bbt, c:d] = 1.0  # Bottom
