<a href="https://www.lgdv.tf.fau.de/"><img align="right" src="images/lgdv_small.png"></a>

![Python3](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)
[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)
[![Standard](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
![GPL-3.0](https://img.shields.io/github/license/Henningson/vocaloid)


# Vocal3D
**Vocal3D** is a library for the real-time reconstruction of human vocal folds using a single shot structured light system.  
This code accompanies the paper **Real-Time 3D Reconstruction of Human Vocal Folds via High-Speed Laser-Endoscopy**.

## Dataset
The HLE Dataset described in the Paper is hosted ![here](https://github.com/Henningson/HLEDataset.git) on GitHub!
We will add it to CERNs Zenodo Platform at a later stage.

## Prerequisites
A CUDA capable GPU is recommended, but not necessary.
However, getting PyTorch3D to work inside the Nurbs-Diff Module without CUDA may require some tinkering.
You have been warned.

## Installation
* Clone this repository
* Download the ![HLE Dataset](https://github.com/Henningson/HLEDataset.git).
* Install PyIGL and Nurbs-Diff as explained in their respective repositories
* Compile the ARAP Code as explained in the PybindARAP submodule
* If you didn't receive any errors run example.py in the source subdirectory.

## Usage
*TODO*

## Examples


## Citation
If you find this work useful please use the following citation:
