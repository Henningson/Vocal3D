![LGDV](images/lgdv_small.png) ![Phoniatric Division](images/Uniklinikum-Erlangen.svg)

![Python3](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)
[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)
[![Standard](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
![GPL-3.0](https://img.shields.io/github/license/Henningson/vocaloid)


# Vocal3D
**Vocal3D** is a library for the real-time reconstruction of human vocal folds using a single shot structured light system.
This is a joint work of the <a href="https://www.lgdv.tf.fau.de/">Chair of Visual Computing</a> of the Friedrich-Alexander University of Erlangen-Nuremberg and the <a href="https://www.hno-klinik.uk-erlangen.de/phoniatrie/">Phoniatric Division</a> of the University Hospital Erlangen. 
This code accompanies the paper **Real-Time 3D Reconstruction of Human Vocal Folds via High-Speed Laser-Endoscopy**.

![Example](images/reco_example.gif)

## Dataset
The HLE Dataset described in the Paper is hosted <a href="https://github.com/Henningson/HLEDataset.git">here on GitHub</a>!  
We will add it to CERNs Zenodo Platform at a later stage.

## Prerequisites
Make sure that you have a Python version >=3.5 installed.
A CUDA capable GPU is recommended, but not necessary.
However, getting PyTorch3D to work inside the Nurbs-Diff Module without CUDA may require some tinkering.

## Installation
First, make sure that conda is installed and clone this repository, including its submodules:
```
git clone https://github.com/Henningson/Vocal3D.git
cd Vocal3D
git submodule update --init --recursive
```

Generate a new conda environment and activate it:
```
conda create --name Vocal3D python=3.8
conda activate Vocal3D
```

Then, install the necessary packages with
```
pip install opencv-python-headless matplotlib scikit-learn tqdm geomdl PyQt5 pyqtgraph ninja
pip install -U fvcore
conda install -c bottler nvidiacub
conda install -c conda-forge igl
```

Install pytorch and pytorch3D
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Download and install <a href="https://github.com/anjanadev96/NURBS_Diff.git">NURBS_Diff</a>
```
git clone https://github.com/anjanadev96/NURBS_Diff.git
cd NURBS_Diff
python setup.py install
```

Download and install our fork of Victor Cornillères <a href="https://github.com/sunreef/PyIGL_viewer">PyIGL Viewer</a>.  
It adds some shadercode that we use for a more domain specific visualization.
```
pip install git+git://github.com/Henningson/PyIGL_viewer.git
```
And finally install our lightweight <a href="https://github.com/Henningson/PybindARAP">C++ ARAP implementation</a>.
```
cd PybindARAP
python setup.py install
```

## Citation
If you use this work in your research, please cite the following paper:

    @article{TBD,
      author = {TBD},
      journal = {TBD},
      title = {TBD},
      year = {TBD}
    }
