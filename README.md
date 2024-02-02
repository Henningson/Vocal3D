![LGDV](images/lgdv_small.png) ![Phoniatric Division](images/Uniklinikum-Erlangen.svg)

![Python3](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)
[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)
[![Standard](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
![GPL-3.0](https://img.shields.io/github/license/Henningson/vocaloid)


# Vocal3D
**Vocal3D** is a library for the real-time reconstruction of human vocal folds using a single shot structured light system.
This is a joint work of the <a href="https://www.lgdv.tf.fau.de/">Chair of Visual Computing</a> of the Friedrich-Alexander University of Erlangen-Nuremberg and the <a href="https://www.hno-klinik.uk-erlangen.de/phoniatrie/">Phoniatric Division</a> of the University Hospital Erlangen. 
This code accompanies the paper <a href="https://henningson.github.io/Vocal3D/assets/Paper.pdf">Real-Time 3D Reconstruction of Human Vocal Folds via High-Speed Laser-Endoscopy</a>.

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

## Usage
An example video and calibration files are given in the assets folder.
Unzip the example folder with `unzip assets/sample_data.zip -d assets/` and run the example using
```
python source/main.py
```

## Things to note
If you are using the supplied viewer, please note that the pipeline will generally be not as performant, as every step of the pipeline will be computed in succession (think of it more like a debug view).
However, you will still be able to generate results in a matter of seconds, provided you do not use a PC that is barely able to run MS-DOS.
We supply three Segmentation algorithms in this repository.
One is especially designed for the silicone videos (that are included in the sample_data.zip file), then we include the one by Koc et al. and finally a Neural Segmentator based on a U-Net architecture.
For first tests, we recommend the U-Net one, as it generally is the most robust (albeit the slowest) one.
A pre-trained model is included in the `assets` folder.

## Implementing your own segmentation algorithm
If you want to integrate your own segmentation algorithm into the viewer, we supply a `BaseSegmentator` class, from which your segmentation class may inherit.
The necessary functions to override are marked by `#TODO: Implement me`.
Please have a look at the supplied segmentation algorithms for some inspiration.

## Limitations
Due to the moisture on top of human vocal folds, the mucuous tissue of in-vivo data often generates specular highlights that influences the performance of segmentation algorithms.
Furthermore, the segmentation algorithm by Koc et al. that we supply in this repository requires well captured data, in which the glottis can be accurately differentiated from the vocal folds.
As of right now, we are working on a system-specific segmentation algorithm that can deal with these harsh cases.

## Citation
Please cite this paper, if this work helps you with your research:
```
@InProceedings{10.1007/978-3-031-16449-1_1,
  author="Henningson, Jann-Ole and Stamminger, Marc and D{\"o}llinger, Michael and Semmler, Marion",
  title="Real-Time 3D Reconstruction of Human Vocal Folds via High-Speed Laser-Endoscopy",
  booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
  year="2022",
  pages="3--12",
  isbn="978-3-031-16449-1"
}
```
A PDF of the Paper is included in the `assets/` Folder of this repository.
However, you can also find it here: <a href="https://link.springer.com/chapter/10.1007/978-3-031-16449-1_1">Springer Link</a>.
