# Band-wise Hyperspectral Image Pansharpening using CNN Model Propagation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.14403)
[![GitHub Stars](https://img.shields.io/github/stars/matciotola/Lambda-PNN?style=social)](https://github.com/matciotola/Lambda-PNN)
![Visitors](https://img.shields.io/endpoint?url=https%3A%2F%2Fhits.dwyl.com%2Fmatciotola%2FLambda-PNN.json\&style=flat\&label=hits\&color=blue)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/fca77356704048f6a47841f73e8c97db)](https://app.codacy.com/gh/matciotola/Lambda-PNN/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![codecov](https://codecov.io/github/matciotola/Lambda-PNN/graph/badge.svg?token=28AINVS2EK)](https://codecov.io/github/matciotola/Lambda-PNN)

[Band-wise Hyperspectral Image Pansharpening using CNN Model Propagation](link 1) ([ArXiv](link 2)) 

Brief Description

## Cite R-PNN

If you use R-PNN in your research, please use the following BibTeX entry.

Bibtex

## Team members

*   Giuseppe Guarino (giuseppe.guarino2@unina.it);

*   Matteo Ciotola (matteo.ciotola@unina.it);

*   Gemine Vivone   ();

*   Giuseppe Scarpa  (giuseppe.scarpa@uniparthenope.it).

## License

Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/giu-guarino/R-PNN/LICENSE.txt)
(included in this package)

## Prerequisites

All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

*   Python 3.10.10
*   PyTorch 2.0.0
*   Cuda 11.7 or 11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

*   Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads)
*   Create a folder in which save the algorithm
*   Download the algorithm and unzip it into the folder or, alternatively, from CLI:

<!---->

    git clone https://github.com/giu-guarino/R-PNN

*   Create the virtual environment with the `r_pnn_env.yaml`

<!---->

    conda env create -n r_pnn_env -f r_pnn_env.yaml

*   Activate the Conda Environment

<!---->

    conda activate r_pnn_env

*   Test it

<!---->

    python test.py -i example/PRISMA_example.mat -o ./Output_folder/ 

## Usage

### Before to start

The easiest way for testing this algorithm is to create a `.mat` file. It must contain:

*   `I_MS_LR`: Original Hyper-Spectral Stack in channel-last configuration (Dimensions: H x W x B);
*   `I_MS`: Interpolated version of Original Hyper-Spectral Stack in channel-last configuration (Dimensions: HR x WR x B);
*   `I_PAN`: Original Panchromatic band, without the third dimension (Dimensions: HR x WR).

where R is the ratio of the sensor.

Please refer to `--help` for more details.

### Testing

The easiest command to use the algorithm on full resolution data:

    python test.py -i path/to/file.mat -s sensor_name

Several options are possible. Please refer to the parser help for more details:

    python test.py -h

### Training

This project provide a set of weights obtained from a previos training. 

If you want to train this network by yourself, you can do it providing in input the path of a folder that contains two subfolders:

*   Dataset
*       Training
*       Validation


## Dataset

You can find the dataset used in this work at this [link](https://openremotesensing.net/knowledgebase/panchromatic-and-hyperspectral-image-fusion-outcome-of-the-2022-whispers-hyperspectral-pansharpening-challenge/)
