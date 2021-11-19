# IC-U-Net

<iframe width="560" height="315" src="https://www.youtube.com/embed/0tHadL3kRjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Introduction
This is the Computational NeuroErgonomics x NeuroEducation ([CNElab](https://sites.google.com/view/chchuang/CNElab?authuser=0)) on EEG artifacts removal end-to-end process: [IC-U-Net](https://github.com/roseDwayane/AIEEG): A U-Net based Denoising Autoencoders using Mixtures of Independent Components for Automatic EEG Artifact Removal, written in Pytorch. The aim of this project is to
* A novel EEG artifact removal method, IC-U-Net, is proposed.
* IC-U-Net is built based on the U-Net architecture with a loss function ensemble.
* IC-U-Net is trained using mixtures of EEG sources decomposed by independent component analysis.
* IC-U-Net does not require parameter tuning and can simultaneously remove various types of artifacts.
* IC-U-Net can facilitate the extraction of relatively more brain sources from noisy EEG signals.

## Requirements
* Python == 3.6
* Pytorch == 1.6.0
* numpy >= 1.19.2

## Evaluated Dataset
We evaluate the model with lane-keeping drive data collected and walking experiment from [scientific data](https://www.nature.com/articles/s41597-019-0027-4) and [mygooglecloud](https://drive.google.com/drive/folders/1B8smvaYGgC-y_TSshIG23JbMmawoaA5E?usp=sharing).

> __Ethics approval__ 
All participants completed informed consent forms after receiving a complete explanation of the study. The Institutional Review Board of Taipei Veterans General Hospital approved the study protocol.

> __Consent to participate__
All of the participants provided written informed consent prior to participation. The consent regarding publishing their data as a scientific report was also included.



## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

- Using pipenv (recommended)

  > Make sure `pipenv` is installed. (If not, simply run `pip install pipenv`.)

  ```sh
  # Install the dependencies
  pipenv install
  # Activate the virtual environment
  pipenv shell
  ```

- Using pip

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```

### (Sim) Prepare training

> The training data can be download from
> `./UNet1D_sim/datalog/train/1-51log.csv`
Each data was randomly generated from the frequency(1-51Hz), amplitude(0-1), phase(0-2pi)
```sh
# Generate the raw data in 
# (function) dataRestore(name)
# you can run the code
python train.py
```

### (Real) Prepare training

> Real EEG training data could not be open right now. But you can prepare for yourself with 30 channel and 1024 points (4sec, 256 sample rate)

```sh
# you can run the code
python train.py
```

## Scripts

### Use pretrained models

1. Download pretrained models

   ```sh
   # Download the pretrained models
   ./UNet1D_real/final_RealEEG_5/modelsave/BEST_checkpoint.pth.tar
   #or
   ./UNet1D_real/final_RealEEG_5/modelsave/checkpoint.pth.tar
   ```

2. You can either perform inference from a trained model:

   ```sh
   # Run inference from a pretrained model
   (writing)
   ```

   or perform interpolation from a trained model:

   ```sh
   # Run interpolation from a pretrained model
   (writing)
   ```

## Sample Results
(writing)

## Papers
Under review on [NeuroImage](https://www.journals.elsevier.com/neuroimage)

## Acknowledgement
This work was supported by the Ministry of Science and Technology, Taiwan (project numbers: MOST 110-2636-E-007-018 and 109-2636-E-007-022), and by the Research Center for Education and Mind Sciences, National Tsing Hua University. No funding source had involved in any of the research procedures.

