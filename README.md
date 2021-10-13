# IC-U-Net

<iframe width="560" height="315" src="https://www.youtube.com/embed/0tHadL3kRjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

[IC-U-Net](https://github.com/roseDwayane/AIEEG) is a project on EEG artifacts removal end-to-end process. 

We evaluate the model with lane-keeping drive data collected and walking experiment from [scientific data](https://www.nature.com/articles/s41597-019-0027-4) and [mygooglecloud](https://drive.google.com/drive/folders/1B8smvaYGgC-y_TSshIG23JbMmawoaA5E?usp=sharing).

Sample results are available
[working]().

> Looking for a PyTorch version? Check out [this repository](https://github.com/roseDwayane/AIEEG/blob/main/UNet1D-real/cumbersome_model.py).

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

## Outputs

(writing)

## Sample Results

(writing)

## Papers

(writing)
