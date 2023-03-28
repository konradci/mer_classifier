# MER classifier

This repository contains python software that can be used for:
- training deep learning model that can be used for the classification of microelectrode recordings
- classification of validation and test chunks 
to obtain ROC curves for the classification of the chunks.
- classification of validation and test recordings from the MER dataset  
to obtain ROC curves for recordings classification.

## Creating the python environment

- install [Anaconda](https://www.anaconda.com/products/distribution)
- open a new Anaconda terminal
- create a new Python environment named **mer**
 

    conda create --name mer python=3.10
- activate the **mer** environment
  
  
    conda activate mer
- install PyTorch 1.13.1


    conda install -y pytorch==1.11.0 torchvision==0.13.1 torchaudio==0.11.0 pytorch-cuda=11.7 -c pytorch -c nvidia
- install PyTorch Ignite

 
    conda install -y -c pytorch ignite
- install scikit-learn


    conda install -y -c conda-forge scikit-learn
- install tqdm


    conda install -y -c conda-forge tqdm 
- install pandas


    conda install -y -c anaconda pandas
- install install tensorboard


    conda install -y -c conda-forge tensorboard 
- install install psutil


    conda install -y -c conda-forge psutil 
- istall matplotlib


    conda install -y -c conda-forge matplotlib
- install pyarrow


    conda install -y -c conda-forge pyarrow

The **mer** environment will take approximately 8 GB of disk space.

## Requirements

Due to the use of half-precision, the script requires Nvidia GPU. 
Because of PyTorch limitations, they will not run on CPU only.
The model was originally trained on V100 GPU having 32GB of RAM.
If your GPU has less than 32GB RAM, the model might not fit into the GPU RAM during training. 
In such cases, decrease batch size from default 128 to a smaller number, e.g., 96 or 64.

## Training the model
### Pull the software from the [GitHub](https://github.com/)

- select directory with writing permissions *\<dir\>*
- enter directory *\<dir\>*
- execute


    git clone git@github.com:konradci/mer_classifier.git
- let *\<dir\>/mer_classifier* be denoted as *\<dir_s\>*

### Download the chunk dataset cache

The chunk dataset consists of spectrograms of chunks taken from recordings.

- select directory with writing permissions *\<dir_d\>*
- download *cache.7z*
- extract *cache.7z* into *\<dir_d\>* using provided password
- *\<dir_d\>/cache* is created with contents

### Update the configuration

- open *\<dir_s\>/src/config/config_coi.ini* with chosen editor
- set the following entries:


    path_cache = <dir_d>/cache
    path_raw = <dir_d>/raw_data
    path_checkpoint = <dir_d>/checkpoint
    path_roc=<dir_d>/roc
    path_tb=<dir_d>/tb
    path_img=<dir_d>/img
    path_log=<dir_d>/log

### running the training

- open the new Anaconda console
- execute


    conda activate mer
    export PYTHONPATH=$PYTHONPATH:<dir_s>/src
    cd <dir_s>/src/work/01_classify
    python ./A_train.py

## Generating ROC curves
For generating ROC curves, you can use checkpoints generated during training, or for reproducibility of results, use ones provided in *checkpoint.7z* archive.
Mind that for each checkpoint file, four ROC charts will be generated:
- script B_roc_curves_chunks.py creates
  - ROC file for validation chunks
  - ROC file for test chunks
- script C_roc_curves_recs.py creates 
  - ROC file for validation recordings
  - ROC file for test recordings

The resulting figures of ROC curves will be stored in eps format in *\<dir_d\>/roc*. 

## Generating ROC curves for chunks
Creating ROC curves for chunks uses the same data source as the training
i.e. *\<dir_d\>/cache*
- if you wish to use provided checkpoints
  - delete any files from *\<dir_d\>/checkpoint*
  - download *checkpoint.7z*
  - extract *checkpoint.7z* into *\<dir_d\>* using provided password
  - *\<dir_d\>/checkpoint* is created with contents
- open the new Anaconda console
- execute


    conda activate torch
    export PYTHONPATH=$PYTHONPATH:<dir_s>/src
    cd <dir_s>/src/work/01_classify
    python ./B_roc_curves_chunks.py

## Generating ROC curves for recordings
Creating ROC curves for chunks uses the raw dataset provided in *raw_data.7z* dataset
- download *raw_data.7z*
- extract *raw_data.7z* into *\<dir_d\>* using provided password
- *\<dir_d\>/raw_data* is created with contents 
- if you wish to use provided checkpoints
  - delete any files from *\<dir_d\>/checkpoint*
  - download *checkpoint.7z*
  - extract *checkpoint.7z* into *\<dir_d\>* using provided password
  - *\<dir_d\>/checkpoint* is created with contents
- open the new Anaconda console
- execute


    conda activate torch
    export PYTHONPATH=$PYTHONPATH:<dir_s>/src
    cd <dir_s>/src/work/01_classify
    python ./C_roc_curves_recs.py

As the *C_roc_curves_recs.py* script takes for classification, raw
recordings from the MER Dataset, it is an example of how to classify
the recordings using the provided model.

Konrad A. Ciecierski