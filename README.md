# SANCS
Code search model based the self-attention

## Dependency
Successfully tested in Ubuntu 18.04
- Python == 3.7
- PyTorch == 1.6.0
- tqdm == 4.48.2
- numpy == 1.16.3
- tables == 3.6.1
- argparse

## Code Structure
- `attention`: Self-attention network and code-description network.
- `method`: Code/desc representation and similarity measure mudule.
- `train.py`: Train and validate code/desc representation models.
- `dataset.py`: Dataset loader.
- `configs`: Basic configuration for the attention and method folder. Each function defines the hyper-parameters for the corresponding model.
- `utils.py`: Utilities for models and training.

## Usage

  ### Data
  In our experiments, we use the dataset shared by @guxd. You can download this shared dataset from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) and add this dataset folder to `/data`.
  
  ### Configuration
  Edit hyper-parameters and settings in `config.py`
  
  ### Train
  ```bash
  python train --mode train
  ```
  
  ### Eval
  ```bash
  python train --mode eval
  ```
## References
Here are some things I looked at while writing this model.

- https://github.com/guxd/deep-code-search
- https://github.com/jadore801120/attention-is-all-you-need-pytorch

