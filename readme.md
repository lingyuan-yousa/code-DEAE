# Experiment Documentation

## Environment Setup

1. **Install Python >= 3.9**  
   Select the appropriate version for your OS.

2. **Install Anaconda**  
   Configure PyCharm to use the Anaconda environment as the Python interpreter.

3. **Install PyTorch**  
   Follow OS-specific and CUDA configurations from [pytorch.org](https://pytorch.org/get-started/locally/).

4. **Install Additional Packages**  
   Run in Conda environment:  
   ```bash
   conda install -c conda-forge lightgbm  
   conda install -c conda-forge catboost
Note: The dataset is proprietary. Partial data samples are provided to demonstrate feature formats.

## Execution Guide
Experiments are conducted on two datasets: Daqing and Changqing.
Execute the following files directly in PyCharm:

### Daqing Dataset
| File                      | Model Type           | Description                                                         |
|---------------------------|----------------------|---------------------------------------------------------------------|
| `supervised-test.py`      | Supervised           | Baseline model with full label rate (100%)                          |
| `self-supervised-deae.py` | Self-Supervised      | `deae_self_att` architecture with adjustable label rate             |
| `semi-supervised-deae.py` | Semi-Supervised      | `deae_semi` model supporting dynamic label rate configuration       |
| `PL-main.py`              | Pseudo-Labeling      | Enhanced semi-supervised model with `deae_1DCNN_MLP` architecture   |
Changqing Dataset
Same file structure and naming convention as Daqing.

Comparative Experiments
All executable files for benchmark comparisons are clearly organized in the directory.