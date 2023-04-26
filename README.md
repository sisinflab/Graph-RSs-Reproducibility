# Graph-RSs-Reproducibility

This is the official repository for the paper "_Challenging the Myth of Graph Collaborative Filtering: a Reasoned and
Reproducibility-driven Analysis_", under review at RecSys 2023 (Reproducibility Track).

This repository is heavily dependent on the framework **Elliot**, so we suggest you refer to the official GitHub [page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

## Pre-requisites

We implemented and tested our models in `PyTorch==1.12.0`, with CUDA `10.2` and cuDNN `8.0`. Additionally, some of graph-based models require `PyTorch Geometric`, which is compatible with the versions of CUDA and `PyTorch` we indicated above.

### Installation guidelines: scenario #1
If you have the possibility to install CUDA on your workstation (i.e., `10.2`), you may create the virtual environment with the requirement files we included in the repository, as follows:

```
# PYTORCH ENVIRONMENT (CUDA 10.2, cuDNN 8.0)

$ python3.8 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install -r requirements_torch_geometric.txt
```

### Installation guidelines: scenario #2
A more convenient way of running experiments is to instantiate a docker container having CUDA `10.2` already installed. Make sure you have Docker and NVIDIA Container Toolkit installed on your machine (you may refer to this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)). Then, you may use the following Docker image to instantiate the container equipped with CUDA `10.2` and cuDNN `8.0`: [link](https://hub.docker.com/layers/nvidia/cuda/10.2-cudnn8-devel-ubuntu18.04/images/sha256-3d1aefa978b106e8cbe50743bba8c4ddadacf13fe3165dd67a35e4d904f3aabe?context=explore).

## Datasets

### Reproducibility datasets
We used Gowalla, Yelp 2018, and Amazon Book datasets. The original links may be found here, where the train/test splitting has already been provided:

- Gowalla: https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/gowalla
- Yelp 2018: https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018
- Amazon Book: https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/amazon-book

After downloading, create three folders ```./data/{dataset_name}```, one for each dataset. Then, run the script ```./map_dataset.py```, by changing the name of the dataset within the script itself. It will generate the train/test files for each dataset in a format compatible for Elliot (i.e., tsv file with three columns referring to user/item).

In case, we also provide the final tsv files for all the datasets at the following links:

- Gowalla: https://drive.google.com/drive/folders/195OrcPsw7gcr_4gkyW-Ms1g73qGnj2Sw?usp=share_link
- Yelp 2018: https://drive.google.com/drive/folders/1iH7iQImtTrOsrXgU0-v1TmgadbTpPHT9?usp=share_link
- Amazon Book: https://drive.google.com/drive/folders/1bylpVH8t_Q_K0du8952oMa1MhUT68DUr?usp=share_link

### Additional datasets
We directly provide the train/validation/test splittings for Allrecipes and BookCrossing. As already stated for Gowalla, Yelp 2018, and Amazon Book, create one folder for each dataset in ```./data/{dataset_name}```:

- Allrecipes: https://drive.google.com/drive/folders/1Mz_dp9S0sToVIkUISYQAhI_Vvxqwp_WK?usp=share_link
- BookCrossing: https://drive.google.com/drive/folders/1L4x_88uyISpUf5hGxkmw6fdLjBzXjCAR?usp=share_link

## Results

### Replication of prior results (RQ1)
To reproduce the results reported in Table 3, run the following:

```
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 python3.8 -u start_experiments.py \
$ --dataset {dataset_name} \
$ --model {model_name} \
$ --gpu {gpu_id}
```
Note that ```CUBLAS_WORKSPACE_CONFIG=:4096:8``` (which may change depending on your configuration) is needed to ensure the complete reproducibility of the experiments (otherwise, PyTorch may run some operations in their non-deterministic version).

The following table provides links to the specific configuration of hyper-parameters we adopted for each graph-based model (derived from the original papers and/or the codes):

|          | **Gowalla**                                                                                                | **Yelp 2018**                                                                                                | **Amazon Book**                                                                                                |
|----------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| NGCF     |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ngcf_gowalla.yml)   |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ngcf_yelp-2018.yml)   |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ngcf_amazon-book.yml)   |
| DGCF     |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/dgcf_gowalla.yml)   |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/dgcf_yelp-2018.yml)   |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/dgcf_amazon-book.yml)   |
| LightGCN | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/lightgcn_gowalla.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/lightgcn_yelp-2018.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/lightgcn_amazon-book.yml) |
| SGL      |                                                     ---                                                    |    [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/sgl_yelp-2018.yml)   |    [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/sgl_amazon-book.yml)   |
| UltraGCN | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ultragcn_gowalla.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ultragcn_yelp-2018.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ultragcn_amazon-book.yml) |
| GFCF     |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/gfcf_gowalla.yml)   |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/gfcf_yelp-2018.yml)   |   [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/gfcf_amazon-book.yml)   |

### Benchmarking graph CF approaches using alternative baselines (RQ2)
In addition to the graph-based models from above, we train and test four classic (and strong) CF baselines. We also provide pointers to their configuration files with the exploration of hyper-parameters, which can be used to reproduce Table 4.

