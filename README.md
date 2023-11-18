# Graph-RSs-Reproducibility

This is the official repository for the paper "_Challenging the Myth of Graph Collaborative Filtering: a Reasoned and
Reproducibility-driven Analysis_", accepted at RecSys 2023 (Reproducibility Track).

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

In case, we also provide the final tsv files for all the datasets in this repo.

### Additional datasets
We directly provide the train/validation/test splittings for Allrecipes and BookCrossing in this repo. As already stated for Gowalla, Yelp 2018, and Amazon Book, create one folder for each dataset in ```./data/{dataset_name}```.

## Results

### Replication of prior results (RQ1)
To reproduce the results reported in Table 3, run the following:

```
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 python3.8 -u start_experiments.py \
$ --dataset {dataset_name} \
$ --model {model_name} 
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
In addition to the graph-based models from above, we train and test four classic (and strong) CF baselines. We also provide pointers to their configuration files with the exploration of hyper-parameters, which can be used to reproduce Table 4. We recall that EASER configuration file is not provided at submission time for Amazon Book due to heavy computational costs.

|         | **Gowalla**                                                                                               | **Yelp 2018**                                                                                               | **Amazon Book**                                                                                               |
|---------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| MostPop | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/mostpop_gowalla.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/mostpop_yelp-2018.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/mostpop_amazon-book.yml) |
| Random  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/random_gowalla.yml)  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/random_yelp-2018.yml)  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/random_amazon-book.yml)  |
| UserkNN | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/userknn_gowalla.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/userknn_yelp-2018.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/userknn_amazon-book.yml) |
| ItemkNN | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/itemknn_gowalla.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/itemknn_yelp-2018.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/itemknn_amazon-book.yml) |
| RP3Beta | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/rp3beta_gowalla.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/rp3beta_yelp-2018.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/rp3beta_amazon-book.yml) |
| EASER   | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/easer_gowalla.yml)   | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/easer_yelp-2018.yml)   | ---                                                                                                           |

The best hyper-parameters for each classic CF model (as found in our experiments) are reported in the following:

- Gowalla
  - UserkNN: ```neighbors: 146.0, similarity: 'cosine'```
  - ItemkNN: ```neighbors: 508.0, similarity: 'dot'```
  - Rp3Beta: ```neighborhood: 777.0, alpha: 0.5663562161452378, beta: 0.001085447926739258, normalize_similarity: True```
  - EASER: ```l2_norm: 15.930101258108873```

- Yelp 2018
  - UserkNN: ```neighbors: 146.0, similarity: 'cosine'```
  - ItemkNN: ```neighbors: 144.0, similarity: 'cosine'```
  - Rp3Beta: ```neighborhood: 342.0, alpha: 0.7681732734954694, beta: 0.4181395996963926, normalize_similarity: True```
  - EASER: ```l2_norm: 212.98774633994572```

- Amazon Book
  - UserkNN: ```neighbors: 146.0, similarity: 'cosine'```
  - ItemkNN: ```neighbors: 125.0, similarity: 'cosine'```
  - Rp3Beta: ```neighborhood: 496.0, alpha: 0.44477903655656115, beta: 0.5968193614337285, normalize_similarity: True```
  - EASER: N.A.

### Extending the experimental comparison to new datasets (RQ3 — RQ4)
We report the configuration files (with hyper-parameter search spaces) for each model/dataset pair for classic + graph CF baselines and Allrecipes and BookCrossing.

|          | **Allrecipes**                                                                                                | **BookCrossing**                                                                                                |
|----------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| MostPop  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/mostpop_allrecipes.yml)  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/mostpop_bookcrossing.yml)  |
| Random   | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/random_allrecipes.yml)   | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/random_bookcrossing.yml)   |
| UserkNN  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/userknn_allrecipes.yml)  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/userknn_bookcrossing.yml)  |
| ItemkNN  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/itemknn_allrecipes.yml)  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/itemknn_bookcrossing.yml)  |
| RP3Beta  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/rp3beta_allrecipes.yml)  | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/rp3beta_bookcrossing.yml)  |
| EASER    | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/easer_allrecipes.yml)    | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/easer_bookcrossing.yml)    |
| NGCF     | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ngcf_allrecipes.yml)     | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ngcf_bookcrossing.yml)     |
| DGCF     | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/dgcf_allrecipes.yml)     | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/dgcf_bookcrossing.yml)     |
| LightGCN | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/lightgcn_allrecipes.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/lightgcn_bookcrossing.yml) |
| SGL      | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/sgl_allrecipes.yml)      | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/sgl_bookcrossing.yml)      |
| UltraGCN | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ultragcn_allrecipes.yml) | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/ultragcn_bookcrossing.yml) |
| GFCF     | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/gfcf_allrecipes.yml)     | [link](https://github.com/sisinflab/Graph-RSs-Reproducibility/blob/main/config_files/gfcf_bookcrossing.yml)     |

The best hyper-parameters for each classic + graph CF model (as found in our experiments) are reported in the following:

- Allrecipes
  - UserkNN: ```neighbors: 863.0, similarity: 'cosine'```
  - ItemkNN: ```neighbors: 508.0, similarity: 'dot'```
  - RP3Beta: ```neighborhood: 777.0, alpha: 0.5663562161452378, beta: 0.001085447926739258, normalize_similarity: True```
  - EASER: ```l2_norm: 555344.9240485814```
  - NGCF: ```lr: 0.0010492631473907471, epochs: 100, factors: 64, batch_size: 128, l_w: 0.08623551848300251, n_layers: 1, weight_size: 64, node_dropout: 0.5704755544541924, message_dropout: 0.37665593943318876, normalize: True```
  - DGCF: ```lr: 0.000313132757493385, epochs: 10, factors: 64, batch_size: 256, l_w_bpr: 3.3519512293075625e-05, l_w_ind: 0.00021537560246909769, n_layers: 2, routing_iterations: 2, intents: 4```
  - LightGCN: ```lr: 0.001, epochs: 10, factors: 64, batch_size: 256, l_w: 0.001288395174690605, n_layers: 4, normalize: True```
  - SGL: ```lr: 0.001, epochs: 10, batch_size: 128, factors: 64, l_w: 1e-4, n_layers: 3, ssl_temp: 0.6492261261178492, ssl_reg: 0.012429441724966553, ssl_ratio: 0.2618285305261178492, sampling: nd```
  - UltraGCN: ```lr: 1e-4, epochs: 240, factors: 64, batch_size: 128, g: 1e-4, l: 0.6421380210212072, w1: 0.026431283275666788, w2: 0.0006086626045670742, w3: 2.3712235041563928e-07, w4: 0.03156224646525972, ii_n_n: 10, n_n: 300, n_w: 300, s_s_p: False, i_w: 1e-4```
  - GFCF: ```svd_factors: 256, alpha: 0.5477395514607551```

- BookCrossing
  - UserkNN: ```neighbors: 360.0, similarity: 'cosine'```
  - ItemkNN: ```neighbors: 125.0, similarity: 'cosine'```
  - RP3Beta: ```neighborhood: 777.0, alpha: 0.5663562161452378, beta: 0.001085447926739258, normalize_similarity: True```
  - EASER: ```l2_norm: 97.97026620421359```
  - NGCF: ```lr: 0.001313040990458504, epochs: 150, factors: 64, batch_size: 128, l_w: 0.007471352712353916, n_layers: 1, weight_size: 64, node_dropout: 0.6222126221705062, message_dropout: 0.2768938386628866, normalize: True```
  - DGCF: ```lr: 0.00033659666428326467, epochs: 112, factors: 64, batch_size: 1024, l_w_bpr: 0.0005015002430942853, l_w_ind: 1.0625908485203885e-05, n_layers: 1, routing_iterations: 2, intents: 4```
  - LightGCN: ```lr: 0.001, epochs: 160, factors: 64, batch_size: 256, l_w: 0.00128839517469060, n_layers: 4, normalize: True```
  - SGL: ```lr: 0.001, epochs: 10, batch_size: 128, factors: 64, l_w: 1e-4, n_layers: 4, ssl_temp: 0.3831504020789032, ssl_reg: 0.14847461762325737, ssl_ratio: 0.18119634034037221, sampling: rw```
  - UltraGCN: ```lr: 1e-4, epochs: 205, factors: 64, batch_size: 128, g: 1e-4, l: 2.1590977284940767, w1: 0.4071845141372458, w2: 2.674735729193082e-06, w3: 0.11655266791027195, w4: 0.05001575677944944, ii_n_n: 10, n_n: 300, n_w: 300, s_s_p: False, i_w: 1e-4```
  - GFCF: ```svd_factors: 64, alpha: 0.4240013631942601```

For RQ4, you need to generate the tsv files where each user from the training set is assigned one of the four quartiles. To do so, run the script ```./quartiles_characteristics.py```, by changing the name of the dataset inside the script accordingly. This will create (for each dataset) 3 tsv files, one for each hop (i.e., 1-hop, 2-hop, 3-hop). In case, we directly provide such files for your convenience in the same folders of Allrecipes and BookCrossing (see above).

Once all models have been trained, and tsv files for the user groups have been downloaded and correctly placed, you may want to generate the recommendation lists ONLY for the best hyper-parameter configuration for each model/dataset pair. This is done by setting the parameter ```save_recs: True``` in each configuration file. 

Now, we are all set to calculate the nDCG on each user group. To do so, run the following script:

```
$ python3.8 -u start_experiments_user_groups.py \
$ --dataset {dataset_name} \
$ --hop {hop_number}
```
