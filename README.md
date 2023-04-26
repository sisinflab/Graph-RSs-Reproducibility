# Graph-RSs-Reproducibility

This is the official repository for the paper "_Challenging the Myth of Graph Collaborative Filtering: a Reasoned and
Reproducibility-driven Analysis_", under review at RecSys 2023 (Reproducibility Track).

This repository is heavily dependent on the framework **Elliot**, so we suggest you refer to the official GitHub [page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

We implemented and tested our models in `PyTorch`, using the version `1.12.0`, with CUDA `10.2` and cuDNN `8.0`. Additionally, graph-based models require `PyTorch Geometric`, which is compatible with the versions of CUDA and `PyTorch` we indicated above.

### Installation guidelines: scenario #1
If you have the possibility to install CUDA on your workstation (i.e., `10.2`), you may create the virtual environment with the requirement file we included in the repository, as follows:

```
# PYTORCH ENVIRONMENT (CUDA 10.2, cuDNN 8.0)

$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
```

### Installation guidelines: scenario #2
A more convenient way of running experiments is to instantiate a docker container having CUDA `10.2` already installed.

Make sure you have Docker and NVIDIA Container Toolkit installed on your machine (you may refer to this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)).

Then, you may use the following Docker image to instantiate the container equipped with CUDA `10.2`:

- Container Docker with CUDA `10.2` and cuDNN `8.0` (the environment for `PyTorch`): [link](https://hub.docker.com/layers/nvidia/cuda/10.2-cudnn8-devel-ubuntu18.04/images/sha256-3d1aefa978b106e8cbe50743bba8c4ddadacf13fe3165dd67a35e4d904f3aabe?context=explore)

### Datasets generation
First, create the three folders under `./data/` with the names "yelp2018", "amazon-book", "gowalla". Second, download the train and test files from the following links, and place them in each folder accordingly. 

- Yelp-2018: https://github.com/tanatosuu/svd_gcn/tree/main/datasets/yelp
- Amazon-Book: https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network/tree/main/data/amazon-book
- Gowalla: https://github.com/kuandeng/LightGCN/tree/master/Data/gowalla

Then, create a single "dataset.tsv" file for each dataset, by running the script `create_dataset.py`. You will need to change the dataset name and the train/test names within the script.

After that, you can generate the sub-datasets by running the following script (with arguments):

```
$ python3 graph_sampling.py \
--dataset <dataset_name> \
--filename <filename_of_the_dataset> \
--sampling_strategies <list_of_sampling_strategies> \
--num_samplings <number_of_sampling> \
--random_seed <random_seed_for_reproducibility>
```

This will create the sub-folders ```./data/<dataset>/node-dropout/``` and/or ```./data/<dataset>/edge-dropout/``` with the tsv files for each sub-dataset within the corresponding sampling strategy. 

Moreover, you will find a file named ```sampling-stats.tsv``` for each of the three datasets, which reports basic statistics about the generated sub-datasets. You will need it for the training and evaluation of the models (see later).

### Models training and evaluation
Now you are all set to train and evaluate the graph-based recommender systems. To do so, you should run the following script (with arguments):

```
$ python3 start_experiments.py \
--dataset <dataset_name> \
--gpu <gpu_id>
```

If you are curious about the hyper-parameter settings for the models, you may refer to the ```./config_files/``` folder, where all configuration files are stored.

Depending on your workstation, the training and evaluation could take very long time. Remember that you are training and evaluating four graph-based recommender systems on 1,800 datasets!

After the training and evaluation are done, you will find all performance files in the folder ```./results/<dataset_name>/performance/```. Do not worry about them, because there is a script that will collect all results and join them to the dataset characteristics (see later).

### Characteristics calculation and regression (RQ1)

To calculate the dataset characteristics, you should run the following script (with arguments):

```
$ python3 generate_characteristics.py \
--dataset <dataset_name> \
--start <start_dataset_id> \
--end <end_dataset_id> \
--characteristics <list_of_characteristics> \
--metric <list_of_metrics> \
--splitting <list_of_sampling_strategies> \
--proc <number_of_cores_for_multiprocessing> \
-mp <if_set_it_will_run_in_multiprocessing_for_speed_up>
```

This will produce the tsv file "characteristics_0_600.tsv" under each dataset folder.

After that, you may want to run the linear regression model on the generated datasets of characteristics/performance metrics. To do so, you should run the following script (with arguments):

```
$ python3 regression.py \
--dataset <dataset_name> \
--start_id <start_dataset_id> \
--end_id <end_dataset_id> \
--characteristics <list_of_characteristics>
```
This will produce the tsv files "regression_<metric>_0_600.tsv", one for each metric, under the dataset folder.

We also provide a script to generate the latex tables (only the results parts of the tables, without row and column headers) starting from the results. To do so, you should run the script:

```
$ python3 generate_table_rq1.py
```

This will produce a tsv file "table_rq1.tsv" in the folder ```./data/```, as it is unique for all datasets.

### Node and Edge Dropout Analysis (RQ2)

You should run the script 

```
$ python3 check_scale_free.py
```

to fit the power-law and exponential functions on the node degree distribution of gowalla. The script will display the plot, and generate the latex code for the plot (used for the paper).

Finally, to reproduce the results for the RQ2 table, you should run the script:

```
$ python3 generate_table_rq2.py
```

that, similarly to above, will produce one tsv file "table_<alpha>_rq2.tsv" for each alpha value in the folder ```./data/```, as it is unique for all datasets. Again, the latex code contains only result cells, but no row and column headers.
