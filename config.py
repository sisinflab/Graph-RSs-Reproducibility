import os

DATA_FOLDER = os.path.abspath('./data')
OUTPUT_FOLDER = os.path.abspath('./data/')
ACCEPTED_DATASETS = ['gowalla', 'amazon-book', 'yelp-2018']
ACCEPTED_CHARACTERISTICS = ['space_size',
                            'shape',
                            'density',
                            'gini_user',
                            'gini_item',
                            'average_degree_users',
                            'average_degree_items',
                            'average_clustering_coefficient_dot_users',
                            'average_clustering_coefficient_dot_items',
                            'degree_assortativity_users',
                            'degree_assortativity_items']
