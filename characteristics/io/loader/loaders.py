from characteristics.io.paths import *
import pandas as pd
from scipy.sparse import csr_matrix


class TsvLoader:
    # TODO: path should be in load function
    def __init__(self, path, return_type=None, directory='', main_directory='', header=None, names=None):

        # check that the path exists
        assert isinstance(path, str), f'{self.__class__.__name__}: path must be a string. \n' \
                                      f'value: {path}\n' \
                                      f'type: {type(path)}'
        self.path = self.find_the_right_path(path, relative_directory=directory, main_directory=main_directory)

        if return_type is None:
            return_type = pd.DataFrame
        self._return_functions = {pd.DataFrame: self._load_dataframe,
                                  csr_matrix: self._load_crs}
        self.accepted_types = self._return_functions.keys()
        assert return_type in self.accepted_types, f'{self.__class__.__name__}: return type not managed by the loader.'

        self._return_type = return_type

        self.header = header
        self.names = names

    def load(self):
        print(f'{self.__class__.__name__}: loading dataset from \'{self.path}\'')
        data = pd.read_csv(self.path, sep='\t', header=self.header, names=self.names)
        return_function = self._return_functions[self._return_type]
        print(f'{self.__class__.__name__}: dataset loaded as {self._return_type}')
        return return_function(data)

    @staticmethod
    def _load_dataframe(data):
        return data

    @staticmethod
    def _load_crs(data):
        return csr_matrix(data.pivot(index=0, columns=1, values=2).fillna(0))

    def find_the_right_path(self, path, relative_directory=None, main_directory=None):

        if relative_directory is None:
            relative_directory = ''
        if main_directory is None:
            main_directory = ''

        for p in [path, relative_directory, main_directory]:
            assert isinstance(p, str), f'must be a string. Found {p} with type {type(p)}'

        if main_directory:
            path_from_main = os.path.join(main_directory, relative_directory, path)
            assert os.path.exists(path_from_main), f'{self.__class__.__name__}: ' \
                                                   f'path \'{path_from_main}\' does not exists.'
            return path_from_main

        path_from_data_dir = os.path.join(DATA_DIR, relative_directory, path)
        assert os.path.exists(path_from_data_dir), f'{self.__class__.__name__}: ' \
                                                   f'path \'{path_from_data_dir}\' does not exists.'
        return path_from_data_dir
