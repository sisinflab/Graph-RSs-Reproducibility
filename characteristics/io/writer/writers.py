import os

from characteristics.io.paths import RESULT_DIR


class Writer:

    def __init__(self, main_directory=None):
        if main_directory is None:
            main_directory = RESULT_DIR
        assert os.path.exists(main_directory), \
            f'{self.__class__.__name__}: main directory not found at \'{main_directory}\''
        self._main_directory = main_directory

        self._last_stored = None

    def write(self, *args, **kwargs):
        pass

    def path(self, relative_path, relative_directory=None):
        if relative_directory is None:
            relative_directory = ''
        folder = os.path.join(self._main_directory, relative_directory)
        assert os.path.exists(folder), \
            f'{self.__class__.__name__}: path not found at \'{folder}\'.'
        path = os.path.join(folder, relative_path)
        return path


class TsvWriter(Writer):

    def __init__(self, main_directory=None, sep=None, header=None, columns=None, drop_index=None,
                         drop_header=None, extension=None):
        super().__init__(main_directory=main_directory)

        if sep is None:
            sep = '\t'
        if header is None:
            header = True
        if columns is None:
            columns = columns
        self._index = True
        if drop_index is None:
            drop_index = True
        if drop_header is None:
            drop_header = True
        if extension is None:
            extension = '\t'

        self._sep = sep
        self._header = header
        self._columns = columns

        self._drop_index = drop_index
        if self._drop_index is True:
            self._index = False

        self._drop_header = drop_header
        if self._drop_header is True:
            self._header = False

        self._extension = extension
        self._drop_index = None
        self._drop_header = None
        self._extension = '.tsv'

    def write(self, dataset, file_name=None, directory=None):

        if directory is None:
            directory = ''

        if file_name is None:
            file_name = 'tsvDataset'

        target_path = self.path(file_name, directory) + self._extension

        dataset.to_csv(target_path, sep=self._sep, header=self._header, index=self._drop_index)

        print(f'{self.__class__.__name__}: tsv dataset stored at: \'{target_path}\'')
        self._last_stored = target_path
        return target_path
