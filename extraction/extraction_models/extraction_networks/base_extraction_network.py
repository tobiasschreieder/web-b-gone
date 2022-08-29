import abc
import importlib
import logging
from abc import ABC
from pathlib import Path
from typing import List, Type, Dict

from keras.models import load_model
from tensorflow import keras

from config import Config

cfg = Config.get()
log = logging.getLogger('category_network')


class ExtractionNetwork(abc.ABC):

    name: str

    def __init__(self, name: str, **kwargs):
        """
        Create a ExtractionNetwork. Don't use this to get an ExtractionNetwork,
        use ExtractionNetwork.get(version) instead.

        :param name: name of the network.
        :param kwargs:
        """
        self.name = name

    @abc.abstractmethod
    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        Train network with given websites.

        :param web_ids: Website to train on.
        :param kwargs:  additional parameters.
        :return: None
        """
        pass

    @abc.abstractmethod
    def predict(self, web_ids: List[str], k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Predict extracted text for given websites.
        Returns a dictionary for each website where the keys are the extracted attributes
        and the values are lists with the extracted text.

        :param k: number of extract values per attribute.
        :param web_ids: websites to extract attribute from.
        :param kwargs: additional parameters.
        :return: List of dictionaries with extracted text.
        """
        pass

    @staticmethod
    def get(version: str) -> Type['ExtractionNetwork']:
        """
        Get the class of the ExtractionNetwork with the specified version.

        :param version: Version of the ExtractionNetwork class
        :return: class of ExtractionNetwork
        """
        found_versions = []
        # Find module with correct ExtractionNetwork class and import it.
        for module_path in Path(__file__).parent.iterdir():
            if module_path.suffix != '.py' or \
                    module_path.stem == '__init__' or module_path.stem == 'base_extraction_network':
                continue
            pkg_name = '.'.join(__name__.split('.')[:-1])
            module = importlib.import_module(f'.{module_path.stem}', package=pkg_name)
            try:
                return getattr(module, f'ExtractionNetwork{version}')
            except AttributeError:
                match = [attr[17:] for attr in module.__dict__.keys() if str(attr).startswith('ExtractionNetwork')]
                found_versions += match
                continue
        else:
            raise ValueError(f"No extraction network with version '{version}' found. "
                             f"Only found versions: {set(found_versions)}")

    @abc.abstractmethod
    def load(self) -> None:
        pass

    @abc.abstractmethod
    def save(self) -> None:
        pass


class BaseExtractionNetwork(ExtractionNetwork, ABC):
    model: keras.Model = None
    version: str
    description: str
    dir_path: Path = cfg.working_dir.joinpath(Path('models/extraction/'))

    def __init__(self, name: str, version: str, description: str, **kwargs):
        super().__init__(name)
        self.version = version
        self.description = description
        self.dir_path = self.dir_path.joinpath(self.version).joinpath(self.name)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """
        Load the keras model from disk.

        :return: None
        :raises ValueError: if the model doesn't exist.
        """
        load_path = self.dir_path.joinpath('model.hS')
        if not load_path.exists():
            raise ValueError(f"The model '{self.name}' for version {self.version} doesn't exit.")
        self.model = load_model(load_path.as_posix(), compile=False)

    def save(self) -> None:
        """
        Saves the keras model to disk.

        :return: None
        :raises ValueError: if the model is None (not trained).
        """
        if self.model is None:
            raise ValueError(f"No model to save. Model '{self.name}' for version {self.version} not set.")
        self.model.save(self.dir_path.joinpath('model.hS').as_posix())
