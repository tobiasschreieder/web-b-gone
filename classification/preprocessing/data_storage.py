import dataclasses
import json
import random
from enum import Enum
from pathlib import Path
from typing import List, Dict, Union

from config import Config
from .swde_setup import convert_category_name

cfg = Config.get()
faulty_ids = []


class Category(Enum):
    AUTO = 'Auto'
    BOOK = 'Book'
    CAMERA = 'Camera'
    JOB = 'Job'
    MOVIE = 'Movie'
    NBA_PLAYER = 'NBA Player'
    RESTAURANT = 'Restaurant'
    UNIVERSITY = 'University'

    def get_attribute_names(self) -> List[str]:
        """
        TODO
        :return:
        """
        att_name = {'Auto': ['model', 'price', 'engine', 'fuel_economy'],
                    'Book': ['title', 'author', 'isbn_13', 'publisher', 'publication_date'],
                    'Camera': ['model', 'price', 'manufacturer'],
                    'Job:': ['title', 'company', 'location', 'date_posted'],
                    'Movie': ['title', 'director', 'genre', 'mpaa_rating'],
                    'NBA Player': ['name', 'team', 'height', 'weight'],
                    'Restaurant': ['name', 'address', 'phone', 'cuisine'],
                    'University': ['name', 'phone', 'website', 'type']}

        if self.value in att_name.keys():
            return att_name[self.value]

        raise ValueError(f'{self.value} is not a valid category name. Maybe you forgot to add it in the mapping?')


@dataclasses.dataclass
class GroundTruth:
    category_name: Category
    attributes: Dict[str, List[str]]

    @classmethod
    def load(cls, file_path: Path) -> 'GroundTruth':
        """
        TODO
        :param file_path:
        :return:
        """
        with file_path.open('r+') as file:
            truth_json = json.load(file)

        return cls(
            category_name=truth_json['category'],
            attributes=truth_json,
        )


@dataclasses.dataclass
class Website:
    web_id: str
    domain_name: str
    url: str
    truth: GroundTruth

    file_path: str

    @classmethod
    def load(cls, web_id: str) -> 'Website':
        """
        TODO
        :param web_id:
        :return:
        """
        rswde = cfg.data_dir.joinpath('restruc_swde/')
        for category in rswde.iterdir():
            web_path = category.joinpath(web_id[:3]).joinpath(web_id)
            if not web_path.exists():
                continue

            # Found correct category
            with web_path.joinpath('website-url.txt').open('r+') as url_file:
                url = url_file.readline()

            domain = url.replace('http://', '').replace('https://', '').split('/')[0].replace('www.', '')

            return cls(
                web_id=web_id,
                domain_name=domain,
                url=url,
                truth=GroundTruth.load(web_path.joinpath('groundtruth.json')),
                file_path=web_path.joinpath('website.htm'),
            )

        raise ValueError(f'Given web_id "{web_id}" does not exist.')

    @staticmethod
    def get_website_ids(max_size: int = -1, rdm_sample: bool = False, seed: str = 'seed',
                        domains: Union[str, List[str]] = None, categories: Union[str, List[str]] = None) -> List[str]:
        """
        Returns number of website ids in a list. If max_size is < 1 return all website ids.
        It is possible to get a random subset of specified website ids.

        :param categories: List of domains where the websites are from. If list is empty all domains are used.
        :param domains: List of domains where the websites are from. If list is empty all domains are used.
        :param max_size: Parameter to determine maximal length of returned list.
        :param rdm_sample: If the returns list should be a random sublist.
        :param seed: Seed for the random generator.
        :return: List of website ids as strings
        """

        rswde = cfg.data_dir.joinpath('restruc_swde')

        if categories is not None:
            if isinstance(categories, str):
                if len(categories) > 0:
                    categories = [categories]
                else:
                    categories = None
            elif isinstance(categories, list):
                if len(categories) == 0:
                    categories = None

        if domains is not None:
            if isinstance(domains, str):
                if len(domains) > 0:
                    domains = [domains]
                else:
                    domains = None
            elif isinstance(domains, list):
                if len(domains) == 0:
                    domains = None

        final_size = max_size
        if rdm_sample:
            max_size = -1

        id_list = []
        count = 0
        check_length = max_size > 0
        for category in rswde.iterdir():

            if categories is not None and convert_category_name(category.name) not in categories:
                continue

            if not category.is_dir():
                continue

            for idir in category.iterdir():
                if not idir.is_dir():
                    continue
                for web_hash in idir.iterdir():
                    if web_hash.name in faulty_ids:
                        continue

                    if domains is not None:
                        entry = Website.load(web_hash.name)
                        if entry.domain_name not in domains:
                            continue

                    id_list.append(web_hash.name)
                    count += 1
                    if check_length and count >= max_size:
                        return id_list

        if rdm_sample and 0 < final_size < len(id_list):
            random.seed(seed)
            return random.sample(id_list, final_size)

        return id_list

    @staticmethod
    def get_all_domains(category: str) -> List[str]:
        """
        TODO
        :param category:
        :return:
        """
        for category_dir in cfg.data_dir.joinpath('restruc_swde').iterdir():
            if convert_category_name(category_dir.name) == category:
                with category_dir.joinpath('domains.json').open('r+') as dom_file:
                    return json.load(dom_file)

        raise ValueError(f'Category "{category}" is not a valid category.')
