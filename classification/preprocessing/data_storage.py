import dataclasses
import json
import random
from aenum import Enum, MultiValue
from typing import List, Dict, Union

from config import Config

cfg = Config.get()
faulty_ids = []


class Category(Enum):
    _init_ = 'value fullname'
    _settings_ = MultiValue

    AUTO = 1, 'Auto'
    BOOK = 2, 'Book'
    CAMERA = 3, 'Camera'
    JOB = 4, 'Job'
    MOVIE = 5, 'Movie'
    NBA_PLAYER = 6, 'NBA Player'
    RESTAURANT = 7, 'Restaurant'
    UNIVERSITY = 8, 'University'
    NONE = 9, 'NONE'

    @staticmethod
    def get(name: str) -> 'Category':
        """
        TODO
        :param name:
        :return:
        """
        name_low = name.lower()
        for cat in Category:
            if name_low == cat.name.lower():
                return cat
        else:
            if name_low == 'nba player' or name_low == 'nbaplayer' or name_low == 'nba_player':
                return Category.NBA_PLAYER
        raise ValueError(f'No Category found with name "{name}"')

    def __int__(self):
        return self.value

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

    @staticmethod
    def get_attr_type(attr_name: str) -> str:
        numbers = ['isbn_13', 'height', 'weight']
        text = ['model', 'engine', 'fuel_economy', 'title', 'mpaa_rating']
        name = ['author', 'publisher', 'manufacturer', 'company',
                'director', 'genre', 'name', 'team', 'cuisine', 'type']
        date = ['publication_date', 'date_posted']
        rest = ['price', 'location', 'address', 'phone', 'website']

        if attr_name in numbers:
            return 'number'
        elif attr_name in text:
            return 'text'
        elif attr_name in name:
            return 'name'
        elif attr_name in date:
            return 'date'
        else:
            return 'other'


@dataclasses.dataclass
class GroundTruth:
    category: Category
    attributes: Dict[str, List[str]]

    @classmethod
    def load(cls, web_id: str) -> 'GroundTruth':
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
            with web_path.joinpath('groundtruth.json').open('r+') as file:
                truth_json = json.load(file)

            try:
                cat = Category.get(truth_json['category'])
            except ValueError as e:
                raise ValueError(f"The saved category '{truth_json['category']}' is not recognizable "
                                 f"for this program.") from e
            return cls(
                category=cat,
                attributes=truth_json,
            )

        raise ValueError(f'Given web_id "{web_id}" does not exist.')


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
                truth=GroundTruth.load(web_id),
                file_path=web_path.joinpath('website.htm'),
            )

        raise ValueError(f'Given web_id "{web_id}" does not exist.')

    @staticmethod
    def get_website_ids(max_size: int = -1, rdm_sample: bool = False, seed: str = 'seed',
                        domains: Union[str, List[str]] = None, sample_size: float = 0.8,
                        categories: Union[Category, List[Category]] = None) -> List[str]:
        """
        Returns number of website ids in a list. If max_size is < 1 return all website ids.
        It is possible to get a random subset of specified website ids. For this a sample size can be set.
        If sample size and max size are given max size is preferred.

        :param sample_size: Percentage of all web_ids in sample.
        :param categories: List of domains where the websites are from. If list is empty all domains are used.
        :param domains: List of domains where the websites are from. If list is empty all domains are used.
        :param max_size: Parameter to determine maximal length of returned list.
        :param rdm_sample: If the returns list should be a random sublist.
        :param seed: Seed for the random generator.
        :return: List of website ids as strings
        """

        rswde = cfg.data_dir.joinpath('restruc_swde')

        if categories is not None:
            if isinstance(categories, Category):
                categories = [categories]
            elif isinstance(categories, list):
                if len(categories) == 0:
                    categories = None
            else:
                raise ValueError(f'categories is not a Category or a List, but {type(categories)}')

        if domains is not None:
            if isinstance(domains, str):
                if len(domains) > 0:
                    domains = [domains]
                else:
                    domains = None
            elif isinstance(domains, list):
                if len(domains) == 0:
                    domains = None
            else:
                raise ValueError(f'domains is not a string or a List, but {type(domains)}')

        final_size = max_size
        if rdm_sample:
            max_size = -1

        id_list = []
        count = 0
        check_length = max_size > 0
        for category in rswde.iterdir():

            if not category.is_dir():
                continue

            try:
                cat = Category.get(category.name)
            except ValueError:
                continue

            if categories is not None and cat not in categories:
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

        if rdm_sample:
            len_sample = len(id_list)

            if final_size > 0:
                len_sample = min(final_size, len_sample)
            elif 0 < sample_size < 1:
                len_sample = int(len_sample * sample_size)

            random.seed(seed)
            return random.sample(id_list, len_sample)

        return id_list

    @staticmethod
    def get_all_domains(category: Category) -> List[str]:
        """
        TODO
        :param category:
        :return:
        """
        for category_dir in cfg.data_dir.joinpath('restruc_swde').iterdir():
            try:
                if Category.get(category_dir.name) == category:
                    with category_dir.joinpath('domains.json').open('r+') as dom_file:
                        return json.load(dom_file)
            except ValueError:
                continue

        raise ValueError(f'Category "{category}" is not in the restructured SWDE dataset.')
