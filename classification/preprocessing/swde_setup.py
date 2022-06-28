import json
import logging
import shutil
from hashlib import md5
from pathlib import Path
from typing import List, Dict

from bs4 import BeautifulSoup, Tag
from pyunpack import Archive

from config import Config

cfg = Config.get()
log = logging.getLogger('swde_setup')


def setup_swde_dataset(zip_path: Path):
    """
    TODO
    :param zip_path:
    :return:
    """

    def extract_and_delete(file_path: Path):
        Archive(file_path).extractall(file_path.parent, auto_create_dir=True)
        file_path.unlink()

    swde = cfg.data_dir.joinpath('swde')
    Archive(zip_path).extractall(swde, auto_create_dir=True)

    extract_and_delete(swde.joinpath('groundtruth.7z'))

    pages = swde.joinpath('webpages')
    extract_and_delete(pages.joinpath('auto.7z'))
    extract_and_delete(pages.joinpath('book.7z'))
    extract_and_delete(pages.joinpath('camera.7z'))
    extract_and_delete(pages.joinpath('job.7z'))
    extract_and_delete(pages.joinpath('movie.7z'))
    extract_and_delete(pages.joinpath('nbaplayer.7z'))
    extract_and_delete(pages.joinpath('restaurant.7z'))
    extract_and_delete(pages.joinpath('university.7z'))


def convert_category_name(cat_name: str) -> str:
    """
    TODO
    :param cat_name:
    :return:
    """
    mapping = {'auto': 'Auto',
               'book': 'Book',
               'camera': 'Camera',
               'job:': 'Job',
               'movie': 'Movie',
               'nbaplayer': 'NBA Player',
               'restaurant': 'Restaurant',
               'university': 'University'}
    if cat_name in mapping.keys():
        return mapping[cat_name]
    else:
        return cat_name


def _calc_attribute_dict(truth_path: Path, site_name: str, attribute_name: str) -> Dict[int, List[str]]:
    temp_dict = {}
    begin_name = site_name.split('(')[0]

    with truth_path.joinpath(f'{begin_name}-{attribute_name}.txt').open('r+', encoding='utf-8') as truth_file:
        line_number = -1
        for line in truth_file:
            line_number += 1
            if line_number < 2:
                continue

            values = line.replace('\n', '').split('\t')
            row_id = int(values[0])
            count = int(values[1])
            if count == 0:
                temp_dict[row_id] = []
            else:
                temp_dict[row_id] = values[2:]
    return temp_dict


def restructure_swde(remove_old: bool = False):
    """
    TODO
    :param remove_old:
    :return:
    """
    swde_page = cfg.data_dir.joinpath('swde/webpages')
    swde_truth = cfg.data_dir.joinpath('swde/groundtruth')
    swde_new = cfg.data_dir.joinpath('restruc_swde')
    swde_new.mkdir(exist_ok=True)

    ground_truth = {'auto': ['model', 'price', 'engine', 'fuel_economy'],
                    'book': ['title', 'author', 'isbn_13', 'publisher', 'publication_date'],
                    'camera': ['model', 'price', 'manufacturer'],
                    'job': ['title', 'company', 'location', 'date_posted'],
                    'movie': ['title', 'director', 'genre', 'mpaa_rating'],
                    'nbaplayer': ['name', 'team', 'height', 'weight'],
                    'restaurant': ['name', 'address', 'phone', 'cuisine'],
                    'university': ['name', 'phone', 'website', 'type']}

    for category in swde_page.iterdir():
        if category.is_dir():
            log.info('Start category: %s', category.name)
            new_path = swde_new.joinpath(category.name)
            truth_path = swde_truth.joinpath(category.name)
            domains = []

            for site in category.iterdir():
                log.info('Start site: %s', site.name)
                att: Dict[str, Dict[int, List[str]]] = \
                    {
                        att_name: _calc_attribute_dict(truth_path, site.name, att_name)
                        for att_name in ground_truth[category.name]
                    }

                for page_path in site.iterdir():
                    with page_path.open(encoding='utf-8') as page_file:
                        # htm file
                        soup = BeautifulSoup(page_file, features="html.parser")
                        base: Tag = soup.findChildren('base')[0]
                        url = base.attrs['href']
                        url_hash = 'W' + md5(url.encode('utf-8')).hexdigest()[:16]
                        website_path = new_path.joinpath(url_hash[:3]).joinpath(url_hash)
                        website_path.mkdir(exist_ok=True, parents=True)
                        shutil.copy2(page_path, website_path.joinpath('website.htm'))

                        # Groundtruth
                        new_truth = {'category': convert_category_name(category.name)}
                        number = int(page_path.stem)
                        for k, v in att.items():
                            new_truth[k] = v[number]

                        with website_path.joinpath('groundtruth.json').open('w+') as truth_file:
                            json.dump(new_truth, truth_file)

                        with website_path.joinpath('website-url.txt').open('w+') as url_file:
                            url_file.write(url)
                            url_file.flush()

                domains.append(url.replace('http://', '').replace('https://', '').split('/')[0].replace('www.', ''))

            with new_path.joinpath('domains.json').open('w+') as dom_file:
                json.dump(domains, dom_file)

    if remove_old:
        log.info('Remove old SWDE.')
        swde_page.unlink()
        swde_truth.unlink()
