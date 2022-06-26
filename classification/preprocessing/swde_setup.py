import logging
from pathlib import Path

from pyunpack import Archive

from config import Config

cfg = Config.get()
log = logging.getLogger('swde_setup')


def setup_swde_dataset(zip_path: Path):

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

    pass

