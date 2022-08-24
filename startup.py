import argparse
import logging
import pathlib
import matplotlib.pyplot as plt
from typing import Any, Dict

from classification.category_models import RandomCategoryModel
from classification.preprocessing import Category, Website, extract_restruc_swde
from config import Config
# from frontend import start_server
from evaluation import extraction, classification
from extraction import StructuredTemplateExtractionModel, RandomExtractionModel, NeuralNetExtractionModel
from utils import setup_logger_handler

args: Dict[str, Any] = None


def init_logging():
    """
    Method where the root logger is setup
    """

    root = logging.getLogger()
    setup_logger_handler(root)
    root.setLevel(logging.INFO)

    root.info('Logging initialised')
    root.setLevel(logging.DEBUG)
    root.debug('Set to debug level')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", default=pathlib.Path('data'), type=pathlib.Path,
                        dest='data_dir', help='Path to input directory.')
    parser.add_argument("-o", "--output-dir", default=pathlib.Path('out'), type=pathlib.Path,
                        dest='out_dir', help='Path to output directory.')

    parser.add_argument("-w", "--working-dir", default=pathlib.Path('working'), type=pathlib.Path,
                        dest='work_dir', help='Path to working directory. (Location of networks)')
    parser.add_argument("-cfg", "--config", default=pathlib.Path('config.json'), type=pathlib.Path,
                        dest='config', help='Path to config.json file.')

    parser.add_argument('-njobs', '--number-jobs', type=int, dest='n_jobs', default=-1,
                        help='Number of processors to use in parallel processes. -1 = all Processors,'
                             ' -2 = all processors but one')

    parser.add_argument('-web', '--website', action='store_true', dest='frontend',
                        help='Start flask web server.')
    parser.add_argument('-p', '--port', type=int, dest='port', default=5000,
                        help='Port for web server.')
    parser.add_argument('-host', '--host', type=str, dest='host', default='0.0.0.0',
                        help='Host address for web server.')

    global args
    args = parser.parse_args()
    args = vars(args)

    if 'config' in args.keys():
        Config._save_path = args['config']
    else:
        cfg = Config.get()
        if 'data_dir' in args.keys():
            cfg.data_dir = args['data_dir']
        if 'out_dir' in args.keys():
            cfg.output_dir = args['out_dir']
        if 'work_dir' in args.keys():
            cfg.working_dir = args['work_dir']

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.working_dir.mkdir(parents=True, exist_ok=True)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.save()


def handle_args():
    # if args['frontend']:
    #     log.info('Start flask frontend')
    #     start_server(host=args['host'], port=args['port'])
    #     sys.exit(0)

    main()


def main():
    """
    normal program run
    :return:
    """

    log.info('do main stuff')

    # results_classification = classification.evaluate_classification(model_cls_classification=RandomCategoryModel,
    #                                                                 train_test_split=0.7,
    #                                                                 max_size=10000,
    #                                                                 split_type="website")
    # log.info(results_classification)

    results_extraction = extraction.evaluate_extraction(model_cls_extraction=NeuralNetExtractionModel,
                                                        category=Category.NBA_PLAYER,
                                                        train_test_split=0.7,
                                                        max_size=10,
                                                        split_type="website",
                                                        **{"name": "text_2", "version": "NerV2"})
    log.info(results_extraction)

    # web_ids = Website.get_website_ids(max_size=10, categories=Category.NBA_PLAYER)

    # struc_temp_model = StructuredTemplateExtractionModel(Category.NBA_PLAYER)
    # struc_temp_model.train(web_ids[0:5])
    # result = struc_temp_model.extract(web_ids[5:10])

    # ner_temp_model = NeuralNetExtractionModel(Category.NBA_PLAYER, 'text_1', 'NerV1')
    # history = ner_temp_model.train(web_ids[0:8])
    # result = ner_temp_model.extract(web_ids[8:10])
    # print(result)

    pass


if __name__ == '__main__':
    parse_args()
    init_logging()
    log = logging.getLogger('startup')
    try:
        handle_args()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
