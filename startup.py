import argparse
import logging
import pathlib
from typing import Any, Dict

from classification import NeuralNetCategoryModel, KeywordModel
from classification.preprocessing import Category, Website
from config import Config
from evaluation import extraction, classification
from extraction import StrucTempExtractionModel, NeuralNetExtractionModel, CombinedExtractionModel
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

    # Dataset setup stuff
    parser.add_argument('-swde', '--setup-swde', type=pathlib.Path, dest='setup_swde',
                        help='ZIP file path to perform the SWDE setup and restructuring method on.')
    parser.add_argument('-reswde', '--setup-restruc-swde')
    parser.add_argument('-cswde', '--compress-restruc-swde')

    # parser.add_argument('-njobs', '--number-jobs', type=int, dest='n_jobs', default=-1,
    #                     help='Number of processors to use in parallel processes. -1 = all Processors,'
    #                          ' -2 = all processors but one')

    # parser.add_argument('-web', '--website', action='store_true', dest='frontend',
    #                     help='Start flask web server.')
    # parser.add_argument('-p', '--port', type=int, dest='port', default=5000,
    #                     help='Port for web server.')
    # parser.add_argument('-host', '--host', type=str, dest='host', default='0.0.0.0',
    #                     help='Host address for web server.')

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

    # Evaluate a CategoryModel, example with NeuralNetCategoryModel
    results_classification = classification.evaluate_classification(
        model_cls_classification=NeuralNetCategoryModel,
        train_test_split=0.7,
        max_size=10000,
        split_type="domain",
        name="domain_10k",
        version="V1",
    )
    log.info(results_classification)

    # Evaluate a ExtractionModel, example with NeuralNetExtractionModel
    results_extraction = extraction.evaluate_extraction(
        model_cls_extraction=NeuralNetExtractionModel,
        category=Category.NBA_PLAYER,
        train_test_split=0.7,
        max_size=-1,
        split_type="website",
        name="nba_player_website",
        version='NerV1',
    )
    log.info(results_extraction)

    # Get a list of website ids
    web_ids = Website.get_website_ids(max_size=-1, categories=Category.NBA_PLAYER)

    # Categorize website with KeywordModel
    cat_key = KeywordModel()
    result = cat_key.classification(web_ids)

    # Categorize website with NeuralNetCategoryModel
    cat_net = NeuralNetCategoryModel('cat_model_1', 'V1')
    cat_net.classification(web_ids)

    # Train and extract from a StrucTempExtractionModel
    struc_temp_model = StrucTempExtractionModel(Category.NBA_PLAYER, 'struc_model_1')
    struc_temp_model.train(web_ids)
    result = struc_temp_model.extract(web_ids, k=3, n_jobs=-2)

    # Train and extract from a Spacy NeuralNetCategoryModel
    ner_spacy = NeuralNetExtractionModel(Category.NBA_PLAYER, 'ner_model_1', 'NerV1')
    ner_spacy.train(web_ids)
    result = ner_spacy.extract(web_ids)

    # Train and extract from a own LSTM NeuralNetCategoryModel
    ner_lstm = NeuralNetExtractionModel(Category.NBA_PLAYER, 'ner_model_1', 'NerV2')
    ner_lstm.train(web_ids)
    result = ner_lstm.extract(web_ids)

    # Extract from a CombinedExtractionModel
    combine_model = CombinedExtractionModel(Category.NBA_PLAYER, ner_name='ner_model_1', struc_name='struc_model_1')
    result = combine_model.extract(web_ids[:2])

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
