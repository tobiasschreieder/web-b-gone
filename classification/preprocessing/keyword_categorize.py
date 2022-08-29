import logging

from flashtext.keyword import KeywordProcessor

from classification.preprocessing import Category
from config import Config

cfg = Config.get()
log = logging.getLogger('keyword_categorize')

key_processors = None


def create_key_processors():
    keys_auto = ['auto', 'car', 'drive', 'driving', 'highway', 'automobile', 'license plate', 'tire', 'trunk',
                 'hubcap', 'bumper', 'windshield', 'windscreen', 'wheel', 'airbag', 'motor', 'seatbelt',
                 'fuel', 'tank', 'brake', 'engine', 'model']
    keys_book = ['book', 'author', 'pages', 'novel', 'spine', 'title', 'literature', 'chapter', 'story',
                 'writer', 'isbn', 'publisher', 'publication']
    keys_camera = ['camera', 'ratio', 'bokeh', 'composition', 'iso', 'pixel', 'lenses', 'resolution', 'shutter',
                   'photo', 'photography', 'filter', 'zoom', 'model', 'manufacturer']
    keys_job = ['job', 'office', 'business', 'employee', 'worker', 'recruitment', 'employer', 'promotion',
                'occupation', 'career', 'applicant', 'application', 'experience', 'company', 'interview',
                'wage', 'shift', 'staff', 'recruiter', 'qualification', 'retirement', 'vacancy']
    keys_movie = ['movie', 'film', 'actor', 'actress', 'regisseur', 'director', 'oscar', 'genre', 'rating']
    keys_nba = ['nba', 'basketball', 'showdown', 'matchup', 'team', 'playoffs', 'dribbling',
                'passing', 'time-out', 'foul', 'overhead pass', 'slam dunk']
    keys_restaurant = ['restaurant', 'cafeteria', 'bar', 'food', 'drinks', 'eat',
                       'cuisine', 'booking', 'menu', 'appetizers', 'dish', 'soup', 'dessert']
    keys_uni = ['university', 'campus', 'graduate', 'graduation', 'study', 'grade', 'undergraduate',
                'postgraduate', 'education', 'tuition', 'exam', 'academic', 'bachelor', 'master',
                'degree', 'thesis', 'research']

    kp0 = KeywordProcessor()
    processors = []

    for key_list in [keys_auto, keys_book, keys_camera, keys_job, keys_movie, keys_nba, keys_restaurant, keys_uni]:
        kp = KeywordProcessor()
        kp.add_keywords_from_list(key_list)
        kp0.add_keywords_from_list(key_list)
        processors.append(kp)

    return [kp0] + processors


def percentage1(dum0, dumx):
    if dum0 == 0:
        return 0

    try:
        ans = float(dumx) / float(dum0)
        ans = ans * 100
    except ZeroDivisionError:
        return 0.0
    else:
        return float(ans)


def find_class(text_from_html: str) -> (Category, float):
    global key_processors
    if key_processors is None:
        key_processors = create_key_processors()

    x = str(text_from_html)
    len_keys = [len(kp.extract_keywords(x)) for kp in key_processors]

    y0 = len_keys.pop(0)
    if y0 == 0:
        return Category.NONE, 0

    per_list = [percentage1(y0, y) for y in len_keys]
    max_per = max(per_list)
    max_index = per_list.index(max_per)
    return Category.get(max_index), max_per
