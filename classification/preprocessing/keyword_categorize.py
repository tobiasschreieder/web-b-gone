import logging

import html2text as html2text
from flashtext.keyword import KeywordProcessor

from config import Config
from data_storage import GroundTruth, Category

cfg = Config.get()
log = logging.getLogger('keyword_categorize')

# https://towardsdatascience.com/industrial-classification-of-websites-by-machine-learning-with-hands-on-python-3761b1b530f1

"""
    Returns the corresponding list of groundtruth categories to the web id list input.
"""
cat_auto_keywords = ['auto', 'car', 'drive', 'driving', 'highway', 'automobile', 'license plate', 'tire', 'trunk',
                     'hubcap', 'bumper', 'windshield', 'windscreen', 'wheel', 'airbag', 'motor', 'seatbelt',
                     'fuel', 'tank', 'brake', 'engine', 'model']
cat_book_keywords = ['book', 'author', 'pages', 'novel', 'spine', 'title', 'literature', 'chapter', 'story', 'writer',
                     'isbn', 'publisher', 'publication']
cat_camera_keywords = ['camera', 'ratio', 'bokeh', 'composition', 'iso', 'pixel', 'lenses', 'resolution', 'shutter',
                       'photo', 'photography', 'filter', 'zoom', 'model', 'manufacturer']
cat_job_keywords = ['job', 'office', 'business', 'employee', 'worker', 'recruitment', 'employer', 'promotion',
                    'occupation', 'career', 'applicant', 'application', 'experience', 'company', 'interview',
                    'wage', 'shift', 'staff', 'recruiter', 'qualification', 'retirement', 'vacancy']
cat_movie_keywords = ['movie', 'film', 'actor', 'actress', 'regisseur', 'director', 'oscar', 'genre', 'rating']
cat_nbaplayer_keywords = ['nba', 'basketball', 'showdown', 'matchup', 'team', 'playoffs', 'dribbling', 'passing',
                          'time-out', 'foul', 'overhead pass', 'slam dunk']
cat_restaurant_keywords = ['restaurant', 'cafeteria', 'bar', 'food', 'drinks', 'eat', 'cuisine', 'booking', 'menu',
                           'appetizers', 'dish', 'soup', 'dessert']
cat_university_keywords = ['university', 'campus', 'graduate', 'graduation', 'study', 'grade', 'undergraduate',
                           'postgraduate', 'education', 'tuition', 'exam', 'academic', 'bachelor', 'master',
                           'degree', 'thesis', 'research']
keywords = cat_auto_keywords + cat_book_keywords + cat_camera_keywords + cat_job_keywords + cat_movie_keywords + cat_nbaplayer_keywords + cat_restaurant_keywords + cat_university_keywords

kp0 = KeywordProcessor()
for word in keywords:
    kp0.add_keyword(word)
kp1 = KeywordProcessor()
for word in cat_auto_keywords:
    kp1.add_keyword(word)
kp2 = KeywordProcessor()
for word in cat_book_keywords:
    kp2.add_keyword(word)
kp3 = KeywordProcessor()
for word in cat_camera_keywords:
    kp3.add_keyword(word)
kp4 = KeywordProcessor()
for word in cat_job_keywords:
    kp4.add_keyword(word)
kp5 = KeywordProcessor()
for word in cat_movie_keywords:
    kp5.add_keyword(word)
kp6 = KeywordProcessor()
for word in cat_nbaplayer_keywords:
    kp6.add_keyword(word)
kp7 = KeywordProcessor()
for word in cat_restaurant_keywords:
    kp7.add_keyword(word)
kp8 = KeywordProcessor()
for word in cat_university_keywords:
    kp8.add_keyword(word)


def percentage1(dum0, dumx):
    try:
        ans = float(dumx) / float(dum0)
        ans = ans * 100
    except:
        return 0
    else:
        return ans


def find_class(text_from_html: str) -> (Category, float):
    x = str(text_from_html)
    y0 = len(kp0.extract_keywords(x))
    y1 = len(kp1.extract_keywords(x))
    y2 = len(kp2.extract_keywords(x))
    y3 = len(kp3.extract_keywords(x))
    y4 = len(kp4.extract_keywords(x))
    y5 = len(kp5.extract_keywords(x))
    y6 = len(kp6.extract_keywords(x))
    y7 = len(kp7.extract_keywords(x))
    y8 = len(kp8.extract_keywords(x))
    Total_matches = y0
    per1 = float(percentage1(y0, y1))
    per2 = float(percentage1(y0, y2))
    per3 = float(percentage1(y0, y3))
    per4 = float(percentage1(y0, y4))
    per5 = float(percentage1(y0, y5))
    per6 = float(percentage1(y0, y6))
    per7 = float(percentage1(y0, y7))
    per8 = float(percentage1(y0, y8))
    percentage = 0
    category = Category.get("NONE")
    if y0 == 0:
        category = Category.get("NONE")
    else:
        if per1 >= per2 and per1 >= per3 and per1 >= per4 and per1 >= per5 and per1 >= per6 and per1 >= per7 \
                and per1 >= per8:
            category = Category.AUTO
            percentage = per1
        elif per2 >= per3 and per2 >= per1 and per2 >= per4 and per2 >= per5 and per2 >= per6 and per2 >= per7 \
                and per2 >= per8:
            category = Category.BOOK
            percentage = per2
        elif per3 >= per1 and per3 >= per2 and per3 >= per4 and per3 >= per5 and per3 >= per6 and per3 >= per7 \
                and per3 >= per8:
            category = Category.CAMERA
            percentage = per3
        elif per4 >= per1 and per4 >= per2 and per4 >= per3 and per4 >= per5 and per4 >= per6 and per4 >= per7 \
                and per4 >= per8:
            category = Category.JOB
            percentage = per4
        elif per5 >= per1 and per5 >= per2 and per5 >= per3 and per5 >= per4 and per5 >= per6 and per5 >= per7 \
                and per5 >= per8:
            category = Category.MOVIE
            percentage = per5
        elif per6 >= per1 and per6 >= per2 and per6 >= per3 and per6 >= per4 and per6 >= per5 and per6 >= per7 \
                and per6 >= per8:
            category = Category.NBA_PLAYER
            percentage = per6
        elif per7 >= per1 and per7 >= per2 and per7 >= per3 and per7 >= per4 and per7 >= per5 and per7 >= per6 \
                and per7 >= per8:
            category = Category.RESTAURANT
            percentage = per7
        elif per8 >= per1 and per8 >= per2 and per8 >= per3 and per8 >= per4 and per8 >= per5 and per8 >= per6 \
                and per8 >= per7:
            category = Category.UNIVERSITY
            percentage = per8

    return (category, percentage)

# maybe also give out percentage for category for neural net
