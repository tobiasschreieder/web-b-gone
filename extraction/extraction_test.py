import random
from copy import copy
from pathlib import Path

from bs4 import BeautifulSoup

import evaluation.text_preprocessing as tp
from classification.preprocessing import Website


class Template:
    def __init__(self):
        self.web_ids_trained_on = []
        self.attributes = []
        self.positions = []
        self.text_info = []

    def get_web_ids(self):
        return self.web_ids_trained_on

    def add_attribute(self, attribute, position, text_info, web_id):
        self.attributes.append(attribute)
        self.positions.append(position)
        self.text_info.append(text_info)
        self.web_ids_trained_on.append(web_id)

    def get_from_attribute(self, attribute):
        if attribute in self.attributes:
            index = [i for i, x in enumerate(self.attributes) if x == attribute]
            all = []
            for i in index:
                all.append({'attribute': attribute,
                            'position': self.positions[i],
                            'text_info': self.text_info[i],
                            'web_id': self.web_ids_trained_on[i]})
            return all
        else:
            return False

    def get_from_web_id(self, web_id):
        if not web_id in self.web_ids_trained_on:
            return False
        else:
            index = [i for i, x in enumerate(self.web_ids_trained_on) if x == web_id]
            all = []
            for i in index:
                all.append({'attribute': self.attributes[i],
                            'position': self.positions[i],
                            'text_info': self.text_info[i],
                            'web_id': web_id})
            return all

    def get_from_web_id_and_attribute(self, web_id, attribute):
        if not web_id in self.web_ids_trained_on:
            return False
        else:
            index = [i for i, x in enumerate(self.web_ids_trained_on) if x == web_id]
            for i in index:
                if self.attributes[i] == attribute:
                    return {'attribute': self.attributes[i],
                            'position': self.positions[i],
                            'text_info': self.text_info[i],
                            'web_id': web_id}
        return False

    def get_all_attributes(self):
        all = []
        for index in range(len(self.attributes)):
            all.append({'attribute': self.attributes[index],
                        'position': self.positions[index],
                        'text_info': self.text_info[index]})
        return all

    def print(self):
        print("TEMPLATE")
        all = self.get_all_attributes()
        for e in all:
            print(e)
        print("---")


def learn_template(web_ids):
    print(web_ids)
    template = Template()
    for web_id in web_ids:
        website = Website.load(web_id)
        with Path(website.file_path).open(encoding='utf-8') as htm_file:
            soup = BeautifulSoup(htm_file, features="html.parser")

        body = soup.find('body')
        # html_tree = build_html_tree(body, [])
        # print_html_tree(html_tree)
        text_position_mapping = build_text_position_mapping(body)

        attributes = website.truth.attributes
        if 'category' in attributes:
            attributes.pop('category')
        for key in attributes:
            attributes[key] = tp.preprocess_text_html(attributes[key][0])
            best_match = 0
            best_position = None
            for mapping in text_position_mapping:
                match = simple_string_match(mapping['text'], website.truth.attributes[key])
                if match > best_match:
                    best_match = match
                    text_correct = str(website.truth.attributes[key]).split(" ")
                    text_found = str(mapping['text']).split(" ")
                    text_info = [0] * len(text_found)
                    for correct_word in text_correct:
                        if correct_word in text_found:
                            pos = text_found.index(correct_word)
                            text_info[pos] = 1

                    best_position = {'attribute': key,
                                     'position': mapping['position'],
                                     'text_info': text_info}

            template.add_attribute(attribute=best_position['attribute'],
                                   position=best_position['position'],
                                   text_info=best_position['text_info'],
                                   web_id=web_id)
    # TODO: cluster templates
    return template


def simple_string_match(html_string, attribute_string):
    n = 3
    grams = [attribute_string[i:i + n] for i in range(len(attribute_string) - n + 1)]

    # to speed-up the process
    if len(grams) > 100:
        grams = random.sample(list, 50)

    matches = 0
    for gram in grams:
        if gram in html_string:
            matches += 1
    if len(grams) * 0.9 > matches:
        return 0

    len_diff = abs(len(html_string) - len(attribute_string))
    return (max(len(html_string), len(attribute_string)) + 1) / (len_diff + 1)


def extract_infos(web_id, template):
    print(web_id)
    website = Website.load(web_id)
    with Path(website.file_path).open(encoding='utf-8') as htm_file:
        soup = BeautifulSoup(htm_file, features="html.parser")

    body = soup.find('body')
    # html_tree = build_html_tree(body, [])
    # print_html_tree(html_tree)
    text_position_mapping = build_text_position_mapping(body)

    attributes = website.truth.attributes
    if 'category' in attributes:
        attributes.pop('category')

    candidates = []
    for key in attributes:
        if key in ['name', 'team']:
            filter_category = 'Name'
        if key in ['height', 'weight']:
            filter_category = 'Number'

        key_candidates = candidates_filter(filter_category, text_position_mapping)
        candidates.append({'attribute': key,
                           'candidates': key_candidates})
    result = find_best_candidate(candidates, template)
    return result


def find_best_candidate(candidates, template):
    """
         c1, c2, c3  c4  (NAME)
    t1 [ .   .   .   . ]
    t2 [ .   .   .   . ]
    t3 [ .   .   .   . ]
    t4 [ .   .   .   . ]

         c1, c2, c3  (TEAM)
    t1 [ .   .   . ]
    t2 [ .   .   . ]
    t3 [ .   .   . ]
    t4 [ .   .   . ]

         c1, c2, c3  (HEIGHT)
    t1 [ .   .   . ]
    t2 [ .   .   . ]
    t3 [ .   .   . ]
    t4 [ .   .   . ]


    --> Add
        Best from NAME_t1 + TEAM_t1 + HEIGHT_t1
        Best from NAME_t2 + TEAM_t2 + HEIGHT_t2
        Best from NAME_t3 + TEAM_t3 + HEIGHT_t3

    --> choose best Sum and use this template and this candidate
    """
    web_ids = template.get_web_ids()
    best_candidate = {'score': 0,
                      'candidate': [],
                      'web_id': None}
    for t in web_ids:
        score_sum = 0
        candidates_sum = []
        for key in candidates:
            attribute = key['attribute']
            candidates_list = key['candidates']
            row = []
            position_t = template.get_from_web_id_and_attribute(web_id=t, attribute=attribute)['position']
            for c in candidates_list:
                position_c = c['position']
                score = position_scoring(position_c, position_t)
                row.append(score)
            best_score_from_row = max(row)
            best_score_candidate = row.index(best_score_from_row)
            score_sum += best_score_from_row

            # apply text_info
            candidate = copy(candidates_list[best_score_candidate])
            correct_words = candidate['text'].split(" ")
            text_info = template.get_from_web_id_and_attribute(web_id=t, attribute=attribute)['text_info']
            if not all(x == 1 for x in text_info):
                # delete leading words
                i = 0
                while text_info[i] == 0:
                    correct_words = correct_words[1:]
                    i += 1
                    if i >= len(text_info):
                        break
                # delete trailing words
                i = len(text_info)
                while text_info[i - 1] == 0:
                    correct_words = correct_words[:-1]
                    i -= 1
                    if i < 0:
                        break
                candidate['text'] = correct_words

            candidates_sum.append({'attribute': attribute, 'candidate': candidate})

        if best_candidate['score'] < score_sum:
            best_candidate['score'] = score_sum
            best_candidate['candidate'] = candidates_sum
            best_candidate['web_id'] = t

    return best_candidate['candidate']


def position_scoring(position_c, position_t):
    n = 5
    grams = [position_c[i:i + n] for i in range(len(position_c) - n + 1)]
    # to speed-up the process
    if len(grams) > 100:
        grams = random.sample(list, 50)

    score = abs(len(position_c) - len(position_t)) * (-2)
    without_position_t = [x[1] for x in position_t]
    for gram in grams:
        gram_without_position = [x[1] for x in gram]
        if sublist(gram_without_position, without_position_t):
            score += 1
            if sublist(gram, position_t):
                score += 1
    return score


def sublist(lst1, lst2):
    if len(lst2) < len(lst1):
        lst1, lst2 = lst2, lst1
    if lst1[0] in lst2:
        indices_apperance = [i for i, x in enumerate(lst2) if x == lst1[0]]
        for i_p in indices_apperance:
            if lst2[i_p:i_p+len(lst1)] == lst1:
                return True
    return False


def candidates_filter(filter_category, text_position_mapping):
    filtered_mapping = []
    if filter_category == 'Name':
        for mapping in text_position_mapping:
            if candidate_filter_name(mapping['text']):
                filtered_mapping.append(mapping)
    elif filter_category == 'Number':
        for mapping in text_position_mapping:
            if candidate_filter_number(mapping['text']):
                filtered_mapping.append(mapping)
    return filtered_mapping


def candidate_filter_name(text):
    if text == "":
        return False
    if len(text) < 5:
        return False
    if text.count(" ") > 20:
        return False
    if any([char in text for char in ['%', '$', '!', '§', '&']]):
        return False
    if any(char.isdigit() for char in text):
        return False
    return True


def candidate_filter_number(text):
    if text == "":
        return False
    if text.count(" ") > 20:
        return False
    if not any(char.isdigit() for char in text):
        return False
    return True


def build_html_tree(section, current_tree):
    tags_to_ignore = ['script']
    children = section.find_all(recursive=False)
    tag = section.name
    text = section.findAll(text=True, recursive=False)
    text = tp.preprocess_text_html(text)

    if tag in tags_to_ignore:
        return {"tag": tag, "text": ""}

    if not children:
        return {"tag": tag, "text": text}

    child_subtree = [{"tag": tag, "text": text}]
    for child_section in children:
        child_subtree.append(build_html_tree(child_section, current_tree))
    return child_subtree


def build_text_position_mapping(section, current_pos=[], section_count=0):
    html_position = copy(current_pos)
    tags_to_ignore = ['script']
    children = section.find_all(recursive=False)
    tag = section.name
    text = section.findAll(text=True, recursive=False)
    text = tp.preprocess_text_html(text)

    if tag in tags_to_ignore:
        return []

    if not children:
        if not text == "":
            html_position.append([0, tag])
            return [{"text": text, "position": html_position}]

    temp_list = []
    if not text == "":
        html_position.append([0, tag])
        temp_list += [{"text": text, "position": html_position}]

    html_position.append([section_count, tag])
    section_count = 0
    for child_section in children:
        section_count += 1
        temp_list += build_text_position_mapping(child_section, html_position, section_count)

    return temp_list


def print_html_tree(lst, level=0):
    print('    ' * (level - 1) + '+---' * (level > 0) + lst[0]["tag"] + " : " + str(lst[0]["text"]))
    for l in lst[1:]:
        if type(l) is list:
            print_html_tree(l, level + 1)
        else:
            print('    ' * level + '+---' + l["tag"] + " : " + str(l["text"]))
