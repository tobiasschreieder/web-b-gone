import random
from copy import copy

import evaluation.text_preprocessing as tp


# TODO: refactor names and write doc


def simple_string_match(reference: str, query: str, n: int = 3) -> float:
    """
    Searches the query in the reference by using ngrams.
    :param n: Length of gram.
    :param reference: String to search in
    :param query: string to compare to reference
    :return: float witch represents the match
    """
    grams = [query[i:i + n] for i in range(len(query) - n + 1)]

    # to speed up the process
    if len(grams) > 100:
        grams = random.sample(grams, 50)  # TODO Before was list? cant be right

    matches = 0
    for gram in grams:
        if gram in reference:
            matches += 1
    if len(grams) * 0.9 > matches:
        return 0

    len_diff = abs(len(reference) - len(query))
    return (max(len(reference), len(query)) + 1) / (len_diff + 1)


def build_text_position_mapping(section,
                                current_pos=None,
                                section_count: int = 0):
    if current_pos is None:
        current_pos = []
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


def candidates_filter(filter_category: str, text_position_mapping):
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
