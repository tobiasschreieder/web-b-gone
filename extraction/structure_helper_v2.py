import dataclasses
import logging
import random
from copy import copy
from typing import Any, Iterable, List, Dict, Tuple, Set
from bs4 import Tag, Comment, NavigableString, PageElement

from classification.preprocessing import Website
import evaluation.text_preprocessing as tp


# TODO: refactor names and write doc

@dataclasses.dataclass
class AttributeData:
    name: str
    positions: Dict[str, str] = dataclasses.field(init=False, default_factory=lambda: dict())
    text_infos: Dict[str, List[int]] = dataclasses.field(init=False, default_factory=lambda: dict())

    def add_website(self, web_id: str, position: str, text_info: List[int]) -> None:
        self.positions[web_id] = position
        self.text_infos[web_id] = text_info


class StructuredTemplate:
    log = logging.getLogger('StrucTemp')

    # TODO use tree rep

    train_data: Dict[str, AttributeData] = dict()
    web_ids: Set[str] = set()

    def add_attribute(self, attribute: str, position, text_info: List[int], web_id: str) -> None:
        attr_data = self.train_data.setdefault(attribute, AttributeData(attribute))
        attr_data.add_website(web_id, position, text_info)
        self.web_ids.add(web_id)

    def get_attr_data(self, attribute: str) -> AttributeData:
        if attribute in self.train_data.keys():
            return self.train_data[attribute]

        raise ValueError(f"{attribute} does't have train data.")

    def get_from_web_id_and_attribute(self, web_id: str, attribute: str) -> Dict[str, Any]:
        attr_data = self.get_attr_data(attribute)

        if web_id not in attr_data.positions.keys():
            raise ValueError(f"Web_id {web_id} wasn't used for training.")

        return {'attribute': attribute,
                'position': attr_data.positions[web_id],
                'text_info': attr_data.text_infos[web_id],
                'web_id': web_id}

    def get_attr_names(self) -> List[str]:
        return list(self.train_data.keys())

    def __repr__(self) -> str:
        result = 'StructuredTemplate(train_data={'
        for k, v in self.train_data.items():
            result += f'{k.__repr__()}: {v.__repr__()}'
        result += '})'
        return result


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


def get_longest_element(iter_obj: Iterable[Any]):
    e = None
    length = 0
    for element in iter_obj:
        if len(element) > length:
            e = iter_obj
            length = len(element)
    return e


@dataclasses.dataclass
class TextPosMapping:
    text: str
    position: List[Tuple[int, str]]

    def __getitem__(self, item):
        if item == 'text':
            return self.text
        elif item == 'position':
            return self.position
        else:
            raise KeyError(f'{item} is not in TextPosMapping.')


def build_mapping(section: Tag,
                  current_pos=None,
                  section_count: int = 0) -> List[TextPosMapping]:
    if current_pos is None:
        current_pos = []

    html_position = copy(current_pos)
    html_position.append((section_count, section.name))

    result_list = []

    # tag_text = section.findAll(text=True, recursive=False)
    # tag_text = tp.preprocess_text_html(tag_text)
    #
    # if not tag_text == '':
    #     result_list.append(TextPosMapping(tag_text, html_position))

    i = 0
    for child in section.children:
        if isinstance(child, NavigableString):
            child: NavigableString
            child_pos = copy(html_position)
            child_pos.append((i, 'NavStr'))
            text = tp.preprocess_text_html(child.text)

            if text == '':
                continue

            result_list.append(TextPosMapping(text, child_pos))
        elif isinstance(child, Comment):
            pass
        else:
            if child.name in ['script']:
                continue
            result_list += build_mapping(child, html_position, i)
            i += 1

    return result_list


def build_text_position_mapping(section: Tag,
                                current_pos=None,
                                section_count: int = 0) -> List[Any]:
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
        text_pos = copy(html_position)
        text_pos.append([0, tag])
        temp_list += [{"text": text, "position": text_pos}]

    html_position.append([section_count, tag])
    section_count = 0
    for child_section in children:
        section_count += 1
        temp_list += build_text_position_mapping(child_section, html_position, section_count)

    return temp_list


def candidates_filter(filter_category: str, text_position_mapping) -> List[Tuple[str, Any]]:
    if filter_category == 'name':
        return [(mapping['text'], mapping['position'])
                for mapping in text_position_mapping if candidate_filter_name(mapping['text'])]
    elif filter_category == 'number':
        return [(mapping['text'], mapping['position'])
                for mapping in text_position_mapping if candidate_filter_number(mapping['text'])]
    else:
        return [(mapping['text'], mapping['position']) for mapping in text_position_mapping]


def candidate_filter_name(text: str) -> bool:
    if text == "":
        return False
    if len(text) < 5:
        return False
    if text.count(" ") > 20:
        return False
    # if any([char in text for char in ['%', '$', '!', '§', '&']]):
    #     return False
    # if any(char.isdigit() for char in text):
    #     return False
    return True


def candidate_filter_number(text: str) -> bool:
    if text == "":
        return False
    if text.count(" ") > 20:
        return False
    if not any(char.isdigit() for char in text):
        return False
    return True


def sort_n_score_candidates(candidates: Dict[str, List[Tuple[str, Any]]],
                            template: StructuredTemplate,
                            k: int = 3) -> Dict[str, List[str]]:
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

        use avg over attr to compensate for missing ones

    --> choose best Sum and use this template and this candidate
    """
    # result_can = [(0, {'name': 'Hans Franz', 'height': '900'}, 'W855695663')]
    result_can = []
    for web_id in template.web_ids:
        web_dict = dict()
        web_score = 0
        used_attr = 0
        for attr in candidates.keys():
            try:
                web_pos = template.get_from_web_id_and_attribute(web_id, attr)['position']
            except ValueError:
                continue

            used_attr += 1

            best_can = (0, '')
            best_pos = None
            for candid in candidates[attr]:
                candid_pos = candid[1]
                candid_text = candid[0]

                can_score = position_scoring(candid_pos, web_pos)
                if can_score > best_can[0]:
                    best_can = (can_score, candid_text)
                    best_pos = candid_pos

            web_dict[attr] = best_can[1]
            web_score += best_can[0]

        if used_attr == 0:
            continue

        web_score = web_score / used_attr

        result_can.append((web_score, web_dict, web_id))
        result_can = sorted(result_can, key=lambda x: x[0], reverse=True)[:k]

    return apply_bit_mask(result_can, template)


def apply_bit_mask(candidates: List[Tuple[int, Dict[str, str], str]],
                   template: StructuredTemplate) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = dict()
    for _, web_dict, web_id in candidates:
        for attr in web_dict.keys():
            bit_mask: List[int] = template.get_from_web_id_and_attribute(web_id, attr)['text_info']
            text = web_dict[attr]

            mask_sum = sum(bit_mask)
            if mask_sum == 0:
                text = ''
            elif mask_sum != len(bit_mask):
                first_word = bit_mask.index(1)
                last_word = len(bit_mask) - bit_mask.index(0, first_word)
                correct_text = text.split(' ')[first_word:-last_word]
                text = ' '.join(correct_text)

            result.setdefault(attr, [])
            result[attr].append(text)

    return result


def position_scoring(position_c, position_t):
    n = 5
    grams = [position_c[i:i + n] for i in range(len(position_c) - n + 1)]
    # to speed-up the process
    if len(grams) > 100:
        grams = random.sample(grams, 50)

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
