import dataclasses
import logging
import pickle
import random
import re
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Union, Iterable

from bs4 import BeautifulSoup, Tag, NavigableString, Comment

from classification.preprocessing import Category, Website
import evaluation.text_preprocessing as tp


@dataclasses.dataclass(eq=True, frozen=True)
class HTMLPosition:
    position: int
    tag: str

    @staticmethod
    def convert_from_old_position(old_pos: List[Tuple[int, str]]) -> List['HTMLPosition']:
        return [HTMLPosition(pos, tag) for pos, tag in old_pos]

    @staticmethod
    def convert_to_old_position(hpos_list: List['HTMLPosition']) -> List[Tuple[int, str]]:
        return [(hpos.position, hpos.tag) for hpos in hpos_list]


@dataclasses.dataclass
class TreeNode:
    parent: 'TreeNode' = dataclasses.field(repr=False)
    root: bool
    hpos: HTMLPosition
    children: Dict[HTMLPosition, 'TreeNode'] = dataclasses.field(init=False, default_factory=lambda: dict())
    bit_mask: List[int] = dataclasses.field(init=False, default=None)
    weight: float = dataclasses.field(init=False, default=1)
    wed_ids: Set[str] = dataclasses.field(init=False, default_factory=lambda: set())

    log = logging.getLogger('TreeNode')

    def get_root(self) -> 'TreeNode':
        if self.root:
            return self
        return self.parent.get_root()

    def get_leaves(self) -> List['TreeNode']:
        tmp_list = []
        if self.bit_mask is not None:
            tmp_list.append(self)
        for child in self.children.values():
            tmp_list += child.get_leaves()
        return tmp_list

    def create_child(self, pos: int, tag: str) -> 'TreeNode':
        hpos = HTMLPosition(pos, tag)
        child = TreeNode(self, False, hpos)
        self.children[hpos] = child
        self.weight += 1
        return child

    def is_leave(self) -> bool:
        return len(self.children) == 0

    def traverse(self, position: List[HTMLPosition], best_bet: bool = False) -> 'TreeNode':
        node = self
        for hpos in position:
            if hpos in node.children.keys():
                node = node.children[hpos]
            elif best_bet:
                return node
            else:
                raise ValueError(f'Nodes doesnt exists. Children pos: {node.children.keys()},\n '
                                 f'Position searched: {position}, \n'
                                 f'Position node:     {node.get_position()}')

    def get_position(self) -> List[HTMLPosition]:
        if self.root:
            return [self.hpos]
        parent_pos = self.parent.get_position()
        parent_pos.append(self.hpos)
        return parent_pos

    def add_leave(self, h_position: List[HTMLPosition], bit_mask: List[int], web_id: str) -> None:
        if not self.root:
            self.log.warning('Tried to add leave to none root node, traverse up')
            return self.parent.add_leave(h_position, bit_mask, web_id)

        node = self
        for hpos in h_position:
            if hpos in node.children.keys():
                node = node.children[hpos]
                node.weight += 1
            else:
                node = node.create_child(hpos.position, hpos.tag)

        node.bit_mask = bit_mask
        node.wed_ids.add(web_id)


class StructuredTreeTemplate:
    log = logging.getLogger('StrucTemp')

    train_data: Dict[str, TreeNode] = {}

    def add_attribute(self, attribute: str, position: List[HTMLPosition],
                      text_info: List[int], web_id: str) -> None:
        attr_data = self.train_data.setdefault(attribute, TreeNode(None, True, HTMLPosition(0, 'root')))
        attr_data.add_leave(position, text_info, web_id)

    def get_attr_root(self, attribute: str) -> TreeNode:
        if attribute in self.train_data.keys():
            return self.train_data[attribute]

        raise ValueError(f"{attribute} doesn't have train data.")

    def get_attr_names(self) -> List[str]:
        return list(self.train_data.keys())

    def __repr__(self) -> str:
        result = 'StructuredTreeTemplate(train_data={'
        for k, v in self.train_data.items():
            result += f'{k.__repr__()}: {v.__repr__()}'
        result += '})'
        return result

    @classmethod
    def load(cls, path: Path) -> 'StructuredTreeTemplate':
        struc_temp = cls()
        with path.joinpath('strucTreeTemp_train_data.pkl').open(mode='rb') as pkl:
            struc_temp.train_data = pickle.load(pkl, fix_imports=False)
        struc_temp.log.debug(f'Loaded from disk {path}')
        return struc_temp

    def save(self, path: Path) -> None:
        with path.joinpath('strucTreeTemp_train_data.pkl').open(mode='wb') as pkl:
            pickle.dump(self.train_data, pkl, fix_imports=False)
        self.log.debug(f'Saved to disk under {path}')

    def extract(self, web_ids: str, category: Category, k: int = 3,
                with_score: bool = False) -> List[Dict[str, Union[List[str], List[Tuple[int, str]]]]]:
        result = []
        for web_id in web_ids:
            website = Website.load(web_id)
            self.log.debug(f'Extract for web_id {web_id}')
            with Path(website.file_path).open(encoding='utf-8') as htm_file:
                soup = BeautifulSoup(htm_file, features="html.parser")

            body = soup.find('body')
            text_position_mapping = build_mapping(body)

            candidates = dict()
            for key in category.get_attribute_names():
                filter_category = Category.get_attr_type(key)
                candidates[key] = candidates_filter(filter_category, text_position_mapping)

            result.append(sort_n_score_candidates(candidates, self, k=k, with_score=with_score))
        return result

    def train(self, web_ids: List[str]) -> None:
        for web_id in web_ids:
            self.log.debug(f'Start learning from web_id {web_id}')
            website = Website.load(web_id)
            with Path(website.file_path).open(encoding='utf-8') as htm_file:
                soup = BeautifulSoup(htm_file, features="html.parser")

            body = soup.find('body')
            text_position_mapping = build_mapping(body)

            attr_truth = website.truth.attributes
            for key in attr_truth:
                if key == 'category':
                    continue

                if len(attr_truth[key]) == 0:
                    continue

                prep_text = str(tp.preprocess_text_html(get_longest_element(attr_truth[key])))
                best_match = 0
                best_position = None
                for mapping in text_position_mapping:
                    match = simple_string_match(mapping['text'], prep_text)
                    if match > best_match:
                        best_match = match
                        text_correct = prep_text.split(" ")
                        text_found = str(mapping['text']).split(" ")
                        text_info = [0] * len(text_found)
                        for correct_word in text_correct:
                            if correct_word in text_found:
                                pos = text_found.index(correct_word)
                                text_info[pos] = 1

                        best_position = {'attribute': key,
                                         'position': mapping['position'],
                                         'text_info': text_info}
                if best_match == 0:
                    continue

                self.add_attribute(attribute=best_position['attribute'],
                                   position=best_position['position'],
                                   text_info=best_position['text_info'],
                                   web_id=web_id)


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
        grams = random.sample(grams, 50)

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


attr_keys = {
    'height', 'method', 'href', 'frameborder', 'marginheight', 'placeholder', 'show_faces',
    'type', 'width', 'hspace', 'vspace', 'onchange', 'selected', 'scrolling', 'id', 'ref',
    'border', 'onsubmit', 'onclick', 'rel', 'data-sportid', 'accept-charset', 'bordercolor',
    'class', 'marginwidth', 'value', 'name', 'cellpadding', 'cellspacing', 'alt', 'colspan',
    'autocomplete', 'style', 'src', 'action', 'valign', 'align', 'layout'
}


def filter_section(section: Tag) -> bool:
    tags_to_ignore = ['script']

    if section.name in tags_to_ignore:
        return True

    # boilerplate with regex search after -nav nav nav-, ggf header
    reg = re.compile('nav')

    if 'class' in section.attrs.keys():
        if any(reg.search(html_class) for html_class in section.attrs['class']):
            return True

    if 'id' in section.attrs.keys():
        if reg.search(section.attrs['id']):
            return True

    return False


@dataclasses.dataclass
class TextPosMapping:
    text: str
    position: List[HTMLPosition]

    def __getitem__(self, item):
        if item == 'text':
            return self.text
        elif item == 'position':
            return self.position
        else:
            raise KeyError(f'{item} is not in TextPosMapping.')


def build_mapping(section: Tag,
                  current_pos: List[HTMLPosition] = None,
                  section_count: int = 0) -> List[TextPosMapping]:
    if current_pos is None:
        current_pos = []

    html_position = copy(current_pos)
    html_position.append(HTMLPosition(section_count, section.name))

    result_list = []

    i = 0
    for child in section.children:
        if isinstance(child, NavigableString):
            child: NavigableString
            child_pos = copy(html_position)
            child_pos.append(HTMLPosition(i, 'NavStr'))
            text = tp.preprocess_text_html(child.text)

            if text == '':
                continue

            result_list.append(TextPosMapping(text, child_pos))
        elif isinstance(child, Comment):
            pass
        else:
            if filter_section(child):
                continue
            result_list += build_mapping(child, html_position, i)
            i += 1

    return result_list


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


def sort_n_score_candidates(candidates: Dict[str, List[Tuple[str, HTMLPosition]]],
                            template: StructuredTreeTemplate, k: int = 3,
                            with_score: bool = False) -> Dict[str, Union[List[str], List[Tuple[int, str]]]]:
    scored_candidates = dict()
    for attr in candidates.keys():
        attr_can = []
        for c_text, c_pos in candidates[attr]:
            best_leave = (-100, None)
            for leave in template.get_attr_root(attr).get_leaves():
                l_pos = leave.get_position()
                l_score = position_scoring(l_pos, c_pos) * leave.weight

                if l_score > best_leave[0]:
                    best_leave = (l_score, leave)

            attr_can.append((best_leave[0], c_text, best_leave[1]))
            attr_can = sorted(attr_can, key=lambda x: x[0], reverse=True)[:k]

        scored_candidates[attr] = attr_can
    return apply_bit_mask(scored_candidates, with_score=with_score)


def apply_bit_mask(candidates: Dict[str, List[Tuple[int, str, TreeNode]]],
                   with_score: bool = False) -> Dict[str, Union[List[str], List[Tuple[int, str]]]]:
    result: Dict[str, List[str]] = dict()
    for attr in candidates.keys():
        for web_score, text, leave in candidates[attr]:
            result.setdefault(attr, [])
            if len(text) == 0:
                if with_score:
                    result[attr].append((web_score, text))
                else:
                    result[attr].append(text)
                continue

            bit_mask: List[int] = leave.bit_mask

            mask_sum = sum(bit_mask)
            if mask_sum == 0:
                text = ''
            elif mask_sum != len(bit_mask):
                first_word = bit_mask.index(1)
                last_word = len(bit_mask) - bit_mask.index(0, first_word)
                correct_text = text.split(' ')[first_word:-last_word]
                text = ' '.join(correct_text)

            if with_score:
                result[attr].append((web_score, text))
            else:
                result[attr].append(text)

    return result


def position_scoring(pos_template: List[HTMLPosition], pos_candid: List[HTMLPosition]):
    position_c = HTMLPosition.convert_to_old_position(pos_template)
    position_t = HTMLPosition.convert_to_old_position(pos_candid)
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
