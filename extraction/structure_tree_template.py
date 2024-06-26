import dataclasses
import logging
import pickle
import random
import re
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Union, Iterable

from bs4 import BeautifulSoup, Tag, NavigableString, Comment
from tqdm import tqdm

import evaluation.text_preprocessing as tp
from classification.preprocessing import Category, Website


@dataclasses.dataclass(eq=True, frozen=True)
class HTMLPosition:
    """
    Representation of a HTML tag
    """
    position: int
    tag: str

    @staticmethod
    def convert_from_old_position(old_pos: List[Tuple[int, str]]) -> List['HTMLPosition']:
        """
        Convert a List of Tuple[int, str] (old datastructure) to a List of HTMLPosition (new datastructure).

        :param old_pos: old datastructure
        :return: converted new datastructure
        """
        return [HTMLPosition(pos, tag) for pos, tag in old_pos]

    @staticmethod
    def convert_to_old_position(hpos_list: List['HTMLPosition']) -> List[Tuple[int, str]]:
        """
        Convert a List of HTMLPosition (new datastructure) to a List of Tuple[int, str] (old datastructure).

        :param hpos_list: new datastructure
        :return: converted old datastructure
        """
        return [(hpos.position, hpos.tag) for hpos in hpos_list]


@dataclasses.dataclass
class TreeNode:
    """
    Datastructure to represent a template tree.
    """

    parent: 'TreeNode' = dataclasses.field(repr=False)
    root: bool
    hpos: HTMLPosition
    children: Dict[HTMLPosition, 'TreeNode'] = dataclasses.field(init=False, default_factory=lambda: dict())
    bit_mask: List[int] = dataclasses.field(init=False, default=None)
    weight: float = dataclasses.field(init=False, default=1)
    wed_ids: Set[str] = dataclasses.field(init=False, default_factory=lambda: set())

    log = logging.getLogger('TreeNode')

    def get_root(self) -> 'TreeNode':
        """
        Get the root of the tree.
        :return: tree root
        """
        if self.root:
            return self
        return self.parent.get_root()

    def get_leafs(self) -> List['TreeNode']:
        """
        Get all leafs of this TreeNode. Calls recursive all child nodes.

        :return: List of leaf nodes.
        """
        tmp_list = []
        if self.bit_mask is not None:
            tmp_list.append(self)
        for child in self.children.values():
            tmp_list += child.get_leafs()
        return tmp_list

    def create_child(self, pos: int, tag: str) -> 'TreeNode':
        """
        Create a child node from this node.

        :param pos: postion of child in parent node.
        :param tag: tag of child node.
        :return: Created child node.
        """
        hpos = HTMLPosition(pos, tag)
        child = TreeNode(self, False, hpos)
        self.children[hpos] = child
        self.weight += 1
        return child

    def is_leave(self) -> bool:
        """
        Is this node a leaf?

        :return: if this node is a leaf.
        """
        return len(self.children) == 0 and self.bit_mask is not None

    def traverse(self, position: List[HTMLPosition], best_bet: bool = False) -> 'TreeNode':
        """
        Traverse the tree using the given HTMLPosition's.
        If best_bet is True, return the best approximation,
        else raise a ValueError because the postion couldn't be fully traversed.

        :param position: List of HTMLPosition to traverse
        :param best_bet: Should the best approximation be returned
        :return: traversed TreeNode
        :raises ValueError: if best_bet is False and position not complete in Tree.
        """
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
        return node

    def get_position(self) -> List[HTMLPosition]:
        """
        Create a List with HTMLPosition's representing the position of this node in the tree.

        :return: Position in Tree
        """
        if self.root:
            return [self.hpos]
        parent_pos = self.parent.get_position()
        parent_pos.append(self.hpos)
        return parent_pos

    def add_leave(self, h_position: List[HTMLPosition], bit_mask: List[int], web_id: str) -> None:
        """
        Add a leaf to the tree, if node not exits yet create it.

        :param h_position: position of the leaf.
        :param bit_mask: bit_mask of the leaf.
        :param web_id: web_id of th leaf.
        :return: None
        """
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
    """
    Represents the structured template for a category.
    """

    log = logging.getLogger('StrucTemp')

    train_data: Dict[str, TreeNode] = {}

    def get_attr_root(self, attribute: str) -> TreeNode:
        """
        Get the root tree node for the attribute.

        :param attribute: which attribute tree
        :return: root of the tree.
        """
        if attribute in self.train_data.keys():
            return self.train_data[attribute]

        raise ValueError(f"{attribute} doesn't have train data.")

    def get_attr_names(self) -> List[str]:
        """
        Return a list with all learned attributes.

        :return: List of learned attributes.
        """
        return list(self.train_data.keys())

    def __repr__(self) -> str:
        result = 'StructuredTreeTemplate(train_data={'
        for k, v in self.train_data.items():
            result += f'{k.__repr__()}: {v.__repr__()}'
        result += '})'
        return result

    @classmethod
    def load(cls, path: Path) -> 'StructuredTreeTemplate':
        """
        Load a StructuredTreeTemplate from disk.

        :param path: path/to/strucTreeTemp_train_data.pkl
        :return: None
        """
        struc_temp = cls()
        with path.joinpath('strucTreeTemp_train_data.pkl').open(mode='rb') as pkl:
            struc_temp.train_data = pickle.load(pkl, fix_imports=False)
        struc_temp.log.debug(f'Loaded from disk {path}')
        return struc_temp

    def save(self, path: Path) -> None:
        """
        Save a StructuredTreeTemplate to disk.

        :param path: path/to/save
        :return: None
        """
        with path.joinpath('strucTreeTemp_train_data.pkl').open(mode='wb') as pkl:
            pickle.dump(self.train_data, pkl, fix_imports=False)
        self.log.debug(f'Saved to disk under {path}')

    def extract(self, web_ids: str, category: Category, k: int = 3,
                with_score: bool = False, only_perfect_match: bool = False
                ) -> List[Dict[str, Union[List[str], List[Tuple[int, str]]]]]:
        """
        Extract attributes from given websites, using the template trees.

        :param web_ids: List of websites to extract from
        :param category: Category to extract
        :param k: number of extracted values per attribute
        :param with_score: Return the calculated score
        :param only_perfect_match: only return candidates with perfect score
        :return: List of dictionaries with attribute as key and extracted values as value.
        """
        result = []
        self.log.debug(f'Extract for web_ids')
        for web_id in tqdm(web_ids, desc='StrucTree Extraction'):
            website = Website.load(web_id)
            # self.log.debug(f'Extract for web_id {web_id}')
            with Path(website.file_path).open(encoding='utf-8') as htm_file:
                soup = BeautifulSoup(htm_file, features="html.parser")

            body = soup.find('body')
            text_position_mapping = build_mapping(body)

            candidates = dict()
            for key in category.get_attribute_names():
                filter_category = Category.get_attr_type(key)
                candidates[key] = candidates_filter(filter_category, text_position_mapping)

            result.append(sort_n_score_candidates(candidates, self, k=k, with_score=with_score,
                                                  only_perfect_match=only_perfect_match))
        return result

    def train(self, web_ids: List[str]) -> None:
        """
        Learn the template trees from given websites.

        :param web_ids: Websites to learn from.
        :return: None
        """
        self.log.debug(f'Start learning from web_ids')
        for web_id in tqdm(web_ids, desc='StrucTree Training'):
            # self.log.debug(f'Start learning from web_id {web_id}')
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

                attr_data = self.train_data.setdefault(best_position['attribute'],
                                                       TreeNode(None, True, HTMLPosition(0, 'root')))
                attr_data.add_leave(best_position['position'], best_position['text_info'], web_id)


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
    """
    Return the longest element of the given Iterable

    :param iter_obj: Iterable
    :return: longest element
    """

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
    """
    Decides if a given HTML section should be filtered. Returns True if section iss a navbar or a script.

    :param section: HTML section
    :return: Should this section be filtered.
    """
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
    """
    Class to represent a text position mapping.
    """
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
    """
    Generate the complete text position mapping.

    :param section: The HTML section to start from.
    :param current_pos: current HTML position.
    :param section_count: position of current HTML section in parent.
    :return: List of all TextPosMapping
    """
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
    """
    Filter extraction candidates, to reduce candidate size.

    :param filter_category: Filter category
    :param text_position_mapping: candidates
    :return: filtered candidates
    """
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
                            with_score: bool = False, only_perfect_match: bool = False
                            ) -> Dict[str, Union[List[str], List[Tuple[int, str]]]]:
    """
    Score all candidates against the template leaf nodes and return the extracted text for the top k candidates.

    :param candidates: candidates to score
    :param template: StructuredTreeTemplate
    :param k: number of extracted values per attribute
    :param with_score: Return the calculated score
    :param only_perfect_match: only return candidates with perfect score
    :return: List of dictionaries with attribute as key and extracted values as value.
    """
    scored_candidates = dict()
    for attr in candidates.keys():
        attr_can = []
        for c_text, c_pos in candidates[attr]:
            best_leave = (-100, None)

            if only_perfect_match:
                try:
                    leave = template.get_attr_root(attr).traverse(c_pos)
                    best_leave = (100, leave)
                except ValueError:
                    pass
            else:
                for leave in template.get_attr_root(attr).get_leafs():
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
    """
    Apply the bit_mask's to the top candidates.

    :param candidates: Selected candidates.
    :param with_score: Return the calculated score
    :return: List of dictionaries with attribute as key and extracted values as value.
    """
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

            if leave is None:
                if with_score:
                    result[attr].append((web_score, ''))
                else:
                    result[attr].append('')
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
    """
    Score two HTMLPosition against each other.

    :param pos_template: HTMLPosition in template tree
    :param pos_candid: HTMLPosition in candidates HTML
    :return: Calculated score
    """
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
    """
    Return if the list are sub list of each other, but don't change the order of entries.

    :param lst1: list 1
    :param lst2: list 2
    :return: result
    """
    if len(lst2) < len(lst1):
        lst1, lst2 = lst2, lst1
    if lst1[0] in lst2:
        indices_apperance = [i for i, x in enumerate(lst2) if x == lst1[0]]
        for i_p in indices_apperance:
            if lst2[i_p:i_p+len(lst1)] == lst1:
                return True
    return False
