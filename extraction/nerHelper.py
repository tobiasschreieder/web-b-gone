from copy import copy
from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Comment

from classification.preprocessing import Website


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_html_text(web_id):
    print(web_id)
    website = Website.load(web_id)
    with Path(website.file_path).open(encoding='utf-8') as htm_file:
        soup = BeautifulSoup(htm_file, features="html.parser")
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    line_list = []
    for line in [t.strip() for t in visible_texts]:
        if line:
            line = ' '.join(line.split())
            line_list.append(line)
    return line_list

def html_text_to_BIO(text, attributes):
    bio_format = []
    for line in text:
        labels = ['O'] * len(copy(line).split(' '))
        line_list = line.split(" ")
        for attr, value in attributes.items():
            value_list = value.split(" ")
            if (value in line) and (value_list[0] in line_list):
                start_index = line_list.index(value_list[0])
                for i in range(len(value_list)):
                    labels[start_index + i] = attr
        bio_line = []
        for i in range(len(line_list)):
            bio_line.append((line_list[i], labels[i]))
        bio_format.append(bio_line)
    return bio_format

def bio_text_to_spacy(text, new_attributes):
    pass