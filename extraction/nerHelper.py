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
    temp_line = []
    for line in [t.strip() for t in visible_texts]:
        if line:
            line = ' '.join(line.split())
            if len(line) < 20:
                temp_line.append(line)
                if len(temp_line) > 20:
                    line_list.append(' '.join(temp_line))
                    temp_line = []
            else:
                if temp_line:
                    line_list.append(' '.join(temp_line))
                    temp_line = []
                line_list.append(line)
    print(line_list)
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


def html_text_to_spacy(html_text, attributes):
    text = ""
    entities = []

    for line in html_text:
        text += str(line) + " "

    text_list = text.split(" ")
    text_len_list = [len(i) for i in text_list]
    for attr, value in attributes.items():
        if value and isinstance(value, list):
            value = value[0]
        elif value:
            pass
        else:
            print("nothing there", attr, value)
            continue
        value = value.strip()
        value = value.replace("  ", " ")
        value = value.replace("\t", " ")
        value_list = value.split(" ")
        print(attr, ":", value)
        indices = [i for i, x in enumerate(text_list) if x == value_list[0]]
        for i in indices:
            if text_list[i:i+len(value_list)] == value_list:
                start_index = sum(text_len_list[:i]) + len(text_len_list[:i])
                pos = i + len(value_list)
                end_index = sum(text_len_list[:pos]) + len(text_len_list[:pos]) - 1
                entities.append((start_index, end_index, attr))
        print({'text': text, 'entities': entities})
    return {'text': text, 'entities': entities}