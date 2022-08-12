def exact_match(text_1: str, text_2: str) -> float:
    """
    Calculate exact match similarity between two given Strings (only 0.0 or 1.0)
    :param text_1: String 1
    :param text_2: String 2
    :return: Similarity between String 1 and String 2 as float
    """
    # At least one of the strings is empty
    if (len(text_1) == 0) or (len(text_2) == 0):
        return 0.0

    # Different strings
    elif text_1 != text_2:
        return 0.0

    # Identical strings
    else:
        return 1.0


def bag_similarity(text_1: str, text_2: str) -> float:
    """
    Calculate bag distance similarity between two given Strings (0.0 - 1.0)
    :param text_1: String 1
    :param text_2: String 2
    :return: Similarity between String 1 and String 2 as float
    """
    # At least one of the strings is empty -> return 0.0
    if (len(text_1) == 0) or (len(text_2) == 0):
        return 0.0

    # Exact match between both texts -> return 1.0
    elif text_1 == text_2:
        return 1.0

    len_text_1 = len(text_1)
    len_text_2 = len(text_2)

    list_1 = list(text_1)
    list_2 = list(text_2)

    for c in text_1:
        if c in list_2:
            list_2.remove(c)

    for c in text_2:
        if c in list_1:
            list_1.remove(c)

    # Calculate bag similarity
    b = max(len(list_1), len(list_2))
    bag_sim = 1.0 - float(b) / float(max(len_text_1, len_text_2))

    assert 0.0 <= bag_sim <= 1.0

    return bag_sim
