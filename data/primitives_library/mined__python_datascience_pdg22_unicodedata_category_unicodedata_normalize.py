# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg22::unicodedata.category+unicodedata.normalize
# name: unicodedata_primitive
# summary: Uses unicodedata.category, unicodedata.normalize across 2 repos
# anchor_symbols: ['unicodedata.category', 'unicodedata.normalize']
# observed in 2 repos: ['explosion__spaCy', 'piskvorky__gensim']...

# --- from explosion__spaCy::spacy/lang/yo/lex_attrs.py::strip_accents_text ---
def strip_accents_text(text):
    """
    Converts the string to NFD, separates & returns only the base characters
    :param text:
    :return: input string without diacritic adornments on base characters
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

# --- from piskvorky__gensim::gensim/utils.py::deaccent ---
def deaccent(text):
    """Remove letter accents from the given string.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Unicode string without accents.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import deaccent
        >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
        u'Sef chomutovskych komunistu dostal postou bily prasek'

    """
    if not isinstance(text, str):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)
