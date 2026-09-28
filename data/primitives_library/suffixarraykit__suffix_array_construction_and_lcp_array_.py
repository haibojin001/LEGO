def build_suffix_array(text):
    suffixes = [(text[i:], i) for i in range(len(text))]
    suffixes.sort() #Ordenacion lexografica
    suffix_array = [s[1] for s in suffixes]
    return suffix_array