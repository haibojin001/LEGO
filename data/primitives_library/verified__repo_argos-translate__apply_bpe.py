import argparse
import codecs
import inspect
import io
import os
import re
import sys
import warnings


class BPE(object):
    def __init__(self, codes, merges=-1, separator="@@", vocab=None, glossaries=None):
        codes.seek(0)
        offset = 1

        first_line = codes.readline()
        if first_line.startswith("#version:"):
            version_text = re.sub(r"(\.0+)*$", "", first_line.split()[-1])
            self.version = tuple(int(part) for part in version_text.split("."))
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        pairs = [
            tuple(line.strip("\r\n ").split(" "))
            for number, line in enumerate(codes)
            if number < merges or merges == -1
        ]

        for number, pair in enumerate(pairs):
            if len(pair) != 2:
                sys.stderr.write(
                    "Error: invalid line {0} in BPE codes file: {1}\n".format(
                        number + offset, " ".join(pair)
                    )
                )
                sys.stderr.write(
                    "The line should exist of exactly two subword units, separated by whitespace\n"
                )
                sys.exit(1)

        self.bpe_codes = dict(
            (pair, rank)
            for rank, pair in reversed(list(enumerate(pairs)))
        )
        self.bpe_codes_reverse = dict(
            (pair[0] + pair[1], pair)
            for pair in self.bpe_codes
        )

        self.separator = separator
        self.vocab = vocab
        self.glossaries = glossaries if glossaries else []
        self.cache = {}

    def process_line(self, line):
        output = ""

        leading = len(line) - len(line.lstrip("\r\n "))
        if leading:
            output += line[:leading]

        output += self.segment(line)

        trailing = len(line) - len(line.rstrip("\r\n "))
        if trailing and trailing != len(line):
            output += line[-trailing:]

        return output

    def segment(self, sentence):
        return " ".join(self.segment_tokens(sentence.strip("\r\n ").split(" ")))

    def segment_tokens(self, tokens):
        output = []

        for word in tokens:
            if not word:
                continue

            encoded = [
                item
                for part in self._isolate_glossaries(word)
                for item in encode(
                    part,
                    self.bpe_codes,
                    self.bpe_codes_reverse,
                    self.vocab,
                    self.separator,
                    self.version,
                    self.cache,
                    self.glossaries,
                )
            ]

            for item in encoded[:-1]:
                output.append(item + self.separator)
            output.append(encoded[-1])

        return output

    def _isolate_glossaries(self, word):
        segments = [word]

        for glossary in self.glossaries:
            segments = [
                subsegment
                for segment in segments
                for subsegment in isolate_glossary(segment, glossary)
            ]

        return segments


def create_parser(subparsers=None):
    if subparsers:
        parser = subparsers.add_parser(
            "apply-bpe",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation",
        )
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation",
        )

    parser.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("r"),
        default=sys.stdin,
        metavar="PATH",
        help="Input file (default: standard input).",
    )
    parser.add_argument(
        "--codes",
        "-c",
        type=argparse.FileType("r"),
        metavar="PATH",
        required=True,
        help="File with BPE codes (created by learn_bpe.py).",
    )
    parser.add_argument(
        "--merges",
        "-m",
        type=int,
        default=-1,
        metavar="INT",
        help="Use this many BPE operations (<= number of learned symbols)"
        + "default: Apply all the learned merge operations",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w"),
        default=sys.stdout,
        metavar="PATH",
        help="Output file (default: standard output)",
    )
    parser.add_argument(
        "--separator",
        "-s",
        type=str,
        default="@@",
        metavar="STR",
        help="Separator between non-final subword units (default: '%(default)s'))",
    )
    parser.add_argument(
        "--vocabulary",
        type=argparse.FileType("r"),
        default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.",
    )
    parser.add_argument(
        "--vocabulary-threshold",
        type=int,
        default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV",
    )
    parser.add_argument(
        "--glossaries",
        type=str,
        nargs="+",
        default=None,
        metavar="STR",
        help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "
        + "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "
        + "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes.",
    )

    return parser


def get_pairs(word):
    pairs = set()
    previous = word[0]

    for symbol in word[1:]:
        pairs.add((previous, symbol))
        previous = symbol

    return pairs


def encode(
    orig,
    bpe_codes,
    bpe_codes_reverse,
    vocab,
    separator,
    version,
    cache,
    glossaries=None,
):
    if orig in cache:
        return cache[orig]

    if glossaries and re.match("^({})$".format("|".join(glossaries)), orig):
        cache[orig] = (orig,)
        return (orig,)

    if version == (0, 1):
        word = tuple(orig) + ("</w>",)
    elif version == (0, 2):
        word = tuple(orig[:-1]) + (orig[-1] + "</w>",)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float("inf")))

        if bigram not in bpe_codes:
            break

        first, second = bigram
        merged = []
        index = 0

        while index < len(word):
            try:
                next_index = word.index(first, index)
            except ValueError:
                merged.extend(word[index:])
                break

            merged.extend(word[index:next_index])
            index = next_index

            if (
                word[index] == first
                and index < len(word) - 1
                and word[index + 1] == second
            ):
                merged.append(first + second)
                index += 2
            else:
                merged.append(word[index])
                index += 1

        word = tuple(merged)

        if len(word) == 1:
            break

        pairs = get_pairs(word)

    if vocab:
        word = check_vocab_and_split(
            word, bpe_codes_reverse, vocab, separator, version
        )

    if version == (0, 1):
        word = word[:-1]
    elif version == (0, 2):
        word = word[:-1] + (word[-1].replace("</w>", ""),)

    cache[orig] = word
    return word


def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    try:
        if final:
            left, right = bpe_codes[segment + "</w>"]
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except KeyError:
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (right if final else right + separator) in vocab:
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item


def check_vocab_and_split(orig, bpe_codes, vocab, separator, version):
    output = []

    for segment in orig:
        if segment[-4:] == "</w>":
            segment = segment[:-4]
            final = True
        else:
            final = False

        if segment + ("" if final else separator) in vocab:
            output.append(segment)
        else:
            for item in recursive_split(
                segment, bpe_codes, vocab, separator, final=final
            ):
                output.append(item)

    return tuple(output)


def read_vocabulary(vocab_file, threshold):
    vocabulary = set()

    for line in vocab_file:
        word, frequency = line.strip("\r\n ").split(" ")
        if int(frequency) >= threshold:
            vocabulary.add(word)

    return vocabulary


def isolate_glossary(word, glossary):
    if re.match("^({})$".format(glossary), word):
        return [word]

    return [
        segment
        for segment in re.split("({})".format(glossary), word)
        if segment
    ]


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.vocabulary:
        if args.vocabulary_threshold is None:
            warnings.warn(
                "Vocabulary threshold not specified. Using default threshold of 0."
            )
            args.vocabulary_threshold = 0
        vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    bpe = BPE(
        args.codes,
        args.merges,
        args.separator,
        vocabulary,
        args.glossaries,
    )

    for line in args.input:
        args.output.write(bpe.process_line(line))


if __name__ == "__main__":
    main()