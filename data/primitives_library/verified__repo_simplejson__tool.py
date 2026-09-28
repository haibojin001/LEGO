"""Validate and format JSON data from standard input or files."""

import sys
import simplejson as json


def main():
    argc = len(sys.argv)

    if argc == 1:
        source = sys.stdin
        target = sys.stdout
    elif argc == 2:
        source = open(sys.argv[1], "r")
        target = sys.stdout
    elif argc == 3:
        source = open(sys.argv[1], "r")
        target = open(sys.argv[2], "w")
    else:
        raise SystemExit("{} [infile [outfile]]".format(sys.argv[0]))

    with source:
        try:
            value = json.load(
                source,
                object_pairs_hook=json.OrderedDict,
                use_decimal=True,
            )
        except ValueError as error:
            raise SystemExit(error)

    with target:
        json.dump(
            value,
            target,
            sort_keys=True,
            indent="    ",
            use_decimal=True,
        )
        target.write("\n")


if __name__ == "__main__":
    main()