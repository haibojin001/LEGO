import argparse
import os

from libretranslate.api_keys import Database
from libretranslate.default_values import DEFAULT_ARGUMENTS as DEFARGS


def manage():
    parser = argparse.ArgumentParser(description="LibreTranslate Manage Tools")
    commands = parser.add_subparsers(
        help="",
        dest="command",
        required=True,
        title="Command List",
    )

    keys = commands.add_parser("keys", help="Manage API keys database")
    keys.add_argument(
        "--api-keys-db-path",
        default=DEFARGS["API_KEYS_DB_PATH"],
        type=str,
        help="Use a specific path inside the container for the local database",
    )
    actions = keys.add_subparsers(
        help="",
        dest="sub_command",
        title="Command List",
    )

    add = actions.add_parser("add", help="Add API keys to database")
    add.add_argument("req_limit", type=int, help="Request Limits (per minute)")
    add.add_argument(
        "--key",
        type=str,
        default="auto",
        required=False,
        help="API Key",
    )
    add.add_argument(
        "--char-limit",
        type=int,
        default=0,
        required=False,
        help="Character limit",
    )

    remove = actions.add_parser("remove", help="Remove API keys from database")
    remove.add_argument("key", type=str, help="API Key")

    arguments = parser.parse_args()

    if arguments.command == "keys":
        if not os.path.exists(arguments.api_keys_db_path):
            print("No such database: %s" % arguments.api_keys_db_path)
            exit(1)

        database = Database(arguments.api_keys_db_path)

        if arguments.sub_command is None:
            api_keys = database.all()
            if not api_keys:
                print("There are no API keys")
            else:
                for api_key in api_keys:
                    print("{}: {}".format(*api_key))
        elif arguments.sub_command == "add":
            print(
                database.add(
                    arguments.req_limit,
                    arguments.key,
                    arguments.char_limit,
                )[0]
            )
        elif arguments.sub_command == "remove":
            print(database.remove(arguments.key))
    else:
        parser.print_help()
        exit(1)