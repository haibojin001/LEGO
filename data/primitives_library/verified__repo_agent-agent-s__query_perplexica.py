import os

import requests
import toml


def query_to_perplexica(query):
    url = os.getenv("PERPLEXICA_URL")
    if not url:
        raise ValueError(
            "PERPLEXICA_URL environment variable not set. It may take the form: "
            "'http://localhost:{port}/api/search'. The port number is set in "
            "the config.toml in the Perplexica directory."
        )

    payload = {
        "focusMode": "webSearch",
        "query": query,
        "history": [["human", query]],
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["message"]
    if response.status_code == 400:
        raise ValueError(
            "The request is malformed or missing required fields, such as FocusModel or query"
        )
    raise ValueError("Internal Server Error")


if __name__ == "__main__":
    query = "What is Agent S?"
    response = query_to_perplexica(query)
    print(response)