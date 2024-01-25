import argparse
import functools
import itertools
import json
import operator
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests


def auth(url, username, password) -> str:
    """Authenticate to the server."""
    r = requests.post(
        url + "/api/v4/users/login",
        json={"login_id": username, "password": password},
    )
    if r.status_code != 200:
        raise ValueError(f"Authentication failed with {r.status_code}.")

    return r.headers["Token"]


def get_posts(url, channel_id, token, after=None):
    """Get the posts."""
    headers = {"Authorization": f"Bearer {token}"}
    params = (
        {
            "per_page": 1000,
        }
        | {"after": after}
        if after is not None
        else None
    )
    r = requests.get(
        url + f"/api/v4/channels/{channel_id}/posts", headers=headers, params=params
    )
    if r.status_code != 200:
        raise ValueError(f"Getting posts failed with {r.status_code}.", r.content)

    return r.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", type=str)
    parser.add_argument("-n", "--username", type=str)
    parser.add_argument("-p", "--password", type=str)
    parser.add_argument("-c", "--channel", type=str)
    parser.add_argument("-a", "--after", type=str)
    args = parser.parse_args()

    token = auth(args.url, args.username, args.password)

    ret = json.loads(get_posts(args.url, args.channel, token, after=args.after))

    peices = [ret["posts"][key] for key in reversed(ret["order"])]
    merged = []

    for p in peices:
        if "__main__" not in p["message"]:
            continue
        elif p["message"].startswith("\nHey"):
            merged.append(
                {
                    "id": p["id"],
                    "message": p["message"],
                    "date": datetime.fromtimestamp(p["create_at"] // 1000),
                }
            )
        else:
            last = merged.pop()
            last["message"] += p["message"]
            merged.append(last)

    results = []
    for m in merged:
        table = "\n".join(m["message"].split("\n")[4:])
        pattern = r"\| (.+) \| (.+) \| (.+) \| (.+) \| (.+) \|"

        # Use the findall function to extract all rows that match the pattern
        matches = re.findall(pattern, table)

        # Extract the header and data rows
        header = [h.strip() for h in matches[0]]
        data = matches[2:]

        df = pd.DataFrame(data, columns=header)

        df["Time"] = df["Time"].apply(lambda x: x.split(" ")[0]).astype(float)

        results.append(df)

    times = list(itertools.chain(*[df["Time"] for df in results]))
    print("Total runs:", len(times))
    print("Total time (hours):", np.sum(times) / 3600)
    print("Average time (seconds):", np.mean(times))
    unique = functools.reduce(operator.or_, [set(df["Overrides"]) for df in results])
    print("Unique runs:", len(unique))
    pass


if __name__ == "__main__":
    main()
