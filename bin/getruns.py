import argparse
import json
import re
from datetime import datetime
from getpass import getpass

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
    parser.add_argument("-t", "--target-url", type=str, required=True, dest="url")
    parser.add_argument("-u", "--username", type=str, required=True)
    parser.add_argument("-c", "--channel", type=str, required=True)
    parser.add_argument("-p", "--password", type=str)
    parser.add_argument("-a", "--after", type=str)
    args = parser.parse_args()

    if args.password is None:
        password = getpass(prompt=f"Mattermost password for {args.username}: ")
    else:
        password = args.password

    token = auth(args.url, args.username, password)

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

    runs = pd.concat(results)
    completed = runs[runs["Status"].str.contains("COMPLETED")]
    times = completed["Time"]
    unique = completed["Overrides"].unique()

    print("Total runs:", len(times))
    print("Total time (hours):", np.sum(times) / 3600)
    print("Average time (seconds):", np.mean(times))
    print("Unique runs:", len(unique))
    pass


if __name__ == "__main__":
    main()
