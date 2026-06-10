from __future__ import annotations


def make_client(ls_url: str, api_key: str):
    from label_studio_sdk import Client

    client = Client(ls_url, api_key)
    if not client.check_connection():
        raise RuntimeError("Label Studio connection failed")
    return client

