import json
import os
from datetime import datetime


def list_dict_to_jsonl(list_dict: list[dict]):
    return "\n".join(json.dumps(list_dict[i]) for i in range(len(list_dict)))


def log(message: str, **kwargs):
    """
    Logs message and timestamp only when it's rank 0
    """
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
        print(f"\n[{timestamp}] {message}", **kwargs)
