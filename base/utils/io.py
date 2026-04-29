import pickle
import os
import json
import time


def pickle_load(filename: str):
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        print(f"Logging Info - Loaded: {filename}")
    except EOFError:
        print(f"Logging Error - Cannot load: {filename}")
        obj = None

    return obj


def pickle_dump(filename: str, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"Logging Info - Saved: {filename}")


def format_filename(_dir: str, filename_template: str, **kwargs):
    """Obtain the filename of data_repository base on the provided template and parameters"""
    filename = os.path.join(_dir, filename_template.format(**kwargs))
    return filename


def write_log(filename: str, log, mode="w"):
    if not os.path.exists(filename):
        with open(filename, mode) as handle:
            json.dump([], handle, indent=4, ensure_ascii=False)

    data = []
    with open(filename, "r") as handle:
        data = json.load(handle)
    endtry_id = len(data)
    entry = {"id": endtry_id, "time": str(time.ctime(time.time())), "result": log}
    data.append(entry)

    with open(filename, mode) as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
