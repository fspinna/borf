import json
import pathlib


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def make_path(folder):
    path = pathlib.Path(folder)
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)
    return path


def dump_json(value, filename):
    with open(filename, "w", encoding="utf8") as json_file:
        json.dump(value, json_file, ensure_ascii=False)
