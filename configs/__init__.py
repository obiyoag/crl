import os
import yaml
import argparse

from ast import literal_eval
from typing import Any, Dict, List, Tuple, Union


class Configs(dict):
    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    @staticmethod
    def load_file(fpath: str) -> dict:
        with open(fpath) as f:
            if fpath.endswith(".yaml"):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format. Use .yaml!")

    def load(self, fpath: str):
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        configs = self.load_file(fpath)
        parent_path = configs.pop("extends", None)

        if parent_path:
            # parent_path = os.path.join(os.path.dirname(fpath), parent_path)
            if not os.path.exists(parent_path):
                raise FileNotFoundError(f"Parent config file not found: {parent_path}")
            # Recursively load parent config
            self.load(parent_path)

        self.update_from_configs(configs)

    def update_from_configs(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], Configs):
                    self[key] = Configs()
                self[key].update(value)
            else:
                self[key] = value

    def update_from_args(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            key = opts[index][2:]
            value = opts[index + 1]
            index += 2

            current = self
            subkeys = key.split(".")
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, Configs())
            current[subkeys[-1]] = value

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, Configs):
                seperator = "\n"
            else:
                seperator = " "
            text = key + ":" + seperator + str(value)
            lines = text.split("\n")
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (" " * 2) + line
            texts.extend(lines)
        return "\n".join(texts)


def get_configs():
    configs = Configs()
    parser = argparse.ArgumentParser("concept logical")
    parser.add_argument("--configs", type=str, required=True, help="configs file")
    args, unknown_args = parser.parse_known_args()
    processed_args = []
    for arg in unknown_args:
        if "=" in arg:
            processed_args.extend(arg.split("="))
        else:
            processed_args.append(arg)
    configs.load(args.configs)
    configs.update_from_args(processed_args)
    return configs
