from myrec.config.abstract_config import AbstractConfig
from myrec.utils.config.load import load_from_json, load_from_argparse
import argparse


class Config(AbstractConfig):
    def __init__(self, config_path, args):
        super().__init__()
        self.__config_path = config_path
        self.args = args
        self.__load_from_file()
        # cmd中参数优先级更高，可以覆盖file中的参数
        self.__load_from_cmd()

    def __load_from_cmd(self):
        self.config_dict.update(load_from_argparse(self.args))

    def __load_from_file(self):
        self.config_dict.update(load_from_json(path=self.__config_path))


class FileConfig(AbstractConfig):
    def __init__(self, config_path):
        super().__init__()
        self.__config_path = config_path
        self.__load_from_file()

    def __load_from_file(self):
        self.config_dict.update(load_from_json(path=self.__config_path))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Run Model.")
#     parser.add_argument("--array", nargs='+', type=int, default=[1, 28])
#     parser.add_argument("--epochs", default=-1)
#     parser.add_argument("--str", type=str, default="fuck")
#     args = parser.parse_args()
#     json_path = "exp_setting.json"
#     config = Config(json_path, args)
#     config['epochs'] = 100
#     print("epochs" in config)
