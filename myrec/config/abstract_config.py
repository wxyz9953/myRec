class AbstractConfig:
    def __init__(self):
        self.config_dict = {}

    def __load_from_file(self):
        pass

    def __load_from_cmd(self):
        pass

    def update(self, config):
        dict2 = config.config_dict
        s = set(self.config_dict.keys()) & set(dict2.keys())
        if len(s) > 0:
            raise KeyError("Key Conflict:" + str(s))
        self.config_dict.update(dict2)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("Wrong Index")
        self.config_dict[key] = value

    def __getitem__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            raise KeyError("Wrong Index: %s" % item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise KeyError("Wrong Index: %s" % key)
        return key in self.config_dict

    def __str__(self):
        config_str = ""
        for (k, v) in self.config_dict.items():
            config_str += "%s:%s\n" % (k, v)
        return config_str

    def __repr__(self):
        self.__str__()
