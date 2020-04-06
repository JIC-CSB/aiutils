import ruamel.yaml

class YAMLConfig(object):

    def __init__(self, config_fpath):
        yaml = ruamel.yaml.YAML()
        with open(config_fpath) as fh:
            self.config_dict = yaml.load(fh)

    def __getitem__(self, key):
        return self.config_dict[key]

    def __getattr__(self, name):
        return self.config_dict[name]