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


class Config(object):

    def __init__(self, raw_config):
        self.raw_config = raw_config

    @classmethod
    def from_fpath(cls, yaml_fpath):
        yaml = ruamel.yaml.YAML()
        with open(yaml_fpath) as fh:
            raw_config = yaml.load(fh)

        return cls(raw_config)

    def __getattr__(self, name):
        return self.raw_config[name]

    def as_readme_format(self):
        yaml = ruamel.yaml.YAML()

        readme_dict = {
            "config": self.raw_config
        }

        with io.StringIO() as f:
            yaml.dump(readme_dict, f)
            readme_content = f.getvalue()

        return readme_content
