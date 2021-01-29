import click

from aiutils.config import YAMLConfig


@click.command()
@click.argument('config_fpath')
def main(config_fpath):
    pass


if __name__ == "__main__":
    main()
