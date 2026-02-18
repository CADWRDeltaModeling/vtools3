import click
from vtools.functions.separate_species import main as separate_main


@click.group(help="vtools CLI tools for time series processing.")
@click.help_option("-h", "--help")  # Add the help option at the group level
def cli():
    """Main entry point for vtools commands."""
    pass


# Register the commands
cli.add_command(separate_main, "separate_species")
