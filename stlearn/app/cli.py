import click
from .. import __version__

import os


@click.group(
    name="stlearn",
    subcommand_metavar="COMMAND <args>",
    options_metavar="<options>",
    context_settings=dict(max_content_width=85, help_option_names=["-h", "--help"]),
)
@click.help_option("--help", "-h", help="Show this message and exit.")
@click.version_option(
    version=__version__,
    prog_name="stlearn",
    message="[%(prog)s] Version %(version)s",
    help="Show the software version and exit.",
)
def main():
    os._exit
    click.echo("Please run `stlearn launch` to start the web app")


@main.command(short_help="Launch the stlearn interactive app")
def launch():
    from .app import app

    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            raise click.ClickException(
                "Port is in use, please specify an open port using the --port flag."
            ) from e
        raise
