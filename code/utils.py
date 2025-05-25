import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

from paths import PUBLICATION_FPATH, ENV_FPATH


def load_publication():
    """
    Load the VAE publication markdown file from the data directory.

    Returns:
        str: Content of the publication as a text string

    Raises:
        FileNotFoundError: If the publication file doesn't exist
        IOError: If there's an error reading the file
    """
    file_path = Path(PUBLICATION_FPATH)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Publication file not found: {file_path}")

    # Read and return the file content
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e


def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str or Path): Path to the YAML file to load

    Returns:
        dict: Parsed YAML content as a dictionary

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If there's an error parsing the YAML
        IOError: If there's an error reading the file
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e


def load_env():
    # Load environment variables from .env file
    load_dotenv(ENV_FPATH, override=True)

    # Check if 'XYZ' has been loaded
    api_key = os.getenv("OPENAI_API_KEY")

    assert api_key, "'api_key' has not been loaded or is not set in the .env file."


def save_text_to_file(text, filepath, header=None):
    """
    Save text content to a file.

    Args:
        text (str): The text content to save
        filepath (str or Path): Path where to save the file
        header (str): Optional header to add at the top of the file

    Raises:
        IOError: If there's an error writing the file
    """
    try:
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if header:
                f.write(f"# {header}\n")
                f.write("# " + "=" * 60 + "\n\n")
            f.write(text)

    except IOError as e:
        raise IOError(f"Error writing to file {filepath}: {e}") from e
