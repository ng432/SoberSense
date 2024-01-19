

import json
from pathlib import Path

""" Functions to load data """


def load_av_normalising_values():
    """ Function to load values for touch acceleration and velocity normalisation.
    These have been pre-extracted from the data. See pilot_data_exploration.py for extraction code."""
  
    package_directory = Path(__file__).parent.parent.parent
    data_file_path = package_directory / 'data_for_normalising' / 'av_normalising_values.json'
    
    with open(data_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    return data
