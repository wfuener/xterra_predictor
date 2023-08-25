import json
from datetime import datetime
import os.path
import time
import mlflow
import requests
from bs4 import BeautifulSoup
BASE_DIR = os.path.join(os.path.dirname(__file__))
INPUT_FILE = f"{BASE_DIR}/working_files/downloads/age_group_full.html"
OUTPUT_FILE = f"{BASE_DIR}/working_files/age_group_full.json"


def main():
    with open(INPUT_FILE, 'r') as file:
        html = file.read()
    parsed = html_to_dict(html)
    transform_types(parsed)
    output(parsed)


def html_to_dict(html):
    """Take html and parse into list of dicts"""
    soup = BeautifulSoup(html, 'html.parser')

    table = soup.find('table')
    thead = table.find('thead')
    tbody = table.find('tbody')
    tr_tags = tbody.find_all('tr')

    parsed_data = []

    for tr_tag in tr_tags:
        times = tr_tag.find_all(class_='time')
        data = {
            'place': tr_tag.find(class_='place').text,
            'bib': tr_tag.find(class_='bib').text,
            'first_name': tr_tag.find(class_='participantName__name__firstName').text,
            'last_name': tr_tag.find(class_='participantName__name__lastName').text,
            'gender': tr_tag.find_all('td')[3].text,
            'age': tr_tag.find_all('td')[4].text,
            'city': tr_tag.find_all('td')[5].text,
            'swim_time': times[0].text,
            't1_time': times[1].text,
            'bike_time': times[2].text,
            't2_time': times[3].text,
            'run_time': times[4].text,
            'chip_time': times[5].text,
        }
        parsed_data.append(data)

    mlflow.log_metric("num_records", len(parsed_data))
    return parsed_data


def transform_types(parsed):
    """Transform data type"""
    for row in parsed:
        row['place'] = str_to_int(row['place'])
        row['bib'] = str_to_int(row['bib'])
        row['age'] = str_to_int(row['age'])


def str_to_int(_str):
    try:
        integer_value = int(_str)
        return integer_value
    except ValueError:
        return None


def output(parsed_data):
    """Write to file and log_artifact"""
    with open(OUTPUT_FILE, 'w') as file:
        file.write(json.dumps(parsed_data, default=str, indent=1))

    mlflow.log_artifact(local_path=OUTPUT_FILE)


if __name__ == '__main__':
    main()

