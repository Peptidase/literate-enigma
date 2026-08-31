import pathlib
import requests
import json
import argparse
import tqdm

argument_parser = argparse.ArgumentParser(description="Download a PDF file from a URL.")
argument_parser.add_argument("json_path", help="The path to the JSON file containing the PDF URL.")
argument_parser.add_argument("save_path", help="The path where the downloaded PDF will be saved.")

def downloadFile(url, fileName):
    with open(fileName, "wb") as file:
        response = requests.get(url)
        file.write(response.content)


def download_pdf(json_path, save_path):
    """
    Downloads a PDF file from the given URL and saves it to the specified path.

    Args:
        json_path (str): The path to the JSON file containing the PDF URL.
        save_path (str): The path where the downloaded PDF will be saved.
    """
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

        for key,value in tqdm.tqdm(data.items(), desc="Downloading PDFs", unit="file"):
            downloadFile(value, pathlib.Path(save_path) / pathlib.Path(value+".pdf").name)


if __name__ == "__main__":
    args = argument_parser.parse_args()
    download_pdf(args.json_path, args.save_path)