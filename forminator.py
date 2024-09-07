"""Convert photos of forms into a CSV file using OpenAI"""

import base64
import configparser
import csv
import json
import os
import pathlib
import sys
import threading
import tkinter as tk
from typing import Callable

import guitk as ui
import openai
import requests
from xdg_base_dirs import xdg_config_home, xdg_data_home

CONFIG_FILE = "forminator.ini"
OUTPUT_FILE = "contacts.csv"

PROMPT = """
Parse the information in the form represented by the attached image and return the results as a JSON object.
The form contains the following fields which should be mapped to the corresponding keys in the JSON object:
    First Name -> first_name
    Last Name -> last_name
    Email -> email
    Mobile -> mobile
    Address -> address
    City -> city
    State -> state
    Zip -> zip
    Newsletter -> newsletter
    Free Seminar Invitations -> free_seminar_invitations
    Consultation Appointment -> consultation_appointment
    Speaker for my group -> speaker_for_my_group

The newsletter, free_seminar_invitations, consultation_appointment, and speaker_for_my_group fields are checkboxes on the form.
If the field is checked, use a value of true and if not, use a value of false.

Also include a field called 'image' which contains the name of the image file and a field called 'confidence'
which contains the confidence score of the extracted information. You may use whatever method you prefer to
define the confidence score so long as a low score indicates low confidence in the extracted information.

If information is blank for a form field value, use an empty string for the value. If information is not blank
but is not legible, use "ERROR" as the value.

The image file name is {}. Please extract the information from this image and return it as a JSON object.
"""


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class APIRequestManager:
    def __init__(
        self,
        directory: str,
        api_key: str,
        progress_callback: Callable,
        completion_callback: Callable,
    ):
        self.directory = directory
        self.api_key = api_key
        self.completed_requests = 0
        self.num_requests = 0
        self.lock = threading.Lock()
        self.completion_callback = completion_callback
        self.progress_callback = progress_callback
        self.files = []
        self.results = []
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        self.collect_files()

    def request_complete(self, result):
        # Update the GUI with the result of each request
        print(result)

        # Lock to safely update the counter across threads
        with self.lock:
            self.completed_requests += 1
            if self.completed_requests == self.num_requests:
                # All requests are completed, invoke the main callback
                self.completion_callback(self.results, self.tokens)

    def parse_openai_json_result(self, result):
        # Navigate to the specific path within the result to access the 'content'
        content = result["choices"][0]["message"]["content"]

        # Remove the code block markers by splitting on the first occurrence of ``` and taking the second part
        json_str = content.split("```", 2)[1].strip()

        # strip the json\n from beginning
        json_str = json_str.lstrip("json\n")

        # Now parse the JSON string
        try:
            parsed_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None

        return parsed_dict

    def collect_files(self):
        """Collect all the files in the given directory"""
        for file in os.listdir(self.directory):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                self.files.append(file)
        self.num_requests = len(self.files)

    def process_files(self):
        """Process all the files in the given directory"""
        thread = threading.Thread(target=self._process_files, daemon=True)
        thread.start()

    def _process_files(self):
        for file in self.files:
            image_path = os.path.join(self.directory, file)
            prompt = PROMPT.format(file)
            self.progress_callback(f"Processing {file}...")
            self.process_image(image_path, prompt)

    def process_image(self, image_path, prompt):
        """Process a single image"""

        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            # "model": "gpt-4o-mini",
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            self.process_result(image_path, response.json())
            self.request_complete(f"Processed image {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            self.request_complete(f"Error processing image {image_path}")

    def process_result(self, image_path, result):
        """Process result from the query"""

        self.tokens["completion_tokens"] += result["usage"]["completion_tokens"]
        self.tokens["prompt_tokens"] += result["usage"]["prompt_tokens"]

        content = self.parse_openai_json_result(result)
        content = content or dict()
        self.results.append(content)
        self.progress_callback(str(content))


class MainWindow(ui.Window):

    def config(self):
        self.title = "Forminator"
        self.geometry = "800x600"

        self.load_settings()

        with ui.VLayout():
            ui.LabelEntry(
                "OpenAI API Key",
                key="api_key",
                width=30,
                default=self.api_key,
                focus=True,
            )
            with ui.HStack(vexpand=False):
                ui.Label("Forms Directory:")
                ui.Entry(key="dirname", default=self.directory, keyrelease=True)
                ui.BrowseDirectoryButton(target_key="dirname", key="browse_dir")
            with ui.HStack(vexpand=False):
                ui.Label("Output Directory:")
                ui.Entry(key="output_dir", default=self.output_dir, keyrelease=True)
                ui.BrowseDirectoryButton(target_key="output_dir")
            with ui.HStack(vexpand=False):
                ui.Label("Output File:")
                ui.Entry(key="output_file", default=self.output_file, keyrelease=True)
            with ui.HStack(vexpand=False):
                ui.Button("Process Files", key="process_files")
                ui.Button("Quit")
            ui.ProgressBar(key="progress", max=100, mode="indeterminate")
            ui.Output(key="output", height=20, width=100)

    def setup(self):
        """"Perform setup before showing window"""
        # If self.directory exists, scan the files
        self.files = []
        if self.directory:
            self.scan_files()

    @ui.on(key="dirname")
    @ui.on(key="browse_dir")
    def on_dirname(self):
        self.directory = (
            os.path.normpath(self.get("dirname").value)
            if self.get("dirname").value
            else None
        )
        if self.directory:
            self.scan_files()

    @ui.on(key="api_key")
    def on_api_key(self):
        self.api_key = self.get("api_key").value

    @ui.on(key="output_dir")
    def on_output_dir(self):
        self.output_dir = (
            os.path.normpath(self.get("output_dir").value)
            if self.get("output_dir").value
            else None
        )

    @ui.on(key="output_file")
    def on_output_file(self):
        self.output_file = (
            self.get("output_file").value if self.get("output_file").value else None
        )

    @ui.on(key="Quit")
    def on_quit(self):
        self.quit()

    @ui.on(key="process_files")
    def on_process_files(self):
        if self.validate():
            self.save_settings()
            self.get("process_files").disabled = True
            # start the progress bar
            self.get("progress").progressbar.start()
            APIRequestManager(
                self.directory, self.api_key, self.on_progress, self.on_complete
            ).process_files()

    def on_progress(self, status: str):
        """Update progress"""
        print(status)

    def on_complete(self, results, tokens):
        self.get("process_files").disabled = False
        progress = self.get("progress")
        progress.progressbar.stop()
        progress.value = 0
        print("Processing complete")
        print(f"Completion tokens: {tokens['completion_tokens']}")
        print(f"Prompt tokens: {tokens['prompt_tokens']}")
        self.write_csv(results)

    def write_csv(self, results):
        """Write the results to a CSV file"""
        if not results:
            return
        output_file = os.path.join(self.output_dir, self.output_file)
        if os.path.exists(output_file):
            """Prompt to overwrite the file"""
            if not tk.messagebox.askyesno(
                "File Exists", f"Overwrite {output_file}?", parent=self.root
            ):
                return
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {output_file}")

    def scan_files(self):
        """Scan the forms directory for image files"""
        self.files = []
        if not self.directory:
            return
        for file in os.listdir(self.directory):
            suffix = pathlib.Path(file).suffix.lower()
            if suffix in (".jpg", ".jpeg", ".png"):
                image_path = os.path.join(self.directory, file)
                self.files.append(image_path)
        print(f"Found {len(self.files)} image files to process in {self.directory}")

    def validate(self):
        """Validate the input fields"""
        if not self.api_key:
            tk.messagebox.showinfo("Error", "Please enter an OpenAI API key")
            return False
        if not self.directory:
            tk.messagebox.showinfo(
                "Error", "Please select a forms directory to process"
            )
            return False
        if not self.output_dir:
            tk.messagebox.showinfo(
                "Error", "Please select an output directory to save the results"
            )
            return False
        if not self.output_file:
            tk.messagebox.showinfo("Error", "Please enter an output file name")
            return False
        if not self.files:
            tk.messagebox.showinfo(
                "Error", f"No files found to process in {self.directory}"
            )
            return False
        return True


    def load_settings(self):
        """Load the configuration from the config file"""
        config = configparser.ConfigParser()
        config_path = os.path.join(xdg_config_home(), CONFIG_FILE)

        # Set default values
        self.api_key = os.getenv("OPENAI_API_KEY", "Please enter OpenAI API Key")
        self.directory = os.path.expanduser("~/")
        self.output_dir = os.path.expanduser("~/Desktop")
        self.output_file = OUTPUT_FILE

        if os.path.exists(config_path):
            config.read(config_path)
            if "Settings" in config:
                self.api_key = config["Settings"].get("api_key", self.api_key)
                self.directory = config["Settings"].get("directory", self.directory)
                self.output_dir = config["Settings"].get("output_dir", self.output_dir)
                self.output_file = config["Settings"].get(
                    "output_file", self.output_file
                )

    def save_settings(self):
        """Save the configuration to the config file"""
        config = configparser.ConfigParser()
        config["Settings"] = {
            "api_key": self.api_key,
            "directory": self.directory,
            "output_dir": self.output_dir,
            "output_file": self.output_file,
        }
        config_path = os.path.join(xdg_config_home(), CONFIG_FILE)
        print(f"Saving settings to {config_path}")
        with open(config_path, "w") as configfile:
            config.write(configfile)


if __name__ == "__main__":
    MainWindow().run()
