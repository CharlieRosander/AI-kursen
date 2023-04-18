import os,sys
from flask import Flask, request
from flaskr.phonebook import Phonebook
from flaskr.mock_phonebook import initialize_mock
from flask import render_template_string

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append("..")

app = Flask(__name__)
phonebook = Phonebook(CURR_DIR_PATH + "/data/database.db")
initialize_mock(phonebook)

# Function to get all entries in the phonebook and return the results
@app.route("/phonebook/")
def get_phonebook():
    phonebook_data, phonebook_df = phonebook.get_all()
    return render_template_string(phonebook_df.to_html(classes='table table-responsive table-striped'))

# Function to GET the phonebook by name and return the results
@app.route("/phonebook/name/<name_query>")
def get_by_name(name_query):
  return phonebook.get_by_name(name_query)

# Function to GET the phonebook by address and return the results
@app.route("/phonebook/address/<address_query>")
def get_by_address(address_query):
  return phonebook.get_by_address(address_query)

@app.route("/phonebook", methods=["POST", "PUT"])
def enter_record():
  json_data = request.get_json()
  phonebook.add(json_data["entry"])
  return "Added to the phonebook", 201

@app.route("/phonebook/delete/<name>", methods=["DELETE"])
def delete_record(name):
  phonebook.delete_entry(name)
  return f"Deleted {name} from the phonebook", 200

# Function to GET the phonebook by number of rows (1-100) and return the results
@app.route("/phonebook/<num_of_rows>")
def get_phonebook_rows(num_of_rows):
  return phonebook.get_rows(num_of_rows)
