import requests

data = {"entry": {
  "name": "Jakob Rolandsson",
  "number": "0725080995",
  "address": "Häraldsvägen 3B"
}}

# Request to add a new entry to the phonebook
r = requests.post("http://127.0.0.1:5000/phonebook", json=data)

# Request to get all entries in the phonebook
r = requests.get("http://127.0.0.1:5000/phonebook/")

# Request to get all entries in the phonebook by name
name = ""
r = requests.get(f"http://127.0.0.1:5000/phonebook/name/{name}")

# Request to get all entries in the phonebook by address
r = requests.get("http://127.0.0.1:5000/phonebook/address/31456 Bruce Mountain")

# Request to get all entries in the phonebook by number of rows (1-100)
r = requests.get("http://127.0.0.1:5000/phonebook/100")

for entry in r.json():
  print(entry["name"], entry["number"], "-", entry["address"], "\n")

# Request to delete an entry in the phonebook by name (full match)
del_entry = requests.delete("http://127.0.0.1:5000/phonebook/delete/Norma Fisher")
if del_entry.status_code == 200:
  print(del_entry.text)

# Request to get all entries in the phonebook by date (month only)
date_start = "06"
date_end = "07"
date_entry = requests.get(f"http://127.0.0.1:5000/phonebook/date/{date_start}/{date_end}")

for entry in date_entry.json():
  print(entry["name"], entry["number"], "-", entry["address"], "\n")

# Request to validate the dates of the entries in the phonebook, should only be between 22-06-01 and today's date
validate_dates = requests.get("http://127.0.0.1:5000/phonebook/validate")
invalid_dates = validate_dates.json()

if invalid_dates:
    for entry in invalid_dates:
        print("Invalid entries:","\n", entry["name"], entry["number"], "-", entry["address"], "\n", entry["added"], "\n")
else:
    print("All dates are valid.")

