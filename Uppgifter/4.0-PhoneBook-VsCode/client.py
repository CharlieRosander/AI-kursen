import requests

data = {"entry": {
  "name": "Jakob Rolandsson",
  "number": "0725080995",
  "address": "Häraldsvägen 3B"
}}


# r = requests.post("http://127.0.0.1:5000/phonebook", json=data)

# r = requests.get("http://127.0.0.1:5000/phonebook/")

# r = requests.get("http://127.0.0.1:5000/phonebook/address/31456 Bruce Mountain")

# r = requests.get("http://127.0.0.1:5000/phonebook/100")

# del_entry = requests.delete("http://127.0.0.1:5000/phonebook/delete/Norma Fisher")
# if del_entry.status_code == 200:
#   print(del_entry.text)

# date_start = "22-06-04"
# date_end = "22-06-15"
# date_entry = requests.get(f"http://127.0.0.1:5000/phonebook/date/{date_start}/{date_end}")

validate_dates = requests.get("http://127.0.0.1:5000/phonebook/validate")

# print(date_entry.json())
# for entry in r.json():
#   print(entry["name"], entry["number"], "-", entry["address"], "\n")

print(validate_dates.text)
# for entry in validate_rows.json():
#   print(entry["name"], entry["number"], "-", entry["address"], "\n")
