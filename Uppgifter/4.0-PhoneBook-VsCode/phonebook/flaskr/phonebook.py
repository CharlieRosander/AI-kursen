from sqlalchemy import create_engine  # sql driver
import pandas as pd

class SQLWriter:
  def __init__(self, db_path):
    print("sqlite://" + db_path)
    self.engine = create_engine("sqlite:///" + db_path)  # local sqlite

  def get_data(self, dbname):
    if not(self.engine.has_table(dbname)):
      return []
    return pd.read_sql(dbname, self.engine, columns=["name", "number", "address"])

  def write_to(self, dbname, df):
    df.to_sql(dbname, self.engine, if_exists="replace")

class Phonebook:
  def __init__(self, db_path):
    self.sql = SQLWriter(db_path)
    self.data = []

  def get_data(self):
    if len(self.data) == 0:
      self.data = self.sql.get_data("phonebook")

    return self.data

  def write_to(self, entry):
    self.data = self.get_data().append(entry, ignore_index=True)

  def save_all_updates(self):
    self.sql.write_to("phonebook", self.data)

  def get_all(self):
    get_all_data = self.get_data()
    return self.get_data().to_json(orient="records"), get_all_data

  def get_by_name(self, name):
    df = self.get_data()
    return df[df["name"].str.contains(name)].to_json(orient="records")

  def get_by_address(self, address):
    df = self.get_data()
    return df[df["address"].str.contains(address)].to_json(orient="records")
  
  # Function with a conditional statement to check if the number of rows is between 1 and 100
  # If not, return an error message, else return the number of rows requested
  def get_rows(self, rows):
    df = self.get_data()
    if int(rows) < 1 or int(rows) > 100:
      return "Only 1 to 100 rows are allowed"
    else:
      return df.head(int(rows)).to_json(orient="records")


  def add(self, entry):
    self.write_to(entry)