1. Familiarize yourself with the existing files: app.py, client.py, and the files inside the "flaskr" folder.

2. Expand the API by implementing the following missing features:
    a. Get parts of the phonebook (return up to 100 rows).
    b. Query by address (returns all matches with an address query, similar to the existing query by name).
    c. Delete entry (deletes ONLY if the name is a full match).

   Implement the feature logic in phonebook.py, while handling the Flask routing and data return in app.py. Access the data in phonebook.py using the method "get_data" rather than directly accessing self.data.

3. Enable a hidden feature: The database has a 4th column called "added" that holds the date a phone number was added. To enable this, change the shown columns in the class "SQLWriter" in phonebook.py.

4. Implement a new feature: Query by added date (returns all phone numbers that were added between the start_month and end_month).

5. Fix specification failure:
    a. Validate that all the added dates are between 1st of June 2022 and today. You can create an ETL file called "date_etl.py" and use either SQLite directly on the database or query the web API using the existing "get_all" method, then save the data using pandas and local CSV formats.
    b. Modify the "add" feature of the REST API to prevent adding a date before today's date.
