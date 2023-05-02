### 1. Install/reset Ubuntu
```
Pc settings -> Apps & Features -> Ubuntu -> Advanced options -> Reset

You might also have to install WSL through Microsoft Store, as systemd/systemctl is not enabled by default when "installing wsl through enabling it in windows". When setting up the pgadmin4-webmode you might get an error if this is not enabled.
```
### 2. Update WSL, install venv + airflow and set up the airflow-directory
```
sudo apt update
sudo apt upgrade -y
sudo apt install python3-venv
mkdir airflow
cd airflow
mkdir dags plugins
python -m venv venv
. ./venv/bin/activate
pip3 install WTForms==2.3.3
pip3 install flask
pip3 install apache-airflow  
```

### 3. Create the airflow start-script: start.sh
```
cat > start.sh

Plug this into the script, press ctrl + o to save:

#!/bin/bash
. ./venv/bin/activate
export AIRFLOW__API__AUTH_BACKEND='airflow.api.auth.backend.basic_auth'
airflow standalone
```

### 4. Install postgresql
```
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql

Change postgres password:
sudo -u postgres psql
ALTER USER postgres PASSWORD 'password';

Restart postgres for changes to take effect:
sudo service postgresql restart

```

### 5. Install and configure PgAdmin4
```
# Setup the repository

# Install the public key for the repository (if not done previously):

curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg

# Create the repository configuration file:

sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'

# Install pgAdmin4

# Install for both desktop and web modes:
sudo apt install pgadmin4

	# Install for desktop mode only:
	sudo apt install pgadmin4-desktop
	
	# Install for web mode only:
	sudo apt install pgadmin4-web

# Configure the webserver, if you installed pgadmin4-web:
sudo /usr/pgadmin4/bin/setup-web.sh

# Edit listening_address in postgres config
sudo nano /etc/postgresql/15/main/postgresql.conf

#Allow port in firewall
sudo ufw allow 5432/tcp

# Open pgadmin webmode and connect to the server in your browser
localhost/pgadmin4
```

### 6. Setup project in airflow dir:
```
# Create a projs dir:
mkdir projs

# Create the project dir:
mkdir <proj-name>

# Create the files used in the proj, .env, main, reqs etc.

Reqs for ETL:
	matplotlib==3.7.1
	pandas==2.0.1
	python-dotenv==1.0.0
	Requests==2.29.0
	psycopg2 --no-binary=psycopg2 <-- Might have to install outside venv
	OR 
	psycopg2-binary
```