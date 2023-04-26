import os
import requests
from dotenv import load_dotenv

load_dotenv()
openweathermap_api_key = os.getenv('OPENWEATHERMAP_API_KEY')

city_query = "stockholm, se"

url = f'http://api.openweathermap.org/data/2.5/weather?q={city_query}&appid={openweathermap_api_key}&units=metric'
response = requests.get(url).json()


print(response)