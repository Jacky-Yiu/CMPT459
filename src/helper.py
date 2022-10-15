import random
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore")


#age convert
def convert_age_data(data):
  if type(data) == str:
    if "-" in data:
      a = data.split("-")[0]
      b = data.split("-")[1]
      if b == "":
        data=a
      else:
        data = (int(a)+int(b))/2
    return round(float(data))
  elif type(data) == float:
    data = int(round(data))
  else:
    return data
  return data

#sex imputation using male to female ratio
def random_sex_genarator(data):
  if data != "male" and data !="female":
    num = random.random()
    if num >= 0.36:
      data = "male"
    else:
      data = "female"
  return data

def impute_date_confirmation(row,table):
  if row.date_confirmation == 0:
    # get mode of 
    mode = table["date_confirmation"][table["country"] == row["country"]].mode()[0]
    if mode == 0:
      return table["date_confirmation"].mode()[0]
    return mode
  elif "-" in (row.date_confirmation) :
    date = row.date_confirmation.strip(" ").split("-")
    date = [i.strip() for i in date]
    date = pd.to_datetime(date, format="%d.%m.%Y")

    mean = (np.array(date, dtype='datetime64[s]')
        .view('i8')
        .mean()
        .astype('datetime64[s]'))
    return mean
  else:
    return row.date_confirmation

#impute province
def impute_province(row, table):
  geolocator = Nominatim(user_agent="milestone")
  if row['province'] == "missing":
    try:
      location = geolocator.reverse(str(row["latitude"])+","+str(row["longitude"]), language='en')
      address = location.raw['address']
      result = address.get("state")
      if result != None:
        return result
      else:
        result = table.province[table["country"] == row["country"]].mode()[0]
        if result == "missing":
          return row['country']
        else:
          return row['province']
    except Exception as e:
      return table.province[table["country"]== row["country"]].mode()[0]
  else:
    return row['province']
