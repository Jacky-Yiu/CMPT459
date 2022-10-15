import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats
import seaborn as sns
import folium
import helper
import swifter

case_train = pd.read_csv('../data/cases_2021_train.csv')
case_location = pd.read_csv('../data/location_2021.csv')
case_test = pd.read_csv('../data/cases_2021_test.csv')

#task 1.1
print("Doing task 1.1 ...")
outcome_group = {"Discharged": "hospitalized",
                 "Discharged from hospital": "hospitalized", 
                 "Hospitalized": "hospitalized",
                 "critical condition": "hospitalized",
                 "discharge": "hospitalized",
                 "discharged": "hospitalized", 
                 "Alive": "nonhospitalized",
                 "Receiving Treatment": "nonhospitalized", 
                 "Stable": "nonhospitalized",
                 "Under treatment": "nonhospitalized", 
                 "recovering at home 03.03.2020": "nonhospitalized", 
                 "released from quarantine": "nonhospitalized", 
                 "stable": "nonhospitalized",
                 "stable condition": "nonhospitalized",
                 "Dead":"deceased",
                 "Death":"deceased", 
                 "Deceased":"deceased", 
                 "Died":"deceased", 
                 "death":"deceased", 
                 "died":"deceased",
                 "Recovered": "recovered", 
                 "recovered": "recovered"}

case_train.outcome_group = case_train.apply(lambda row: outcome_group[row.outcome], axis=1)
case_train = case_train.drop("outcome", axis=1)

# task 1.4
print("Doing task 1.4")
#drop row with NA as age
case_train = case_train.dropna(subset=['age'])
case_test = case_test.dropna(subset=['age'])

# convert age range to average age and turn float into int
case_train.age = case_train.swifter.apply(lambda row: helper.convert_age_data(str(row.age)), axis=1)
case_test.age = case_test.swifter.apply(lambda row: helper.convert_age_data(str(row.age)), axis=1)

#print age histogram
plt.figure(0)
plt.hist(case_train['age'], bins=100)
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_train Age Histogram')
plt.savefig('../plots/task-1.4/case_train_age_histogram.png')

plt.figure(1)
plt.hist(case_test['age'], bins=100)
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_test Age Histogram')
plt.savefig('../plots/task-1.4/case_test_age_histogram.png')

#inpute Sex (missing 2.5% data), we found out that female to male ratio is 0.36:0.64
print("imputing case_train sex")
case_train.sex = case_train.swifter.apply(lambda row: helper.random_sex_genarator(row.sex), axis=1)
print("imputing case_test sex")
case_test.sex = case_test.swifter.apply(lambda row: helper.random_sex_genarator(row.sex), axis=1)
plt.figure(2)
plt.hist(case_train['sex'])
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_train Sex Histogram')
plt.savefig('../plots/task-1.4/case_train_sex_histogram.png')

plt.figure(3)
plt.hist(case_test['sex'])
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_test Sex Histogram')
plt.savefig('../plots/task-1.4/case_test_sex_histogram.png')

#impute missing date in train and test
print("imputing case_train date_confirmation")
case_train.date_confirmation.fillna(0, inplace=True)
case_train["date_confirmation"] = case_train.swifter.apply(lambda row: helper.impute_date_confirmation( row, case_train), axis=1)

print("imputing case_train date_confirmation")
case_test.date_confirmation.fillna(0, inplace=True)
case_test["date_confirmation"] = case_test.swifter.apply(lambda row: helper.impute_date_confirmation( row, case_test), axis=1)



case_train['date_confirmation'] = pd.to_datetime(case_train['date_confirmation'],infer_datetime_format=True)
case_test['date_confirmation'] = pd.to_datetime(case_test['date_confirmation'],infer_datetime_format=True)


plt.figure(4)
plt.hist(case_train['date_confirmation'])
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_train date_confirmation Histogram')
plt.savefig('../plots/task-1.4/case_train_date_confirmation_histogram.png')

plt.figure(5)
plt.hist(case_test['date_confirmation'])
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_test date_confirmation Histogram')
plt.savefig('../plots/task-1.4/case_test_date_confirmation_histogram.png')

#impute province
print("imputing case_train province")
case_train.province.fillna("missing", inplace=True)
case_train["province"] = case_train.swifter.apply(lambda row: helper.impute_province( row, case_train), axis=1)

print("imputing case_test province")
case_test.province.fillna("missing", inplace=True)
case_test["province"] = case_test.swifter.apply(lambda row: helper.impute_province( row, case_test), axis=1)



#impute country
case_train["country"] = case_train["country"].fillna(value="Taiwan")
case_test["country"] = case_test["country"].fillna(value="Taiwan")

#Task 1.5
print("Doing task 1.5")
#remove age outlier in training and testing
case_train = case_train[(np.abs(stats.zscore(case_train["age"])) < 3)]
case_test = case_test[(np.abs(stats.zscore(case_test["age"])) < 3)]

plt.figure(6)
plt.hist(case_train['age'], bins=100)
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_train Age Histogram')
plt.savefig('../plots/task-1.5/case_train_age_histogram.png')

plt.figure(7)
plt.hist(case_test['age'], bins=100)
plt.ylabel('Frequency count')
plt.xlabel('Data');
plt.title('case_test Age Histogram')
plt.savefig('../plots/task-1.5/case_test_age_histogram.png')

#Task 1.6
print("Doing task 1.6")
print("Joining location and test/train data")
case_location = case_location.replace("W.P. Kuala Lumpur","Wilayah Persekutuan Kuala Lumpur")
case_location = case_location.replace("Korea, South","South Korea")   
case_location = case_location.replace("Taiwan*","Taiwan") 
case_location = case_location.replace("Bayern","Bavaria") 
case_location = case_location[case_location.Country_Region!="US"] 

case_train = case_train.replace("Kanagawa Prefecture","Kanagawa")
case_train = case_train[case_train.country!="United States"]
case_test = case_test.replace("Kanagawa Prefecture","Kanagawa")
case_test = case_test[case_test.country!="United States"]


jointed_train = pd.merge(case_train, case_location, left_on=['province','country'],right_on=["Province_State","Country_Region"], how="left", indicator=True)
jointed_test = pd.merge(case_test, case_location, left_on=['province','country'],right_on=["Province_State","Country_Region"], how="left", indicator=True)

index_list = jointed_train[(jointed_train["_merge"] != "both") == True].index
case_train_subset = case_train.iloc[index_list]

index_list = jointed_test[(jointed_test["_merge"] != "both") == True].index
case_test_subset = case_test.iloc[index_list]

jointed_train2 = pd.merge(case_train_subset, case_location, left_on=['country'],right_on=["Combined_Key"], how="left", indicator=True)
jointed_test2 = pd.merge(case_test_subset, case_location, left_on=['country'],right_on=["Combined_Key"], how="left", indicator=True)

jointed_train3 = pd.concat([jointed_train[(jointed_train["_merge"] == "both") == True],jointed_train2])
jointed_train3.reset_index(inplace=True,drop=True)

jointed_test3 = pd.concat([jointed_test[(jointed_test["_merge"] == "both") == True],jointed_test2])
jointed_test3.reset_index(inplace=True,drop=True)

jointed_train3.to_csv('../results/cases_2021_train_processed.csv')
jointed_test3.to_csv('../results/cases_2021_test_processed.csv')
case_location.to_csv('../results/location_2021_processed.csv')

#1.7
print("Doing task 1.7")
final_train = jointed_train3.drop(['additional_information','source','Province_State',"Country_Region","Last_Update","Lat","Long_","Combined_Key","_merge"],axis=1)
final_test = jointed_test3.drop(['additional_information','source','Province_State',"Country_Region","Last_Update","Lat","Long_","Combined_Key","_merge"],axis=1)

final_train.to_csv('../results/cases_2021_train_processed_features.csv')
final_test.to_csv('../results/cases_2021_test_processed_features.csv')