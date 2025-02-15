import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("/Users/in22417145/PycharmProjects/Logistic Regression - Rain Prediction/weatherAUS.csv")
print(data.head())
print(data.info())
print(data.describe())

sns.heatmap(data.isnull())
plt.show()

data.dropna(subset=["RainToday", "RainTomorrow"], inplace=True)
print(data.info())

#Exploratory Data Analysis and Visualization
sns.set_style("darkgrid")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["figure.facecolor"] = "#00000000"
fig1 = px.histogram(data, x="Location", title="Location vs Rainy Days", color="RainToday")
fig1.show()
fig2 = px.histogram(data, x="Temp3pm", title="Temp at 3pm vs Rain Tomorrow", color="RainTomorrow")
fig2.show()
fig3 = px.histogram(data, x="RainTomorrow", title="Rain Tomorrow vs Rain Today", color="RainToday")
fig3.show()
fig4 = px.scatter(data.sample(2000), x="MinTemp", y="MaxTemp", color="RainToday")
fig4.show()
fig5 = px.strip(data.sample(2000), x="Temp3pm", y="Humidity3pm", title="Temp vs Humidity", color="RainTomorrow")
fig5.show()

#When working with large datasets, it is ideal to use a sample data to setup our model.
use_sample = True
sample_fraction = 0.1
if use_sample:
    data_sample = data.sample(frac=sample_fraction).copy()
print(data_sample.head())

#Good practice to split the dataset into train set, validation set and test set (60-20-20). If a separate test dataset is provided,
#then split the dataset into 75-25 ratio.  
from sklearn.model_selection import train_test_split
train_val, df_test = train_test_split(data_sample, test_size=0.2,random_state=42)
df_train, df_val = train_test_split(train_val, test_size=0.25,random_state=42)
print("shape of training dataset", df_train.shape)
print("shape of validating dataset", df_val.shape)
print("shape of testing dataset", df_test.shape)

#However while working with dates, its better to split the train, validate and test datasets with date, so that the model is trained
#on past data and evaluated on the data from future.
#For our current sample, we can use Date column to create another column called Year. We will use last 2 years as test and a year
#before that as validate.
plt.title("No of Rows per Year")
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["text.color"] = "#000000"
sns.countplot(x=pd.to_datetime(data_sample["Date"]).dt.year, palette="magma")
plt.show()

#from above we see data from 2008 to 2017. So we will use 2008-2014 for train, 2015 for validate and 2016-2017 for test.
year = pd.to_datetime(data_sample["Date"]).dt.year
df_train = data_sample[year < 2015]
df_val = data_sample[year == 2015]
df_test = data_sample[year > 2015]

'''
the above code can also be written as below:
data_sample["year"] = pd.to_datetime(data_sample["Date"]).dt.year
df_train = data_sample[data_sample["year"] < 2015]
df_val = data_sample[data_sample["year"] == 2015]
df_test = data_sample[data_sample["year"] > 2015]
'''

print("shape of training dataset", df_train.shape)
print("shape of validating dataset", df_val.shape)
print("shape of testing dataset", df_test.shape)

#Identify the target and predictors
#Exclude Date and RainTomorrow column
input_cols = list(data_sample.columns)[1:-1]
target_cols = "RainTomorrow"

train_inputs = df_train[input_cols].copy()
train_targets = df_train[target_cols].copy()
val_inputs = df_val[input_cols].copy()
val_targets = df_val[target_cols].copy()
test_inputs = df_test[input_cols].copy()
test_targets = df_test[target_cols].copy()

#Identify Numerical and Categorical Columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.to_list()
categorical_cols = train_inputs.select_dtypes("object").columns.to_list()
print(f"Numerical columns: {numeric_cols}")
print(f"Categoical columns: {categorical_cols}")

train_inputs[numeric_cols].describe()

#Impute missing numeric data
#There are several techniques but we shall use the basic to replace the missing data with mean value
print(data_sample[numeric_cols].isna().sum())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")

#Fit i.e calculate the imputed value (mean in this case) to each column
imputer.fit(data_sample[numeric_cols])
print(list(imputer.statistics_))

#Fill the missing data into the dataframes
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

#Scaling Numeric Features
#MinMaxScaler scales between 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_sample[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

#Encoding Categorical Data
#One hot encoding involves adding a new binary(0/1) column for each unique category of a categorical column
print(data_sample[categorical_cols].nunique())
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(data_sample[categorical_cols])

encoder_cols = list(encoder.get_feature_names_out(categorical_cols))
print(f"Encoder Columns: {encoder_cols}")

train_inputs[encoder_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoder_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoder_cols] = encoder.transform(test_inputs[categorical_cols])
print(train_inputs)

#Model Training
#We can optimize the result by either using parameter max_iter (eg:100) or setting tol to a value (eg: 0.001) which would
# stop if the loss or tolerance is less than the value set. 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="liblinear", max_iter=100)
#Ensure there are no categorical columns as we have transformed it to new categories using one hot encoder.
model.fit(train_inputs[numeric_cols + encoder_cols], train_targets)
#we can check the weight of each column on the target column by checking the coefficient
print(model.coef_.tolist())
weight_df = pd.DataFrame({"feature" : (numeric_cols + encoder_cols), "weight": model.coef_.tolist()[0]})
plt.figure(figsize=(15,50))
sns.barplot(data=weight_df, x="weight", y="feature")
plt.show()

#Making Prediction and Evaluating the model
X_train = train_inputs[numeric_cols + encoder_cols]
X_val = val_inputs[numeric_cols + encoder_cols]
X_test = test_inputs[numeric_cols + encoder_cols]

train_pred = model.predict(X_train)

from sklearn.metrics import accuracy_score
train_score = accuracy_score(train_targets, train_pred)
print(f"Train Accuracy Score: {train_score}")

#We can also check the probability. It shows how confidence it is for Yes and No for each row
train_prob = model.predict_proba(X_train)
print(model.classes_)
print(f"Probability Score: {train_prob}")

#Confusion matrix. It gives percantage of TN, FP, FN and TP.
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(train_targets, train_pred, normalize="true")
print(matrix)

#Computung the model's accuracy on validation and test datasets
val_pred = model.predict(X_val)
val_score = accuracy_score(val_targets, val_pred)
print(f"Validate Accuracy Score: {val_score}")

test_pred = model.predict(X_test)
test_score = accuracy_score(test_targets, test_pred)
print(f"Test Accuracy Score: {test_score}")

#Now how good is the accuracy we got? This depends on business scenario but a good way to determine whether its useful is by
#comparing its result to "random" or "dumb" model.
def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))

def all_no(inputs):
    return np.full(len(inputs), "No")

random_guess(X_val)
all_no(X_val)

#Lets check the accuracy of these random/dumb models
random_accuracy = accuracy_score(test_targets, random_guess(X_test))
dumb_accuracy = accuracy_score(test_targets, all_no(X_test))
print(f"Random Accuracy: {random_accuracy}\nDumb Accuracy: {dumb_accuracy}")

#Making predictions on a single input using our model
new_input = {"Date" : "2025-01-21",
             "Location" : "GoldCoast",
             "MinTemp" : 25.2,
             "MaxTemp" : 36.5,
             "Rainfall" : 0.0,
             "Evaporation" : 6.2,
             "Sunshine" : np.nan,
             "WindGustDir" : "NNW",
             "WindGustSpeed" : 70.7,
             "WindDir9am" : "NW",
             "WindDir3pm" : "NNE",
             "WindSpeed9am" : 12.0,
             "WindSpeed3pm" : 22.0,
             "Humidity9am" : 55.0,
             "Humidity3pm" : 88.0,
             "Pressure9am" : 1000.1,
             "Pressure3pm" : 1004.3,
             "Cloud9am" : 7.0,
             "Cloud3pm" : 4.0,
             "Temp9am" : 27.8,
             "Temp3pm" : 35.5,
             "RainToday": "No"
             }

new_input_df = pd.DataFrame([new_input])
print(new_input_df)

new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoder_cols] = encoder.transform(new_input_df[categorical_cols])

X_new_input = new_input_df[numeric_cols + encoder_cols]

prediction = model.predict(X_new_input)[0]
print(f"New input Prediction: {prediction}")

probability = model.predict_proba(X_new_input)[0]
print(f"New input Probability: {probability}")

#Saving and Loading Trained Models
import joblib

#create dictionary containing all the required objects
rain_prediction = {
    "model" : model,
    "imputer" : imputer,
    "scaler" : scaler,
    "encoder" : encoder,
    "target_cols" : target_cols,
    "numeric_cols" : numeric_cols,
    "categorical_cols" : categorical_cols,
    "encoder_cols" : encoder_cols
}

joblib.dump(rain_prediction, "rain_prediction.joblib")

