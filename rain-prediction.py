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

# sns.heatmap(data.isnull())
# plt.show()

data.dropna(subset=["RainToday", "RainTomorrow"], inplace=True)
print(data.info())

#Exploratory Data Analysis and Visualization
sns.set_style("darkgrid")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["figure.facecolor"] = "#00000000"
# fig1 = px.histogram(data, x="Location", title="Location vs Rainy Days", color="RainToday")
# fig1.show()
# fig2 = px.histogram(data, x="Temp3pm", title="Temp at 3pm vs Rain Tomorrow", color="RainTomorrow")
# fig2.show()
# fig3 = px.histogram(data, x="RainTomorrow", title="Rain Tomorrow vs Rain Today", color="RainToday")
# fig3.show()
# fig4 = px.scatter(data.sample(2000), x="MinTemp", y="MaxTemp", color="RainToday")
# fig4.show()
# fig5 = px.strip(data.sample(2000), x="Temp3pm", y="Humidity3pm", title="Temp vs Humidity", color="RainTomorrow")
# fig5.show()

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
# plt.show()

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
