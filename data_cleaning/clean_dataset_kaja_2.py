#Import libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

#Read datasets
train_df = pd.read_csv('data/train.csv', sep = ',')
test_df = pd.read_csv('data/test.csv', sep = ',')


#Test of columns
#print(train_df.SaleType.unique())
#print(train_df['Exterior2nd'].isna().sum())


#Make columns in training-set and test-set to be of the same type.
#Saw in the analysis that the some of the features in the testset 
#are of value float, while it is int in the training set

for col in test_df.columns:
    if test_df[col].dtype != train_df[col].dtype:
        #print(f"Column: {col}, Test dtype: {test_df[col].dtype}, Train dtype: {train_df[col].dtype}")
        test_df[col] = test_df[col].fillna(0)
        test_df[col] = test_df[col].astype(int)


#Identification of features with missing values: 
missing_values = [(col,train_df[col].isna().sum()) for col in train_df]
missing_values_percent = [(col,train_df[col].isna().mean() * 100) for col in train_df]
missing_values = pd.DataFrame(missing_values, columns=["column_name", "count"])
missing_values_percent = pd.DataFrame(missing_values_percent, columns = ["column_name", "percentage"])
missing_data = missing_values.merge(missing_values_percent, on = ["column_name"]).sort_values(by="count", ascending=False)
print(missing_data[missing_data['count'] > 0])

#Find columns with more than 80% missing values and drop them
missing_feat_to_drop = missing_values_percent[missing_values_percent.percentage > 80]['column_name'].to_list()
print(missing_feat_to_drop)
train1_df = train_df.copy()
train1_df = train1_df.drop(missing_feat_to_drop, axis = 1)

#Handle missing categorical values 

#For features where missing values signify the absence of a future we fill the missing values with None
features_none = ['MasVnrType','BsmtQual','BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2',
                 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual']
for feature in features_none:
    train1_df[feature] = train1_df[feature].fillna('None')

#For other categorical features we impute with mode
features_mode = ['Electrical']

for feature in features_mode:
    train1_df[feature] = train1_df[feature].fillna(train1_df[feature].mode()[0])


#Handle missing numerical values

#If missing value implies zero. E.g. No garage or no MasVnrType
features_zero = ['MasVnrArea', 'GarageYrBlt']
for feature in features_zero:
    train1_df[feature] = train1_df[feature].fillna(0)

#LotFrontage:
impute_features = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'Neighborhood']
impute_df = train1_df[impute_features].copy()
impute_df = pd.get_dummies(impute_df, columns = ['Neighborhood'], drop_first=True) #Encode categorigcal valyes 
num_cols = impute_df.columns

scaler = StandardScaler()
impute_df_scaled = pd.DataFrame(scaler.fit_transform(impute_df), columns=num_cols) #Scale the features
knn_imputer = KNNImputer(n_neighbors=5)
impute_df_imputed = pd.DataFrame(knn_imputer.fit_transform(impute_df_scaled), columns=num_cols) #Apply KNN imputer
impute_df_imputed = pd.DataFrame(scaler.inverse_transform(impute_df_imputed), columns=num_cols) #Inverse transform
#Replace Lot frontage with imputed values
train1_df['LotFrontage'] = impute_df_imputed['LotFrontage']


#LotFrontage: Uses the median of 'LotFrontage' per neighborhood
#train1_df['LotFrontage'] = train1_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

print(train1_df.isnull().sum().any())

sns.histplot(train1_df['LotFrontage'])
plt.show()