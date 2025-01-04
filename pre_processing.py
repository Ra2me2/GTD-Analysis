import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# **SUMMARY OF THE PREPROCESSING STEPS**

# Input Dataset Cleaning:
# Retains specified columns (train_columns) relevant for training or testing.
# Drops rows with missing values in critical columns (nkill, city, gname) and removes invalid values (e.g., "unknown" or empty strings).
# Filters nkill values to a maximum of 500 for training data to remove outliers.

# Feature Engineering:
# Applies frequency encoding to columns like gname, city, and country by mapping their occurrence counts.
# Merges precomputed severity scores and kill sums from external CSV files for categorical columns (e.g., attacktype1, gname, city) and renames columns for clarity.
# Computes a likelihood score (nkill_likelihood_score) as a weighted combination of kill sums, adjusted by the success value of incidents.

# Categorical Encoding:
# Converts categorical variables (e.g., region) into one-hot encoded features.

# Normalization:
# Normalizes numerical features using MinMaxScaler.

# Return Values:
# For training data (test_data=False): Splits the dataset into feature matrix X and target variable y (nkill) and returns their values and the final dataset.
# For testing data (test_data=True): Returns only the normalized feature matrix X.


def preprocess_data(dataset, test_data = False):
    train_columns = ['country', 'city', 'success',
                     'attacktype1', 'targtype1', 'weaptype1','region','gname']

    if test_data == False:
        dataset = dataset.dropna(subset=['nkill','city'])
        dataset = dataset[dataset['nkill'] <= 500]
        train_columns.append('nkill')

    dataset = dataset[train_columns]
    dataset.fillna(0, inplace=True)
    dataset = dataset.dropna(subset=['gname', 'city'])
    dataset['gname'] = dataset['gname'].str.lower()
    dataset = dataset[dataset['gname'] != ''] 
    dataset = dataset[dataset['gname'] != 'unknown']

    dataset['city'] = dataset['city'].str.lower()
    dataset = dataset[dataset['city'] != '']
    dataset = dataset[dataset['city'] != 'unknown']

    # Frequency Encoding
    dataset['gname_freq'] = dataset['gname'].map(dataset['gname'].value_counts())
    dataset['city_freq'] = dataset['city'].map(dataset['city'].value_counts())
    dataset['country_freq'] = dataset['country'].map(dataset['country'].value_counts())


 ############################################################################
    kill_sum_gname = pd.read_csv("metrics\kill_sum_gname.csv")
    kill_sum_city = pd.read_csv("metrics\kill_sum_city.csv")
    kill_sum_country = pd.read_csv("metrics\kill_sum_country.csv")

    kill_sum_attacktype1 = pd.read_csv("metrics\kill_sum_attacktype1.csv")
    kill_sum_targtype1 = pd.read_csv("metrics\kill_sum_targtype1.csv")
    kill_sum_weaptype1 = pd.read_csv("metrics\kill_sum_weaptype1.csv")


    dataset = dataset.merge(kill_sum_attacktype1[['attacktype1', 'severity_score']], on='attacktype1', how='left')
    dataset.rename(columns={'severity_score': 'attacktype1_score'}, inplace=True)

    dataset = dataset.merge(kill_sum_targtype1[['targtype1', 'severity_score']], on='targtype1', how='left')
    dataset.rename(columns={'severity_score': 'targtype1_score'}, inplace=True)

    dataset = dataset.merge(kill_sum_weaptype1[['weaptype1', 'severity_score']], on='weaptype1', how='left')
    dataset.rename(columns={'severity_score': 'weaptype1_score'}, inplace=True)

    dataset = dataset.merge(kill_sum_gname[['gname', 'severity_score']], on='gname', how='left')
    dataset.rename(columns={'severity_score': 'gname_score'}, inplace=True)

    dataset = dataset.merge(kill_sum_country[['country', 'severity_score']], on='country', how='left')
    dataset.rename(columns={'severity_score': 'country_score'}, inplace=True)

    dataset = dataset.merge(kill_sum_city[['city', 'severity_score']], on='city', how='left')
    dataset.rename(columns={'severity_score': 'city_score'}, inplace=True)


    dataset['gname_kill_sum'] = dataset['gname'].map(kill_sum_gname.set_index('gname')['kill_sum']).fillna(0)
    dataset['city_kill_sum'] = dataset['city'].map(kill_sum_city.set_index('city')['kill_sum']).fillna(0)
    dataset['country_kill_sum'] = dataset['country'].map(kill_sum_country.set_index('country')['kill_sum']).fillna(0)
    dataset['attacktype1_kill_sum'] = dataset['attacktype1'].map(kill_sum_attacktype1.set_index('attacktype1')['kill_sum']).fillna(0)
    dataset['targtype1_kill_sum'] = dataset['targtype1'].map(kill_sum_targtype1.set_index('targtype1')['kill_sum']).fillna(0)
    dataset['weaptype1_kill_sum'] = dataset['weaptype1'].map(kill_sum_weaptype1.set_index('weaptype1')['kill_sum']).fillna(0)

    success_weight = dataset['success'].apply(lambda x: 0.9 if x == 1 else 0.1)

    # Calculate the nkill_likelihood_score
    dataset['nkill_likelihood_score'] = (
        0.2 *  dataset['gname_kill_sum'] +
        0.05 * dataset['city_kill_sum'] +
        0.05 * dataset['country_kill_sum'] +
        0.3 * dataset['attacktype1_kill_sum'] +
        0.1 * dataset['targtype1_kill_sum'] +
        0.3 * dataset['weaptype1_kill_sum']
    ) * success_weight


 ############################################################################
    categorical_columns = ['region']
    dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)

    dataset = dataset.drop(columns=[ 'attacktype1', 'targtype1', 'weaptype1','gname','country','city'])
   

    scaler = MinMaxScaler()

    if test_data == False:
        X = dataset.drop('nkill', axis=1)
        y = dataset['nkill'].astype(float)
                        
        X_normalized = scaler.fit_transform(X)
        X = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)

        dataset = pd.concat([X, y], axis=1)
        dataset = dataset.astype(float)

        return X.values, y.values, dataset
    
    else:
        X = dataset
        
        X_normalized = scaler.fit_transform(X)
        X = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
        
        return X.values
