import pandas as pd
import numpy as np
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model('BEST') # searching for the best model which changes based on modeling
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'},
                                            inplace=True)
    predictions.rename({'Score': 'Percentage'}, axis=1, inplace=True)
    return predictions[predictions.columns[6:8]]


if __name__ == "__main__":
    """
    Runs full script if main is loaded
    Transforms new data to create matching features to the model
    """
    df = load_data('data/new_churn_data_unmodified.csv')

    df.fillna(df['TotalCharges'].median(), inplace=True)
    df.at[df['tenure'] == 0, 'tenure'] = np.nan
    df['tenure'].fillna(df['tenure'].median(), inplace=True)
    
    df['PhoneService'] = df['PhoneService'].replace({'No': 0, 'Yes': 1})
    df['Contract'] = df['Contract'].replace({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].replace({'Electronic check': 0, 'Mailed check': 1,
                'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})

    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
