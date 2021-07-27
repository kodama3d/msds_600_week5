import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads diabetes data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model('LR')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'},
                                            inplace=True)
    return predictions['Churn_prediction']


if __name__ == "__main__":
    df = load_data('data/new_churn_data.csv')
    
    df['NoContract'] = df['Contract']   # create a new column based on contract
    df['NoContract'] = df['NoContract'].replace({2 : 1})
    df['NoContract'] = 1 - df['NoContract']   # kind of wacky but it swaps
    
    df['ElectronicCheck'] = df['PaymentMethod']   # create a new column based on payment
    df['ElectronicCheck'] = df['ElectronicCheck'].replace({2 : 1, 3 : 1})
    df['ElectronicCheck'] = 1 - df['ElectronicCheck']   # kind of wacky but it swaps
    
    df['Difference'] = df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])
    
    df['MonthlyRatio'] = df['TotalCharges'] / df['tenure']

    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
