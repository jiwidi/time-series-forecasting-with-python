import fbprophet
import pandas as pd
import json
from utils.metrics import evaluate_all





def get_data():
    df = pd.read_csv("datasets/air_pollution.csv",parse_dates=['date']).sort_values(by='date')
    df = df.rename(columns={'date': 'ds', 'pollution': 'y'})
    split = int(len(df)*0.8)
    train = df[: split]
    test = df[split :]
    
    return train,test


def get_model():
    return fbprophet.Prophet(changepoint_prior_scale=0.15)
    
    
    
def train_model(prophetModel,df):
    prophetModel.fit(df)
    return prophetModel

def predict(prophetModel,df):
    return model.predict(df)

def evaluate(original, predicted):
    original = original.y.values
    predicted = predicted.yhat.values
    result = evaluate_all(original,predicted)
    json.dump(result, open("results_prophet.txt",'w'))
    return result
    
    
train,test = get_data()
model = get_model()
model = train_model(model,train)
test_predictions = predict(model,test)
result = evaluate(test,test_predictions)