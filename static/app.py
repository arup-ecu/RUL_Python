import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    output = {
        "message": "Hello Buddy"
    }
    return jsonify(output)

@app.route('/RUL', methods=['POST'])
def getRUL():
    print(request)
    dischargeTime = request.json['dischargeTime']
    maxVoltageDischarge = request.json['maxVoltageDischarge']
    minVoltageDischarge = request.json['minVoltageDischarge']
    cycleIndex = request.json['cycleIndex']
    cycleIndexMultiplier = request.json['cycleIndexMultiplier'] #default value .5
    outputMultiplier = request.json['outputMultiplier'] #default value 2
    mileage = request.json['mileage']

    battery = pd.read_csv("Battery_RUL.csv")
    # Filter rows where Discharge Time <9000
    battery = battery[battery['Discharge Time (s)'] < 9000]

    target = battery['RUL']
    feature = battery.drop(['RUL', 'Decrement 3.6-3.4V (s)',
                            'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)'], axis=1)

    scaler = StandardScaler()
    feature_std = scaler.fit_transform(feature)
    feature_std = pd.DataFrame(feature_std, columns=feature.columns)
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.1, random_state=2404)

    pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])

    model = Pipeline(steps=[('preprocessing', pipeline), ('algorithm', RandomForestRegressor())])
    model.fit(X_train, y_train)

    testData = [[cycleIndex, dischargeTime, maxVoltageDischarge, minVoltageDischarge]]
    pred = model.predict(testData)
    print(pred[0])
    print(pred[0]*outputMultiplier)

    rul = 0
    if outputMultiplier == 0:
        rul = pred[0]
    else:
        rul = pred[0]*outputMultiplier
    output = {
        "remainingBatteryLife": rul,
        "mileage": mileage
    }
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)

# inputJson = {
#     'dischargeTime': 2595.30,
#     'maxVoltageDischarge': 3.670,
#     'minVoltageDischarge': 3.211
# }

#https://www.moesif.com/blog/technical/api-development/Building-RESTful-API-with-Flask/
