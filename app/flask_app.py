#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
import os
import logg
import pandas as pd
import numpy as np
from pipeline import Model_Pipeline,CleaningTextData,FillingNaN,TfIdf
from sklearn.ensemble import GradientBoostingRegressor
import nltk
import ssl
import datetime

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import request, jsonify

log = None
app = Flask(__name__)

data=""
prediction = None
gbr_result = None


@app.route('/')
def hello_world():
	return str(data)

@app.route('/api/v1/resources/conv', methods=['GET'])
def api_filter():
    query_parameters = request.args


    id = query_parameters.get('id')
    campaign_id = query_parameters.get('campaign_id')
    country_code = query_parameters.get('country_code')
    date = query_parameters.get('date')
    original_spend = query_parameters.get('original_spend')
    impression = query_parameters.get('impression')
    reach = query_parameters.get('reach')
    click = query_parameters.get('click')
    conversion_values_usd = query_parameters.get('conversion_values_usd')
    project_id = query_parameters.get('project_id')
    fb_id = query_parameters.get('fb_id')
    campaign_name = query_parameters.get('campaign_name')
    start_time = query_parameters.get('start_time')
    currency_code = query_parameters.get('currency_code')
    
    d = {
        'id':id,
        'campaign_id':campaign_id,
        'country_code':country_code,
        'date':date,
        'original_spend':float(original_spend),
        'impression':impression,
        'reach':reach,
        'click':click,
        'conversion_values_usd':float(conversion_values_usd),
        'project_id':project_id,
        'fb_id':fb_id,
        'campaign_name':campaign_name,
        'start_time':start_time,
        'currency_code':currency_code,
    }

    print(d)
    
    
    
    if not (id or campaign_id or country_code or date or original_spend or impression or reach or click or project_id or fb_id):
        return page_not_found(404)
        
    

    df = pd.DataFrame(d, index=[0])

    #df = pd.read_csv("try_to_test.csv")
    df = df.set_index('id')

    print(df)

    submition_res = func(df)



    return jsonify(submition_res.to_json(orient="records"))

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

def func(X_test):


    y_test = X_test['conversion_values_usd']

    del X_test['conversion_values_usd']

    data = pipeline.pipelineData(X_test)

    print("sdfknksdnfklsldnfnklsdnf")

    print(type(data))

    data['conversion_values_usd'] = y_test.values


    print("sdfknksdnfsdfklaslflnksalfdkklsldnfnklsdnf")
    data = pd.DataFrame(data)

    print(data)

    print(type(data))



    del data["start_time"]
    del data["currency_code"]
    del data["country_code"]

    data["date"] = data["date"].apply(lambda x: int(x.strftime('%w')))

    data = data.astype('float')

    prediction = gbr_result.predict(data)

    submition_res = pd.DataFrame({'id': list(X_test.index), 'conversion_predict': prediction})
    print(submition_res)

    return submition_res


if __name__ == "__main__":
	
    log_directory = 'log'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)


    pipeline = Model_Pipeline("dft_idf_final.joblib")
    gbr_result = Model_Pipeline("XGBRegressor.joblib")
    

    #log = logg.setup_logging('Server')
    #log = logg.get_log("Web-server")

    app.run(debug=False,host='0.0.0.0')