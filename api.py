import pathlib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import json
from sklearn.neighbors import NearestNeighbors
import shap

app = Flask(__name__)
# Load the model
model = joblib.load('notebook/best_lgbm.joblib')
data_test=pd.read_csv('notebook/data_to_test.csv')
data_train=pd.read_csv('notebook/data_to_train.csv')
data_final_test=pd.read_csv('notebook/test_final.csv')
# On crée la liste des ID clients qui nous servira dans l'API
id_client = data_test["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

@app.route("/", methods=["GET"])
def home():
    return "Bienvenue sur l'API du projet Implémentez un modèle de scoring"

@app.route("/load_data", methods=["GET"])
def load_data():
    
    return   jsonify ({"response": "these are the id _clients loaded:" + id_client.to_json(orient='values')})



@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

@app.route("/class_target", methods=["GET"])
def class_target():

    df_target = data_train["TARGET"].value_counts()

    return jsonify ({"response": "these are the frequency of the targets data" + df_target.to_json(orient='values')})

@app.route("/infos_client", methods=["GET"])
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == float(id)]
    
    dict_infos = {
       "Sex" : data_client["CODE_GENDER_M"].item(),
       "Statut Famille" : data_client["NAME_FAMILY_STATUS_Married"].item(),
       "age" : int(data_client["DAYS_BIRTH"].values / -365),
       "Emploi" : int(data_client["DAYS_EMPLOYED"].values / -365),
       "revenus" : data_client["AMT_INCOME_TOTAL"].item(),
       "montant_credit" : data_client["AMT_CREDIT"].item(),
       "annuites" : data_client["AMT_ANNUITY"].item(),

       }
    print(dict_infos)
    
    response = json.loads(data_client.to_json(orient='index'))

    return jsonify( response)

# Calcul des ages de la population pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():
    
    df_age = round((data_train["DAYS_BIRTH"] / -365), 2)
    return ({"response": "l'age des clients:" + df_age.to_json(orient='values')})

@app.route("/load_revenus/population", methods=["GET"])
def load_revenus_population():
    
    df_revenus = data_train["AMT_INCOME_TOTAL"] 
    return ({"response": "Revenus des clients:" + df_revenus.to_json(orient='values')})

@app.route("/predict", methods=["GET"])
def predict():
    
    id = request.args.get("id_client")
    new_data=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    idx=data_final_test.loc[data_final_test["SK_ID_CURR"] ==float (id)].index
    
    data_clients=new_data
    data_client = data_clients.iloc[idx]

    prediction = model.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return jsonify(prediction)


@app.route("/model_interpretation", methods=["GET"])
def model_interpretation():
    
    id = request.args.get("id_client")
    idx=data_final_test.loc[data_final_test["SK_ID_CURR"] ==float (id)].index
    new_data_test=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    new_data_train=data_train.drop(['Unnamed: 0', 'TARGET'], axis=1)
    data_client = new_data_test.iloc[idx]
    shap.initjs()
    model_explainer=shap.TreeExplainer(model,new_data_train )
    shap_values = model_explainer.shap_values(data_client, check_additivity=False)
    shap_values=shap_values.tolist()
    return  jsonify (shap_values)

@app.route("/load_features", methods=["GET"])
def load_features():
    new_data_test=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    features=new_data_test.columns.values.tolist()
    
    return jsonify (features)
   

@app.route("/load_data_shap", methods=["GET"])
def load_data_shap():
    id = request.args.get("id_client")
    idx=data_final_test.loc[data_final_test["SK_ID_CURR"] ==float (id)].index
    new_data_test=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    data_client = new_data_test.iloc[idx]
    response = json.loads(data_client.to_json(orient='index'))
    return jsonify (response)

@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    
    id = request.args.get("id_client")
    new_data=data_test.drop(['Unnamed: 0'], axis=1)
    new_data_train=data_train.drop(['Unnamed: 0', 'TARGET'], axis=1)
    data_client = new_data[new_data["SK_ID_CURR"] == float(id)]
    print(data_test.columns)
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(new_data_train)

    distances, indices = knn.kneighbors(data_client)

    df_voisins = data_train.iloc[indices[0], :]
    
    response = json.loads(df_voisins.to_json(orient='index'))

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)