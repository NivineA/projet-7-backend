import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, make_response
import joblib
import json
from sklearn.neighbors import NearestNeighbors
import shap

app = Flask(__name__)
# Load the model
model = joblib.load('notebook//best_lgbm.joblib')
Knn = joblib.load('notebook//knn_model.joblib')
data_test_initial=pd.read_parquet('notebook/data_test_initial.parquet.gzip')
data_test=pd.read_csv('notebook//data_to_test_sample.csv.zip')
data_train=pd.read_csv('notebook//data_to_train_api.csv')
data_final_test=pd.read_csv('notebook//test_final_sample.csv.zip')
# On crée la liste des ID clients qui nous servira dans l'API
id_client = data_test_initial["SK_ID_CURR"][:1000].values
id_client = pd.DataFrame(id_client)

def generateMetrics():# page de bienvenue
    description= "Bienvenue sur l'API du projet Implémentez un modèle de scoring:\n\nVoici les endpoints de cette API\n\n load_data: requête pour le chargement du jeu de données\n load_age_population: requête pour charger les âges de la population\n load_revenus/population: visualiser le revenu de tous les clients\n predict: requête pour prédire le score d'un client sélectionné(accepté ou refusé)\n load_data_predict: requête pour charger les données avec le score prédit\n model_interpretation_shap: requête renvoyant les shap_values du modèle d'un client sélectionné\n load_data_shap: requête permet de charger les données utilisés pour le calcul des shap values\n load_features: requête illustrant les features du modèle\n load_feature_importance: requête permettant d'illustrer les importances des différents features du modèle\n load_voisins: visualise la liste des dossiers similaire à un certain client"
    return(description)   

@app.route("/", methods=["GET"])# afficher la description en format texte
def home():
    response = make_response(generateMetrics(), 200)
    response.mimetype = "text/plain"
    return response

@app.route("/load_data", methods=["GET"]) # charger les id d'un client sélectionné
def load_data():
    return  id_client.to_json(orient='values')


@app.route("/load_data_test", methods=["GET"]) # charger les données de tous les clients pour la visualisation graphique des distributions
def load_data_test():
    new_data=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    response = json.loads(new_data.to_json(orient='index'))

    return jsonify( response)
    

@app.route("/infos_client", methods=["GET"])# charger les données d'un client sélectionné
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test_initial[data_test_initial["SK_ID_CURR"] == float(id)]
    
    dict_infos = {
       "age" : int(data_client["DAYS_BIRTH"].values / -365),
       "Sex" : data_client["CODE_GENDER"].item(),
       "Statut Famille" : data_client["NAME_FAMILY_STATUS"].item(),
       "Education" : data_client["NAME_EDUCATION_TYPE"].item(),
       "Emploi" : int(data_client["DAYS_EMPLOYED"].values / -365),
       "Type_revenus":data_client["NAME_INCOME_TYPE"].item(),
       "Revenus" : data_client["AMT_INCOME_TOTAL"].item(),
       "Type contrat" : data_client["NAME_CONTRACT_TYPE"].item(),
       "Montant_credit" : data_client["AMT_CREDIT"].item(),
       "Annuites" : data_client["AMT_ANNUITY"].item(),
    }
    print(dict_infos)
    
    response = json.loads(data_client.to_json(orient='index'))

    return jsonify( response)

# Calcul des ages de la population utilisé pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():
    
    df_age = round((data_test_initial["DAYS_BIRTH"] / -365), 2)
    return  df_age.to_json(orient='values')

# Calcul des années d'emploi de la population utilisé pour le graphique

@app.route("/load_days_employed_population", methods=["GET"])
def load_days_employed_population():
  
    df_days_employed = round((data_test_initial["DAYS_EMPLOYED"] / -365), 2)
    return  df_days_employed.to_json(orient='values')

# renvoie le genre de la population utilisé pour le graphique

@app.route("/load_sex_population", methods=["GET"])
def load_sex_population():
    
    df_sex = (data_test_initial["CODE_GENDER"] )
    return  df_sex.to_json(orient='values')

# renvoie le statut de famille de la population utilisé pour le graphique
@app.route("/load_family_status_population", methods=["GET"])
def load_family_status_population():
    
    df_family = (data_test_initial["NAME_FAMILY_STATUS"] )
    return  df_family.to_json(orient='values')

#renvoie le niveau d'éducation de la population utilisé pour le graphique
@app.route("/load_education_population", methods=["GET"])
def load_education_population():
    
    df_education = (data_test_initial["NAME_EDUCATION_TYPE"] )
    return  df_education.to_json(orient='values')

#renvoie le type de revenu de la population utilisé pour le graphique
@app.route("/load_income_type_population", methods=["GET"])
def load_income_type_population():
    df_income = (data_test_initial["NAME_INCOME_TYPE"] )
    return  df_income.to_json(orient='values')

#renvoie le revenu de la population utilisé pour le graphique
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():
    
    df_revenus = data_test_initial["AMT_INCOME_TOTAL"] 
    return  df_revenus.to_json(orient='values')

#renvoie le type de contrat de la population utilisé pour le graphique
@app.route("/load_contract_population", methods=["GET"])
def load_contract_population():
    df_contract = (data_test_initial["NAME_CONTRACT_TYPE"] )
    return  df_contract.to_json(orient='values')

#renvoie le montant du crédit  de la population utilisé pour le graphique
@app.route("/load_credit_population", methods=["GET"])
def load_credit_population():
    
    df_credit = data_test_initial["AMT_CREDIT"] 
    return  df_credit.to_json(orient='values')

#renvoie le montant des annuités  de la population utilisé pour le graphique
@app.route("/load_annuity_population", methods=["GET"])
def load_annuity_population():
    
    df_annuity = data_test_initial["AMT_ANNUITY"] 
    return  df_annuity.to_json(orient='values')

#prédit le score de faillite d'un client sélectionné
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

#permet de charger les données  des client incluant  le score calculé
@app.route("/load_data_predict", methods=["GET"])
def load_data_predict():
    score=[]
    df=pd.DataFrame()
    id_client = data_test_initial["SK_ID_CURR"][:100].values
    id_client = pd.DataFrame(id_client)
    for id in id_client.loc[:,0]:
        new_data=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
        idx=data_final_test.loc[data_final_test["SK_ID_CURR"] ==float (id)].index
        data_clients=new_data
        data_client = data_clients.iloc[idx]
        prediction = model.predict_proba(data_client)
        prediction = prediction[0].tolist()
        score.append(prediction[1])
        data_client['score']=prediction[1]
        df = df.append(data_client, ignore_index=True)
    df["SK_ID_CURR"]=id_client.loc[:,0]
    test=pd.merge(df, data_test_initial.iloc[:100,:], on=['SK_ID_CURR'], how='inner')
    response = json.loads(test.to_json(orient='index'))
    return jsonify( response)

# permet de calculer les shap values du modèle pour un client sélectionné
@app.route("/model_interpretation_shap", methods=["GET"])
def model_interpretation_shap():
    id = request.args.get("id_client")
    idx=data_final_test.loc[data_final_test["SK_ID_CURR"] ==float (id)].index
    new_data_test=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    data_client = new_data_test.iloc[idx]
    shap.initjs()
    model_explainer=shap.TreeExplainer(model, feature_perturbation="tree_path_dependent" )
    shap_values = model_explainer.shap_values(data_client, check_additivity=False)
    shap_values=np.asarray(shap_values)
    shap_values=shap_values.tolist()
    return  jsonify (shap_values)

# calcul de l'importance des features
@app.route("/load_feature_importance", methods=["GET"])
def load_feature_importance():
    features_importance = model.feature_importances_
    features_importance=features_importance.tolist()
    return jsonify(features_importance)

# charger les features du modèle
@app.route("/load_features", methods=["GET"])
def load_features():
    new_data_test=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    features=new_data_test.columns.values.tolist() 
    return jsonify (features)

# charger les données utilisé par shap après la sélection finale des features
@app.route("/load_data_shap", methods=["GET"])
def load_data_shap():
    id = request.args.get("id_client")
    idx=data_final_test.loc[data_final_test["SK_ID_CURR"] ==float (id)].index
    new_data_test=data_final_test.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1)
    data_client = new_data_test.iloc[idx]
    response = json.loads(data_client.to_json(orient='index'))
    return jsonify (response)

# afficher les 10 dossiers similaires au client sélectionné
@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    
    id = request.args.get("id_client")
    new_data=data_test.drop(['Unnamed: 0'], axis=1)
    new_data_train=data_train.drop(['Unnamed: 0'], axis=1)
    data_client = new_data[new_data["SK_ID_CURR"] == float(id)]
    print(data_test.columns)

    distances, indices = Knn.kneighbors(data_client)

    df_voisins = new_data_train.iloc[indices[0], :]
    
    response = json.loads(df_voisins.to_json(orient='index'))

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)