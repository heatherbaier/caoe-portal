########################################################################################
######################          Import packages      ###################################
########################################################################################
from flask import Blueprint, render_template, flash
from flask_login import login_required, current_user
from __init__ import create_app, db

import datetime

# [START gae_python38_auth_verify_token]
# [START gae_python3_auth_verify_token]
from flask import Flask, render_template, request

import pandas as pd
import json
import os

from app_helpers import *


########################################################################################
# our main blueprint
main = Blueprint('main', __name__)

@main.route('/') # home page that return 'index'
def index():
    return render_template('index.html')

@main.route('/profile') # profile page that return 'profile'
@login_required
def profile():


    print("WORKING DIRECTORY: ", os.getcwd())


    # Read in census and migration data
    df = pd.read_csv(DATA_PATH)

    # mig_6months = pd.read_csv(MONTH6_PATH)
    mig_12months = pd.read_csv(MONTH12_PATH)

    # print(df.head())

    # with open(MIGRATION_PATH) as m:
    #     mig_data = json.load(m)

    # # Get total # of migrants and a list of muni ID's
    # total_migrants = sum(list(mig_data.values()))
    # municipality_ids = list(mig_data.keys())

    # Calculate the average age of migrants per muni
    # df['avg_age_weight'] = df['avg_age'] * df['sum_num_intmig']
    # avg_age = df['avg_age_weight'].sum() / df['sum_num_intmig'].sum()

    # Open the variables JSON and the JSON containing the readable translation of the variables
    with open("./vars.json", "r") as f:
        grouped_vars = json.load(f)

    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)

    # print(grouped_vars)

    # Get all of the variables to send to Flask for dropdown options
    demog, family, edu, employ, hhold, crime = get_column_lists(df, var_names, grouped_vars)


    # return render_template(
    #                     'profile.html',
                        # demog_data = demog,
                        # family_data = family,
                        # edu_data = edu,
                        # employ_data = employ,
                        # hhold_data = hhold,
                        # crime_data = crime,
                        # month6_migs = int(mig_6months["serial"].sum()),
                        # month12_migs = int(mig_12months["serial"].sum(),
    #                     name = current_user.name)
    #                 )

    return render_template('profile.html', name=current_user.name,
                            demog_data = demog,
                            family_data = family,
                            edu_data = edu,
                            employ_data = employ,
                            hhold_data = hhold,
                            crime_data = crime,
                            month6_migs = int(mig_12months["serial"].sum()),
                            month12_migs = int(mig_12months["serial"].sum()))


@main.route('/geojson-features', methods=['GET'])
@login_required
def get_all_points():

    """
    Grabs the polygons from the geojson, converts them to JSON format with geometry and data 
    features and sends back to the webpage to render on the Leaflet map
    """

    print("here!!")

    # Convert the geoJSON to a dataframe and merge it to the migration data
    feature_df = convert_to_pandas(geodata_collection, MATCH_PATH, MONTH12_PATH)
    feature_df['sum_num_intmig'] = feature_df['serial'].fillna(0)
    feature_df['perc_migrants'] = feature_df['sum_num_intmig']# / feature_df['total_pop']
    
    print(feature_df)

    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    num_migrants = feature_df['perc_migrants']
    shapeIDs = feature_df['shapeID']
    shapeNames = feature_df["properties.ipumns_simple_wgs_wdata_geo2_mx1960_2015_ADMIN_NAME"]

    # For each of the polygons in the data frame, append it and it's data to a list 
    # of dicts to be sent as a JSON back to the Leaflet map
    features = []
    for i in range(0, len(feature_df)):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": types[i],
                "coordinates": coords[i]
            },
            "properties": {'num_migrants': num_migrants[i],
                           'shapeID': str(shapeIDs[i]),
                           'shapeName': shapeNames[i]
                          }
        })

    print("done up to here!!")


    response = jsonify(features)

    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")        

    print("returning MAP!!")

    return response



@main.route('/predict_migration', methods=['GET', 'POST'])
def predict_migration():

    print(request.json)

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Starting predictions."}, outfile)

    # Parse the selected municipalities and get their unique B ID's
    selected_municipalities = request.json['selected_municipalities']

    print("LEN SELECTED MUNIS: ", len(selected_municipalities))

    # TEMPORARY UNTIL YOU GET THE BIG IMAGES DOWNLOADED
    selected_municipalities = [sm for sm in selected_municipalities if sm in munis_available]
    # selected_municipalities = [sm for sm in selected_municipalities if graph_id_dict[sm] not in BAD_IDS]

    # Read in the migration data and subset it to the selected municipalities
    dta = pd.read_csv(MODEL_DATA_PATH)
    dta = dta.dropna(subset = ['muni_id'])

    dta_ids = dta["muni_id"].to_list()
    selected_municipalities = [sm for sm in selected_municipalities if int(sm) in dta_ids]

    print("IN DF: ", selected_municipalities)

    # If no muni's are selected, select them all
    if len(selected_municipalities) == 0:
        selected_municipalities = [str(i) for i in dta['GEO2_MX'].to_list()]
        selected_municipalities = [sm for sm in selected_municipalities if sm in munis_available]
        # selected_municipalities = [sm for sm in selected_municipalities if graph_id_dict[sm] not in BAD_IDS]
        print("Selected municipalities since none were selected: ", selected_municipalities)

    print(dta.head())

    for muni in selected_municipalities:

        cur_dta = dta[dta["muni_id"] == int(muni)].fillna(0)
        cur_dta = np.array(dta.drop(["muni_id", "year", "month", "migrants"], axis = 1).values, dtype = np.float32)
        cur_dta[cur_dta != cur_dta] = 0
        print(cur_dta)

    # dta_selected, dta_dropped, muns_to_pred = prep_dataframes(dta, request, selected_municipalities)

    #######################################################################
    # Create some sort of dictionary with references to the graph_id_dict # 
    #######################################################################
    selected_muni_ref_dict = {}
    for muni in selected_municipalities:
        muni_ref = graph_id_dict[muni]
        selected_muni_ref_dict[muni] = muni_ref

    #######################################################################
    # Create a dictionary with graph_id_dict                              #
    # references mapped to the new census data                            #
    #######################################################################
    new_census_vals = {}
    for sm in range(0, len(selected_municipalities)):
        new_census_vals[selected_muni_ref_dict[selected_municipalities[sm]]] = muns_to_pred[sm]

    #######################################################################
    # Predict the new data                                                # 
    #######################################################################
    predictions = predict(graph, selected_muni_ref_dict, new_census_vals, selected_municipalities)

    #######################################################################
    # Update the new predictions in the dta_selected dataframe and append #
    # that to all of the data in dta_dropped that wan't selected to       #
    # create a full dataframe with everything                             #
    #######################################################################
    dta_selected['sum_num_intmig'] = predictions
    dta_final = dta_selected.append(dta_dropped)
    print("ALL DATA SHAPE: ", dta_final.shape)
    print("DTA FINAL HEAD: ", dta_final.head())

    #######################################################################
    # Normalize the geoJSON as a pandas dataframe and merge in the new    #
    # census & migration data                                             #
    #######################################################################
    dta_final['GEO2_MX'] = dta_final['GEO2_MX'].astype(str)
    dta_final[['GEO2_MX', 'sum_num_intmig']].to_csv("./map_layers/sum_num_intmig.csv", index = False)
    geoDF = json_normalize(geodata_collection["features"])
    merged = pd.merge(geoDF, dta_final, left_on = "properties.shapeID", right_on = "GEO2_MX")
    merged['sum_num_intmig'] = merged['sum_num_intmig'].fillna(0)
    merged['perc_migrants'] = merged['sum_num_intmig'] / merged['total_pop']

    dta_final['perc_migrants'] = dta_final['sum_num_intmig'] / dta_final['total_pop']
    dta_final[['GEO2_MX', 'perc_migrants']].to_csv("./map_layers/perc_migrants.csv", index = False)

    og_df = pd.read_csv(DATA_PATH)
    og_df = og_df[['GEO2_MX', 'sum_num_intmig', 'total_pop']].rename(columns = {'sum_num_intmig': 'sum_num_intmig_og'})
    og_df['GEO2_MX'] = og_df['GEO2_MX'].astype(str)
    change_df = pd.merge(og_df, dta_final[['GEO2_MX', 'sum_num_intmig']])
    change_df['absolute_change'] = change_df['sum_num_intmig'] - change_df['sum_num_intmig_og']
    change_df[['GEO2_MX', 'absolute_change']].to_csv("./map_layers/absolute_change.csv", index = False)
    change_df['perc_change'] = (change_df['sum_num_intmig'] - change_df['sum_num_intmig_og']) / change_df['sum_num_intmig_og']
    change_df = change_df.replace([np.inf, -np.inf], np.nan)
    change_df = change_df.fillna(0)
    change_df[['GEO2_MX', 'perc_change']].to_csv("./map_layers/perc_change.csv", index = False)

    #######################################################################
    # Aggregate statistics and send to a JSON                             #
    #######################################################################

    total_pred_migrants = merged['sum_num_intmig'].sum()
    merged['avg_age_weight'] = merged['avg_age'] * merged['sum_num_intmig']
    avg_age = merged['avg_age_weight'].sum() / merged['sum_num_intmig'].sum()
    migration_statistics = {'avg_age': avg_age, "total_pred_migrants": float(total_pred_migrants)}
    with open('predicted_migrants.json', 'w') as outfile:
        json.dump(migration_statistics, outfile)

    #######################################################################
    # Convert features to a gejson for rendering in Leaflet               #
    #######################################################################
    features = convert_features_to_geojson(merged, column = 'perc_migrants')

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Rendering new migration map..."}, outfile)

    return jsonify(features)

app = create_app() # we initialize our flask app using the __init__.py function
if __name__ == '__main__':
    db.create_all(app=create_app()) # create the SQLite database
    app.run(debug=True) # run the flask app on debug mode