function predict_migration() {

    // Make two new variables to store column name sand their associated percentage change values
    var column_names = [],
        percent_changes = [];

    // Grab all of the percent change inputs
    var inputs = document.getElementsByTagName('input');

    // For all of the inputs, append the ID and value to their respective lists
    for (i = 0; i < inputs.length; ++i) {
        column_names.push(inputs[i].id);
        percent_changes.push(inputs[i].value);
    }

    // If there are no municipalities selected, provide the user with the option to either go back and pick some or add the inputted increases to all of them
    if (window.selected_polys.length == 0) {
        if (confirm("\nNo municipalities have been selected on the map.\n\nPress 'OK' to apply your variable changes to all of the municipalities in Mexico or 'Cancel' to go back and choose a subset of municipalities.") != true) {
            return 
        }
    }

    // If the user hasn't made any changes to the data, don't allow them to run the model
    var num_changes = percent_changes.filter(x => x != '100').length
    if (num_changes == 0) {
        if (alert("\nNo changes have been made to the input variables. Please edit the desired variables before predicting again.") != true) {
            return 
        }
    }

    // Post all of the variable ID's and their percent changes back to Flaks
    fetch('/predict_migration', {

        // When the data gets POSTed back to Flask, it'll be in JSON format
        headers: {
            'Content-Type': 'application/json'
        },

        // Send the data back as a POST request
        method: 'POST',

        // Here's where you construct the JSON
        body: JSON.stringify({

            "selected_municipalities": window.selected_polys,
            "column_names": column_names,
            "percent_changes": percent_changes

        })

        // ...and then send it off
        }).then(function (response) {
            return response.text();
        }).then(function (text) {

            // Remove the current migration data layer from the map
            mymap.removeLayer(window.poly);

            window.poly = [];

            // Remove the drawn polygons from the map and re-initalize the drawnPolys group as empty
            mymap.removeLayer(window.drawnPolys);
            window.drawnPolys = new L.featureGroup().addTo(mymap);

            // Convert the new migration data into a leaflet geoJSON
            var polys = L.geoJSON(JSON.parse(text), {style: polygon_style, onEachFeature: onEachFeature})//.addTo(mymap);

            console.log("POLYS: ", polys.getLayers().length)

            // Update the global window.poly variable & add it to the map
            window.poly = polys;
            window.poly.addTo(mymap);

            window.selected_polys = [];

            // Zoom the map back out to all of Mexico                
            // mymap.fitBounds(window.poly.getBounds());
            mymap.setView(new L.LatLng(23.6345, -102.5528), 6);
            
            // Function to get the data from the Flask function/URL (TO-DO: REMOVE ALL OF THE FUNCTIONS FROM HERE AND USE WINDOW.POLY TO EDIT THEM)
            axios.get('http://127.0.0.1:5000/update_stats')

            .then(response => {


                var map_button_list = ["sum_num_intmig_button", "perc_migrants_button", "absolute_change_button", "perc_change_button"]

                // Hack for switching button colors because you suck at CSS
                for (var mb = 0; mb < map_button_list.length; mb++) {
                    document.getElementById(map_button_list[mb]).style.backgroundColor = "#0E0C28";
                    document.getElementById(map_button_list[mb]).style.color = "white";
                    document.getElementById(map_button_list[mb]).style.fontWeight = "normal";
                }



                document.getElementById("absolute_change_button").style.display = 'block';
                document.getElementById("perc_change_button").style.display = 'block';
                document.getElementById("perc_migrants_button").style.backgroundColor = 'white';
                document.getElementById("perc_migrants_button").style.color = '#0E0C28';
                document.getElementById("perc_migrants_button").style.fontWeight = 'bold';

                document.getElementById("absolute_change_button").style.display = 'inline';
                document.getElementById("absolute_change_label").style.display = 'inline';
                document.getElementById("perc_change_button").style.display = 'inline';
                document.getElementById("perc_change_label").style.display = 'inline';




                // Update all of the HTML text that doesn't involve the trending icon
                document.getElementById("grid-container").style.gridTemplateColumns = "auto auto auto auto";
                document.getElementById("total_migrants").innerHTML = response.data['predicted_migrants'].toLocaleString();

                document.getElementById("change_migrants").style.display = "inline";
                document.getElementById("change_migrants_t").innerHTML = response.data['change'].toLocaleString();
                document.getElementById("p_change_migrants_t").innerHTML = response.data['p_change'].toString().concat("%");
                
                document.getElementById("avg_age").innerHTML = response.data['avg_age'];
                
                document.getElementById("avg_age_change").style.display = "inline";
                document.getElementById("avg_age_change_t").innerHTML = response.data['avg_age_change'].toString();
                document.getElementById("pavg_age_change_t").innerHTML = response.data['pavg_age_change'].toString();//.concat(" years");



                // if p_change is greater than 1, make icon green & trending_up and vice versa
                if (response.data['p_change'] > 0) {
                    document.getElementById("pchange_icon").innerHTML = 'trending_up'
                    document.getElementById("pchange_icon").style.boxShadow = "0 0 0 0 white"
                    // document.getElementById("pchange_icon").style.color = 'red'
                } else {
                    // document.getElementById("pchange_migrants").innerHTML = response.data['p_change'].toString().concat("%");
                    document.getElementById("pchange_icon").innerHTML = 'trending_down'
                    document.getElementById("pchange_icon").style.boxShadow = "0 0 0 0 white"
                    // document.getElementById("pchange_icon").style.color = 'green'
                }

                // if pavg_age_change is greater than 1, make icon green & trending_up and vice versa
                if (response.data['pavg_age_change'] > 0) {
                    document.getElementById("page_change_icon").innerHTML = 'trending_up'
                    document.getElementById("page_change_icon").style.color = 'black'
                    document.getElementById("page_change_icon").style.boxShadow = "0 0 0 0 white"
                } else {
                    // document.getElementById("pavg_age_change").innerHTML = response.data['pavg_age_change'].toString().concat("%");
                    document.getElementById("page_change_icon").innerHTML = 'trending_down'
                    document.getElementById("page_change_icon").style.color = 'black'
                    document.getElementById("page_change_icon").style.boxShadow = "0 0 0 0 white"
                }

                // Update the status so the user knows everything is done
                document.getElementById("status").innerHTML = "Done."

                border_sector_breakout(response.data);

                make_corr_tables(response.data);

                update_top_ten(response.data);

            });

        });

}