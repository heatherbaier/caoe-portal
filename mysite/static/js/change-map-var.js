// Function to color the polygons by total number of migrants
function total_mig_color(d) {

    return d > 5000 ? '#800026' :
           d > 2500  ? '#BD0026' :
           d > 500  ? '#E31A1C' :
           d > 250  ? '#FC4E2A' :
           d > 100   ? '#FD8D3C' :
           d > 50   ? '#FEB24C' :
           d > 10   ? '#FED976' :
                      '#FFEDA0';

}

// Function to style the polygons
function total_mig_style(feature) {
    return {
        fillColor: total_mig_color(feature.properties.num_migrants),
        weight: .2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.8
    };
}


// Function to color the polygons by percentage change in migrants
function perc_change_color(d) {

    return  d > 1               ? '#69B34C' : // positive change
            d > .1              ? '#ACB334' : // positive change
            d > .0000001        ? '#FAB733' :
            d > -.0000001       ? '#656565' : // zero change
            d > -.1             ? '#FF8E15' :
            d > -1              ? '#FF4E11' :
            d > -100000         ? '#FF0D0D' : // negative change
                                  '#808080' ; // other

}

// Function to style the polygons
function perc_change_style(feature) {
    return {
        fillColor: perc_change_color(feature.properties.num_migrants),
        weight: .2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.8
    };
}



// Function to color the polygons by change in migrants
function abs_change_color(d) {

    return  d > 10000           ? '#69B34C' : // positive change
            d > 100              ? '#ACB334' : // positive change
            d > .0000001         ? '#FAB733' :
            d > -.0000001        ? '#656565'   : // zero change
            d > -10              ? '#FF8E15' :
            d > -100             ? '#FF4E11' :
                                   '#FF0D0D' ; // negative change
                                //    '#808080' ; // other

}

// Function to style the polygons
function abs_change_style(feature) {
    return {
        fillColor: abs_change_color(feature.properties.num_migrants),
        weight: .2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.8
    };
}


var map_button_list = ["sum_num_intmig_button", "perc_migrants_button", "absolute_change_button", "perc_change_button"]


function change_map_var(variable) {

    console.log("here!!", variable)

    var to_send = {'variable': variable}

    window.map_var = variable

    axios.post('http://127.0.0.1:5000/update_map', to_send)

        .then(response => {

            // Hack for switching button colors because you suck at CSS
            for (var mb = 0; mb < map_button_list.length; mb++) {
                document.getElementById(map_button_list[mb]).style.backgroundColor = "#0E0C28";
                document.getElementById(map_button_list[mb]).style.color = "white";
                document.getElementById(map_button_list[mb]).style.fontWeight = "normal";
            }

            var var_button = document.getElementById(variable + "_button");
            // var_button.classList.toggle("active");
            var_button.style.backgroundColor = "white";
            var_button.style.color = "#0E0C28";
            var_button.style.fontWeight = "bold";

            console.log(var_button.classList);

            // Remove the current migration data layer from the map
            mymap.removeLayer(window.poly);
            mymap.removeControl(legend);

            console.log("removed legend");

            // console.log('')
            // getColor


            legend = L.control({position: 'bottomleft'});
            legend.onAdd = function (map) {

                var div = L.DomUtil.create('div', 'info legend');
                var labels = [];

                if (variable == "sum_num_intmig") {


                    var grades = [0, 10, 50, 100, 250, 500, 2500, 5000];
                    
                    for (var i = 0; i < grades.length; i++) {
                        console.log(i, total_mig_color(grades[i] + 1));
                        div.innerHTML +=
                            '<i style="background:' + total_mig_color(grades[i] + 1) + '"></i> ' +
                            grades[i] + (grades[i + 1] ? ' to ' + grades[i + 1] + '<br>' : '+');
                    }

                } else if (variable == "perc_migrants") {

                    var grades = [0, 0.02, .04, .06, .08, .1, .5, .8];
                    for (var i = 0; i < grades.length; i++) {
                        div.innerHTML +=
                            '<i style="background:' + getColor(grades[i] + .01) + '"></i> ' +
                            grades[i] + (grades[i + 1] ? ' to ' + grades[i + 1] + '<br>' : '+');
                    }

                } else if (variable == "absolute_change") {


                    var grades = [100005, 100000, 1000, .0000001, -.0000001, -10, -100, -100000];
                    var labels = ["> 1000", "1000 to 100", "100 to 0", "No change", "-1 to -10", "-10 to -100", "-100 to -1000", "< -1000"]
                    for (var i = 0; i < grades.length; i++) {
                        console.log(i, abs_change_color(grades[i]));

                        div.innerHTML +=
                        '<i style="background:' + abs_change_color(grades[i]) + '"></i> ' +
                        labels[i] + '<br>';

                    }

                } else {

                    var grades = [2, 1, .1, .0000001, -.0000001, -.1, -1, -2];
                    var labels = ["> 100%", "100% to 1%", "1% to 0", "No change", "0 to -.1%", "-.1% to -1%", "-1% to -100%", "< -100%"]
                    for (var i = 0; i < grades.length; i++) {
                        console.log(i, getColor(grades[i]));

                        div.innerHTML +=
                        '<i style="background:' + perc_change_color(grades[i]) + '"></i> ' +
                        labels[i] + '<br>';

                    }

                }

                return div;

            };

            legend.addTo(mymap);

            window.poly = [];

            // Remove the drawn polygons from the map and re-initalize the drawnPolys group as empty
            mymap.removeLayer(window.drawnPolys);
            window.drawnPolys = new L.featureGroup().addTo(mymap);

            // Convert the new migration data into a leaflet geoJSON

            if (variable == "sum_num_intmig") {
                var polys = L.geoJSON(response.data, {style: total_mig_style, onEachFeature: onEachFeature})//.addTo(mymap);
            } else if (variable == "perc_migrants") {
                var polys = L.geoJSON(response.data, {style: polygon_style, onEachFeature: onEachFeature})//.addTo(mymap);
            } else if (variable == "absolute_change") {
                var polys = L.geoJSON(response.data, {style: abs_change_style, onEachFeature: onEachFeature})//.addTo(mymap);
            } else {
                var polys = L.geoJSON(response.data, {style: perc_change_style, onEachFeature: onEachFeature})//.addTo(mymap);
            }

            console.log("POLYS: ", polys.getLayers().length)

            // Update the global window.poly variable & add it to the map
            window.poly = polys;
            window.poly.addTo(mymap);

            window.selected_polys = [];

            // Zoom the map back out to all of Mexico                
            mymap.setView(new L.LatLng(23.6345, -102.5528), 6);

        })

}



