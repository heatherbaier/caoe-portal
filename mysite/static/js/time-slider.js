function change_year(direc) {

    var cur_year = parseInt(document.getElementById("ts-year").innerHTML)

    if (direc == "increase") {
        cur_year += 5
    } else {
        cur_year -= 5
    }

    if (cur_year == 2010) {
        document.getElementById("lc").style.visibility = "hidden";
        document.getElementById("rc").style.visibility = "visible";
    } else {
        document.getElementById("rc").style.visibility = "hidden";
        document.getElementById("lc").style.visibility = "visible";
    }

    document.getElementById("ts-year").innerHTML = cur_year

    // if (window.map_var == null) {
    //     window.map_var = "perc_migrants"
    // }

    // var to_send = {'year': document.getElementById("ts-year").innerHTML, 'var': window.map_var}

    // axios.post('http://127.0.0.1:5000/change_year', to_send)

    //     .then(response => {

    //         var variable = window.map_var

    //         // Remove the current migration data layer from the map
    //         mymap.removeLayer(window.poly);
    //         mymap.removeControl(legend);

    //         console.log("removed legend");

    //         // console.log('')
    //         // getColor


    //         legend = L.control({position: 'bottomleft'});
    //         legend.onAdd = function (map) {

    //             var div = L.DomUtil.create('div', 'info legend');
    //             var labels = [];

    //             if (variable == "sum_num_intmig") {


    //                 var grades = [0, 10, 50, 100, 250, 500, 2500, 5000];
                    
    //                 for (var i = 0; i < grades.length; i++) {
    //                     console.log(i, total_mig_color(grades[i] + 1));
    //                     div.innerHTML +=
    //                         '<i style="background:' + total_mig_color(grades[i] + 1) + '"></i> ' +
    //                         grades[i] + (grades[i + 1] ? ' to ' + grades[i + 1] + '<br>' : '+');
    //                 }

    //             } else if (variable == "perc_migrants") {

    //                 var grades = [0, 0.02, .04, .06, .08, .1, .5, .8];
    //                 for (var i = 0; i < grades.length; i++) {
    //                     div.innerHTML +=
    //                         '<i style="background:' + getColor(grades[i] + .01) + '"></i> ' +
    //                         grades[i] + (grades[i + 1] ? ' to ' + grades[i + 1] + '<br>' : '+');
    //                 }

    //             } else if (variable == "absolute_change") {


    //                 var grades = [100005, 100000, 1000, .0000001, -.0000001, -10, -100, -100000];
    //                 var labels = ["> 1000", "1000 to 100", "100 to 0", "No change", "-1 to -10", "-10 to -100", "-100 to -1000", "< -1000"]
    //                 for (var i = 0; i < grades.length; i++) {
    //                     console.log(i, abs_change_color(grades[i]));

    //                     div.innerHTML +=
    //                     '<i style="background:' + abs_change_color(grades[i]) + '"></i> ' +
    //                     labels[i] + '<br>';

    //                 }

    //             } else {

    //                 var grades = [2, 1, .1, .0000001, -.0000001, -.1, -1, -2];
    //                 var labels = ["> 100%", "100% to 1%", "1% to 0", "No change", "0 to -.1%", "-.1% to -1%", "-1% to -100%", "< -100%"]
    //                 for (var i = 0; i < grades.length; i++) {
    //                     console.log(i, getColor(grades[i]));

    //                     div.innerHTML +=
    //                     '<i style="background:' + perc_change_color(grades[i]) + '"></i> ' +
    //                     labels[i] + '<br>';

    //                 }

    //             }

    //             return div;

    //         };

    //         legend.addTo(mymap);

    //         window.poly = [];

    //         // Remove the drawn polygons from the map and re-initalize the drawnPolys group as empty
    //         mymap.removeLayer(window.drawnPolys);
    //         window.drawnPolys = new L.featureGroup().addTo(mymap);

    //         // Convert the new migration data into a leaflet geoJSON

    //         var polys = L.geoJSON(response.data, {style: total_mig_style, onEachFeature: onEachFeature})//.addTo(mymap);

    //         console.log("POLYS: ", polys.getLayers().length)

    //         // Update the global window.poly variable & add it to the map
    //         window.poly = polys;
    //         window.poly.addTo(mymap);

    //         window.selected_polys = [];

    //         // Zoom the map back out to all of Mexico                
    //         mymap.setView(new L.LatLng(23.6345, -102.5528), 6);


    //     })




}