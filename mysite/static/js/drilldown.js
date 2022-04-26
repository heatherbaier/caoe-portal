var bs_config = {
    tooltips: {
        titleFontSize: 20,
        bodyFontSize: 18
    },
    maintainAspectRatio: false,
    legend: { 
        display: false 
    },
    title: {
        display: true,
        fontSize: 20
    },
    scales: {
            y: {
                beginAtZero: true
            },
            yAxes: [{
                ticks: {
                    fontSize: 20
                }
            }],
            xAxes: [{
                ticks: {
                    fontSize: 15
                }
            }]
    },
    annotation: {
        annotations: [
          {
            drawTime: "afterDatasetsDraw",
            type: "line",
            mode: "vertical",
            scaleID: "x-axis-0",
            value: 30000,
            borderWidth: 5,
            borderColor: "red",
            label: {
              content: "TODAY",
              enabled: true,
              position: "top"
            }
          }
        ]
      }
}


function make_perc_chart(counts, bins) {


    var plot1_div = document.getElementById("dd-plot1")
    plot1_div.style.height = "35%";


    var plot1_check = document.getElementById('plot1');

    console.log("plot1_check", plot1_check)

    if (plot1_check != null){
        console.log("removing old canvas!!!")
        plot1_div.removeChild(plot1_check);
    }

    var plot1 = document.createElement('canvas');

    plot1.id = 'plot1';
    plot1_div.appendChild(plot1); // append the left chart to the left canvas

    var ctx1 = document.getElementById('plot1').getContext('2d');
    var myChart1 = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: bins,
            datasets: [{
                label: 'Number of migrants',
                data: counts,
                backgroundColor: 'rgba(70, 109, 29, 0.5)',
                borderColor: 'rgba(70, 109, 29, 1)',
                borderWidth: 1
            }]
        },
        options: bs_config
    });

}




function drilldown(target) {

    console.log("in drill down!!")

    console.log(target.feature.properties.shapeName);

    document.getElementById("nav").style.width = "25%"
    document.getElementById("drill").style.display = "block"
    document.getElementById("drill").style.width = "25%"
    document.getElementById("data").style.width = "50%"
    document.getElementById("radio-layers").style.marginRight = "28%"

    document.getElementById("drill-muni").innerHTML = target.feature.properties.shapeName.concat(" Analytics");

    fetch('/drilldown', {

        // When the data gets POSTed back to Flask, it'll be in JSON format
        headers: {
            'Content-Type': 'application/json'
        },

        // Send the data back as a POST request
        method: 'POST',

        // Here's where you construct the JSON
        body: JSON.stringify({
            "drilldown_muni": target.feature.properties.shapeID
        })

        // ...and then send it off
        }).then(function (response) {
            return response.text();
        }).then(function (text) {

            // Parse data from Flask into JSON
            data = JSON.parse(text);

            // Update text on screen to say:
            // XX MUNI is in YY Percentile...
            document.getElementById("mig_perc").style.fontSize = "28px";
            document.getElementById("mig_perc").style.textAlign = "center";
            document.getElementById("mig_perc").innerHTML = target.feature.properties.shapeName + " is in the " + data['mig_perc'] + "st percentile of all municipalities in number of migrants."

            // Make the histogram chart with percentiles
            // TO-DO: ADD VERTICAL LINE FOR MUNI'S PLACE ON HISTOGRAM
            make_perc_chart(data['mig_hist_counts'], data['mig_hist_bins'])

            // Parent node id="drill"
            var parent = document.getElementById("drill");

            // Create the uncertainty header
            var unc_title = document.createElement("h3")
            unc_title.id = "unc_title"
            unc_title.innerHTML = "Uncertainty";

            // Create the text that says..
            // XX MUNI has a YY level of uncertainty
            var unc_level = document.createElement("h5")
            unc_level.id = "unc_level"
            unc_level.innerHTML = target.feature.properties.shapeName + " has a HIGH level of uncertainty"
            unc_level.style.fontSize = "28px";
            unc_level.textAlign = "center";

            var hr = document.createElement("hr")
            hr.id = "unc_hr"

            // Append the title, an hr and the uncertainity text to the document
            parent.appendChild(unc_title);
            parent.appendChild(hr);
            parent.appendChild(unc_level);

        })

}