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
                    }
                }

axios.get('http://127.0.0.1:5000/get_border_data')

    .then(response => {


        var bsdivL = document.getElementById("bs-divL")

        var bs_title = document.getElementById("bs-title");
        bs_title.innerHTML = 'Number of migrants to each Southwest Border Sector';
    
        var bs_hist = document.createElement('canvas');
        bs_hist.id = 'bs-hist';
        bsdivL.appendChild(bs_hist); // append the left chart to the left canvas

        var ctx1 = document.getElementById('bs-hist').getContext('2d');
        var myChart1 = new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: response.data["bs_fractions_labels"],
                datasets: [{
                    label: 'Number of migrants',
                    data: response.data["bs_fractions_values"],
                    backgroundColor: 'rgba(70, 109, 29, 0.5)',
                    borderColor: 'rgba(70, 109, 29, 1)',
                    borderWidth: 1
                }]
            },
            options: bs_config
        });


    });



function border_sector_breakout(data) {

    var bsdivL = document.getElementById("bs-divL")

    // Remove the old canvas elements from the document
    var old_bs_canv = document.getElementById('bs-hist');
    console.log(old_bs_canv);
    if(old_bs_canv != null){
        console.log("removing old canvas!!!")
        bsdivL.removeChild(old_bs_canv);
    }

    var bs_title = document.getElementById("bs-title");
    bs_title.innerHTML = 'Number of migrants to each Southwest Border Sector';

    var bs_hist = document.createElement('canvas');
    bs_hist.id = 'bs-hist';
    bsdivL.appendChild(bs_hist); // append the left chart to the left canvas

    var ctx1 = document.getElementById('bs-hist').getContext('2d');
    var myChart1 = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: data["bs_fractions_labels"],
            datasets: [{
                label: 'Number of migrants',
                data: data["bs_fractions_values"],
                backgroundColor: 'rgba(70, 109, 29, 0.5)',
                borderColor: 'rgba(70, 109, 29, 1)',
                borderWidth: 1
            }]
        },
        options: bs_config
    });


}