function update_top_ten(data) {

    document.getElementById('bs-divR').style.display = "block";
    document.getElementById('top-ten-title').style.display = "block";

    for (i = 0; i < 10; ++i) {

        cur_id = "m-".concat(i.toString())
        console.log(cur_id);
        document.getElementById(cur_id).innerHTML = data.top_munis[i]

        cur_p = "c-".concat(i.toString())
        console.log(cur_p);
        document.getElementById(cur_p).innerHTML = data.top_changes[i].toString().concat(" migrants")

        cur_id = "mb-".concat(i.toString())
        console.log(cur_id);
        document.getElementById(cur_id).innerHTML = data.bottom_munis[i]

        cur_p = "cb-".concat(i.toString())
        console.log(cur_p);
        document.getElementById(cur_p).innerHTML = data.bottom_changes[i].toString().concat(" migrants")


    }

}