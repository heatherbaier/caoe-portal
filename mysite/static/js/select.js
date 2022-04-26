function choose_var(elem) {


    var elem_id = elem.id;
    var parent = document.getElementById("cat-vars");
    var elemL = document.getElementById(elem.id + " L");
    var elemB = document.getElementById(elem.id + " B");

    var new_parent = document.getElementById("chosen-vars");

    // remove both elements from selection list
    parent.removeChild(elem);
    parent.removeChild(elemL);
    parent.removeChild(elemB);

    // create the minus icon and append to page
    var newVarI;
    newVarI = document.createElement("i");
    newVarI.classList.add("material-icons");
    newVarI.classList.add("remove");
    newVarI.id = elem.id;
    newVarI.innerHTML = "remove";
    newVarI.style.fontSize = "40px";
    newVarI.style.padding = "10px";
    newVarI.style.textAlign = "left";
    newVarI.style.marginLeft = "40px";
    newVarI.style.marginRight = "25px";
    // newVarI.style.paddingLeft = "40px";
    newVarI.style.backgroundColor = "gray";
    newVarI.style.boxShadow = "10px 5px 5px lightgray";
    newVarI.style.color = "white";
    newVarI.style.borderRadius = "8px";
    
    console.log(newVarI);
    new_parent.appendChild(newVarI)


    // remove class tage from label
    elemL.classList.remove("variable");

    // append label to new div
    new_parent.appendChild(elemL);


    var newInput;
    newInput = document.createElement("input");
    newInput.type = "number"
    newInput.value = "100"
    newInput.style.width = "100px"
    newInput.style.marginLeft = "40px";
    new_parent.appendChild(newInput);


    // line break
    var br = document.createElement("br");
    // br.classList.add("variable");
    new_parent.appendChild(br);
    
}


function change_var_options() {

    console.log("in here yo!!")
    var select = document.querySelector('#cats');
    console.log(select.value)

    fetch('/cat_select', {

        // When the data gets POSTed back to Flask, it'll be in JSON format
        headers: {
            'Content-Type': 'application/json'
        },

        // Send the data back as a POST request
        method: 'POST',

        // Here's where you construct the JSON
        body: JSON.stringify({
            "selected_cat": select.value
        })

        // ...and then send it off
        }).then(function (response) {
            return response.text();
        }).then(function (text) {

            data = JSON.parse(text);

            var vars = data['categories']
            var varsOnPage = document.getElementsByClassName("variable");
            var appendTo = document.getElementById("cat-vars");

            // console.log(data);
            // console.log("append to: ", appendTo)
            // console.log(varsOnPage);

            if (varsOnPage.length == 0) {

                console.log("here in length 0")

                var newVarI;
                var newVarL;
                for (var i = 0; i < vars.length; i++) {

                    // create the input and append to page
                    newVarI = document.createElement("i");
                    newVarI.classList.add("material-icons");
                    newVarI.classList.add("add");
                    newVarI.classList.add("variable");
                    newVarI.id = vars[i];
                    newVarI.innerHTML = "add";
                    newVarI.style.fontSize = "40px";
                    newVarI.style.padding = "10px";
                    newVarI.style.textAlign = "left";
                    newVarI.style.marginLeft = "40px";
                    newVarI.style.marginRight = "25px";
                    // newVarI.style.paddingLeft = "40px";
                    newVarI.style.backgroundColor = "gray";
                    newVarI.style.boxShadow = "10px 5px 5px lightgray";
                    newVarI.style.color = "white";
                    newVarI.style.borderRadius = "8px";
                    console.log(newVarI);
                    newVarI.setAttribute("onclick","choose_var(this)");
                    appendTo.appendChild(newVarI)

                    // create the label and append to page
                    newVarL = document.createElement("label");
                    newVarL.classList.add("variable");
                    newVarL.id  = vars[i] + " L";
                    newVarL.innerHTML = vars[i];
                    console.log(newVarL);
                    appendTo.appendChild(newVarL)

                    // line break
                    var br = document.createElement("br");
                    br.classList.add("variable");
                    br.id = vars[i] + " B";
                    appendTo.appendChild(br);
                
                }


            } else {

                // console.log(varsOnPage.length);


                var elements = document.getElementsByClassName("variable");
                while(elements.length > 0){
                    elements[0].parentNode.removeChild(elements[0]);
                }

                var newVarI;
                var newVarL;
                for (var i = 0; i < vars.length; i++) {

                    // create the input and append to page
                    newVarI = document.createElement("i");
                    newVarI.classList.add("material-icons");
                    newVarI.classList.add("add");
                    newVarI.classList.add("variable");
                    newVarI.id = vars[i];
                    newVarI.innerHTML = "add";
                    newVarI.style.fontSize = "40px";
                    newVarI.style.padding = "10px";
                    newVarI.style.textAlign = "left";
                    newVarI.style.marginLeft = "40px";
                    newVarI.style.marginRight = "25px";
                    // newVarI.style.paddingLeft = "40px";
                    newVarI.style.backgroundColor = "gray";
                    newVarI.style.boxShadow = "10px 5px 5px lightgray";
                    newVarI.style.color = "white";
                    newVarI.style.borderRadius = "8px";
                    console.log(newVarI);
                    newVarI.setAttribute("onclick","choose_var(this)");
                    appendTo.appendChild(newVarI)

                    // create the label and append to page
                    newVarL = document.createElement("label");
                    newVarL.classList.add("variable");
                    newVarL.id  = vars[i] + " L";
                    newVarL.innerHTML = vars[i];
                    console.log(newVarL);
                    appendTo.appendChild(newVarL)

                    // line break
                    var br = document.createElement("br");
                    br.classList.add("variable");
                    br.id = vars[i] + " B";
                    appendTo.appendChild(br);

                }




            }

            // document.getElementById("mig_perc").innerHTML = target.feature.properties.shapeName + " is in the " + data['mig_perc'] + "st percentile of all municipalities in number of migrants."
        })
    

}



