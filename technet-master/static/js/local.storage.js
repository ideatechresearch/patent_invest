// Function to save selected values for a specific group to localStorage
function saveSelectedValues(group) {
    var selectBoxes = document.querySelectorAll('.selectBox.' + group);

    selectBoxes.forEach(function (selectBox, index) {
        try {
            var selectedValue = selectBox.options[selectBox.selectedIndex].value;
            localStorage.setItem(group + "_selectedValue" + (index + 1), selectedValue);
        } catch (error) {
            console.error('Error saving selected value:',selectBox.name,error);
        }
    });
}

// Function to set select box values for a specific group from localStorage
function setSelectBoxValues(group) {
    var selectBoxes = document.querySelectorAll('.selectBox.' + group);

    selectBoxes.forEach(function (selectBox, index) {
        var selectedValue = localStorage.getItem(group + "_selectedValue" + (index + 1));
        if (selectedValue) {
            for (var i = 0; i < selectBox.options.length; i++) {
                if (selectBox.options[i].value === selectedValue) {
                    selectBox.selectedIndex = i;
                    break;
                }
            }
        }
    });
}

// Function to save selected value to localStorage
function saveSelectedValue(selectBox) {
    var selectedValue = selectBox.options[selectBox.selectedIndex].value;
    localStorage.setItem(selectBox.id, selectedValue);
}

// Function to set select box value from localStorage
function setSelectBoxValue(selectBox) {
    var selectedValue = localStorage.getItem(selectBox.id);
    if (selectedValue) {
        for (var i = 0; i < selectBox.options.length; i++) {
            if (selectBox.options[i].value === selectedValue) {
                selectBox.selectedIndex = i;
                break;
            }
        }
    }
}
