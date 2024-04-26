
{
var clase = NaN;
var currentImageName;
var knowledge = NaN;
var role;

run = function(){
    $("#startBtn").on('click', function(){
        if (isNaN(knowledge)) {
            window.alert('Please select your expertise level');
            return;
		};

		if (!role || role.length === 0) {
            window.alert('Please select your role');
            return;
		};

		localStorage.setItem('knowledge', knowledge);
		localStorage.setItem('role', role);

		window.location.href = "experiment.html";
    });
	loadData();
};

function sendVote(value) {
    knowledge = localStorage.getItem('knowledge');
	role = localStorage.getItem('role');
    clase = value;

    if (X.img >= X.images.length) {
        var buttons = document.querySelectorAll("button");
        buttons.forEach(function(button) {
          button.onclick = null;
        });
    }

    $.get("vote.php",
    {
        image: currentImageName,
        clase: clase,
        knowledge: knowledge,
        role: role
    },
    function(data, status){
    });

    X.showImg();
}

function updateKnowledge(value) {
    knowledge = value;
}

function updateRole(value) {
    role = value;
}

loadData = function(){
	$.get("images/images.txt",function(txt){
		var lines = txt.split("\n");
        for (var i = 0, len = lines.length; i < len; i++) {
            if (lines[i].trim() === "") {
                continue;
            }
            X.addImage(lines[i].split(";")[0]);
        }
	X.showImg();
	});
}


var X = new function() {
 this.images =[];
 this.img = 0;

 this.showImg = function()
 {
	var totalImages = this.images.length;
	console.log(this.img+" -->"+totalImages);

	valeur = Math.floor(this.img*100/totalImages);
	$('#globalprogress').css('width', valeur+'%').attr('aria-valuenow', valeur); 
	$('#globalprogress').html(valeur+'%');

	if(this.img>=totalImages) {
	    $(".clickable").off('click');
		window.alert('Thank you!');
		return;
	}

	var imageName = this.images[this.img];
    $("#imgOrig").attr('src','images/original/'+imageName);

    currentImageName = imageName;
    console.log(currentImageName);
	
	this.img++;
  };

  this.addImage = function(img){
  	this.images.push(img);
  };

};

}
