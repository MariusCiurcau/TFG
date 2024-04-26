
{
var rating = NaN;
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

	$("#sendBtn").on('click', function(){
	    knowledge = localStorage.getItem('knowledge');
		role = localStorage.getItem('role');
	    var image = $(this).attr('data-img').trim();
        var feedback = $("#feedbackInput").val().trim();

        if (isNaN(rating)) {
            window.alert('Please provide a rating for this explanation');
            return;
		};

		var radioButtons = document.getElementsByName("rating");
        for (var i = 0; i < radioButtons.length; i++) {
            radioButtons[i].checked = false;
        }

        $.get("vote.php",
		  {
		    image: image,
		    rating: rating,
			feedback: feedback,
			knowledge: knowledge,
			role: role
		  },
		  function(data, status){
		  });
		$("#feedbackInput").val("");
        rating = NaN;
    	X.showImg();
    });
};

function updateKnowledge(value) {
    knowledge = value;
}

function updateRole(value) {
    role = value;
}

function updateRating(value) {
    rating = value;
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

        $.get("images/images.csv",function(txt){
		var lines = txt.split("\n");
        for (var i = 0, len = lines.length; i < len; i+=1) {
			split = lines[i].split(";");
			realClass = split[1];
			explanation = split[2];
            X.addClass(realClass, explanation);
        }
		X.showImg();
		}); 
	}); 
}


var X = new function() {
 this.images =[];
 this.img = 0;
 this.realClasses = [];
 this.explanations = [];

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

	$("#realClass").text(this.realClasses[this.img]);

	$("#explanation").text(this.explanations[this.img]);

	var imageName = this.images[this.img];
    $("#imgOrig").attr('src','images/original/'+imageName);

    $("#sendBtn").attr('data-img', imageName);
	
	this.img++;
  };

  this.addImage = function(img){
  	this.images.push(img);
  };

  this.addClass = function(realClass, explanation)
  {
	this.realClasses.push(realClass);
	this.explanations.push(explanation);
  }
};

}
