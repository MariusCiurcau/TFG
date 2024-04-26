
{
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
    $(".clickable").on('click', function(){
		knowledge = localStorage.getItem('knowledge');
		role = localStorage.getItem('role');
		
    	var method = $(this).attr('data-xm').trim();
    	var image = $(this).attr('data-img').trim();

    	if(method == null) return;
    	if(image == null) return;

    	$.get("vote.php",
		  {
		    image: image,
		    method: method,
			knowledge: knowledge,
			role: role
		  },
		  function(data, status){
		  });
    	X.showImg();
    });

};

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

        $.get("images/images.csv",function(txt){
		var lines = txt.split("\n");
        for (var i = 0, len = lines.length; i < len; i+=1) {
			split = lines[i].split(";");
			realClass = split[1];
            X.addClass(realClass);
        }
		X.showImg();
		}); 
	}); 
}


var X = new function() {
 this.images =[];
 this.img = 0;
 this.realClasses = [];

 this.showImg = function()
 {
	folders = ['two_class_gradcam', 'three_class_gradcam', 'vargrad', 'saliency'];
	folders.sort(function() {
			return Math.random() - 0.5;
		});
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

	var imageName = this.images[this.img];
    $("#imgOrig").attr('src','images/original/'+imageName);


	$("#img1").attr('src','images/'+folders[0]+"/"+imageName);
	$("#img2").attr('src','images/'+folders[1]+"/"+imageName);
	$("#img3").attr('src','images/'+folders[2]+"/"+imageName);
	$("#img4").attr('src','images/'+folders[3]+"/"+imageName);

	$("#img1").attr('data-xm', folders[0]);
	$("#img2").attr('data-xm', folders[1]);
	$("#img3").attr('data-xm', folders[2]);
	$("#img4").attr('data-xm', folders[3]);

	$("#img1").attr('data-img', imageName);
	$("#img2").attr('data-img', imageName);
	$("#img3").attr('data-img', imageName);
	$("#img4").attr('data-img', imageName);

	
	this.img++;
  };

  this.addImage = function(img){
  	this.images.push(img);
  };

  this.addClass = function(realClass)
  {
	this.realClasses.push(realClass);
  }
};

}
