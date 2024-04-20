
{
var clase = NaN;
var currentImageName;

run = function(){
	loadData();
};

function sendVote(value) {
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
        clase: clase
    },
    function(data, status){
    });

    X.showImg();
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
		window.alert('Thank you!!!. You have finished all');
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
