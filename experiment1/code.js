
{
run = function(){
	loadData();
    $(".clickable").on('click', function(){
    	var method = $(this).attr('data-xm').trim();
    	var image = $(this).attr('data-img').trim();

    	if(method == null) return;
    	if(image == null) return;

    	$.get("vote.php",
		  {
		    image: image,
		    method: method
		  },
		  function(data, status){
		  });
    	X.showImg();
    });
};

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
			console.log("split " + split)
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
	folders = ['two_class_gradcam', 'three_class_gradcam', 'xplique'];
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

	$("#img1").attr('data-xm', folders[0]);
	$("#img2").attr('data-xm', folders[1]);
	$("#img3").attr('data-xm', folders[2]);

	$("#img1").attr('data-img', imageName);
	$("#img2").attr('data-img', imageName);
	$("#img3").attr('data-img', imageName);

	
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
