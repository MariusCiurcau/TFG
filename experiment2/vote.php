<html>
 <head>
  <title>Prueba de PHP</title>
 </head>
 <body>
 <?php 
if ($_SERVER["REQUEST_METHOD"] == "GET") {
  // collect value of input field
  $image = $_GET['image'];
  $rating = $_GET['rating'];
  $feedback = $_GET['feedback'];
  $knowledge = $_GET['knowledge'];
  $role = $_GET['role'];

  $file = fopen('votesMulticlass.txt', 'a');
  
	if (flock ($file, LOCK_EX)) { // exclusive lock
  		fwrite($file, date("Y/m/d H:i").";".$_SERVER['REMOTE_ADDR'].";".$knowledge.";".$role.";".$image.";".$rating.";".$feedback.PHP_EOL);
  		fclose($file);
	}
}

?>
 </body>
</html>
