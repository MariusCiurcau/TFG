<html>
 <head>
  <title>Prueba de PHP</title>
 </head>
 <body>
 <?php 
if ($_SERVER["REQUEST_METHOD"] == "GET") {
  // collect value of input field
  $image = $_GET['image'];
  $clase = $_GET['clase'];
  $knowledge = $_GET['knowledge'];
  $role = $_GET['role'];

  $file = fopen('votesMulticlass.txt', 'a');
  
	if (flock ($file, LOCK_EX)) { // exclusive lock
  		fwrite($file, date("Y/m/d").";".$_SERVER['REMOTE_ADDR'].";".$knowledge.";".$role.";".$image.";".$clase.PHP_EOL);
  		fclose($file);
	}
}

?>
 </body>
</html>
