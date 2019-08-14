const CSV_URL = "News_Final.csv"
async function app(){
	const dataset = tf.data.csv(CSV_URL);
	console.log("testing");
	console.log(dataset.columnNames());
}
app();
