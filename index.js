const CSV_URL = "https://raw.githack.com/janithwanni/tf-all-around-2019/master/News_Extract.csv";
const API_KEY = "fdc7167d66b548568f9eb32bc54cbb33";
const url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey='+API_KEY;
const prediction_element = document.getElementById('news_prediction');
const table_body_element = document.getElementById('tableBody');
const article_stream_length = 20;
const sample_size = 40;
const print_size = 1;
const number_of_layers_in_model = 2;
const train_slice = 30;
const batchSize = 1;
const epochs = 1;

async function app(){
	const dataset = tf.data.csv(CSV_URL,{
		columnConfigs:{
			"Title":{
				required: true
			},
			"SentimentTitle":{
				isLabel:true
			}
		},
		configuredColumnsOnly: true
	});
	console.log("Loading Encoding Model");
	const use_model = await use.load();
	console.log("Loaded Encoding Model");

	console.log("Features and Target");
	console.log(await dataset.columnNames());
	//await dataset.take(print_size).forEachAsync(e => console.log(e));
	
	console.log("Flattening Dataset");
	const flatdataset = dataset.map(({xs, ys}) =>
      		{
        	return {xs:Object.values(xs), ys:Object.values(ys)};
	});
	console.log("Flattenned dataset");
	
	console.log("Visualizing Sentiment Scores");
	const scatterplot_surface = {name:"Plot of Sentiment Scores",tab:"Data Visualization"};
	flatdataset.take(sample_size).toArray().then(slice_dataset => {
		console.log("Converted Dataset to Array");
		let index = -1;
		let series1 = slice_dataset.map(value => {
			index+=1;
			return {x:index, y:value.ys[0]};
		});
		const series = ['Sentiment Score of Title'];
		const data = { values: [series1], series }
		tfvis.render.scatterplot(scatterplot_surface, data,{xLabel:"Index",yLabel:"Score",yAxisDomain:[-1,1]});
		console.log("Visualized Sentiment Scores");
	});
	
	//await flatdataset.take(print_size).forEachAsync(e => console.log(e));
	const embedDataset = await flatdataset.mapAsync(value => { return getEmbeddings(value,use_model);});
	console.log("Embeded Dataset");
	//await embedDataset.take(print_size).forEachAsync(e=>console.log(e));
	
	console.log("Visualizing Embeddings with Sentiment Scores");
	//TODO: Add this
	console.log("Visualized Embeddings with Sentiment Scores")

	console.log("Setting batch size for dataset");
	const batchDataset = embedDataset.take(train_slice).batch(batchSize);
	console.log("Set batch size. NO MORE PIPELINING");
	//await batchDataset.take(print_size).forEachAsync(e=>console.log(e));


	console.log("Creating Model");
	const our_model = createModel();
	console.log("Created Model");
	our_model.summary();
	console.log("Visualizing Model Layer Summary")
	for(var i = 0;i<number_of_layers_in_model;i++){
		let layer_surface = { name: 'Layer '+i+' Summary', tab: 'Model Inspection'};
		tfvis.show.layer(layer_surface, our_model.getLayer(undefined, (i)));
	}
	console.log("Visualized Model Layer Summary")

	console.log("Training Dataset");

	console.log("Compiling Model");
	our_model.compile({
		optimizer: tf.train.adam(),
		loss: tf.losses.meanSquaredError,
		metrics: ['mse'],
	  });
	console.log("Compiled Model");

	const history_surface = {name: "Model Performance", tab: "Model Performance"};
	const history = [];
	console.log("Fitting Dataset");
	await our_model.fitDataset(batchDataset,{
		epochs: epochs,
		callbacks:{
			onEpochBegin:(epoch,log) => {
				console.log("Epoch "+epoch);
			},
			onBatchBegin:(batch,log) => {
				console.log("Batch "+batch);
			},
			onEpochEnd: (epoch,log) => {
				history.push(log);
				console.log("Epoch End "+epoch);
				tfvis.show.history(history_surface, history, ['mse',]);
			}
		}
	});
	console.log("Fitted Dataset");

	console.log("Trained Dataset");
	
	prediction_element.style['display'] = 'block';

	var req = new Request(url);
	const response = await fetch(req);
	console.log(response);
	const rjson = await response.json();
	console.log(rjson);
	const articles = rjson.articles;
	let titles = [];
	const elem = document.getElementById('titles');
	articles.forEach(article =>{
		let title_token_arr = article.title.split("-")
		title_token_arr = title_token_arr.slice(0,title_token_arr.length-1);
		let title = title_token_arr.join(" ");
		titles.push(title);	
	});
	embed_titles = await use_model.embed(titles);
	embed_titles.print(true);
	console.log(embed_titles.array());
	embed_titles_arr = await embed_titles.array()[0];
	for(var i = 0;i<article_stream_length;i++){
		score = our_model.predict(tf.tensor1d(embed_titles_arr[i]));
		elem.innerHTML += "<tr> <td>"+title + "</td>" + "<td>"+score+"</td>";
	}
	
	

}

async function getEmbeddings(data,model){
	const embeds = await model.embed(data.xs);
	const embeds_array = await embeds.array();
	embeds.dispose();
	return {xs:tf.tensor1d(embeds_array[0]),ys:tf.tensor1d(data.ys)};
	// return new Promise((resolve,reject) =>{
	// 	resolve({xs:embeds_array,ys:data.ys});
	// });
}
function createModel(){
	const model = tf.sequential();
	model.add(tf.layers.dense({inputShape:[batchSize,512],units:8}));
	model.add(tf.layers.dense({units: 1}));
	return model;
}
//FUNCTION TO PUT CONSOLE LOG TO HTML PAGE
//SOURCE: https://stackoverflow.com/a/35449256/1941842
(function () {
    var old = console.log;
    var logger = document.getElementById('log');
    console.log = function () {
      for (var i = 0; i < arguments.length; i++) {
        if (typeof arguments[i] == 'object') {
            logger.innerHTML += (JSON && JSON.stringify ? JSON.stringify(arguments[i], undefined, 2) : arguments[i]) + '<br />';
        } else {
            logger.innerHTML += arguments[i] + '<br />';
        }
      }
    }
})();

app();



