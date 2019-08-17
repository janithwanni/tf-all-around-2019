const CSV_URL = "https://raw.githack.com/janithwanni/tf-all-around-2019/master/News_Extract.csv";
//const CSV_URL = "News_Final.csv"
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
	console.log("testing");
	console.log(await dataset.columnNames());
	await dataset.take(3).forEachAsync(e=>console.log(e));
	console.log("flat dataset");
	const flatdataset = dataset.map(({xs, ys}) =>
      		{
        	return {xs:Object.values(xs), ys:Object.values(ys)};
		});
	await flatdataset.take(3).forEachAsync(e=>console.log(e));

	const inputs = flatdataset.map(values => { return values.xs;});
	//const labels = flatdataset.map(values => { return values.ys;});
	console.log(inputs);
	//const labelsDataset = tf.data.array(labels);
	
	const use_model = await use.load();
	//const embeddings = await getEmbeddings(inputs,use_model);
	//console.log("embeddings made with shape");
	
	//console.log(embeddings.print(true));
	//const inputsDataset = tf.data.array(embeddings);
	//console.log(inputsDataset);
	// const Dataset = tf.data.zip(inputsDataset,labelsDataset);
	// await Dataset.take(3).forEachAsync(e=>console.log(e));
	
}
async function getEmbeddings(inputs,model){
	return await model.embed(inputs).then(embeddings => {
		return new Promise((resolve,reject) => {
			resolve(embeddings);
		});
	});
}
function createModel(){
	const model = tf.sequential();
	model.add(tf.layers.dense({inputShape:[512],units:2,useBias:true}));
	model.add(tf.layers.dense({units: 1,useBias: true}));
	return model;
}
app();
