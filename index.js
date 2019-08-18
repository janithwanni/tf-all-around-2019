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
	const model = await use.load();
	console.log("Loaded Model");
	console.log("Features and Target");
	console.log(await dataset.columnNames());
	const flatdataset = dataset.map(({xs, ys}) =>
      		{
        	return {xs:Object.values(xs), ys:Object.values(ys)};
	});
	console.log("Flattenned dataset");
	
	const embedDataset = await flatdataset.mapAsync(value => { return getEmbeddings(value,model);});
	console.log("Embeded Dataset");
	await embedDataset.take(3).forEachAsync(e=>console.log(e));
	
	
}
async function getEmbeddings(data,model){
	const embeds = await model.embed(data.xs);
	const embeds_array = await embeds.array();
	return new Promise((resolve,reject) =>{
		resolve({xs:embeds_array,ys:data.ys});
	});
	
}
function createModel(){
	const model = tf.sequential();
	model.add(tf.layers.dense({inputShape:[512],units:2,useBias:true}));
	model.add(tf.layers.dense({units: 1,useBias: true}));
	return model;
}
app();


