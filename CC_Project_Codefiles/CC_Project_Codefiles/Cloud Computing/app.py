from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName('SentimentAnalysis').getOrCreate()

# Load your trained PipelineModel
model = PipelineModel.load("path/to/your/saved/model")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the request
    data = request.json
    review_text = data['text']

    # Create a DataFrame
    review_df = spark.createDataFrame([(review_text,)], ["clean_tweet"])

    # Predict sentiment
    prediction = model.transform(review_df)
    predicted_label_index = prediction.select("prediction").collect()[0]["prediction"]
    labels = model.stages[-2].labels
    predicted_label = labels[int(predicted_label_index)]

    # Send back the response
    return jsonify({'sentiment': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
