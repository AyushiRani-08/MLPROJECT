from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Fetching elements matching your exact student data types
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Clip score between 0 and 100 since it's an exam grade
        final_result = max(0, min(100, round(results[0], 2)))
        
        return render_template('index.html', results=final_result)

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000, debug=True)