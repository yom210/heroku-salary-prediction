from logging import debug

from flask import Flask , render_template , request
import joblib
app=Flask(__name__)

#load model
model=joblib.load("hiring_model.pkl")

@app.route('/')
def welcome():
    return render_template('base.html')

@app.route('/predict' , methods=["POST"])
def predict():
    exp=request.form.get("experience")
    score=request.form.get("test_score")
    interview_score=request.form.get("Interview_score")

    prediction=model.predict([[int(exp),int(score),int(interview_score)]])

    output = round(prediction[0],2)

    return render_template("base.html",prediction_text= f"Employee salary will be $ {output}")


    return "YOUR FORM IS SUBMITTED"

# @app.route('/contact')
# def cont():
#     return "welcome to contact page"

# @app.route('/feedback')
# def feed():
#     return "welcome to feedback page"

# @app.route('/help')
# def help():
#     return "welcome to help page"

# @app.route('/data science')
# def ds():
#     return "welcome to data science page"
if __name__=="__main__":
    app.run(debug=True)