# Importing essential libraries and modules
from flask import Flask, render_template, request
import numpy as np
import pickle
from warnings import filterwarnings
filterwarnings('ignore')

################### LOADING TRAINED MODELS ##########################
###########  YIELD PREDICTION #################################
forest = pickle.load(open('models/yield1.pkl', 'rb'))  # yield
###########  PRICE PREDICTION #################################
cp = pickle.load(open('models/forest.pkl', 'rb'))  # price
###########  FERTILIZER PREDICTION #################################
model = pickle.load(open('models/classifier.pkl','rb'))
ferti = pickle.load(open('models/fertilizer.pkl','rb'))
###########  CROP RECOMMENDATION MODEL #################################
cr = pickle.load(open('models/RandomForest.pkl', 'rb'))


# ################################  FLASK APP ################################


app = Flask(__name__)

############################################## RENDERING HOME PAGE #################################################################


@ app.route('/')
def home():
    title = 'Crop harvest'
    return render_template('index.html', title=title)



############################################## CROP RECOMMENDATION CODES #################################################################


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop Recommendation'
    return render_template('crop.html', title=title)

@ app.route('/crop_predict', methods=['POST'])
def crop_predict():
    title = 'Crop Recommended'

    if request.method == 'POST':
        N = request.form['nitrogen']
        P = request.form['phosphorous']
        K = request.form['pottasium']
        ph = request.form['ph']
        rainfall = request.form['rainfall']
        hum = request.form['hum']
        temp = request.form['temp']

        data = np.array([[N, P, K, temp, hum, ph, rainfall]])
        my_prediction = cr.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction, title=title)


############################################## YIELD PREDICITON CODES #################################################################

@ app.route('/yeild')
def yeild():
    title = 'crop yeild prediction'
    return render_template('crop_yeild.html', title=title)

# render disease prediction input page

@ app.route('/yeild-predict', methods=['POST'])
def yeild_predict():
    title = 'yeild predicted'

    if request.method == 'POST':
        state = request.form['stt']
        district = request.form['city']
        year = request.form['year']
        season = request.form['season']
        crop = request.form['crop']
        Temperature = request.form['Temperature']
        humidity = request.form['humidity']
        soilmoisture = request.form['soilmoisture']
        area = request.form['area']

        out_1 = forest.predict([[float(state),
                                 float(district),
                                 float(year),
                                 float(season),
                                 float(crop),
                                 float(Temperature),
                                 float(humidity),
                                 float(soilmoisture),
                                 float(area)]])
        print("the yield is --->   {}    tons".format(out_1[0]))
        out_yield="{:.2f}".format(out_1[0])


        return render_template('yeild_prediction.html', prediction=out_yield, title=title)

    return render_template('try_again.html', title=title)


############################################## FERTILIZER PREDICITON CODES #################################################################

@app.route('/crop_fer', methods=['GET', 'POST'])
def crop_fer():
    # return "this is crop prediction page"
    title = 'crop Fertilizer'
    return render_template('fer.html', title=title)

@app.route('/fer_predict',methods=['POST'])
def fer_predict():
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    input = [float(temp),float(humi),float(mois),float(soil),float(crop),float(nitro),float(pota),float(phosp)]

    res = ferti.classes_[model.predict([input])]

    return render_template('fer_predict.html',res = res[0])


############################################## PRICE PREDICITON CODES #################################################################


@app.route('/crop_price', methods=['GET', 'POST'])
def crop_price():
    # return "this is crop prediction page"
    title = 'crop price'
    return render_template('crop_price.html', title=title)




@ app.route('/price_predict', methods=['POST'])
def price_predict():
    title = 'price Suggestion'
    if request.method == 'POST':
        state = int(request.form['stt'])
        district = int(request.form['city'])
        year = int(request.form['year'])
        season = int(request.form['season'])
        crop = int(request.form['crop'])

        p_result = cp.predict([[float(state),
                                float(district),
                                float(year),
                                float(season),
                                float(crop)]])[0]

        return render_template('price_prediction.html', title=title, p_result=p_result)
    return render_template('try_again.html', title=title)


############################################## Local Host Link CODE #################################################################

if __name__ == '__main__':
    app.run(debug=True)
