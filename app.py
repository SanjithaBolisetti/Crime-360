from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from India import districtwise, data01to14, multi_crime_plot, plot_map_any, states_to_ui
from predictions import children_prediction, children_crimes, children_states, children_years, pred_crime_plot, women_prediction, women_crimes, women_states, women_years

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.debug = True

db = SQLAlchemy(app)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    state = db.Column(db.String(200), nullable=False)
    district = db.Column(db.String(200), nullable=False)
    crime = db.Column(db.String(200), nullable=False)
    number = db.Column(db.Integer)
    year = db.Column(db.Integer)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Task {self.id}>'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/select')
def select():
    return render_template('select.html', data=[{'name': i} for i in states_to_ui])

@app.route('/selected', methods=['GET', 'POST'])
def selected():
    district_selected = request.form.get('district')
    crime_selected = request.form.get('crime')
    districtwise(crime_selected, district_selected)
    return render_template('display.html')

@app.route('/multi_select')
def multi_select():
    return render_template('multiselect.html', data=[{'name': i} for i in data01to14['DISTRICT'].unique()], crimes=[{'name': i} for i in data01to14])

@app.route('/multi_selected', methods=['GET', 'POST'])
def multi_selected():
    district_selected = request.form.get('district')
    crimes_selected = request.form.getlist('crime')
    multi_crime_plot(district_selected, crimes_selected)
    return render_template('display_multi.html')

@app.route('/select_any')
def select_any():
    return render_template('select_any.html', crimes=[{'name': i} for i in data01to14])

@app.route('/selected_any', methods=['GET', 'POST'])
def selected_any():
    crime_selected = request.form.getlist('crime')
    plot_map_any(crime_selected)
    return render_template('display_any.html')

@app.route('/plotly_children')
def plotly_children():
    return render_template('plotly_children.html')

@app.route('/children_select')
def children_select():
    return render_template('select_pred_children.html', states=[{'name': i} for i in children_states], crimes=[{'name': i} for i in children_crimes], years=[{'name': i} for i in children_years])

@app.route('/children', methods=['POST', 'GET'])
def children():
    year = request.form.get("year")
    crime = request.form.get("crime")
    state = request.form.get("state")
    y, years, output = children_prediction(state, year, crime)
    pred_crime_plot(state, crime, y, years)
    return render_template('display_multi.html')

@app.route('/women_select')
def women_select():
    return render_template('select_pred_women.html', states=[{'name': i} for i in women_states], crimes=[{'name': i} for i in women_crimes], years=[{'name': i} for i in women_years])

@app.route('/women', methods=['POST', 'GET'])
def women():
    year = request.form.get("year")
    crime = request.form.get("crime")
    state = request.form.get("state")
    y, years, output = women_prediction(state, year, crime)
    pred_crime_plot(state, crime, y, years)
    return render_template('display_multi.html')

@app.route('/insert', methods=['POST', 'GET'])
def insert():
    if request.method == 'POST':
        form_year = request.form['year']
        form_state = request.form['state']
        form_district = request.form['district']
        form_crime = request.form['crime']
        form_number = request.form['number']
        new_data = Data(state=form_state, district=form_district, crime=form_crime, number=form_number, year=form_year)
        if int(form_year) <= 2014:
            return 'Data before 2014 cannot be inserted'
        try:
            db.session.add(new_data)
            db.session.commit()
            return redirect('/insert')
        except:
            return 'Error adding data'
    else:
        data_sorted = Data.query.all()
        return render_template('insert.html', data_sorted=data_sorted)

@app.route('/delete/<int:id>')
def delete(id):
    entry_to_delete = Data.query.get_or_404(id)
    try:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return redirect('/insert')
    except:
        return 'There was an issue deleting that entry'

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)
