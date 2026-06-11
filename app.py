from flask import Flask, jsonify, render_template
import os
import csv
import glob

app = Flask(__name__)

def get_latest_session():
    files = glob.glob("data/session_*.csv")
    if not files:
        return []
    latest = max(files, key=os.path.getmtime)
    rows = []
    with open(latest, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def get_all_sessions():
    files = sorted(glob.glob("data/session_*.csv"), key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/session")
def session_data():
    rows = get_latest_session()
    return jsonify(rows)

@app.route("/api/sessions")
def all_sessions():
    return jsonify(get_all_sessions())

if __name__ == "__main__":
    app.run(debug=True, port=5000)
