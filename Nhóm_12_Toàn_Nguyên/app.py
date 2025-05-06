from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from clustering import perform_clustering
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload-form')
def upload_form():
    return render_template('index.html')

@app.route('/customers')
def customers():
    file_path = 'uploads/customers.csv'
    if not os.path.exists(file_path):
        return "Customer data not found. Please upload the file first."

    df = pd.read_csv(file_path)
    customers_data = df.to_dict(orient='records')
    return render_template('customers.html', customers=customers_data)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    file_path = 'uploads/' + file.filename
    file.save(file_path)

    n_clusters_input = request.form.get('n_clusters')
    n_clusters = int(n_clusters_input) if n_clusters_input.strip() else None

    cluster_results, plot_path = perform_clustering(file_path, n_clusters)

    # Return result page with clustering results and plot
    return render_template('result.html', cluster_results=cluster_results, plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
