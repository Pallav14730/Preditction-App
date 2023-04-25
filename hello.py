from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/')
def navbar():
    return "<h1>Hello world</h1>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    # app.run(debug=True)
