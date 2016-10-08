from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route("/<name>")
def index(name):
    #return '<h>Hello world!</h>'
    return render_template('index3.html')


if __name__ == "__main__":
    app.run(host='192.168.56.101')
