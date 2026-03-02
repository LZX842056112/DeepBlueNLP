# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/5 20:59
Create User : 19410
Desc : xxx
"""

from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!--->NLP+CV'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
