#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)

activate_this = 'venv/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

from main import app as application