import time
from config import *
from models import *
from views import *


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        database_initialization_sequence()

    app.run(debug=True, host='0.0.0.0')
