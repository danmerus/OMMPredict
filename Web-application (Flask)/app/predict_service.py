from pathlib import Path
import numpy as np
import pickle

MODEL_PATH = Path(__file__).resolve().parent / "static" / "model.pt"

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def predict(relapse, vegfa634gg, vegfa634c, periods, tp53gg, mecho, vegfa936cc, first_symptom, kitlg80441cc, emergency_birth, vleft, fsh, vright):
    data = np.array([relapse, vegfa634gg, vegfa634c, periods, tp53gg, mecho,
                    vegfa936cc, first_symptom, kitlg80441cc, emergency_birth, vleft, fsh, vright])
    prediction = model.predict(data)
    if prediction:
        return True
    else:
        return False
