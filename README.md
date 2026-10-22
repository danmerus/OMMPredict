# Fast start OMMPredict:

docker-compose up --build

# Interface:

![main](https://github.com/glkhv/OMMPredict/blob/main/Screenshots/main.png)
![create](https://github.com/glkhv/OMMPredict/blob/main/Screenshots/create.png)
![search](https://github.com/glkhv/OMMPredict/blob/main/Screenshots/search.png)
![sidepage-top](https://github.com/glkhv/OMMPredict/blob/main/Screenshots/sidepage-top.png)
![sidepage-down](https://github.com/glkhv/OMMPredict/blob/main/Screenshots/sidepage-down.png)

# Streamlit app (`Web-application (Flask)/app/streamlit_app.py`)

Required Streamlit secrets:

```toml
[supabase]
url = "https://<project>.supabase.co"
SUPABASE_SERVICE_KEY = "<service role key>"

[auth]
password = "<shared password for the app>"   # the app stays locked until this is set
```

Units: ovary volumes are entered and stored in cm³. The baseline CSV (`app/data/baseline_examples.csv`) stores
length × width × height in mm and is converted with the ellipsoid formula (× 0.523 / 1000) when training.

Tests: `python -m pytest tests`
