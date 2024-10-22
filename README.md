Endogenous Variables: Palo Alto Networks (PANW) Closing Price
Exogenous Variables: S&P500 Companies with highest comovement

Environment Setup

```bash
git init
git remote add origin https://github.com/yammei/statistical-models.git
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
```

Running the model

```bash
cd scripts
python3 analyze_comovement.py
python3 retrieve_data.py
python3 train_arimax.py
python3 model_inference.py
```

Resources?

https://www.analyticsvidhya.com/blog/2021/11/basic-understanding-of-time-series-modelling-with-auto-arimax/
