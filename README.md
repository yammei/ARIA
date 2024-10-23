<h1>ARIMAX: PANW (endo) + Friends (exo)</h1> 

Model Metrics:

```bash
MAE (%): 1.4106491746705319
RMSE (%): 1.6691941227145795
```

Model Output

```bash
Actual vs Forecast (Last 10 Days):
                Actual    Forecast
2024-10-08  355.130005  346.308547
2024-10-09  362.869995  351.796046
2024-10-10  369.399994  360.582167
2024-10-11  373.200012  369.425346
2024-10-14  373.910004  374.379146
2024-10-15  374.440002  380.446158
2024-10-16  373.230011  382.199641
2024-10-17  376.149994  382.862606
2024-10-18  374.829987  381.371636
2024-10-21  378.410004  381.935894
```

Endogenous Variables: Palo Alto Networks (PANW) Closing Price<br>
Exogenous Variables: PANW Adjacent/Competitor stock data: Closing Price, EMA 12, EMA 26, Volume

Environment Setup

```bash
git init
git remote add origin https://github.com/yammei/statistical-models.git
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Running the model

```bash
cd scripts
python3 analyze_comovement.py
python3 retrieve_data.py
python3 train_arimax.py
python3 model_inference.py
```
