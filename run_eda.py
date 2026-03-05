from src.preprocess import load_and_preprocess_2018
from src.eda import run_eda

X, y = load_and_preprocess_2018("data/CSE-CICIDS-2018/02-14-2018.csv")

run_eda(X, y)