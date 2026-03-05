from src.sliding_window_eval import sliding_window_experiment

ALL_FILES = [
    "data/CSE-CICIDS-2018/02-14-2018.csv",
    "data/CSE-CICIDS-2018/02-15-2018.csv",
    "data/CSE-CICIDS-2018/02-16-2018.csv",
    "data/CSE-CICIDS-2018/02-20-2018.csv",
    "data/CSE-CICIDS-2018/02-21-2018.csv",
    "data/CSE-CICIDS-2018/02-22-2018.csv",
    "data/CSE-CICIDS-2018/02-23-2018.csv",
    "data/CSE-CICIDS-2018/02-28-2018.csv",
    "data/CSE-CICIDS-2018/03-01-2018.csv",
    "data/CSE-CICIDS-2018/03-02-2018.csv",
]

sliding_window_experiment(ALL_FILES, window_size=3)
