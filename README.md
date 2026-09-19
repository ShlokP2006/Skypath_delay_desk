# SkyPath Delay Desk

A Streamlit app that forecasts whether a US departure will leave more than 15 minutes late, built on the flight delay dataset (`DEP_DEL15` target). You fill in a flight plan, and the app returns a boarding-pass style briefing with the delay odds, the best slot of the day for the same flight, and the factors moving the odds.

Delay risk is graded with the same four colours pilots use for METAR flight categories: VFR (low), MVFR (slightly elevated), IFR (high) and LIFR (very high).

## Quick start

```bash
pip install -r requirements.txt

# 1. Put train.csv and test.csv from the dataset zip into data/
# 2. Train (reads the CSVs, writes model + summaries to artifacts/)
python train_model.py --sample-frac 0.4

# 3. Launch
streamlit run app.py
```

Want to see it running first? `python train_model.py --demo` trains on synthetic flights in about a minute. The app shows a banner in that mode, because every number is made up.

## What is in the folder

```
skypath_delay_desk/
├── app.py                  Streamlit app (six tabs)
├── train_model.py          Reads the CSVs, trains LightGBM, writes artifacts/
├── make_demo_data.py       Synthetic train/test files for a smoke test
├── requirements.txt
├── .streamlit/config.toml  Theme colours
├── src/
│   ├── config.py           Column lists, labels, colours: one source of truth
│   ├── data.py             Chunked CSV loader with optional sampling
│   ├── lookups.py          Typical values for the form + chart summaries
│   ├── modeling.py         Training, evaluation, inference, risk categories
│   └── ui.py               CSS, boarding-pass ticket, Plotly styling
├── docs/
│   ├── raw_data_documentation.txt     Original, unchanged
│   ├── train_sets_documentation.txt   Original, unchanged
│   ├── important_topics.md            What matters in this dataset
│   └── data_dictionary.md             Every column and how the app fills it
├── data/                   Put train.csv and test.csv here
└── artifacts/              Created by train_model.py
```

## The tabs

- **Pre-flight briefing**: the flight plan form, the ticket, the same flight across every departure slot, and the factors pushing the odds up or down (LightGBM's built-in SHAP values).
- **Airport radar**: a map of airports sized by traffic and coloured by late rate, plus a watch list and a best-performers list.
- **Airlines and fleet**: late rate by airline, aircraft age and aircraft size.
- **Sky and clock**: day-by-hour heatmap, month, weather, ramp congestion and the leg-of-the-day effect.
- **Model bay**: ROC-AUC, PR-AUC, lift, calibration, feature importance and the confusion matrix on the test file.
- **Field notes**: the topics and data dictionary from `docs/`.

## How it works

1. `train_model.py` reads `train.csv` in chunks, keeping only the 25 feature columns and the target. `--sample-frac 0.4` keeps a random 40% of every chunk; use `1.0` for everything.
2. It builds small lookup tables (airport size, airline staffing, typical traffic per slot, typical weather per month). This is what lets the form ask for things a passenger knows and fill in the rest.
3. It fits a LightGBM classifier with native categorical handling for airline, airport, previous airport and time block, with early stopping on a 10% validation slice. The alert threshold is the one that maximises F1 on that slice.
4. It scores `test.csv` and stores ROC-AUC, PR-AUC, calibration, lift and feature importance.
5. The app only loads `artifacts/`, so it starts fast and never touches the 1 GB CSV.

The model uses only information available before pushback. The raw on-time table also holds actual times and delay-cause flags, which would leak the answer, and the train file already drops them.

## Check these on the first real run

This project was written from the two documentation files and tested on synthetic data with the same columns. It has not been run on the real CSVs, so look at these first:

- **Column names.** The docs call the time block `DEP_BLOCK`; the loader also accepts `DEP_TIME_BLK`. If training stops with a "columns are missing" message, compare the CSV header with `docs/train_sets_documentation.txt`.
- **`PREVIOUS_AIRPORT`.** The form assumes first flights of the day are labelled `NONE` (missing values are mapped to it too). If your file uses another label, set `FIRST_LEG` in `src/config.py` so the "first leg" shortcut matches.
- **Weather units.** The docs give inches for rain and snow but not units for `TMAX` and `AWND`. The weather bins in `src/lookups.py` assume Fahrenheit and miles per hour. Adjust them if the values look different.
- **Memory.** The default `--sample-frac 0.4` keeps memory use down. If you still run out, lower the fraction.
- **Training time.** Minutes on a sample, longer on the full file. `--trees 400 --learning-rate 0.1` is a quick pass.

## Ideas for next steps

- Add the previous flight's actual delay as a feature. It is a strong real-world signal but is not in this dataset.
- Hourly weather (METAR) instead of one daily reading.
- Compare LightGBM with XGBoost or a small neural net, and check calibration for each.
- Add a route-level view once destination airport is available.

The forecast is a statistical estimate from historical data, not flight status.
