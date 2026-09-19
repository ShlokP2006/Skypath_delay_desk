### The topics that matter in this dataset

**1. The target: a 15-minute departure delay.**
`DEP_DEL15` is 1 when the departure was delayed by more than 15 minutes. It is a yes/no label about the *departure*, not the arrival, and delayed flights are the minority, so the classes are imbalanced. That is why the app reports ROC-AUC, PR-AUC, lift and calibration rather than accuracy.

**2. Two layers of data.**
The raw folder is the US on-time reporting table plus supporting tables: airport coordinates, aircraft inventory (seats, manufacture year), carrier names, airline employee counts, airport activity (departures and passengers enplaned) and daily airport weather. `train.csv` and `test.csv` are the same information already joined and boiled down to one row per flight.

**3. Feature families in the train file.**

| Family | Columns |
| --- | --- |
| Calendar | `MONTH`, `DAY_OF_WEEK` |
| The flight | `DEP_BLOCK`, `DISTANCE_GROUP`, `SEGMENT_NUMBER`, `PREVIOUS_AIRPORT` |
| Airport traffic | `CONCURRENT_FLIGHTS`, `AIRPORT_FLIGHTS_MONTH`, `AVG_MONTHLY_PASS_AIRPORT` |
| Airline scale and staffing | `AIRLINE_FLIGHTS_MONTH`, `AIRLINE_AIRPORT_FLIGHTS_MONTH`, `AVG_MONTHLY_PASS_AIRLINE`, `FLT_ATTENDANTS_PER_PASS`, `GROUND_SERV_PER_PASS` |
| Aircraft | `NUMBER_OF_SEATS`, `PLANE_AGE` |
| Location | `DEPARTING_AIRPORT`, `LATITUDE`, `LONGITUDE` |
| Weather (daily) | `PRCP`, `SNOW`, `SNWD`, `TMAX`, `AWND` |

**4. Congestion and knock-on delays.**
`CONCURRENT_FLIGHTS` counts other departures from the same airport in the same time block, a direct measure of ramp and runway pressure. `SEGMENT_NUMBER` and `PREVIOUS_AIRPORT` hint at "late aircraft" delays: an aircraft on its fifth leg has had more chances to fall behind than one on its first.

**5. Data leakage.**
The raw on-time table contains columns that only exist after the flight happens: actual departure and arrival times, delay minutes, cancellation flags and the carrier / weather / NAS / security / late-aircraft delay flags. None of them can be used to predict a delay, and the train file leaves them out. Anything you add later has to be knowable before pushback.

**6. Yearly aggregates describe size, not the day.**
The "per month" and "per passenger" columns are averages, so they say how big an airport or airline is, not how busy it was on a given day. Treat them as context, not live conditions.

**7. Weather is coarse.**
One daily reading per airport. It captures snow days and heavy rain, but not a thunderstorm that hits for one hour. Check the units of `TMAX` and `AWND` against the GHCND documentation before interpreting them.

**8. High-cardinality categories.**
Airports (and previous airports) have dozens to hundreds of values. Gradient-boosted trees with native categorical handling (LightGBM here) deal with this without one-hot encoding.

**9. Size.**
`train.csv` is over a gigabyte once unzipped. `train_model.py` reads it in chunks, keeps only the needed columns and can sample a fraction of rows.

**10. Ideas that would help the most.**
The delay of the aircraft's previous flight is a very strong signal for real-world operations, but it is not in this dataset. Hour-level weather (METAR) and airline on-time history by route would be the next upgrades.
