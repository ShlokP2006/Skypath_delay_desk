### Data dictionary: train.csv and test.csv

Descriptions come from `train_sets_documentation.txt`. The last column says how the app gets each value when you build a flight plan.

| Column | Meaning | Where the app gets it |
| --- | --- | --- |
| `DEP_DEL15` | Target: departure delayed by more than 15 minutes (1 = yes) | Predicted |
| `MONTH` | Month | You choose |
| `DAY_OF_WEEK` | Day of week (1 = Monday) | You choose |
| `DEP_BLOCK` | Departure time block | You choose |
| `DISTANCE_GROUP` | Distance group of the flight (250-mile bands) | You choose |
| `SEGMENT_NUMBER` | Which leg of the day this tail number is on | You choose (1 if the aircraft's first flight) |
| `PREVIOUS_AIRPORT` | Airport the aircraft departed from before this one | You choose (`NONE` for a first flight) |
| `CONCURRENT_FLIGHTS` | Departures from the airport in the same block | Starts at the typical value for airport and slot; you can change it |
| `NUMBER_OF_SEATS` | Seats on the aircraft | Starts at the airline's median; you can change it |
| `PLANE_AGE` | Age of the departing aircraft | Starts at the airline's median; you can change it |
| `CARRIER_NAME` | Airline | You choose |
| `DEPARTING_AIRPORT` | Departing airport | You choose |
| `LATITUDE`, `LONGITUDE` | Position of the departing airport | Looked up from the airport |
| `AIRPORT_FLIGHTS_MONTH` | Average flights per month at the airport | Looked up from the airport |
| `AVG_MONTHLY_PASS_AIRPORT` | Average monthly passengers at the airport | Looked up from the airport |
| `AIRLINE_FLIGHTS_MONTH` | Average flights per month for the airline | Looked up from the airline |
| `AVG_MONTHLY_PASS_AIRLINE` | Average monthly passengers for the airline | Looked up from the airline |
| `FLT_ATTENDANTS_PER_PASS` | Flight attendants per passenger, airline-wide | Looked up from the airline |
| `GROUND_SERV_PER_PASS` | Ground service staff per passenger, airline-wide | Looked up from the airline |
| `AIRLINE_AIRPORT_FLIGHTS_MONTH` | Average flights per month for that airline at that airport | Looked up from the airline and airport pair |
| `PRCP` | Precipitation for the day (inches) | Starts at the airport's typical value for the month; you can change it |
| `SNOW` | Snowfall for the day (inches) | Same |
| `SNWD` | Snow on the ground (inches) | Same |
| `TMAX` | Maximum temperature for the day | Same |
| `AWND` | Wind speed for the day | Same |

The documentation names the time block column `DEP_BLOCK`. Some copies of the file call it `DEP_TIME_BLK`; the loader accepts both.
