# 🗽 New York City Taxi Trip Duration Prediction 🚕

  <img src="Figures/header.jpeg" width="100%" />

## **Project Overview** 📌
The goal of this project is to predict the duration of taxi trips in New York City using the Kaggle `NYC Taxi Trip Duration Dataset`. Reliable trip-time estimates are essential for ride-hailing services to optimize routing, set dynamic pricing, and provide accurate ETAs to users.

**Objectives:**
  - Perform thorough data exploration to understand spatiotemporal and passenger-related patterns.

  - Engineer features that capture geographic distance, directionality, and temporal context.

  - Train and compare machine learning models to achieve the lowest possible prediction error.

## **Dataset** 📂
- Source: Kaggle NYC Taxi Trip Duration competition [Download](https://www.kaggle.com/c/nyc-taxi-trip-duration).
- The training set contains `1.45 million records` with these core fields:

  - `id`: a unique identifier for each trip.
  - `vendor_id`: a code indicating the provider associated with the trip record.
  - `passenger_count`: the number of passengers in the vehicle (driver entered value).
  - `pickup_datetime`: date and time when the meter was engaged.
  - `pickup_longitude`: the longitude where the meter was engaged.
  - `pickup_latitude`: the latitude where the meter was engaged.
  - `dropoff_longitude`: the longitude where the meter was disengaged.
  - `dropoff_latitude`: the latitude where the meter was disengaged.
  - `store_and_fwd_flag:` This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server, Y = store and forward, N = not a store and forward trip.
  - `trip_duration:` duration of the trip in seconds.

- Preprocessing Steps:
  - Removed duplicate trip records to ensure data integrity.
  - Filtered out trips with `passenger_count = 0` (no meaningful pickup) and counts of 7–9 as unrealistic.
  - Excluded records with missing or zero GPS coordinates.
  - Filtered out spatial outliers by restricting pickup and dropoff coordinates to NYC bounding box (longitude: -74.03 to -73.75, latitude: 40.63 to 40.85).
  - Removed `dropoff_datetime` from features to prevent target leakage in a real-time prediction scenario.
  - Applied log transformation to the `trip_duration` target to address right-skew and stabilize variance.
  - Removed outliers in log(duration) beyond mean ± 3 standard deviations to further mitigate extreme values.


## Exploratory Data Analysis (EDA) 🔍

- Data Quality Checks: Assessed null values, coordinate bounds, and distribution of passenger counts.
- Duration Distribution: Revealed right skew; applied log transformation to stabilize variance and improve model convergence.
<p align="center">
  <img src="Figures/right skewed.png" width="45%" />
  <img src="Figures/4. Log of Trip Duration Plot.png" width="45%" />
</p>

- Temporal Trends: Examined trip counts and average durations by hour.

![Avg Duration by Hour](Figures/8.%20Pickup%20Hours.png)


## 🎯 **Project Goals**
- **Build models:** to estimate taxi trip duration in New York City using historical trip data.
- **Exploratory Data Analysis (EDA):** to uncover patterns and relationships in spatial and temporal features.
- **Engineer meaningful features:** to enhance model performance, including:
  - `Haversine distance:` The Haversine distance measures the shortest distance between two points over the Earth's surface as the crow flies, taking the Earth's curvature into account.
  ```text
  def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371    # Average radius of Earth in kilometers
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    distance = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return distance
  ```
  - `Manhattan distance:` Also called Taxicab distance or L1 distance, it measures how far two points are if you can only travel horizontally and vertically (like driving through a city grid).
  ```text
  def manhattan_distance(lat1, lng1, lat2, lng2):
    a  = np.abs(newDf['dropoff_longitude'] - newDf['pickup_longitude']) 
    b = np.abs(newDf['dropoff_latitude'] - newDf['pickup_latitude'])
    return a + b
  ```

  - `Bearing direction:` Bearing is the direction or angle from one point to another, usually measured in degrees from North (0°) clockwise (90° East, 180° South, etc.).
  ```text
  def bearing_direction(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
    ```
- **Extract time-based features:**
  -  `Hour of day`
  -  `Day of week`
  -  `Month`
  -  `Time of day`
  -  `Rush hour`
  -  `Holiday`


## 🧹 Data Cleaning
To ensure high-quality input for model training, the dataset was carefully cleaned and prepared:

- **Skewness Correction:**
  - Observed that the `trip_duration` feature was right-skewed, with a long tail of very long trips.

  - Applied a logarithmic transformation (log1p) to trip_duration to normalize its distribution, improving model performance.

  ![NYC Taxi Banner](Figures/4.%20Log%20of%20Trip%20Duration%20Plot.png)

- **Statistical Filtering:**

  - For key numerical features, outliers were further removed by applying mean ± 3 × standard deviation (STD) thresholds, keeping only values within a statistically reasonable range.

  ```text
  mean = np.mean(newDf["log_trip_duraion"])
  std = np.std(newDf["log_trip_duraion"])
  newDf = newDf[newDf['log_trip_duraion'] <= mean + 3*std]
  newDf = newDf[newDf['log_trip_duraion'] >= mean - 3*std]
  ```
  ![NYC Taxi Banner](Figures/5.%20Remove%20outliers.png)

- **Outlier Removal:**

  - Removed records with extreme values in pickup and dropoff latitude/longitude coordinates, which likely represented GPS errors.

  - Detected and eliminated trips with unrealistic distances or durations (e.g., extremely long or short trips).
<p align="center">
  <img src="Figures/6.%20Pickup%20Location.png" width="45%" />
  <img src="Figures/7.%20Dropoff%20Locations.png" width="45%" />
</p>

---

## 🧠 **Model Development**
Tried multiple regression algorithms:

  - Ridge (baseline)
  - Random Forest Regressor
  - Decision Tree Regressor
  - XGBoost Regressor (best performance)
