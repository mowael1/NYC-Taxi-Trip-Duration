# 🗽 New York City Taxi Trip Duration Prediction 🚕

  <img src="Figures/header.jpeg" width="100%" />

## **1. Project Overview** 📌
The goal of this project is to predict the duration of taxi trips in New York City using the Kaggle `NYC Taxi Trip Duration Dataset`. Reliable trip-time estimates are essential for ride-hailing services to optimize routing, set dynamic pricing, and provide accurate ETAs to users.

**Objectives:**
  - Perform thorough data exploration to understand spatiotemporal and passenger-related patterns.

  - Engineer features that capture geographic distance, directionality, and temporal context.

  - Train and compare machine learning models to achieve the lowest possible prediction error.

## **2. Dataset** 📂
- **Source:** [Kaggle NYC Taxi Trip Duration competition](https://www.kaggle.com/c/nyc-taxi-trip-duration).
- The dataset contains `1.45 million records` with these core fields:

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


## **3. Exploratory Data Analysis (EDA)** 🔍

- **Data Quality Checks:** Assessed null values, coordinate bounds, and distribution of passenger counts.

- **Duration Distribution:** Revealed right skew, applied log transformation to stabilize variance and improve model convergence.

<p align="center">
  <img src="Figures/right skewed.png" width="45%" />
  <img src="Figures/4. Log of Trip Duration Plot.png" width="45%" />
</p>

- **Temporal Trends:** Examined trip counts and average durations by hour.

![Avg Duration by Hour](Figures/8.%20Pickup%20Hours.png)


## **4. Feature Engineering** ✨

- **Haversine Distance:** Calculates the great-circle distance between pickup and dropoff coordinates.

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
- **Manhattan Distance:** Approximates grid-based distance using east–west and north–south differentials.

  ```text
  def manhattan_distance(lat1, lng1, lat2, lng2):
    a  = np.abs(newDf['dropoff_longitude'] - newDf['pickup_longitude']) 
    b = np.abs(newDf['dropoff_latitude'] - newDf['pickup_latitude'])
    return a + b
  ```
- **Bearing:** Computes the compass direction (0–360°) from pickup to dropoff.

  ```text
  def bearing_direction(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
    ```
- **Time-based Features:**
  - `Hour of Day:` Encodes diurnal traffic patterns.
  - `Day of Week:` Differentiates weekday vs. weekend behavior.
  - `Month:` Captures seasonal variation over the dataset’s January–June time frame.
  - `Time of Day:` Categorical buckets (e.g. Morning, Afternoon, Evening, Night).
  - `Rush Hour Flag:` Binary indicator for typical peak commute periods.
  - `Holiday Flag:` Marks federal and city-wide holidays based on U.S. holiday calendar.


## **5. Modeling Methodology 🛠️**

- **Data Split:** Partitioned the preprocessed data into 75% train, 12.5% validation, and 12.5% test sets to rigorously tune and evaluate model performance.

- **Preprocessing Pipeline:** Applied log-transformation to trip durations; standardized continuous features; one-hot encoded categorical variables using `pandas.get_dummies` to prepare features for model ingestion.

- **Baseline Model:** Employed Ridge Regression as a simple linear baseline on log(duration).

- **Additional Models:** Trained `Decision Tree Regressor` and `Random Forest Regressor` to capture non-linear relationships.

- **Advanced Model:** Tuned and evaluated XGBoost Regressor with hyperparameter optimization via RandomizedSearchCV and cross-validation, achieving the best validation performance.

- **Evaluation Metrics:** Used R² on log-transformed trip durations as the primary error metric
