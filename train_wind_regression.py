import csv
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "wind_dataset.csv"
OUTPUT_PATH = BASE_DIR / "predictions.csv"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FEATURE_COLUMNS = [
    "wind_speed_ms",
    "wind_speed_ms_1h_before",
    "wind_speed_ms_2h_before",
    "temperature_C",
    "humidity_pct",
]
OUTPUT_COLUMNS = [
    "forecast_time",
    "target_time",
    "actual_wind_speed_ms",
    "persistence_prediction_ms",
    "linear_regression_prediction_ms",
    "random_forest_prediction_ms",
]
TRAIN_START = datetime(2023, 1, 1)
EVALUATION_START = datetime(2024, 12, 1)
EVALUATION_END = datetime(2025, 1, 1)


def main():
    with DATASET_PATH.open("r", newline="", encoding="utf-8-sig") as dataset_file:
        reader = csv.DictReader(dataset_file)
        required_columns = set(FEATURE_COLUMNS) | {
            "forecast_time",
            "target_time",
            "actual_wind_speed_ms",
        }
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Dataset must contain columns: {sorted(required_columns)}")
        rows = list(reader)

    rows.sort(key=lambda row: datetime.strptime(row["target_time"], TIME_FORMAT))
    parsed_targets = [
        datetime.strptime(row["target_time"], TIME_FORMAT) for row in rows
    ]
    if parsed_targets != sorted(parsed_targets):
        raise ValueError("Dataset rows could not be sorted chronologically.")

    train_rows = [
        row
        for row in rows
        if TRAIN_START
        <= datetime.strptime(row["target_time"], TIME_FORMAT)
        < EVALUATION_START
    ]
    evaluation_rows = [
        row
        for row in rows
        if EVALUATION_START
        <= datetime.strptime(row["target_time"], TIME_FORMAT)
        < EVALUATION_END
    ]
    if not train_rows or not evaluation_rows:
        raise ValueError("Training or evaluation period contains no rows.")

    def features(row):
        return [float(row[column]) for column in FEATURE_COLUMNS]

    training_features = [features(row) for row in train_rows]
    training_targets = [float(row["actual_wind_speed_ms"]) for row in train_rows]
    evaluation_features = [features(row) for row in evaluation_rows]
    actual_values = [
        float(row["actual_wind_speed_ms"]) for row in evaluation_rows
    ]

    model = LinearRegression()
    model.fit(training_features, training_targets)
    regression_predictions = [
        max(0.0, float(value))
        for value in model.predict(evaluation_features)
    ]
    forest_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    forest_model.fit(training_features, training_targets)
    forest_predictions = [
        max(0.0, float(value))
        for value in forest_model.predict(evaluation_features)
    ]
    persistence_predictions = [
        float(row["wind_speed_ms"]) for row in evaluation_rows
    ]
    persistence_mae = mean_absolute_error(actual_values, persistence_predictions)
    regression_mae = mean_absolute_error(actual_values, regression_predictions)
    forest_mae = mean_absolute_error(actual_values, forest_predictions)

    output_rows = []
    for row, actual, persistence, regression, forest in zip(
        evaluation_rows,
        actual_values,
        persistence_predictions,
        regression_predictions,
        forest_predictions,
    ):
        output_rows.append(
            {
                "forecast_time": row["forecast_time"],
                "target_time": row["target_time"],
                "actual_wind_speed_ms": actual,
                "persistence_prediction_ms": persistence,
                "linear_regression_prediction_ms": regression,
                "random_forest_prediction_ms": forest,
            }
        )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    with OUTPUT_PATH.open("r", newline="", encoding="utf-8") as output_file:
        saved_rows = list(csv.DictReader(output_file))
    if len(saved_rows) != len(evaluation_rows):
        raise RuntimeError("Saved prediction count does not match evaluation rows.")

    print(f"Training rows: {len(train_rows)}")
    print(f"Evaluation rows: {len(evaluation_rows)}")
    print(f"Persistence MAE: {persistence_mae:.6f} m/s")
    print(f"Linear regression MAE: {regression_mae:.6f} m/s")
    print(f"Random forest MAE: {forest_mae:.6f} m/s")
    print(f"MAE change, linear regression vs persistence: {regression_mae - persistence_mae:+.6f} m/s")
    print(f"MAE change, random forest vs persistence: {forest_mae - persistence_mae:+.6f} m/s")
    print(f"MAE change, random forest vs linear regression: {forest_mae - regression_mae:+.6f} m/s")
    print("First evaluation result:")
    print(",".join(OUTPUT_COLUMNS))
    print(",".join(saved_rows[0][column] for column in OUTPUT_COLUMNS))


if __name__ == "__main__":
    main()
