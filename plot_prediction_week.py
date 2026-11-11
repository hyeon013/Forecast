import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "predictions.csv"
OUTPUT_PATH = BASE_DIR / "prediction_week.png"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
WEEK_START = datetime(2024, 12, 1, 0)
WEEK_END = datetime(2024, 12, 7, 23)
DECEMBER_START = datetime(2024, 12, 1)
JANUARY_START = datetime(2025, 1, 1)


def main():
    with PREDICTIONS_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("predictions.csv contains no prediction rows.")

    december_rows = [
        row
        for row in rows
        if DECEMBER_START
        <= datetime.strptime(row["target_time"], TIME_FORMAT)
        < JANUARY_START
    ]
    week_rows = [
        row
        for row in december_rows
        if WEEK_START
        <= datetime.strptime(row["target_time"], TIME_FORMAT)
        <= WEEK_END
    ]
    if not december_rows or not week_rows:
        raise ValueError("No rows found for the requested December evaluation period.")

    def mae(column):
        return sum(
            abs(float(row["actual_wind_speed_ms"]) - float(row[column]))
            for row in december_rows
        ) / len(december_rows)

    target_times = [
        datetime.strptime(row["target_time"], TIME_FORMAT) for row in week_rows
    ]
    actual = [float(row["actual_wind_speed_ms"]) for row in week_rows]
    persistence = [
        float(row["persistence_prediction_ms"]) for row in week_rows
    ]
    regression = [
        float(row["linear_regression_prediction_ms"]) for row in week_rows
    ]

    fig, axis = plt.subplots(figsize=(13, 6))
    axis.plot(target_times, actual, color="black", linewidth=1.5, label="Actual")
    axis.plot(
        target_times,
        persistence,
        color="gray",
        linestyle="--",
        linewidth=1.3,
        label="Persistence baseline",
    )
    axis.plot(
        target_times,
        regression,
        color="blue",
        linewidth=1.3,
        label="Linear regression",
    )
    axis.set_xlabel("Target time (KST)")
    axis.set_ylabel("Wind speed (m/s)")
    axis.set_title("Wind speed predictions: December 1-7, 2024 (KST)")
    axis.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)

    print(f"Plotted rows: {len(week_rows)}")
    print(f"December evaluation rows used for MAE: {len(december_rows)}")
    print(f"December persistence MAE: {mae('persistence_prediction_ms'):.6f} m/s")
    print(f"December linear regression MAE: {mae('linear_regression_prediction_ms'):.6f} m/s")
    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
