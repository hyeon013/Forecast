import csv
from datetime import datetime, timedelta
from pathlib import Path


SOURCE_PATH = Path(__file__).with_name("asos_108_2023_2024.csv")
OUTPUT_PATH = Path(__file__).with_name("wind_dataset.csv")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OUTPUT_COLUMNS = [
    "forecast_time",
    "target_time",
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_ms_1h_before",
    "wind_speed_ms_2h_before",
    "actual_wind_speed_ms",
]


def main():
    observations = {}
    with SOURCE_PATH.open("r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        required_columns = {
            "datetime",
            "wind_speed_ms",
            "temperature_C",
            "humidity_pct",
        }
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.strptime(row["datetime"], TIME_FORMAT)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid datetime on CSV line {line_number}: {row['datetime']!r}"
                ) from error
            if timestamp in observations:
                raise ValueError(f"Duplicate observation timestamp: {row['datetime']}")
            observations[timestamp] = row

    if not observations:
        raise ValueError("The source CSV contains no observations.")

    first_time = min(observations)
    last_time = max(observations)
    hourly_count = int((last_time - first_time) / timedelta(hours=1)) + 1
    aligned_times = [
        first_time + timedelta(hours=offset) for offset in range(hourly_count)
    ]
    aligned_observations = {time: observations.get(time) for time in aligned_times}

    feature_columns = [
        "wind_speed_ms",
        "temperature_C",
        "humidity_pct",
    ]
    saved_rows = []
    for forecast_time in aligned_times:
        target_time = forecast_time + timedelta(hours=1)
        one_hour_before = forecast_time - timedelta(hours=1)
        two_hours_before = forecast_time - timedelta(hours=2)
        needed_times = [
            forecast_time,
            one_hour_before,
            two_hours_before,
            target_time,
        ]
        if any(
            time not in aligned_observations or aligned_observations[time] is None
            for time in needed_times
        ):
            continue

        current = aligned_observations[forecast_time]
        previous = aligned_observations[one_hour_before]
        two_previous = aligned_observations[two_hours_before]
        target = aligned_observations[target_time]
        required_values = [
            current[column] for column in feature_columns
        ] + [
            previous["wind_speed_ms"],
            two_previous["wind_speed_ms"],
            target["wind_speed_ms"],
        ]
        if any(value is None or not value.strip() for value in required_values):
            continue

        saved_rows.append(
            {
                "forecast_time": forecast_time.strftime(TIME_FORMAT),
                "target_time": target_time.strftime(TIME_FORMAT),
                "wind_speed_ms": current["wind_speed_ms"],
                "temperature_C": current["temperature_C"],
                "humidity_pct": current["humidity_pct"],
                "wind_speed_ms_1h_before": previous["wind_speed_ms"],
                "wind_speed_ms_2h_before": two_previous["wind_speed_ms"],
                "actual_wind_speed_ms": target["wind_speed_ms"],
            }
        )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(saved_rows)

    with OUTPUT_PATH.open("r", newline="", encoding="utf-8") as output_file:
        output_reader = csv.DictReader(output_file)
        written_rows = list(output_reader)
    if len(written_rows) != len(saved_rows):
        raise RuntimeError("The saved dataset row count does not match the generated data.")

    print(f"Hourly-aligned rows: {len(aligned_times)}")
    print(f"Saved dataset rows: {len(written_rows)}")
    print("First 4 saved rows:")
    print(",".join(OUTPUT_COLUMNS))
    for row in written_rows[:4]:
        print(",".join(row[column] for column in OUTPUT_COLUMNS))


if __name__ == "__main__":
    main()
