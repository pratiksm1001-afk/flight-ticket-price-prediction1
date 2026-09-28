# ============================================================
# FLIGHT TICKET PRICE PREDICTION USING XGBOOST
# predict_price.py
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\mini project\flight ticket price prediction"

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Clean_Dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_flight_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ordinal_encoder.pkl"
)

FEATURE_INFO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_info.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 75)
print("        ✈️ FLIGHT TICKET PRICE PREDICTION")
print("                 USING XGBOOST")
print("=" * 75)

print("\n🤖 Loading trained ML model...")


try:

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)

except FileNotFoundError:

    print("\n❌ XGBoost model not found!")

    print(
        f"Expected:\n{MODEL_PATH}"
    )

    exit()


# ============================================================
# 3. LOAD ENCODER
# ============================================================

try:

    with open(
        ENCODER_PATH,
        "rb"
    ) as file:

        encoder = pickle.load(file)

except FileNotFoundError:

    print("\n❌ Encoder not found!")

    print(
        f"Expected:\n{ENCODER_PATH}"
    )

    exit()


# ============================================================
# 4. LOAD FEATURE INFORMATION
# ============================================================

try:

    with open(
        FEATURE_INFO_PATH,
        "rb"
    ) as file:

        feature_info = pickle.load(file)

except FileNotFoundError:

    print("\n❌ Feature information not found!")

    print(
        f"Expected:\n{FEATURE_INFO_PATH}"
    )

    exit()


all_columns = feature_info[
    "all_columns"
]

categorical_cols = feature_info[
    "categorical_columns"
]


# ============================================================
# 5. LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(
        DATASET_PATH
    )

except FileNotFoundError:

    print("\n❌ Dataset not found!")

    print(
        f"Expected:\n{DATASET_PATH}"
    )

    exit()


if "Unnamed: 0" in df.columns:

    df.drop(
        "Unnamed: 0",
        axis=1,
        inplace=True
    )


print(
    "\n✅ Model, encoder and dataset loaded successfully."
)


# ============================================================
# 6. DISPLAY AVAILABLE CITIES
# ============================================================

available_sources = sorted(
    df["source_city"]
    .dropna()
    .unique()
)

available_destinations = sorted(
    df["destination_city"]
    .dropna()
    .unique()
)

print("\n" + "=" * 75)

print(
    "AVAILABLE SOURCE CITIES"
)

print(
    ", ".join(
        available_sources
    )
)

print("\n" + "-" * 75)

print(
    "AVAILABLE DESTINATION CITIES"
)

print(
    ", ".join(
        available_destinations
    )
)

print("=" * 75)


# ============================================================
# 7. USER INPUT - SOURCE CITY
# ============================================================

user_source = input(
    "\nEnter origin city (FROM): "
).strip()


# Find actual dataset value ignoring case
source_match = next(
    (
        city
        for city in available_sources
        if str(city).lower()
        == user_source.lower()
    ),
    None
)


if source_match is None:

    print(
        f"\n❌ '{user_source}' "
        "is not available in the dataset."
    )

    print(
        "\nPlease choose one of:"
    )

    print(
        ", ".join(
            available_sources
        )
    )

    exit()


user_source = source_match


# ============================================================
# 8. USER INPUT - DESTINATION CITY
# ============================================================

user_dest = input(
    "Enter destination city (TO): "
).strip()


destination_match = next(
    (
        city
        for city in available_destinations
        if str(city).lower()
        == user_dest.lower()
    ),
    None
)


if destination_match is None:

    print(
        f"\n❌ '{user_dest}' "
        "is not available in the dataset."
    )

    print(
        "\nPlease choose one of:"
    )

    print(
        ", ".join(
            available_destinations
        )
    )

    exit()


user_dest = destination_match


# ============================================================
# 9. FILTER ROUTE
# ============================================================

route_matches = df[
    (
        df["source_city"]
        == user_source
    )
    &
    (
        df["destination_city"]
        == user_dest
    )
]


if route_matches.empty:

    print(
        f"\n❌ No flights found from "
        f"{user_source} to {user_dest}."
    )

    exit()


print(
    f"\n✅ Route found: "
    f"{user_source} → {user_dest}"
)

print(
    f"Available records: "
    f"{len(route_matches)}"
)


# ============================================================
# 10. USER INPUT - CLASS
# ============================================================

available_classes = sorted(
    route_matches["class"]
    .dropna()
    .unique()
)

print(
    "\nAvailable classes:"
)

for item in available_classes:

    print(
        f"  • {item}"
    )


user_class = input(
    "\nSelect Class (Economy/Business): "
).strip()


class_match = next(
    (
        value
        for value in available_classes
        if str(value).lower()
        == user_class.lower()
    ),
    None
)


if class_match is None:

    print(
        f"\n❌ Class '{user_class}' "
        "is not available on this route."
    )

    exit()


user_class = class_match


# ============================================================
# 11. USER INPUT - DAYS LEFT
# ============================================================

while True:

    try:

        user_days = int(
            input(
                "\nEnter days remaining "
                "before departure (1-49): "
            )
        )

        if user_days < 1:

            print(
                "❌ Days must be at least 1."
            )

            continue

        if user_days > 49:

            print(
                "⚠️ Dataset normally contains "
                "days_left values up to 49."
            )

            confirm = input(
                "Continue anyway? (y/n): "
            ).strip().lower()

            if confirm != "y":

                continue

        break

    except ValueError:

        print(
            "❌ Please enter a valid number."
        )


# ============================================================
# 12. FILTER ROUTE + CLASS
# ============================================================

route_class_matches = route_matches[
    route_matches["class"]
    == user_class
]


if route_class_matches.empty:

    print(
        "\n❌ No flights found for this "
        "route and class."
    )

    exit()


# ============================================================
# 13. FIND AVAILABLE AIRLINES
# ============================================================

unique_airlines = sorted(
    route_class_matches[
        "airline"
    ]
    .dropna()
    .unique()
)


print("\n" + "=" * 75)

print(
    f"🔍 Analyzing "
    f"{len(unique_airlines)} airlines..."
)

print("=" * 75)


# ============================================================
# 14. PREDICT PRICE FOR EACH AIRLINE
# ============================================================

report_data = []


for airline in unique_airlines:

    airline_matches = route_class_matches[
        route_class_matches["airline"]
        == airline
    ]


    if airline_matches.empty:

        continue


    # --------------------------------------------------------
    # Representative values from dataset
    # --------------------------------------------------------

    historical_avg = airline_matches[
        "price"
    ].mean()


    # Duration
    if "duration" in airline_matches.columns:

        inferred_duration = float(
            airline_matches[
                "duration"
            ].median()
        )

    else:

        inferred_duration = 5.0


    # Departure time
    if "departure_time" in airline_matches.columns:

        departure_mode = (
            airline_matches[
                "departure_time"
            ]
            .mode()
        )

        if len(departure_mode) > 0:

            inferred_dep_time = (
                departure_mode.iloc[0]
            )

        else:

            inferred_dep_time = (
                airline_matches[
                    "departure_time"
                ].iloc[0]
            )

    else:

        inferred_dep_time = "Morning"


    # Stops
    if "stops" in airline_matches.columns:

        stops_mode = (
            airline_matches[
                "stops"
            ]
            .mode()
        )

        if len(stops_mode) > 0:

            inferred_stops = (
                stops_mode.iloc[0]
            )

        else:

            inferred_stops = (
                airline_matches[
                    "stops"
                ].iloc[0]
            )

    else:

        inferred_stops = "zero"


    # Arrival time
    if "arrival_time" in airline_matches.columns:

        arrival_mode = (
            airline_matches[
                "arrival_time"
            ]
            .mode()
        )

        if len(arrival_mode) > 0:

            inferred_arr_time = (
                arrival_mode.iloc[0]
            )

        else:

            inferred_arr_time = (
                airline_matches[
                    "arrival_time"
                ].iloc[0]
            )

    else:

        inferred_arr_time = "Evening"


    # Flight number
    if "flight" in airline_matches.columns:

        flight_mode = (
            airline_matches[
                "flight"
            ]
            .mode()
        )

        if len(flight_mode) > 0:

            inferred_flight = (
                flight_mode.iloc[0]
            )

        else:

            inferred_flight = (
                airline_matches[
                    "flight"
                ].iloc[0]
            )

    else:

        inferred_flight = "Unknown"


    # --------------------------------------------------------
    # Create model input
    # --------------------------------------------------------

    raw_input = {

        "airline": airline,

        "flight": inferred_flight,

        "source_city": user_source,

        "departure_time":
            inferred_dep_time,

        "stops": inferred_stops,

        "arrival_time":
            inferred_arr_time,

        "destination_city":
            user_dest,

        "class": user_class,

        "duration":
            inferred_duration,

        "days_left":
            user_days
    }


    # --------------------------------------------------------
    # Make sure input categories exactly match dataset
    # --------------------------------------------------------

    for column in categorical_cols:

        if column not in raw_input:

            continue


        possible_values = (
            df[column]
            .dropna()
            .unique()
        )


        matched_value = None


        for value in possible_values:

            if (
                str(value).lower()
                ==
                str(
                    raw_input[column]
                ).lower()
            ):

                matched_value = value

                break


        if matched_value is not None:

            raw_input[column] = (
                matched_value
            )


    # --------------------------------------------------------
    # Create DataFrame in exact training-column order
    # --------------------------------------------------------

    try:

        user_frame = pd.DataFrame(
            [raw_input]
        )

        user_frame = user_frame[
            all_columns
        ]

    except Exception as error:

        print(
            f"\n❌ Error creating input "
            f"for {airline}: {error}"
        )

        continue


    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    user_frame_encoded = (
        user_frame.copy()
    )


    if len(categorical_cols) > 0:

        user_frame_encoded[
            categorical_cols
        ] = encoder.transform(
            user_frame[
                categorical_cols
            ]
        )


    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            user_frame_encoded
        )[0]

    except Exception as error:

        print(
            f"\n❌ Prediction failed "
            f"for {airline}: {error}"
        )

        continue


    # Prevent negative prediction
    prediction = max(
        0,
        float(prediction)
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    report_data.append({

        "Airline":
            airline,

        "Duration":
            f"{inferred_duration:.1f} hrs",

        "Stops":
            inferred_stops,

        "Historical Avg":
            historical_avg,

        "Predicted Price":
            prediction
    })


# ============================================================
# 15. CHECK RESULTS
# ============================================================

if not report_data:

    print(
        "\n❌ No predictions could be generated."
    )

    exit()


# ============================================================
# 16. CREATE RESULT DATAFRAME
# ============================================================

comparison_df = pd.DataFrame(
    report_data
)


# Sort by predicted price
comparison_df = comparison_df.sort_values(
    by="Predicted Price"
).reset_index(
    drop=True
)


# ============================================================
# 17. DISPLAY FORMATTED REPORT
# ============================================================

print("\n")
print("=" * 90)

print(
    "📊 FLIGHT FARE COMPARISON REPORT"
)

print(
    f"{user_source.upper()} ➜ "
    f"{user_dest.upper()}"
)

print(
    f"Class: {user_class}"
)

print(
    f"Days before departure: "
    f"{user_days}"
)

print("=" * 90)


display_df = comparison_df.copy()


display_df[
    "Historical Avg"
] = display_df[
    "Historical Avg"
].apply(
    lambda x: f"₹{x:,.2f}"
)


display_df[
    "Predicted Price"
] = display_df[
    "Predicted Price"
].apply(
    lambda x: f"₹{x:,.2f}"
)


print(
    display_df.to_string(
        index=False
    )
)


print("=" * 90)


# ============================================================
# 18. LOWEST PREDICTED PRICE
# ============================================================

cheapest = comparison_df.iloc[0]


print(
    "\n💰 Lowest predicted fare:"
)

print(
    f"Airline       : "
    f"{cheapest['Airline']}"
)

print(
    f"Predicted fare: "
    f"₹{cheapest['Predicted Price']:,.2f}"
)


# ============================================================
# 19. SAVE REPORT TO CSV
# ============================================================

safe_source = (
    str(user_source)
    .replace(" ", "_")
)

safe_dest = (
    str(user_dest)
    .replace(" ", "_")
)

safe_class = (
    str(user_class)
    .replace(" ", "_")
)


report_filename = (
    f"prediction_"
    f"{safe_source}_"
    f"to_"
    f"{safe_dest}_"
    f"{safe_class}.csv"
)


report_path = os.path.join(
    OUTPUT_DIR,
    report_filename
)


comparison_df.to_csv(
    report_path,
    index=False
)


print(
    f"\n📁 Prediction report saved:"
)

print(
    report_path
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 90)

print(
    "✅ PRICE PREDICTION COMPLETED SUCCESSFULLY!"
)

print("=" * 90)