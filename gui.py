import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\mini project\flight ticket price prediction"

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "Clean_Dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "xgboost_flight_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "models", "ordinal_encoder.pkl"
)

FEATURE_INFO_PATH = os.path.join(
    BASE_DIR, "models", "feature_info.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL FILES
# ============================================================

try:

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    with open(FEATURE_INFO_PATH, "rb") as f:
        feature_info = pickle.load(f)

    df = pd.read_csv(DATASET_PATH)

except Exception as e:

    print("Error loading project files:")
    print(e)

    raise SystemExit


# ============================================================
# REMOVE UNNECESSARY COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# ============================================================
# FEATURE INFORMATION
# ============================================================

# The training script may save feature information in
# different formats. Handle both common formats.

if isinstance(feature_info, dict):

    if "categorical_cols" in feature_info:
        categorical_cols = feature_info["categorical_cols"]

    elif "categorical_columns" in feature_info:
        categorical_cols = feature_info["categorical_columns"]

    else:
        categorical_cols = [
            "airline",
            "flight",
            "source_city",
            "departure_time",
            "stops",
            "arrival_time",
            "destination_city",
            "class"
        ]

    if "numerical_cols" in feature_info:
        numerical_cols = feature_info["numerical_cols"]

    elif "numerical_columns" in feature_info:
        numerical_cols = feature_info["numerical_columns"]

    else:
        numerical_cols = [
            "duration",
            "days_left"
        ]

else:

    categorical_cols = [
        "airline",
        "flight",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class"
    ]

    numerical_cols = [
        "duration",
        "days_left"
    ]


feature_cols = categorical_cols + numerical_cols
# ============================================================
# AVAILABLE VALUES
# ============================================================

source_cities = sorted(
    df["source_city"].dropna().unique().tolist()
)

destination_cities = sorted(
    df["destination_city"].dropna().unique().tolist()
)

classes = sorted(
    df["class"].dropna().unique().tolist()
)


# ============================================================
# FUNCTIONS
# ============================================================

def predict_price():

    try:

        source = source_var.get()
        destination = destination_var.get()
        flight_class = class_var.get()
        days_left_text = days_var.get().strip()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not source:
            messagebox.showwarning(
                "Input Required",
                "Please select source city."
            )
            return

        if not destination:
            messagebox.showwarning(
                "Input Required",
                "Please select destination city."
            )
            return

        if source == destination:
            messagebox.showwarning(
                "Invalid Route",
                "Source and destination cities cannot be the same."
            )
            return

        if not flight_class:
            messagebox.showwarning(
                "Input Required",
                "Please select class."
            )
            return

        if not days_left_text:
            messagebox.showwarning(
                "Input Required",
                "Please enter days left."
            )
            return

        try:
            days_left = int(days_left_text)
        except ValueError:

            messagebox.showerror(
                "Invalid Input",
                "Days Left must be a whole number."
            )
            return

        if days_left <= 0:

            messagebox.showerror(
                "Invalid Input",
                "Days Left must be greater than 0."
            )
            return

        if days_left > 49:

            messagebox.showwarning(
                "Outside Dataset Range",
                "The dataset contains approximately 1-49 days left.\n\n"
                "Please enter a value between 1 and 49."
            )
            return

        # ----------------------------------------------------
        # FILTER ROUTE + CLASS
        # ----------------------------------------------------

        route_df = df[
            (df["source_city"] == source) &
            (df["destination_city"] == destination) &
            (df["class"] == flight_class)
        ].copy()

        if route_df.empty:

            messagebox.showerror(
                "No Data",
                f"No historical data found for:\n\n"
                f"{source} → {destination}\n"
                f"Class: {flight_class}"
            )

            return

        # ----------------------------------------------------
        # AVAILABLE AIRLINES
        # ----------------------------------------------------

        airlines = sorted(
            route_df["airline"].dropna().unique().tolist()
        )

        results = []

        # ----------------------------------------------------
        # PREDICT FOR EACH AIRLINE
        # ----------------------------------------------------

        for airline in airlines:

            airline_df = route_df[
                route_df["airline"] == airline
            ].copy()

            if airline_df.empty:
                continue

            # Representative historical values
            duration = airline_df["duration"].median()

            departure_time = (
                airline_df["departure_time"]
                .mode()
                .iloc[0]
            )

            stops = (
                airline_df["stops"]
                .mode()
                .iloc[0]
            )

            arrival_time = (
                airline_df["arrival_time"]
                .mode()
                .iloc[0]
            )

            flight = (
                airline_df["flight"]
                .mode()
                .iloc[0]
            )

            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame([{

                "airline": airline,

                "flight": flight,

                "source_city": source,

                "departure_time": departure_time,

                "stops": stops,

                "arrival_time": arrival_time,

                "destination_city": destination,

                "class": flight_class,

                "duration": duration,

                "days_left": days_left

            }])

            # ------------------------------------------------
            # ENCODE CATEGORICAL FEATURES
            # ------------------------------------------------

            encoded = encoder.transform(
                input_data[categorical_cols]
            )

            encoded_df = pd.DataFrame(
                encoded,
                columns=categorical_cols
            )

            final_input = pd.concat(
                [
                    encoded_df.reset_index(drop=True),
                    input_data[numerical_cols].reset_index(drop=True)
                ],
                axis=1
            )

            # Make sure column order is correct
            final_input = final_input[feature_cols]

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(final_input)[0]

            historical_average = airline_df["price"].mean()

            results.append({

                "Airline": airline,

                "Duration": f"{duration:.1f} hrs",

                "Stops": stops,

                "Historical Avg": historical_average,

                "Predicted Price": prediction

            })

        # ----------------------------------------------------
        # CHECK RESULTS
        # ----------------------------------------------------

        if not results:

            messagebox.showerror(
                "Prediction Error",
                "Unable to generate predictions."
            )

            return

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            "Predicted Price"
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # CLEAR TABLE
        # ----------------------------------------------------

        for item in result_tree.get_children():
            result_tree.delete(item)

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        for index, row in results_df.iterrows():

            result_tree.insert(
                "",
                "end",
                values=(

                    row["Airline"],

                    row["Duration"],

                    row["Stops"],

                    f"₹{row['Historical Avg']:,.2f}",

                    f"₹{row['Predicted Price']:,.2f}"

                )
            )

        # ----------------------------------------------------
        # LOWEST PRICE
        # ----------------------------------------------------

        cheapest = results_df.iloc[0]

        lowest_label.config(
            text=
            f"Lowest Predicted Fare: "
            f"{cheapest['Airline']}  |  "
            f"₹{cheapest['Predicted Price']:,.2f}"
        )

        # ----------------------------------------------------
        # ROUTE INFO
        # ----------------------------------------------------

        route_label.config(
            text=
            f"{source}  →  {destination}   |   "
            f"{flight_class}   |   "
            f"{days_left} days remaining"
        )

        # ----------------------------------------------------
        # SAVE CSV
        # ----------------------------------------------------

        filename = (
            f"prediction_"
            f"{source}_to_{destination}_"
            f"{flight_class}.csv"
        )

        filename = filename.replace(" ", "_")

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        results_df.to_csv(
            output_path,
            index=False
        )

        status_label.config(
            text=f"Report saved: {filename}"
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


def clear_results():

    source_var.set("")
    destination_var.set("")
    class_var.set("")
    days_var.set("")

    for item in result_tree.get_children():
        result_tree.delete(item)

    route_label.config(
        text="Enter flight details and click Predict Price"
    )

    lowest_label.config(
        text="Lowest Predicted Fare: --"
    )

    status_label.config(
        text="Ready"
    )


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Flight Ticket Price Prediction - XGBoost"
)

root.geometry("1000x650")

root.minsize(900, 600)

root.configure(
    bg="#f2f4f7"
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(

    root,

    text="✈ Flight Ticket Price Prediction",

    font=("Arial", 24, "bold"),

    bg="#f2f4f7",

    fg="#1f2937"

)

title_label.pack(
    pady=(20, 5)
)


subtitle_label = tk.Label(

    root,

    text="Machine Learning Model: XGBoost",

    font=("Arial", 12),

    bg="#f2f4f7",

    fg="#6b7280"

)

subtitle_label.pack(
    pady=(0, 20)
)


# ============================================================
# INPUT FRAME
# ============================================================

input_frame = tk.Frame(

    root,

    bg="white",

    bd=1,

    relief="solid"

)

input_frame.pack(

    padx=40,

    fill="x"

)


# ------------------------------------------------------------
# SOURCE
# ------------------------------------------------------------

tk.Label(

    input_frame,

    text="From City",

    font=("Arial", 11, "bold"),

    bg="white"

).grid(

    row=0,

    column=0,

    padx=20,

    pady=(20, 5),

    sticky="w"

)


source_var = tk.StringVar()

source_combo = ttk.Combobox(

    input_frame,

    textvariable=source_var,

    values=source_cities,

    state="readonly",

    width=25

)

source_combo.grid(

    row=1,

    column=0,

    padx=20,

    pady=(0, 20)

)


# ------------------------------------------------------------
# DESTINATION
# ------------------------------------------------------------

tk.Label(

    input_frame,

    text="To City",

    font=("Arial", 11, "bold"),

    bg="white"

).grid(

    row=0,

    column=1,

    padx=20,

    pady=(20, 5),

    sticky="w"

)


destination_var = tk.StringVar()

destination_combo = ttk.Combobox(

    input_frame,

    textvariable=destination_var,

    values=destination_cities,

    state="readonly",

    width=25

)

destination_combo.grid(

    row=1,

    column=1,

    padx=20,

    pady=(0, 20)

)


# ------------------------------------------------------------
# CLASS
# ------------------------------------------------------------

tk.Label(

    input_frame,

    text="Class",

    font=("Arial", 11, "bold"),

    bg="white"

).grid(

    row=0,

    column=2,

    padx=20,

    pady=(20, 5),

    sticky="w"

)


class_var = tk.StringVar()

class_combo = ttk.Combobox(

    input_frame,

    textvariable=class_var,

    values=classes,

    state="readonly",

    width=20

)

class_combo.grid(

    row=1,

    column=2,

    padx=20,

    pady=(0, 20)

)


# ------------------------------------------------------------
# DAYS LEFT
# ------------------------------------------------------------

tk.Label(

    input_frame,

    text="Days Left",

    font=("Arial", 11, "bold"),

    bg="white"

).grid(

    row=0,

    column=3,

    padx=20,

    pady=(20, 5),

    sticky="w"

)


days_var = tk.StringVar()

days_entry = tk.Entry(

    input_frame,

    textvariable=days_var,

    width=15,

    font=("Arial", 11)

)

days_entry.grid(

    row=1,

    column=3,

    padx=20,

    pady=(0, 20)

)


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(

    root,

    bg="#f2f4f7"

)

button_frame.pack(

    pady=20

)


predict_button = tk.Button(

    button_frame,

    text="🔮 Predict Price",

    command=predict_price,

    font=("Arial", 12, "bold"),

    bg="#2563eb",

    fg="white",

    padx=25,

    pady=10,

    cursor="hand2"

)

predict_button.grid(

    row=0,

    column=0,

    padx=10

)


clear_button = tk.Button(

    button_frame,

    text="Clear",

    command=clear_results,

    font=("Arial", 12),

    bg="#6b7280",

    fg="white",

    padx=25,

    pady=10,

    cursor="hand2"

)

clear_button.grid(

    row=0,

    column=1,

    padx=10

)


# ============================================================
# ROUTE LABEL
# ============================================================

route_label = tk.Label(

    root,

    text="Enter flight details and click Predict Price",

    font=("Arial", 12, "bold"),

    bg="#f2f4f7",

    fg="#374151"

)

route_label.pack(
    pady=5
)


# ============================================================
# RESULTS FRAME
# ============================================================

result_frame = tk.Frame(

    root,

    bg="white",

    bd=1,

    relief="solid"

)

result_frame.pack(

    padx=40,

    pady=10,

    fill="both",

    expand=True

)


columns = (

    "Airline",

    "Duration",

    "Stops",

    "Historical Avg",

    "Predicted Price"

)


result_tree = ttk.Treeview(

    result_frame,

    columns=columns,

    show="headings",

    height=8

)


for column in columns:

    result_tree.heading(

        column,

        text=column

    )

    result_tree.column(

        column,

        anchor="center",

        width=170

    )


result_tree.pack(

    fill="both",

    expand=True,

    padx=10,

    pady=10

)


# ============================================================
# LOWEST PRICE
# ============================================================

lowest_label = tk.Label(

    root,

    text="Lowest Predicted Fare: --",

    font=("Arial", 15, "bold"),

    bg="#f2f4f7",

    fg="#15803d"

)

lowest_label.pack(

    pady=10

)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(

    root,

    text="Ready",

    font=("Arial", 9),

    bg="#f2f4f7",

    fg="#6b7280"

)

status_label.pack(

    pady=(0, 10)

)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()