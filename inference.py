#!/usr/bin/env python3
import os
import warnings
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.drop_width.model import build_lstm_model
from src.drop_width.preprocessing import DropDataProcessor
from src.drop_width.side_to_width import width_estim

# Suppress TensorFlow messages and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

INFERENCE_TO_MODEL_COLUMNS = {
    "Advancing (degree)": "Advancing (degree)",
    "Receding (degree)": "Receding (degree)",
    "Drop length (cm)": "Drop length (cm)",
    "Drop height (cm)": "Drop height (cm)",
    "Velocity (cm/s)": "Velocity (cm/s)",
    "Middle line angle (degree)": "Middle line angle (degree)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a new SROF Excel file using training normalization from dataset.xlsx."
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        required=True,
        help="Path to the new SROF Excel file for inference.",
    )
    parser.add_argument(
        "--reference-data-file",
        "-r",
        type=Path,
        default=Path("data/dataset.xlsx"),
        help="Path to the reference dataset used for training normalization.",
    )
    parser.add_argument(
        "--weights-file",
        "-w",
        type=Path,
        default=Path("models/lstm_weights.h5"),
        help="Path to the pre-trained LSTM weights (.h5).",
    )
    parser.add_argument(
        "--window-size",
        "-s",
        type=int,
        default=20,
        help="Sliding window size for slicing the time series.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional extra path to also save the plot (PNG). The plot is always saved to the output folder.",
    )
    return parser.parse_args()


def prepare_inference_input(df):
    missing_columns = [column for column in INFERENCE_TO_MODEL_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "The inference Excel file is missing required columns: "
            f"{missing_text}"
        )

    prepared_df = pd.DataFrame(
        {
            model_column: df[source_column]
            for source_column, model_column in INFERENCE_TO_MODEL_COLUMNS.items()
        }
    )

    if "Drop width (cm)" in df.columns:
        prepared_df["Drop width (cm)"] = df["Drop width (cm)"]

    return prepared_df


def get_alignment_slices(window_size):
    left_trim = window_size // 2
    right_trim = window_size // 2
    prediction_trim = 1 if window_size % 2 == 0 else 0
    return left_trim, right_trim, prediction_trim


def trim_rows(df, left_trim, right_trim):
    if right_trim == 0:
        return df.iloc[left_trim:].copy()
    return df.iloc[left_trim:-right_trim].copy()


def main():
    args = parse_args()

    if args.window_size < 2:
        raise ValueError("window-size must be at least 2.")

    reference_df = pd.read_excel(args.reference_data_file)
    input_df = pd.read_excel(args.input_file)

    if len(input_df) < args.window_size:
        raise ValueError(
            f"The input file must contain at least {args.window_size} rows, "
            f"but it has {len(input_df)}."
        )

    processor = DropDataProcessor(reference_df)
    train_data, _, train_label, _, train_data_notnormalized = processor.extract_data_label()

    train_data_sliced, train_label_sliced = processor.slice_patch(
        args.window_size, train_data, train_label
    )

    model, _, _, _, _ = build_lstm_model(
        train_data_sliced, train_label_sliced, args.window_size
    )
    model.load_weights(args.weights_file)

    prepared_input_df = prepare_inference_input(input_df)
    y_pred, measured_width = width_estim(
        prepared_input_df,
        model,
        processor,
        train_data_notnormalized,
        args.window_size,
    )

    left_trim, right_trim, prediction_trim = get_alignment_slices(args.window_size)
    output_df = trim_rows(input_df, left_trim, right_trim)

    if prediction_trim:
        y_pred = y_pred[:-prediction_trim]

    output_df["Estimated Width (\u03bcm)"] = y_pred

    has_measured_width = "Drop width (cm)" in prepared_input_df.columns and len(measured_width) == len(y_pred)
    if has_measured_width:
        output_df["Measured Width (\u03bcm)"] = measured_width

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_stem = f"inference_{args.input_file.stem}"
    excel_path = output_dir / f"{output_stem}.xlsx"
    plot_path = output_dir / f"{output_stem}.png"

    output_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    plt.rcParams["figure.figsize"] = (20, 5)
    font_size = 32

    if "file number" in output_df.columns:
        x_values = output_df["file number"].to_numpy()
        x_label = "File Number"
    elif "Frame" in output_df.columns:
        x_values = output_df["Frame"].to_numpy()
        x_label = "Frames"
    else:
        x_values = np.arange(len(output_df))
        x_label = "Index"

    plt.xlabel(x_label, fontdict={"size": font_size})
    plt.ylabel("Width (\u03bcm)", fontdict={"size": font_size})
    plt.plot(
        x_values,
        y_pred,
        color="red",
        linestyle="dotted",
        linewidth=5,
        label="Estimated",
    )
    if has_measured_width:
        plt.plot(x_values, measured_width, "black", linewidth=5, label="Measured")
    plt.rc("xtick", labelsize=font_size)
    plt.rc("ytick", labelsize=font_size)
    plt.legend(fontsize=24, loc="upper left")
    plt.tight_layout()

    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
        print(f"Additional plot saved to {args.output}")

    plt.show()


if __name__ == "__main__":
    main()
