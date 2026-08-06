"""Render publication-style comparison curves for all price-response models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.price_response import (  # noqa: E402
    AggregateElasticityModel,
    BoundedRationalAgentModel,
    UtilityChoiceModel,
)


COLORS = ("#2563EB", "#EA580C", "#059669")
LINE_STYLES = ("-", "--", "-.")
MODELS = (
    ("Huang aggregate CDF", AggregateElasticityModel()),
    ("Mo reduced-form utility", UtilityChoiceModel()),
    ("Xie bounded rational", BoundedRationalAgentModel()),
)


def curve(value, axis):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return np.full_like(axis, float(value), dtype=float)
    return np.broadcast_to(value, axis.shape).astype(float)


def style_axis(axis, title, xlabel):
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=9)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel("Probability", fontsize=9)
    axis.set_ylim(-0.02, 1.02)
    axis.set_yticks(np.linspace(0, 1, 5))
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.65, alpha=0.65)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#94A3B8")
    axis.tick_params(colors="#475569", labelsize=8, length=3)
    axis.set_facecolor("#F8FAFC")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "price_response_validation.png",
    )
    args = parser.parse_args()

    price = np.linspace(0.25, 4.0, 151)
    wait = np.linspace(0.0, 600.0, 121)
    pickup = np.linspace(0.0, 3.0, 121)
    income = np.linspace(10.0, 50.0, 161)
    base_fare = 10.0
    reference_payment = 7.5

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Passenger and Driver Price-Response Models",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.952,
        "Theory-constrained calibration • probabilities evaluated under common reference scenarios",
        ha="center",
        fontsize=9,
        color="#64748B",
    )

    reposition_results = []
    for index, (label, model) in enumerate(MODELS):
        plot_style = {
            "color": COLORS[index],
            "linestyle": LINE_STYLES[index],
            "linewidth": 2.25,
            "label": label,
        }
        axes[0, 0].plot(
            price,
            curve(
                model.passenger_accept_probability(
                    base_fare * price,
                    base_fare,
                    expected_wait_time=0.0,
                    trip_time=600.0,
                    maximum_wait_time=300.0,
                ),
                price,
            ),
            **plot_style,
        )
        axes[0, 1].plot(
            price,
            curve(
                model.driver_accept_probability(
                    reference_payment * price,
                    reference_payment,
                    pickup_distance=0.5,
                    trip_distance=3.0,
                ),
                price,
            ),
            **plot_style,
        )
        axes[0, 2].plot(
            wait,
            curve(
                model.passenger_accept_probability(
                    base_fare,
                    base_fare,
                    expected_wait_time=wait,
                    trip_time=600.0,
                    maximum_wait_time=300.0,
                ),
                wait,
            ),
            **plot_style,
        )
        axes[1, 0].plot(
            pickup,
            curve(
                model.driver_accept_probability(
                    reference_payment * 1.5,
                    reference_payment,
                    pickup_distance=pickup,
                    trip_distance=3.0,
                ),
                pickup,
            ),
            **plot_style,
        )
        axes[1, 1].plot(
            income,
            curve(
                model.driver_online_probability(
                    expected_hourly_income=income,
                    reference_hourly_income=30.0,
                ),
                income,
            ),
            **plot_style,
        )
        reposition_results.append(
            model.driver_reposition_probabilities(
                expected_payments=np.array([20.0, 30.0, 25.0]),
                reposition_costs=np.array([0.0, 2.0, 1.0]),
            )
        )

    style_axis(axes[0, 0], "A  Passenger request vs. price", "Price multiplier")
    style_axis(axes[0, 1], "B  Driver acceptance vs. payment", "Payment multiplier")
    style_axis(axes[0, 2], "C  Passenger request vs. waiting", "Expected wait (seconds)")
    style_axis(axes[1, 0], "D  Driver acceptance vs. pickup", "Pickup distance (km)")
    style_axis(axes[1, 1], "E  Driver participation vs. income", "Expected hourly income")

    bar_axis = axes[1, 2]
    positions = np.arange(3)
    width = 0.23
    for index, (label, _) in enumerate(MODELS):
        bar_axis.bar(
            positions + (index - 1) * width,
            reposition_results[index],
            width=width,
            color=COLORS[index],
            alpha=0.88,
            label=label,
        )
    style_axis(bar_axis, "F  Reposition choice", "Candidate destination")
    bar_axis.set_xticks(positions, ["A\nnet 20", "B\nnet 28", "C\nnet 24"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=3.4,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
