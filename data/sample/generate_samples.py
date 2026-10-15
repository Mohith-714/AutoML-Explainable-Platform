"""Generate realistic synthetic tabular domain datasets for testing and demonstrations."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent


def generate_manufacturing_data() -> None:
    """Generate industrial sensor dataset: reference, stable monitoring, and drifted monitoring."""
    np.random.seed(42)
    n_ref = 1000
    n_drift = 350
    n_stable = 200

    shifts = ["Morning", "Evening", "Night"]

    def _make_batch(n: int, vibration_mean: float, temp_mean: float, defect_boost: float = 0.0) -> pd.DataFrame:
        vibration = np.random.normal(vibration_mean, 6.0, n).clip(15.0, 110.0)
        temperature = np.random.normal(temp_mean, 8.0, n).clip(30.0, 120.0)
        pressure = np.random.normal(101.3, 4.5, n).clip(80.0, 130.0)
        speed = np.random.normal(1500, 120, n).clip(900, 2100)
        lubricant = np.random.uniform(40.0, 95.0, n)
        shift = np.random.choice(shifts, n, p=[0.4, 0.35, 0.25])
        machine_id = [f"MCH-{np.random.randint(100, 999):03d}" for _ in range(n)]

        # Defect probability formula
        logits = (
            0.08 * (vibration - 45.0)
            + 0.06 * (temperature - 65.0)
            - 0.03 * (lubricant - 60.0)
            + (0.3 if defect_boost else 0.0)
            - 2.2
        )
        probs = 1.0 / (1.0 + np.exp(-logits))
        defects = (np.random.rand(n) < probs).astype(int)

        return pd.DataFrame({
            "machine_id": machine_id,
            "vibration_hz": np.round(vibration, 2),
            "temperature_c": np.round(temperature, 2),
            "pressure_kpa": np.round(pressure, 2),
            "rotational_speed_rpm": np.round(speed, 1),
            "lubricant_level_pct": np.round(lubricant, 1),
            "operator_shift": shift,
            "defect_detected": defects,
        })

    # Reference (normal operating baseline)
    ref_df = _make_batch(n_ref, vibration_mean=45.0, temp_mean=65.0)
    ref_df.to_csv(SAMPLE_DIR / "manufacturing_reference.csv", index=False)

    # Stable monitoring batch (no drift)
    stable_df = _make_batch(n_stable, vibration_mean=45.2, temp_mean=65.5)
    stable_df.to_csv(SAMPLE_DIR / "manufacturing_monitoring_stable.csv", index=False)

    # Drifted monitoring batch (bearing wear: vibration & temp elevated)
    drift_df = _make_batch(n_drift, vibration_mean=72.0, temp_mean=84.0, defect_boost=0.3)
    drift_df.to_csv(SAMPLE_DIR / "manufacturing_monitoring_drifted.csv", index=False)


def generate_churn_data() -> None:
    """Generate telecom customer churn dataset."""
    np.random.seed(123)
    n = 800

    contracts = ["Month-to-month", "One year", "Two year"]
    payments = ["Electronic check", "Bank transfer", "Credit card"]

    customer_id = [f"CUST-{10000 + i}" for i in range(n)]
    tenure = np.random.exponential(18, n).clip(1, 72).astype(int)
    monthly = np.random.normal(65, 25, n).clip(20, 120)
    total = (tenure * monthly + np.random.normal(0, 50, n)).clip(20, 9000)
    contract = np.random.choice(contracts, n, p=[0.55, 0.25, 0.20])
    support_calls = np.random.poisson(1.8, n).clip(0, 9)
    payment = np.random.choice(payments, n)

    # Probability of churn
    logits = (
        -0.05 * tenure
        + 0.02 * (monthly - 60)
        + 0.45 * (support_calls - 2)
        + np.where(contract == "Month-to-month", 0.8, -0.6)
        - 1.2
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    churn = (np.random.rand(n) < probs).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure,
        "monthly_charges": np.round(monthly, 2),
        "total_charges": np.round(total, 2),
        "contract_type": contract,
        "support_calls": support_calls,
        "payment_method": payment,
        "churn": churn,
    })
    df.to_csv(SAMPLE_DIR / "customer_churn_sample.csv", index=False)


def generate_housing_data() -> None:
    """Generate residential real estate price valuation dataset (regression)."""
    np.random.seed(999)
    n = 600

    zones = ["North", "South", "East", "West", "Downtown"]
    zone_premiums = {"North": 20000, "South": -15000, "East": 5000, "West": 10000, "Downtown": 75000}

    prop_id = [f"PROP-{2000 + i}" for i in range(n)]
    sqft = np.random.normal(1850, 450, n).clip(600, 4200)
    beds = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    baths = (beds * 0.6 + np.random.choice([0.5, 1.0, 1.5], n)).clip(1.0, 4.5)
    zone = np.random.choice(zones, n, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    age = np.random.uniform(1, 45, n)
    lot_size = (sqft * np.random.uniform(2.0, 5.0, n)).clip(1500, 20000)

    # Price function
    base_price = (
        150.0 * sqft
        + 18000.0 * beds
        + 12000.0 * baths
        - 1200.0 * age
        + np.array([zone_premiums[z] for z in zone])
        + 50000.0
        + np.random.normal(0, 18000, n)
    )
    price = np.round(base_price.clip(60000, 950000), -2)

    df = pd.DataFrame({
        "property_id": prop_id,
        "square_feet": np.round(sqft, 0),
        "bedrooms": beds,
        "bathrooms": np.round(baths, 1),
        "location_zone": zone,
        "age_years": np.round(age, 1),
        "lot_size_sqft": np.round(lot_size, 0),
        "price_usd": price,
    })
    df.to_csv(SAMPLE_DIR / "housing_price_sample.csv", index=False)


if __name__ == "__main__":
    generate_manufacturing_data()
    generate_churn_data()
    generate_housing_data()
    print("Synthetic sample datasets generated successfully in:", SAMPLE_DIR)
