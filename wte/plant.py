"""Synthetic boiler dynamics shared by the live simulator and the forecast training data.

The operator loads a weighed batch of waste roughly every hour and logs it, together with the
plan for the next two batches (when, and how heavy). Plans slip a little: a batch goes in a
couple of minutes early or late and weighs a few percent more or less than planned.

Each batch burns off over about half an hour, heating the boiler; temperature follows the heat
release with a lag, and steam pressure follows temperature. On top of that the waste's
calorific value drifts slowly, the heat exchanger fouls over a run, and every reading carries
sensor noise and the occasional pressure spike. A run starts at first steam (~95 °C).

None of this is measured plant data: it is a model chosen to give the forecast realistic
time patterns to learn.
"""

import math
import random

from wte.config import PRESSURE_LIMIT_PSI

BURN_TAU_MIN = 30.0      # a batch of waste burns off with this time constant
THERMAL_TAU_MIN = 12.0   # boiler temperature lags the heat release by this much
BASE_TEMP_C = 95.0       # first steam
HEAT_GAIN = 960.0        # °C above base per kg/min of waste burned
MAX_RISE_C = 22.0        # soft ceiling on the rise above base


def power_w(temperature, pressure):
    """Generator output of the ~6 W prototype for a given boiler state (no noise)."""
    return 0.045 * temperature + 0.02 * pressure


class PlantProcess:
    """One run of the plant from first steam. Call `read()` once per sensor sample."""

    def __init__(self, rng: random.Random | None = None, fault_rate: float = 0.1):
        self.rng = rng or random.Random()
        self.fault_rate = fault_rate
        self.minutes = 0.0
        self.temperature = BASE_TEMP_C + self.rng.uniform(-0.5, 0.5)
        self.feed_interval = self.rng.uniform(45, 75)      # this operator's rhythm, minutes
        self.quality = self.rng.uniform(0.9, 1.1)          # calorific value of the waste
        self.fouling_per_hour = self.rng.uniform(0.005, 0.015)
        self.fuel = 0.0
        self.feed_log: list[dict] = []                     # feeds not yet collected by `pop_feeds()`
        self.plan: list[tuple[float, float]] = []          # (planned minute, planned kg), two batches ahead
        self._plan_batch(self.feed_interval * self.rng.uniform(0.3, 1.0))
        self._plan_batch()
        self._load(self.rng.uniform(0.3, 0.6))

    @property
    def pressure(self) -> float:
        return 50.0 + 0.35 * (self.temperature - BASE_TEMP_C)

    @property
    def next_feed(self) -> float:
        """When the next planned batch really goes in: a couple of minutes off the plan."""
        return self.plan[0][0] + self._slip

    def _plan_batch(self, wait: float | None = None) -> None:
        after = self.plan[-1][0] if self.plan else self.minutes
        wait = wait if wait is not None else max(20.0, self.feed_interval + self.rng.gauss(0, 4))
        self.plan.append((after + wait, self.rng.uniform(0.55, 0.95)))

    def _load(self, mass_kg: float) -> None:
        """Load a batch and log it with the plan for the next two, as the operator would."""
        self.fuel += mass_kg
        self._slip = self.rng.gauss(0, 2)
        (next_at, next_kg), (then_at, then_kg) = self.plan
        self.feed_log.append({
            "mass_kg": round(mass_kg, 3),
            "next_feed_in_min": round(next_at - self.minutes, 1),
            "next_mass_kg": round(next_kg, 3),
            "then_feed_in_min": round(then_at - self.minutes, 1),
            "then_mass_kg": round(then_kg, 3),
        })

    def pop_feeds(self) -> list[dict]:
        feeds, self.feed_log = self.feed_log, []
        return feeds

    def step(self, dt_seconds: float) -> None:
        dt = dt_seconds / 60.0
        self.minutes += dt

        while self.minutes >= self.next_feed:
            _, planned_kg = self.plan.pop(0)
            self._plan_batch()
            self._load(planned_kg * self.rng.uniform(0.95, 1.05))  # weighs a few percent off the plan

        heat = self.fuel / BURN_TAU_MIN
        self.fuel -= heat * dt

        # Ornstein-Uhlenbeck drift of the waste quality around 1.0, time constant ~6 h.
        self.quality += (1.0 - self.quality) * dt / 360.0 + 0.004 * math.sqrt(dt) * self.rng.gauss(0, 1)

        efficiency = max(0.7, 1.0 - self.fouling_per_hour * self.minutes / 60.0)
        rise = MAX_RISE_C * (1.0 - math.exp(-HEAT_GAIN * heat * self.quality * efficiency / MAX_RISE_C))
        self.temperature += (BASE_TEMP_C + rise - self.temperature) * dt / THERMAL_TAU_MIN

    def true_power(self) -> float:
        return power_w(self.temperature, self.pressure)

    def read(self, dt_seconds: float) -> dict:
        """Advance the plant by `dt_seconds` and return one noisy sensor reading."""
        self.step(dt_seconds)
        if self.rng.random() < self.fault_rate:
            pressure = self.rng.uniform(PRESSURE_LIMIT_PSI, PRESSURE_LIMIT_PSI + 6)
        else:
            pressure = self.pressure + self.rng.gauss(0, 0.4)
        voltage = 13.5 + 0.25 * (self.true_power() - 5.3)
        return {
            "temperature": round(self.temperature + self.rng.gauss(0, 0.3), 2),  # °C
            "pressure": round(pressure, 3),                                     # PSI
            "voltage": round(voltage + self.rng.gauss(0, 0.02), 2),             # V
        }
