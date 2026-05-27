"""
SEIRD Agent-Based Simulation Engine
-------------------------------------
Uses NumPy for fully vectorized position updates and
pairwise distance calculations — no Python loops in hot paths.

State encoding:
  0 = Susceptible (S)
  1 = Infected    (I)
  2 = Recovered   (R)
  3 = Dead        (D)
  4 = Exposed     (E)  ← new: incubating, not yet infectious
"""

import numpy as np


class SEIRDEngine:
    """
    Population of agents moving in a [0, 1] x [0, 1] arena.

    Parameters
    ----------
    population       : total number of agents
    infection_radius : Euclidean distance threshold for transmission
    beta             : probability of transmission per contact per step (0-1)
    incubation_range : (min, max) steps an agent stays exposed before becoming
                       infectious — each agent draws a unique random duration
                       from this range, modelling biological variability
    recovery_time    : steps an agent stays infected before outcome
    mortality_rate   : probability of dying instead of recovering (0-1)
    quarantine_pct   : fraction of agents that remain stationary (0-1)
    seed             : optional RNG seed for reproducibility
    """

    SUSCEPTIBLE = 0
    INFECTED    = 1
    RECOVERED   = 2
    DEAD        = 3
    EXPOSED     = 4

    def __init__(
        self,
        population: int         = 200,
        infection_radius: float = 0.05,
        beta: float             = 0.40,
        incubation_range: tuple   = (20, 140),
        recovery_time: int      = 100,
        mortality_rate: float   = 0.02,
        quarantine_pct: float   = 0.0,
        seed: int | None        = None,
    ):
        self.population       = population
        self.infection_radius = infection_radius
        self.beta             = beta
        self.incubation_range = tuple(incubation_range)
        self.recovery_time    = recovery_time
        self.mortality_rate   = mortality_rate
        self.quarantine_pct   = quarantine_pct
        self.rng              = np.random.default_rng(seed)

        self.day     = 0
        self.history = {"S": [], "E": [], "I": [], "R": [], "D": []}

        self._init_agents()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_agents(self) -> None:
        """Allocate and randomise all agent arrays."""
        n = self.population

        self.pos = self.rng.random((n, 2), dtype=np.float32)

        speed    = 0.008
        self.vel = (self.rng.random((n, 2), dtype=np.float32) - 0.5) * speed * 2

        # Quarantined agents never move
        n_quar           = int(n * self.quarantine_pct)
        self.quarantined = np.zeros(n, dtype=bool)
        if n_quar > 0:
            idx = self.rng.choice(n, size=n_quar, replace=False)
            self.quarantined[idx] = True
            self.vel[idx]         = 0.0

        self.state            = np.zeros(n, dtype=np.int8)
        self.incubation_counter = np.zeros(n, dtype=np.int32)
        self.recovery_counter = np.zeros(n, dtype=np.int32)

        # Pre-roll each agent's fate: will they die or recover?
        # True  = will die when timer expires
        # False = will recover
        self.will_die = self.rng.random(n) < self.mortality_rate

        # Seed one exposed agent (patient zero enters incubation, not yet infectious)
        patient_zero = self.rng.integers(0, n)
        self.state[patient_zero]              = self.EXPOSED
        lo, hi = self.incubation_range
        self.incubation_counter[patient_zero] = int(self.rng.integers(lo, hi + 1))

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """Advance by one frame. Returns current SEIRD counts + day."""
        self._move()
        self._spread()
        self._incubate()  # E → I transition
        self._resolve()   # recover or die

        self.day += 1
        c = self.counts()
        for k in ("S", "E", "I", "R", "D"):
            self.history[k].append(c[k])
        return c

    def step_move_only(self) -> None:
        """Move agents without any disease progression — for smooth slow-mo rendering."""
        self._move()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _move(self) -> None:
        """Bounce-back boundary movement. Dead agents stay still."""
        alive = self.state != self.DEAD
        self.pos[alive] += self.vel[alive]

        for axis in range(2):
            over_max = (self.pos[:, axis] > 1.0) & alive
            over_min = (self.pos[:, axis] < 0.0) & alive
            self.vel[over_max, axis] *= -1
            self.vel[over_min, axis] *= -1
            self.pos[:, axis] = np.clip(self.pos[:, axis], 0.0, 1.0)

    def _spread(self) -> None:
        """
        Vectorized SEIRD infection spread via NumPy broadcasting.
        Only INFECTED agents spread the disease (EXPOSED do not transmit).
        Dead and Exposed agents cannot catch the disease.
        Newly exposed agents enter the E state with a fresh incubation counter.
        """
        infected_mask    = self.state == self.INFECTED
        susceptible_mask = self.state == self.SUSCEPTIBLE

        infected_pos    = self.pos[infected_mask]
        susceptible_pos = self.pos[susceptible_mask]

        if infected_pos.size == 0 or susceptible_pos.size == 0:
            return

        # Pairwise squared distances (nS, nI)
        diff    = susceptible_pos[:, np.newaxis, :] - infected_pos[np.newaxis, :, :]
        sq_dist = (diff ** 2).sum(axis=2)
        in_range = sq_dist < (self.infection_radius ** 2)

        any_in_range = in_range.any(axis=1)

        coin_flip         = self.rng.random(any_in_range.sum()) < self.beta
        new_exposed_local = np.where(any_in_range)[0][coin_flip]

        susceptible_indices                    = np.where(susceptible_mask)[0]
        newly_exposed                          = susceptible_indices[new_exposed_local]
        # S → E: each agent gets its own random incubation duration
        self.state[newly_exposed]              = self.EXPOSED
        lo, hi = self.incubation_range
        n_new  = len(newly_exposed)
        if n_new:
            self.incubation_counter[newly_exposed] = self.rng.integers(
                lo, hi + 1, size=n_new
            )

    def _incubate(self) -> None:
        """
        Decrement incubation counters for EXPOSED agents.
        When the counter hits 0: E → I, initialize recovery counter,
        and (re-)roll mortality fate for this agent.
        """
        exposed_mask = self.state == self.EXPOSED
        self.incubation_counter[exposed_mask] -= 1

        graduating = exposed_mask & (self.incubation_counter <= 0)
        self.state[graduating]            = self.INFECTED
        self.recovery_counter[graduating] = self.recovery_time
        # Roll mortality fate now that incubation is complete
        self.will_die[graduating] = self.rng.random(graduating.sum()) < self.mortality_rate

    def _resolve(self) -> None:
        """
        Decrement recovery timers.
        When timer hits 0: agents with will_die=True → DEAD, others → RECOVERED.
        Dead agents stop moving (velocity zeroed).
        """
        infected_mask = self.state == self.INFECTED
        self.recovery_counter[infected_mask] -= 1

        timer_done = infected_mask & (self.recovery_counter <= 0)

        dying                      = timer_done & self.will_die
        recovering                 = timer_done & ~self.will_die
        self.state[dying]          = self.DEAD
        self.vel[dying]            = 0.0          # dead agents stop
        self.state[recovering]     = self.RECOVERED

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def counts(self) -> dict:
        return {
            "S"  : int((self.state == self.SUSCEPTIBLE).sum()),
            "E"  : int((self.state == self.EXPOSED).sum()),
            "I"  : int((self.state == self.INFECTED).sum()),
            "R"  : int((self.state == self.RECOVERED).sum()),
            "D"  : int((self.state == self.DEAD).sum()),
            "day": self.day,
        }

    def is_over(self) -> bool:
        no_exposed  = int((self.state == self.EXPOSED).sum()) == 0
        no_infected = int((self.state == self.INFECTED).sum()) == 0
        return no_exposed and no_infected and self.day > 0

    def peak_infected(self) -> int:
        return max(self.history["I"]) if self.history["I"] else 0

    def total_deaths(self) -> int:
        return int((self.state == self.DEAD).sum())

    def reset(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.day     = 0
        self.history = {"S": [], "E": [], "I": [], "R": [], "D": []}
        self._init_agents()