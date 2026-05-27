# 🦠 Plague Simulator

An agent-based **SEIRD** epidemic simulator built with **Dash** and **NumPy**.

Each agent is a moving dot in a 2-D arena. Disease spreads by proximity, progresses through an incubation period, and resolves into recovery or death. Includes real-world and fictional disease presets based on WHO/CDC data and pop-culture lore.

---

## Project structure

```
plague-simulator/
├── app.py            # Dash UI — sidebar controls, animation loop, charts
├── core/
│   ├── __init__.py
│   └── engine.py     # Vectorised SEIRD engine (NumPy)
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

Then open your browser at `http://localhost:8050`.

---

## Parameters

| Parameter | Description |
|---|---|
| Speed | Steps computed per tick — controls simulation speed (1x default) |
| Agents | Number of agents in the arena |
| Quarantine % | Fraction of agents that remain stationary from day one |
| Infection radius | Euclidean distance within which transmission can occur |
| Transmission probability (β) | Chance of infection per contact per step |
| Incubation period | Steps an exposed agent incubates before becoming infectious |
| Recovery time (γ) | Steps an infected agent takes to resolve (recover or die) |
| Case Fatality Rate (CFR) | Probability of dying instead of recovering |

---

## Model

The simulation uses a **SEIRD** compartmental model with agent-based movement:

```
Susceptible
    │
    └─(proximity + β roll)──► Exposed (incubating, not infectious)
                                    │
                              (incubation timer)
                                    │
                                    ▼
                                Infected (infectious)
                                    │
                              (recovery timer)
                               ┌────┴────┐
                           (1-CFR)      (CFR)
                               │          │
                               ▼          ▼
                           Recovered    Dead
```

- **Exposed** agents move but cannot transmit the disease.
- **Infected** agents move and actively spread the disease within their infection radius.
- **Recovered** agents are permanently immune.
- **Dead** agents remain on the map and stop moving.
- **Quarantined** agents (shown as squares) never move regardless of state.

---

## Disease presets

### Real-world

| Preset | Based on |
|---|---|
| COVID-19 (Wuhan Strain) | WHO early outbreak data |
| COVID-19 (Delta Variant) | CDC Delta variant estimates |
| Ebola (Zaire Strain) | WHO Ebola response data |
| Ebola (Bundibugyo Strain) | WHO outbreak reports |
| Hantavirus (Pulmonary Syndrome) | CDC Hantavirus data |
| Dengue (Severe/Hemorrhagic) | WHO Dengue guidelines |

### Fictional

| Preset | Source |
|---|---|
| Wildfire Virus | The Walking Dead |
| T-Virus | Resident Evil |
| Infestation Virus | WarZ |
| Krippin Virus | I Am Legend |
| Cordyceps | The Last of Us |
| Kharaa Bacterium | Subnautica |

---
