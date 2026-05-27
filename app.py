"""
Plague Simulator — Dash App (SEIRD model)
==========================================
dcc.Interval at 33 ms (~30 FPS). Only figures update each tick.
engine.py upgraded from SIRD to SEIRD (Exposed incubation state).

Project layout:
    app.py
    core/__init__.py
    core/engine.py

Run with:
    python app.py
"""

import numpy as np
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

from core.engine import SEIRDEngine

# ── Colours ───────────────────────────────────────────────────────────────────
C_S  = "#4ade80"
C_E  = "#f59e0b"   # amber — exposed / incubating
C_I  = "#f87171"
C_R  = "#64748b"
C_D  = "#e2e8f0"
C_Q  = "#fbbf24"
BG   = "#0d1117"
SURF = "#161b22"
CARD = "#1e2432"
GRID = "#252d3d"
TEXT = "#94a3b8"
TX2  = "#e2e8f0"
BRD  = "#2d3748"

DISEASE_PRESETS = {
    "custom": "Custom (Manual Parameters)",

    # ── Real-world (Scientifically Correct / WHO & CDC Data) ──────────────────
    "covid_wuhan": {
        "name": "COVID-19 (Wuhan Strain)",
        "radius": 0.04, "beta": 0.30, "incub": [20, 50], "gamma": 100, "mort": 0.01,
    },
    "covid_delta": {
        "name": "COVID-19 (Delta Variant)",
        "radius": 0.06, "beta": 0.80, "incub": [10, 30], "gamma": 80, "mort": 0.01,
    },
    "ebola_zaire": {
        "name": "Ebola (Zaire Strain)",
        "radius": 0.02, "beta": 0.85, "incub": [20, 100], "gamma": 120, "mort": 0.88,
    },
    "ebola_bundibugyo": {
        "name": "Ebola (Bundibugyo Strain)",
        "radius": 0.02, "beta": 0.65, "incub": [30, 140], "gamma": 110, "mort": 0.34,
    },
    "hantavirus": {
        "name": "Hantavirus (Pulmonary Syndrome)",
        "radius": 0.03, "beta": 0.25, "incub": [140, 300], "gamma": 140, "mort": 0.38,
    },
    "dengue": {
        "name": "Dengue (Severe/Hemorrhagic)",
        "radius": 0.04, "beta": 0.45, "incub": [40, 100], "gamma": 90, "mort": 0.01,
    },

    # ── Fictional (Strict Pop-Culture Lore Scaled) ────────────────────────────
    "twd_wildfire": {
        "name": "Wildfire Virus (The Walking Dead)",
        "radius": 0.01, "beta": 1.00, "incub": [10, 20], "gamma": 30, "mort": 1.00,
    },
    "t_virus": {
        "name": "T-Virus (Resident Evil)",
        "radius": 0.04, "beta": 0.85, "incub": [10, 30], "gamma": 50, "mort": 0.85,
    },
    "infestation_warz": {
        "name": "Infestation Virus (WarZ)",
        "radius": 0.03, "beta": 0.75, "incub": [10, 50], "gamma": 70, "mort": 0.60,
    },
    "krippin_virus": {
        "name": "Krippin Virus (I Am Legend)",
        "radius": 0.12, "beta": 0.95, "incub": [20, 60], "gamma": 90, "mort": 0.94,
    },
    "cordyceps": {
        "name": "Cordyceps (The Last of Us)",
        "radius": 0.03, "beta": 0.95, "incub": [10, 30], "gamma": 40, "mort": 1.00,
    },
    "kharaa": {
        "name": "Kharaa Bacterium (Subnautica)",
        "radius": 0.05, "beta": 0.90, "incub": [15, 45], "gamma": 60, "mort": 1.00,
    },
}

BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, size=11, family="'IBM Plex Mono', monospace"),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(
        bgcolor="rgba(22,27,34,0.9)", bordercolor=BRD, borderwidth=1,
        font=dict(size=10), orientation="h",
        yanchor="bottom", y=1.02, xanchor="left", x=0,
    ),
)

GLOBAL_CSS = f"""
* {{box-sizing:border-box;margin:0;padding:0}}
body {{background:{BG};color:{TX2};
      font-family:'IBM Plex Sans',sans-serif;overflow:hidden}}
html,body,#react-entry-point,#react-entry-point>div {{height:100%}}
.plague-slider .rc-slider-rail {{background:{BRD};height:3px}}
.plague-slider .rc-slider-track {{background:{C_I};height:3px}}
.plague-slider .rc-slider-handle {{
    background:{C_I};border:2px solid {BG};
    width:14px;height:14px;margin-top:-5px;opacity:1}}
.plague-slider .rc-slider-handle:hover,
.plague-slider .rc-slider-handle:active {{
    box-shadow:0 0 0 4px rgba(248,113,113,0.2)}}
.plague-slider .rc-slider-mark {{display:none}}
.plague-slider .rc-slider-mark-text {{display:none}}
.rc-slider-tooltip {{display:none !important}}
::-webkit-scrollbar {{width:4px}}
::-webkit-scrollbar-track {{background:transparent}}
::-webkit-scrollbar-thumb {{background:{BRD};border-radius:2px}}
/* Force sidebar text colors — override Bootstrap */
span, label, div, p {{color:inherit}}
input[type=range] {{accent-color:{C_I}}}
"""

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap",
    ],
    title="Plague Simulator",
)
server = app.server  # Expose Flask server for gunicorn

# ── Helpers ───────────────────────────────────────────────────────────────────

def hex_rgba(hex_color: str, alpha: float = 0.08) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def agent_figure(eng: SEIRDEngine) -> go.Figure:
    traces = []

    def scatter(mask, name, color, symbol, size, border=None):
        pos = eng.pos[mask]
        if not pos.size:
            return
        traces.append(go.Scattergl(
            x=pos[:, 0], y=pos[:, 1], mode="markers",
            name=name, hoverinfo="skip",
            marker=dict(color=color, symbol=symbol, size=size,
                        line=dict(width=1.5 if border else 0,
                                  color=border or color)),
        ))

    mob  = ~eng.quarantined
    quar =  eng.quarantined
    S, E, I, R, D = (SEIRDEngine.SUSCEPTIBLE, SEIRDEngine.EXPOSED,
                     SEIRDEngine.INFECTED,    SEIRDEngine.RECOVERED,
                     SEIRDEngine.DEAD)

    scatter((eng.state == S) & mob,  "Susceptible",     C_S, "circle", 5)
    scatter((eng.state == E) & mob,  "Exposed",         C_E, "circle", 6)
    scatter((eng.state == I) & mob,  "Infected",        C_I, "circle", 7)
    scatter((eng.state == R) & mob,  "Recovered",       C_R, "circle", 5)
    scatter((eng.state == S) & quar, "Susceptible (Q)", C_S, "square", 6, C_Q)
    scatter((eng.state == E) & quar, "Exposed (Q)",     C_E, "square", 6, C_Q)
    scatter((eng.state == I) & quar, "Infected (Q)",    C_I, "square", 7, C_Q)
    scatter((eng.state == R) & quar, "Recovered (Q)",   C_R, "square", 6, C_Q)

    # Dead agents: large red X with dark border — hard to miss
    dead_pos = eng.pos[eng.state == D]
    if dead_pos.size:
        traces.append(go.Scattergl(
            x=dead_pos[:, 0], y=dead_pos[:, 1], mode="markers",
            name="Dead", hoverinfo="skip",
            marker=dict(
                color="#ff2244", symbol="x", size=10,
                line=dict(width=2.5, color="#1a0008"),
            ),
        ))

    shapes = []
    for pos in eng.pos[eng.state == I][:60]:
        r = eng.infection_radius
        shapes.append(dict(
            type="circle", xref="x", yref="y",
            x0=float(pos[0]) - r, y0=float(pos[1]) - r,
            x1=float(pos[0]) + r, y1=float(pos[1]) + r,
            line=dict(color="rgba(248,113,113,0.15)", width=1),
            fillcolor="rgba(248,113,113,0.03)",
        ))

    return go.Figure(data=traces, layout={
        **BASE_LAYOUT,
        "margin": dict(l=2, r=2, t=2, b=2),
        "shapes": shapes,
        "xaxis": dict(range=[0, 1], showgrid=False, zeroline=False,
                      showticklabels=False, constrain="domain"),
        "yaxis": dict(range=[0, 1], showgrid=False, zeroline=False,
                      showticklabels=False,
                      scaleanchor="x", scaleratio=1, constrain="domain"),
        "uirevision": "agents",
        "dragmode": False,
    })


def curve_figure(eng: SEIRDEngine) -> go.Figure:
    days = list(range(1, len(eng.history["S"]) + 1))

    def line(key, name, color, dash="solid"):
        return go.Scatter(
            x=days, y=eng.history[key], name=name,
            line=dict(color=color, width=2, dash=dash),
            fill="tozeroy", fillcolor=hex_rgba(color, 0.08),
        )

    traces = [
        line("S", "Susceptible", C_S),
        line("E", "Exposed",     C_E, dash="dash"),
        line("I", "Infected",    C_I),
        line("R", "Recovered",   C_R),
        line("D", "Dead",        C_D, dash="dot"),
    ]
    annotations = []
    if eng.history["I"]:
        pv = max(eng.history["I"])
        pd = eng.history["I"].index(pv) + 1
        annotations.append(dict(
            x=pd, y=pv, text=f"peak {pv}",
            showarrow=True, arrowhead=2, arrowcolor=C_I,
            font=dict(color=C_I, size=10), bgcolor="rgba(22,27,34,0.9)",
        ))

    return go.Figure(data=traces, layout=dict(
        **BASE_LAYOUT, annotations=annotations,
        xaxis=dict(gridcolor=GRID, zeroline=False,
                   title="Day", title_font=dict(size=10)),
        yaxis=dict(gridcolor=GRID, zeroline=False,
                   title="Agents", title_font=dict(size=10),
                   range=[0, eng.population * 1.05]),
        uirevision="curve",
    ))


def blank_fig(msg: str) -> go.Figure:
    return go.Figure(layout=dict(
        **BASE_LAYOUT,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(color=TEXT, size=14))],
    ))


def metric_card(label: str, value, color: str = TX2) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": TEXT,
                               "letterSpacing": "0.08em",
                               "textTransform": "uppercase",
                               "marginBottom": "4px"}),
        html.Div(str(value), style={"fontSize": "22px", "fontWeight": "500",
                                    "color": color,
                                    "fontFamily": "'IBM Plex Mono', monospace"}),
    ], style={"background": CARD, "border": f"1px solid {BRD}",
              "borderRadius": "8px", "padding": "10px 14px", "flex": "1"})


import base64

_HINT_SVG_RAW = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<circle cx="12" cy="12" r="10" fill="none" stroke="white" stroke-width="2"/>'
    '<text x="12" y="16.5" text-anchor="middle" font-size="13" font-weight="bold" '
    'font-family="Arial,sans-serif" fill="white">?</text>'
    '</svg>'
)
_HINT_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(_HINT_SVG_RAW.encode()).decode()

HINT_SVG = lambda hint_id: html.Img(
    id=hint_id,
    src=_HINT_DATA_URI,
    style={
        "width": "13px", "height": "13px", "minWidth": "13px",
        "opacity": "0.4", "cursor": "pointer",
        "flexShrink": "0",
    },
)


def slider_row(label: str, id_: str, min_, max_, step, value, hint_id=None) -> html.Div:
    right_children = [
        html.Span(id=f"{id_}-display",
                  style={"fontSize": "12px", "color": TX2,
                         "fontFamily": "'IBM Plex Mono', monospace",
                         "textAlign": "right", "whiteSpace": "nowrap"}),
    ]
    if hint_id:
        right_children.append(
            html.Div(HINT_SVG(hint_id),
                     style={"width": "22px", "display": "flex",
                            "justifyContent": "center", "flexShrink": "0"}),
        )
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "12px", "color": TEXT, "flex": "1"}),
            html.Div(right_children,
                     style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
        dcc.Slider(id=id_, min=min_, max=max_, step=step, value=value,
                   marks=None,
                   updatemode="drag", className="plague-slider"),
    ], style={"marginBottom": "14px"})


# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div([


    dcc.Store(id="sim-state", data={"running": False, "engine_data": None}),

    html.Div([

        # ── Sidebar ───────────────────────────────────────────────────────────
        html.Div([
            html.Div("🦠 PLAGUE", style={
                "fontSize": "18px", "fontWeight": "600", "color": C_I,
                "letterSpacing": "0.15em", "marginBottom": "2px",
                "fontFamily": "'IBM Plex Mono', monospace",
            }),
            html.Div("SEIRD epidemic simulator", style={
                "fontSize": "10px", "color": TEXT,
                "letterSpacing": "0.08em", "marginBottom": "20px",
            }),

            html.Div("SIMULATION", style={
                "fontSize": "10px", "color": TEXT, "letterSpacing": "0.12em",
                "marginBottom": "12px", "borderBottom": f"1px solid {BRD}",
                "paddingBottom": "6px",
            }),
            slider_row("Speed", "sl-speed", -4, 10, 1, 0),

            html.Div("POPULATION", style={
                "fontSize": "10px", "color": TEXT, "letterSpacing": "0.12em",
                "marginBottom": "12px", "borderBottom": f"1px solid {BRD}",
                "paddingBottom": "6px",
            }),
            slider_row("Agents",       "sl-pop",    50,   500, 10,   500, hint_id="hint-population"),
            slider_row("Quarantine %", "sl-quar",    0,    90,  5,     0, hint_id="hint-quarantine"),

            html.Div("VIRUS", style={
                "fontSize": "10px", "color": TEXT, "letterSpacing": "0.12em",
                "margin": "6px 0 12px", "borderBottom": f"1px solid {BRD}",
                "paddingBottom": "6px",
            }),

            # ── Disease Presets Dropdown ───────────────────────────────────────
            dcc.Dropdown(
                id="preset-dropdown",
                options=[
                    {"label": (v if isinstance(v, str) else v["name"]), "value": k}
                    for k, v in DISEASE_PRESETS.items()
                ],
                value="custom",
                clearable=False,
                searchable=False,
                optionHeight=32,
                className="preset-select",
            ),
            slider_row("Infection radius",   "sl-radius",  0.01, 0.15, 0.01,  0.05, hint_id="hint-radius"),
            slider_row("Transmission prob.", "sl-beta",    0.05, 1.00, 0.05,  0.40, hint_id="hint-transmission"),

            # ── Incubation range ──────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("Incubation Period (Steps)",
                              style={"fontSize": "12px", "color": TEXT, "flex": "1"}),
                    html.Div([
                        html.Span(id="sl-incubation-range-display",
                                  style={"fontSize": "12px", "color": TX2,
                                         "fontFamily": "'IBM Plex Mono', monospace",
                                         "textAlign": "right", "whiteSpace": "nowrap"}),
                        html.Div(HINT_SVG("hint-incubation"),
                                 style={"width": "22px", "display": "flex",
                                        "justifyContent": "center", "flexShrink": "0"}),
                    ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
                dcc.RangeSlider(
                    id="sl-incubation-range",
                    min=0, max=300, step=10,
                    value=[20, 140],
                    marks=None,
                    updatemode="drag",
                    className="plague-slider",
                ),
            ], style={"marginBottom": "14px"}),

            slider_row("Recovery time",      "sl-gamma",     20,  300,  10,   100, hint_id="hint-recovery"),
            slider_row("Case Fatality Rate (CFR)", "sl-mort", 0.00, 1.00, 0.01, 0.02, hint_id="hint-cfr"),

            # ── Tooltips ──────────────────────────────────────────────────────
            dbc.Tooltip("Total number of active individuals in the arena. Higher values increase crowd density and accelerate transmission chains.",
                        target="hint-population", placement="right"),
            dbc.Tooltip("Percentage of the population that remains stationary from day one, simulating lockdowns or isolation compliance.",
                        target="hint-quarantine", placement="right"),
            dbc.Tooltip("The danger zone radius around an infected agent. For reference: 0.03 represents close contact, while 0.08 represents airborne spread.",
                        target="hint-radius", placement="right"),
            dbc.Tooltip("Probability of transmission per contact tick. For example: a value of 0.40 means there is a 40% chance of transmission per exposure.",
                        target="hint-transmission", placement="right"),
            dbc.Tooltip("Silent incubation window. Exposed agents (orange) cannot transmit during this period. Note: 10 simulation steps equal 1 full day.",
                        target="hint-incubation", placement="right"),
            dbc.Tooltip("Active infectious duration (red dots). The agent recovers or dies after this timer ends. Note: 10 simulation steps equal 1 full day.",
                        target="hint-recovery", placement="right"),
            dbc.Tooltip("Case Fatality Rate The probability of an infected individual dying after the recovery timer ends. For example: 0.06 equals a 6% mortality rate.",
                        target="hint-cfr", placement="right"),

            html.Div(style={"flex": "1", "minHeight": "20px"}),

            html.Button("▶  SIMULATE", id="btn-start", n_clicks=0, style={
                "width": "100%", "padding": "10px",
                "background": C_S, "color": "#0d1117", "border": "none",
                "borderRadius": "6px", "fontSize": "12px", "fontWeight": "600",
                "letterSpacing": "0.12em", "cursor": "pointer",
                "fontFamily": "'IBM Plex Mono', monospace", "marginBottom": "8px",
            }),
            html.Button("↺  RESET", id="btn-reset", n_clicks=0, style={
                "width": "100%", "padding": "10px",
                "background": "transparent", "color": TEXT,
                "border": f"1px solid {BRD}", "borderRadius": "6px",
                "fontSize": "12px", "letterSpacing": "0.12em", "cursor": "pointer",
                "fontFamily": "'IBM Plex Mono', monospace",
            }),

            html.Div([
                html.Div("Created by Felix Loaiza", style={
                    "fontSize": "9px", "color": TEXT,
                    "letterSpacing": "0.06em",
                }),
                html.Div(
                    f"© {__import__('datetime').datetime.now().year} All rights reserved.",
                    style={"fontSize": "9px", "color": TEXT, "letterSpacing": "0.06em"},
                ),
            ], style={"marginTop": "14px", "textAlign": "center"}),
        ], style={
            "width": "260px", "flexShrink": "0",
            "background": SURF, "borderRight": f"1px solid {BRD}",
            "padding": "20px 16px", "display": "flex",
            "flexDirection": "column", "overflowY": "auto",
        }),

        # ── Main ──────────────────────────────────────────────────────────────
        html.Div([

            html.Div(id="metrics-row", style={"display": "flex", "gap": "10px"}),

            html.Div([
                # Agent Map card
                html.Div([
                    html.Div([
                        html.Div("AGENT MAP", style={
                            "fontSize": "10px", "color": TEXT, "letterSpacing": "0.12em",
                        }),
                        html.Button("⛶", id="btn-expand-agents", n_clicks=0, title="Expand",
                            style={"background": "transparent", "border": "none",
                                   "color": TEXT, "cursor": "pointer", "fontSize": "15px",
                                   "lineHeight": "1", "padding": "0 2px", "opacity": "0.6"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "marginBottom": "8px"}),
                    dcc.Graph(id="graph-agents", figure=blank_fig("Press ▶ START"),
                              config={"displayModeBar": False},
                              style={"flex": "1", "minHeight": "0"}),
                ], style={"flex": "2", "background": CARD, "border": f"1px solid {BRD}",
                          "borderRadius": "8px", "padding": "12px",
                          "display": "flex", "flexDirection": "column"}),

                # Epidemic Curve card
                html.Div([
                    html.Div([
                        html.Div("EPIDEMIC CURVE", style={
                            "fontSize": "10px", "color": TEXT, "letterSpacing": "0.12em",
                        }),
                        html.Button("⛶", id="btn-expand-curve", n_clicks=0, title="Expand",
                            style={"background": "transparent", "border": "none",
                                   "color": TEXT, "cursor": "pointer", "fontSize": "15px",
                                   "lineHeight": "1", "padding": "0 2px", "opacity": "0.6"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "marginBottom": "8px"}),
                    dcc.Graph(id="graph-curve", figure=blank_fig("Press ▶ START"),
                              config={"displayModeBar": False},
                              style={"flex": "1", "minHeight": "0"}),
                ], style={"flex": "1", "background": CARD, "border": f"1px solid {BRD}",
                          "borderRadius": "8px", "padding": "12px",
                          "display": "flex", "flexDirection": "column"}),

            ], style={"display": "flex", "gap": "14px", "flex": "1", "minHeight": "0"}),

            html.Div(id="banner", style={"display": "none"}),

            # Fullscreen modal overlay
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(id="modal-title", style={
                            "fontSize": "10px", "color": TEXT, "letterSpacing": "0.12em",
                        }),
                        html.Button("✕", id="btn-close-modal", n_clicks=0,
                            style={"background": "transparent", "border": "none",
                                   "color": TEXT, "cursor": "pointer", "fontSize": "18px",
                                   "lineHeight": "1", "padding": "0 4px", "opacity": "0.7"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center", "marginBottom": "10px"}),
                    dcc.Graph(id="graph-modal", figure=blank_fig(""),
                              config={"displayModeBar": False},
                              style={"flex": "1", "minHeight": "0"}),
                ], style={
                    "background": CARD, "border": f"1px solid {BRD}",
                    "borderRadius": "10px", "padding": "16px",
                    "width": "92vw", "height": "88vh",
                    "display": "flex", "flexDirection": "column",
                }),
            ], id="modal-overlay", style={
                "display": "none", "position": "fixed",
                "top": "0", "left": "0", "width": "100vw", "height": "100vh",
                "background": "rgba(13,17,23,0.85)", "backdropFilter": "blur(4px)",
                "zIndex": "10000", "alignItems": "center", "justifyContent": "center",
            }),

        ], style={
            "flex": "1", "display": "flex", "flexDirection": "column",
            "padding": "20px", "gap": "14px", "overflowY": "auto",
            "background": BG, "minHeight": "0",
        }),

    ], style={"display": "flex", "height": "100vh"}),

    # 33 ms = 30 FPS
    dcc.Interval(id="interval", interval=33, n_intervals=0, disabled=True),

], style={"height": "100vh", "overflow": "hidden"})


# ── Inject global CSS via clientside callback ─────────────────────────────────
app.clientside_callback(
    f"""
    function() {{
        if (!document.getElementById('plague-global-css')) {{
            var s = document.createElement('style');
            s.id = 'plague-global-css';
            s.textContent = `{GLOBAL_CSS}`;
            document.head.appendChild(s);
        }}
        return window.dash_clientside.no_update;
    }}
    """,
    Output("sim-state", "data", allow_duplicate=True),
    Input("interval",   "n_intervals"),
    prevent_initial_call=True,
)


# ── Slider display labels ─────────────────────────────────────────────────────
_SLIDERS = [
    ("sl-pop",    "{:.0f}"),
    ("sl-quar",   "{:.0f}%"),
    ("sl-radius", "{:.2f}"),
    ("sl-beta",   "{:.2f}"),
    ("sl-gamma",  "{:.0f}"),
    ("sl-mort",   "{:.2f}"),
    ("sl-speed",  "__skip__"),
]

for _sid, _fmt in _SLIDERS:
    if _fmt == "__skip__":
        continue
    @app.callback(
        Output(f"{_sid}-display", "children"),
        Input(_sid, "value"),
    )
    def _disp(val, fmt=_fmt):
        return fmt.format(val)


@app.callback(
    Output("sl-incubation-range-display", "children"),
    Input("sl-incubation-range", "value"),
)
def _disp_incub_range(val):
    return f"{val[0]}–{val[1]}"


@app.callback(
    Output("sl-speed-display", "children"),
    Input("sl-speed", "value"),
)
def _disp_speed(val):
    if val >= 0:
        return f"{val + 1}x"
    else:
        return f"1/{abs(val) + 1}x"


def speed_to_params(speed):
    """Return (steps_per_tick, skip_ticks) from slider value."""
    if speed >= 0:
        return speed + 1, 0      # e.g. speed=0 → 1 step/tick
    else:
        return 1, abs(speed)     # e.g. speed=-3 → 1 step every 4 ticks


# ── Engine serialisation ──────────────────────────────────────────────────────

def eng_to_dict(eng: SEIRDEngine) -> dict:
    return {
        "population":           eng.population,
        "infection_radius":     eng.infection_radius,
        "beta":                 eng.beta,
        "incubation_range":     list(eng.incubation_range),
        "recovery_time":        eng.recovery_time,
        "mortality_rate":       eng.mortality_rate,
        "quarantine_pct":       eng.quarantine_pct,
        "day":                  eng.day,
        "pos":                  eng.pos.tolist(),
        "vel":                  eng.vel.tolist(),
        "state":                eng.state.tolist(),
        "incubation_counter":   eng.incubation_counter.tolist(),
        "recovery_counter":     eng.recovery_counter.tolist(),
        "will_die":             eng.will_die.tolist(),
        "quarantined":          eng.quarantined.tolist(),
        "history":              eng.history,
    }


def dict_to_eng(d: dict) -> SEIRDEngine:
    eng = SEIRDEngine(
        population       = d["population"],
        infection_radius = d["infection_radius"],
        beta             = d["beta"],
        incubation_range = d["incubation_range"],
        recovery_time    = d["recovery_time"],
        mortality_rate   = d["mortality_rate"],
        quarantine_pct   = d["quarantine_pct"],
    )
    eng.day                = d["day"]
    eng.pos                = np.array(d["pos"],                dtype=np.float32)
    eng.vel                = np.array(d["vel"],                dtype=np.float32)
    eng.state              = np.array(d["state"],              dtype=np.int8)
    eng.incubation_counter = np.array(d["incubation_counter"], dtype=np.int32)
    eng.recovery_counter   = np.array(d["recovery_counter"],   dtype=np.int32)
    eng.will_die           = np.array(d["will_die"],           dtype=bool)
    eng.quarantined        = np.array(d["quarantined"],        dtype=bool)
    eng.history            = d["history"]
    return eng


# ── Start / Pause / Reset ─────────────────────────────────────────────────────
@app.callback(
    Output("sim-state",    "data",     allow_duplicate=True),
    Output("interval",     "disabled", allow_duplicate=True),
    Output("btn-start",    "children", allow_duplicate=True),
    Output("btn-start",    "style",    allow_duplicate=True),
    Output("banner",       "style",    allow_duplicate=True),
    Output("graph-agents", "figure",   allow_duplicate=True),
    Output("graph-curve",  "figure",   allow_duplicate=True),
    Input("btn-start",   "n_clicks"),
    Input("btn-reset",   "n_clicks"),
    State("sim-state",   "data"),
    State("sl-pop",              "value"),
    State("sl-quar",             "value"),
    State("sl-radius",           "value"),
    State("sl-beta",             "value"),
    State("sl-incubation-range", "value"),
    State("sl-gamma",            "value"),
    State("sl-mort",             "value"),
    prevent_initial_call=True,
)
def handle_controls(start_n, reset_n, state,
                    pop, quar, radius, beta, incub_range, gamma, mort):
    ctx = callback_context
    if not ctx.triggered:
        return (no_update,) * 7

    trigger = ctx.triggered[0]["prop_id"]

    BTN_BASE = {
        "width": "100%", "padding": "10px", "border": "none",
        "borderRadius": "6px", "fontSize": "12px", "fontWeight": "600",
        "letterSpacing": "0.12em", "cursor": "pointer",
        "fontFamily": "'IBM Plex Mono', monospace", "marginBottom": "8px",
        "color": "#0d1117",
    }

    if "btn-reset" in trigger:
        return (
            {"running": False, "engine_data": None},
            True, "▶  SIMULATE",
            {**BTN_BASE, "background": C_S},
            {"display": "none"},
            blank_fig("Press ▶ SIMULATE"),
            blank_fig("Press ▶ SIMULATE"),
        )

    running     = state.get("running", False)
    engine_data = state.get("engine_data")

    if engine_data is None:
        eng = SEIRDEngine(
            population       = pop,
            infection_radius = radius,
            beta             = beta,
            incubation_range = incub_range,
            recovery_time    = gamma,
            mortality_rate   = mort,
            quarantine_pct   = quar / 100,
        )
        engine_data = eng_to_dict(eng)

    new_running = not running
    label = "⏹  STOP" if new_running else "▶  SIMULATE"
    bg    = C_I if new_running else C_S
    return (
        {"running": new_running, "engine_data": engine_data},
        not new_running,
        label,
        {**BTN_BASE, "background": bg},
        {"display": "none"},
        no_update,
        no_update,
    )


# ── Animation tick ────────────────────────────────────────────────────────────
@app.callback(
    Output("graph-agents", "figure"),
    Output("graph-curve",  "figure"),
    Output("metrics-row",  "children"),
    Output("sim-state",    "data",     allow_duplicate=True),
    Output("interval",     "disabled", allow_duplicate=True),
    Output("btn-start",    "children", allow_duplicate=True),
    Output("btn-start",    "style",    allow_duplicate=True),
    Output("banner",       "children", allow_duplicate=True),
    Output("banner",       "style",    allow_duplicate=True),
    Input("interval",      "n_intervals"),
    State("sim-state",     "data"),
    State("sl-speed",      "value"),
    prevent_initial_call=True,
)
def tick(n, state, speed):
    BTN_BASE = {
        "width": "100%", "padding": "10px", "border": "none",
        "borderRadius": "6px", "fontSize": "12px", "fontWeight": "600",
        "letterSpacing": "0.12em", "cursor": "pointer",
        "fontFamily": "'IBM Plex Mono', monospace", "marginBottom": "8px",
        "color": "#0d1117",
    }
    if not state or not state.get("running") or not state.get("engine_data"):
        return (no_update,) * 9

    steps_per_tick, skip_ticks = speed_to_params(speed or 0)

    # Handle slow-down: move agents visually but don't advance disease
    ticks_remaining = state.get("ticks_remaining", 0)
    if ticks_remaining > 0:
        eng = dict_to_eng(state["engine_data"])
        eng.step_move_only()
        new_state = {**state,
                     "engine_data": eng_to_dict(eng),
                     "ticks_remaining": ticks_remaining - 1}
        return (agent_figure(eng), no_update, no_update, new_state,
                no_update, no_update, no_update, no_update, no_update)

    eng = dict_to_eng(state["engine_data"])

    counts = None
    for _ in range(steps_per_tick):
        counts = eng.step()
        if eng.is_over():
            break

    metrics = [
        metric_card("📅 Day",         counts["day"],  TX2),
        metric_card("🟢 Susceptible", counts["S"],    C_S),
        metric_card("🟡 Exposed",     counts["E"],    C_E),
        metric_card("🔴 Infected",    counts["I"],    C_I),
        metric_card("⚫ Recovered",   counts["R"],    C_R),
        metric_card("💀 Dead",        counts["D"],    C_D),
    ]

    new_state = {"running": True, "engine_data": eng_to_dict(eng),
                 "ticks_remaining": skip_ticks}

    if eng.is_over():
        banner_txt = (
            f"✅  Epidemic over — Day {counts['day']}  |  "
            f"Peak: {eng.peak_infected()}  |  "
            f"Deaths: {eng.total_deaths()}  |  "
            f"Recovered: {counts['R']}"
        )
        ban_sty = {
            "background": "#0f2d1a", "border": "1px solid #166534",
            "color": "#4ade80", "borderRadius": "8px",
            "padding": "10px 16px", "fontSize": "13px",
            "fontFamily": "'IBM Plex Mono', monospace",
        }
        return (agent_figure(eng), curve_figure(eng), metrics,
                {"running": False, "engine_data": None}, True,
                "▶  SIMULATE", {**BTN_BASE, "background": C_S},
                banner_txt, ban_sty)

    return (agent_figure(eng), curve_figure(eng), metrics,
            new_state, no_update, no_update, no_update, no_update, {"display": "none"})


# ── Initial metrics ───────────────────────────────────────────────────────────
@app.callback(
    Output("metrics-row", "children", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call='initial_duplicate',
)
def init_metrics(_):
    return [
        metric_card("📅 Day",         "—", TX2),
        metric_card("🟢 Susceptible", "—", C_S),
        metric_card("🟡 Exposed",     "—", C_E),
        metric_card("🔴 Infected",    "—", C_I),
        metric_card("⚫ Recovered",   "—", C_R),
        metric_card("💀 Dead",        "—", C_D),
    ]


# ── Preset Engine Callback ────────────────────────────────────────────────────
@app.callback(
    Output("sl-radius",           "value"),
    Output("sl-beta",             "value"),
    Output("sl-incubation-range", "value"),
    Output("sl-gamma",            "value"),
    Output("sl-mort",             "value"),
    Output("sl-radius-display",           "children"),
    Output("sl-beta-display",             "children"),
    Output("sl-incubation-range-display", "children"),
    Output("sl-gamma-display",            "children"),
    Output("sl-mort-display",             "children"),
    Input("preset-dropdown", "value"),
    prevent_initial_call=True,
)
def apply_disease_preset(preset_key):
    if preset_key == "custom" or preset_key not in DISEASE_PRESETS:
        return [no_update] * 10

    p = DISEASE_PRESETS[preset_key]
    r, b, i, g, m = p["radius"], p["beta"], p["incub"], p["gamma"], p["mort"]

    return (
        r, b, i, g, m,
        f"{r:.2f}",
        f"{b:.2f}",
        f"{i[0]}–{i[1]}",
        f"{g:.0f}",
        f"{m:.2f}",
    )


# ── Expand modal callback ─────────────────────────────────────────────────────
@app.callback(
    Output("modal-overlay",  "style"),
    Output("graph-modal",    "figure"),
    Output("modal-title",    "children"),
    Input("btn-expand-agents", "n_clicks"),
    Input("btn-expand-curve",  "n_clicks"),
    Input("btn-close-modal",   "n_clicks"),
    State("graph-agents",      "figure"),
    State("graph-curve",       "figure"),
    prevent_initial_call=True,
)
def toggle_modal(n_agents, n_curve, n_close, fig_agents, fig_curve):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    trigger = ctx.triggered[0]["prop_id"]

    _open = {
        "display": "flex", "position": "fixed",
        "top": "0", "left": "0", "width": "100vw", "height": "100vh",
        "background": "rgba(13,17,23,0.88)", "backdropFilter": "blur(6px)",
        "zIndex": "10000", "alignItems": "center", "justifyContent": "center",
    }
    _closed = {"display": "none"}

    if "btn-close-modal" in trigger:
        return _closed, no_update, no_update
    if "btn-expand-agents" in trigger:
        return _open, fig_agents, "AGENT MAP"
    if "btn-expand-curve" in trigger:
        return _open, fig_curve, "EPIDEMIC CURVE"
    return no_update, no_update, no_update


if __name__ == "__main__":
    app.run(debug=False, port=8050)