import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pd_settings import configure_pandas_display
configure_pandas_display()

# ------------------------------
# 1. Setup & File Loading
# ------------------------------
eh = pd.read_csv("eh_sorted.csv", encoding="utf-8-sig")

# ------------------------------
# 2. Rename & Clean Columns
# ------------------------------

# eh.columns = ['title','published','views','likes','comments','video id','duration','url','series name','is series','episode type','episode number']

# eh['title'] = eh['title'].str.replace(r'- Extra History ', '', regex=True)
eh['published'] = pd.to_datetime(eh['published'])
eh['year'] = eh['published'].dt.year.astype(str)

# ------------------------------
# 3. Derivative Columns
# ------------------------------

today = pd.Timestamp.today()
eh['daysSince'] = (today - eh['published']).dt.days
eh['viewsPerDay'] = (eh['views'] / eh['daysSince']).round(0)

# Episode number within series
# eh['epNo'] = eh.groupby('series').cumcount() + 1
eh['seriesLength'] = eh.groupby('series name')['series name'].transform('count')

# Final episode per series
last_eps = eh.groupby('series name')['published'].idxmax()
last_eps_df = eh.loc[last_eps].sort_values('published')
last_eps_df['seriesNo'] = range(1, len(last_eps_df) + 1)

# Merge series number back to main df
eh = eh.merge(last_eps_df[['series name', 'seriesNo']], on='series name', how='left')

# Convert year to ordered category
years_ordered = sorted(eh['year'].unique())
eh['year'] = pd.Categorical(eh['year'], categories=years_ordered, ordered=True)

# ------------------------------
# 4. Summary: Views by Series
# ------------------------------

sum_views = (
    eh.groupby(['seriesNo', 'series name'])
    .agg(seriesViews=('views', 'sum'))
    .reset_index()
    .sort_values('seriesNo')
)

eh_epis = eh[eh['episode type'] == 'Episode'].copy()
eh_epis = eh_epis[eh_epis['episode type'] == 'Episode'].copy()

# have to convert to int(y) b/c ['year'] is a categorical type 
eh_epis["odd even year"] = eh_epis["year"].apply(lambda y: "Even Year" if int(y) % 2 == 0 else "Odd Year")

eh_epis['episode number'] = eh_epis['episode number'].astype(int)

# Get today's date (normalized to remove time)
today = pd.Timestamp.today().normalize()

# Calculate difference in days
eh_epis['days since published'] = (today - eh_epis['published']).dt.days

# print(eh_epis[['title', 'series name', 'episode number']])
# print(eh_epis.columns)

import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_daq as daq


# --- Assumes you already have 'eh_epis' DataFrame preloaded ---
# Columns: ["series name", "views", "odd even year", "episode number", "title"]

# Configuration
height_per_series = 20
color_map = {"Odd Year": "red", "Even Year": "blue"}

# Y-axis numeric mapping
series_order = eh_epis["series name"].unique()[::-1]
series_to_num = {name: i for i, name in enumerate(series_order)}
eh_epis["y_numeric"] = eh_epis["series name"].map(series_to_num)

# Jitter
np.random.seed(42)
vertical_jitter_strength = 0.2
horizontal_jitter_strength = 0.05

eh_epis["y_jittered"] = eh_epis["y_numeric"] + np.random.uniform(
    -vertical_jitter_strength, vertical_jitter_strength, size=len(eh_epis)
)
eh_epis["x_jittered"] = eh_epis["views"] * (
    1 + np.random.uniform(-horizontal_jitter_strength, horizontal_jitter_strength, size=len(eh_epis))
)

# Episode options
available_episodes = sorted(eh_epis["episode number"].unique())

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Views by Series (Toggle Scale & Episodes)"),

    html.Div([
        html.Label("X-Axis Scale:"),
        daq.ToggleSwitch(
            id="xaxis-scale",
            label=["Linear", "Log"],
            value=True,  # True = log, False = linear
            persistence=True,
            persistence_type="local",  # Remember across reloads
            style={"margin-bottom": "10px"}
        ),
    ]),

    dcc.Checklist(
    id="episode-selector",
    options=[{"label": f"Ep {ep}", "value": ep} for ep in available_episodes],
    value=available_episodes,
    inline=True,
    inputStyle={"margin-right": "5px", "margin-left": "10px"},
    persistence=True,
    persistence_type="local"  # persists across browser reloads
    ),

    dcc.Graph(id="scatter-plot", config={"displayModeBar": True})
])


@app.callback(
    Output("scatter-plot", "figure"),
    [Input("episode-selector", "value"),
     Input("xaxis-scale", "value")]
)
def update_figure(selected_episodes, toggle_value):
    # Convert toggle value to axis type
    xaxis_type = "log" if toggle_value else "linear"

    filtered_df = eh_epis[eh_epis["episode number"].isin(selected_episodes)]

    fig = go.Figure()

    for group, df_group in filtered_df.groupby("odd even year"):
        fig.add_trace(go.Scatter(
            x=df_group["x_jittered"],
            y=df_group["y_jittered"],
            text=df_group["episode number"].astype(str),
            mode="text",
            name=f"{group}",
            textfont=dict(color=color_map.get(group, "black"), size=12),
            hovertext=[
                f"{row['title']}<br>"
                f"Published: {row['published']} ({row['days since published']} days ago)<br>"
                f"Views: {row['views']:,}<br>"
                f"Likes: {row['likes']:,}<br>"
                f"Comments: {row['comments']:,}<br>"
                f"Views/day: {row['views']/row['days since published']:.2f}<br>"
                f"Views/likes: {row['views']/row['likes']:.2f}<br>"
                for _, row in df_group.iterrows()
            ],
            hoverinfo="text"
        ))

    min_y = -0.5
    max_y = len(series_order) - 0.5
    fig_height = max(len(series_order) * height_per_series, 400)

    fig.update_layout(
        height=fig_height,
        yaxis=dict(
            tickmode="array",
            tickvals=list(series_to_num.values()),
            ticktext=list(series_to_num.keys()),
            range=[min_y, max_y],
            autorange=False,
            automargin=True,
            title="Series"
        ),
        xaxis=dict(
            title="Views",
            type=xaxis_type
        ),
        legend_title_text="Year Type",
        margin=dict(l=220, r=20, t=40, b=40),
        uirevision="static"
    )

    return fig



if __name__ == "__main__":
    app.run(debug=True)
