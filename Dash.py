# %%
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

# %%
eh_epis = eh[eh['episode type'] == 'Episode'].copy()
eh_epis = eh_epis[eh_epis['episode type'] == 'Episode'].copy()

# have to convert to int(y) b/c ['year'] is a categorical type 
eh_epis["odd even year"] = eh_epis["year"].apply(lambda y: "Even Year" if int(y) % 2 == 0 else "Odd Year")

eh_epis['episode number'] = eh_epis['episode number'].astype(int)

# print(eh_epis[['title', 'series name', 'episode number']])
# print(eh_epis.columns)


# %%
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output

# Replace this with your actual `eh_epis` DataFrame
np.random.seed(42)
n_samples = 100
eh_epis = pd.DataFrame({
    "series name": np.random.choice([f"Series {i}" for i in range(10)], size=n_samples),
    "views": np.random.lognormal(mean=10, sigma=1, size=n_samples),
    "odd even year": np.random.choice(["Odd Year", "Even Year"], size=n_samples),
    "episode number": np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples),
    "title": [f"Episode {i}" for i in range(n_samples)]
})

# Prepare categorical y-axis
series_order = eh_epis["series name"].unique()[::-1]
series_to_y = {s: i for i, s in enumerate(series_order)}
eh_epis["y_numeric"] = eh_epis["series name"].map(series_to_y)

# Jitter
vertical_jitter_strength = 0.2
horizontal_jitter_strength = 0.05
eh_epis["y_jittered"] = eh_epis["y_numeric"] + np.random.uniform(-vertical_jitter_strength, vertical_jitter_strength, len(eh_epis))
eh_epis["x_jittered"] = eh_epis["views"] * (1 + np.random.uniform(-horizontal_jitter_strength, horizontal_jitter_strength, len(eh_epis)))

# Color map
color_map = {"Odd Year": "red", "Even Year": "blue"}
episode_numbers = [1, 2, 3, 4, 5, 6]
fig_height = len(series_order) * 20

# Build figure function
def create_figure(selected_eps):
    fig = go.Figure()
    for ep_num in selected_eps:
        df_ep = eh_epis[eh_epis["episode number"] == ep_num]
        for group, df_group in df_ep.groupby("odd even year"):
            fig.add_trace(go.Scatter(
                x=df_group["x_jittered"],
                y=df_group["y_jittered"],
                text=[str(ep_num)] * len(df_group),
                mode="text",
                name=f"Ep {ep_num}",
                textfont=dict(color=color_map.get(group, "black"), size=12),
                hovertext=df_group["title"],
                hoverinfo="text"
            ))
    fig.update_layout(
        height=fig_height,
        title="Views by Series (Log Scale, Multiselect Toggle)",
        xaxis=dict(title="Views", type="log"),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(series_order))),
            ticktext=list(series_order),
            range=[-0.5, len(series_order) - 0.5]
        )
    )
    return fig

# Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.H2("Toggle Episode Numbers"),
    dcc.Checklist(
        id="episode-checklist",
        options=[{"label": f"Episode {i}", "value": i} for i in episode_numbers],
        value=episode_numbers,
        labelStyle={"display": "inline-block", "margin-right": "10px"}
    ),
    dcc.Graph(id="episode-graph", figure=create_figure(episode_numbers))
])

@app.callback(
    Output("episode-graph", "figure"),
    Input("episode-checklist", "value")
)
def update_figure(selected_eps):
    return create_figure(selected_eps)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)



