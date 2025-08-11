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

print(eh_epis[['title', 'series name', 'episode number']])

import plotly.express as px

fig = px.strip(
    eh,
    x="views",
    y="series name",
    color="year",  # Custom column based on year (odd/even)
    hover_name="title",
    stripmode='overlay'
)
fig.show()
