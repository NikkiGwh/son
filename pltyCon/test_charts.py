import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio
from plotly.subplots import make_subplots
from plotly_templates import export_with_bugfix, load_templates

loaded_dpi = load_templates(
    # base_width_in_px=400,
    dpi=96
)

df = px.data.gapminder()
df_2007 = df.query("year==2007")


fig2 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["title" for _ in range(2)],
            )

fig2.update_layout(
    # template="oscilloscope+SC--3-4th-CW",
    template="oscilloscope+HC--2-3rd-CW"
)
fig2.update_xaxes(type="log")

fig0 = go.Scatter(
    
)

fig2.add_scatter(x=df_2007["gdpPercap"], y=df_2007["lifeExp"],
                 marker=dict(color=[index for index, _ in enumerate(df_2007["continent"])]),
                 mode= "markers",
                 row=1, col=1)
fig2.add_scatter(x=df_2007["gdpPercap"], y=df_2007["lifeExp"],
                 marker=dict(color=[index for index, _ in enumerate(df_2007["continent"])]),
                 mode= "markers",
                 row=1, col=2)

# fig2.add_scatter(x=df_2007["gdpPercap"], y=df_2007["lifeExp"],
#                  marker=dict(color=[index for index, _ in enumerate(df_2007["continent"])]),
#                  mode= "markers",
#                  row=2, col=1)
# fig2.add_scatter(x=df_2007["gdpPercap"], y=df_2007["lifeExp"],
#                  marker=dict(color=[index for index, _ in enumerate(df_2007["continent"])]),
#                  mode= "markers",
#                  row=2, col=2)
# fig2.add_scatter(x=df_2007["gdpPercap"], y=df_2007["lifeExp"],
#                  marker=dict(color=[index for index, _ in enumerate(df_2007["continent"])]),
#                  mode= "markers",
#                  row=3, col=1)
# fig2.add_scatter(x=df_2007["gdpPercap"], y=df_2007["lifeExp"],
#                  marker=dict(color=[index for index, _ in enumerate(df_2007["continent"])]),
#                  mode= "markers",
#                  row=3, col=2)



fig2.show()

export_with_bugfix(fig_handle=fig2, filename="test_einzeln.pdf")

