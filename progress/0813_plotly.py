import plotly.express as px
import pandas as pd
import numpy as np
from palmerpenguins import  load_penguins

penguins = load_penguins()
penguins.head()


fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color= "species"
)

fig.show()

# subplot 관련 패키지 설치
from plotly.subplots import make_subplots

fig_subplot = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Adelie', 'Gentoo', 'Chinstrap')
)

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Adelie"')['bill_length_mm'],
   'y' : penguins.query('species=="Adelie"')['bill_depth_mm'],
   'name' : 'Adelie'
  },
  row=1, col=1
 )

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Gentoo"')['bill_length_mm'],
   'y' : penguins.query('species=="Gentoo"')['bill_depth_mm'],
   'name' : 'Gentoo'
  },
  row=1, col=2
 )

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Chinstrap"')['bill_length_mm'],
   'y' : penguins.query('species=="Chinstrap"')['bill_depth_mm'],
   'name' : 'Chinstrap'
  },
  row=1, col=3
 )
 