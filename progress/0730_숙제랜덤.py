!pip install gspread oauth2client pandas
import pandas as pd

gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet={Sheet2}"
df = pd.read_csv(gsheet_url)
df.head()
