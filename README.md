# Pronosticos
# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
import datetime

# Para subir archivo en Colab
from google.colab import files
uploaded = files.upload()

# Obtener el nombre del archivo subido
file_path = list(uploaded.keys())[0]

# === Paso 1: Cargar datos ===
df = pd.read_excel(file_path)
df['ds'] = pd.to_datetime(df['ds'])
df = df.set_index('ds').asfreq('D')

# Rellenar valores faltantes por interpolación
df['y'] = df['y'].interpolate(method='linear')

# === Paso 2: Crear variables de calendario ===
def create_calendar_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

# Definir festivos en México
class MexicoHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        Holiday('Constitution Day', month=2, day=5, observance=nearest_workday),
        Holiday('Benito Juarez', month=3, day=21, observance=nearest_workday),
        Holiday('Labor Day', month=5, day=1, observance=nearest_workday),
        Holiday('Independence Day', month=9, day=16, observance=nearest_workday),
        Holiday('Revolution Day', month=11, day=20, observance=nearest_workday),
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

holidays = MexicoHolidayCalendar().holidays(start=df.index.min(), end=df.index.max())
df['holiday'] = df.index.isin(holidays).astype(int)

# Añadir temporadas pico
def add_peak_season_flags(df):
    df['peak'] = 0
    for year in df.index.year.unique():
        df.loc[f'{year}-05-20':f'{year}-05-31', 'peak'] = 1  # Hot Sale
        df.loc[f'{year}-11-10':f'{year}-11-20', 'peak'] = 1  # Buen Fin
        df.loc[f'{year}-12-10':f'{year}-12-25', 'peak'] = 1  # Navidad
    return df

df = create_calendar_features(df)
df = add_peak_season_flags(df)

# === Paso 3: Modelo SARIMA ===
sarima_model = SARIMAX(df['y'], order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)
forecast_sarima = sarima_result.forecast(steps=30)

# === Paso 4: Modelo XGBoost para residuos ===
df['sarima_fitted'] = sarima_result.fittedvalues
df['residuals'] = df['y'] - df['sarima_fitted']

features = ['dayofweek', 'weekofyear', 'month', 'day', 'quarter', 'year', 'is_weekend', 'holiday', 'peak']
X = df[features]
y = df['residuals']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
xgb.fit(X_train, y_train)

# === Paso 5: Crear dataframe para fechas futuras y predecir ===
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame(index=future_dates)

future_df = create_calendar_features(future_df)
future_df['holiday'] = future_df.index.isin(holidays).astype(int)
future_df = add_peak_season_flags(future_df)

future_df['sarima'] = forecast_sarima.values

X_future = future_df[features]
future_df['residuals_xgb'] = xgb.predict(X_future)

# Pronóstico final
future_df['forecast'] = future_df['sarima'] + future_df['residuals_xgb']

# === Paso 6: Guardar resultados en Excel ===
output = future_df[['forecast']].reset_index()
output.columns = ['ds', 'y_forecast']
output.to_excel('pronostico_hibrido.xlsx', index=False)

print("Pronóstico generado y guardado en 'pronostico_hibrido.xlsx'")

