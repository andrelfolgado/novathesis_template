from gemlib.database import load_data_azure
from gemlib.database.utils import load_data_oracle, load_data_hub
import pandas as pd
# import warnings to avoid getting sqlalchemy warnings from load_data_hub
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def get_wind_prev_pivot(zones_names: tuple, hist_num_year: int):

    df_zones_props = get_facilities_props(zones_names)

    df_index = get_wind_prev_hist(zones_names)

    column_types = {
        "dia_aplicacao": "datetime64[ns]",
        "hora_aplicacao": np.uint8,
        "percentil": "category",
        "valor": "float64",
        "nomeZona": "string",
        "zonaDescr": "string",
        "unidade": "string",
        "nome_Modelo": "string",
        "modelo_Descr": "string"
    }

    df_index = df_index.astype(column_types)
    # df_index = df.copy()

    # filter datetime to get the last 2 years
    # hist_num_year = 4
    df_index = df_index[
        df_index['dia_aplicacao'] >= df_index['dia_aplicacao'].max() - pd.DateOffset(months=hist_num_year * 12)]

    # create datetimeindex based on dia_aplicacao(datetime) e hora_aplicacao(float)
    df_index['datetime'] = (
            pd.to_datetime(df_index['dia_aplicacao']) + pd.to_timedelta(df_index['hora_aplicacao'] - 1, unit='h')
    )

    df_index.set_index('datetime', inplace=True)

    df_index.drop(
        columns=['dia_aplicacao', 'hora_aplicacao', 'zonaDescr', 'unidade', 'percentil', 'nomeZona', 'modelo_Descr'],
        inplace=True)
    df_index.sort_index(inplace=True)

    # calculate the mean of the duplicates,
    # keeping the original index and nome_modelo eliminating the duplicates reset the nome_modelo index
    df_silver = (
        df_index
        .groupby(['datetime', 'nome_Modelo'])
        .mean()
        .reset_index(level='nome_Modelo')
    )

    df_silver['installed_capacity'] = df_silver['nome_Modelo'].map(
        df_zones_props.set_index('FACILITY_UNIT_ID_SHORT')['INSTALLED_CAPACITY']
    ) / 1000

    df_silver['capacity_factor'] = df_silver['valor'] / df_silver['installed_capacity']

    df_silver.drop(columns=['installed_capacity', 'valor'], inplace=True)

    df_silver_pivot = df_silver.pivot(columns='nome_Modelo', values='capacity_factor')

    # substitute nan values with 0, to fill projects that started later than historical data retrieval
    df_silver_pivot.fillna(0, inplace=True)

    return df_silver_pivot


def get_wind_prev_hist(zones_names):
    orc_wind_prev_hist = f"""
        SELECT *
        FROM POWERBI_EOL_PREV_HIST
        WHERE NOME_MODELO IN {zones_names}
    """
    df = load_data_azure(orc_wind_prev_hist)
    return df


def get_facilities_props(zones_names):
    dh_qry = """
        SELECT * 
        FROM raw_tdmi.lov_edpr_facilities
    """
    df_fac_props = load_data_hub(dh_qry)
    df_fac_props['facility_name'] = df_fac_props['FACILITY_UNIT_NAME'].apply(
        lambda x: x.split(' -')[0] if ' -' in x else x)
    df_fac_props['FACILITY_UNIT_ID_SHORT'] = df_fac_props['FACILITY_UNIT_ID'].apply(
        lambda x: '_'.join(x.split('_')[1:]) if '_' in x else x)
    df_zones_props = df_fac_props[df_fac_props['FACILITY_UNIT_ID_SHORT'].isin(zones_names)]

    df_zones_props = (
        df_zones_props
        .sort_values(by="UPDATED_DATE")
        .drop_duplicates(subset=["FACILITY_UNIT_ID_SHORT"], keep="last")
    )
    return df_zones_props
