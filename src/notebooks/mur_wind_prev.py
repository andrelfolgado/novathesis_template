from gemlib.database.utils import load_data_oracle, load_data_hub
import pandas as pd
# import warnings to avoid getting sqlalchemy warnings from load_data_hub
import warnings

warnings.filterwarnings('ignore')

ZONES = (
    "22Bolo", "24Ama1", "26Fanh", "27Arr1", "29Sob2", "44Milagre", "48Vale_de", "60Maravil", "61MaraviII", "68Jogui"
)


def get_wind_prev_pivot(zones_names, hist_num_year):
    df_zones_props = get_facilities_props(zones_names)

    df_index = get_wind_prev_hist(zones_names)

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
        columns=['dia_aplicacao', 'hora_aplicacao', 'zonadescr', 'unidade', 'percentil', 'nomezona', 'modelo_descr'],
        inplace=True)
    df_index.sort_index(inplace=True)

    # calculate the mean of the duplicates,
    # keeping the original index and nome_modelo eliminating the duplicates reset the nome_modelo index
    df_silver = (
        df_index
        .groupby(['datetime', 'nome_modelo'])
        .mean()
        .reset_index(level='nome_modelo')
    )

    df_silver['installed_capacity'] = df_silver['nome_modelo'].map(
        df_zones_props.set_index('FACILITY_UNIT_ID_SHORT')['INSTALLED_CAPACITY']
    ) / 1000

    df_silver['capacity_factor'] = df_silver['valor'] / df_silver['installed_capacity']

    df_silver.drop(columns=['installed_capacity', 'valor'], inplace=True)

    df_silver_pivot = df_silver.pivot(columns='nome_modelo', values='capacity_factor')

    return df_silver_pivot


def get_wind_prev_hist(zones_names):
    orc_wind_prev_hist = f"""
        SELECT *
        FROM POWERBI_EOL_PREV_HIST
        WHERE NOME_MODELO IN {zones_names}
    """
    df = load_data_oracle(orc_wind_prev_hist)
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
    return df_zones_props
