import pandas as pd
from typing import List, Optional

"""
    Carga un CSV y retorna el DataFrame numérico (sin las columnas especificadas) 
    y el DataFrame completo.
    
    Args:
        ruta (str): Ruta del archivo CSV.
        columnas (List[str], opcional): Columnas a excluir del DataFrame numérico.
    
    Returns:
        df_numerico (pd.DataFrame): DataFrame solo con variables numéricas.
        df (pd.DataFrame): DataFrame completo original.
    """

def cargar_csv(ruta: str, columnas: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    if columnas:
        df_numerico = df.drop(columns=columnas)
    else:
        df_numerico = df
    return df_numerico,df