
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def aplicar_kmeans(df: pd.DataFrame, n_clusters: int , random_state: int ):
    """
    Aplica KMeans a un DataFrame numérico, escala los datos,
    entrena el modelo y devuelve los datos con clusters y los centroides originales.

    Args:
        df (pd.DataFrame): DataFrame con solo variables numéricas
        n_clusters (int): número de clusters
        random_state (int): semilla para reproducibilidad

    Returns:
        df_clusters (pd.DataFrame): DataFrame con columna "Cluster"
        df_centroides (pd.DataFrame): centroides en escala original
        model (KMeans): modelo entrenado
        scaler (StandardScaler): escalador entrenado
    """
    
    # Escalado
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Modelo KMeans
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    grupos = model.fit_predict(df_scaled)
    
    # DataFrame con clusters
    df_clusters = df.copy()
    df_clusters["Cluster"] = grupos
    
    # Centroides en escala original
    centroides_escalados = model.cluster_centers_
    centroides = scaler.inverse_transform(centroides_escalados)
    df_centroides = pd.DataFrame(centroides, columns=df.columns)
    df_centroides["Cluster"] = df_centroides.index
    
    return df_clusters, df_centroides, model, scaler