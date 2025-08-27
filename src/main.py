import pandas as pd
import seaborn as sns

from data_loader import cargar_csv
from clustering import aplicar_kmeans
from config import DATA_PATH, N_CLUSTERS, RANDOM_STATE,COLUMNAS
from plots import (
    plot_medias_cluster,
    plot_relaciones_variables,
    plot_matriz_correlacion_por_cluster
)

sns.set(style="whitegrid")  # Estilo de los gráficos


def main():
    # ============================
    # CARGA DE DATOS
    # ============================
    df_numerico, df_original = cargar_csv(DATA_PATH, columnas=COLUMNAS)

    # ============================
    # APLICACIÓN DE K-MEANS
    # ============================
    df_clusters, df_centroides, model, scaler = aplicar_kmeans(
        df_numerico, N_CLUSTERS, RANDOM_STATE
    )

    print("\nCentroides del modelo K-Means:")
    print(df_centroides)

    # ============================
    # ANÁLISIS EXPLORATORIO
    # ============================
    print("\nTipos de variables:")
    print(df_original.dtypes)

    print("\nEstadísticas descriptivas:")
    print(df_original.describe(include='all'))

    print("\nValores nulos por columna:")
    print(df_original.isnull().sum())

    # ============================
    # VISUALIZACIONES
    # ============================

    print("\nDistribución de variables por clúster:")
    medias_cluster = df_clusters.groupby("Cluster")[
        ["Edad", "Ingresos Anueales (k$)", "Puntuacion Gastos (1-100)"]
    ].mean().reset_index()

    plot_medias_cluster(medias_cluster)
    plot_relaciones_variables(df_clusters)
    plot_matriz_correlacion_por_cluster(df_clusters)


if __name__ == "__main__":
    main()