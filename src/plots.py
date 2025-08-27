import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_medias_cluster(medias_cluster: pd.DataFrame):
    """
    Genera gráficos de barras para cada variable mostrando la media por clúster.
    
    Args:
        medias_cluster (pd.DataFrame): DataFrame con medias por clúster.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Edad
    sns.barplot(data=medias_cluster, x="Cluster", y="Edad", ax=axes[0], palette="tab10")
    axes[0].set_title("Edad Promedio por Clúster")
    axes[0].set_ylabel("Edad")
    
    # Ingresos
    sns.barplot(data=medias_cluster, x="Cluster", y="Ingresos Anueales (k$)", ax=axes[1], palette="tab10")
    axes[1].set_title("Ingresos Promedios por Clúster")
    axes[1].set_ylabel("Ingresos Anuales (k$)")
    
    # Puntuación de Gastos
    sns.barplot(data=medias_cluster, x="Cluster", y="Puntuacion Gastos (1-100)", ax=axes[2], palette="tab10")
    axes[2].set_title("Puntuación Promedio de Gastos por Clúster")
    axes[2].set_ylabel("Puntuación de Gastos")
    
    plt.tight_layout()
    plt.show()


def plot_relaciones_variables(df_clusters: pd.DataFrame):
    """
    Genera un pairplot para observar relaciones entre variables numéricas coloreadas por clúster.
    
    Args:
        df_clusters (pd.DataFrame): DataFrame con columna 'Cluster'.
    """
    sns.pairplot(df_clusters, hue="Cluster", palette="tab10")
    plt.suptitle("Relaciones entre Variables Numéricas por Clúster", y=1.02)
    plt.show()


def plot_matriz_correlacion_por_cluster(df_clusters: pd.DataFrame):
    """
    Genera un heatmap de la matriz de correlación para cada clúster.
    
    Args:
        df_clusters (pd.DataFrame): DataFrame con columna 'Cluster'.
    """
    for i in sorted(df_clusters["Cluster"].unique()):
        plt.figure(figsize=(8, 6))
        cluster_corr = df_clusters[df_clusters["Cluster"] == i].drop(columns=["Cluster"]).corr()
        sns.heatmap(cluster_corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title(f"Matriz de Correlación - Clúster {i}")
        plt.show()