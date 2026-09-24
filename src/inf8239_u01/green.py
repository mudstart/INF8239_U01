"""Utilidades Green AI: frontera de Pareto desempeño vs. costo."""
import pandas as pd


def pareto_flags(df, score="f1_macro", cost="fit_median_s"):
    """Marca qué filas están en la frontera de Pareto (no dominadas).

    Una fila está dominada si existe otra igual o mejor en `score` y
    igual o más barata en `cost`, y estrictamente mejor en al menos uno.
    Devuelve una lista de booleanos (True = pertenece a la frontera).
    """
    flags = []
    for _, row in df.iterrows():
        dominated = ((df[score] >= row[score]) & (df[cost] <= row[cost]) &
          ((df[score] > row[score]) | (df[cost] < row[cost]))).any()
        flags.append(not bool(dominated))
    return flags