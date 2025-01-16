import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import nltk
from nltk.corpus import stopwords

def load_data(folder_path, subyacentes_file):
    """
    Carga los datos de subyacentes y archivos de clientes desde una carpeta específica.
    
    Args:
        folder_path (str): Ruta a la carpeta que contiene los archivos
        subyacentes_file (str): Nombre del archivo de subyacentes
        
    Returns:
        tuple: (DataFrame de subyacentes, DataFrame combinado de clientes)
    """
    # Definir columnas para subyacentes
    columns = [
        'codigo_subyacente_caracteristica', 'descripcion', 'estado',
        'permitido_orf', 'desc_subya', 'desc_caracteristica',
        'desc_unidad', 'desc_empaque', 'desc_naturaleza'
    ]
    
    # Cargar datos de subyacentes
    subyacentes_path = os.path.join(folder_path, subyacentes_file)
    db_subyacentes = pd.read_excel(subyacentes_path, names=columns)
    
    # Cargar y combinar archivos de clientes
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx') and file not in [subyacentes_file, 'final_merged_data.xlsx']:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            dataframes.append(df)
    
    combined_df_clientes = pd.concat(dataframes, ignore_index=True)
    return db_subyacentes, combined_df_clientes

def clean_client_data(df):
    """
    Limpia y preprocesa el DataFrame de clientes.
    
    Args:
        df (pd.DataFrame): DataFrame de clientes original
        
    Returns:
        pd.DataFrame: DataFrame limpio
    """
    # Combinar columnas DESC y DESCRIPCION
    df['DESCRIPCION_CLIENTE'] = df['DESC'].fillna(df['DESCRIPCION'])
    df = df.drop(columns=['DESC', 'DESCRIPCION'])
    
    # Eliminar columnas con mayoría de NaN
    columns_to_drop = ['FECHA HORA', 'Nit. Cliente', 'CODIGO PROD. CLIENTE', 
                      'DESCRIPCION PRODUCTOS', 'TIPO ID', 'ID', 'FACTOR CONVERSION']
    df = df.drop(columns=columns_to_drop)
    
    # Convertir tipos de datos
    df['CODIGO BMC'] = df['CODIGO BMC'].fillna(0).astype('int')
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    return df

def standardize_text(text):
    """
    Estandariza el texto aplicando varias transformaciones.
    
    Args:
        text (str): Texto a estandarizar
        
    Returns:
        str: Texto estandarizado
    """
    if not isinstance(text, str):
        return text
        
    # Convertir a minúsculas
    text = text.lower()
    
    # Separar números y palabras
    text = re.sub(r'(\D)(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)(\D)', r'\1 \2', text)
    
    # Eliminar paréntesis
    text = text.replace('(', ' ').replace(')', ' ')
    
    # Limpiar espacios
    text = ' '.join(text.split())
    
    # Reemplazar abreviaturas comunes
    replacements = {
        r'\bg\b': 'gramos',
        r'\bgms\b': 'gramos',
        r'\bgrs\b': 'gramos',
        r'\bgr\b': 'gramos',
        r'\bkg\b': 'kilogramo',
        r'\bml\b': 'mililitros',
        r'\blt\b': 'litro',
        r'\bcc\b': 'mililitros',
        r'\bund\b': 'unidades',
        r'\buni\b': 'unidades'
    }
    
    for old, new in replacements.items():
        text = re.sub(old, new, text)
    
    # Eliminar caracteres especiales
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text

def remove_stopwords(text):
    """
    Elimina las stopwords del texto.
    
    Args:
        text (str): Texto de entrada
        
    Returns:
        str: Texto sin stopwords
    """
    if not isinstance(text, str):
        return text
        
    stop_words = set(stopwords.words('spanish'))
    words = text.split()
    clean_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(clean_words)

def main():
    """
    Función principal que ejecuta el pipeline completo de preprocesamiento.
    """
    # Descargar recursos necesarios de NLTK
    nltk.download('stopwords')
    
    # Definir rutas (ajustar según sea necesario)
    FOLDER_PATH = 'ruta/a/tu/carpeta/de/datos'
    SUBYACENTES_FILE = 'SUBYACENTES.xlsx'
    OUTPUT_FILE = 'datos_preprocesados.xlsx'
    
    # Cargar datos
    print("Cargando datos...")
    db_subyacentes, df_clientes = load_data(FOLDER_PATH, SUBYACENTES_FILE)
    
    # Limpiar datos de clientes
    print("Limpiando datos de clientes...")
    df_clientes_limpio = clean_client_data(df_clientes)
    
    # Preparar datos de subyacentes para el merge
    print("Preparando datos para la integración...")
    db_subyacentes = db_subyacentes.rename(columns={
        'descripcion': 'DESCRIPCION_BMC',
        'desc_subya': 'DESCRIPCION_SUBYACENTE',
        'codigo_subyacente_caracteristica': 'CODIGO BMC'
    })
    
    # Integrar datos
    print("Integrando datos...")
    resultado_df = pd.merge(
        df_clientes_limpio,
        db_subyacentes[['CODIGO BMC', 'DESCRIPCION_BMC', 'DESCRIPCION_SUBYACENTE']],
        on='CODIGO BMC',
        how='left'
    )
    
    # Eliminar filas con NaN
    resultado_df = resultado_df.dropna()
    
    # Aplicar estandarización de texto
    print("Estandarizando textos...")
    for col in ['DESCRIPCION_CLIENTE', 'DESCRIPCION_BMC', 'DESCRIPCION_SUBYACENTE']:
        resultado_df[col] = resultado_df[col].apply(standardize_text)
        resultado_df[col] = resultado_df[col].apply(remove_stopwords)
    
    # Guardar resultados
    print("Guardando resultados...")
    resultado_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Proceso completado. Resultados guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()