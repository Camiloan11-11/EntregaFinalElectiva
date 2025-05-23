import pandas as pd
from prophet import Prophet
import pickle
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TarifaPredictor:
    def __init__(self):
        self.models = {}
        self.municipios = []
        
    def load_data(self, file_path):
        """Cargar y procesar los datos"""
        print("Cargando datos...")
        self.df = pd.read_csv(file_path)
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        self.municipios = sorted(self.df['Municipio'].unique())
        print(f"Datos cargados. Municipios encontrados: {len(self.municipios)}")
        
    def prepare_data_for_municipio(self, municipio):
        """Preparar datos para un municipio específico"""
        df_municipio = self.df[self.df['Municipio'].str.lower() == municipio.lower()]
        
        if df_municipio.empty:
            raise ValueError(f"No se encontraron datos para el municipio: {municipio}")
            
        # Agrupar por mes y calcular promedio
        df_municipio_prom = df_municipio.groupby('Fecha')['Cargo Fijo'].mean().reset_index()
        df_municipio_prom.rename(columns={'Fecha': 'ds', 'Cargo Fijo': 'y'}, inplace=True)
        
        return df_municipio_prom
    
    def train_model_for_municipio(self, municipio):
        """Entrenar modelo Prophet para un municipio"""
        print(f"Entrenando modelo para {municipio}...")
        
        # Preparar datos
        data = self.prepare_data_for_municipio(municipio)
        
        # Crear y entrenar modelo
        model = Prophet(
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=10.0,
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=False
        )
        
        model.fit(data)
        
        # Guardar modelo y datos
        self.models[municipio] = {
            'model': model,
            'training_data': data,
            'last_date': data['ds'].max(),
            'mean_value': data['y'].mean()
        }
        
        return model
    
    def train_all_models(self):
        """Entrenar modelos para todos los municipios"""
        print("Iniciando entrenamiento de todos los modelos...")
        
        for municipio in self.municipios:
            try:
                self.train_model_for_municipio(municipio)
                print(f"✓ Modelo entrenado para {municipio}")
            except Exception as e:
                print(f"✗ Error entrenando modelo para {municipio}: {str(e)}")
        
        print(f"Entrenamiento completado. Modelos entrenados: {len(self.models)}")
    
    def evaluate_model(self, municipio, test_months=6):
        """Evaluar modelo usando validación temporal"""
        if municipio not in self.models:
            raise ValueError(f"Modelo no encontrado para {municipio}")
            
        data = self.models[municipio]['training_data']
        
        # Dividir datos para validación
        split_date = data['ds'].iloc[-test_months]
        train_data = data[data['ds'] < split_date]
        test_data = data[data['ds'] >= split_date]
        
        if len(test_data) == 0:
            print(f"No hay suficientes datos para validar {municipio}")
            return None
        
        # Entrenar modelo con datos de entrenamiento
        temp_model = Prophet(
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=10.0,
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=False
        )
        temp_model.fit(train_data)
        
        # Predecir
        future = temp_model.make_future_dataframe(periods=len(test_data), freq='MS')
        forecast = temp_model.predict(future)
        
        # Calcular métricas
        y_true = test_data['y'].values
        y_pred = forecast.tail(len(test_data))['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'municipio': municipio
        }
    
    def predict(self, municipio, months_ahead=12):
        """Realizar predicción para un municipio"""
        if municipio not in self.models:
            raise ValueError(f"Modelo no encontrado para {municipio}")
            
        model = self.models[municipio]['model']
        
        # Crear dataframe futuro
        future = model.make_future_dataframe(periods=months_ahead, freq='MS')
        
        # Realizar predicción
        forecast = model.predict(future)
        
        # Extraer predicciones futuras
        future_forecast = forecast.tail(months_ahead)
        
        predictions = []
        for _, row in future_forecast.iterrows():
            predictions.append({
                'fecha': row['ds'].strftime('%Y-%m-%d'),
                'prediccion': round(row['yhat'], 2),
                'limite_inferior': round(row['yhat_lower'], 2),
                'limite_superior': round(row['yhat_upper'], 2)
            })
            
        return predictions
    
    def save_models(self, directory='trained_models'):
        """Guardar todos los modelos entrenados"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Guardar cada modelo individualmente
        for municipio, model_data in self.models.items():
            filename = f"{directory}/model_{municipio.lower().replace(' ', '_')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
        
        # Guardar lista de municipios
        with open(f"{directory}/municipios.pkl", 'wb') as f:
            pickle.dump(self.municipios, f)
            
        print(f"Modelos guardados en {directory}/")
    
    def load_models(self, directory='trained_models'):
        """Cargar modelos previamente entrenados"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directorio {directory} no encontrado")
            
        # Cargar lista de municipios
        with open(f"{directory}/municipios.pkl", 'rb') as f:
            self.municipios = pickle.load(f)
            
        # Cargar cada modelo
        for municipio in self.municipios:
            filename = f"{directory}/model_{municipio.lower().replace(' ', '_')}.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    self.models[municipio] = pickle.load(f)
                    
        print(f"Modelos cargados: {len(self.models)}")

def main():
    """Función principal para entrenar y guardar modelos"""
    try:
        # Inicializar predictor
        predictor = TarifaPredictor()
        
        # Cargar datos - ajustar la ruta según tu estructura
        data_file = "tarifas_con_indicadores.csv"
        if not os.path.exists(data_file):
            data_file = "../data/tarifas_con_indicadores.csv"
        if not os.path.exists(data_file):
            print("Error: No se encuentra el archivo tarifas_con_indicadores.csv")
            print("Asegúrate de que esté en el directorio actual o en ../data/")
            return
            
        predictor.load_data(data_file)
        
        # Entrenar modelos
        predictor.train_all_models()
        
        # Evaluar algunos modelos
        print("\nEvaluando modelos...")
        for i, municipio in enumerate(predictor.municipios[:3]):  # Evaluar primeros 3
            try:
                metrics = predictor.evaluate_model(municipio)
                if metrics:
                    print(f"{municipio}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
            except Exception as e:
                print(f"Error evaluando {municipio}: {e}")
        
        # Guardar modelos
        predictor.save_models()
        
        # Ejemplo de predicción
        if predictor.municipios:
            print(f"\nEjemplo de predicción para {predictor.municipios[0]}:")
            predictions = predictor.predict(predictor.municipios[0], months_ahead=6)
            for pred in predictions[:3]:
                print(f"  {pred['fecha']}: ${pred['prediccion']:.2f}")
        
        print("\n¡Entrenamiento completado!")
        print(f"Modelos guardados para {len(predictor.models)} municipios")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()