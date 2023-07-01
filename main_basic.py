from rasa.model_training import train_nlu
import asyncio
from rasa.model_testing import test_nlu

async def run_test_nlu(modelo, test_data, predictions_folder):
    await test_nlu(model=modelo, nlu_data = test_data, 
                   output_directory = predictions_folder, additional_arguments={'errors': True})  
    # El parámetro 'errors' : True indica que queremos que guarde el listado de errores que se han cometido al probar el modelo.

if __name__ == "__main__":
    
    domain_path = "domain.yml"
    config = 'spacy1_EXPERIMENTO1.yml' # Ruta al archido de configuración (donde está el pipeline)
    output_path_model = "models/"   # Carpeta donde se guardará el modelo entrenado
    output_path_test = 'spacy1_EXPERIMENTO4/' # Carpeta donde se guarda el report, la matriz de confusión y el histograma
    model_name = 'spacy1_EXPERIMENTO4' # nombre del modelo

    nlu_train_path = 'data_train.yml'  #ruta a los datos de entrenamiento
    nlu_test_path = 'data_test.yml'    #ruta a los datos de prueba
    
    
    modelo = train_nlu(config = config, nlu_data= nlu_train_path, output= output_path_model,
                        fixed_model_name = model_name)
    # Rendimiento del modelo sobre el conjunto de datos de prueba
    asyncio.run(run_test_nlu(modelo, nlu_test_path, output_path_test))

    # Si se quiere calcular el rendimiento del modelo sobre el conjunto de datos de entrenamiento para comprobar que no hay
    # overfitting o underfitting pueden ejecutarse las siguientes lineas
    # output_path_train = 'spacy1_EXPERIMENTO4_prueba/'  # carpeta donde se guarda el report, la matriz de confusión y el histograma de los datos de prueba
    # asyncio.run(run_test_nlu(modelo, nlu_train_path, output_path_train))
