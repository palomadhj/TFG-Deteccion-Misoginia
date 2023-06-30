from rasa.model_training import train_nlu
import asyncio
from rasa.model_testing import test_nlu

async def run_test_nlu(modelo, test_data, predictions_folder):
    await test_nlu(model=modelo, nlu_data = test_data, 
                   output_directory = predictions_folder, additional_arguments={})

if __name__ == "__main__":
    
    domain_path = "domain.yml"
    config = 'spacy1_EXPERIMENTO1.yml'
    output_path_model = "models/"
    #output_path_train = 'spacy1_train/'
    output_path_test = 'spacy1_EXPERIMENTO4/'
    model_name = 'spacy1_EXPERIMENTO4'

    nlu_train_path = 'data_train.yml'
    nlu_test_path = 'data_test.yml'
    
    
    modelo = train_nlu(config = config, nlu_data= nlu_train_path, output= output_path_model,
                        fixed_model_name = model_name)
    # Rendimiento del modelo sobre el conjunto de datos de entrenamiento
    
    # asyncio.run(run_test_nlu(modelo, nlu_train_path, output_path_train))

    # Rendimiento del modelo sobre el conjunto de datos de prueba
    asyncio.run(run_test_nlu(modelo, nlu_test_path, output_path_test))