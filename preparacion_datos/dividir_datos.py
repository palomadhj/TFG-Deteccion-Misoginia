from sklearn.model_selection import train_test_split
import argparse
import yaml

def split_nlu_data(args, nlu_data_path):
    # Cargar los datos de entrenamiento NLU
    with open(nlu_data_path, 'r', encoding='utf-8') as file:
        nlu_data = yaml.safe_load(file)

    # Obtener ejemplos de frases y etiquetas
    examples = []
    labels = []
    for intent_data in nlu_data['nlu']:
        intent = intent_data['intent']
        string_examples = intent_data['examples'].replace('- ', '')
        list_examples = string_examples.split('\n')
        list_examples.remove('')
        for example in list_examples:
            examples.append(example)
            labels.append(intent)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(examples, labels, test_size=args.test_size, random_state=args.random_state)

    print('the length of train data is', len(y_train))
    print('the length of test data is', len(y_test))
    # Pasar los datos a la función para escribirlos en un archivo YALM
    write_rasa_yaml(y_train, X_train, 'data_train.yml')
    write_rasa_yaml(y_test, X_test, 'data_test.yml')

    return

def write_rasa_yaml(labels, examples, output_file):
    # Obtener los intents únicos a partir de las etiquetas
    intents = []
    [intents.append(label) for label in labels if label not in intents]

    nlu_data=[]

    # Crear los datos del NLU para cada intento
    for intent in intents:
        examples_new = []
        [examples_new.append(text) for text in examples if intent == labels[examples.index(text)]]
        intent_data = {
            'intent': intent,
            'examples': examples_new
        }
        nlu_data.append(intent_data)

    # Escribir los datos en el archivo YAML
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("version: '3.1'\n\nnlu:\n")

        # Escribir cada intento y sus ejemplos
        for item in nlu_data:
            file.write(f"- intent: {item['intent']}\n")
            file.write("  examples: |\n")
            for example in item['examples']:
                file.write(f"    - {example}\n")
            file.write("\n")
    return

if __name__ == "__main__":
    
    nlu_data_path = "nlu.yml"

    parser = argparse.ArgumentParser(description="Split NLU Data")
    
    # Agregar argumentos
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for data splitting")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for data splitting")
    
    # Parsear los argumentos de la línea de comandos
    args = parser.parse_args()
    
    # Llamar a la función split_nlu_data con los argumentos procesados
    split_nlu_data(args, nlu_data_path)

    nlu_train_path = 'data_train.yml'
    nlu_test_path = 'data_test.yml'
