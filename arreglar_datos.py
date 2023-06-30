# En este documento tomamos los datos del archivo nlu.xls, eliminamos los enlaces 
# y las menciones y lo escribimos en formato YALM.
import pandas as pd
import re


def clean_text(text):
    pattern1 = re.compile('((www\.[^\s]+)|(https?://[^\s]+))')
    # Reemplazar los enlaces con una cadena vacía
    text = re.sub(pattern1, '', text)


    pattern2 = re.compile(r'@[\w\d_]+')
    # Reemplazar las menciones con una cadena vacía
    text = re.sub(pattern2, '', text)

    # Quitar espacios en blanco al principio y al final
    text = text.strip()
    return text

excel_data = pd.read_excel("nlu.xlsx")

data = pd.DataFrame(excel_data, columns=['id', 'tweet', 'misogynous'])

intents_list = ['no_misos', 'misos']

file_name = "nlu.yml"
with open(file_name, 'w', encoding='utf-8') as file:
    file.write("version: '3.1'\n\nnlu:\n")
    for intent in intents_list:
        j=0
        file.write(f"- intent: {intent}\n")
        file.write("  examples: |\n")
        for i in range(0, len(data.id)):
            if data.misogynous[i] == intents_list.index(intent):
                file.write(f"    - {clean_text(data.tweet[i])}\n")
                j=j+1
        print('para el intent', intent, 'tenemos ', j, 'ejemplos')
