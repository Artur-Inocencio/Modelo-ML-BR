from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

# Configuração da API Flask
app = Flask(__name__)
CORS(app)

# Função para normalizar os dados de entrada
def preprocess_input(data):
    # Ajustar para incluir as colunas corretas no DataFrame de entrada
    input_data = pd.DataFrame({
        'Pessoas': [data['Pessoas']],
        'Veículos': [data['Veículos']],
        #'Data e Hora': [pd.Timestamp(data['Data e Hora'])],
        'Sentido': [data['Sentido']],
        'Clima': [data['Clima']],
        'Pista': [data['Pista']],
        'Traçado': [data['Traçado']],
        'UF': [data['UF']],
        'BR': [data['BR']]
    })
    return input_data

# Carregar o modelo treinado
with open('./modelo-prf-br.pkl', 'rb') as model_file:
    modelo_completo = pickle.load(model_file)

@app.route('/prever', methods=['POST'])
def prever_probabilidade():
    try:
        # Receber dados da solicitação em formato JSON
        dados = request.get_json() 
        # Pré-processar os dados
        input_data = preprocess_input(dados)
        if modelo_completo is None: 
            print("Dados:", input_data) 
        # Fazer a previsão
        probabilidade_ileso = modelo_completo.predict_proba(input_data)[:, 1]
        
        # Construir a resposta
        resultado = {
            'probabilidade_ileso': float(probabilidade_ileso[0])
        }

        # Retornar a resposta como JSON
        return jsonify(resultado)
    except Exception as e:
        # Tratar erros e retornar mensagem de erro
        return jsonify({'erro': str(e)}), 400

# Executar o servidor Flask
if __name__ == '__main__':
    app.run(debug=True)
