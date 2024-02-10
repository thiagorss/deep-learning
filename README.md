# README: Implementação de Rede Neural

## Visão Geral
Este código em Python fornece uma implementação básica de uma rede neural feedforward com camadas personalizáveis, funções de ativação e opções de treinamento. A rede neural é implementada usando classes para camadas, unidades (neurônios), funções de ativação e cálculo de gradientes.

## Componentes

### Classe Gradiente
- A classe `Gradiente` representa o gradiente da rede neural, contendo gradientes para pesos (`arr_dw`) e viés (`db`).

### Classe FuncaoAtivacao
- A classe `FuncaoAtivacao` define funções de ativação usadas na rede neural. Inclui a função de ativação em si, sua derivada e a derivada para a última camada, se aplicável.

### Classe Unidade
- A classe `Unidade` representa uma única unidade (neurônio) em uma camada da rede neural. Inclui métodos para propagação para frente e para trás, além de atualização de pesos.

### Classe Camada
- A classe `Camada` representa uma camada na rede neural. Contém várias unidades e facilita a propagação para frente e para trás através da camada.

### Classe RedeNeural
- A classe `RedeNeural` orquestra a rede neural. Permite configurar a rede com a arquitetura desejada, treiná-la e fazer previsões.

## Uso

1. **Inicialização**: Crie uma instância de `RedeNeural` com os parâmetros de arquitetura desejados, como o número de unidades por camada e funções de ativação.

```python
# Exemplo de Inicialização
arr_qtd_un_por_camada = [tamanho_entrada, tamanho_camada_oculta, tamanho_saida]
arr_func_a_por_camada = [funcao_ativacao_entrada, funcao_ativacao_oculta, funcao_ativacao_saida]
num_iteracoes = 1000

rede_neural = RedeNeural(arr_qtd_un_por_camada, arr_func_a_por_camada, num_iteracoes)
```

2. **Treinamento**: Treine a rede neural usando o método `fit` fornecendo os dados de entrada (`mat_x`) e os rótulos correspondentes (`arr_y`).

```python
# Exemplo de Treinamento
mat_x = ...  # Dados de entrada
arr_y = ...  # Rótulos

rede_neural.fit(mat_x, arr_y)
```

3. **Predição**: Após o treinamento, faça previsões usando o método `predict` com novos dados de entrada.

```python
# Exemplo de Predição
rotulos_previstos = rede_neural.predict(novos_dados)
```

## Personalização

- **Funções de Ativação**: Defina funções de ativação personalizadas criando instâncias da classe `FuncaoAtivacao` com funções e derivadas apropriadas.

- **Arquitetura da Rede**: Personalize a arquitetura da rede neural especificando o número de camadas, unidades por camada e funções de ativação para cada camada.

- **Parâmetros de Treinamento**: Ajuste os parâmetros de treinamento, como o número de iterações (`num_iteracoes`) e a taxa de aprendizado (`learning_rate`), para otimizar o desempenho.

## Requisitos

- Este código requer Python 3.x com a biblioteca NumPy instalada.

## Observações

- Esta implementação fornece um framework básico para entender redes neurais e pode ser estendida para aplicativos mais complexos.
- Garanta o pré-processamento adequado dos dados e a validação para tarefas de treinamento e previsão.