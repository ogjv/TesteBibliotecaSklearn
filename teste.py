import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Criar dados de exemplo: valores de X e seus correspondentes Y
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Criar um modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo usando os dados
modelo.fit(X.reshape(-1, 1), Y)

# Fazer previsões usando o modelo treinado
previsoes = modelo.predict(X.reshape(-1, 1))

# Visualizar os resultados
plt.scatter(X, Y, label='Dados reais', color='blue')
plt.plot(X, previsoes, label='Previsões', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
