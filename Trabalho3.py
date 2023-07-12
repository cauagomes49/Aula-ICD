from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()
X = wine.data
y = wine.target
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia: {:.2f}%".format(accuracy * 100))

novo_valor = [[13.2, 2.77, 2.51, 18.5, 96.6, 2.6, 2.51, 0.31, 1.29, 4.14, 1.06, 3.37, 1050]]

previsao = model.predict(novo_valor)

classes = wine.target_names
classe_prevista = classes[previsao[0]]

print("Classe prevista para o novo valor: ", classe_prevista)
