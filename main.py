import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ustawienie rysunków
plt.rc('font', size=10)
plt.rc('axes', labelsize=10, titlesize=10)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# 0cieżki dostępu
IMAGES_PATH = Path() / "images" / "training_linear_models"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# zapisywanie grafik
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# 1) Generowanie zaszumionego sygnału sinusoidalnego
LiczbaCykli = 5  # Modyfikuj liczbę cykli od 1 do 5 i porównuj wyniki,
x = np.linspace(0, LiczbaCykli * np.pi, 1000)
y_true = np.sin(x)
PoziomSzumu = 0.1  # Modyfikuj Zaszumienie i porównuj wyniki,
noise = np.random.normal(0, PoziomSzumu, len(x))
y_noisy = y_true + noise

# 2a) Dopasowanie modelu Fourier'a do dopasowania
def fourier_model(params, x, y):
    a0, a1, b1, w = params
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) - y
# Definicja funkcji modelu dla predykcji
def fourier_predict(x, params):
    a0, a1, b1, w = params
    return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w)

# Początkowe punkty dla parametrów modelu - estymowane ręcznie :) można poszukać lepszych
initial_guess = [0, 0, 0, 0.0628947478196155]
# Dopasowanie modelu do danych
params_Fourier = least_squares(fourier_model, initial_guess, args=(x, y_noisy))
print(params_Fourier)

# 2b) Definicja modelu funkcji sinusoidalnej
def sin1_model(params, x, y):
    A, omega, phi, offset = params
    return A * np.sin(omega * x + phi) + offset - y
# Definicja funkcji modelu dla predykcji
def sin1_predict(x, params):
    A, omega, phi, offset = params
    return A * np.sin(omega * x + phi) + offset

# Początkowe punkty dla dopasowania - estymowane ręcznie :) można poszukać lepszych
initial_guess = [0.99, 0.0628, -0.0539, 0]
# Dopasowywanie modelu do danych
params_sin = least_squares(sin1_model, initial_guess, args=(x, y_noisy))
print(params_sin)

# 2c)  Dopasowanie modelu regresji liniowej do danych wielomianowych
poly_degree = 5  # Stopień wielomianu, zmodyfikuj i porównaj wyniki
poly_features = PolynomialFeatures(degree=poly_degree)
x_poly = poly_features.fit_transform(x.reshape(-1, 1))
modelPoly = LinearRegression()
modelPoly.fit(x_poly, y_noisy)

# 2d)  Dopasowanie modelu regresji z wykorzystaniem NARNET
# Normalizacja danych
scaler = MinMaxScaler(feature_range=(0, 1))
y_noisy_scaled = scaler.fit_transform(y_noisy.reshape(-1, 1))

# Konwersja danych do formatu przyjmowanego przez sieć RNN w Keras  czyli modelu autogresywnego
def create_dataset(data, n):
    X, y = [], []
    for i in range(n, len(data)):
         X.append(data[i - n:i])
         y.append(data[i])
    return np.array(X), np.array(y)

# Liczba poprzednich kroków używanych do prognozy wmodelu typu AR
look_back = 3  # feedbackDelays - zmodyfikuj od 1 do 10 i porównaj wyniki
X, Y = create_dataset(y_noisy_scaled, look_back)

# na dalsze potrzeby odwracamy normalizację
y_true = scaler.inverse_transform(Y.reshape(-1, 1))

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Definiowanie modelu sieci NARNET o 10 neuronach w pojedynczej warstwie ukrytej typu feed forward - zmodyfikuj/poeksperymentuj
model_NARNET = Sequential([
    SimpleRNN(10, input_shape=(look_back, 1)),  # hiddenLayerSize = 10
    Dense(1)
])
model_NARNET.compile(optimizer='adam', loss='mean_squared_error')
# Trenowanie modelu
model_NARNET.fit(X, Y, epochs=10, batch_size=1, verbose=0)
# Przewidywanie



# 2e)  Dopasowanie modelu regresji z wykorzystaniem perceptronu
model_perceptron = Sequential([Dense(1, input_dim=look_back, activation='linear')])
model_perceptron.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
model_perceptron.fit(X, Y, epochs=10, verbose=0)

# 3) Predykcja modelu
y_pred_poly = modelPoly.predict(x_poly)
y_pred_sin = sin1_predict(x, params_sin.x)
y_pred_Fourier = fourier_predict(x, params_Fourier.x)
y_pred_scaled = model_NARNET.predict(X)
y_pred_NARNET = scaler.inverse_transform(y_pred_scaled) # transformacja sporwrotem
y_pred_scaled = model_perceptron.predict(X)
y_pred_perceptron = scaler.inverse_transform(y_pred_scaled)  # transformacja sporwrotem

# 4) Obliczanie R^2 i RMSE

r2 = r2_score(y_noisy, y_pred_Fourier)
rmse = np.sqrt(mean_squared_error(y_noisy, y_pred_Fourier))

r2_poly = r2_score(y_noisy, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_noisy, y_pred_poly))

r2_sin = r2_score(y_noisy, y_pred_sin)
rmse_sin = np.sqrt(mean_squared_error(y_noisy, y_pred_sin))

r2Perc = r2_score(y_true, y_pred_perceptron.flatten())
rmsePerc = np.sqrt(mean_squared_error(y_true, y_pred_perceptron.flatten()))

r2NARNN = r2_score(y_true, y_pred_NARNET)
rmseNARNN = np.sqrt(mean_squared_error(y_true, y_pred_NARNET))

# 5) Obliczanie reszt, rysowanie wykresów
residuals_furier = y_noisy - y_pred_Fourier
residuals_poly = y_noisy - y_pred_poly
residuals_sin = y_noisy - y_pred_sin
residuals_perc = y_true.flatten() - y_pred_perceptron.flatten()  # e(k) = a(k) - â(k)
residuals_NARNET = y_true - y_pred_NARNET

# Tworzenie subplotów
fig, axs = plt.subplots(3, 1, figsize=(8, 6))
#% dopasowanie x-ów dla perceptronu i narneta
x_axis_for_predictions = x[look_back:len(y_pred_NARNET) + look_back]

# Dane i model
axs[0].plot(x, y_noisy, label='Dane zaszumione', linestyle='--', alpha=0.5)
axs[0].plot(x, y_pred_Fourier, label='Fourier', color='red')
axs[0].plot(x, y_pred_poly, label=f'Poly {poly_degree}', color='green')
axs[0].plot(x, y_pred_sin, label='Sinus', color='blue')
axs[0].plot( x_axis_for_predictions,y_pred_perceptron, label='Perceptron', color='black')
axs[0].plot( x_axis_for_predictions,y_pred_NARNET, label='NARNET', color='yellow')
axs[0].set_title('Dane i model dopasowany')
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=5)

# Reszty
axs[1].plot(x, residuals_sin, label='Sin')
axs[1].plot(x, residuals_poly, label='Poly')
axs[1].plot(x, residuals_furier, label='Fourier')
axs[1].plot(x_axis_for_predictions, residuals_perc, label='Perceptron')
axs[1].plot(x_axis_for_predictions, residuals_NARNET, label='NARNET')

axs[1].set_title('Reszty')
axs[1].hlines(0, x[0], x[-1], colors='r', linestyles='--')
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=5)

# Histogram reszt
axs[2].hist(residuals_sin, bins=30, label='Sin', color='green',rwidth=0.8,)
axs[2].hist(residuals_poly, bins=30, label='Poly', color='red',rwidth=0.7,)
axs[2].hist(residuals_furier, bins=30, label='Fourier', color='skyblue',rwidth=0.6,)
axs[2].hist(residuals_perc, bins=30, label='Perceptron', color='black',rwidth=0.5,)
axs[2].hist(residuals_NARNET, bins=30, label='NARNET', color='yellow',rwidth=0.3,)
axs[2].set_title('Histogram reszt')
axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=5)

plt.tight_layout()
save_fig("regresja_SIN")
plt.show()

# Wypisanie statystyk (bez zmian)
print(f'R^2 (model sinusoidalny): {r2_sin}')
print(f'R^2 (model wielomianowy): {r2_poly}')
print(f'R^2 (model Fourierowski): {r2 }')
print(f'R^2 (model Perceptron): {r2Perc }')
print(f'R^2 (model Perceptron): {r2NARNN }')
print(f'RMSE (model sinusoidalny): {rmse_sin}')
print(f'RMSE (model wielomianowy): {rmse_poly}')
print(f'RMSE (model Fourierowski): {rmse}')
print(f'RMSE (model Perceptron): {rmsePerc}')
print(f'RMSE (model Perceptron): {rmseNARNN}')