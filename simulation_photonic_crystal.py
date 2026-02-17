import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

n_Si = 3.5
lambda0 = 330e-9
omega0 = 2 * np.pi * constants.c / lambda0

a = 150e-9
r = 0.3 * a

N_periods = 10
L = N_periods * a

defect_position = N_periods // 2

Nx, Ny, Nz = 100, 100, 100
x = np.linspace(-L / 2, L / 2, Nx)
y = np.linspace(-L / 2, L / 2, Ny)
z = np.linspace(-L / 2, L / 2, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

epsilon = np.ones((Nx, Ny, Nz))

for i in range(-N_periods // 2, N_periods // 2 + 1):
    for j in range(-N_periods // 2, N_periods // 2 + 1):
        x0 = i * a
        y0 = j * a

        if i == 0 and j == 0:
            continue

        dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

        mask = (dist <= r) & (np.abs(Z) <= L / 2)
        epsilon[mask] = n_Si ** 2

X_rel = X - x[defect_position]
Y_rel = Y - y[defect_position]
Z_rel = Z - z[defect_position]
R = np.sqrt(X_rel ** 2 + Y_rel ** 2 + Z_rel ** 2)

sigma_x = 100e-9
sigma_y = 100e-9
sigma_z = 100e-9

field = np.exp(-(X_rel / sigma_x) ** 2 - (Y_rel / sigma_y) ** 2 - (Z_rel / sigma_z) ** 2)

t = np.linspace(0, 2e-12, 1000)
field_t = np.exp(-t / 5e-10) * np.cos(omega0 * t)

alpha_Si = 1e4
loss_absorption = np.exp(-alpha_Si * L)

roughness = 1e-9
loss_scattering = np.exp(-(4 * np.pi * roughness / lambda0) ** 2)

total_loss = loss_absorption * loss_scattering

Q = 2 * np.pi / (1 - total_loss)
print(f"Расчётная добротность: Q = {Q:.2e}")

tau_photon = Q / omega0
print(f"Время жизни фотона: τ = {tau_photon * 1e12:.2f} пс")

E_pulse = 1e-6

V_mode = (2 * np.pi) ** (3 / 2) * sigma_x * sigma_y * sigma_z
print(f"Объём моды: V = {V_mode:.2e} м³")

rho_peak = (E_pulse * Q) / (2 * np.pi * V_mode)
print(f"Пиковая плотность энергии: ρ = {rho_peak:.2e} Дж/м³")

rho_threshold = 1.97e8
print(f"Порог рождения пары: {rho_threshold:.2e} Дж/м³")

if rho_peak >= rho_threshold:
    print("✅ Порог ДОСТИГНУТ!")
else:
    ratio = rho_threshold / rho_peak
    print(f"❌ Порог НЕ достигнут. Нужно увеличить параметры в {ratio:.1f} раз")

E_field = np.sqrt(2 * rho_peak / (constants.epsilon_0 * n_Si ** 2))
E_schwinger = 1.3e18
print(f"Напряжённость поля: E = {E_field:.2e} В/м")
print(f"Доля от швингеровского предела: {E_field / E_schwinger:.2e}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(field[:, :, Nz // 2].T, extent=[x[0] * 1e9, x[-1] * 1e9, y[0] * 1e9, y[-1] * 1e9],
           cmap='hot', origin='lower')
plt.colorbar(label='Нормированная интенсивность |E|²')
plt.xlabel('x (нм)')
plt.ylabel('y (нм)')
plt.title('Распределение поля (сечение z=0)')

plt.subplot(1, 3, 2)
plt.imshow(field[Nx // 2, :, :].T, extent=[y[0] * 1e9, y[-1] * 1e9, z[0] * 1e9, z[-1] * 1e9],
           cmap='hot', origin='lower')
plt.colorbar(label='Нормированная интенсивность |E|²')
plt.xlabel('y (нм)')
plt.ylabel('z (нм)')
plt.title('Распределение поля (сечение x=0)')

plt.subplot(1, 3, 3)
profile_x = field[:, Ny // 2, Nz // 2]
plt.plot(x * 1e9, profile_x, 'b-', linewidth=2)
plt.xlabel('x (нм)')
plt.ylabel('|E|² (норм.)')
plt.title('Профиль поля вдоль оси x')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_field.png', dpi=300)
print("\n✅ График 1 сохранён: simulation_field.png")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t * 1e12, field_t, 'b-', linewidth=1)
plt.xlabel('Время (пс)')
plt.ylabel('Напряжённость поля (отн. ед.)')
plt.title('Временная эволюция поля в дефектной моде')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
energy_t = np.exp(-t / tau_photon)
plt.semilogy(t * 1e12, energy_t, 'r-', linewidth=2)
plt.xlabel('Время (пс)')
plt.ylabel('Энергия в моде (отн. ед.)')
plt.title(f'Затухание энергии (τ = {tau_photon * 1e12:.2f} пс)')
plt.grid(True, alpha=0.3, which='both')

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

popt, _ = curve_fit(exp_decay, t[:500], energy_t[:500], p0=[1, tau_photon])
print(f"Аппроксимированное время жизни: τ_fit = {popt[1] * 1e12:.2f} пс")

plt.tight_layout()
plt.savefig('simulation_temporal.png', dpi=300)
print("✅ График 2 сохранён: simulation_temporal.png")

plt.figure(figsize=(10, 6))

dt = t[1] - t[0]
freq = np.fft.fftfreq(len(t), dt)
spectrum = np.abs(np.fft.fft(field_t))

positive = freq > 0
freq_pos = freq[positive] / 1e12
spectrum_pos = spectrum[positive] / spectrum[positive].max()

plt.plot(freq_pos, spectrum_pos, 'b-', linewidth=2)
plt.axvline(x=omega0 / (2 * np.pi) / 1e12, color='r', linestyle='--', label='Резонансная частота')
plt.xlabel('Частота (ТГц)')
plt.ylabel('Спектральная амплитуда (норм.)')
plt.title('Спектр дефектной моды')
plt.grid(True, alpha=0.3)
plt.legend()

half_max = 0.5
idx = np.where(spectrum_pos >= half_max)[0]
if len(idx) > 0:
    fwhm = freq_pos[idx[-1]] - freq_pos[idx[0]]
    print(f"Полуширина спектра: Δf = {fwhm:.2f} ТГц")
    print(f"Добротность из спектра: Q = {freq_pos[idx[0]] / fwhm:.2e}")

plt.tight_layout()
plt.savefig('simulation_spectrum.png', dpi=300)
print("✅ График 3 сохранён: simulation_spectrum.png")

print("\n" + "=" * 50)
print("ИТОГИ МОДЕЛИРОВАНИЯ")
print("=" * 50)
print(f"Материал: кремний (n = {n_Si})")
print(f"Длина волны: {lambda0 * 1e9:.1f} нм")
print(f"Добротность: Q = {Q:.2e}")
print(f"Объём моды: V = {V_mode:.2e} м³")
print(f"Плотность энергии: ρ = {rho_peak:.2e} Дж/м³")
print(f"Порог рождения пары: {rho_threshold:.2e} Дж/м³")
print(f"Напряжённость поля: E = {E_field:.2e} В/м")
print(f"Швингеровский предел: {E_schwinger:.2e} В/м")
print("=" * 50)
