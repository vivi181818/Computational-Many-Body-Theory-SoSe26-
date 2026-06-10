import numpy as np
import matplotlib.pyplot as plt

U =2
Ed =-1

miu_values = np.linspace(-5, 5, 500)
B_values = [1, 3, 10]

def Occupation_Number(beta, miu_values, U, Ed):
    Z = (1
         + 2*np.exp(-beta*(Ed - miu_values))
         + np.exp(-beta*(2*Ed + U - 2*miu_values)))

    Ne = (2*np.exp(-beta*(Ed - miu_values))
          + 2*np.exp(-beta*(2*Ed + U - 2*miu_values))) / Z
    return Ne

plt.figure(figsize=(10, 6))

for beta in B_values:
    Ne = Occupation_Number(beta, miu_values, U, Ed)
    plt.plot(miu_values, Ne, label=f"β = {beta}")

plt.xlabel("Chemical potential μ")
plt.ylabel("Occupation number Nₑ")
plt.title("Occupation Number vs Chemical Potential (Hubbard Atom Model)")
plt.axhline(y=1, color='r', linestyle='--', label='Nₑ = 1')

plt.legend()
plt.grid()
plt.show()