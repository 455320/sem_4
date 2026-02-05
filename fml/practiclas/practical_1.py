"""
Practical 1: NumPy & Matplotlib Operations
Unit: VI | Hours: 4

Objectives:
- Perform various NumPy operations (arrays, mathematical operations, indexing, slicing)
- Create visualizations using Matplotlib (plots, charts, graphs)
"""

import numpy as np 
import random 
import matplotlib.pyplot as plt


zeros_array = np.zeros((3,3))
print(zeros_array)

print("\n")

ones_array = np.ones((3,3))
print(ones_array)

print("\n")

random_array = np.random.randint(0,10,(3,3))
print(random_array)

print("\n")




indexing_array = random_array[1,1]
print(indexing_array)

print("\n")

slicing_array = random_array[0:2,0:2]
print(slicing_array)

print("\n")



plt.plot(random_array)
plt.show()

