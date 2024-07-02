import os
import numpy as np
import matplotlib.pyplot as plt

result_dir =  "./result/numpy"

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith("label")]
lst_input = [f for f in lst_data if f.startswith("input")]
lst_output = [f for f in lst_data if f.startswith("output")]


lst_label.sort()
lst_input.sort()
lst_output.sort()


ids = 0

label = np.load(os.path.join(result_dir, lst_label[ids]))
inputs = np.load(os.path.join(result_dir, lst_input[ids]))
output = np.load(os.path.join(result_dir, lst_output[ids]))




plt.subplot(131)
plt.imshow(inputs, cmap="gray")
plt.title("Input")

plt.subplot(132)
plt.imshow(label, cmap="gray")
plt.title("Label")

plt.subplot(133)
plt.imshow(output, cmap="gray")
plt.title("Output")

plt.show()