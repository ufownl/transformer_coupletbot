import re
import numpy as np
import matplotlib.pyplot as plt

training_loss = []
validating_loss = []

regex = re.compile(".* training_loss (\S+).* validating_loss (\S+)")

while True:
    try:
        line = input()
    except EOFError:
        break
    m = regex.match(line)
    if m:
        training_loss.append(float(m.group(1)))
        validating_loss.append(float(m.group(2)))

plt.plot(np.array(training_loss), label="training loss")
plt.plot(np.array(validating_loss), label="validating loss")
plt.grid(True)
plt.legend()
plt.show()
