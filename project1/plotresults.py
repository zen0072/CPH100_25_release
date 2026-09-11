import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("grid_results.csv")

plt.scatter(
    results["regularization_lambda"],
    results["train_loss"]
)

plt.xlabel("L2 regularization")
plt.ylabel("Training loss")
plt.title("L2 Regularization vs Training Loss")
#plt.show()
