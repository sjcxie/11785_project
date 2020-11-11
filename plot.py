import numpy as np
import matplotlib.pyplot as plt

layers_pgdinf = np.loadtxt("./layers_pgdinf.csv", delimiter=",", skiprows=1)
trades_pgdinf = np.loadtxt("./trades_pgdinf.csv", delimiter=",", skiprows=1)
layers_pgdl2_1 = np.loadtxt("./layers_pgdl2_1.csv", delimiter=",", skiprows=1)
trades_pgdl2_1 = np.loadtxt("./trades_pgdl2_1.csv", delimiter=",", skiprows=1)


def train_val_accuracy_over_attacks():
	plt.figure(figsize=(14, 6))

	plt.subplot(121)
	plt.plot(layers_pgdinf[:,0])
	plt.plot(layers_pgdinf[:,4])
	plt.plot(trades_pgdinf[:,0])
	plt.plot(trades_pgdinf[:,4])
	plt.title("PGD INF")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC train", "LDC val", "TRADES train", "TRADES val"], loc="upper left")

	plt.subplot(122)
	plt.plot(layers_pgdl2_1[:,0])
	plt.plot(layers_pgdl2_1[:,4])
	plt.plot(trades_pgdl2_1[:,0])
	plt.plot(trades_pgdl2_1[:,4])
	plt.title("PGD L2")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC train", "LDC val", "TRADES train", "TRADES val"], loc="upper left")

	plt.savefig("./train_val_accuracy_over_attacks.png")


def val_accuracy_over_clean_and_adv():
	plt.figure(figsize=(18, 12))

	plt.subplot(231)
	plt.plot(layers_pgdinf[:,6])
	plt.plot(trades_pgdinf[:,6])
	plt.title("Validation Accuracy over Clean Data - PGD INF")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC", "TRADES"], loc="upper left")

	plt.subplot(232)
	plt.plot(layers_pgdinf[:,5])
	plt.plot(trades_pgdinf[:,5])
	plt.title("Validation Accuracy over Adversarial Data - PGD INF")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC", "TRADES"], loc="upper left")

	plt.subplot(233)
	plt.plot(layers_pgdinf[:,4])
	plt.plot(trades_pgdinf[:,4])
	plt.title("Total Accuracy - PGD INF")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC", "TRADES"], loc="upper left")

	plt.subplot(234)
	plt.plot(layers_pgdl2_1[:,6])
	plt.plot(trades_pgdl2_1[:,6])
	plt.title("Validation Accuracy over Clean Data - PGD L2")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC", "TRADES"], loc="upper left")

	plt.subplot(235)
	plt.plot(layers_pgdl2_1[:,5])
	plt.plot(trades_pgdl2_1[:,5])
	plt.title("Validation Accuracy over Adversarial Data - PGD L2")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC", "TRADES"], loc="upper left")

	plt.subplot(236)
	plt.plot(layers_pgdl2_1[:,4])
	plt.plot(trades_pgdl2_1[:,4])
	plt.title("Total Accuracy - PGD L2")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["LDC", "TRADES"], loc="upper left")

	plt.savefig("./val_accuracy_over_clean_and_adv.png")


if __name__ == '__main__':
	train_val_accuracy_over_attacks()
	val_accuracy_over_clean_and_adv()
