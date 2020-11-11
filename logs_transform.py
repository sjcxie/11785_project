import re

in_filename = "./111"
out_filename = "./trades_pgdl2_1.csv"
res = []

with open(in_filename) as f:
	text = f.read()
	for match in re.finditer(r".*train_accuracy=([0-9.]*), train_adv_accuracy=([0-9.]*), train_clean_accuracy=([0-9.]*), train_loss=([0-9.]*)]\n.*val_accuracy=([0-9.]*), val_adv_accuracy=([0-9.]*), val_clean_accuracy=([0-9.]*), val_loss=([0-9.]*)", text):
		res.append(",".join(match.groups()))

with open(out_filename, "w") as f:
	print(len(res))
	f.write("train_accuracy,train_adv_accuracy,train_clean_accuracy,train_loss,val_accuracy,val_adv_accuracy,val_clean_accuracy,val_loss\n")
	for item in res:
		f.write(item + "\n")

