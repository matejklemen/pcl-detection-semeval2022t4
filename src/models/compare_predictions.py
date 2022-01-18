import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
""" A script for comparing two models in terms of differently classified examples. """


def create_table(column_names, rows):
	table_header = "<thead>\n<tr>\n{}\n</tr>\n</thead>".format(
		"\n".join([f"<th>{curr_col}</th>" for curr_col in column_names])
	)

	table_rows = []
	for curr_row in rows:
		table_rows.append("<tr>\n" + "\n".join([f"<td>{el}</td>" for el in curr_row]) + "\n</tr>")

	table_html = "<table class='table table-hover'><thead>{}</thead><tbody>{}</tbody></table>".format(
		table_header, "\n".join(table_rows)
	)
	return table_html


# TODO: Appropriately modify these for your models
model1 = "/home/matej/Documents/multiview-pcl-detection/models/pcla_baseline_roberta_base1e-5_158"
model2 = "/home/matej/Documents/multiview-pcl-detection/models/pcla_aftertr_labelprobas_roberta_base_mcd50_1e-5_158"
TARGET_PATH = "model_comparison.html"

if model1.endswith(os.path.sep):
	model1 = model1[:-1]

if model2.endswith(os.path.sep):
	model2 = model2[:-1]

model1_name = model1.split(os.path.sep)[-1]
model2_name = model2.split(os.path.sep)[-1]

result_dir = f"diff_{model1_name}_{model2_name}"
if not os.path.exists(result_dir):
	os.makedirs(result_dir)

m1_better = []
m2_better = []
all_tp_diff = []
all_tn_diff = []
for idx_fold in range(10):
	pred1_path = os.path.join(model1, f"{model1_name}_f{idx_fold}", f"pred_dev_f{idx_fold}.tsv")
	m1_preds = pd.read_csv(pred1_path, sep="\t")

	pred2_path = os.path.join(model2, f"{model2_name}_f{idx_fold}", f"pred_dev_f{idx_fold}.tsv")
	m2_preds = pd.read_csv(pred2_path, sep="\t")

	data_path = os.path.join(model1, f"{model1_name}_f{idx_fold}", f"dev_f{idx_fold}.tsv")
	data = pd.read_csv(data_path, sep="\t")

	assert m1_preds.shape[0] == m2_preds.shape[0]
	correct_labels = data["binary_label"].values
	is_pos = correct_labels == 1
	is_neg = correct_labels == 0
	ex_inds = np.arange(correct_labels.shape[0], dtype=int)

	m1_labels = m1_preds["pred_binary_label"].values
	m2_labels = m2_preds["pred_binary_label"].values

	# TP/TN/FP/FNs of model2 that differ from model1
	tp_diff = list(set(ex_inds[np.logical_and(is_pos, m2_labels == 1)]) -
				   set(ex_inds[np.logical_and(is_pos, m1_labels == 1)]))
	tn_diff = list(set(ex_inds[np.logical_and(is_neg, m2_labels == 0)]) -
				   set(ex_inds[np.logical_and(is_neg, m1_labels == 0)]))
	fp_diff = list(set(ex_inds[np.logical_and(is_neg, m2_labels == 1)]) -
				   set(ex_inds[np.logical_and(is_neg, m1_labels == 1)]))
	fn_diff = list(set(ex_inds[np.logical_and(is_pos, m2_labels == 0)]) -
				   set(ex_inds[np.logical_and(is_pos, m1_labels == 0)]))
	for _idx in ex_inds[tp_diff]:
		all_tp_diff.append([data.iloc[_idx]["label"], m1_labels[_idx], "", m2_labels[_idx], "", data.iloc[_idx]["text"], data.iloc[_idx]["keyword"]])
	for _idx in ex_inds[tn_diff]:
		all_tn_diff.append([data.iloc[_idx]["label"], m1_labels[_idx], "", m2_labels[_idx], "", data.iloc[_idx]["text"], data.iloc[_idx]["keyword"]])

	diff_m2_better = np.logical_and(m1_labels != m2_labels, m2_labels == correct_labels)
	diff_m1_better = np.logical_and(m1_labels != m2_labels, m1_labels == correct_labels)

	for _idx in ex_inds[diff_m2_better]:
		m2_better.append([data.iloc[_idx]["label"], m1_labels[_idx], "", m2_labels[_idx], "", data.iloc[_idx]["text"], data.iloc[_idx]["keyword"]])
	for _idx in ex_inds[diff_m1_better]:
		m1_better.append([data.iloc[_idx]["label"], m1_labels[_idx], "", m2_labels[_idx], "", data.iloc[_idx]["text"], data.iloc[_idx]["keyword"]])

	m1_confusion = confusion_matrix(y_true=correct_labels, y_pred=m1_labels)
	m1_metrics = {
		"f1_score": f1_score(y_true=correct_labels, y_pred=m1_labels, pos_label=1, average='binary'),
		"p_score": precision_score(y_true=correct_labels, y_pred=m1_labels, pos_label=1, average='binary'),
		"r_score": recall_score(y_true=correct_labels, y_pred=m1_labels, pos_label=1, average='binary')
	}

	m2_confusion = confusion_matrix(y_true=correct_labels, y_pred=m2_labels)
	m2_metrics = {
		"f1_score": f1_score(y_true=correct_labels, y_pred=m2_labels, pos_label=1, average='binary'),
		"p_score": precision_score(y_true=correct_labels, y_pred=m2_labels, pos_label=1, average='binary'),
		"r_score": recall_score(y_true=correct_labels, y_pred=m2_labels, pos_label=1, average='binary')
	}
	print(f"Fold #{idx_fold}:")

	print("--[model1]--")
	print(f"P={m1_metrics['p_score']:.3f}, R={m1_metrics['r_score']:.3f}, F1={m1_metrics['f1_score']:.3f}")
	print(m1_confusion)
	print("------------")
	print("--[model2]--")
	print(f"P={m2_metrics['p_score']:.3f}, R={m2_metrics['r_score']:.3f}, F1={m2_metrics['f1_score']:.3f}")
	print(m2_confusion)
	print("------------")
	print("\n\n")

# sort by uncertain 5-scale label
m2_better = [m2_better[_i] for _i in np.argsort(list(map(lambda _l: _l[0], m2_better)))]
m1_better = [m1_better[_i] for _i in np.argsort(list(map(lambda _l: _l[0], m1_better)))]
all_tp_diff = [all_tp_diff[_i] for _i in np.argsort(list(map(lambda _l: _l[0], all_tp_diff)))]
all_tn_diff = [all_tn_diff[_i] for _i in np.argsort(list(map(lambda _l: _l[0], all_tn_diff)))]

m2_better_tab_html = \
	create_table(["Uncertain-label", "M1", "P<sub>M1</sub>(+)", "M2", "P<sub>M2</sub>(+)", "Text", "Keyword"],
				 rows=m2_better)

m1_better_tab_html = \
	create_table(["Uncertain-label", "M1", "P<sub>M1</sub>(+)", "M2", "P<sub>M2</sub>(+)", "Text", "Keyword"],
				 rows=m1_better)

tp_diff_tab_html = \
	create_table(["Uncertain-label", "M1", "P<sub>M1</sub>(+)", "M2", "P<sub>M2</sub>(+)", "Text", "Keyword"],
				 rows=all_tp_diff)

tn_diff_tab_html = \
	create_table(["Uncertain-label", "M1", "P<sub>M1</sub>(+)", "M2", "P<sub>M2</sub>(+)", "Text", "Keyword"],
				 rows=all_tn_diff)

s = \
"""
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Model comparison</title>
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
</head>
<body style="font-size:18px;">
<div class="container">
<div class="row">
<h1> Model comparison</h1>
<h2> Where M2 gets it right </h2>
... and M1 does not. ({} examples)
{}

<h2> Where M1 gets it right </h2>
... and M2 does not. ({} examples)
{}

<h2>TP diff</h2>
TPs that M2 gets that M1 doesn't ({} examples)
{}

<h2>TN diff</h2>
TNs that M2 gets that M1 doesn't ({} examples)
{}
</div>
</div>
</body>
</html>
""".format(len(m2_better), m2_better_tab_html,
		   len(m1_better), m1_better_tab_html,
		   len(all_tp_diff), tp_diff_tab_html,
		   len(all_tn_diff), tn_diff_tab_html)

print(f"Saving comparison to '{TARGET_PATH}'")
with open(TARGET_PATH, "w", encoding="utf-8") as f:
	print(s, file=f)







