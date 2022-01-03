import numpy as np


def visualize_bin_predictions(texts, preds, correct, mean_pos_probas, sd_pos_probas=None,
                              visualization_save_path="pred_vis.html"):
    # mean_pos_probas... mean probabilities for positive label
    visualization_html_rows = []
    safe_sd_probas = np.zeros_like(mean_pos_probas, dtype=np.float32) if sd_pos_probas is None else sd_pos_probas

    visualization_html_rows.append("<h1>False positives (true = 0, pred = 1)</h1>")
    fp_indices = np.nonzero(np.logical_and(correct == 0, preds == 1))[0]
    visualization_html_rows.append(f"<h5>{fp_indices.shape[0]} examples</h5>")
    for idx_ex in fp_indices:
        visualization_html_rows.append(
            f"<div><i>P(y=PCL) = {mean_pos_probas[idx_ex]:.2f} ({safe_sd_probas[idx_ex]})</i>:"
            f"{texts[idx_ex]}"
            f"</div><br />")

    visualization_html_rows.append("<h1>False negatives (true = 1, pred = 0)</h1>")
    fn_indices = np.nonzero(np.logical_and(correct == 1, preds == 0))[0]
    visualization_html_rows.append(f"<h5>{fn_indices.shape[0]} examples</h5>")
    for idx_ex in fn_indices:
        visualization_html_rows.append(
            f"<div><i>P(y=PCL) = {mean_pos_probas[idx_ex]:.2f} ({safe_sd_probas[idx_ex]})</i>:"
            f"{texts[idx_ex]}"
            f"</div><br />")

    visualization_html_rows.append("<h1>True positives (true = 1, pred = 1)</h1>")
    tp_indices = np.nonzero(np.logical_and(correct == 1, preds == 1))[0]
    visualization_html_rows.append(f"<h5>{tp_indices.shape[0]} examples</h5>")
    for idx_ex in tp_indices:
        visualization_html_rows.append(
            f"<div><i>P(y=PCL) = {mean_pos_probas[idx_ex]:.2f} ({safe_sd_probas[idx_ex]})</i>:"
            f"{texts[idx_ex]}"
            f"</div><br />")

    visualization_html_rows.append("<h1>True negatives (true = 0, pred = 0)</h1>")
    tn_indices = np.nonzero(np.logical_and(correct == 0, preds == 0))[0]
    visualization_html_rows.append(f"<h5>{tn_indices.shape[0]} examples</h5>")
    for idx_ex in tn_indices:
        visualization_html_rows.append(
            f"<div><i>P(y=PCL) = {mean_pos_probas[idx_ex]:.2f} ({safe_sd_probas[idx_ex]})</i>:"
            f"{texts[idx_ex]}"
            f"</div><br />")

    with open(visualization_save_path, "w", encoding="utf-8") as f:
        print("<html><body>{}</body></html>".format("\n".join(visualization_html_rows)), file=f)


