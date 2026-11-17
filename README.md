# Winning Culers: D150 experimental candidate

Run `python predict.py --data-dir C:/path/to/data --output C:/path/to/output.csv` after installing requirements.txt. Inputs are the tactical ATD XML/TXT and automatic Smart Tags, under data/tactical. No video, broadcast, commentary, manual labels or challenge reference is used.

The classifier, features and H/A Viterbi cost16 are unchanged. The only decision change multiplies the existing signed D evidence by 1.5 before geometric decoding, then fixes that new D mask for H/A inference. This scales positive stoppage evidence AND negative automatic-tag evidence; external tests have no such tags, which remains a transfer risk. It is not a probability calibration. Kickoff overrides remain unchanged. X outside official graded intervals is preserved.

The checkpoint is trained on IDSSE J03WMX. D variants were compared on development J03WN1/J03WOH; D150 was frozen before testing new J03WPY. J03WOY was not reused for D tuning. All data from https://huggingface.co/datasets/pysport/idsse-data ; provider BallPossession/BallStatus mapping differs from private challenge labels. Same-provider held-out performance does not certify challenge accuracy.

Checkpoint provenance, training details, measured results and limitations are documented in TECHNICAL_REPORT.md in this folder. This package is experimental and does not overwrite previous candidates. Restricted match data and prediction CSVs must stay local. The optional --baseline-output writes the modified geometric result, not the original backup.
