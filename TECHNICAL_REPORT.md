# Winning Culers — Technical report

More Than A Hack 2027, Initial Challenge. Prepared 17 November 2026.

Team: Ibtissam Benchayb, Khaoula El Hattabi, Othman Moussaoui, Hafsa Aziz.

## Approach

We estimate a frame-level H/A/D/X sequence from tactical tracking and automatic Smart Tags. A geometric detector estimates dead-ball intervals, a supervised binary classifier estimates the controlling team during predicted-live intervals, and Viterbi decoding discourages unsupported fragmentation. This CPU pipeline produces 199,319 rows on the tactical video timeline. No manual match corrections are applied.

The geometric detector combines distances to the pitch boundary, ball motion, missingness following a near-boundary observation, collective player slowing and automatic restart context. Ball absence alone does not imply D. The selected D150 variant multiplies the existing signed dead-ball evidence by 1.5 before geometric decoding. This strengthens positive stoppage cues and negative tag evidence; it is not probability calibration. Automatic kickoff overrides are unchanged.

The HistGradientBoostingClassifier uses 21 features: per-team nearest-player distances and motion, missingness, boundary distances, distances to the last observed ball position and observation age. Predictions are averaged with team-swapped predictions for team symmetry. H/A decoding maximizes the sum of log probabilities minus 16 per team switch, separately inside each predicted-live interval. D intervals therefore reset H/A decoding; D is not learned by this classifier. X is emitted outside the published graded intervals.

## Inputs, attempted alternatives and possession conventions

Inference uses the tactical ATD XML/TXT and automatic Smart Tag XML. ATD row indices are converted to zero-based video indices. Normalized coordinates are converted to metres using 105.3 by 68.0. Track slots are not treated as persistent player identities.

Team control is inferred from tracking evidence over time. Loose balls and uncertain contacts do not automatically start a new possession: the temporal decoder weighs sustained evidence. This is an implemented approximation, not an assertion that every football possession rule is fully observed. Restart tags influence D evidence but are not treated as ground truth. Fouls and offside away from boundaries remain difficult.

We explored geometric baselines, short-gap bridging, control holding, supervised variants, missingness augmentation and decoder costs. Additional experiments included a specialized ball detector, local tiny VLM, local audio transcription and partial tracking-based camera synchronization. None of the detector, VLM, transcript or broadcast results enter this candidate. No hosted model inference over restricted match assets is part of the submitted pipeline. No hand-labelled match intervals or manually observed replay times are used.

## Training and external labels

The supplied checkpoint was trained only on public IDSSE match J03WMX, using live H/A provider labels sampled every fifth frame and team-swap augmentation. Training includes fixed missing-ball masks of 0, 1, 3 and 8 seconds. The estimator has 100 iterations, 15 leaves, learning rate 0.08, L2 regularization 1, minimum leaf size 50, no early stopping, random state 2026. The bundled checkpoint and its hash permit inference reproduction without retraining.

Source: https://huggingface.co/datasets/pysport/idsse-data ; original dataset DOI: https://doi.org/10.6084/m9.figshare.28196177 ; paper DOI: https://doi.org/10.1038/s41597-025-04505-y . The dataset card identifies Sportec Solutions/TRACAB and DFL provenance and CC BY 4.0. Credit the dataset authors (Bassek et al., 2025). We used provider BallPossession (1/2 mapped to H/A) and BallStatus (0 mapped to D; missing status to X). We did not manually annotate these labels. The individual annotators and exact equivalence to the challenge conventions have not been independently verified.

J03WN1/J03WOH (four halves, two matches) were development data. J03WOY was a frozen test of cost16 before D tuning. D150 was chosen from five bounded development alternatives, then tested on new J03WPY. No J03WOY labels were reused for D tuning. Tags are absent and passed as zero in these external comparisons. Synthetic gaps are controlled stress tests, not natural occlusions or independent matches.

## Measured results

On new J03WPY, unweighted two-half mean spellF1 improved from 0.464033 to 0.469894 on complete tracking, 0.468196 to 0.468821 with short gaps, and 0.431180 to 0.433179 with mixed gaps. Five of six half-condition comparisons improved; second-half short-gap spellF1 regressed by 0.001311. On complete tracking, D150 recovered 1,321 additional true D frames and added 158 false D frames. These modest same-provider results are not a demonstrated full-challenge improvement.

After freezing the candidate hash, we performed one official public-five-minute sanity check without choosing among variants or adjusting parameters using that score. Results: spellF1 0.450000, macroF1 0.599063, accuracy 0.733600, 23 predicted H/A spells versus 17 reference spells, 7,500 graded frames and no abstentions within them. D precision was 0.456284 and recall 0.113683. This is a material weakness, not a solved component. The full-match reference remains private. No confidence interval was computed.

## Limitations

External data and challenge tracking differ in sensor quality, missingness, player identity stability and potentially annotation conventions. The public check demonstrates weak D performance on that slice; it does not establish its cause. A stationary-ball shortcut can be provider-specific. D150 also scales signed Smart Tag contributions that are absent from the external tests. Long occlusions beyond the training masks, restart ambiguity, tracking duplicates and stale last-seen positions remain risks. The method can miss interior stoppages and overmerge short team possessions. Improvements on external data do not guarantee a top-20 finish or whole-match accuracy.

## Reproduction

Tested with Python 3.12.8 on Windows and pinned requirements. From the public code folder:

```powershell
python -m pip install -r requirements.txt
python predict.py --data-dir C:/path/to/data --output "C:/path/to/Winning Culers.csv"
```

The data folder must contain tactical/BAR-ATH_tactical_atd.xml, tactical/BAR-ATH_tactical_atd.txt and tactical/BAR-ATH_tactical_smart_tagging.xml. No video or network request is needed for inference once dependencies are installed. Two CPU threads are used. Loading is restricted to the bundled checkpoint hash. The optional baseline-output is the modified geometric result, not the original backup.

The candidate was generated twice with identical bytes and passed the official validator. CSV SHA256: ce7eeb4086b42d58f3c73706b0ebcdaba8ea027849f2bc12fd604089cd5be9f9. Only source, checkpoint, dependencies and documentation belong in the public repository. Restricted tracking, videos and full prediction CSVs remain outside it.
