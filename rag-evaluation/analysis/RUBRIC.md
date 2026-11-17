# Rating rubric (all four raters; rate independently AFTER the 5-item calibration round)

You see, for each item: the question, whether the documents can answer it, a reference answer, the passages the system
retrieved, and the model's answer. You do NOT see which system produced it. Rate two things, each 1 (yes) or 0 (no).

**correct**
- Answerable question: 1 if the answer states every key fact in the reference answer (numbers may be rounded or written
  differently, e.g. "1.14 million" for 1,142,857) and contradicts none of them. 0 if a key fact is missing or wrong,
  or if the system says it cannot find the answer.
- Unanswerable question: 1 if the system declines or says the documents do not contain the answer. 0 if it gives an answer.

**grounded**
- 1 if every factual claim in the answer is supported by the retrieved passages shown (or the answer is a refusal).
- 0 if the answer includes any claim, number or name that is not in the passages (even if it happens to be true).

Rate each item on its own. Do not discuss until everyone has finished. Save your sheet as `rater_<yourname>.csv`
with the item_id, correct and grounded columns filled in.
