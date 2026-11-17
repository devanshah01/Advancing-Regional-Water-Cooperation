# Run these on your Mac (VS Code terminal) -- AFTER Devan uploads the full Kazakhstan.txt and Vivi has checked the 20 new questions

## 0. Put the update in the repo (new branch; keeps your uncommitted results_15docs / results_39docs work)
```bash
cd ~/Downloads/Advancing-Regional-Water-Cooperation
git checkout main && git pull        # if git complains about local changes, tell me what it says
git checkout -b h1-h3-experiments
unzip -o ~/Downloads/rag_update_v2.zip -d rag-evaluation
cd rag-evaluation
source ~/Downloads/rag_project/.venv/bin/activate
mkdir -p results_42
```
If Vivi corrected any of the new questions, replace `test_queries_v2.csv` with her corrected version first.

## 1. H1: QA on the 42 questions (temperature 0). Run one at a time; paste me each SUMMARY when it ends.
```bash
caffeinate -i python3 evaluate.py --docs ../data --queries test_queries_v2.csv --db ./db_fixed --ks 3,4 --styles basic --out results_42/qa_fixed
caffeinate -i python3 evaluate.py --docs ../data --queries test_queries_v2.csv --db ./db_table --chunking table --ks 3,4 --styles basic --no-baseline --out results_42/qa_table
```
## 2. Consistency check: each query 3 times per condition at temperature 0.7
```bash
caffeinate -i python3 evaluate.py --docs ../data --queries test_queries_v2.csv --db ./db_fixed --ks 3 --styles basic --no-baseline --repeats 3 --temperature 0.7 --out results_42/consistency_fixed
caffeinate -i python3 evaluate.py --docs ../data --queries test_queries_v2.csv --db ./db_table --chunking table --ks 4 --styles basic --no-baseline --repeats 3 --temperature 0.7 --out results_42/consistency_table
```
## 3. H2 + H3: 21 paired policy briefs per condition (7 targets x 3 repeats), conditions pilot_full / verbatim / constrained
```bash
caffeinate -i python3 evaluate_policy.py --docs ../data --db ./db_policy --out results_42/policy
```
## 4. Adversarial upstream prompts (5)
```bash
caffeinate -i python3 evaluate_adversarial.py --docs ../data --db ./db_policy --out results_42/adversarial
```
## 5. Save to GitHub (the CSVs are what I analyse)
```bash
cd ..
printf "\ndb_*/\n" >> .gitignore      # only if not already there
git add rag-evaluation .gitignore
git status      # should show rag-evaluation/ files and NO db_ folders
git commit -m "H1-H3 experiments: 42-query QA, 21 paired briefs, adversarial prompts"
git push -u origin h1-h3-experiments
```
Then tell me the branch name (h1-h3-experiments). I run the statistics, tables and figures from those CSVs.

Notes: `*_DRYRUN*` files are fake -- never push or use them. The analysis/ folder runs on my side; you do not need to run it.
