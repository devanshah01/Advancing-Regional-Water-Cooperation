**Multi Agent Negotiation**

A RAG grounded, hub and spoke multiagent workflow in which LLM delegates for Afghanistan, Tajikistan, Uzbekistan and Turkmenistan negotiate Amu Darya water allocation through a neutral mediator. It doubles as an experimental harness: persona framing, payoff visibility and document grounding are switchable factors, so every run measures how those design choices change the outcome.
Written for: your teammates, as a plain-text summary to paste into a chat or doc.

**Synthetic persona results (llama3.2, RAG database, 10 personas, one run each)**

**Setup**
We built 10 test personas from the 4 country worksheets. 4 are the worksheets unchanged, and 6 change one thing each: loaded wording, hardline goals, cooperative goals, no sources, planted false figures, and no background at all. Each persona ran as a negotiating agent, gave an opening statement and voted on the Protocol 566 allocation. We then checked every claim it made against the document it cited.

**What went well**
- No invented numbers. All 13 figures the agents stated came from the documents or the negotiation text they were given.
- The false facts did not spread. The agent given 7 planted false figures repeated none of them.
- Our checks caught the bad worksheets. The source check flagged the planted false figures: 72% of that persona's figures appear in its sources, against 100% for every real worksheet. The cooperative persona scored 83% because its new red line had no source.
- The Part C checklist flagged the loaded-wording persona (8 issues) and the no-background persona (4 issues).
- Claims were on topic. Each claim was clearly closer to the passage it cited than to random passages (similarity 0.75 against 0.61).
- Every model reply came back in the right format.

**What went wrong**
- Citations were badly formatted. As generated, only 24% of claims cited a passage id correctly; the model added brackets or extra characters to real ids. After we fixed the matching, 83% were valid, and 17% cited passages that do not exist.
- Most claims were not backed by their source. Only 34% of claims were judged supported or partly supported by the passage they cited.
- The verifier is unreliable. It judged the same 7 claims twice and agreed with itself only 3 times.
- Agents ignored their own payoffs. Afghanistan accepted a deal worth 367 to it when walking away was worth 805, in both of its personas.
- Afghanistan has the weakest sources. Only 38% of the passages it was given were about Afghanistan, against 50 to 75% for the other countries.
- The sample is small: one run per persona and 29 claims in total, so differences between personas are not reliable yet.

**Next steps**
1. Use a bigger model (at least 14B, for example qwen2.5:14b) for the agents, and run the verifier at temperature 0 or on a stronger model.
2. Run each persona 5 or more times, so we can compare personas properly.
3. Rebuild the database index to include the new documents Devan added, and add more sources and primary treaty texts.
4. Review the 4 country worksheets, and find sources for the lines marked NO SOURCE YET.
5. Replace the placeholder numbers in basin.yaml marked ILLUSTRATIVE with sourced values.
6. Look into why agents accept deals worth less than walking away, starting with Afghanistan.
