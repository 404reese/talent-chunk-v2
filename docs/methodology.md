# Methodology

The ranker is intentionally transparent. The competition rewards matching the JD's meaning, not only matching terms in the skills list.

## Candidate fit

The scoring function prioritizes:

- Direct production evidence of search, retrieval, ranking, recommendation, matching, vector search, and embedding systems.
- Evaluation maturity, including NDCG, MRR, MAP, A/B testing, ranking evaluation, and feedback loops.
- Strong Python and ML engineering implementation signals.
- Seniority close to the requested 5 to 9 year range.
- Product-company context over service-only consulting histories.
- Location and logistics fit for Pune, Noida, Delhi NCR, Hyderabad, Mumbai, and Bangalore.

## Behavioral signals

Redrob signals are used as a capped modifier. Recent activity, open-to-work status, high recruiter response, short notice period, recruiter saves, verification, and strong GitHub activity improve the score. Stale activity, low response, long notice, and weak contactability reduce it.

## Trap handling

The model penalizes:

- AI keyword-heavy skills without matching career history.
- Disfavored current titles without retrieval or production evidence.
- Service-company-only career histories.
- Primary AI depth in computer vision, speech, robotics, or GANs when NLP/search evidence is weak.
- Skill-duration inconsistencies, especially expert skills with zero duration.
- Stale and low-response candidates.

## Reasoning

Reasoning strings are assembled from facts extracted from the candidate record and score components. This keeps the reasoning specific and reduces hallucination risk during manual review.
