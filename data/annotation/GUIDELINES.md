# Annotation guidelines: Pidgin finance and crypto sentiment

These rules were written **before** any labelling started, and they are not
changed once labelling begins. If a post does not fit the rules, it gets
excluded rather than labelled by instinct, and the reason is recorded.

This matters for the report: a single annotator labelling to taste is weak
evidence. A single annotator labelling to written rules, with a measured
agreement check, is defensible.

## What is being labelled

The **sentiment the author expresses towards the money, crypto or financial
thing they are talking about.** Not whether the post is good news in general,
and not whether the post is polite.

Three labels, matching AfriSenti so the two test sets are comparable:

| label | meaning |
|---|---|
| `positive` | the author is pleased, approving, optimistic, or recommending |
| `negative` | the author is unhappy, complaining, warning, accusing, or pessimistic |
| `neutral` | factual, informational, or a question with no stance |

## Rules

1. **Label the author's stance, not the event.** "Bitcoin crash again" is
   factual reporting and is `neutral`. "Bitcoin crash again, I done tire for
   this market" is `negative`, because the author adds a stance.

2. **Scam warnings and accusations are `negative`.** "Dis platform na scam, make
   you no put money" is negative. The target is the platform, not the reader.

3. **Price direction alone is not sentiment.** "BTC reach 100k" is `neutral`.
   "BTC reach 100k, we dey eat" is `positive`.

4. **Questions with no stance are `neutral`.** "How I go withdraw from Binance
   P2P?" is neutral even if the author is probably frustrated. Only label
   `negative` if frustration is actually expressed: "Why dis withdrawal dey slow
   like dis na?" is negative.

5. **Advice and recommendations take the sentiment of the advice.** "Use
   stablecoin, e better pass holding naira" is `positive` towards stablecoins.

6. **Mixed posts take the dominant stance.** If a post praises one thing and
   criticises another, label whichever the post spends more of its words on. If
   genuinely balanced, exclude it.

7. **Sarcasm is labelled by intended meaning, not literal words.** "Na very
   good platform, dem swallow my money since January" is `negative`. Flag these
   in the `notes` column, because they are worth pulling out in error analysis.

8. **Ignore how the post treats other users.** An insult aimed at another forum
   member is not financial sentiment. If the post has no financial stance at
   all, exclude it.

## What to exclude

Mark `label` as `exclude` and give a reason in `notes`:

- Not actually about money, crypto, payments or trading.
- Not recognisably Pidgin (pure English, or pure Hausa/Yoruba/Igbo).
- Pure advertising, referral links, or signature spam.
- Too short to carry a stance ("lol", "true").
- Genuinely ambiguous after applying the rules above.

Exclusions get counted and reported. A high exclusion rate is itself a finding
about the quality of forum data.

## Language note

Posts on these forums code-switch constantly. Keep a post if it is
*recognisably Pidgin* — at least some Pidgin grammar or vocabulary (`dey`, `na`,
`wetin`, `abeg`, `don`, `wahala`, serial verbs). Drop posts that are just
standard Nigerian English, since the point of the test set is Pidgin.

## Reliability check

Two things make the labels defensible:

1. **Second annotator on a subset.** Ask one other Pidgin speaker to label the
   same 50 posts, independently, using this document and nothing else. Cohen's
   kappa on that subset gets reported. This is the stronger check and it costs
   the other person about twenty minutes.

2. **Self-agreement.** Re-label 40 posts in a shuffled order without looking at
   the first pass, and report the agreement. Weaker than a second annotator, but
   better than nothing if no one is available.

`src/annotation_agreement.py` computes both.

## Sources and permission

Posts are collected manually from public Nigerian crypto discussion:
Bitcointalk's Nigeria (Naija) local board and Nairaland's cryptocurrency
section. Collection is manual reading and copying, not automated scraping, and
the volume is small. Record the source board and the collection date for each
post. Do not record usernames: the posts are quoted for linguistic analysis and
author identity is not needed.
