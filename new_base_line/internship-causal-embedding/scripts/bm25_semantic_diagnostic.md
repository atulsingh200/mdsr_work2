# BM25 hard negatives — semantic similarity vs. the gold positive

Encoder: `sentence-transformers/all-MiniLM-L6-v2` (off-the-shelf, mean pool, L2-normalized).

Cosine similarity = dot product of L2-normalized embeddings.

All values are *averages over the queried anchors*.


## Aggregate semantic similarities

| dataset | N queries | sim(anchor, **gold**) | sim(anchor, BM25 neg) avg | sim(anchor, **hardest** BM25 neg) | gap (gold − hardest neg) | % anchors where a neg outscores gold |
|---|---:|---:|---:|---:|---:|---:|
| `followupqg` | 501 | **0.4529** | 0.1267 | 0.2728 | +0.1801 | 20.2% |

> If `gap (gold − hardest neg)` is **positive**, the gold has a higher mean semantic similarity to the anchor than any BM25 distractor — that's why MiniLM still wins despite BM25 picking lexically similar texts.

> If it's **negative**, BM25 is genuinely semantically hard there — every dataset besides `followupqg` is in this regime.


---

## followupqg — examples

Each row is one randomly chosen anchor sampled near the dataset's 10th, 50th, and 90th percentiles of the per-anchor (gold − hardest neg) gap. Higher gap = easier for the model on that anchor.


**gap = +0.1926**

- **anchor.** ELI5: How does the McGurk Effect happen? The Ben 10 toy that produces that sample (which correctly says “Brainstorm”as that is the name of the toy character) has a fairly low sample rate and a hanging frequency in the background. Br and Gr sound similar enough to be confused if you have one in mind…
- **gold positive**  (sim 0.3996)  ·  (I'm not english, I'm sorry for any mistake I make) So it's basically our brain choosing to interpret a frequency and the words when associated with the words on the screen? But, if I'm not looking at the screen to read or I'm thinking about something unrelated to the audio, I can still only hear G…
- BM25 neg #1 (sim 0.0113)  ·  Thank you for the detailed response. I have heard about a similar manual, systematic approach determining edibility in the past. Is there anything else to test on a molecular basis to determine if it will effect the various cells in the human body?
- BM25 neg #2 (sim 0.1741)  ·  > In those moments while it's in the air, does that 10 pound rock add weight to the box? During a ballistic arc neither the rock or a fly would be weighing the box down (although the mass would be included). The issue there though is that *part* of the box is falling. You can't gauge the force requ…
- BM25 neg #3 (sim 0.2032)  ·  Thanks. I’m listening to the audiobooks and totally missed it.I have watched the movies many times and am just on the second time through the books. Is it spoke of during the battle at Pelennor Fields? there is a lot going on there that is new and I must have glossed over it.
- BM25 neg #4 (sim 0.2070)  ·  Gotcha, that makes a lot of sense! So it's a combination of inefficient sound capturing (would need a VERY sensitive diaphragm) and the general low dBs that could be captured.. Would there be some sort of way to boost the dB frequency just before it reaches the diaphragm yet contain it in a way tha…


**gap = -0.1087**

- **anchor.** ELI5: How can a food high in fat have zero cholesterol? How can I high cholesterol food have low fat? Cholesterol is made in your body to repair damaged cells like blood vessels, so if you eat a lot of sugars, your cholesterol will be released into the blood to repair the damage caused by the sugar…
- **gold positive**  (sim 0.1571)  ·  It looks like you're suggesting a vegetarian keto plan, right?
- BM25 neg #1 (sim 0.2658)  ·  >Because all of the food now is riddled with terrible ingredients we have not evolved to eat naturally I'd love you to tell me which common foods we aren't able to eat naturally? Humans are built to eat almost everything
- BM25 neg #2 (sim -0.0085)  ·  Rescue breaths aren't as important if the heart is stopped, right? Like if the body isn't moving the blood around there's no reason to blow air into their lungs.
- BM25 neg #3 (sim 0.1709)  ·  So the reason fruits taste good is because the tree/plant knows that animals will be more likely to eat it then? Wow that's so cool
- BM25 neg #4 (sim 0.2178)  ·  What if we’re fat and short? I’m just the latter but wouldn’t we be using more resources if we’re fat?


**gap = +0.4594**

- **anchor.** ELI5: Why does in bass in speakers seem to limit at higher volumes? The speaker may have circuitry designed to limit bass at higher volume to prevent damage to itself. Bass frequencies require a lot of speaker movement to create. Small speakers aren't good and producing bass anyway.
- **gold positive**  (sim 0.7503)  ·  Thank you!! Completely different random question. But is there a reason why speakers that hold more bass, are quieter than say a speaker that pushes the mids and highs more?? Sorry for the random question, just really interested in learning in the physics of audio.
- BM25 neg #1 (sim 0.1630)  ·  Thank you. What causes the air in the atmosphere to have higher pressure than the heavy air closer to the surface?
- BM25 neg #2 (sim 0.0622)  ·  Ok, that's a pretty good analogy. So a follow-up of sorts: when someone is badly injured and is to be 'transferred' up the chain to a higher level hospital, is it usually due to experienced staff, equipment or something else?
- BM25 neg #3 (sim 0.2909)  ·  >Your ears pick up more bass from your voice resonating in your head which is why it sounds different and often worse on recordings so is the voice you hear on recording more accurate to what other people hear? for instance people say i sound very much like my dad but my recorded voice sounds nothi…
- BM25 neg #4 (sim -0.0644)  ·  are ya saying the intensity of energy? sunlight having low energy not allowing movement than are feelable/observable, while electricity at the end of a wire is high hence the movement is immense enough to be feelable and observable?
