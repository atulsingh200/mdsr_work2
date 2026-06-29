# Hard-Negative Mining Results

This folder contains training data, checkpoints, and evaluation metrics for
the fine-tuned bi-encoder, one sub-folder per dataset.

---

## What are hard negatives?

For each training pair `(anchor, true_positive)` we mine **4 hard negatives**
using `sentence-transformers/all-MiniLM-L6-v2` (frozen).

```
For every anchor x with true positive y:
    Encode y and every OTHER y' in the training set with MiniLM.
    Take the top-4 nearest y' by cosine similarity.
    These become the hard negatives y*_1 … y*_4.
```

The hard negatives are semantically close to the true positive (same
vocabulary, same topic) but are **not** the actual causal/conversational
continuation. The model must learn a finer distinction than topic overlap —
exactly the signal needed for follow-up retrieval.

Hard negative indices are stored as `hard_negatives.npy` / `test_hard_negatives.npy` (shape `[N, 4]`).
Each row contains 4 integer indices into the corresponding `train_pairs.jsonl` / `test_pairs.jsonl`.

---

## Dataset inventory

| Dataset        | Train pairs | k hard negs | Mining model             |
|----------------|-------------|-------------|--------------------------|
| aep_causal     | 5,487       | 4           | all-MiniLM-L6-v2         |
| followupqg     | 2,790       | 4           | all-MiniLM-L6-v2         |
| multiwoz_v24   | 15,000      | 4           | all-MiniLM-L6-v2         |
| qrecc          | 15,000      | 4           | all-MiniLM-L6-v2         |
| workflow       | 20,000      | 4           | all-MiniLM-L6-v2         |

---

## Annotated samples per dataset (test split)

All samples below are from `test_pairs.jsonl` + `test_hard_negatives.npy`.
Each block shows the anchor, the single true positive, and the 4 hard
negatives mined by MiniLM cosine-KNN — these are the negatives the model is
evaluated against in `eval_hardneg.json`.

---

### aep\_causal

Anchor is an AEP documentation excerpt; the true positive is the specific
doc section that causally follows it. Hard negatives are other doc chunks
that are topically close but wrong.

---

**Sample 1** *(test idx 0)*

> **Anchor**
> Audience and Profile Activation to Enterprise Destinations — Share profile
> and audience changes and events in streaming or batch from Real-time
> Customer Data Platform to enterprise data stores and applications. These
> profile and audience events can be used to initiate a sales or support
> action to the customer…

> **True Positive**
> Guardrails reflect system constraints, expected latencies, and performance
> expectations to optimize customer architecture and use case performance and
> help to ensure stability, avoid errors or unexpected results. Types of
> Guardrails — Performance guardrail (Soft limit)…

| # | Hard Negative |
|---|---------------|
| 1 | Guardrails are thresholds that guide data and system usage, performance optimization, and avoidance of errors or unexpected results in Adobe Experience Platform. This document provides default usage limits for Query Service data… |
| 2 | Default guardrails for Real-Time Customer Profile data and segmentation — Adobe Experience Platform enables you to deliver personalized cross-channel experiences based on behavioral insights and customer attributes… |
| 3 | Decisioning guardrails & limitations — Applies to: Journey Orchestration. To ensure optimal use of Decisioning, keep the following guardrails and limitations in mind. The complete list of Journey Optimizer guardrails & limitations… |
| 4 | Guardrails and limitations — Applies to: Campaign Orchestration. You will find below guardrails and limitations when using Orchestrated campaigns. Dataflow limitations / Data Design & Storage… |

*Why hard:* All four are "guardrails" doc pages from different AEP products
(Query Service, RTCDP, Decisioning, Campaign). They share the exact keyword
"guardrails" with the true positive but address different subsystems.

---

**Sample 2** *(test idx 50)*

> **Anchor**
> Resources exist in different states and they depend on one another. Before
> you delete a resource, you must make sure it is in a state where it can be
> deleted. Preparing a resource for deletion consists of two basic steps:
> Resolve dependencies. Remove from libraries…

> **True Positive**
> When you no longer want a resource to have an effect inside a build, you
> must remove it from the library that contains that resource and create a
> new build. IMPORTANT — Resources in libraries are interdependent. Removing
> a resource from a build may change the behavior of other resources…

| # | Hard Negative |
|---|---------------|
| 1 | Libraries — A library is a set of instructions for how extensions, data elements, and rules interact with one another after they are deployed. When creating a library, you specify the changes you want to make to your library… |
| 2 | When a library is compiled into a build in the Reactor API, the exact contents of the build depend upon the environment settings and the resources included in the library. Specifically, the environment determines the Destination… |
| 3 | Resources in the Reactor API are often related to each other. This document provides an overview of how resource relationships are established in the API, and the relationship requirements of each resource type… |
| 4 | Builds — A build is the set of files containing all the code that runs on the client device. It is a composite of the changes you specified within your library, as well as everything that has been submitted, approved, or published… |

*Why hard:* All four are about the Reactor API library/build lifecycle —
same product area, same vocabulary (library, build, resources) — but each
covers a different phase of the workflow (creating, compiling, relationships,
deploying) rather than the deletion/removal step the anchor is about.

---

### followupqg

Anchor is an ELI5 Reddit explanation; true positive is a templated follow-up
question from that exact thread. Hard negatives are follow-up questions from
other threads that share surface vocabulary.

---

**Sample 1** *(test idx 0)*

> **Anchor**
> ELI5 Do animals tan? Animals can get sunburned like we do — pigs for
> example root in mud to cool off and because mud is a form of sun
> protection. Animals with fur cannot get sunburned where and when the fur
> covers them but can get burned on their lips and eyelids.

> **True Positive**
> But can they tan? Does their body create color to protect them?

| # | Hard Negative |
|---|---------------|
| 1 | Thank you! So if I'm understanding this correctly — UV-A and UV-B are what tans skin, enables plants to conduct photosynthesis, etc, and they're just wavelengths of light? I honestly thought they were a light-transmitted chemical or something. |
| 2 | But the only pigment in our eye is melanin which is brown right? |
| 3 | Does the same thing happen in babies when fathers do skin to skin contact? |
| 4 | Why does electricity penetrate skin but not light, they are both energy right? |

*Why hard:* HN 1 and 2 mention UV/melanin/pigment — same biological skin
topic — but come from unrelated ELI5 threads. HN 3 and 4 share "skin"
surface vocabulary but are completely different questions.

---

**Sample 2** *(test idx 50)*

> **Anchor**
> ELI5: How can choosing pictures of certain objects (stop signs, crosswalks,
> cars, etc.) prove that you are not a robot on websites? Because how bots
> select images on those questions can be detected as automated by the
> website. People tend to wander with their mouse, and can pretty easily tell
> the difference between a stop sign and a house…

> **True Positive**
> But why haven't the bot makers made their software behave more like a human?

| # | Hard Negative |
|---|---------------|
| 1 | So we didn't get better at programming OSs, we just have more computer power to spare now? They would've done it if they could, couldn't they? |
| 2 | Why isn't this automated? |
| 3 | But how do they make sure every single person does every task? Especially when they give more than three and even 100s or more folks participate, it seems like a lot of tedious work. Unless there is some sort of bot that does it for you? |
| 4 | But why don't they do any redenomination to make things easier? |

*Why hard:* HN 2 ("Why isn't this automated?") and HN 3 (bots doing tasks)
share the automation framing and bot vocabulary with the true follow-up, but
are from threads about crowdsourcing and financial systems respectively.

---

### multiwoz\_v24

Anchor is a multi-turn task-oriented dialogue; true positive is the agent's
immediate next utterance. Hard negatives are paraphrases of the same
dialogue act from different conversations.

---

**Sample 1** *(test idx 0)*

> **Anchor**
> I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.

> **True Positive**
> What time do you want to leave and what time do you want to arrive by?

| # | Hard Negative |
|---|---------------|
| 1 | What time do you want to leave or arrive by? |
| 2 | Can you tell me what time you would like to leave or arrive by? |
| 3 | What time would you like to leave or arrive by? |
| 4 | What time would you like to leave or arrive by? |

*Why hard:* All four are paraphrases of the same taxi-booking slot-filling
question. The distinction is purely stylistic — the model must learn to
match the exact conversational context, not just the dialogue act.

---

**Sample 2** *(test idx 50)*

> **Anchor**
> *(hotel booking dialogue)* Yes please, I would like to book it for 7
> people for 5 nights on Saturday, and I need a reference number. I'm very
> sorry I was unable to book at acorn guest house for 5 nights, would you
> like to try for a shorter stay? Yes, please try to book it for four nights
> instead.

> **True Positive**
> Your booking was successful, the reference number is WAEYAQ0M. May I
> assist you with anything else today?

| # | Hard Negative |
|---|---------------|
| 1 | Your booking was successful. Reference number is: 6I8873PA. Is there anything else I can assist you with today? |
| 2 | OK, the booking was successful. Your reference number is 5CB11JQ8. Is there anything else I can help you with today? |
| 3 | Booking was successful. Reference number is: 8OE7IXND. Is there anything else I can help you with today? |
| 4 | Booking was successful! Your reference number is LNDHTG39. Is there anything else I can help you with today? |

*Why hard:* Same confirmation template, different reference numbers — the
model cannot distinguish by intent or wording alone. The correct positive is
identified only by the full conversation context.

---

### qrecc

Anchor is a conversational context (prior question + retrieved passage); true
positive is the user's next question in that specific thread. Hard negatives
are next-questions from different threads on similar topics.

---

**Sample 1** *(test idx 0)*

> **Anchor**
> What is a physician's assistant? Physician assistants are medical providers
> who are licensed to diagnose and treat illness and disease and to prescribe
> medication for patients.

> **True Positive**
> What are the educational requirements required to become one?

| # | Hard Negative |
|---|---------------|
| 1 | What requirements were made for a woman to attend school? |
| 2 | What school subjects are needed to become a registered nurse? |
| 3 | Did he receive any education? |
| 4 | Did he receive formal education? |

*Why hard:* HN 1 and 2 are about educational requirements for other medical/
professional roles — same domain, plausible follow-up structure — but wrong
entity. HN 3 and 4 are about a historical person's education, same "education"
keyword but entirely different context.

---

**Sample 2** *(test idx 100)*

> **Anchor**
> Describe experiences of some people who have done LASIK. Many developed
> symptoms within six months, such as seeing starbursts (30.3%), halos
> (26.2%), and double images (5.2%). Are there any good alternatives to
> LASIK? LASEK (laser-assisted subepithelial keratomileusis): While LAS…

> **True Positive**
> Once it's done, what kind of precautions do I need to take?

| # | Hard Negative |
|---|---------------|
| 1 | How do I prepare for it? |
| 2 | What are the risks? |
| 3 | How is it treated? |
| 4 | Describe experiences of some people who had it done. |

*Why hard:* All four are natural next-questions for a medical procedure thread
— preparation, risks, treatment, experiences — but belong to different
conversations (different procedures). HN 4 is especially tricky: it mirrors
the anchor's own structure ("describe experiences") for a different procedure.

---

### workflow

Anchor is a natural-language description of a code step; true positive is
the description of the immediately next step in the same workflow. Hard
negatives are descriptions of similar operations from other workflows.

---

**Sample 1** *(test idx 0)*

> **Anchor**
> Retrieves the current content from the clipboard and assigns it to the
> variable `clipboard_content`.

> **True Positive**
> Uses a regular expression pattern to search for an arXiv ID in the
> clipboard content; result is stored in `arxiv_id_match_found`.

| # | Hard Negative |
|---|---------------|
| 1 | Another retrieval of clipboard content, this time naming the variable `Enter_one_or_more_tracking_ids` for clarity. |
| 2 | If 'What's in My Clipboard' is mentioned, it retrieves the clipboard content. |
| 3 | This line searches the web for content matching the data stored in `clipboard_data`. |
| 4 | Matches a text pattern in the clipboard content using a regex pattern and stores the result in `matched_text_pattern`. |

*Why hard:* HN 4 is nearly identical in structure (regex match on clipboard →
store result) but belongs to a different workflow. HN 1 and 2 also read from
clipboard. The model must understand that the true next step is specifically
arXiv ID extraction, not just any clipboard operation.

---

**Sample 2** *(test idx 1)*

> **Anchor**
> Uses a regular expression pattern to search for an arXiv ID in the
> clipboard content; result is stored in `arxiv_id_match_found`.

> **True Positive**
> Extracts the matched arXiv ID from the previous search results, naming it
> `arxiv_id`.

| # | Hard Negative |
|---|---------------|
| 1 | Uses a regular expression pattern to search for an arXiv ID in the clipboard content; result is stored in `arxiv_id_match_found`. (exact anchor text — mined as its own hard negative) |
| 2 | Assigns the matched search results to `search_results`. |
| 3 | Retrieves the single search result from the previous query. |
| 4 | Retrieves the matched bibcode groups from the previous search results. |

*Why hard:* HN 1 is the anchor itself — MiniLM correctly identifies it as
maximally similar to itself, making it the hardest possible negative. HN 2–4
all describe extracting a match from search results, same operation type but
different variable names and workflows.

---

## Folder structure

```
results/
├── README.md                    ← this file
├── mining_summary.json          ← per-dataset mining stats
├── eval_summary.json            ← test-set metrics (standard eval)
├── eval_hardneg_summary.json    ← test-set metrics (hard-neg eval)
├── val_mining_summary.json      ← val split mining stats
├── test_mining_summary.json     ← test split mining stats
└── <dataset>/
    ├── hard_negatives.npy       (N, 4) int indices into train_pairs.jsonl
    ├── train_pairs.jsonl        anchor + positive pairs (train split)
    ├── val_pairs.jsonl          anchor + positive pairs (val split)
    ├── test_pairs.jsonl         anchor + positive pairs (test split)
    ├── val_hard_negatives.npy   (N_val, 4)
    ├── test_hard_negatives.npy  (N_test, 4)
    ├── mining_info.json         mining metadata (miner, timing, paths)
    ├── val_mining_info.json
    ├── test_mining_info.json
    ├── checkpoint_best.pt       best epoch by val MRR
    ├── checkpoint_latest.pt
    ├── config.json              training run config
    ├── training_history.json    per-epoch train/val metrics
    ├── eval.json                test-set metrics
    └── eval_hardneg.json        test-set metrics on hard-neg eval set
```

## Test-set results

Candidate pool of 1000 (gold + 999 random distractors) per anchor.

| Dataset      |     N |  AUC   |  P@1   |  MRR   |  R@1   |  R@5   |  R@10  |
|--------------|------:|-------:|-------:|-------:|-------:|-------:|-------:|
| aep_causal   |   562 | 0.9712 | 0.9128 | 0.3525 | 0.1637 | 0.5854 | 0.7153 |
| followupqg   |   501 | 0.9827 | 0.9541 | 0.5844 | 0.4032 | 0.8064 | 0.8543 |
| multiwoz_v24 | 7,368 | 0.9526 | 0.8694 | 0.2493 | 0.1606 | 0.3275 | 0.4256 |
| qrecc        | 5,204 | 0.9227 | 0.8113 | 0.2758 | 0.1822 | 0.3653 | 0.4570 |
| workflow     |  260k | 0.9662 | 0.9104 | 0.5500 | 0.4656 | 0.6428 | 0.7055 |
