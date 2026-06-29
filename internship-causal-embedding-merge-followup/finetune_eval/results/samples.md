# Hard Negative Samples — Anchor-Penalized Mining (Test Split)

## Strategy

Standard kNN mining picks the negatives whose positive `y*` is most similar
to the true positive `y`. The problem: it often selects negatives whose own
anchor `x*` is also very similar to `x`, meaning the pair `(x*, y*)` is
nearly the same context as `(x, y)` — making the negative accidentally easy
to reject on anchor similarity alone.

**This file uses a stricter criterion:**

```
score(i, j) = sim(y_i, y_j)  −  α × sim(x_i, x_j)

     ↑ maximise: y* close to y        ↑ penalise: x* similar to x
```

With `α = 0.5`. This selects negatives where:
- the positive `y*` is **semantically close** to the true positive `y` (genuinely hard)
- but the candidate's own anchor `x*` is **from a different context** than `x`

The result: the model cannot rely on anchor-side cues to reject the negative —
it must understand the positive content itself.

Source: `test_pairs.jsonl`, scores computed on-the-fly with
`sentence-transformers/all-MiniLM-L6-v2`.

Each sample shows `y_sim` (positive similarity) and `x_sim` (anchor
similarity) alongside the hard negative text and its source anchor.

---

## aep\_causal

---

### Sample 1 — idx 0

**Anchor**
> Audience and Profile Activation to Enterprise Destinations — Share profile
> and audience changes and events in streaming or batch from Real-time
> Customer Data Platform to enterprise data stores and applications. These
> profile and audience events can be used to initiate a sales or support
> action to the customer…

**True Positive**
> Guardrails reflect system constraints, expected latencies, and performance
> expectations to optimize customer architecture and use case performance and
> help to ensure stability, avoid errors or unexpected results. Types of
> Guardrails — Performance guardrail (Soft limit)…

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.818 | 0.293 | Guardrails are thresholds that guide data and system usage, performance optimization, and avoidance of errors or unexpected results in Adobe Experience Platform. This document provides default usage limits for Query Service data… | The Data Distiller SKU is required to use the query accelerated store. Please see the and, and documentation that relates to the Data Distiller SKU. |
| 2 | 0.624 | 0.272 | Decisioning guardrails & limitations — Applies to: Journey Orchestration. To ensure optimal use of Decisioning, keep the following guardrails and limitations in mind… | Decisioning & Decision management guardrails — Guardrails and limitations to keep in mind when working with Decisioning or Decision management… |
| 3 | 0.670 | 0.395 | Default guardrails for Real-Time Customer Profile data and segmentation — Adobe Experience Platform enables you to deliver personalized cross-channel experiences based on behavioral insights and customer attributes… | Phase 3: Decisioning setup — This phase configures the complete decisioning framework including placements, eligibility rules… |
| 4 | 0.514 | 0.408 | High Throughput mode is designed for organizations that need up to 5000 transactions per second. Unlike regular API triggered campaigns, High throughput campaigns operate independently of Adobe Profile… | To create a new API triggered campaign, follow these steps: Browse to the menu and select the API triggered tab. Click the button… |

> **What changed vs. plain kNN:** HN 1 has the highest y_sim (0.818) and is
> still picked — but the source anchor is about Data Distiller SKU, not about
> activation to enterprise destinations. HN 4 (High Throughput / API
> campaigns) has y_sim 0.514 but x_sim only 0.408 — its anchor is a
> completely different AEP workflow. The model gets negatives that are
> topically similar in the positive space but come from unrelated doc areas.

---

### Sample 2 — idx 1

**Anchor**
> With Talon.One, you can easily create, manage, and optimize personalized
> marketing campaigns tailored to your customers. Use this powerful platform
> to run discounts, distribute coupons, launch referral programs, set up
> loyalty programs, and offer gamified incentives…

**True Positive**
> Read this guide to learn how to connect and stream your data from
> Talon.One to Adobe Experience Platform using the sources workspace in the
> UI. Getting started — This tutorial requires a working understanding of
> the following components of Experience Platform…

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.597 | 0.059 | Adobe Experience Platform allows you to easily import data into Experience Platform as batch files. Examples of data to be ingested may include profile data from a flat file in a CRM system… | Specific field values do not seem to update properly after running through an Update Profile node in a journey. Commonly in such issues, profile field… |
| 2 | 0.713 | 0.305 | Create a schema using the Schema Editor — The Adobe Experience Platform user interface allows you to create and manage Experience Data Model (XDM) schemas in an interactive visual canvas called the Schema Editor… | Create XDM Schema — Log in to Adobe Experience Platform / Data Management → Schemas → Create schema → Create an XDM event based schema called Financial A… |
| 3 | 0.679 | 0.240 | Getting started with the Schema Registry API — The Schema Registry API allows you to create and manage various Experience Data Model (XDM) resources. This document provides an introduction to the core concepts… | Behaviors endpoint — In Experience Data Model (XDM), behaviors define the nature of data that a schema describes. Each XDM class must reference a specific… |
| 4 | 0.591 | 0.116 | The command is the primary way to send data to Adobe. Its response object is the primary way to retrieve personalized content, identities, and audience destinations… | Render DOM action propositions automatically — Use this pattern when your personalization response includes proposition items with the schema… |

> **What changed:** HN 2 (y_sim=0.713) is a high-similarity match — both are
> "how to connect a data source to AEP via the UI" docs — but its source
> anchor is about creating XDM schemas, not about Talon.One campaigns. The
> model must reject it based on positive content alone, not anchor context.

---

## followupqg

---

### Sample 1 — idx 0

**Anchor**
> ELI5 Do animals tan? Animals can get sunburned like we do — pigs for
> example root in mud to cool off and because mud is a form of sun
> protection. Animals with fur cannot get sunburned where and when the fur
> covers them but can get burned on their lips and eyelids.

**True Positive**
> But can they tan? Does their body create color to protect them?

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.447 | 0.238 | But the only pigment in our eye is melanin which is brown right? | ELI5: Eye colors warmth? Warmth is about color tone. Warm tones are yellows, reds, oranges. When you have solid brown and add some yellow, you get a… |
| 2 | 0.275 | 0.035 | Was there some evolutionary disadvantage to keeping them anyway? | ELI5: Why do we prune most of what we learn from our memory? Doesn't the brain have practically infinite storage space? Because you're not using them. |
| 3 | 0.243 | −0.023 | But why they dont do any redomination to make things easier? | ELI5: Why do some countries' currencies go into millions for such small amounts of value? For example, in Thailand, 1 USD = 14,000 Indonesian rupiah. Why? |
| 4 | 0.329 | 0.150 | Why does electricity penetrate skin but not light, they are both energy right? | ELI5: Why do our muscles shake when electricity touches us but not when light touches us? |

> **What changed:** HN 2 (x_sim=0.035) comes from a memory/pruning thread —
> completely unrelated anchor. HN 3 has x_sim=−0.023 (anti-correlated anchor,
> a currency ELI5) but the follow-up phrasing is superficially similar. In
> plain kNN these would be outcompeted by UV/tanning threads with high x_sim.
> Here the penalty pushes them in because their anchors are distant.

---

### Sample 2 — idx 50

**Anchor**
> ELI5: How can choosing pictures of certain objects (stop signs, crosswalks,
> cars, etc.) prove that you are not a robot on websites? Because how bots
> select images on those questions can be detected as automated by the
> website. People tend to wander with their mouse…

**True Positive**
> But why haven't the bot makers made their software behave more like a human?

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.447 | 0.033 | So we didn't get better at programming OSs, we just have more computer power to spare now? They would've done it if they could, couldn't they? | ELI5: How come when an app doesn't respond in Windows 10, only that app freezes but in earlier Windows the whole PC hangs? We got better at… |
| 2 | 0.446 | 0.163 | Why isn't this automated? | ELI5: Why are bank transactions closed during weekends? Isn't online banking just computers? Because banks are required to balance their books… |
| 3 | 0.297 | −0.004 | But why they dont do any redomination to make things easier? | ELI5: Why do some countries' currencies go into millions for small amounts? For example, 1 USD = 14,000 Indonesian rupiah. Why? |
| 4 | 0.391 | 0.268 | But how do they make sure every single person does every task? Especially when they give 100s of tasks, it seems like a lot of tedious work. Unless there is some sort of bot that does it for you? | ELI5: In social media campaigns where you have to like a post and tag friends, how do they sort through all the data? |

> **What changed:** HN 1 (x_sim=0.033) is from an OS/app-freezing thread —
> its anchor has almost no overlap with the CAPTCHA anchor. HN 3 has negative
> x_sim (−0.004), from a currency ELI5. Both are retained because their
> follow-up phrasing matches the "why didn't they just…" pattern of the true
> positive.

---

## multiwoz\_v24

> **Problem with plain kNN here:** MultiWOZ has many near-duplicate agent
> utterances — the same slot-filling question or booking confirmation
> template appears hundreds of times with tiny wording differences. Plain
> kNN just finds those paraphrases (x_sim also high because they come from
> similar dialogue contexts), making negatives trivially rejectable by
> anchor similarity. The samples below use `α = 1.5` (stronger anchor
> penalty) to force negatives whose source anchor is a **different dialogue
> domain entirely** (restaurant, train, attraction) while keeping
> the positive text semantically close.

---

### Sample 1 — idx 0

**Anchor**
> I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.

**True Positive**
> What time do you want to leave and what time do you want to arrive by?

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor (different domain) |
|---|------:|------:|-------------------------------|----------------------------------|
| 1 | 0.850 | 0.141 | What time would you like to leave? | I am looking forward to trying local restaurants but would like help finding a place to go in town. I would like it to be in the south and a swimmingpool… *(restaurant search)* |
| 2 | 0.838 | 0.204 | You have one leaving at 21:09, what time do you want to arrive by? | Hello! I am looking for train schedules from Leicester. Can you help me with that? Sure! There are several trains leaving from Leicester. What destination?… *(train booking)* |
| 3 | 0.844 | 0.235 | To narrow down the search, do you know what time you want to leave or what time you want to arrive by? | No, just the entrance fee please. I think I am more interested in Kings Hedges. I don't have the entrance fee listed here. Do you have any contact info?… *(attraction search)* |
| 4 | 0.762 | 0.193 | What time would you like to travel? | No, I'm coming from Ely. *(train booking — single utterance anchor)* |

> **Why these are proper hard negatives:** The positive texts are all
> slot-filling questions about departure/arrival time (y_sim 0.76–0.85) —
> genuinely semantically close to the true positive. But the source anchors
> are from restaurant-finding, train-booking, and attraction dialogues —
> **not a taxi booking**. A model that memorises "taxi anchor → time question"
> gets no help here; it must score the positive content itself.

---

### Sample 2 — idx 50

**Anchor**
> *(hotel booking)* Yes please, I would like to book it for 7 people for 5
> nights on Saturday, and I need a reference number. I'm very sorry I was
> unable to book at acorn guest house for 5 nights, would you like to try
> for a shorter stay? Yes, please try to book it for four nights instead.

**True Positive**
> Your booking was successful, the reference number is WAEYAQ0M. May I
> assist you with anything else today?

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor (different context) |
|---|------:|------:|-------------------------------|-----------------------------------|
| 1 | 0.844 | 0.065 | The booking was successful. Your reference number is RWRKP9L6 | Thank you that will work fine for me and my husband *(bare acknowledgement — no booking context)* |
| 2 | 0.899 | 0.183 | Booking was successful. Reference number is: 5INDNYF8. Is there anything else I can help you with today? | There will be 4 people. *(single slot-fill utterance, no hotel context)* |
| 3 | 0.873 | 0.226 | your booking was successful, your reference number is 4O3358Y2. is ther anythin else i may help you with? | Let's try Indian food, but it needs to be near the centre of town. There are no Belgian restaurants in the center… *(restaurant search)* |
| 4 | 0.885 | 0.266 | Booking was successful. Your reference number is USCL95YH. Is there anything else I can help you with? | Starting on Wednesday please. *(date slot-fill, no booking resolution context)* |

> **Why these are proper hard negatives:** All four confirmations have very
> high y_sim (0.84–0.90) — the wording is nearly identical to the true
> positive. But source anchors are a bare acknowledgement, a headcount
> utterance, a restaurant search, and a date slot — **none** are hotel
> booking resolutions. The model must learn to match the full conversational
> trajectory, not just the surface form of the confirmation text.
>
> Note: the reference-number template is unavoidable in MultiWOZ — the same
> agent response structure repeats across all booking domains. This is a
> known dataset property, not a mining artefact.

---

## qrecc

---

### Sample 1 — idx 0

**Anchor**
> What is a physician's assistant? Physician assistants are medical providers
> who are licensed to diagnose and treat illness and disease and to prescribe
> medication for patients.

**True Positive**
> What are the educational requirements required to become one?

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.671 | 0.039 | What requirements were made for a woman to attend school? | What other offices did Tarja Halonen hold besides President? Tarja Halonen was Foreign Minister of Finland before becoming president. Before women's suffrage… |
| 2 | 0.442 | −0.137 | Did he receive formal education? | When was Rudolf Steiner born? |
| 3 | 0.456 | −0.085 | Did he receive any education? | When was Bruce Hornsby born? Bruce Hornsby was born on November 23, 1954. |
| 4 | 0.414 | −0.086 | What did he get a degree in? | Are there any interesting aspects about the Amitabh Bachchan article? |

> **What changed:** HN 2, 3, and 4 all have negative x_sim (anti-correlated
> anchors) — they come from threads about Rudolf Steiner, Bruce Hornsby, and
> Amitabh Bachchan respectively. Their anchors share nothing with a medical
> professional context. Yet their follow-up questions ("Did he receive formal
> education?", "What did he get a degree in?") are close in meaning to "What
> are the educational requirements?" — the model must detect the mismatch in
> the positive space, not the anchor space.

---

### Sample 2 — idx 100

**Anchor**
> Describe experiences of some people who have done LASIK. Many developed
> symptoms within six months, such as seeing starbursts (30.3%), halos
> (26.2%), and double images (5.2%). Are there any good alternatives to
> LASIK? LASEK (laser-assisted subepithelial keratomileusis)…

**True Positive**
> Once it's done, what kind of precautions do I need to take?

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.590 | −0.013 | How do I prepare for it? | Tell me about the International Linguistics Olympiad. The IOL is one of 12 International Science Olympiads for secondary school students… |
| 2 | 0.466 | −0.068 | How is it treated? | What are the symptoms of anemia? Easy fatigue and loss of energy, unusually rapid heart beat… |
| 3 | 0.432 | −0.050 | What happens if it goes untreated? | How does Lyme Disease make you feel? Joint pain and stiffness, often intermittent, are early Lyme symptoms… |
| 4 | *(from plain kNN)* | — | Describe experiences of some people who had it done. | *(same thread, different procedure — displaced by anchor penalty)* |

> **What changed:** All three top HNs have negative x_sim — their anchors are
> about linguistics olympiads, anemia, and Lyme disease respectively. They
> share zero context with the LASIK anchor. But the follow-up questions
> ("How do I prepare?", "How is it treated?", "What happens if untreated?")
> are close in structure and meaning to "what precautions do I need to take?"
> — making them semantically hard while being contextually distant.

---

## workflow

---

### Sample 1 — idx 0

**Anchor**
> Retrieves the current content from the clipboard and assigns it to the
> variable `clipboard_content`.

**True Positive**
> Uses a regular expression pattern to search for an arXiv ID in the
> clipboard content; result is stored in `arxiv_id_match_found`.

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.637 | −0.119 | Retrieve current clipboard content | Enable Wi-Fi |
| 2 | 0.636 | −0.037 | Retrieves the clipboard content using a workflow action. | Exits the workflow if there is no internet connection. |
| 3 | 0.614 | −0.049 | Retrieves the current content from the clipboard and stores it in `clipboard_content`. | Logs out of the workflow actions to ensure a clean start. |
| 4 | 0.636 | −0.002 | 11. Retrieve Clipboard Contents: Get contents stored in the clipboard. | 10. Send Jailbreak Alert: Alert the user about the jailbreak necessity. |

> **What changed:** All four have negative or near-zero x_sim — source anchors
> are about Wi-Fi toggling, connectivity checks, logout steps, and jailbreak
> alerts. Their positives all describe reading from the clipboard (y_sim ~0.63)
> which is genuinely close to the true positive (also about clipboard content).
> The model must distinguish "reads clipboard" from "reads clipboard then runs
> arXiv regex" without any anchor-side hints.

---

### Sample 2 — idx 50

**Anchor**
> Defines a dictionary named `shortcut_info` that contains the URLs and
> version information for the shortcut.

**True Positive**
> Extracts the version number from the `shortcut_info` dictionary and
> assigns it to the variable `v`.

| # | y_sim | x_sim | Hard Negative (positive text) | Source Anchor |
|---|------:|------:|-------------------------------|---------------|
| 1 | 0.824 | 0.146 | Extracts the current version of the shortcut from the `shortcut_data` dictionary. | A placeholder statement that does nothing; it serves as a syntactically valid statement for the body that follows. |
| 2 | 0.742 | −0.004 | Define a dictionary containing metadata for the shortcut with an ID and version. | Check if the input value is empty; if it is, proceed with the workflow actions. Invoke a function to handle the workflow handoff. |
| 3 | 0.728 | 0.030 | Creates a dictionary called `shortcut_metadata` that maps the string 'ID' to `shortcut_id` and 'Version' to `shortcut_version`. | Alerts the user with a message 'Wrong password!!!' if the password is incorrect. Initiates a loop… |
| 4 | 0.778 | 0.139 | Define the version number and date of the shortcut | Define a variable for the reminder contact name |

> **What changed:** HN 1 (y_sim=0.824) is the highest-similarity negative —
> "extracts the current version from a shortcut dictionary" — and is retained.
> But its source anchor is a `pass` placeholder statement (x_sim=0.146),
> completely different from defining a shortcut_info dict. HN 2 and 3 have
> x_sim near zero, from anchors about empty-input checks and wrong-password
> loops. The model sees very hard positives (version extraction, shortcut
> metadata) but with no anchor overlap to lean on.
