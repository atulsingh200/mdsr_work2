### aep\_causal
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

### followupqg
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


---

### multiwoz\_v24

**Sample 1** *(test idx 0)*

> **Anchor**
> I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.

> **True Positive**
> What time do you want to leave and what time do you want to arrive by?

| # | Hard Negative |
|---|---------------|
| 1 | What time would you like to leave? |
| 2 | You have one leaving at 21:09, what time do you want to arrive by? |
| 3 | To narrow down the search, do you know what time you want to leave or what time you want to arrive by? |
| 4 | What time would you like to travel? | 

**Sample 2** *(test idx 50)*

> **Anchor**
> Yes please, I would like to book it for 7 people for 5 nights on Saturday, and I need a reference number. I'm very sorry I was unable to book at acorn guest house for 5 nights, would you like to try for a shorter stay? Yes, please try to book it for four nights instead.

> **True Positive**
> Your booking was successful, the reference number is WAEYAQ0M. May I assist you with anything else today?

| # | Hard Negative |
|---|---------------|
| 1 | The booking was successful. Your reference number is RWRKP9L6 |
| 2 | Booking was successful. Reference number is: 5INDNYF8. Is there anything else I can help you with today? | 
| 3 | your booking was successful, your reference number is 4O3358Y2. is ther anythin else i may help you with? | 
| 4 | Booking was successful. Your reference number is USCL95YH. Is there anything else I can help you with? | 

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
