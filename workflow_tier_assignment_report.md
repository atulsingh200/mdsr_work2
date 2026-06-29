# Which Tier Does Each Workflow Text Belong To?

Tier definitions are from `build_classification_data.py` (the 15-tier curriculum used to build the dataset). Each text is assigned by combining (a) what the tier *means* and (b) the tier of its most-similar training texts, under both **semantic** (MiniLM) and **lexical** (BM25) retrieval.

---

## `t1` — Select all profiles who added items to their cart but did not complete a purchase in the last 24 hours. Use the rule builder to filter by cart status equals 'active' and no order event.

| Method | Predicted tier | Tier name | Vote |
|---|---|---|---|
| Semantic | **15** | Use cases, labs, capstones | 5/5 |
| BM25 | **15** | Use cases, labs, capstones | 4/5 |

### Similar texts and why they point to a tier

**Semantic top-5:**

1. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), sim `0.499`
   > Since the events are followed by a 3 day wait before we check if the customer has been inactive since the Product View. So how do we check if the profile engaged with the brand? Remember, engagement with the brand means browsing the website
2. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), sim `0.458`
   > And instead of listening to the Add to Cart after the Product View for an hour, you can listen for any brand engagement for 3 days. In that case, we won’t require a wait activity and the profile will exit the journey as soon as they engage 
3. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), sim `0.451`
   > We’re going to showcase this in a retail scenario, but you can apply this to almost any industry. Maybe you’re a financial institution and somebody reads about your auto loans but doesn’t apply for one. Next is abandoned cart. In this one, 
4. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), sim `0.445`
   > If your data structure has all event information in one schema, you can simply create a brand engagement event. And instead of listening to the Add to Cart after the Product View for an hour, you can listen for any brand engagement for 3 da
5. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), sim `0.443`
   > Like the cart ID, the ID and value for the product list ads, as well as the required product catalog data. Such as the image URL, the SKU, product name, price and so on. Once the events are available, you can add them to your journey. You c

**BM25 top-5:**

1. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), BM25 `46.7`
   > We’re going to showcase this in a retail scenario, but you can apply this to almost any industry. Maybe you’re a financial institution and somebody reads about your auto loans but doesn’t apply for one. Next is abandoned cart. In this one, 
2. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), BM25 `41.7`
   > So that’s our first audience. We save it as a batch audience, which is entirely suitable because of the time horizon. Now, because of that one hour buffer, we need to be careful. We just created a blind spot and we don’t want people who mak
3. **Tier 8** (Personalization & decisioning) — sub `11b` = personalization editor: helper functions, contextual events, dynamic content, BM25 `40.9`
   > You may want to construct if then conditions where you can change what appears in the message based on the conditional value of an attribute. Another use could be in grabbing a list of items where you want to display values for each. For al
4. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), BM25 `39.2`
   > Like the cart ID, the ID and value for the product list ads, as well as the required product catalog data. Such as the image URL, the SKU, product name, price and so on. Once the events are available, you can add them to your journey. You c
5. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), BM25 `37.7`
   > Depending on when within the 24 hours after the Product View the first audience calculation runs, the timeframe of this blind spot can be longer or shorter, but it will never exceed 24 hours. So, to bridge the gap, we will define a second a

---

## `t2` — Enrich each profile with the product names, images, and prices from their abandoned cart. This data will be used to personalize the email content.

| Method | Predicted tier | Tier name | Vote |
|---|---|---|---|
| Semantic | **6** | Channels + campaigns (first sends) | 3/5 |
| BM25 | **15** | Use cases, labs, capstones | 2/5 |

### Similar texts and why they point to a tier

**Semantic top-5:**

1. **Tier 6** (Channels + campaigns (first sends)) — sub `4c` = orchestrated campaigns (build audience, enrich, personalize, send), sim `0.543`
   > We are also able to manipulate the data that’s brought into this campaign by creating a filter, and sorting. For our campaign, we want to pull a few other attributes. We can pull in the product description, and we can pull in the product pr
2. **Tier 6** (Channels + campaigns (first sends)) — sub `4c` = orchestrated campaigns (build audience, enrich, personalize, send), sim `0.530`
   > Next, let’s create the email. We can start by personalizing the subject line of our email. We’ll pull in first name from each recipient’s profile attributes, and can access the enrichment data, our wishlist items that we brought into the ca
3. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), sim `0.530`
   > In our use case, there are different events used in this scenarios like viewing a product, adding it to their shopping cart or purchasing something accompanying these events. You need things like timestamps because the passage of time is an
4. **Tier 6** (Channels + campaigns (first sends)) — sub `4c` = orchestrated campaigns (build audience, enrich, personalize, send), sim `0.499`
   > After setting the conditions, we can view the resulting audience, and add in any attributes that we want to see, for a more detailed preview of our target audience and specific recipients. Now that we’ve built our audience for this re-engag
5. **Tier 9** (Advanced journey patterns & experimentation) — sub `5c` = advanced journey use-cases (transactional, audience qualification), sim `0.496`
   > And the supplemental identifier is also the product identifier, but it is within the collection within the object array. All right, now let’s go into the journey. So we’re going to fire off an event here and then we’re going to receive an e

**BM25 top-5:**

1. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), BM25 `33.2`
   > In our use case, there are different events used in this scenarios like viewing a product, adding it to their shopping cart or purchasing something accompanying these events. You need things like timestamps because the passage of time is an
2. **Tier 6** (Channels + campaigns (first sends)) — sub `4c` = orchestrated campaigns (build audience, enrich, personalize, send), BM25 `28.8`
   > We can see who was targeted, who was excluded, and why. Next, let’s proof our message content. We can jump into the email designer, and simulate the content to see how it will appear for test profiles. We can see that while more formatting 
3. **Tier 9** (Advanced journey patterns & experimentation) — sub `12b` = content experiments for emails, BM25 `27.6`
   > In A-J-O, the marketer can create a campaign to test and use an experiment report to see which users interact with different subject lines. After naming the campaign, providing a description, and choosing an audience, the marketer will desi
4. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), BM25 `26.5`
   > How do we collect the product details in case we want to display things like product names and images in our re-engagement messages? I’m collecting the product SKUs and names here in the Web SDK implementation. You can see them in this prod
5. **Tier 1** (Product orientation) — sub `2a` = Journey Optimizer overview / personas, BM25 `26.1`
   > They create and manage all the various components of these personalized journeys, including email and push messages. They also configure decision management and all of its components to intelligently personalize message content, media, and 

---

## `t3` — Send a personalized email reminding the customer of their left-behind items and offering a 10% discount to complete the purchase. Use the enriched cart data to populate the product block in the email template.

| Method | Predicted tier | Tier name | Vote |
|---|---|---|---|
| Semantic | **6** | Channels + campaigns (first sends) | 3/5 |
| BM25 | **8** | Personalization & decisioning | 3/5 |

### Similar texts and why they point to a tier

**Semantic top-5:**

1. **Tier 8** (Personalization & decisioning) — sub `11b` = personalization editor: helper functions, contextual events, dynamic content, sim `0.457`
   > And the next thing I want to do is I want to take the ending each tag, and I’m going to place it below the bottom of this block. So that I know this whole item is going to be repeated for each product, this whole section. Or validate that j
2. **Tier 6** (Channels + campaigns (first sends)) — sub `4c` = orchestrated campaigns (build audience, enrich, personalize, send), sim `0.447`
   > And more specifically, clicked on a link within that message. We’ll enhance this campaign further by splitting engagement by channel preference, and delivering a follow-up via their preferred channel, SMS, or push notification, with the goa
3. **Tier 6** (Channels + campaigns (first sends)) — sub `4c` = orchestrated campaigns (build audience, enrich, personalize, send), sim `0.445`
   > After setting the conditions, we can view the resulting audience, and add in any attributes that we want to see, for a more detailed preview of our target audience and specific recipients. Now that we’ve built our audience for this re-engag
4. **Tier 6** (Channels + campaigns (first sends)) — sub `9d` = email channel, sim `0.443`
   > If we’re happy with this personalized preview, we can then send a proof as the final test for our email. Sending the proof is easy. Here we can directly enter the emails that we want to send the proof to. If you have different offers or con
5. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), sim `0.440`
   > Now let’s talk about personalization of the messages. Let me navigate into the email. Now, if you would like to refer to the product the customer reviewed in your communication, you have two options. You can personalize the message either b

**BM25 top-5:**

1. **Tier 8** (Personalization & decisioning) — sub `11b` = personalization editor: helper functions, contextual events, dynamic content, BM25 `48.6`
   > You may want to construct if then conditions where you can change what appears in the message based on the conditional value of an attribute. Another use could be in grabbing a list of items where you want to display values for each. For al
2. **Tier 8** (Personalization & decisioning) — sub `13a` = decisioning (offers / decision management), BM25 `43.6`
   > As the marketer, I plan to send a push notification to my entire customer base of opted-in push subscribers, but I want to show an offer that’s more compelling to the recipient, rather than a generic one-size-fits-all discount. We’ll go ahe
3. **Tier 9** (Advanced journey patterns & experimentation) — sub `5c` = advanced journey use-cases (transactional, audience qualification), BM25 `43.3`
   > The journey is triggered by a product purchase event followed by a product purchase confirmation email, the product shipped event, and a condition check, then trigger the product sent email. In this example, we want to send the same profile
4. **Tier 15** (Use cases, labs, capstones) — sub `20a` = abandoned-cart / customer-onboarding use-case (Luma retail scenario), BM25 `42.7`
   > We’re going to showcase this in a retail scenario, but you can apply this to almost any industry. Maybe you’re a financial institution and somebody reads about your auto loans but doesn’t apply for one. Next is abandoned cart. In this one, 
5. **Tier 8** (Personalization & decisioning) — sub `11b` = personalization editor: helper functions, contextual events, dynamic content, BM25 `42.3`
   > All right, will validate this make sure there are no errors and then I can click save. All right, that should be it. Let’s take a look at how these Helper Functions worked in my emails. So in this first email, I can see that the customer na

---
