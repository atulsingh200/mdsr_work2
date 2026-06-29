# Workflow Text → Tier Report (BM25 Lexical Search)

**Corpus:** `directional_train.jsonl` (924 unique texts, 15 tiers)  
**Method:** BM25Okapi keyword scoring over lowercased word tokens.  
For each query, the 5 highest-BM25 training texts; predicted tier = majority vote.

---

## Query `t1`

> Select all profiles who added items to their cart but did not complete a purchase in the last 24 hours. Use the rule builder to filter by cart status equals 'active' and no order event.

**Predicted tier: 15** (won 4/5 of top neighbors)  
**Tier vote spread:** {15: 4, 8: 1}

### Top 5 BM25 matches

**1. Tier 15**  (sub `20a`, BM25 score = `46.74`)

> We’re going to showcase this in a retail scenario, but you can apply this to almost any industry. Maybe you’re a financial institution and somebody reads about your auto loans but doesn’t apply for one. Next is abandoned cart. In this one, a customer adds a product to their shopping cart but doesn’t complete the purchase. Finally, we have an order confirmation scenario which is sending a message after a purchase or other conversion event.

**2. Tier 15**  (sub `20a`, BM25 score = `41.70`)

> So that’s our first audience. We save it as a batch audience, which is entirely suitable because of the time horizon. Now, because of that one hour buffer, we need to be careful. We just created a blind spot and we don’t want people who make a purchase or add a product to their cart during that hour to qualify for our paid media campaign. We don’t want to spend ad dollars on people who just bought something from us and people who added something to their cart but didn’t purchase.

**3. Tier 8**  (sub `11b`, BM25 score = `40.86`)

> You may want to construct if then conditions where you can change what appears in the message based on the conditional value of an attribute. Another use could be in grabbing a list of items where you want to display values for each. For all of these scenarios and many others you’ll want to use the Helper Functions and the personalization editor. Let’s take a look. In this example, I have a Journey Optimizer message that will be sent to a customer if they’ve left some items in their shopping cart, but have not completed the purchase.

**4. Tier 15**  (sub `20a`, BM25 score = `39.20`)

> Like the cart ID, the ID and value for the product list ads, as well as the required product catalog data. Such as the image URL, the SKU, product name, price and so on. Once the events are available, you can add them to your journey. You can just work with a Product View event, but in this retail scenario, listening to an Add to Cart for an hour after a Product View allows you to exit profiles who have a buying intention from your journey earlier. Without this event, all profiles will stay in the journey for at least 3 days.

**5. Tier 15**  (sub `20a`, BM25 score = `37.75`)

> Depending on when within the 24 hours after the Product View the first audience calculation runs, the timeframe of this blind spot can be longer or shorter, but it will never exceed 24 hours. So, to bridge the gap, we will define a second audience, which looks for profile engagement within the last 24 hours. This audience can be of type streaming or will be of type streaming, so a profile gets added in real time. Unfortunately, we do have yet another issue. Because we added an hour wait if our audience calculation runs within an hour of the condition check, then you can see here all profiles in the journey will disqualify from the batch, because the first rule of our batch audience, the Product View, falls out of the lookback timeframe.

---

## Query `t2`

> Enrich each profile with the product names, images, and prices from their abandoned cart. This data will be used to personalize the email content.

**Predicted tier: 15** (won 2/5 of top neighbors)  
**Tier vote spread:** {15: 2, 6: 1, 9: 1, 1: 1}

### Top 5 BM25 matches

**1. Tier 15**  (sub `20a`, BM25 score = `33.15`)

> In our use case, there are different events used in this scenarios like viewing a product, adding it to their shopping cart or purchasing something accompanying these events. You need things like timestamps because the passage of time is another important element of qualification. Finally, we have messaging. What message do you want to show them? If you want to show them the actual products they browsed, abandoned or purchased, then you need to collect SKUs, product names, images or things like that.

**2. Tier 6**  (sub `4c`, BM25 score = `28.85`)

> We can see who was targeted, who was excluded, and why. Next, let’s proof our message content. We can jump into the email designer, and simulate the content to see how it will appear for test profiles. We can see that while more formatting is needed, we are getting the fully personalized three wishlist items in the message, with the right information, images, and prices for this profile. We can additionally send proofs to others for reviews, finalization, and publishing the campaign.

**3. Tier 9**  (sub `12b`, BM25 score = `27.57`)

> In A-J-O, the marketer can create a campaign to test and use an experiment report to see which users interact with different subject lines. After naming the campaign, providing a description, and choosing an audience, the marketer will design the first email. For this experiment, the same email template will be used for each treatment. The images, links, and text in each email will be the same. The only changes will be in the subject line of the email.

**4. Tier 15**  (sub `20a`, BM25 score = `26.50`)

> How do we collect the product details in case we want to display things like product names and images in our re-engagement messages? I’m collecting the product SKUs and names here in the Web SDK implementation. You can see them in this product list items array. The image URLs. I don’t collect client side.

**5. Tier 1**  (sub `2a`, BM25 score = `26.07`)

> They create and manage all the various components of these personalized journeys, including email and push messages. They also configure decision management and all of its components to intelligently personalize message content, media, and other creative assets that will be used in messages and offers. They also define audiences for targeting groups of customers within specific journeys. Another important persona for Journey Optimizer is the Data Architect or Engineer, and this person would be primarily involved in setting up and maintaining the customer profile data and other data sources that will be used to power the experiences that are orchestrated by Journey Optimizer. This can include modeling customer profile data and business data into schemas, and configuring source connectors to ingest that data into Adobe Experience Platform.

---

## Query `t3`

> Send a personalized email reminding the customer of their left-behind items and offering a 10% discount to complete the purchase. Use the enriched cart data to populate the product block in the email template.

**Predicted tier: 8** (won 3/5 of top neighbors)  
**Tier vote spread:** {8: 3, 9: 1, 15: 1}

### Top 5 BM25 matches

**1. Tier 8**  (sub `11b`, BM25 score = `48.58`)

> You may want to construct if then conditions where you can change what appears in the message based on the conditional value of an attribute. Another use could be in grabbing a list of items where you want to display values for each. For all of these scenarios and many others you’ll want to use the Helper Functions and the personalization editor. Let’s take a look. In this example, I have a Journey Optimizer message that will be sent to a customer if they’ve left some items in their shopping cart, but have not completed the purchase.

**2. Tier 8**  (sub `13a`, BM25 score = `43.63`)

> As the marketer, I plan to send a push notification to my entire customer base of opted-in push subscribers, but I want to show an offer that’s more compelling to the recipient, rather than a generic one-size-fits-all discount. We’ll go ahead and create three separate offers. The first, a 25% off running shoes offer. The second, a 25% off shorts and leggings offer. And lastly, a 10% off site-wide generic fallback.

**3. Tier 9**  (sub `5c`, BM25 score = `43.30`)

> The journey is triggered by a product purchase event followed by a product purchase confirmation email, the product shipped event, and a condition check, then trigger the product sent email. In this example, we want to send the same profile through this event twice. We’re going to trigger it once for a product P1 and then a second time for product P2. We’ll see that coming through in the emails. However, when we get to our second event, we’re going to be using an object array that’s going to have P1 and P2 in it.

**4. Tier 15**  (sub `20a`, BM25 score = `42.72`)

> We’re going to showcase this in a retail scenario, but you can apply this to almost any industry. Maybe you’re a financial institution and somebody reads about your auto loans but doesn’t apply for one. Next is abandoned cart. In this one, a customer adds a product to their shopping cart but doesn’t complete the purchase. Finally, we have an order confirmation scenario which is sending a message after a purchase or other conversion event.

**5. Tier 8**  (sub `11b`, BM25 score = `42.32`)

> All right, will validate this make sure there are no errors and then I can click save. All right, that should be it. Let’s take a look at how these Helper Functions worked in my emails. So in this first email, I can see that the customer name Brenda has been transformed to all uppercase and the each function is now showing all the different items that were left in the cart. Because the Juno jacket is not one of these items I am not seeing those special product notes that we had conditionally added to indicate that there was a two week lead time in shipping that particular product.

---
