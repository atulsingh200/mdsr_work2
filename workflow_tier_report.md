# Workflow Text → Tier Report (Nearest-Neighbor Mapping)

**Corpus:** `directional_train.jsonl` (924 unique texts, 15 tiers)  
**Embedder:** `all-MiniLM-L6-v2` (cosine similarity)  
**Method:** for each query, the 5 most similar training texts; predicted tier = majority vote.

---

## Query `t1`

> Select all profiles who added items to their cart but did not complete a purchase in the last 24 hours. Use the rule builder to filter by cart status equals 'active' and no order event.

**Predicted tier: 15** (won 5/5 of top neighbors)  
**Tier vote spread:** {15: 5}

### Top 5 most similar training texts

**1. Tier 15**  (sub `20a`, similarity = `0.499`)

> Since the events are followed by a 3 day wait before we check if the customer has been inactive since the Product View. So how do we check if the profile engaged with the brand? Remember, engagement with the brand means browsing the website or interacting with the app, adding a product to a cart or an online or offline purchase. So the online and offline purchase is important. Depending on your data structure, you have two options to implement this use case.

**2. Tier 15**  (sub `20a`, similarity = `0.458`)

> And instead of listening to the Add to Cart after the Product View for an hour, you can listen for any brand engagement for 3 days. In that case, we won’t require a wait activity and the profile will exit the journey as soon as they engage with the brand. But if you have the event data in different schemas, one for each data source for example like Luma does, then you cannot use an event to listen for overall brand engagement as each event can only link to one schema and Journey Optimizer can only listen to one event at a time per journey. So if you have your online and offline purchases in two different schemas, we will need to work with audiences and a condition that checks if the profile is part of an audience. If they are, they will exit the journey.

**3. Tier 15**  (sub `20a`, similarity = `0.451`)

> We’re going to showcase this in a retail scenario, but you can apply this to almost any industry. Maybe you’re a financial institution and somebody reads about your auto loans but doesn’t apply for one. Next is abandoned cart. In this one, a customer adds a product to their shopping cart but doesn’t complete the purchase. Finally, we have an order confirmation scenario which is sending a message after a purchase or other conversion event.

**4. Tier 15**  (sub `20a`, similarity = `0.445`)

> If your data structure has all event information in one schema, you can simply create a brand engagement event. And instead of listening to the Add to Cart after the Product View for an hour, you can listen for any brand engagement for 3 days. In that case, we won’t require a wait activity and the profile will exit the journey as soon as they engage with the brand. But if you have the event data in different schemas, one on your data structure, you have two options to implement this use case. If your data structure has all event information in one schema, you can simply create a brand engagement event.

**5. Tier 15**  (sub `20a`, similarity = `0.443`)

> Like the cart ID, the ID and value for the product list ads, as well as the required product catalog data. Such as the image URL, the SKU, product name, price and so on. Once the events are available, you can add them to your journey. You can just work with a Product View event, but in this retail scenario, listening to an Add to Cart for an hour after a Product View allows you to exit profiles who have a buying intention from your journey earlier. Without this event, all profiles will stay in the journey for at least 3 days.

---

## Query `t2`

> Enrich each profile with the product names, images, and prices from their abandoned cart. This data will be used to personalize the email content.

**Predicted tier: 6** (won 3/5 of top neighbors)  
**Tier vote spread:** {6: 3, 15: 1, 9: 1}

### Top 5 most similar training texts

**1. Tier 6**  (sub `4c`, similarity = `0.543`)

> We are also able to manipulate the data that’s brought into this campaign by creating a filter, and sorting. For our campaign, we want to pull a few other attributes. We can pull in the product description, and we can pull in the product price. Lastly, we can pull in the image URL for each of the products. With that, we have all the data that we need to create an individually personalized re-engagement message for each customer.

**2. Tier 6**  (sub `4c`, similarity = `0.530`)

> Next, let’s create the email. We can start by personalizing the subject line of our email. We’ll pull in first name from each recipient’s profile attributes, and can access the enrichment data, our wishlist items that we brought into the campaign, in case we’d like to add that to the subject line as well. We’ll start with an email template in the message designer, where we have access to all of our existing content fragments, and content for the brand. Let’s drag and drop a content fragment that we’ve saved previously.

**3. Tier 15**  (sub `20a`, similarity = `0.530`)

> In our use case, there are different events used in this scenarios like viewing a product, adding it to their shopping cart or purchasing something accompanying these events. You need things like timestamps because the passage of time is another important element of qualification. Finally, we have messaging. What message do you want to show them? If you want to show them the actual products they browsed, abandoned or purchased, then you need to collect SKUs, product names, images or things like that.

**4. Tier 6**  (sub `4c`, similarity = `0.499`)

> After setting the conditions, we can view the resulting audience, and add in any attributes that we want to see, for a more detailed preview of our target audience and specific recipients. Now that we’ve built our audience for this re-engagement campaign, we can personalize this campaign by enriching it with wishlist and product information. We can bring in recipient data, like their first name or preferred brand, and we can bring in non-recipient business data, such as the wishlist and product information. In this case, we can gather all wishlist item data. By default, we are retrieving three items per wishlist.

**5. Tier 9**  (sub `5c`, similarity = `0.496`)

> And the supplemental identifier is also the product identifier, but it is within the collection within the object array. All right, now let’s go into the journey. So we’re going to fire off an event here and then we’re going to receive an email. I’ll show you the email personalization before we fire off the event. The personalization here is just picking up the product name, the purchase date, the identifier and category from the journey context.

---

## Query `t3`

> Send a personalized email reminding the customer of their left-behind items and offering a 10% discount to complete the purchase. Use the enriched cart data to populate the product block in the email template.

**Predicted tier: 6** (won 3/5 of top neighbors)  
**Tier vote spread:** {6: 3, 8: 1, 15: 1}

### Top 5 most similar training texts

**1. Tier 8**  (sub `11b`, similarity = `0.457`)

> And the next thing I want to do is I want to take the ending each tag, and I’m going to place it below the bottom of this block. So that I know this whole item is going to be repeated for each product, this whole section. Or validate that just to make sure there are no errors, and then I’ll click save. Another user Helper Functions could be in this block here. So, I want to make sure that the customer sees any special product notes for a particular product.

**2. Tier 6**  (sub `4c`, similarity = `0.447`)

> And more specifically, clicked on a link within that message. We’ll enhance this campaign further by splitting engagement by channel preference, and delivering a follow-up via their preferred channel, SMS, or push notification, with the goal of driving conversion. The content for messages in each of these channels is created in the message designer, just as we did for email. And we can bring in profile attributes, such as the recipient’s name, and all of that enrichment data, in this case, the wishlist items, to include in those messages. For the final branch of our campaign, let’s design a notification to let our customers know when their wishlist items are back in stock.

**3. Tier 6**  (sub `4c`, similarity = `0.445`)

> After setting the conditions, we can view the resulting audience, and add in any attributes that we want to see, for a more detailed preview of our target audience and specific recipients. Now that we’ve built our audience for this re-engagement campaign, we can personalize this campaign by enriching it with wishlist and product information. We can bring in recipient data, like their first name or preferred brand, and we can bring in non-recipient business data, such as the wishlist and product information. In this case, we can gather all wishlist item data. By default, we are retrieving three items per wishlist.

**4. Tier 6**  (sub `9d`, similarity = `0.443`)

> If we’re happy with this personalized preview, we can then send a proof as the final test for our email. Sending the proof is easy. Here we can directly enter the emails that we want to send the proof to. If you have different offers or content in the email, you can add a prefix to the subject line to directly recognize which proof this is. So let’s just send this proof.

**5. Tier 15**  (sub `20a`, similarity = `0.440`)

> Now let’s talk about personalization of the messages. Let me navigate into the email. Now, if you would like to refer to the product the customer reviewed in your communication, you have two options. You can personalize the message either by using the contextual attributes coming from the Product View event. In this case it is the first product the user viewed, the product view that triggered the journey.

---
