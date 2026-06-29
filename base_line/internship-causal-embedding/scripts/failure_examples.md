# Failure examples — where the model picks a negative over the positive

For each example: the anchor, the gold positive (along with the model's similarity score), and the 4 negatives (BM25 or semantic kNN, depending on the experiment). The negative marked **(beats pos)** is the one whose similarity to the anchor exceeded the gold positive's — that's what causes the failure.


---

## BM25 hard negatives · followupqg · MiniLM-L6-v2 (Exp 2)

### Example 1  (gap = +0.5809)

**Anchor.** ELI5: Why do gas prices include the extra 9/10 cent on the end? That is an ancient consumer marketing technique. A very long time ago, people in marketing figured out that consumers perceive a price of, say, $1.99 to be significantly less than a price of $2.00. Of course, those prices are not actually very different, but people tend to round down …

**Gold positive  (sim = -0.0244)**  
> So what you’re saying is that we’re all being hornswoggled?

**Negatives:**

- neg #1  sim = 0.5564  **(beats pos)**  
  > If the prices aren’t even that high than why is gas price higher? And surely there’s the same demand right?
- neg #2  sim = 0.4907  
  > so why do gasoline prices wax and wane consistently via different presidential administrations with different agendas? why do gasoline prices vary so widely between nations and different states, regardless of their purported subsidization of "big oil" companies? are you intimating that the rapid increase of price per gallon "at the pump" in the u.…
- neg #3  sim = 0.1435  
  > So, and affirming that you are no expert, you would assume that the only difference is the size of the L/P but even across species they are physically the same? (like if you were to compare my L/P to a mouse's L/P they would look exactly the same but be different size)
- neg #4  sim = 0.3094  
  > So basically, in my example, the guy who owns one share only has the benefit of being able to sell that share in the future for a (hopefully) greater price than he originally bought it for? How is that any different than buying an NFT?


### Example 2  (gap = +0.5206)

**Anchor.** ELI5: If an insect is inside of a container, but flying, is it adding weight to the container? The crux of the answer is, it depends if there's a lid on the container. If there's no lid then it's not adding weight, if there is a lid then it is.

**Gold positive  (sim = 0.2184)**  
> I'm skeptical. What if the lid is there, but open just a crack? What if the lid is open halfway?

**Negatives:**

- neg #1  sim = 0.3045  
  > I heard on a podcast that putting a lid on a pot helps it boil because it increases the pressure. I guess what it is saying is that it helps contain the water vapor and increase the vapor pressure. I suppose putting a lid on the pot wouldn’t increase the ambient pressure because it isn’t an airtight seal?
- neg #2  sim = 0.0141  
  > Rescue breaths aren't as important if the heart is stopped, right? Like if the body isn't moving the blood around there's no reason to blow air into their lungs.
- neg #3  sim = 0.7389  **(beats pos)**  
  > Will the added weight be as much as the weight of the insect itself?
- neg #4  sim = 0.2813  
  > Also if there is no wind, the air is moving at same speed (velocity??) also so there's no force moving you out of position.


### Example 3  (gap = +0.4816)

**Anchor.** ELI5: Why does taking a sip of coffee make someone who has adhd very sleepy instead of energized? Coffee doesn’t energize you much, it just stops you from from feeling sleepy. However, it also seems to have some effects resembling those of ADHD medications. So, when you drink coffee it makes you feel normal in terms of sleepiness, and then also cu…

**Gold positive  (sim = 0.1633)**  
> Speed? Yes. Because they're both Stimulants

**Negatives:**

- neg #1  sim = 0.6448  **(beats pos)**  
  > I can't say whether they're similar or not but I can say for sure that every type of coffee makes me feel like that? Take out, homemade, from espresso to latte, so I bet I sometimes consume less caffeine than in an energy drink and still feel that way. That's why I was wondering if there's some other factor that differentiates coffee from energy d…
- neg #2  sim = -0.0009  
  > Is that why when you touch something you feel the sensation instantly but the pain takes longer? Also when you stub your toe you can sense the incoming pain before it actually starts. Why’s that?
- neg #3  sim = 0.4775  
  > Is caffeine kind of just stuck on coffee like dust and washes off?
- neg #4  sim = 0.0703  
  > It's untenable from a practical standpoint. What if you don't have the money to pay it? Are you forced to sell? Will you then get charged capital gains on selling? Do you get a tax refund for taxes paid on gains that are never realized because of losses on the value? The second hand negative effects of something like this are insane. So insane it …


---

## BM25 hard negatives · workflow · MiniLM-L6-v2 (Exp 2)

### Example 1  (gap = +1.0489)

**Anchor.** Check if the folder already exists, if not create it

**Gold positive  (sim = -0.0489)**  
> Function to retrieve user settings

**Negatives:**

- neg #1  sim = 1.0000  **(beats pos)**  
  > Check if the folder already exists, if not create it
- neg #2  sim = 0.9567  
  > Check if the folder exists; if not, create it
- neg #3  sim = 0.8191  
  > Function to create a folder if it does not exist
- neg #4  sim = 0.2068  
  > Check if the current follower is not already counted in the previous follower count


### Example 2  (gap = +1.0379)

**Anchor.** Open the game developer's website Step 21: Open the developer's website

**Gold positive  (sim = -0.1271)**  
> Retrieve related podcast episodes

**Negatives:**

- neg #1  sim = 0.9107  **(beats pos)**  
  > Open the game developer's website
- neg #2  sim = 0.1868  
  > Open the Allrecipes website
- neg #3  sim = 0.0899  
  > Open the Nutrition Data website
- neg #4  sim = 0.4616  
  > Open the website if available


### Example 3  (gap = +0.9990)

**Anchor.** If the workflow is active, proceed with further actions.

**Gold positive  (sim = 0.0010)**  
> Get the type of item based on the URL component retrieved.

**Negatives:**

- neg #1  sim = 1.0000  **(beats pos)**  
  > If the workflow is active, proceed with further actions.
- neg #2  sim = 0.4886  
  > Step 6: If the task type is 'Reminder', proceed with further actions.
- neg #3  sim = 0.4000  
  > If the document picker is not open, proceed with the following actions.
- neg #4  sim = 0.5241  
  > If the name does not match, proceed with the following actions.


---

## Semantic kNN hard negatives · followupqg · fine-tuned BERT (Exp 6)

### Example 1  (gap = +0.2373)

**Anchor.** ELI5: I’m often shocked at the way people sound talking vs singing, especially when it comes to accents. Why does the singing voice sound different from the speaking voice? (extra emphasis on accents but also tone, timbre, etc.) Tone changes based on the position and tension of the tongue, larynx and soft palatte. Mixing different combos of those …

**Gold positive  (sim = 0.5118)**  
> What about accents though? There's a guy a follow on youtube that makes metal covers of songs, but he's got a thick Norwegian accent that's super hard to understand as an english-only speaker, so at the end of his videos when he does his little outro, he kind of half-sings it and makes his accent almost completely go away and I can understand him …

**Negatives:**

- neg #1  sim = 0.7491  **(beats pos)**  
  > Ok, so how do people sing in tune if their voice sounds different?
- neg #2  sim = 0.6364  
  > >2 people with identical accents might argue over how to pronounce scone - does it rhyme with cone or gone? Or, indeed, with [spoon](https://forvo.com/word/stone_of_scone/)?
- neg #3  sim = 0.4970  
  > Do you know what the purpose of the intro overdub is? Asking because as a listener I've always detested it. When the talking stops and the volume goes up it's like suddenly finding yourself in the middle of a lake (of sound) when you weren't expecting to be swimming. Similarly, I imagine the artists must hate it because a song is meticulously craf…
- neg #4  sim = 0.7052  
  > Is it a good enough explanation that if every language picked the tone randomly independently, 70-30 split would not be that unusual?


### Example 2  (gap = +0.2361)

**Anchor.** ELI5:What makes colours? I'll take a stab at this, even though I'm sure someone here can do it better. Light — which contains all of the possible colors we can see — is absorbed into everything. When we look at something and see its color, what we see is the frequency (or color) of light that isn't absorbed by it. So think about it this way. You h…

**Gold positive  (sim = 0.4774)**  
> That makes perfect sense thanks so much! Edit, so when we paint something a different colour, all were doing is applying a substance that spits out a different frequency of colour?

**Negatives:**

- neg #1  sim = 0.6128  
  > Ok that makes sense. But then the question is: do we know why our brains assign the same or similar perceived color for a mix of blue and red (purple) as "true" violet?
- neg #2  sim = 0.4537  
  > Thanks for the reply! Why can’t white surfaces be polished or microscopically made to be flat?
- neg #3  sim = 0.7135  **(beats pos)**  
  > yes this makes sense, a lot of sense. So to make sure I understand... my "green" cones can detect every color, just in the same way I can see every color, even that I may have green sunglasses on (yes it is properly more the complimentary color, but just for fun) and the same with blue and red. Things are just easier to see / clear if the color is…
- neg #4  sim = 0.6285  
  > I may sound stupid here, but doesn't there needs to be something to not reflect the white color to make the light color white?


### Example 3  (gap = +0.2164)

**Anchor.** ELI5: How are allergies related to the immune system? Your digestive tract is essentially a tube going through your body. It's where elements from outside of your body have an opportunity to go inside your body. A strong digestive tract prevents this. Between the cells that line your intestines, you have something called tight-junction. When your …

**Gold positive  (sim = 0.4543)**  
> She was telling me about diet and I think about this too!! Saying stuff about leaky gut... I thought it sounded suspicious though but ultimately you have had good results?!

**Negatives:**

- neg #1  sim = 0.3620  
  > This really needs to be higher up. "Use more calories than you eat" is reductionist. Sure it'll work for a lot of people but there are *many, many* people for whom this is just not medically, physiologically true. I decided to lose weight in 2018, as well as improve all the numbers actually associated with health-- resting heart rate, etc. In orde…
- neg #2  sim = 0.6706  **(beats pos)**  
  > She suggested i take an IgG food allergy test? How are IgE and IgG related to allergies specifically? So can you have a hyper sensitive super active and strong immune system that also leads to allergies ?
- neg #3  sim = 0.4628  
  > This is what I was looking for. I took some antibiotics (Amoxicillin) that threw off the balance in my gut. Lots of pickles has helped. May I ask what else has worked for you?
- neg #4  sim = 0.5345  
  > This is super interesting. Just curious if you by chance have any idea why you also get that taste when you're given IV fluids?


---

## Semantic kNN hard negatives · workflow · fine-tuned BERT (Exp 6)

### Example 1  (gap = +0.6008)

**Anchor.** Initializes a variable 'sunglasses_art' containing the visual setup of this ASCII art.

**Gold positive  (sim = 0.3215)**  
> Shows the setup phase of the sunglass art.

**Negatives:**

- neg #1  sim = 0.5245  
  > Displays what this artwork will represent to the user.
- neg #2  sim = 0.9222  **(beats pos)**  
  > Initializes a variable 'sunglasses_art' containing the visual setup of this ASCII art.
- neg #3  sim = 0.6376  
  > This line replaces the 'Setup' part of the sunglasses art with the user's provided setup input.
- neg #4  sim = 0.1949  
  > Display the artwork.


### Example 2  (gap = +0.5887)

**Anchor.** Comments stating that the function 'completion' is called to indicate the end of the script execution.

**Gold positive  (sim = 0.3247)**  
> Calls the completion function with 'true' to indicate that the inversion was successful.

**Negatives:**

- neg #1  sim = 0.5767  
  > Calls a function named completion with the 'result' variable, presumably to handle or display the final output.
- neg #2  sim = 0.9134  **(beats pos)**  
  > Comments stating that the function 'completion' is called to indicate the end of the script execution.
- neg #3  sim = 0.5321  
  > Calls the completion function to finalize the JavaScript operation.
- neg #4  sim = 0.5301  
  > Calls a completion function passing the result of the operation, signaling that the JavaScript execution is completed.


### Example 3  (gap = +0.5878)

**Anchor.** Checks if 'IsValidTimeStamp' is True; if valid, proceeds with updating the configuration.

**Gold positive  (sim = 0.3141)**  
> Updates the 'Time' key in the 'Config' dictionary with the validated segment timestamp.

**Negatives:**

- neg #1  sim = 0.1699  
  > Displays an alert indicating that the segment time is not valid, with the specific time entered.
- neg #2  sim = 0.2830  
  > Checks if the updated time entry details exist.
- neg #3  sim = 0.9019  **(beats pos)**  
  > Checks if 'IsValidTimeStamp' is True; if valid, proceeds with updating the configuration.
- neg #4  sim = 0.0762  
  > Step 12: Update the time entry
