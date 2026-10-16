# Error analysis

Predictions file: `predictions_afro-xlmr-base-weighted-seed42_test.csv`
Posts: 4154  |  correct: 2882  |  wrong: 1272  |  accuracy: 0.694

## Which classes get confused

| true -> predicted | count | share of that true class |
|---|---|---|
| negative -> negative | 1975 | 84.9% |
| negative -> neutral <- error | 7 | 0.3% |
| negative -> positive <- error | 344 | 14.8% |
| neutral -> negative <- error | 264 | 61.3% |
| neutral -> positive <- error | 167 | 38.7% |
| positive -> negative <- error | 490 | 35.1% |
| positive -> positive | 907 | 64.9% |

## Error rate by post length

| words | posts | error rate |
|---|---|---|
| 1-10 | 1189 | 0.405 |
| 11-20 | 1503 | 0.310 |
| 21-35 | 995 | 0.229 |
| 36+ | 467 | 0.206 |

## Error rate by vocabulary coverage

| share of words unseen in training | posts | error rate |
|---|---|---|
| (-0.001, 0.0476] | 2093 | 0.312 |
| (0.0476, 0.105] | 1027 | 0.273 |
| (0.105, 0.667] | 1034 | 0.329 |

## Emoji

| contains emoji | posts | error rate |
|---|---|---|
| no | 3912 | 0.310 |
| yes | 242 | 0.244 |

## Errors that are actually label noise

- Test posts that also appear in the training split: 926
- Of those, posts whose training label contradicts the test label: 113
- Posts where the model predicted the *training* label and was marked wrong: 24

Those last ones are cases where the model learned the label the dataset taught it and was then penalised for it by the same dataset. They are a ceiling on the achievable score, not a modelling failure.

Examples:

- train says `negative`, test says `positive`, model said `positive`: and then you go ahead and fuck him because the thing dey sweet man as e dey sweet woman
- train says `negative`, test says `positive`, model said `positive`: abeg come and buy na i sell laptops and phones iphone and samsung only i sell all brandskind of lapt
- train says `negative`, test says `positive`, model said `positive`: i get wise friends abeg ola told me something remarkable lately he was like bruh abeg in all you do 
- train says `negative`, test says `positive`, model said `positive`: if you love rap song go and listen to ft eriggaraw dey burst my head abeg every verse is raw even th
- train says `negative`, test says `positive`, model said `positive`: today marks one small milestone in the beautiful journey of lifei believe dat im the equivalent to a
- train says `neutral`, test says `positive`, model said `positive`: na dream shank just followed me im so emotional im so grateful to god for making me withness this mo
- train says `negative`, test says `positive`, model said `positive`: lmafoooooooooooooooooooooo na why i dey always use good morningevening maam yes maam for my married 
- train says `negative`, test says `positive`, model said `positive`: body hot like rihana na na kimani kill e show wah you into i want to really know

## Sample errors to read out loud

### True label: negative

- said `positive`: since burna dey make music for this kind p when the time reach na fem guys con dey jam i no wan push no agenda e jus funny
- said `positive`: we don waka be dat chaiiii
- said `positive`: punching bag yu are hit me hian na so dem teach u the floor while yu squating what to hit
- said `positive`: people don dey find themselves from twitter oo and that ilorin where everyone knows everyone twitter drama can become real very fa
- said `positive`: wetin i do una so them dey keep friendshipwell do ur tinno wahala
- said `positive`: pedro sef na wa

### True label: neutral

- said `negative`: me wey dey use light sleep n wake
- said `positive`: sugarman dis ur tips hmmmm ok we dey wait dem
- said `negative`: maybe coz dem nor who get the belle wey egg dey inside
- said `negative`: pikin wey say e wan get big teeth must grow big lip to cover am
- said `negative`: nah we were both in a hurry it was yrs ago sef i remembercus it was a day to oshuns wedding went to pick up my suit den
- said `negative`: make it pm abeg i de sleep

### True label: positive

- said `negative`: oya kolibi your phone make we collabo for the with jennifer omoko base on say we wan laughplay and dey serious weti be the biggest
- said `negative`: face palm knock knock na who be that na brother folo o which brother folo brother folo di lada folo di lada
- said `negative`: my last visit was chilling at marco polo hotel n suite na kilimanjaro everyday went to mainland to visit a family friend as uber d
- said `negative`: god come and do miracle for your boy ohhhh cause on my wedding day video wey i go dey ehn e go bloody oh salt on earth worm kinda 
- said `negative`: wahala for those wey sell
- said `negative`: breaking news the matter wey dey our table now be say aka starboy don talk say him go die for the matter one go let settle so simi

## Sample correct predictions

- `negative`: please stop late night meals your belle don dey big oo
- `positive`: i don go follow and patiently waiting for her for apple music should i blow powder before you notice me abeg na
