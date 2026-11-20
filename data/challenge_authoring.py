"""AI-authored synthetic challenge set, frozen before model evaluation.
Author: Codex data_evaluation agent; all examples written independently of model outcomes.
This is diagnostic synthetic evaluation, not naturally collected customer traffic.
"""
from pathlib import Path
import csv, hashlib, json, random
ROOT=Path(__file__).resolve().parents[1]
examples={
'cancel_order':[
'Please stop purchase 712 before the warehouse packs it.',
'I changed my mind about the headphones. Void the order, please.',
'Those shoes are no longer needed; can you cancel my purchase?',
'The gift arrived from someone else, so terminate my order.',
'Do not send the blender I bought yesterday. Cancel it.',
'I accidentally bought twice. Cancel the second purchase only.',
'Please withdraw order A19 completely; I do not want the items.'],
'change_order':[
'I bought a medium jacket, but I need a large on the same order.',
'Could you replace the blue mug in my purchase with the green one?',
'Make the quantity on my current order three instead of two.',
'Can another battery be added to the order I already submitted?',
'I selected the wrong laptop configuration. Edit the existing purchase.',
'Keep my order active but remove one of its four items.',
'I want to substitute the paperback for the hardback in order K27.'],
'change_shipping_address':[
'The parcel has not left yet. Send my current order to my office instead.',
'There is a typo in the delivery address on order C81. Please correct it.',
'My existing purchase needs to go to apartment 9, not apartment 6.',
'I moved since checkout. Redirect that order to my new street address.',
'Can I replace the destination for the camera I ordered this morning?',
'Please update the shipping location on my already-paid order.',
'The order confirmation lists my old home; change its delivery destination.'],
'check_cancellation_fee':[
'Will you charge anything if I cancel the purchase?',
'Before stopping my order, I need to know the cancellation penalty.',
'Is there a fee for backing out after checkout?',
'How much money would cancellation itself cost me?',
'Would you deduct a cancellation charge from what I paid?',
'Are late order cancellations subject to an extra charge?',
'Please explain the amount charged for withdrawing an order.'],
'check_invoice':[
'The tax total on my invoice looks incorrect. Can you check it?',
'Please explain why my invoice lists an extra service charge.',
'Can you verify the billing details on the invoice for order 83?',
'I already have the invoice; I need help understanding a line item.',
'Why does my invoice total differ from the checkout receipt?',
'Please inspect invoice INV91 for a duplicated charge.',
'Is the company name on my issued invoice correct?'],
'check_payment_methods':[
'Do you let customers pay with a debit card?',
'Which payment options are supported at checkout?',
'Can I use a bank transfer for this purchase?',
'Is cash on delivery an available way to pay?',
'Are digital wallets accepted by your store?',
'Can an order be paid for using a gift card?',
'What kinds of cards will your payment system accept?'],
'check_refund_policy':[
'What conditions must a purchase satisfy to qualify for a refund?',
'How many days do I have to ask for my money back?',
'Are opened products eligible under your refund rules?',
'Please show me the refund policy for discounted items.',
'Would a missing receipt make a purchase ineligible for reimbursement?',
'What is your policy on refunds after the return window closes?',
'Before buying, I want to understand when you allow refunds.'],
'complaint':[
'I want to file a formal complaint about how your staff treated me.',
'This is the third damaged package. Please register my complaint.',
'Your team repeatedly ignored my messages and I want to complain.',
'I am reporting unacceptable service from the delivery contractor.',
'Please escalate my complaint about the rude response I received.',
'I want this poor service incident recorded as an official grievance.',
'The way my support case was handled is unacceptable; lodge a complaint.'],
'contact_customer_service':[
'What is the email address for your customer service department?',
'Where can I find the customer support phone number?',
'How do I get in touch with the support team?',
'Please give me the contact details for customer care.',
'Is there a support email inbox I can write to?',
'Where is the customer service contact page on your site?',
'What channels can I use to reach your help department?'],
'contact_human_agent':[
'Please transfer this conversation to a real person.',
'I need a human representative to take over this chat now.',
'Can you connect me directly to a live support agent?',
'I would like to speak with a person rather than continue with the bot.',
'Hand this chat to a member of your staff, please.',
'Please put a live adviser on this conversation.',
'I need a human operator in this session to handle my case.'],
'create_account':[
'I am new here. Help me open a customer account.',
'How can I sign up for a profile with your store?',
'I want to create my first account on this website.',
'Please guide me through becoming a registered customer.',
'Where do new users go to make an account?',
'Can you help me set up a login for your shop?',
'I have never registered here and would like to join.'],
'delete_account':[
'Please permanently close my customer account.',
'I want my profile removed from your service.',
'How can I erase my account and stop being a member?',
'I will not use the shop again. Delete my user account.',
'Please explain how to terminate my registration permanently.',
'I want to remove my account entirely, not just log out.',
'Can you shut down the profile associated with my email?'],
'delivery_options':[
'Can I collect a purchase at a pickup point instead of home delivery?',
'Do you offer express shipping as an alternative to standard shipping?',
'What delivery services can I choose at checkout?',
'Is store pickup available for online purchases?',
'Can packages be delivered to a parcel locker?',
'Which shipping options are offered for my area?',
'Is there a courier delivery option for this product?'],
'delivery_period':[
'How long does standard delivery usually take?',
'If I order today, roughly when should it arrive?',
'What is the expected delivery time for a laptop?',
'How many business days does shipping normally need?',
'Will a new purchase made now arrive before next Friday?',
'What delivery window should I expect after checkout?',
'How soon could a newly placed order reach me?'],
'edit_account':[
'I need to change the email address on my customer profile.',
'How do I update my phone number in account settings?',
'Please help me correct my name on my profile.',
'I want to update my account preferences.',
'Where can I edit the personal details stored in my account?',
'My profile has my old surname; how do I change it?',
'Can I replace the contact number linked to my login?'],
'get_invoice':[
'Please send me a PDF invoice for order 302.',
'Where can I download the invoice for my purchase?',
'I need a copy of my invoice for an expense claim.',
'Could you email the billing invoice for last week\'s order?',
'I lost the invoice and need another copy.',
'Please issue an invoice document for the printer I bought.',
'How do I obtain the invoice for my most recent transaction?'],
'get_refund':[
'The item was returned yesterday. Please refund my payment.',
'I want my money back for the broken kettle.',
'Please start a refund for order 99.',
'The purchase was cancelled. I am requesting reimbursement.',
'Can you return the amount I paid for this faulty product?',
'I need to submit a refund request for the missing item.',
'Please process a refund for the order that never arrived.'],
'newsletter_subscription':[
'Please remove me from your promotional mailing list.',
'I would like to subscribe to your store newsletter.',
'How can I stop receiving your marketing emails?',
'Add my email to the weekly offers newsletter.',
'Where can I manage my newsletter subscription?',
'I unsubscribed by mistake; sign me up for updates again.',
'Please turn off newsletter emails while keeping my account.'],
'payment_issue':[
'My card was declined even though there is enough money in the account.',
'The payment page freezes each time I press pay.',
'I was charged twice when completing checkout.',
'The bank shows a payment, but your website says it failed.',
'Checkout keeps rejecting my payment and I need help.',
'I paid successfully but have not received confirmation of payment.'],
'place_order':[
'I would like to buy two desk lamps. How do I place the order?',
'Help me submit a new purchase through your website.',
'Where do I go to order this coffee machine?',
'I am ready to buy the items in my basket. What is the next step?',
'Can you guide me through purchasing a new pair of shoes?',
'How can I order the tablet shown in your catalogue?'],
'recover_password':[
'I forgot my password and need to reset it.',
'How do I regain access when I cannot remember my login password?',
'Please help me request a password reset link.',
'I am locked out because I lost my password.',
'Where is the forgotten-password option?',
'I cannot recall the password for my existing account.'],
'registration_problems':[
'The sign-up form says my email is invalid even though it is correct.',
'I tried to register, but the activation email never arrived.',
'Account creation fails with an error after I enter my details.',
'The register button does nothing when I press it.',
'I cannot finish signing up because the verification code is rejected.',
'The site keeps looping back to the registration page.'],
'review':[
'I want to leave a review of the headphones I bought.',
'Where can I submit a star rating for this product?',
'How do I write a public review after a purchase?',
'I would like to rate my shopping experience.',
'Can I edit the product review I posted yesterday?',
'Please show me where customer reviews can be submitted.'],
'set_up_shipping_address':[
'How do I save my home address for future deliveries?',
'I want to add a shipping address to my account before buying anything.',
'Where can I set my default delivery address?',
'Please help me enter my first shipping address in my profile.',
'Can I store a second delivery address for future purchases?',
'I need to configure a shipping address before placing my first order.'],
'switch_account':[
'I am signed in to my work profile and need to switch to my personal account.',
'How do I change from this login to my other existing account?',
'Please help me sign out of one account and into the other.',
'I have two customer accounts. How can I switch between them?',
'The site opened my partner\'s profile; I want to use my own account.',
'I want to use a different account for this session.'],
'track_order':[
'Where is order 501 right now?',
'Has my parcel left the warehouse yet?',
'Please show the tracking status for my existing purchase.',
'The carrier number is not in my email. How do I track the shipment?',
'Can you locate the package I ordered last Tuesday?',
'What is the current status of the order already on its way?'],
'track_refund':[
'My refund was approved last week. Where is the money now?',
'Can you check the status of the reimbursement already requested?',
'I am waiting for an approved refund to appear in my bank account.',
'Has refund request R41 been processed yet?',
'Please tell me what stage my existing refund is at.',
'When will the refund that you confirmed yesterday reach my card?']}
ambiguous=[
'I need to cancel my order and delete my account.',
'Please send the invoice and tell me where the package is.',
'I want a refund and to leave a review.',
'Change the size in my order and update my account phone number.',
'Please stop the newsletter and reset my forgotten password.',
'I want to open an account and place an order.',
'Can you check the refund status and the shipment status?',
'Connect me to a human and register a complaint.',
'Tell me your payment methods and your refund rules.',
'Change my current delivery address and send my invoice.',
'I need to switch accounts and cancel a purchase.',
'Check the fee for cancellation and the usual delivery time.',
'I want to save an address and edit my account name.',
'Help with my failed payment and my failed registration.',
'Please delete my account and unsubscribe me from offers.',
'I want to buy a lamp and return my old purchase for a refund.',
'Explain the invoice charge and tell me your support number.',
'Add an item to my order and tell me whether parcel lockers are available.',
'Reset my password and download my last invoice.',
'Where is my refund, and how can I rate the product?',
'I want to change my account email and join the newsletter.',
'Cancel order 7 and change the shipping destination of order 8.',
'Explain my refund eligibility and connect me to a live representative.',
'How do I register, and which cards do you accept?',
'I need the delivery estimate and a copy of my invoice.',
'Help me switch accounts and recover the password for the other one.',
'Please investigate the payment error and the extra charge on my invoice.',
'I need to complain about a package and check whether my refund is complete.',
'How do I choose express delivery and save a default shipping address?',
'Help me finish registration and stop the promotional emails.',
'Can you fix it for me?',
'I need to change something.',
'I am having trouble with my account.',
'Please check the status.',
'I need help with a charge.',
'Can I get it back?',
'I want to stop this.',
'The details are wrong.',
'I cannot get in.',
'I need a document.',
'How does this work?',
'I want to speak about my purchase.',
'It did not go through.',
'Please sort out the problem from yesterday.',
'Could you update that?',
'I need to get in touch.',
'Can you check my request?',
'Nothing has arrived.',
'Please remove the old one.',
'I want to add a new one.',
'Why am I still waiting?',
'I have a question about an email.',
'Please help with the address.',
'There is something wrong with the total.',
'I need to make a change before it is too late.',
'I want information about the policy.',
'Could you cancel that and maybe keep the account? I am not sure.',
'I cannot decide whether to return this or change the order.',
'I need either a new account or my old password; I am unsure which.',
'Can you look at my payment or invoice problem? I cannot tell which it is.'
]
ood=[
'What causes a lunar eclipse?',
'Write a poem about a sleeping fox.',
'How do I solve a quadratic equation?',
'Give me a recipe for spinach soup.',
'What is the capital of Mongolia?',
'Explain how photosynthesis works.',
'Who composed The Four Seasons?',
'Help me learn the rules of chess.',
'How many moons does Mars have?',
'Translate good morning into Finnish.',
'What is the difference between a comet and an asteroid?',
'Suggest a stretching routine after a run.',
'How do I prune a rose bush?',
'Tell me the history of the Roman Republic.',
'What is a prime number?',
'Explain the water cycle to a child.',
'How can I tune a guitar by ear?',
'Give me ideas for a science fair experiment.',
'Why do leaves change colour in autumn?',
'How do I bake sourdough bread?',
'What does a black hole look like?',
'Teach me a basic knitting stitch.',
'How does a steam engine operate?',
'Name three novels by Jane Austen.',
'Explain the offside rule in football.',
'What is the chemical symbol for tungsten?',
'How can I identify clouds before rain?',
'Write a bedtime story about a lighthouse.',
'How do I calculate the area of a triangle?',
'What animals live in the deepest ocean?',
'Help me plan a vegetable garden.',
'What is the origin of the word algebra?',
'Explain recursion with a simple programming example.',
'How do birds navigate during migration?',
'What is the difference between weather and climate?',
'Suggest a chord progression in a minor key.',
'What happened during the Renaissance?',
'How are mountains formed?',
'Give me a practice question about fractions.',
'How do I make a paper crane?',
'Why does metal expand when heated?',
'What is the purpose of sleep?',
'Tell me about the rings of Saturn.',
'How can I improve my freestyle swimming technique?',
'What is a metaphor?',
'Explain the structure of a sonnet.',
'How do I compost vegetable scraps?',
'What distinguishes a mammal from a reptile?',
'Write a short riddle about the moon.',
'How do I read a topographic map?',
'What is the Fibonacci sequence?',
'Explain how bees communicate.',
'How is glass manufactured?',
'What are the major parts of a violin?',
'Give me a breathing exercise before a presentation.',
'How do I tell the age of a tree?',
'What is the difference between a lake and a pond?',
'Why does bread rise?',
'How can I practice drawing perspective?',
'What does the periodic table organise?'
]
dev={
'cancel_order':'Please cancel purchase D55; I no longer need the charger.',
'change_order':'The active order should contain black socks instead of white socks.',
'change_shipping_address':'For the order already submitted, replace the destination with my parents\' home.',
'check_cancellation_fee':'Would stopping the purchase incur a penalty?',
'check_invoice':'Could you verify the VAT calculation on the invoice I received?',
'check_payment_methods':'Can I pay using a prepaid card?',
'check_refund_policy':'Does your refund policy cover products bought during a sale?',
'complaint':'I want to formally complain about a staff member\'s behaviour.',
'contact_customer_service':'How can I find your customer-care telephone details?',
'contact_human_agent':'Please hand me over to a live colleague in this chat.',
'create_account':'Show me the steps to register as a new shopper.',
'delete_account':'I would like my membership account permanently erased.',
'delivery_options':'Is weekend courier delivery an option?',
'delivery_period':'What is the normal transit time to receive a purchase?',
'edit_account':'How can I update the date of birth in my account?',
'get_invoice':'I need to retrieve the invoice PDF for the monitor purchase.',
'get_refund':'Please arrange reimbursement for the damaged suitcase.',
'newsletter_subscription':'I want to opt out of the weekly promotional bulletin.',
'payment_issue':'My payment is stuck processing and checkout will not finish.',
'place_order':'How do I complete a new purchase of a keyboard?',
'recover_password':'I need to reset the password I cannot remember.',
'registration_problems':'The new-user form will not accept my verification token.',
'review':'I want to post a rating for the suitcase I received.',
'set_up_shipping_address':'Help me store a default delivery destination for later orders.',
'switch_account':'Switch from my personal login to my business login.',
'track_order':'Has purchase D66 reached the local delivery depot?',
'track_refund':'What is happening with the reimbursement you already authorised?'
}
dev_ambiguous=['I need to stop an order and request an invoice.','Track my parcel and remove me from the newsletter.','Please reset my password and change my account phone.','I want to review a product and report bad service.','How can I register and what payment choices exist?','I need a refund and your customer support email.','Can you fix the login?','Something went wrong yesterday.','Please change the number.','I want to close it.','The email never came.','Where is it now?','Help me with my bill or payment, whichever it is.','I want to update my address but do not know whether the order exists yet.','Please check and change my details.','I have a problem with the thing I bought.']
dev_ood=['How does a rainbow form?','Write a limerick about a badger.','What is the square root of 144?','Suggest herbs for a pasta sauce.','Explain the tides.','What is the tallest mountain in Africa?','Teach me the first steps of juggling.','How is wool spun into yarn?','What are the colours of the visible spectrum?','Explain what a musical octave is.','How do butterflies emerge from a chrysalis?','Name the layers of the Earth.','Give me a beginner astronomy quiz.','What is an alliteration?','How do I fold a paper boat?','Tell me how a pendulum works.','Why do penguins huddle?']

# Added before v2 freeze after independent scope review, without inspecting model outputs.
near_ood = [
'Will this USB-C dock drive two 4K monitors at sixty hertz?',
'Is the silver laptop compatible with an external graphics card?',
'Does this phone work with an eSIM?',
'How much memory can I add to the desktop you sell?',
'Can this camera record video while connected to mains power?',
'Is the fabric of this jacket waterproof or just water resistant?',
'Does the blender jug tolerate boiling soup?',
'Will these headphones pair with a television over Bluetooth?',
'What is the maximum weight supported by the folding chair?',
'Is this toy suitable for a child under three years old?',
'Where can I find the instruction manual for the espresso machine?',
'What replacement filter fits the air purifier model AP40?',
'Is the blue backpack currently in stock?',
'When will size nine of the brown boots be restocked?',
'Do you have a demonstration unit of this monitor in the shop?',
'Are spare lids for the travel mug available?',
'Can you check whether the downtown branch has the green bicycle?',
'Is the older model of this speaker still being manufactured?',
'How long does the manufacturer warranty last on the vacuum?',
'Where is the nearest authorised repair centre for this camera?',
'Can a cracked screen be repaired under the extended warranty?',
'How do I register the serial number for warranty coverage?',
'Is battery replacement included in the service warranty?',
'What maintenance should I perform on this treadmill?',
'How many loyalty points are currently in my rewards balance?',
'When do the points from last year expire?',
'Can I transfer loyalty points to another family member?',
'How many rewards points are needed for gold membership?',
'Are points earned on gift-card purchases?',
'Can my loyalty points be redeemed for cinema tickets?'
]
near_dev_ood = [
'Will this docking station charge my laptop at ninety watts?',
'Is the winter coat machine washable?',
'Do you sell replacement wheels for the suitcase model S20?',
'Is the red kettle available in the uptown branch today?',
'When will the small yoga mat return to stock?',
'Does the toaster warranty include heating-element repairs?',
'Where should I take the drill for warranty servicing?',
'Can reward points be combined across my two loyalty cards?',
'Which membership tier gives double points on weekends?'
]

def write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def build():
    rows=[]
    for label,texts in examples.items():
        for text in texts: rows.append({'text':text,'label':label,'kind':'supported','acceptable_labels':json.dumps([label]),'expected_action':'route'})
    for kind,texts,action in [('ambiguous',ambiguous,'clarify_or_abstain'),('out_of_domain',ood,'abstain')]:
        for text in texts: rows.append({'text':text,'label':'__abstain__','kind':kind,'acceptable_labels':'[]','expected_action':action})
    development=[{'text':text,'label':label,'kind':'supported','acceptable_labels':json.dumps([label]),'expected_action':'route'} for label,text in dev.items()]
    for kind,texts,action in [('ambiguous',dev_ambiguous,'clarify_or_abstain'),('out_of_domain',dev_ood,'abstain')]:
        for text in texts:development.append({'text':text,'label':'__abstain__','kind':kind,'acceptable_labels':'[]','expected_action':action})
    assert len(rows)==300,len(rows)
    assert len(development)==60
    assert len(set(r['text'].casefold() for r in rows+development))==360
    replaced_ids=[]
    original_ood = [f'challenge_test_{i:03d}' for i in range(240,300)]
    reviewed_ood = set(sorted(original_ood, key=lambda i:hashlib.sha256(i.encode()).hexdigest())[:20])
    replacements = iter(near_ood)
    for i,row in enumerate(rows):
        row['ood_subtype'] = 'far_domain' if row['kind']=='out_of_domain' else ''
        key = f'challenge_test_{i:03d}'
        if row['kind']=='out_of_domain' and key not in reviewed_ood and len(replaced_ids)<30:
            row['text']=next(replacements);row['ood_subtype']='near_domain';replaced_ids.append(key)
    for i,row in enumerate(development):
        row['ood_subtype'] = 'far_domain' if row['kind']=='out_of_domain' else ''
        if row['kind']=='out_of_domain' and i-43 < len(near_dev_ood):
            row['text']=near_dev_ood[i-43];row['ood_subtype']='near_domain'
    assert len({r['text'].casefold() for r in rows+development})==360
    for split,items in [('test',rows),('development',development)]:
        for i,row in enumerate(items):
            row['example_id']=f'challenge_{split}_{i:03d}'
            row['author_provenance']='AI-authored by Codex data_evaluation agent before model outcomes; synthetic diagnostic request'
        random.Random(20261121).shuffle(items)
        write(ROOT/f'data/processed/challenge_{split}.csv',items,['example_id','text','label','kind','acceptable_labels','expected_action','author_provenance','ood_subtype'])
    blind=[]
    for kind in ['supported','ambiguous','out_of_domain']:
        selected=sorted([r for r in rows if r['kind']==kind],key=lambda r:hashlib.sha256(r['example_id'].encode()).hexdigest())[:20]
        blind.extend({'example_id':r['example_id'],'text':r['text']} for r in selected)
    random.Random(71).shuffle(blind)
    write(ROOT/'data/review_blind.csv',blind,['example_id','text'])
    additional=[{'example_id':r['example_id'],'text':r['text']} for r in rows if r['example_id'] in replaced_ids]
    random.Random(28).shuffle(additional)
    write(ROOT/'data/review_near_domain_blind.csv',additional,['example_id','text'])
    ontology={'labels':sorted(examples),'rules':{'route':'One clearly supported customer-support intent. Assign one ontology label.','clarify_or_abstain':'Multiple explicit intents or insufficient information to distinguish intents; label __abstain__.','abstain':'Outside this store-support ontology; label __abstain__. Product specifications and compatibility, manuals and maintenance, stock availability, warranty repairs, and loyalty points are unsupported by the 27-intent ontology even though they concern the same store.'},'definitions':{'contact_customer_service':'Ask for department contact details or general channels.','contact_human_agent':'Ask for a live person to join/take over current interaction.','check_invoice':'Inspect/understand/verify an existing invoice.','get_invoice':'Obtain/download/request an invoice copy.','delivery_period':'General expected time for a new purchase.','track_order':'Locate/status of an existing order.','set_up_shipping_address':'Add or set saved/default address for future purchases.','change_shipping_address':'Change destination on an existing order.','create_account':'How to create/register an account.','registration_problems':'A registration attempt is failing.','get_refund':'Initiate a refund.','track_refund':'Follow status of an already requested/approved refund.','complaint':'Lodge grievance about service.','review':'Write/edit a public rating or review.'}}
    (ROOT/'data/ontology.json').write_text(json.dumps(ontology,indent=2)+'\n',encoding='utf-8')
    manifest={'schema_version':2,'freeze_version':'v2-near-domain-ood','changes_from_v1':{'reason':'Independent review identified distant-only OOD as too easy before model freeze','replaced_final_ids':replaced_ids,'original_blind_sample_preserved':True,'test_ood_subtypes':{'near_domain':30,'far_domain':30},'development_ood_subtypes':{'near_domain':9,'far_domain':8}},'author':'Codex data_evaluation agent','creation_context':'No model predictions or evaluation results were viewed. Authored from ontology and task goals.','synthetic':True,'test_counts':{'supported':180,'ambiguous':60,'out_of_domain':60},'development_counts':{'supported':27,'ambiguous':16,'out_of_domain':17},'blind_review':{'sample_size':60,'selection':'20 deterministic hash-sampled requests per kind; reviewer receives no kinds or labels','status':'Initial 60 reviewed independently, with 60/60 agreement; additional 30 near-domain rows pending independent review','human_review':False},'files':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest() for name in ['processed/challenge_test.csv','processed/challenge_development.csv','review_blind.csv','review_near_domain_blind.csv','ontology.json']},'limitations':['Synthetic examples authored by one AI agent; diagnostic examples are not representative real production traffic.','A separate AI agent review is not a human annotation study.','English-only and narrow support domain.','Small per-intent challenge counts make individual class estimates noisy.']}
    (ROOT/'data/challenge_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print({'test':len(rows),'development':len(development),'blind':len(blind)})
if __name__=='__main__': build()
