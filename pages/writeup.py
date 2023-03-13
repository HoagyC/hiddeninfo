# So the overall plan is something like:
# We want to be able to train a model, where the neuron activations have a particular signal, 
# which guides them to activate according to a particular feature.
# There are two main ways that we can train this feature - we either give a training signal which is purely the signal

# Benefits of pure training:
# Confidence that the network is intending to mean what we want it to mean

# Benefits of mixed training:
# Allows the use of the downstream signal to disambiguate between possible generalizations of the concept that we want
# This was the initial motivation in the ELK lens - we use the downstream training signal to push the question answerer towards
# the direct reporter, rather than the human simulator, because the former is more useful on the downstream tasks.

# This should also show up in working better with sparse labels, which we expect to be very useful (though autmoating the labelling 
# might make this less useful). We see this in our early experiments using autoencoders, though it would be good to be back try to 
# demonstrate this in a feed-forward setting - I ran some basic experiments on this but it wasn't working well - this might be a 
# serious issue but it's not the issue faced by the experiments at the moment. Hypothesis - it depends on the relative complexity of the 
# pre and post true functions.

# My hope was that if we trained separate Joint models, and then 'crossed the wires', so the image -> concept part of model 1 was passing its outputs  
# to the concept -> class function of model 2 50% of the time, while 50% it still passed it to model 1, and these concept -> image models
# were frozen, the model would be forced to switch to a concept representation that matched to only that which the models had in common.

# We can see this in the clustering of the results here. Looks like there's 

# We can show this 

# Aside: Can we look at this through the lens of single-basin theory and see which attributes match up well to something that arises in the non-attribute loss case.?

