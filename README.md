# TFJS Colour Theme Prediction
This simple model gives you occassionally random and occassionally themes from the model itself and asks you to rate them.
This uses a simple three-layer model with 25 hidden neurons and updates the weights each rating.

# What I found
I found that with the current 'all sigmoid' model, the model tends to be more confident in a preferred colour choice and either flat out ignores your recommendations or takes them a little too strongly into consideration. However it seems to work well, even with the random behaviour. It manages to figure out the gist of your preference (e.g. black text / light background or darker backgrounds) and applies them to the recommendations.

Going with other activations such as relu, the model immediately tries to go down the path of "darker colours" and is a lot more hard to converge on other colour choices.

Any recommendations on the model will be appreciated!
