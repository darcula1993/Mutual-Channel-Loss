# Mutual-Channel-Loss

This is an unofficial implementation of Mutual-Channel-Loss:The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification (TIP 2020) DOI

The official pytorch code is here:https://github.com/PRIS-CV/Mutual-Channel-Loss

### Requirements:

tensorflow 2.0+

numpy

## Sample Usage:
```
model = create_model() #model should have two outputs:[predictions,featuremap]

### "predcitions" and "featuremap" are corresponding ouput layers' names.

losses = {
"predictions": "categorical_crossentropy",
"featuremap": MutualChannelLoss,
}

lossWeights = {"predictions": 1.0, "featuremap": 0.05}

model.compile(
    loss = losses,
    loss_weights = lossWeights,
    optimizer = opt,
    metrics = {'predictions': 'accuracy'}
)
```

