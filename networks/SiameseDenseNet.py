import torch
import torch.nn as nn
import torchvision.models as models


class SiameseDenseNet(nn.Module):
    """A second attacker construction, for testing whether the audit's conclusions depend on
    one architecture.

    It differs from `SiameseNetwork` in two ways at once, which is deliberate: the backbone is
    DenseNet-121 rather than ResNet-50, and it is initialized from a model pretrained on chest
    radiographs (the CheXNet-style abnormality classifier shipped with this repository) rather
    than on ImageNet. A second attacker that shared both the architecture and the pretraining
    domain would test very little.

    The head is kept identical to `SiameseNetwork` -- 128-d sigmoid embeddings merged by
    absolute difference and scored by a single linear layer -- so that any change in measured
    leakage is attributable to the backbone and its pretraining, not to a different scoring
    rule.
    """

    def __init__(self, classifier_checkpoint=None):
        super(SiameseDenseNet, self).__init__()

        if classifier_checkpoint is not None:
            # Reuse the radiograph-pretrained feature extractor; its 14-way sigmoid head is
            # discarded and replaced by the 128-d embedding layer.
            ckpt = torch.load(classifier_checkpoint, weights_only=False)
            self.model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        else:
            self.model = models.densenet121(pretrained=True)

        n_features = self.model.classifier[0].in_features \
            if isinstance(self.model.classifier, nn.Sequential) \
            else self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=n_features, out_features=128, bias=True)

        self.fc_end = nn.Linear(128, 1)

    def forward_once(self, x):
        return torch.sigmoid(self.model(x))

    def forward(self, input1, input2):
        return self.fc_end(torch.abs(self.forward_once(input1) - self.forward_once(input2)))
