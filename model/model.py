import torch.nn as nn


def _conv_block(in_ch, out_ch, dropout_p=0.2):
    """Conv → BN → ReLU → MaxPool → Dropout.

    BatchNorm before ReLU stabilises training and acts as implicit
    regularisation — this alone typically reduces overfitting more than
    doubling the dropout rate does.  The small per-block dropout (0.2)
    discourages co-adaptation of spatial features early, where it matters
    most for generalisation.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # bias=False: BN absorbs it
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(p=dropout_p),   # spatial dropout: drops whole feature maps
    )


class CNN(nn.Module):
    def __init__(self, num_classes: int, dropout_classifier: float = 0.5):
        super().__init__()

        # --- Feature extractor ---
        # Wider channels (32 → 64 → 128) give the network more capacity to
        # represent subtle inter-class differences (disgust vs angry, fear vs
        # surprise) without adding depth.
        self.features = nn.Sequential(
            _conv_block(3,   32,  dropout_p=0.1),   # 48→24  (or 224→112 etc.)
            _conv_block(32,  64,  dropout_p=0.15),
            _conv_block(64,  128, dropout_p=0.2),
        )

        # AdaptiveAvgPool decouples the classifier from input resolution.
        # Useful for real-time inference where you might feed 48×48 or 96×96
        # frames — the Linear shapes never change.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))    # → (B, 128, 4, 4)

        # --- Classifier head ---
        # Larger first FC (128*4*4 = 2048 → 256) then one more BN+Dropout
        # before the logits.  BN here helps calibrate the scale of pre-logit
        # activations, which directly reduces softmax saturation.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(256, num_classes),            # raw logits — no Softmax here
        )

        # Kaiming init for every Conv/Linear; constant init for BN
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x                                    # (B, num_classes) logits