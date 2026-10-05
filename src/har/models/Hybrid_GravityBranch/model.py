import torch
from torch import nn

class HybridCNNLSTM_GravityBranch(nn.Module):
    """Input: (batch, 9, 128). Output: (batch, 6)."""

    def __init__(self, input_channels=9, num_classes=6, dropout_cnn=0.2, dropout_fc=0.2):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(2),
            
            nn.Dropout1d(p=dropout_cnn),
            
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AvgPool1d(2),
            
            nn.Dropout1d(p=dropout_cnn)
        )
        
        self.lstm = nn.LSTM(
            input_size=48, 
            hidden_size=32, 
            num_layers=1, 
            batch_first=True 
            # droput has effect here only if num_layers > 1
        )
        
        # droput layer before the final linear layer to regularize
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_fc), 
            nn.Linear(38, num_classes) #raised to account for the 6 additional features (default) from the gravity branch
        )

    def forward(self, x, gravity_features):
        cnn_out = self.cnn(x) 
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, (hn, cn) = self.lstm(lstm_in)
        
        # Concatenate the last hidden state (or the mean of the hidden states) with the gravity features
        combined = torch.cat([lstm_out.mean(dim=1), gravity_features], dim=1)
        return self.classifier(combined)

if __name__ == "__main__":
    model = HybridCNNLSTM_GravityBranch()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {count:,}")
    # parameters = 16.854  