import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STAE(nn.Module):
    def __init__(self, n_time_series: int, hidden_dim: int, num_layers: int, n_target: int, dropout: float,
                 forecast_length=1, use_hidden=False, probabilistic=False, mlp_hidden_dim=32, mlp_num_layers=1):
        super(STAE, self).__init__()

        self.layer_dim = num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.use_hidden = use_hidden
        self.forecast_length = forecast_length
        self.probablistic = probabilistic
        self.gru = nn.GRU(n_time_series, hidden_dim, num_layers, batch_first=True, dropout=dropout).to(device)

        self.attention = nn.Linear(hidden_dim, 1).to(device)

        self.fc_combined = nn.Linear(hidden_dim + n_target, n_target).to(device)
        self.fc_prediction = nn.Linear(hidden_dim, n_target).to(device)
        self.learnable_sigmoid = nn.Linear(hidden_dim, 1).to(device)
        self.mlp = nn.Sequential()
        for i in range(mlp_num_layers):
            if i == 0:
                self.mlp.add_module(f"fc_{i}", nn.Linear(n_target, mlp_hidden_dim))
            else:
                self.mlp.add_module(f"fc_{i}", nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())

        self.fc_final = nn.Linear(mlp_hidden_dim, n_target).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden is not None and self.use_hidden:
            h0 = self.hidden.to(device)
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, self.hidden = self.gru(x.to(device), h0.detach())

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(out), dim=1)
        attention_output = torch.sum(attention_weights * out, dim=1)
        prediction = self.fc_prediction(out[:, -1, :])

        combined_output = torch.cat((attention_output, prediction), dim=1)
        final_output = self.fc_combined(combined_output)

        output = self.mlp(final_output)
        final_output = self.fc_final(output)

        return final_output
