import torch

class LinearRegression(torch.nn.Module):
    """Linear regression model for churn/tabular classification."""
    def __init__(self, n):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(n-1, 1)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs
    
class LogisticRegression(torch.nn.Module):
    """Logistic regression model for churn/tabular classification."""
    def __init__(self, n):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n-1, 1)

    def forward(self, inputs):
        pred = torch.sigmoid(self.linear(inputs))
        return pred