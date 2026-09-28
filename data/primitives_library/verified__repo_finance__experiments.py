import numpy as np
import pandas as pd

from finance._validation import series, window_size
from finance.models.forecasting import ForecastEvaluation, _scores


def evaluate_sequence(
    close: pd.Series,
    *,
    architecture: str = "lstm",
    lookback: int = 20,
    train_fraction: float = 0.75,
    epochs: int = 20,
    seed: int = 0,
) -> ForecastEvaluation:
    """One-step return prediction with a small LSTM or causal-window CNN; CPU, fixed epoch budget."""
    import torch
    from torch import nn

    close = series(close, positive=True, missing=False)
    window_size(lookback, 3)
    window_size(epochs)

    if architecture not in ("lstm", "cnn") or not 0.2 < train_fraction < 0.9:
        raise ValueError("architecture lstm/cnn and chronological split in (.2,.9) required")

    returns = close.pct_change(fill_method=None).dropna()
    features = np.array(
        [
            returns.iloc[position - lookback : position].to_numpy()
            for position in range(lookback, len(returns))
        ]
    )
    targets = returns.iloc[lookback:].to_numpy()

    split = int(len(targets) * train_fraction)
    if split < 40 or len(targets) - split < 10:
        raise ValueError("insufficient sequence history")

    mean = features[:split].mean()
    scale = features[:split].std()
    if scale == 0:
        raise ValueError("training returns are constant")

    values = torch.tensor((features - mean) / scale, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor((targets - mean) / scale, dtype=torch.float32).unsqueeze(-1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        if architecture == "lstm":

            class LSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.sequence = nn.LSTM(1, 8, batch_first=True)
                    self.output = nn.Linear(8, 1)

                def forward(self, inputs):
                    output, _ = self.sequence(inputs)
                    return self.output(output[:, -1])

            model = LSTM()
        else:

            class CNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Conv1d(1, 8, 3),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(8 * (lookback - 2), 1),
                    )

                def forward(self, inputs):
                    return self.network(inputs.transpose(1, 2))

            model = CNN()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(values[:split]), labels[:split])
            if not torch.isfinite(loss):
                raise ValueError("neural training diverged")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predicted = model(values[split:]).numpy().ravel() * scale + mean

    index = returns.index[lookback:]
    predictions = pd.DataFrame(
        {
            "actual": targets[split:],
            architecture: predicted,
            "zero_return": 0.0,
        },
        index=index[split:],
    )
    return ForecastEvaluation(
        predictions,
        _scores(predictions),
        index[split - 1],
        index[split],
    )


def evaluate_prophet(close: pd.Series, train_fraction: float = 0.8) -> ForecastEvaluation:
    """Fixed-origin log-price forecast on observed held-out dates, compared with persistence."""
    from prophet import Prophet

    close = series(close, positive=True, missing=False)

    if not isinstance(close.index, pd.DatetimeIndex) or not 0.2 < train_fraction < 0.9:
        raise ValueError("dated observations and chronological split required")

    split = int(len(close) * train_fraction)
    train = close.iloc[:split]
    test = close.iloc[split:]

    if len(train) < 60 or len(test) < 10:
        raise ValueError("insufficient history")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        uncertainty_samples=0,
    )
    model.fit(
        pd.DataFrame(
            {
                "ds": train.index.tz_localize(None),
                "y": np.log(train.to_numpy()),
            }
        )
    )

    forecast = model.predict(pd.DataFrame({"ds": test.index.tz_localize(None)}))
    predictions = pd.DataFrame(
        {
            "actual": test,
            "prophet": np.exp(forecast.yhat.to_numpy()),
            "persistence": train.iloc[-1],
        },
        index=test.index,
    )

    return ForecastEvaluation(
        predictions,
        _scores(predictions),
        train.index[-1],
        test.index[0],
    )