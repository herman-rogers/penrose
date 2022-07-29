defmodule Penrose.Models.SimpleNeuralNetwork do
  @moduledoc """
  Simple Neural Network model using Axon
  """

  def generate_model(train_x, train_y) do
    model =
      Axon.input({nil, 30})
      |> Axon.dense(256)
      |> Axon.relu()
      |> Axon.dense(256)
      |> Axon.relu()
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(256)
      |> Axon.relu()
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(1)
      |> Axon.sigmoid()

    train_model(model, train_x, train_y)
  end

  def train_model(model, training_data, y_train) do
    fraud =
      Nx.sum(y_train)
      |> Nx.to_number()

    legit = Nx.size(y_train) - fraud

    loss =
      &Axon.Losses.binary_cross_entropy(
        &1,
        &2,
        negative_weight: 1 / legit,
        positive_weight: 1 / fraud,
        reduction: :mean
      )

    optimizer = Axon.Optimizers.adam(0.01)

    model_state = model
    |> Axon.Loop.trainer(loss, optimizer)
    |> Axon.Loop.metric(:precision)
    |> Axon.Loop.metric(:recall)
    |> Axon.Loop.run(training_data, epochs: 30, compiler: EXLA)

    model
    |> Axon.serialize(%{ model_state: model_state })
    |> then(&File.write!("simple_model.axon", &1))
  end
end
