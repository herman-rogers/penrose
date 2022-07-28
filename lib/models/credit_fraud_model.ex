defmodule Penrose.Models.SimpleNeuralNetwork do
  @moduledoc """
  Simple Neural Network model using Axon
  """

  def generate_model(train_x, train_y) do
    # fraud_df = Explorer.DataFrame.read_csv!("./data/creditcard.csv", dtypes: [{"Time", :float}])

    # example_size = Explorer.DataFrame.n_rows(fraud_df)
    # train_size = ceil(0.85 * example_size)
    # test_size = example_size - train_size

    # train_df = Explorer.DataFrame.slice(fraud_df, 0, train_size)
    # test_df = Explorer.DataFrame.slice(fraud_df, train_size, test_size)

    # x_train_df = Explorer.DataFrame.select(train_df, &(&1 == "Class"), :drop)
    # y_train_df = Explorer.DataFrame.select(train_df, &(&1 == "Class"), :keep)

    # x_test_df = Explorer.DataFrame.select(test_df, &(&1 == "Class"), :drop)
    # y_test_df = Explorer.DataFrame.select(test_df, &(&1 == "Class"), :keep)

    # x_train = DataTools.to_tensor(x_train_df)
    # y_train = DataTools.to_tensor(y_train_df)

    # x_test = DataTools.to_tensor(x_test_df)
    # y_test = DataTools.to_tensor(y_test_df)

    # batched_train_inputs = Nx.to_batched_list(x_train, 2048)
    # batched_train_targets = Nx.to_batched_list(y_train, 2048)
    # batched_train = Stream.zip(batched_train_inputs, batched_train_targets)

    # batched_test_inputs = Nx.to_batched_list(x_test, 2048)
    # batched_test_targets = Nx.to_batched_list(y_test, 2048)
    # batched_test = Stream.zip(batched_test_inputs, batched_test_targets)

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

    model
    |> Axon.Loop.trainer(loss, optimizer)
    |> Axon.Loop.metric(:precision)
    |> Axon.Loop.metric(:recall)
    |> Axon.Loop.run(training_data, epochs: 30, compiler: EXLA)
  end
end
