defmodule Penrose.Examples do
  @moduledoc """
  Collection of examples for machine learning computation
  """

  alias Penrose.Models.SimpleNeuralNetwork
  alias Penrose.DataTools

  @doc """
  Simple Neural Network running over credit fraud data
  """
  def credit_fraud() do
    # Load our credit data into a dataframe
    fraud_df = Explorer.DataFrame.read_csv!("./data/creditcard.csv", dtypes: [{"Time", :float}])
    sample_size = Explorer.DataFrame.n_rows(fraud_df)

    train_size = ceil(0.85 * sample_size)
    test_size = sample_size - train_size

    train_df = Explorer.DataFrame.slice(fraud_df, 0, train_size)
    test_df = Explorer.DataFrame.slice(fraud_df, 0, test_size)

    x_train_df = Explorer.DataFrame.select(train_df, &(&1 == "Class"), :drop)
    y_train_df = Explorer.DataFrame.select(train_df, &(&1 == "Class"), :keep)

    x_test_df = Explorer.DataFrame.select(test_df, &(&1 == "Class"), :drop)
    y_test_df = Explorer.DataFrame.select(test_df, &(&1 == "Class"), :keep)

    x_train = DataTools.to_tensor(x_train_df)
    y_train = DataTools.to_tensor(y_train_df)

    x_test = DataTools.to_tensor(x_test_df)
    y_test = DataTools.to_tensor(y_test_df)

    batched_train_inputs = Nx.to_batched_list(x_train, 2048)
    batched_train_targets = Nx.to_batched_list(y_train, 2048)
    batched_train = Stream.zip(batched_train_inputs, batched_train_targets)

    # batched_test_inputs = Nx.to_batched_list(x_test, 2048)
    # batched_test_targets = Nx.to_batched_list(y_test, 2048)
    # batched_test = Stream.zip(batched_test_inputs, batched_test_targets)

    SimpleNeuralNetwork.generate_model(batched_train, y_train)
  end
end
