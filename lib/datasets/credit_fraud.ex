defmodule Penrose.Datasets.CreditFraud do
  alias Penrose.DataTools

  def normalized_training_data() do
    {batched_train, _x_train, y_train} = training_data()
    train_max = get_trained_max()

    normalize = fn {batch, target} ->
      {Nx.divide(batch, train_max), target}
    end

    batched_train = batched_train |> Stream.map(&Nx.Defn.jit(normalize, [&1], compiler: EXLA))
    {batched_train, y_train}
  end

  def normalized_test_data() do
    {batched_test, x_test, y_test} = test_data()
    train_max = get_trained_max()

    normalize = fn {batch, target} ->
      {Nx.divide(batch, train_max), target}
    end

    batched_test = batched_test |> Stream.map(&Nx.Defn.jit(normalize, [&1], compiler: EXLA))
    {batched_test, x_test, y_test}
  end

  def training_data do
    fraud_df = Explorer.DataFrame.read_csv!("./data/creditcard.csv", dtypes: [{"Time", :float}])
    sample_size = Explorer.DataFrame.n_rows(fraud_df)

    train_size = ceil(0.85 * sample_size)

    train_df = Explorer.DataFrame.slice(fraud_df, 0, train_size)

    x_train_df = Explorer.DataFrame.select(train_df, &(&1 == "Class"), :drop)
    y_train_df = Explorer.DataFrame.select(train_df, &(&1 == "Class"), :keep)

    x_train = DataTools.to_tensor(x_train_df)
    y_train = DataTools.to_tensor(y_train_df)

    batched_train_inputs = Nx.to_batched_list(x_train, 2048)
    batched_train_targets = Nx.to_batched_list(y_train, 2048)
    batched_train = Stream.zip(batched_train_inputs, batched_train_targets)

    {batched_train, x_train, y_train}
  end

  def test_data do
    fraud_df = Explorer.DataFrame.read_csv!("./data/creditcard.csv", dtypes: [{"Time", :float}])
    sample_size = Explorer.DataFrame.n_rows(fraud_df)

    train_size = ceil(0.85 * sample_size)
    test_size = sample_size - train_size

    test_df = Explorer.DataFrame.slice(fraud_df, 0, test_size)

    x_test_df = Explorer.DataFrame.select(test_df, &(&1 == "Class"), :drop)
    y_test_df = Explorer.DataFrame.select(test_df, &(&1 == "Class"), :keep)

    x_test = DataTools.to_tensor(x_test_df)
    y_test = DataTools.to_tensor(y_test_df)

    batched_test_inputs = Nx.to_batched_list(x_test, 2048)
    batched_test_targets = Nx.to_batched_list(y_test, 2048)
    batched_test = Stream.zip(batched_test_inputs, batched_test_targets)

    {batched_test, x_test, y_test}
  end

  defp get_trained_max() do
    {_batched_train, x_train, _y_train} = training_data()
    Nx.reduce_max(x_train, axes: [0], keep_axes: true)
  end
end
