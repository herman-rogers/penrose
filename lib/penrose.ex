defmodule Penrose do
  @moduledoc """
  Collection of examples for machine learning computation
  """

  alias Penrose.Models.SimpleNeuralNetwork
  alias Penrose.Datasets.CreditFraud

  def train_credit_fraud_model() do
    {batched_train, y_train} = CreditFraud.normalized_training_data()
    SimpleNeuralNetwork.generate_model(batched_train, y_train)
  end

  @doc """
  Simple Neural Network running over credit fraud data
  """
  def credit_fraud() do

    # Nx.shape(y_test)
    # Nx.sum(y_test)

    # Temp generate model
    {batched_train, y_train} = CreditFraud.normalized_training_data()
    test = SimpleNeuralNetwork.generate_model(batched_train, y_train)

    {batched_test, _x_test, _y_test} = CreditFraud.normalized_test_data()
    {model, _params} = File.read!("simple_model.axon") |> Axon.deserialize()

    model
    |> Axon.Loop.evaluator(test)
    |> Axon.Loop.metric(:true_positives, "fraud_declined", :running_sum)
    |> Axon.Loop.run(batched_test)

    # IO.inspect("Fradulent Data in test set: #{Nx.sum(y_test)}")
  end
end
