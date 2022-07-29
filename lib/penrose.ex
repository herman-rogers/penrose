defmodule Penrose do
  @moduledoc """
  Collection of examples for machine learning computation
  """

  alias Penrose.Models.SimpleNeuralNetwork
  alias Penrose.Datasets.CreditFraud

  @doc """
  Simple Neural Network running over credit fraud data
  """
  def train_credit_fraud_model() do
    {batched_train, y_train} = CreditFraud.normalized_training_data()
    SimpleNeuralNetwork.generate_model(batched_train, y_train)
  end

  def credit_fraud_model_metrics() do
    # Functions for investigating data
    # Nx.shape(y_test)
    # Nx.sum(y_test)

    {batched_test, _x_test, _y_test} = CreditFraud.normalized_test_data()
    {model, params} = File.read!("simple_model.axon") |> Axon.deserialize()

    IO.inspect(">> Fradulent data set output")
    model
    |> Axon.Loop.evaluator(params[:model_state])
    |> Axon.Loop.metric(:true_positives, "fraud_declined", :running_sum)
    |> Axon.Loop.metric(:true_negatives, "legit_accepted", :running_sum)
    |> Axon.Loop.metric(:false_positives, "legit_declined", :running_sum)
    |> Axon.Loop.metric(:false_negatives, "fraud_accepted", :running_sum)
    |> Axon.Loop.run(batched_test, compiler: EXLA)
  end
end
