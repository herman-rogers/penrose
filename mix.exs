defmodule Penrose.MixProject do
  use Mix.Project

  def project do
    [
      app: :penrose,
      version: "0.2.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps()
    ]
  end

  # Configuration for the OTP application.
  #
  # Type `mix help compile.app` for more information.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Specifies your project dependencies.
  #
  # Type `mix help deps` for examples and options.
  defp deps do
    [
      {:floki, ">= 0.30.0", only: :test},
      # {:swoosh, "~> 1.3"},
      # {:telemetry_metrics, "~> 0.6"},
      # {:telemetry_poller, "~> 1.0"},
      # {:gettext, "~> 0.18"},
      # {:jason, "~> 1.2"},
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
      {:explorer, "~> 0.1.0-dev", github: "elixir-nx/explorer"}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  # For example, to install project dependencies and perform other setup tasks, run:
  #
  #     $ mix setup
  #
  # See the documentation for `Mix` for more info on aliases.
  defp aliases do
    []
  end
end
